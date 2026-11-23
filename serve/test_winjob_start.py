"""serve/winjob.py start_contained(): the child is in the kill-on-close job before it runs, so a process it starts at
once (the npx / uvx wrappers of MCP servers) is in the job too and ends with the server.

    python -m unittest serve.test_winjob_start      (Windows only; skipped elsewhere)
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve import winjob  # noqa: E402

# stands in for the server: starts a child that starts a grandchild at once; the assignment to the job comes 1.5 s
# late (a slow moment on a busy PC), prints both pids, then waits to be killed
PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve import winjob
on_time = winjob.contain
def late(proc):
    time.sleep(1.5)
    return on_time(proc)
winjob.contain = late
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
args = [sys.executable, "-c", code]
if sys.argv[1] == "start_contained":
    child = winjob.start_contained(args, stdout=subprocess.PIPE, text=True)
else:                                   # the old way: started, then contained
    child = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    late(child)
print(child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""

PY = getattr(sys, "_base_executable", None) or sys.executable   # a venv's python.exe is a launcher with a child


def running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def kill_server(how: str):
    """Start the stand-in server, kill it outright; the pids of its child and grandchild, and whether each outlived it."""
    parent = subprocess.Popen([PY, "-c", PARENT, how], stdout=subprocess.PIPE, text=True)
    pids = []
    try:
        pids = [int(p) for p in parent.stdout.readline().split()]
        parent.kill()                           # TerminateProcess: no cleanup - like closing the console window
        parent.wait(10)
        deadline = time.monotonic() + 10
        while any(running(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.2)
        return pids, [running(p) for p in pids]
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in pids:
            if running(pid):
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)


class Recorder:
    """subprocess.Popen as start_contained calls it: the real class, every call's creationflags and process kept."""

    def __init__(self):
        self.calls = []

    def __call__(self, args, **kw):
        proc = winjob._REAL_POPEN(args, **kw)
        self.calls.append((kw.get("creationflags", 0), proc))
        return proc


SAY = [sys.executable, "-c", "print('ran')"]


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestStartContained(unittest.TestCase):
    def test_a_grandchild_started_at_once_ends_with_the_server(self):
        pids, alive = kill_server("start_contained")
        self.assertEqual(len(pids), 2)
        self.assertEqual(alive, [False, False], "the child or its grandchild outlived the server")

    def test_control_contained_after_popen_the_grandchild_escapes(self):
        # the race this fixes: the grandchild started before the (late) assignment is not in the job
        pids, alive = kill_server("popen_then_contain")
        self.assertEqual(len(pids), 2)
        self.assertEqual(alive, [False, True])

    def run_say(self, **kw):
        rec = Recorder()
        with mock.patch.object(winjob.subprocess, "Popen", rec):
            proc = winjob.start_contained(SAY, stdout=subprocess.PIPE, text=True, **kw)
        out, _ = proc.communicate(timeout=30)
        self.assertEqual((out.strip(), proc.returncode), ("ran", 0))
        return rec.calls

    def test_created_suspended_with_the_callers_flags_then_runs(self):
        calls = self.run_say(creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual([f for f, _ in calls], [subprocess.CREATE_NO_WINDOW | winjob.CREATE_SUSPENDED])

    def test_a_refused_assignment_still_runs_the_child(self):
        with mock.patch.object(winjob, "contain", return_value=False):
            self.assertEqual(len(self.run_say()), 1)

    def test_no_job_is_plain_popen(self):
        with mock.patch.object(winjob, "_job", 0):
            self.assertEqual([f for f, _ in self.run_say()], [0])

    def test_a_child_that_cannot_be_resumed_is_ended_and_started_the_old_way(self):
        with mock.patch.object(winjob, "_resume", return_value=False):
            calls = self.run_say()
        self.assertEqual([f for f, _ in calls], [winjob.CREATE_SUSPENDED, 0])
        self.assertIsNotNone(calls[0][1].poll(), "the suspended child was left behind")

    def test_an_error_before_the_resume_ends_the_child(self):
        rec = Recorder()
        with mock.patch.object(winjob.subprocess, "Popen", rec), \
                mock.patch.object(winjob, "contain", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                winjob.start_contained(SAY, stdout=subprocess.PIPE, text=True)
        self.assertEqual(len(rec.calls), 1)
        self.assertIsNotNone(rec.calls[0][1].poll(), "the suspended child was left behind")

    def test_a_stand_in_is_returned_as_is(self):
        # tests elsewhere replace subprocess.Popen with objects that have no process behind them
        fake = mock.Mock(pid=4242)
        with mock.patch.object(winjob.subprocess, "Popen", return_value=fake) as popen, \
                mock.patch.object(winjob, "_resume") as resume:
            self.assertIs(winjob.start_contained(["x"]), fake)
        self.assertEqual(popen.call_count, 1)
        resume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
