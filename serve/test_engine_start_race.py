"""serve/test_engine_start_race.py - a failed engine start is reported cleanly when another thread closed the process.

StrataEngine.__init__ cleans up a start that ended before READY.  restart() / close() on another thread set self.proc
to None, so the cleanup read None.stdin: an AttributeError instead of the start failure (EngineDied, a 503).  The
cleanup now uses the process object of THIS start, kept in a local.  After upstream Xolotl 2dd7c80a.

No GPU or model: the "engine" is a Python script that exits before READY.

    python -m unittest serve.test_engine_start_race -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402

FAILING = "import sys; sys.exit(3)\n"                 # exits at once, before READY


class EngineStartRace(unittest.TestCase):
    def test_a_failed_start_survives_a_concurrent_close(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "engine.py"
        script.write_text(FAILING, encoding="utf-8")
        holder = {}
        real_init, real_popen = server.StrataEngine.__init__, server.subprocess.Popen

        def init_keeping_self(self, *a, **k):          # the instance exists before __init__ runs
            holder["eng"] = self
            return real_init(self, *a, **k)

        def racing_popen(cmd, **kw):
            proc = real_popen([sys.executable, str(script), *cmd[1:]], **kw)
            real_wait = proc.wait

            def wait_then_another_thread_closes(*a, **k):
                out = real_wait(*a, **k)
                holder["eng"].proc = None               # what restart() / close() do on another thread
                return out

            proc.wait = wait_then_another_thread_closes
            return proc

        with mock.patch.object(server.StrataEngine, "__init__", init_keeping_self), \
                mock.patch.object(server.subprocess, "Popen", racing_popen), \
                mock.patch.object(server, "START_RETRIES", 0):       # one attempt: the cleanup runs at once
            with self.assertRaises(server.EngineDied) as caught:
                server.StrataEngine(sys.executable, ["--max-context", "4096"])
        self.assertIn("exited before it was ready", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
