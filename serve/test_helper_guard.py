"""tools/v41/serve/ds41_helper_guard.py: a multi-GPU helper that stops never stops the model (CPU, seconds)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "v41" / "serve"))
import ds41_helper_guard as G  # noqa: E402

R = G.REMOTE


def child(code, line=None, sleep_s=0.0):
    """A stand-in helper: writes `line` to its log, waits sleep_s, exits with `code`."""
    src = (f"import sys, time\nif {line!r}: print({line!r}, flush=True)\ntime.sleep({sleep_s})\nsys.exit({code})")
    return [sys.executable, "-I", "-c", src]


class Fail(Exception):
    pass


class Engine:
    """Records each start's environment; fails while STRATA_DS41_REMOTE is in it if `needs_helper` is False."""

    def __init__(self, fail_with_remote=True, fail_always=False):
        self.envs, self.fail_with_remote, self.fail_always = [], fail_with_remote, fail_always

    def init(self, obj, exe, args, cwd=None, log=None, env=None, lazy=False, cpus=None):
        self.envs.append(dict(env) if env is not None else None)
        if self.fail_always or (self.fail_with_remote and env is not None and R in env):
            raise Fail("the engine exited before it was ready")


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)   # Windows: a just-killed child's log handle
        self.log = os.path.join(self.dir.name, "helper.log")
        self.said = []
        self.g = G.HelperGuard(say=self.said.append)

    def tearDown(self):
        if self.g.proc is not None and self.g.proc.poll() is None:
            self.g.proc.kill()
            self.g.proc.wait()
        self.dir.cleanup()

    def start(self, cmd):
        with open(self.log, "ab") as f:
            f.write(b"ds41_helper: an earlier start's line\n")
            at = f.tell()
            p = subprocess.Popen(cmd, stdout=f, stderr=f, stdin=subprocess.DEVNULL)
        self.g.started(p, self.log, at)
        return p

    def test_reason_reads_this_start_only(self):
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("ds41_helper: old\n")
            at = f.tell()
            f.write("ds41_helper: NVIDIA GeForce RTX 3060 ...\nsomething else\nds41_helper: cannot allocate 7.5 GiB\n")
        self.assertEqual(G.reason(self.log, at, 1), "cannot allocate 7.5 GiB")
        self.assertEqual(G.reason(self.log, os.path.getsize(self.log), 3), "exit code 3")
        self.assertEqual(G.reason(os.path.join(self.dir.name, "none.log"), 0, None), "it stopped")

    def test_without_remote(self):
        env = {R: "ds41_remote", "A": "1"}
        a, k = G.without_remote(("exe", [], None, None), {"env": env})
        self.assertEqual(k["env"], {"A": "1"})
        self.assertIn(R, env)                                     # the caller's dict is not changed
        a, k = G.without_remote(("exe", [], None, None, env, False), {})
        self.assertEqual((a[4], len(a), k), ({"A": "1"}, 6, {}))  # a restart passes env by position (self.spawn)
        with mock.patch.dict(os.environ, {R: "ds41_remote"}):
            a, k = G.without_remote(("exe", []), {})
            self.assertNotIn(R, k["env"])
        same = ("exe", [], None, None, {"A": "1"})
        self.assertEqual(G.without_remote(same, {}), (same, {}))

    def test_an_early_exit_turns_it_off(self):
        self.start(child(1, "ds41_helper: model: no such file"))
        self.assertEqual(self.g.off, "model: no such file")
        self.assertEqual(self.said, ["[strata] the second card's helper stopped (model: no such file): running on one card"])
        eng = Engine(fail_with_remote=False)
        self.g.init(eng.init, None, ("exe", []), {"env": {R: "ds41_remote", "A": "1"}})
        self.assertEqual(eng.envs, [{"A": "1"}])                 # straight to one card, no failed start first

    def test_a_running_helper_stays_on(self):
        with mock.patch.object(G, "EARLY_S", 0.2):
            self.start(child(0, sleep_s=30))
        self.assertIsNone(self.g.off)
        eng = Engine(fail_with_remote=False)
        self.g.init(eng.init, None, ("exe", []), {"env": {R: "ds41_remote"}})
        self.assertEqual(eng.envs, [{R: "ds41_remote"}])
        self.assertEqual(self.said, [])

    def test_failed_while_the_engine_started(self):
        with mock.patch.object(G, "EARLY_S", 0.05):
            self.start(child(1, "ds41_helper: cannot allocate 7.5 GiB", sleep_s=0.5))
        self.assertIsNone(self.g.off)                             # still loading when the engine started
        eng = Engine()
        self.g.init(eng.init, None, ("exe", [], None, None, {R: "ds41_remote", "A": "1"}), {})
        self.assertEqual(eng.envs, [{R: "ds41_remote", "A": "1"}, {"A": "1"}])
        self.assertEqual(self.said, ["[strata] the second card's helper stopped (cannot allocate 7.5 GiB): running on one card"])
        self.g.init(eng.init, None, ("exe", [], None, None, {R: "ds41_remote"}), {})   # a later restart: one card too
        self.assertEqual(eng.envs[-1], {})

    def test_the_engines_own_failure_goes_up(self):
        with mock.patch.object(G, "EARLY_S", 0.05):               # the helper stopped cleanly (told to): not its fault
            self.start(child(0, sleep_s=0.3))
        eng = Engine()
        with self.assertRaises(Fail):
            self.g.init(eng.init, None, ("exe", []), {"env": {R: "x"}})
        self.assertEqual((len(eng.envs), self.g.off), (1, None))

    def test_a_helper_still_running_is_not_blamed(self):
        with mock.patch.object(G, "EARLY_S", 0.05), mock.patch.object(G, "EXIT_WAIT_S", 0.2):
            self.start(child(0, sleep_s=30))
            eng = Engine()
            with self.assertRaises(Fail):
                self.g.init(eng.init, None, ("exe", []), {"env": {R: "x"}})
        self.assertEqual((len(eng.envs), self.g.off), (1, None))

    def test_no_helper_no_retry(self):
        eng = Engine(fail_always=True)
        with self.assertRaises(Fail):
            self.g.init(eng.init, None, ("exe", []), {"env": {"A": "1"}})
        self.assertEqual(len(eng.envs), 1)

    def test_one_card_start_that_fails_too_goes_up(self):
        self.start(child(1, "ds41_helper: no CUDA device 0"))
        eng = Engine(fail_always=True)
        with self.assertRaises(Fail):
            self.g.init(eng.init, None, ("exe", []), {"env": {R: "x"}})


if __name__ == "__main__":
    unittest.main()
