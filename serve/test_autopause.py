"""Opt-in auto-pause (serve/autopause.py): the model is unloaded when another program takes the graphics card while
the server is idle - never mid-request, never when it is off - and GET /health says why."""
import threading
import unittest

from serve import autopause, loadstate


class FakeEngine:
    def __init__(self):
        self.on, self.starting = True, False

    def alive(self):
        return self.on

    def unload(self):
        self.on = False


class FakeSvc:
    def __init__(self):
        self.engine = FakeEngine()
        self.status_lock = threading.Lock()
        self.status = {"busy": False, "queued": 0}
        self.last_request_at = 0.0
        self.started_at = 0.0
        self.auto_pause_mib = 1024
        self.unloads = 0

    def loaded(self):
        return self.engine.alive()

    def unload(self):
        with self.status_lock:
            if self.status["busy"] or self.status["queued"]:
                return "busy"
        self.unloads += 1
        self.engine.unload()
        return "unloaded"


class AutoPauseTest(unittest.TestCase):
    def setUp(self):
        self.svc = FakeSvc()
        self.now = 1000.0
        self.reading = {"used_mib": 20000, "util": 0}
        self.said = []
        self.ap = autopause.AutoPause(self.svc, read=lambda: dict(self.reading), clock=lambda: self.now,
                                      say=lambda *a, **k: self.said.append(a[0]))

    def tick(self, n=1, dt=5.0):
        out = None
        for _ in range(n):
            self.now += dt
            out = self.ap.step() or out
        return out

    def test_memory_taken_by_another_program_pauses(self):
        self.assertIsNone(self.tick())                 # the base
        self.reading["used_mib"] += 500
        self.assertIsNone(self.tick())
        self.reading["used_mib"] += 600
        why = self.tick()
        self.assertIn("1.1 GB", why)
        self.assertEqual(self.svc.unloads, 1)
        self.assertIn("Paused by itself", loadstate._state["reason"])
        self.assertIsNone(self.tick(5))                # unloaded: nothing more

    def test_steady_load_pauses_a_short_one_does_not(self):
        self.tick()
        self.reading["util"] = 90
        self.assertIsNone(self.tick(3))                # 15 s
        self.reading["util"] = 5
        self.assertIsNone(self.tick())
        self.reading["util"] = 90
        self.assertIsNone(self.tick(4))                # 20 s from the new start: not yet past BUSY_S at the 4th
        self.assertIn("90% load", self.tick() or "")

    def test_never_mid_request_nor_right_after_one(self):
        self.tick()
        self.svc.status["busy"] = True
        self.reading["used_mib"] += 5000
        self.assertIsNone(self.tick(5))
        self.svc.status["busy"] = False
        self.svc.last_request_at = self.now            # it just finished: quiet first, then a new base
        self.assertIsNone(self.tick(6))
        self.assertIsNone(self.tick(2))                # the base includes the 5 GB taken during the request
        self.assertEqual(self.svc.unloads, 0)

    def test_off_does_nothing(self):
        self.svc.auto_pause_mib = 0
        self.tick()
        self.reading.update(used_mib=40000, util=100)
        self.assertIsNone(self.tick(10))
        self.assertEqual(self.svc.unloads, 0)

    def test_a_request_arriving_wins(self):
        self.tick()
        self.reading["used_mib"] += 3000
        real = self.svc.unload
        self.svc.unload = lambda: "busy"
        self.assertIsNone(self.tick())
        self.svc.unload = real
        self.assertTrue(self.svc.engine.on)


if __name__ == "__main__":
    unittest.main()
