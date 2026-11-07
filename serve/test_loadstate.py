"""The loading screen and Pause / Resume: GET /health's model_state and load steps, the last start's duration, and
--background-load (the model loads on a thread; a failed start ends the server with code 1)."""
import json
import os
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import loadstate
from serve import server as S
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


def health(base):
    return json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())


def post(base, path):
    req = urllib.request.Request(base + path, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


class LoadStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        loadstate.last_start_path = os.path.join(self.tmp.name, "cfg.last-start.json")
        loadstate.done()                                 # a clean state (no file: t0 is None)
        Path(loadstate.last_start_path).unlink(missing_ok=True)
        tok = ByteTokenizer()
        self.engine = loadstate.LoadingMock(MockEngine(tok, "Hello."), 1.0)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        loadstate.last_start_path = None
        self.tmp.cleanup()

    def wait_state(self, want, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = health(self.base)
            if h.get("model_state") == want:
                return h
            time.sleep(0.05)
        self.fail(f"no {want}: {h}")

    def test_background_load_shows_the_steps_then_remembers_the_time(self):
        self.assertEqual(health(self.base)["model_state"], "paused")
        notes, ready = [], []
        S.start_in_background(self.svc, lambda: ready.append(1), lambda: notes.append(1))
        h = self.wait_state("loading")
        self.assertEqual(h["load"]["steps"], len(loadstate.STEPS))
        self.assertIsNone(h["load"]["eta_s"])           # no start before: no estimate
        self.wait_state("loaded")
        time.sleep(0.1)
        self.assertEqual((ready, notes), ([1], [1]))
        took = json.loads(Path(loadstate.last_start_path).read_text())["seconds"]
        self.assertGreater(took, 0.5)
        # Pause, then Resume: the second start has an estimate from the first
        self.assertEqual(post(self.base, "/unload")["status"], "unloaded")
        h = self.wait_state("paused")
        self.assertAlmostEqual(h["load"]["last_s"], took)
        S.start_in_background(self.svc, lambda: None)
        h = self.wait_state("loading")
        self.assertAlmostEqual(h["load"]["last_s"], took)
        self.assertIsNotNone(h["load"]["eta_s"])
        self.wait_state("loaded")

    def test_a_failed_background_start_ends_the_server(self):
        def broken():
            self.engine.starting = True
            time.sleep(0.2)
            self.engine.starting = False
            raise EngineDied("the engine exited before it was ready (test)")
        exits = []
        with mock.patch.object(self.engine, "restart", broken), mock.patch.object(S.os, "_exit", exits.append):
            t = S.start_in_background(self.svc, lambda: None, grace_s=0.3)
            h = self.wait_state("failed")
            self.assertIn("exited before it was ready", h["load"]["error"])
            t.join(5)
        self.assertEqual(exits, [1])

    def test_narrator_steps_and_unknown_keys(self):
        loadstate.begin()
        loadstate.step("cache", "1,234 experts")
        loadstate.step("nonsense")
        h = loadstate.health(self.svc)
        self.assertEqual(h["model_state"], "loading")
        self.assertEqual(h["load"]["step"], "cache")
        self.assertIn("1,234 experts", h["load"]["text"])
        loadstate.failed("x")
        self.assertEqual(loadstate.health(self.svc)["model_state"], "failed")
        loadstate.clear_error()


if __name__ == "__main__":
    unittest.main()
