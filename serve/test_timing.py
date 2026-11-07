"""serve/test_timing.py - the opt-in timing line (serve/timing.py): off by default and nothing runs then; on, one line per
chat request with the ms between the steps it passed, in order.

    python -m unittest serve.test_timing -v
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import timing  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Server:
    def __enter__(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello there"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.out = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.out)
        self.redirect.__enter__()
        return self

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}{path}",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                              "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()

    def lines(self, n, wait_s=3.0):
        """The timing lines, once there are n of them (the handler prints its line as the request ends)."""
        end = time.monotonic() + wait_s
        while True:
            got = [x for x in self.out.getvalue().splitlines() if x.startswith("[strata] timing ")]
            if len(got) >= n or time.monotonic() > end:
                return got

    def __exit__(self, *exc):
        self.redirect.__exit__(*exc)
        self.httpd.shutdown()
        self.httpd.server_close()


CHAT = {"model": "x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}


class Off(unittest.TestCase):
    def test_off_by_default_and_nothing_runs(self):
        self.assertFalse(timing.ON)
        with mock.patch.object(timing, "stamp", side_effect=AssertionError("stamped while off")), \
                mock.patch.object(timing, "start", side_effect=AssertionError("started while off")), \
                mock.patch.object(timing, "finish", side_effect=AssertionError("finished while off")), Server() as s:
            s.post("/v1/chat/completions", dict(CHAT, stream=True))
            s.post("/v1/messages", CHAT)
            self.assertEqual(s.lines(0, 0), [])

    def test_env(self):
        try:
            for value, on in (("1", True), ("0", False), ("", False), ("true", True)):
                with self.subTest(value=value), mock.patch.dict(os.environ, {"STRATA_TIMING": value}):
                    self.assertEqual(importlib.reload(timing).ON, on)
        finally:
            with mock.patch.dict(os.environ):
                os.environ.pop("STRATA_TIMING", None)
                importlib.reload(timing)
        self.assertFalse(timing.ON)


class On(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(timing, "ON", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def steps(self, line):
        return re.findall(r"([a-z][a-z ]*?) \+\d+", line.split(": ", 1)[1])

    def test_streamed_chat(self):
        with Server() as s:
            s.post("/v1/chat/completions", dict(CHAT, stream=True))
            lines = s.lines(1)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("[strata] timing /v1/chat/completions: parsed +"), lines[0])
        self.assertEqual(self.steps(lines[0]), ["parsed", "converted", "images", "rendered", "tokenized", "prepared",
                                                "turn", "loaded", "sent", "first token", "first sent", "done"])
        self.assertRegex(lines[0], r" = \d+ ms \(chars \d+, tokens \d+\)$")

    def test_anthropic_whole_answer(self):
        with Server() as s:
            s.post("/v1/messages", CHAT)
            lines = s.lines(1)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("timing /v1/messages:", lines[0])
        self.assertEqual(self.steps(lines[0])[-2:], ["first sent", "done"])

    def test_other_paths_have_no_line(self):
        with Server() as s:
            s.post("/v1/messages/count_tokens", CHAT)
            s.post("/v1/chat/completions", CHAT)
            lines = s.lines(1)
            time.sleep(0.2)
            lines = s.lines(2, 0)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("/v1/chat/completions", lines[0])

    def test_line(self):
        ladder = {"what": "/v1/messages", "marks": {"received": 1.0, "parsed": 1.002, "rendered": 1.0405,
                                                    "done": 2.0}, "info": {"chars": 10}}
        self.assertEqual(timing.line(ladder),
                         "[strata] timing /v1/messages: parsed +2, rendered +38, done +960 = 1000 ms (chars 10)")

    def test_first_stamp_counts_and_no_ladder_is_a_no_op(self):
        timing.stamp("parsed")                            # no request on this thread: nothing happens
        self.assertIsNone(timing.finish())
        timing.start("/x")
        timing.stamp("sent")
        first = timing._local.ladder["marks"]["sent"]
        timing.stamp("sent")
        self.assertEqual(timing._local.ladder["marks"]["sent"], first)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIn("sent +", timing.finish())
        self.assertIsNone(timing.finish())


if __name__ == "__main__":
    unittest.main()
