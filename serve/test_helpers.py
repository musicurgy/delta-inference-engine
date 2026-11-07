"""serve/test_helpers.py - Claude Code's helper calls (serve/helpers.py): the classifier's narrow signatures, the quota
probe answered without the model, thinking off for the small helpers, and a prompt suggestion that yields to the main
conversation - against the mock engine (no GPU).

    python -m unittest serve.test_helpers -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import helpers as H  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The shapes Claude Code sends (abridged from third-party captures; the parts the rules read are verbatim)
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.300.abc; cch=12345;"}
QUOTA = {"model": "claude-haiku-4-5", "max_tokens": 1, "messages": [{"role": "user", "content": "quota"}],
         "metadata": {"user_id": "u"}}
TITLE = {"model": "claude-haiku-4-5", "max_tokens": 512, "system": [BILLING, {"type": "text", "text":
         "Generate a concise, sentence-case title (3-7 words) that captures the main topic or goal of this coding "
         "session. Return JSON with a single \"title\" field."}],
         "messages": [{"role": "user", "content": "fix the failing unit tests in serve/"}]}
TOPIC = {"model": "claude-haiku-4-5", "max_tokens": 512, "system": [{"type": "text", "text":
         "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that "
         "captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and "
         "'title' (string, or null if isNewTopic is false)."}],
         "messages": [{"role": "user", "content": [{"type": "text", "text": "now look at the README"}]}]}
BASH = {"model": "claude-haiku-4-5", "max_tokens": 512, "system": "Your task is to process Bash commands that an AI "
        "coding agent wants to run.", "messages": [{"role": "user", "content": "<policy_spec>\n# Claude Code Code "
        "Bash command prefix detection\n...\n</policy_spec>\n\nCommand: git status --short\n"}]}
PATHS = {"model": "claude-haiku-4-5", "max_tokens": 512, "system": "Extract any file paths that this command reads "
         "or modifies. Format your response as:\n<filepaths>\npath/to/file1\n</filepaths>",
         "messages": [{"role": "user", "content": "Command: cat README.md\nOutput: # Title\n"}]}
TOOLS = [{"name": "Bash", "description": "run a command", "input_schema": {"type": "object", "properties": {
    "command": {"type": "string"}}}}]
MAIN = {"model": "claude-opus-4-7", "max_tokens": 4096, "system": [BILLING, {"type": "text", "text": "You are "
        "Claude Code."}], "tools": TOOLS, "messages": [{"role": "user", "content": "hello"}]}
SUGGESTION = dict(MAIN, messages=MAIN["messages"] + [
    {"role": "assistant", "content": "Hi! What shall we do?"},
    {"role": "user", "content": "[SUGGESTION MODE: Suggest what the user might naturally type next into Claude "
                                "Code.]\n\nReply with ONLY the suggestion."}])


class Classify(unittest.TestCase):
    def test_each_signature(self):
        for req, want in ((QUOTA, H.QUOTA), (TITLE, H.TITLE), (TOPIC, H.TOPIC), (BASH, H.BASH_PREFIX),
                          (PATHS, H.FILE_PATHS), (SUGGESTION, H.SUGGESTION), (MAIN, None)):
            with self.subTest(want=want):
                self.assertEqual(H.classify(req), want)

    def test_near_misses_stay_main(self):
        cases = {
            # a user who types "quota" in a real conversation
            "quota with a budget": dict(QUOTA, max_tokens=100),
            "quota with tools": dict(QUOTA, tools=TOOLS),
            "quota in a sentence": dict(QUOTA, messages=[{"role": "user", "content": "what is my quota?"}]),
            # a title-like system prompt in a request with tools is a real agent turn
            "title with tools": dict(TITLE, tools=TOOLS),
            "title-ish long conversation": dict(TITLE, messages=[{"role": "user", "content": "a"},
                                                                 {"role": "assistant", "content": "b"},
                                                                 {"role": "user", "content": "c"}]),
            "only the word title": dict(TITLE, system="Write a title for my blog post."),
            "policy_spec without Command:": dict(BASH, messages=[{"role": "user", "content": "<policy_spec> x"}]),
            "bash check with history": dict(BASH, messages=BASH["messages"] + [{"role": "assistant", "content": "x"}]),
            "Command/Output without filepaths": dict(PATHS, system="Summarize this."),
            "suggestion mark mid-text": dict(MAIN, messages=[{"role": "user", "content":
                                                              "what does [SUGGESTION MODE: mean?"}]),
            "suggestion mark in an assistant turn": dict(MAIN, messages=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "[SUGGESTION MODE: x]"}]),
            "no messages": {"model": "m", "max_tokens": 1},
            "not an object": "quota",
        }
        for name, req in cases.items():
            with self.subTest(name):
                self.assertIsNone(H.classify(req))

    def test_apply(self):
        req, kw = H.apply(H.TITLE, {"m": 1}, {"reasoning_effort": "high"})
        self.assertEqual(kw, {"enable_thinking": False})
        req, kw = H.apply(H.SUGGESTION, {"m": 1}, {"reasoning_effort": "high"})
        self.assertEqual(kw, {"reasoning_effort": "high"})         # unchanged: the main conversation's prompt
        self.assertTrue(req["strata_preemptible"])
        self.assertEqual(req["reasoning_budget_tokens"], H.SUGGESTION_BUDGET)
        req, _ = H.apply(H.SUGGESTION, {"reasoning_budget_tokens": 9}, {})
        self.assertEqual(req["reasoning_budget_tokens"], 9)
        req, kw = H.apply(None, {"m": 1}, {"x": 1})
        self.assertEqual((req, kw), ({"m": 1}, {"x": 1}))


class MainFirstOrder(unittest.TestCase):
    def test_a_main_request_stops_a_queued_or_decoding_suggestion_at_once(self):
        o = H.MainFirst()
        for state in ("queued", "decoding"):
            with self.subTest(state=state):
                c = threading.Event()
                e = o.enter(c, True)
                e["state"] = state
                m = o.enter(threading.Event(), False)
                self.assertTrue(c.is_set())
                o.leave(m)
                o.leave(e)

    def test_a_suggestion_reading_its_prompt_stops_at_its_first_token(self):
        o = H.MainFirst()
        c = threading.Event()
        e = o.enter(c, True)
        lock = threading.Lock()
        self.assertTrue(o.acquire(e, lock, 1.0))
        o.took(e)
        m = o.enter(threading.Event(), False)
        self.assertFalse(c.is_set())          # a cancelled read would make the engine drop the conversation
        o.first_token(e)
        self.assertTrue(c.is_set())
        lock.release()
        o.leave(e)
        o.leave(m)

    def test_a_suggestion_does_not_take_the_turn_while_a_main_request_waits(self):
        o = H.MainFirst()
        lock = threading.Lock()
        m = o.enter(threading.Event(), False)            # waiting (the turn is free, but it has not taken it yet)
        e = o.enter(threading.Event(), True)
        t0 = time.monotonic()
        self.assertFalse(o.acquire(e, lock, 0.3))
        self.assertGreaterEqual(time.monotonic() - t0, 0.25)
        self.assertTrue(o.acquire(m, lock, 1.0))
        o.took(m)
        lock.release()
        o.leave(m)
        o.leave(e)
        self.assertEqual(o.main_waiting, 0)


class SlowEngine(MockEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        self.prompts.append(bytes(t for t in ids if t < 256).decode("utf-8", "replace"))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = SlowEngine(tok, "</think>\n\n" + "ab" * 40, max_context=16384, delay_s=0.01)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.helper_calls = True
        self.engine.delay = 0.01

    def post(self, body, raw=False):
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return r.status, (data.decode() if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_quota_probe_is_answered_without_the_model(self):
        before = self.engine.calls
        s, b = self.post(QUOTA)
        self.assertEqual(s, 200, b)
        self.assertEqual(b["type"], "message")
        self.assertEqual(b["stop_reason"], "max_tokens")
        self.assertEqual(b["content"], [{"type": "text", "text": "ok"}])
        s, text = self.post(dict(QUOTA, stream=True), raw=True)
        self.assertEqual(s, 200)
        self.assertIn("event: message_stop", text)
        self.assertEqual(self.engine.calls, before)

    def test_quota_probe_reaches_the_model_when_turned_off(self):
        self.svc.helper_calls = False
        before = self.engine.calls
        s, b = self.post(QUOTA)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.calls, before + 1)

    def test_small_helpers_run_without_thinking(self):
        for req in (TITLE, TOPIC, BASH, PATHS):
            with self.subTest(system=str(req.get("system"))[:30]):
                s, b = self.post(req)
                self.assertEqual(s, 200, b)
                # thinking off: the template closes the thinking in the prompt itself
                self.assertTrue(self.engine.prompts[-1].rstrip().endswith("</think>"), self.engine.prompts[-1][-40:])
        self.svc.helper_calls = False
        self.post(TITLE)
        self.assertFalse(self.engine.prompts[-1].rstrip().endswith("</think>"))

    def test_main_request_stops_a_running_suggestion(self):
        self.engine.delay = 0.05                      # 80 tokens: ~4 s
        out = {}

        def suggest():
            out["s"] = self.post(SUGGESTION)
        t = threading.Thread(target=suggest)
        t.start()
        deadline = time.monotonic() + 10
        while not self.svc.status.get("busy") and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.2)                               # decoding by now
        t0 = time.monotonic()
        self.engine.delay = 0.0
        s, b = self.post(MAIN)
        t.join(30)
        self.assertEqual(s, 200, b)
        self.assertLess(time.monotonic() - t0, 3.0)   # did not wait for the whole suggestion
        s2, b2 = out["s"]
        self.assertEqual(s2, 200, b2)
        self.assertLess(b2["usage"]["output_tokens"], 80)   # stopped part-way
        self.assertEqual(b["usage"]["output_tokens"], len(self.engine.script))   # the main reply is whole


if __name__ == "__main__":
    unittest.main()
