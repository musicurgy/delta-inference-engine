"""DeepSeek's own API shape (deepseek-ai/chat-demo, 11-15): /anthropic/v1/... paths, /v1/tools/asr and /v1/tools/tts,
the `title` extension and the `latest_reminder` role - against a real server on a mock engine and a stub voice child."""
import http.client
import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

from serve.server import deepseek_api_path, request_title, tts_plain_text
from serve.voice import VoiceServer


class _Stub(http.server.BaseHTTPRequestHandler):
    seen = None

    def do_GET(self):
        self._send(200, b"ok", "text/plain")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Stub.seen = (self.path, self.headers.get("Content-Type"), body)
        if self.path == "/v1/audio/speech":
            self._send(200, b"\x01\x00\x02\x00", "audio/pcm")
        else:
            self._send(200, json.dumps({"text": "hello world"}).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Helpers(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(deepseek_api_path("/anthropic/v1/messages"), "/v1/messages")
        self.assertEqual(deepseek_api_path("/anthropic/v1/messages?beta=true"), "/v1/messages?beta=true")
        self.assertEqual(deepseek_api_path("/v1/tools/asr"), "/v1/audio/transcriptions")
        self.assertEqual(deepseek_api_path("/v1/tools/tts"), "/v1/tools/tts")
        self.assertEqual(deepseek_api_path("/anthropic/x"), "/anthropic/x")
        self.assertEqual(deepseek_api_path("/v1/messages"), "/v1/messages")

    def test_title_only_when_asked(self):
        msgs = [{"role": "latest_reminder", "content": "2026-11-15,Sunday,,Web,English"},
                {"role": "user", "content": "How do magnets work?\nPlease explain."}]
        self.assertEqual(request_title({"messages": msgs, "title": {"enabled": True}}), "How do magnets work")
        self.assertIsNone(request_title({"messages": msgs}))
        self.assertIsNone(request_title({"messages": msgs, "title": {"enabled": False}}))
        self.assertEqual(request_title({"messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 50}]}],
                                        "title": {"enabled": True}}), "x" * 36)

    def test_chat_completions_thinking_switches(self):
        from serve.frontend import openai_to_messages
        base = {"messages": [{"role": "user", "content": "hi"}]}
        kw = lambda **extra: openai_to_messages({**base, **extra})[2]
        off = {"enable_thinking": False}
        self.assertEqual(kw(chat_template_kwargs={"enable_thinking": False}), off)      # vLLM / llama.cpp
        self.assertEqual(kw(chat_template_kwargs={"thinking": False}), off)             # SGLang / V4.1 recipes
        self.assertEqual(kw(thinking={"type": "disabled"}), off)                        # DeepSeek's API
        self.assertNotEqual(kw(chat_template_kwargs={"thinking": True}), off)
        self.assertNotEqual(kw(thinking={"type": "enabled"}), off)
        self.assertNotEqual(kw(), off)

    def test_openrouter_reasoning_enabled_false(self):
        """0.95 (pico-dgx-ds4 9ec2d0f): OpenRouter's {"reasoning": {"enabled": false}} turns thinking off, and wins
        over an effort in the same request; enabled true / absent leaves the request as it was."""
        from serve.frontend import openai_to_messages
        base = {"messages": [{"role": "user", "content": "hi"}]}
        kw = lambda **extra: openai_to_messages({**base, **extra})[2]
        off = {"enable_thinking": False}
        self.assertEqual(kw(reasoning={"enabled": False}), off)
        self.assertEqual(kw(reasoning={"enabled": False, "effort": "high"}), off)
        self.assertEqual(kw(reasoning={"enabled": False}, reasoning_effort="max"), off)
        self.assertEqual(kw(reasoning={"enabled": True}), kw())
        self.assertEqual(kw(reasoning={"enabled": True, "effort": "high"}), kw(reasoning={"effort": "high"}))
        self.assertEqual(kw(reasoning={"enabled": None}), kw())

    def test_inline_system_switch(self):
        import serve.frontend as F
        req = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"},
                            {"role": "system", "content": "late"}, {"role": "user", "content": "v"}]}
        old = F.LATE_SYSTEM_INLINE
        try:
            F.LATE_SYSTEM_INLINE = False
            self.assertEqual([m["role"] for m in F.openai_to_messages(req)[0]], ["system", "user", "user", "user"])
            F.LATE_SYSTEM_INLINE = True
            self.assertEqual([m["role"] for m in F.openai_to_messages(req)[0]], ["system", "user", "system", "user"])
        finally:
            F.LATE_SYSTEM_INLINE = old

    def test_tts_plain_text(self):
        md = "# Title\nSome **bold** and `code`, a [link](http://a.b) [1].\n```py\nprint(1)\n```\n| a | b |\n- item"
        self.assertEqual(tts_plain_text(md), "Title\nSome bold and code, a link .\n \nitem")

    def test_link_and_cite_filters_match_their_rules_in_linear_time(self):
        import random
        import time
        from serve import server as S
        rng = random.Random(11)
        pieces = ["[", "]", "(", ")", "!", "^", "1", "23", "a", " ", "【", "】", "x](", "[a](b)"]
        for _ in range(4000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 14)))
            self.assertEqual(S._md_links(text), S._MD_LINK.sub(r"\1", text), repr(text))
            self.assertEqual(S._md_cites(text), S._MD_CITE.sub("", text), repr(text))
        for text in ("[" * 20000, "【" * 20000, "[a](" * 20000, "![" * 20000 + "]", "[1" * 20000):
            t0 = time.perf_counter()
            tts_plain_text(text)
            self.assertLess(time.perf_counter() - t0, 0.5, text[:8])


class ContentLoop(unittest.TestCase):
    """A reply repeating one 32-256-token passage 8 times is ended as "length" (MiaAI-Lab's loop_abort, 11-15)."""

    def test_periodic_tail(self):
        from serve.server import periodic_tail
        block = list(range(40))
        self.assertEqual(periodic_tail([7] * 5 + block * 8, 8), 40)
        self.assertEqual(periodic_tail(block * 7, 8), 0)                    # 7 repeats: not yet
        self.assertEqual(periodic_tail(list(range(400)), 8), 0)             # no repetition
        self.assertEqual(periodic_tail([1, 2, 3] * 200, 8), 0)              # a short loop is #606's (own switch)
        self.assertEqual(periodic_tail([5] * 600, 8), 0)
        varied = [x for i in range(10) for x in block[:-1] + [100 + i]]     # same lines, a new number each time
        self.assertEqual(periodic_tail(varied, 8), 0)
        self.assertEqual(periodic_tail(block * 8, 0), 0)

    def _service(self, body):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, body, max_context=1 << 16), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return svc, httpd

    def _ask(self, httpd):
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=60)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(
                {"model": "x", "max_tokens": 4000, "messages": [{"role": "user", "content": "go"}]}).encode(),
                {"Content-Type": "application/json"})
            out = json.loads(c.getresponse().read())["choices"][0]
            msg = out["message"]          # (the generic test template treats the text as thinking)
            return {"finish_reason": out["finish_reason"],
                    "text": (msg.get("reasoning_content") or "") + (msg.get("content") or "")}
        finally:
            c.close()

    def test_a_looping_reply_is_ended_when_on(self):
        from serve.server import LOOP_STOP_SUGGESTED
        line = "The same sentence over and over, again and again. "    # 51 byte tokens
        svc, httpd = self._service(line * 30)
        svc.loop_stop_repeats = LOOP_STOP_SUGGESTED
        out = self._ask(httpd)
        self.assertEqual(out["finish_reason"], "length")
        self.assertLess(len(out["text"]), len(line) * 12)

    def test_off_by_default(self):
        line = "The same sentence over and over, again and again. "
        svc, httpd = self._service(line * 30)
        self.assertEqual(svc.loop_stop_repeats, 0)            # opt-in, as #728's passage handling
        self.assertEqual(self._ask(httpd)["text"], line * 30)

    def test_ordinary_text_is_untouched(self):
        from serve.server import LOOP_STOP_SUGGESTED
        body = " ".join(f"Item {i}: the value is {i * i}." for i in range(120))
        svc, httpd = self._service(body)
        svc.loop_stop_repeats = LOOP_STOP_SUGGESTED
        self.assertEqual(self._ask(httpd)["text"], body)


class ThroughTheServer(unittest.TestCase):
    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        root = Path(__file__).resolve().parents[1]
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.server_close)
        self.addCleanup(self.stub.shutdown)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        servers = {}
        for kind in ("asr", "tts"):
            v = VoiceServer(kind, {"cmd": sys.executable, "port": self.stub.server_address[1]})
            v.ensure = (lambda v=v: v.base_url)
            servers[kind] = v
        self.svc.voice_servers = servers
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=60)
        try:
            c.request(method, path, body=body, headers=headers or {})
            r = c.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.read()
        finally:
            c.close()

    def stream(self, body):
        status, _, raw = self.call("POST", "/anthropic/v1/messages", json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "x-api-key": "k"})
        self.assertEqual(status, 200, raw[:300])
        return [json.loads(line[5:]) for line in raw.decode().splitlines() if line.startswith("data:")]

    def test_messages_under_anthropic_prefix_with_title(self):
        # (the latest_reminder role is DeepSeek-template only - this mock server's generic template refuses it; the
        # DS4.1 server renders it with dsv41_encoding, covered by DeepSeek's golden case test_input_3 in
        # tools/v41/serve/test_template_conformance.py)
        req = {"model": "deepseek-flash", "max_tokens": 16, "stream": True, "system": "S",
               "thinking": {"type": "disabled"}, "title": {"enabled": True}, "safety_check": {"level": "high"},
               "messages": [{"role": "user", "content": "Name a colour."}]}
        events = self.stream(req)
        deltas = [e for e in events if e.get("type") == "message_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["delta"].get("title"), "Name a colour")
        self.assertEqual(events[-1]["type"], "message_stop")
        req.pop("title")
        deltas = [e for e in self.stream(req) if e.get("type") == "message_delta"]
        self.assertNotIn("title", deltas[0]["delta"])

    def test_models_under_anthropic_prefix(self):
        self.assertEqual(self.call("GET", "/anthropic/v1/models")[0], 200)

    def test_tools_asr_goes_to_the_transcriber(self):
        status, _, raw = self.call("POST", "/v1/tools/asr", b"--x--", {"Content-Type": "multipart/form-data; boundary=x"})
        self.assertEqual((status, json.loads(raw)["text"]), (200, "hello world"))
        self.assertEqual(_Stub.seen[:2], ("/v1/audio/transcriptions", "multipart/form-data; boundary=x"))

    def test_tools_tts_is_translated(self):
        body = {"text": "Hi **there** `x`", "voice_id": "stella", "format": "pcm", "filter_markdown": True}
        status, ctype, raw = self.call("POST", "/v1/tools/tts", json.dumps(body).encode(),
                                       {"Content-Type": "application/json"})
        self.assertEqual((status, ctype, raw), (200, "audio/pcm", b"\x01\x00\x02\x00"))
        path, sent_type, sent = _Stub.seen
        self.assertEqual((path, sent_type), ("/v1/audio/speech", "application/json"))
        self.assertEqual(json.loads(sent), {"input": "Hi there x", "response_format": "pcm", "voice": "stella"})

    def test_tools_tts_without_a_sidecar_is_404_before_the_filter(self):
        from unittest import mock
        from serve import server as S
        self.svc.voice_servers = {k: v for k, v in self.svc.voice_servers.items() if k != "tts"}
        body = json.dumps({"text": "[" * 50000, "filter_markdown": True}).encode()
        with mock.patch.object(S, "tts_plain_text", side_effect=AssertionError("filtered before the sidecar check")):
            self.assertEqual(self.call("POST", "/v1/tools/tts", body, {"Content-Type": "application/json"})[0], 404)

    def test_tools_tts_needs_text(self):
        self.assertEqual(self.call("POST", "/v1/tools/tts", b"{}", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.call("POST", "/v1/tools/tts", json.dumps({"text": "```\nx\n```", "filter_markdown": True})
                                   .encode(), {"Content-Type": "application/json"})[0], 400)


if __name__ == "__main__":
    unittest.main()
