"""serve/test_stream_tool_deltas.py - streamed tool calls on /v1/chat/completions in the shape strict clients parse
(OpenCode reportedly fails on `"name": null` and on an empty `"tool_calls": []`, STATUS_094 row 85): a call's first
delta carries its id, type and function name, the later ones only arguments; no key is ever null, no array empty.

    python -m unittest serve.test_stream_tool_deltas -v
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, Event, ToolCall  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect, openai_tool_delta, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TWO_CALLS = ("</think>\n\nLet me look.\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.py\n</parameter>\n"
             "</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
             "</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}}}}]


def nulls(v, path=""):
    """Every path inside v whose value is null or an empty tool_calls array."""
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            if x is None and k != "finish_reason":            # finish_reason: null is OpenAI's own until the end
                out.append(path + "." + k)
            elif k == "tool_calls" and x == []:
                out.append(path + "." + k + "=[]")
            out += nulls(x, path + "." + k)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            out += nulls(x, f"{path}[{i}]")
    return out


def check_deltas(test, chunks):
    """The tool-call deltas of a chat stream follow the rule; -> {index: (name, arguments)}."""
    seen = {}
    for c in chunks:
        test.assertEqual(nulls(c["choices"][0]["delta"]), [], c)
        for tc in c["choices"][0]["delta"].get("tool_calls", []):
            fn = tc["function"]
            if tc["index"] not in seen:
                test.assertEqual(set(tc), {"index", "id", "type", "function"}, tc)
                test.assertEqual(tc["type"], "function")
                test.assertTrue(isinstance(tc["id"], str) and tc["id"], tc)
                test.assertTrue(isinstance(fn.get("name"), str) and fn["name"], tc)
                seen[tc["index"]] = [fn["name"], fn.get("arguments", "")]
            else:
                test.assertEqual(set(tc), {"index", "function"}, tc)
                test.assertEqual(set(fn), {"arguments"}, tc)
                seen[tc["index"]][1] += fn["arguments"]
    return {k: tuple(v) for k, v in seen.items()}


class OverHttp(unittest.TestCase):
    def test_streamed_calls(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, TWO_CALLS), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = {"model": "x", "max_tokens": 400, "stream": True, "tools": TOOLS,
                    "messages": [{"role": "user", "content": "look at main.py"}]}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertNotIn('"name": null', raw)
        self.assertNotIn('"tool_calls": []', raw)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        calls = check_deltas(self, chunks)
        self.assertEqual({k: (n, json.loads(a)) for k, (n, a) in calls.items()},
                         {0: ("read", {"path": "src/main.py"}), 1: ("bash", {"command": "ls -la"})})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))


class Chunks(unittest.TestCase):
    """openai_chunks over events the parsers can give: a call streamed piece by piece, one sent whole, a name missing."""

    def chunks(self, events):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ""), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        run = [("event", e) for e in events] + [("done", {"finish": "stop", "completion_tokens": 3})]
        return list(openai_chunks(svc, {"model": "x"}, [1, 2], True, TOOLS, 10, threading.Event(), run=iter(run)))

    def test_streamed_and_whole(self):
        a, b = ToolCall("read", {"path": "x"}), ToolCall("bash", {"command": "ls"})
        out = self.chunks([Event("tool_start", call=a), Event("tool_args", '{"path":', call=a),
                           Event("tool_args", '"x"}', call=a), Event("tool_call", call=a), Event("tool_call", call=b)])
        calls = check_deltas(self, out)
        self.assertEqual(calls, {0: ("read", '{"path":"x"}'), 1: ("bash", '{"command": "ls"}')})
        msg = openai_collect(out)["choices"][0]["message"]
        self.assertEqual([(c["id"], c["function"]["name"]) for c in msg["tool_calls"]], [(a.id, "read"), (b.id, "bash")])

    def test_no_null_name(self):
        """A call whose name is unknown (None) leaves the key out instead of sending null."""
        c = ToolCall(None, {})
        out = self.chunks([Event("tool_start", call=c), Event("tool_args", "{}", call=c), Event("tool_call", call=c)])
        first = [tc for ch in out for tc in ch["choices"][0]["delta"].get("tool_calls", [])][0]
        self.assertNotIn("name", first["function"])
        self.assertEqual(first["id"], c.id)
        for ch in out:
            self.assertEqual(nulls(ch["choices"][0]["delta"]), [])

    def test_empty_args_piece_is_not_sent(self):
        c = ToolCall("read", {})
        out = self.chunks([Event("tool_start", call=c), Event("tool_args", "", call=c), Event("tool_args", "{}", call=c),
                           Event("tool_call", call=c)])
        pieces = [tc for ch in out for tc in ch["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(len(pieces), 2)

    def test_delta_helper(self):
        c = ToolCall("read", {})
        self.assertEqual(openai_tool_delta(0, c, ""), {"index": 0, "id": c.id, "type": "function",
                                                       "function": {"name": "read", "arguments": ""}})
        self.assertEqual(openai_tool_delta(2, None, "{}"), {"index": 2, "function": {"arguments": "{}"}})
        self.assertEqual(openai_tool_delta(1, ToolCall("", {}), "{}")["function"], {"arguments": "{}"})


if __name__ == "__main__":
    unittest.main()
