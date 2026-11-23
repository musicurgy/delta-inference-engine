"""0.95: four server fixes ported from upstream Strata (0.2.6 / 0.2.9 / 0.2.12), each first written as a test of the
behaviour this server had before the port.

1. Forced tool_choice "required" / Anthropic "any" with several tools (Strata 0.2.12, PR #4828: 0 of 24 forced runs
   made a real call, 24 of 24 after).  The server wrote the call's opening up to the tool's name (`<function=`; ours
   in DSML: `<｜DSML｜ invoke name="`) and the model went on with names the request never offered.  Now the server
   writes only the wrapper (`<tool_call>\n`; DSML: `<｜DSML｜ calls>\n`) and the model writes the whole invoke tag.  A
   named choice, or "required" with one tool, still gets the whole opening (ForcedToolChoice in test_server.py
   covers the Qwen form).
2. Calls shown inside a code fence or an indented block are text, never run (0.2.9, #4702 / #4703): a fence closes
   only on a run of at least as many of its character, and a 4-space / tab block after a blank line is code.
3. A chat request with no message, or only messages with nothing in them, is a 400 (0.2.6), not a generation.
4. `undeclared_tool_calls` (0.2.12, #4622; the rule itself since 0.2.2, #3075 / #3796): "call" (our default, upstream
   31c02061) returns a well-formed call of a tool the request did not declare as a tool call; "text" (opt-in, Strata's
   default) keeps it as the text the model wrote, with `strata_rejected_tool_calls` beside it.

The DeepSeek layer (tools/v41/serve/serve_ds41.py) monkeypatches serve.frontend and serve.server on import, so its
probes run in a child process, as in test_sweep_1115.py.  Nothing here starts an engine.
"""
import json
import os
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import serve_ds41 as D
F, E = D.F, D.E
out = {}
TWO = [{"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
       {"name": "done", "parameters": {"type": "object", "properties": {}}}]

# 1. forced tool_choice openings
out["forced_required_two"] = D.S.forced_call("required", TWO)
out["forced_any_two"] = D.S.forced_call({"type": "any"}, TWO)
out["forced_required_one"] = D.S.forced_call("required", TWO[:1])
out["forced_named"] = D.S.forced_call({"type": "function", "function": {"name": "done"}}, TWO)
out["forced_tool"] = D.S.forced_call({"type": "tool", "name": "done"}, TWO)
out["calls_open"] = D.CALLS_OPEN

def parse(text, think=False, tools=TWO, step=3, **kw):
    p = F.OutputParser(thinking=think, tools=tools, **kw)
    ev = []
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish()
    return {"calls": [[e.call.name, e.call.arguments] for e in ev if e.kind == "tool_call"],
            "content": "".join(e.text for e in ev if e.kind == "content"),
            "reasoning": "".join(e.text for e in ev if e.kind == "reasoning"),
            "unread": getattr(p, "unread", None), "unread_names": getattr(p, "unread_names", None)}

P = '<｜DSML｜ parameter name="q" string="true">2+2</｜DSML｜ parameter>'
def block(name="search"):
    return D.CALLS_OPEN + '\n<｜DSML｜ invoke name="' + name + '">\n' + P + '\n</｜DSML｜ invoke>\n' + D.CALLS_CLOSE
out["block"] = block()
indent = "    " + block().replace("\n", "\n    ")

# 2. calls inside code
cases = {
    "fence": "Here is how a call looks:\n\n```xml\n" + block() + "\n```\n\nDone.",
    "fence_nested": "A file:\n\n````markdown\n```\n" + block() + "\n```\n````\n\nDone.",
    "fence_tilde": "Example:\n\n~~~\n" + block() + "\n~~~\nDone.",
    "indented": "Example:\n\n" + indent + "\n\nDone.",
    "indented_tab": "Example:\n\n\t" + block().replace("\n", "\n\t") + "\n\nDone.",
    "after_fence": "Example:\n\n```\ncode\n```\n\n" + block(),
    "after_nested_fence": "A file:\n\n````\n```\nx\n```\n````\n\n" + block(),
    "after_indented": "Example:\n\n    code line\n\nNow the call.\n\n" + block(),
    "plain": "Calling it.\n\n" + block(),
}
for k, text in cases.items():
    for step in (1, 3, 1000):
        out[f"code:{k}:{step}"] = parse(text, step=step)
        out[f"code_think:{k}:{step}"] = parse("I think." + E.thinking_end_token + text, think=True, step=step)
# a fence the thinking left open does not hide the answer's call
out["open_fence_in_thinking"] = parse("Look:\n```\nunclosed" + E.thinking_end_token + "\n\n" + block(), think=True)

# 4. undeclared tool calls
out["undeclared"] = parse("Let me.\n\n" + block("rm_rf"), deferred=False)
out["undeclared_default"] = parse("Let me.\n\n" + block("rm_rf"))
out["declared"] = parse("Let me.\n\n" + block("search"), deferred=False)
out["no_tools_declared_only"] = parse("Let me.\n\n" + block("search"), tools=None, declared_only=True, deferred=False)
out["no_tools_text"] = parse("Let me.\n\n" + block("search"), tools=None, deferred=False)
out["no_tools_default"] = parse("Let me.\n\n" + block("search"), tools=None, declared_only=True)

print("\n@@JSON@@" + json.dumps(out, ensure_ascii=False))
'''

_RESULTS = None


def ds41() -> dict:
    global _RESULTS
    if _RESULTS is None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        run = subprocess.run([sys.executable, "-c", CHILD, str(ROOT / "tools" / "v41" / "serve")], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        if "@@JSON@@" not in run.stdout:
            raise RuntimeError("probe child failed:\n" + run.stdout[-2000:] + run.stderr[-4000:])
        _RESULTS = json.loads(run.stdout.split("@@JSON@@", 1)[1])
    return _RESULTS


# ------------------------------------------------------------------------------------------------ 1. forced calls
class Ds41ForcedSeveralTools(unittest.TestCase):
    def test_required_with_several_tools_opens_only_the_calls_block(self):
        r = ds41()
        for key in ("forced_required_two", "forced_any_two"):
            with self.subTest(key=key):
                self.assertEqual(r[key], r["calls_open"] + "\n")     # the model writes `<｜DSML｜ invoke name="X">`

    def test_a_known_name_is_still_opened_whole(self):
        r = ds41()
        self.assertEqual(r["forced_required_one"], r["calls_open"] + '\n<｜DSML｜ invoke name="search">\n')
        for key in ("forced_named", "forced_tool"):
            with self.subTest(key=key):
                self.assertEqual(r[key], r["calls_open"] + '\n<｜DSML｜ invoke name="done">\n')


# ------------------------------------------------------------------------------------------------ 2. code fences
class Ds41CallsInCode(unittest.TestCase):
    QUOTED = ("fence", "fence_nested", "fence_tilde", "indented", "indented_tab")
    REAL = ("after_fence", "after_nested_fence", "after_indented", "plain")

    def test_a_call_inside_code_is_text_written_as_the_model_wrote_it(self):
        r = ds41()
        for k in self.QUOTED:
            for prefix in ("code", "code_think"):
                for step in (1, 3, 1000):
                    with self.subTest(case=k, thinking=prefix != "code", step=step):
                        got = r[f"{prefix}:{k}:{step}"]
                        self.assertEqual(got["calls"], [])
                        # the calls block stays in the text (its wrapper too), line for line
                        for line in r["block"].split("\n"):
                            self.assertIn(line, got["content"])

    def test_a_call_after_the_code_is_still_a_call(self):
        r = ds41()
        for k in self.REAL:
            for prefix in ("code", "code_think"):
                for step in (1, 3, 1000):
                    with self.subTest(case=k, thinking=prefix != "code", step=step):
                        got = r[f"{prefix}:{k}:{step}"]
                        self.assertEqual(got["calls"], [["search", {"q": "2+2"}]])
                        self.assertNotIn("DSML", got["content"])

    def test_a_fence_left_open_in_the_thinking_does_not_hide_the_answers_call(self):
        got = ds41()["open_fence_in_thinking"]
        self.assertEqual(got["calls"], [["search", {"q": "2+2"}]])


class QwenCallsInCode(unittest.TestCase):
    """The server's own parser (the Qwen form), as Strata #4702 / #4703."""
    TOOLS = [{"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}]
    CALL = "<tool_call>\n<function=search>\n<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"

    def parse(self, text, step):
        p = OutputParser(thinking=False, tools=self.TOOLS)
        ev = []
        for i in range(0, len(text), step):
            ev += p.feed(text[i:i + step])
        ev += p.finish()
        return [e.call.name for e in ev if e.kind == "tool_call"], "".join(e.text for e in ev if e.kind == "content")

    def test_quoted(self):
        cases = {"nested": "A file:\n\n````markdown\n```\n" + self.CALL + "\n```\n````\nDone.",
                 "indented": "Example:\n\n    " + self.CALL.replace("\n", "\n    ") + "\n\nDone.",
                 "tab": "Example:\n\n\t" + self.CALL.replace("\n", "\n\t") + "\n\nDone."}
        for k, text in cases.items():
            for step in (1, 4, 1000):
                with self.subTest(case=k, step=step):
                    calls, content = self.parse(text, step)
                    self.assertEqual(calls, [])
                    self.assertIn("<function=search>", content)

    def test_real(self):
        cases = {"after_nested": "A file:\n\n````\n```\nx\n```\n````\n\n" + self.CALL,
                 "after_indented": "Example:\n\n    code\n\nNow.\n\n" + self.CALL,
                 "indented_list_text": "Steps:\n1. first\n    more of the first\n" + self.CALL}
        for k, text in cases.items():
            for step in (1, 4, 1000):
                with self.subTest(case=k, step=step):
                    calls, _ = self.parse(text, step)
                    self.assertEqual(calls, ["search"])


# ------------------------------------------------------------------------------------------------ HTTP helpers
class _Http(unittest.TestCase):
    TEMPLATE = None
    SCRIPT = "Hello."

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, self.SCRIPT, max_context=32768)
        self.engine.calls = 0
        gen = self.engine.generate

        def generate(*a, **k):
            self.engine.calls += 1
            yield from gen(*a, **k)
        self.engine.generate = generate
        self.svc = Service(self.engine, self.tok, (self.TEMPLATE or ChatTemplate)(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps({"model": "m", "max_tokens": 400, **body}).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if not body.get("stream") else raw)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


# ------------------------------------------------------------------------------------------------ 3. empty requests
class _Permissive(ChatTemplate):
    """Renders a conversation with no message without an error, as DeepSeek's encoder does (serve_ds41.Ds41Template:
    "<｜begin▁of▁sentence｜>" alone, which the engine would continue)."""

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        if not messages:
            return ""
        return super().render(messages, tools, add_generation_prompt, **kw)


class EmptyChatRequest(_Http):
    TEMPLATE = _Permissive
    EMPTY = {"no messages": [], "empty text": [{"role": "user", "content": ""}],
             "blank text": [{"role": "system", "content": ""}, {"role": "user", "content": "  \n"}],
             "empty parts": [{"role": "user", "content": []}],
             "empty text part": [{"role": "user", "content": [{"type": "text", "text": ""}]}]}

    def test_chat_completions(self):
        for label, messages in self.EMPTY.items():
            for stream in (False, True):
                with self.subTest(label=label, stream=stream):
                    code, b = self.post("/v1/chat/completions", {"messages": messages, "stream": stream})
                    self.assertEqual(code, 400, b)
                    self.assertIn("messages", b["error"]["message"])
        self.assertEqual(self.engine.calls, 0)

    def test_anthropic_messages(self):
        for label, messages in self.EMPTY.items():
            with self.subTest(label=label):
                code, b = self.post("/v1/messages", {"messages": messages})
                self.assertEqual(code, 400, b)
                self.assertEqual(b["type"], "error")
        self.assertEqual(self.engine.calls, 0)

    def test_a_message_with_something_in_it_is_answered(self):
        for messages in ([{"role": "user", "content": "hi"}],
                         [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                         [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]):
            with self.subTest(messages=messages):
                code, b = self.post("/v1/chat/completions", {"messages": messages})
                self.assertEqual(code, 200, b)


# ------------------------------------------------------------------------------------------------ 4. undeclared calls
class UndeclaredToolCalls(_Http):
    SCRIPT = ("I could look this up.</think>\n\nLet me.\n<tool_call>\n<function=rm_rf>\n<parameter=path>\n/\n"
              "</parameter>\n</function>\n</tool_call>")
    TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}}}}]
    A_TOOLS = [{"name": "search", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]
    MSG = [{"role": "user", "content": "clean up"}]

    def test_call_is_the_default(self):
        self.assertEqual(self.svc.undeclared_tool_calls, "call")
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, b = self.post("/v1/chat/completions", {"messages": self.MSG, "tools": self.TOOLS, "stream": stream})
                self.assertEqual(code, 200, b)
                if not stream:
                    self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
                    self.assertEqual(b["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "rm_rf")
                    self.assertNotIn("strata_rejected_tool_calls", b)
                else:
                    self.assertIn('"name": "rm_rf"', b)
                    self.assertNotIn("strata_rejected_tool_calls", b)
        code, b = self.post("/v1/messages", {"messages": self.MSG, "tools": self.A_TOOLS})
        self.assertEqual((b["stop_reason"], b["content"][-1]["type"], b["content"][-1]["name"]), ("tool_use", "tool_use", "rm_rf"))

    def test_text_with_the_marker(self):
        self.svc.undeclared_tool_calls = "text"
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, b = self.post("/v1/chat/completions", {"messages": self.MSG, "tools": self.TOOLS, "stream": stream})
                self.assertEqual(code, 200, b)
                if stream:
                    chunks = [json.loads(line[6:]) for line in b.splitlines() if line.startswith("data: {")]
                    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
                    self.assertFalse(any(c["choices"][0]["delta"].get("tool_calls") for c in chunks))
                    self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
                    self.assertEqual(chunks[-1]["strata_rejected_tool_calls"], [{"name": "rm_rf", "reason": "undeclared"}])
                else:
                    msg = b["choices"][0]["message"]
                    content = msg["content"]
                    self.assertNotIn("tool_calls", msg)
                    self.assertEqual(b["choices"][0]["finish_reason"], "stop")
                    self.assertEqual(b["strata_rejected_tool_calls"], [{"name": "rm_rf", "reason": "undeclared"}])
                self.assertIn("<function=rm_rf>", content)

    def test_anthropic_text(self):
        self.svc.undeclared_tool_calls = "text"
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, b = self.post("/v1/messages", {"messages": self.MSG, "tools": self.A_TOOLS, "stream": stream})
                self.assertEqual(code, 200, b)
                if stream:
                    evs = [json.loads(line[6:]) for line in b.splitlines() if line.startswith("data: {")]
                    self.assertFalse(any(e.get("content_block", {}).get("type") == "tool_use" for e in evs))
                    delta = [e for e in evs if e["type"] == "message_delta"][0]
                    self.assertEqual(delta["delta"]["stop_reason"], "end_turn")
                    self.assertEqual(delta["strata_rejected_tool_calls"], [{"name": "rm_rf", "reason": "undeclared"}])
                else:
                    self.assertEqual([c["type"] for c in b["content"]], ["thinking", "text"])
                    self.assertIn("<function=rm_rf>", b["content"][1]["text"])
                    self.assertEqual(b["stop_reason"], "end_turn")
                    self.assertEqual(b["strata_rejected_tool_calls"], [{"name": "rm_rf", "reason": "undeclared"}])

    def test_no_tools_declared_stays_a_call_by_default(self):
        code, b = self.post("/v1/chat/completions", {"messages": self.MSG})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(b["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "rm_rf")

    def test_no_tools_declared_is_text_with_text(self):
        self.svc.undeclared_tool_calls = "text"
        code, b = self.post("/v1/chat/completions", {"messages": self.MSG})
        self.assertEqual(code, 200, b)
        self.assertNotIn("tool_calls", b["choices"][0]["message"])
        self.assertIn("<function=rm_rf>", b["choices"][0]["message"]["content"])

    def test_call_hands_the_call_to_the_client(self):
        self.svc.undeclared_tool_calls = "call"
        code, b = self.post("/v1/chat/completions", {"messages": self.MSG, "tools": self.TOOLS})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        call = b["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("rm_rf", {"path": "/"}))
        self.assertNotIn("strata_rejected_tool_calls", b)

    def test_a_declared_call_is_unchanged(self):
        self.engine.scripts = [self.tok.encode(self.SCRIPT.replace("rm_rf", "search").replace("path", "q"),
                                               parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        code, b = self.post("/v1/chat/completions", {"messages": self.MSG, "tools": self.TOOLS})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("strata_rejected_tool_calls", b)

    def test_setting(self):
        from serve.server import undeclared_tool_calls_setting
        env = {k: v for k, v in os.environ.items() if k != "STRATA_UNDECLARED_TOOL_CALLS"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(undeclared_tool_calls_setting({}), "call")
            self.assertEqual(undeclared_tool_calls_setting({"undeclared_tool_calls": "text"}), "text")
        with unittest.mock.patch.dict(os.environ, {"STRATA_UNDECLARED_TOOL_CALLS": "text"}):
            self.assertEqual(undeclared_tool_calls_setting({}), "text")
            self.assertEqual(undeclared_tool_calls_setting({"undeclared_tool_calls": "call"}), "call")   # the config wins
        for bad in ("yes", True, 1):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                undeclared_tool_calls_setting({"undeclared_tool_calls": bad})


class Ds41UndeclaredToolCalls(unittest.TestCase):
    def test_an_undeclared_call_is_text_with_text(self):
        got = ds41()["undeclared"]
        self.assertEqual(got["calls"], [])
        self.assertIn('invoke name="rm_rf"', got["content"])
        self.assertEqual((got["unread"], got["unread_names"]), (1, ["rm_rf"]))

    def test_default_and_declared_are_calls(self):
        r = ds41()
        self.assertEqual(r["undeclared_default"]["calls"], [["rm_rf", {"q": "2+2"}]])
        self.assertEqual(r["undeclared_default"]["unread"], 0)
        self.assertEqual(r["declared"]["calls"], [["search", {"q": "2+2"}]])
        self.assertEqual(r["declared"]["unread"], 0)

    def test_no_tools(self):
        r = ds41()
        self.assertEqual(r["no_tools_declared_only"]["calls"], [])                      # "text", chat / messages
        self.assertEqual(r["no_tools_text"]["calls"], [["search", {"q": "2+2"}]])       # "text", the Responses route (#3075)
        self.assertEqual(r["no_tools_default"]["calls"], [["search", {"q": "2+2"}]])    # "call" (default): a call, as before


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
