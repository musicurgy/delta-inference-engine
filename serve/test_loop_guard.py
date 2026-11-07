"""serve/test_loop_guard.py - the tool-call loop guard (serve/loop_guard.py, BACKLOG #69): what counts as the same call
N times in a row, the note it adds before the model's turn, and that every other request renders as before.

    python -m unittest serve.test_loop_guard -v
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import loop_guard as LG  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "v41"))
import dsv41_encoding as E  # noqa: E402  (DeepSeek V4.1's own encoder: pure Python)

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}]


def call(name="bash", args=None, cid="c"):
    args = {"command": "pytest -x"} if args is None else args
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str)
                                                         else json.dumps(args)}}


def agent(*turns, user="fix the failing test", tail=None):
    """An OpenAI chat request: the user's message, then for each turn an assistant message with its calls and one tool
    result per call.  A turn is a list of calls, or a string (an assistant message without a call)."""
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": user}]
    for k, turn in enumerate(turns):
        if isinstance(turn, str):
            msgs.append({"role": "assistant", "content": turn})
            continue
        calls = [dict(c, id=f"call_{k}_{j}") for j, c in enumerate(turn)]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        msgs += [{"role": "tool", "tool_call_id": c["id"], "content": "1 failed"} for c in calls]
    if tail:
        msgs += tail
    return {"model": "x", "max_tokens": 50, "tools": TOOLS, "messages": msgs}


def template_messages(req):
    return openai_to_messages(req)[0]


class Repeated(unittest.TestCase):
    def test_four_in_a_row(self):
        self.assertEqual(LG.repeated(template_messages(agent(*[[call()]] * 4))), ("bash", 4))
        self.assertEqual(LG.repeated(template_messages(agent(*[[call()]] * 6))), ("bash", 6))

    def test_three_is_not_a_loop(self):
        self.assertIsNone(LG.repeated(template_messages(agent(*[[call()]] * 3))))

    def test_configured_repeats(self):
        msgs = template_messages(agent(*[[call()]] * 3))
        self.assertEqual(LG.repeated(msgs, 3), ("bash", 3))
        self.assertIsNone(LG.repeated(msgs, 0))                           # off

    def test_other_arguments_break_the_run(self):
        turns = [[call()], [call(args={"command": "pytest -x -k one"})], [call()], [call()], [call()]]
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))
        self.assertEqual(LG.repeated(template_messages(agent(*turns, [call()]))), ("bash", 4))

    def test_other_tool_breaks_the_run(self):
        turns = [[call()]] * 2 + [[call("read", {"command": "pytest -x"})]] + [[call()]] * 3
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))

    def test_a_reply_without_a_call_breaks_the_run(self):
        turns = [[call()]] * 2 + ["Still failing; let me look again."] + [[call()]] * 2
        self.assertIsNone(LG.repeated(template_messages(agent(*turns))))

    def test_argument_spelling_does_not_matter(self):
        """A client's JSON string with other spacing or key order is the same call (Maya's args_key)."""
        a = call(args={"command": "ls", "timeout": 5})
        b = call(args='{"timeout": 5,   "command": "ls"}')
        msgs = template_messages(agent([a], [b], [a], [b]))
        self.assertEqual(LG.repeated(msgs), ("bash", 4))
        self.assertEqual(LG.args_key('{"a": 1}'), LG.args_key({"a": 1}))
        self.assertEqual(LG.args_key("not json"), LG.args_key("not json"))

    def test_a_batch_repeated(self):
        batch = [call(), call("read", {"path": "a.py"})]
        self.assertEqual(LG.repeated(template_messages(agent(*[batch] * 4))), ("bash, read", 4))
        self.assertIsNone(LG.repeated(template_messages(agent(*[batch] * 3, [call()]))))

    def test_only_when_the_request_answers_the_call(self):
        """The user wrote after the loop: no note (they are steering); the results with a user text beside them (Claude
        Code's reminders): still the loop."""
        stop = [{"role": "user", "content": "stop repeating yourself"}]
        req = agent(*[[call()]] * 4)
        req["messages"] = req["messages"][:-1]                            # the last call's result not sent yet
        self.assertIsNone(LG.repeated(template_messages(req)))
        msgs = template_messages(agent(*[[call()]] * 4))
        self.assertIsNone(LG.repeated(msgs + [{"role": "assistant", "content": "Done."}] + stop))
        self.assertEqual(LG.repeated(msgs + [{"role": "user", "content": "<system-reminder>x</system-reminder>"}]),
                         ("bash", 4))
        self.assertIsNone(LG.repeated([]))
        self.assertIsNone(LG.repeated(template_messages({"messages": [{"role": "user", "content": "hi"}]})))

    def test_anthropic_conversation(self):
        msgs = [{"role": "user", "content": "fix it"}]
        for k in range(4):
            msgs.append({"role": "assistant", "content": [
                {"type": "text", "text": "Running the tests."},
                {"type": "tool_use", "id": f"t{k}", "name": "Bash", "input": {"command": "npm test"}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}",
                                                      "content": "1 failing"}]})
        got = LG.repeated(anthropic_to_messages({"messages": msgs, "max_tokens": 10})[0])
        self.assertEqual(got, ("Bash", 4))

    def test_setting(self):
        self.assertEqual(LG.setting(None), 4)
        self.assertEqual(LG.setting(True), 4)
        self.assertEqual(LG.setting(False), 0)
        self.assertEqual(LG.setting({"repeats": 3}), 3)
        self.assertEqual(LG.setting({}), 4)
        for bad in ({"repeats": 1}, {"repeats": True}, {"repeats": "4"}, {"times": 4}, "on", 4):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LG.setting(bad)


class QwenPrompt(unittest.TestCase):
    """The server's own (Qwen) template refuses a late system message: the note is a user turn there."""

    def svc(self, repeats=LG.REPEATS):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.loop_guard_repeats = repeats
        return svc

    def prompt(self, svc, req):
        messages, tools, kw = openai_to_messages(req)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids, _, _ = svc.prepare(messages, tools, kw, 10)
        return svc.tok.decode(ids), out.getvalue()

    def test_no_repeats_render_as_before(self):
        """Byte for byte: guard on and guard off give the same ids for every request that is not a loop."""
        reqs = [agent(*[[call()]] * 3), agent([call()], [call("read", {"path": "x"})], [call()], [call()]),
                agent(*[[call()]] * 2 + ["thinking out loud"] + [[call()]] * 3), {"model": "x", "messages": [
                    {"role": "user", "content": "hi"}]}]
        for req in reqs:
            with self.subTest(n=len(req["messages"])):
                on, log = self.prompt(self.svc(), req)
                off, _ = self.prompt(self.svc(0), req)
                self.assertEqual(on, off)
                self.assertEqual(log, "")

    def test_note_before_the_models_turn(self):
        req = agent(*[[call()]] * 4)
        before = copy.deepcopy(req)
        on, log = self.prompt(self.svc(), req)
        off, _ = self.prompt(self.svc(0), req)
        self.assertEqual(req, before)                                    # the client's conversation is not changed
        self.assertIn("[strata] loop guard: bash repeated 4 times", log)
        self.assertEqual(log.count("loop guard"), 1)
        note = LG.note("bash", 4)
        self.assertIn(note, on)
        head = off[:off.rindex("<|im_start|>assistant")]                 # everything up to the last tool result
        self.assertTrue(on.startswith(head))
        self.assertEqual(on[len(head):], "<|im_start|>user\n" + note + "<|im_end|>\n" + off[len(head):])

    def test_over_http(self):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "ok", max_context=1 << 20)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                for api, path, body in (("openai", "/v1/chat/completions", agent(*[[call()]] * 5)),
                                        ("anthropic", "/v1/messages", None)):
                    if body is None:
                        msgs = [{"role": "user", "content": "go"}]
                        for k in range(4):
                            msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{k}", "name": "bash",
                                                                         "input": {"command": "make"}}]},
                                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}",
                                                                   "content": "error"}]}]
                        body = {"model": "x", "max_tokens": 20, "messages": msgs, "tools": [
                            {"name": "bash", "input_schema": {"type": "object", "properties": {}}}]}
                    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                 data=json.dumps(body).encode(), headers={
                                                     "Content-Type": "application/json",
                                                     "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        r.read()
                    with self.subTest(api=api):
                        self.assertIn("[Note from the server, not the user", tok.decode(eng.last_prompt))
            log = out.getvalue()
            self.assertIn("[strata] loop guard: bash repeated 5 times", log)
            self.assertIn("[strata] loop guard: bash repeated 4 times", log)
        finally:
            httpd.shutdown()
            httpd.server_close()


class Ds41Template:
    """DeepSeek V4.1's encoder behind the server's template interface (as tools/v41/serve/serve_ds41.py renders it,
    without importing that module, which patches the server for good)."""

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"type": "function", "id": None, "function": dict(c["function"])}
                                   for c in m["tool_calls"]]
            out.append(m)
        if tools:
            tools = [{"type": "function", "function": t} for t in tools]
            if out and out[0].get("role") == "system":
                out[0] = dict(out[0], tools=tools)
            else:
                out.insert(0, {"role": "system", "content": "", "tools": tools})
        return E.encode_messages(out, thinking_mode="chat" if kw.get("enable_thinking") is False else "thinking",
                                 reasoning_effort=kw.get("reasoning_effort"))


class DeepSeekPrompt(unittest.TestCase):
    """DeepSeek V4.1 takes a system message mid-conversation: the note is one, after the tool result, and the
    calls in the prompt stay DSML (Maya #844: the DSML case)."""

    def prompt(self, req, repeats=LG.REPEATS):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, Ds41Template())
        svc.loop_guard_repeats = repeats
        messages, tools, kw = openai_to_messages(req)
        with contextlib.redirect_stdout(io.StringIO()):
            ids, _, _ = svc.prepare(messages, tools, kw, 10)
        return tok.decode(ids), svc

    def test_note_as_a_system_turn(self):
        dsml = call(args={"command": "cat log | tail -n 40", "timeout": 30})
        req = agent(*[[dsml]] * 4)
        on, svc = self.prompt(req)
        off, _ = self.prompt(req, 0)
        self.assertEqual(svc.note_role, "system")
        note = LG.note("bash", 4)
        self.assertEqual(on.count(note), 1)
        self.assertEqual(on.count(E.dsml_token + E.tool_calls_block_name), off.count(E.dsml_token + E.tool_calls_block_name))
        self.assertIn('string="false">30<', on)                       # a typed argument, as DSML writes it
        head = off[:off.rindex(E.ASSISTANT_SP_TOKEN)]
        self.assertTrue(on.startswith(head))
        self.assertEqual(on[len(head):], E.SYSTEM_SP_TOKEN + note + off[len(head):])
        self.assertTrue(on.endswith(E.ASSISTANT_SP_TOKEN + E.thinking_start_token))

    def test_no_repeats_render_as_before(self):
        for req in (agent(*[[call()]] * 3), agent([call()], [call(args={"command": "ls"})], [call()], [call()])):
            with self.subTest(n=len(req["messages"])):
                self.assertEqual(self.prompt(req)[0], self.prompt(req, 0)[0])


if __name__ == "__main__":
    unittest.main()
