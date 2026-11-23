"""serve/test_restore_reasoning.py - an earlier answer sent back without its thinking gets the thinking this server wrote
for it (serve/reasoning_memory.py, after upstream 3f0cb0c2; mock engine, no GPU).

The prompt with the thinking put back must equal, byte for byte, the prompt of a client that kept the thinking - i.e.
what the model actually read and wrote, so the engine's conversation cache matches it.

    python -m unittest serve.test_restore_reasoning -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v41"))
import dsv41_encoding as E  # noqa: E402
from serve.frontend import ChatTemplate, ToolCall  # noqa: E402
from serve.reasoning_memory import ReasoningMemory  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

CALL = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
SCRIPTS = ["<think>\nI should read a.txt first.\n</think>\n\n" + CALL,
           "<think>\nThe file says hello.\n</think>\n\nIt says hello."]
TOOLS_OAI = [{"type": "function", "function": {"name": "read", "description": "read a file",
                                               "parameters": {"type": "object",
                                                              "properties": {"path": {"type": "string"}}}}}]
TOOLS_ANT = [{"name": "read", "description": "read a file",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]


class _Http(unittest.TestCase):
    scripts = SCRIPTS

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, cls.scripts, max_context=16384)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.reasoning_memory.clear()
        self.svc.reasoning_memory.on = True
        self.engine.turns = 0

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        return raw, self.tok.decode(self.engine.last_prompt)


class OpenCodeLike(_Http):
    """OpenAI chat completions with tools; the client sends the tool loop back without reasoning_content."""

    def chat(self, messages, stream=False):
        raw, prompt = self.post("/v1/chat/completions", {"messages": messages, "tools": TOOLS_OAI, "stream": stream})
        return (None if stream else json.loads(raw)), prompt

    def history(self, keep_thinking, first, second=None):
        m1 = first["choices"][0]["message"]
        turn1 = {"role": "assistant", "content": m1.get("content") or "", "tool_calls": m1["tool_calls"],
                 **({"reasoning_content": m1["reasoning_content"]} if keep_thinking else {})}
        msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "what is in a.txt?"},
                turn1, {"role": "tool", "tool_call_id": m1["tool_calls"][0]["id"], "content": "hello"}]
        if second is not None:
            m2 = second["choices"][0]["message"]
            msgs += [{"role": "assistant", "content": m2["content"],
                      **({"reasoning_content": m2["reasoning_content"]} if keep_thinking else {})},
                     {"role": "user", "content": "thanks; and b.txt?"}]
        return msgs

    def test_tool_loop_prompt_equals_the_kept_thinking_one(self):
        first, _ = self.chat([{"role": "system", "content": "You are a coding agent."},
                              {"role": "user", "content": "what is in a.txt?"}])
        self.assertIn("read a.txt first", first["choices"][0]["message"]["reasoning_content"])
        second, dropped = self.chat(self.history(False, first))
        self.assertIn("I should read a.txt first.", dropped)
        # the reference: a client that kept the thinking (the reply list replays from the start for it)
        self.svc.reasoning_memory.clear()
        self.engine.turns = 1
        _, kept = self.chat(self.history(True, first))
        self.assertEqual(dropped, kept)

    def test_next_user_turn_gets_both_back(self):
        first, _ = self.chat([{"role": "system", "content": "You are a coding agent."},
                              {"role": "user", "content": "what is in a.txt?"}])
        second, _ = self.chat(self.history(False, first))
        _, dropped = self.chat(self.history(False, first, second))
        self.assertIn("I should read a.txt first.", dropped)
        self.assertIn("The file says hello.", dropped)
        _, kept = self.chat(self.history(True, first, second))
        self.assertEqual(dropped, kept)

    def test_streamed_reply_is_remembered(self):
        self.chat([{"role": "system", "content": "You are a coding agent."},
                   {"role": "user", "content": "what is in a.txt?"}], stream=True)
        self.assertEqual(len(self.svc.reasoning_memory), 1)

    def test_changed_history_and_other_answers_stay_as_sent(self):
        first, _ = self.chat([{"role": "system", "content": "You are a coding agent."},
                              {"role": "user", "content": "what is in a.txt?"}])
        msgs = self.history(False, first)
        msgs[1] = {"role": "user", "content": "what is in a.txt??"}       # an earlier message differs
        _, prompt = self.chat(msgs)
        self.assertNotIn("I should read a.txt first.", prompt)
        msgs = self.history(False, first)
        msgs[2] = dict(msgs[2], tool_calls=[dict(msgs[2]["tool_calls"][0], function={
            "name": "read", "arguments": json.dumps({"path": "b.txt"})})])  # another call
        _, prompt = self.chat(msgs)
        self.assertNotIn("I should read a.txt first.", prompt)
        msgs = self.history(False, first)
        msgs[2] = dict(msgs[2], reasoning_content="the client's own")      # the client's thinking wins
        _, prompt = self.chat(msgs)
        self.assertIn("the client's own", prompt)
        self.assertNotIn("I should read a.txt first.", prompt)

    def test_switched_off(self):
        first, _ = self.chat([{"role": "system", "content": "You are a coding agent."},
                              {"role": "user", "content": "what is in a.txt?"}])
        self.svc.reasoning_memory.on = False
        _, prompt = self.chat(self.history(False, first))
        self.assertNotIn("I should read a.txt first.", prompt)


class ClaudeCodeLike(_Http):
    """Anthropic messages with tools; the history's assistant turns come back as text + tool_use, no thinking blocks."""

    def msg(self, messages):
        raw, prompt = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "system": "You are Claude Code.",
                                                  "tools": TOOLS_ANT, "messages": messages})
        return json.loads(raw), prompt

    def turns(self, keep_thinking, first):
        blocks = [b for b in first["content"] if keep_thinking or b["type"] != "thinking"]
        use = next(b for b in first["content"] if b["type"] == "tool_use")
        return [{"role": "user", "content": "what is in a.txt?"}, {"role": "assistant", "content": blocks},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "hello"}]}]

    def test_thinking_blocks_dropped(self):
        first, _ = self.msg([{"role": "user", "content": "what is in a.txt?"}])
        self.assertTrue(any(b["type"] == "thinking" for b in first["content"]))
        _, dropped = self.msg(self.turns(False, first))
        self.assertIn("I should read a.txt first.", dropped)
        self.svc.reasoning_memory.clear()
        self.engine.turns = 1
        _, kept = self.msg(self.turns(True, first))
        self.assertEqual(dropped, kept)


class SameAnswerTwice(_Http):
    """The same answer after different messages: each turn gets its own thinking."""
    scripts = ["<think>\nplan one\n</think>\n\nok", "<think>\nplan two\n</think>\n\nok", "<think>\nx\n</think>\n\ndone"]

    def chat(self, messages):
        raw, prompt = self.post("/v1/chat/completions", {"messages": messages, "tools": TOOLS_OAI})
        return json.loads(raw), prompt

    def test_each_gets_its_own(self):
        self.chat([{"role": "user", "content": "a"}])
        self.chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "b"}])
        _, prompt = self.chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": "ok"},
                               {"role": "user", "content": "b"}, {"role": "assistant", "content": "ok"},
                               {"role": "user", "content": "c"}])
        self.assertLess(prompt.index("plan one"), prompt.index("plan two"))


class Bounds(unittest.TestCase):
    def test_entries_and_chars(self):
        mem = ReasoningMemory(keep=2, keep_chars=10)
        _, ctx, _, _ = mem.restore([{"role": "user", "content": "q"}])
        for i in range(3):
            mem.remember(ctx, f"answer {i}", [], "abc")
        self.assertEqual(len(mem), 2)
        mem.remember(ctx, "long", [], "x" * 9)
        self.assertEqual(len(mem), 1)
        self.assertLessEqual(mem.chars, 10)

    def test_pending_handoff_is_bounded_and_by_identity(self):
        mem = ReasoningMemory()
        ids = [1, 2, 3]
        mem.prepared(ids, b"c")
        self.assertIsNone(mem.context_of([1, 2, 3]))
        self.assertEqual(mem.context_of(ids), b"c")
        self.assertIsNone(mem.context_of(ids))
        for i in range(20):
            mem.prepared([i], b"x")
        self.assertLessEqual(len(mem.pending), 8)

    def test_callers_dicts_untouched(self):
        mem = ReasoningMemory()
        hist = [{"role": "user", "content": "q"}]
        _, ctx, _, _ = mem.restore(hist)
        mem.remember(ctx, "a", [], "t")
        turn = {"role": "assistant", "content": "a"}
        out, _, n, _ = mem.restore(hist + [turn, {"role": "user", "content": "q2"}])
        self.assertEqual(n, 1)
        self.assertEqual(out[1]["reasoning_content"], "t")
        self.assertNotIn("reasoning_content", turn)


class DeepSeekEncoder(unittest.TestCase):
    """What DeepSeek V4.1's encoder does with earlier-turn thinking, and the put-back through it."""

    SYS = {"role": "system", "content": "agent", "tools": TOOLS_OAI}

    def history(self, with_thinking, tools=True):
        call = {"type": "function", "function": {"name": "read", "arguments": {"path": "a.txt"}}}
        a1 = {"role": "assistant", "content": "", "tool_calls": [call]}
        a2 = {"role": "assistant", "content": "It says hello."}
        if with_thinking:
            a1, a2 = dict(a1, reasoning_content="read it"), dict(a2, reasoning_content="it says hello")
        return ([self.SYS] if tools else [{"role": "system", "content": "agent"}]) + [
            {"role": "user", "content": "a.txt?"}, a1, {"role": "tool", "content": "hello"}, a2,
            {"role": "user", "content": "b.txt?"}]

    def test_template_keeps_old_thinking_with_tools(self):
        kept = E.encode_messages(self.history(True), "thinking")
        self.assertIn("read it</think>", kept)
        self.assertIn("it says hello</think>", kept)
        self.assertNotEqual(kept, E.encode_messages(self.history(False), "thinking"))

    def test_template_drops_thinking_before_the_last_user_without_tools(self):
        self.assertEqual(E.encode_messages(self.history(True, tools=False), "thinking"),
                         E.encode_messages(self.history(False, tools=False), "thinking"))

    def test_put_back_renders_as_kept(self):
        mem = ReasoningMemory()
        full = self.history(False)
        _, ctx1, _, _ = mem.restore(full[:2])
        mem.remember(ctx1, "", [ToolCall("read", {"path": "a.txt"})], "read it")
        _, ctx2, n, _ = mem.restore(full[:4])
        self.assertEqual(n, 1)
        mem.remember(ctx2, "It says hello.", [], "it says hello")
        out, _, n, _ = mem.restore(full)
        self.assertEqual(n, 2)
        self.assertEqual(E.encode_messages(out, "thinking"), E.encode_messages(self.history(True), "thinking"))


if __name__ == "__main__":
    unittest.main()
