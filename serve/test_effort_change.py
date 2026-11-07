"""serve/test_effort_change.py - C11 / R4: DeepSeek V4.1 puts "Reasoning Effort: N" at the start of the prompt, so a
request that changes the effort, or turns thinking on or off, in a running conversation reads it all again.  The
server says so in one log line; nothing else changes.

    python -m unittest serve.test_effort_change -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, effort_header  # noqa: E402
from serve.test_loop_guard import Ds41Template  # noqa: E402  (DeepSeek's own encoder behind the template interface)

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a coding agent. " + "Follow the repository's rules. " * 80       # an agent's long system prompt


def conversation(turns, system=SYSTEM):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "fix the bug"}]
    for k in range(turns):
        msgs += [{"role": "assistant", "content": f"step {k} done"}, {"role": "user", "content": f"next {k}"}]
    return msgs


class Header(unittest.TestCase):
    def test_labels(self):
        t = Ds41Template()
        msgs = conversation(1)
        heads = set()
        for kw, label in (({}, "high"), ({"reasoning_effort": 75}, "high"), ({"reasoning_effort": 100}, "max"),
                          ({"reasoning_effort": 62}, "medium"), ({"reasoning_effort": 50}, "low"),
                          ({"reasoning_effort": 33}, "33"), ({"enable_thinking": False}, "off")):
            with self.subTest(kw=kw):
                got, head, length = effort_header(t.render(msgs, **kw))
                self.assertEqual(got, label)
                self.assertNotIn("Reasoning Effort", head)
                heads.add(head)
        self.assertEqual(len(heads), 1)                   # without the header every one starts the same

    def test_no_header(self):
        self.assertEqual(effort_header("<|im_start|>system\nhi")[0], "off")
        self.assertEqual(effort_header("x" * 600 + "Reasoning Effort: 5 ")[0], "off")   # only at the start


class Log(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, Ds41Template())

    def send(self, msgs, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids, _, _ = self.svc.prepare(msgs, None, dict(kw), 10)
        self.assertEqual(ids, self.svc.encode_prompt(msgs, None, dict(kw)))   # the prompt itself is unchanged
        return [line for line in out.getvalue().splitlines() if "effort changed" in line]

    def test_change_in_one_conversation(self):
        self.assertEqual(self.send(conversation(1)), [])
        self.assertEqual(self.send(conversation(2)), [])                       # same effort: nothing
        self.assertEqual(self.send(conversation(3), reasoning_effort=100),
                         ["[strata] effort changed (high -> max): the whole conversation is read again"])
        self.assertEqual(self.send(conversation(4), reasoning_effort=100), [])
        self.assertEqual(self.send(conversation(5), enable_thinking=False),
                         ["[strata] effort changed (max -> off): the whole conversation is read again"])
        self.assertEqual(self.send(conversation(6), reasoning_effort=50),
                         ["[strata] effort changed (off -> low): the whole conversation is read again"])

    def test_other_conversation_is_not_a_change(self):
        self.send(conversation(2))
        self.assertEqual(self.send(conversation(1, system="Write a title for this chat. " * 60),
                                   enable_thinking=False), [])

    def test_helper_calls_in_between(self):
        """Claude Code's helper calls (their own prompts, thinking off) come between the main turns: the change is
        still seen against the main conversation's last request."""
        self.send(conversation(2))
        self.send([{"role": "user", "content": "Command: ls\nOutput: a.py\nfilepaths?"}], enable_thinking=False)
        self.assertEqual(self.send(conversation(3), reasoning_effort=100),
                         ["[strata] effort changed (high -> max): the whole conversation is read again"])

    def test_qwen_template_never_logs(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(self.send(conversation(1)), [])
        self.assertEqual(self.send(conversation(2), enable_thinking=False), [])
        self.assertEqual(self.send(conversation(3), reasoning_effort="low"), [])


if __name__ == "__main__":
    unittest.main()
