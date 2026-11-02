"""A whitespace-only replayed reasoning renders exactly like an absent one (antirez/ds4 a8ce74a): OpenAI chat,
Anthropic thinking blocks and the Responses API.    python -m unittest serve.test_blank_reasoning"""
import unittest

from serve import frontend as F
from serve import responses as R


class BlankReasoning(unittest.TestCase):
    CALL = [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]

    def test_openai_blank_is_absent(self):
        pad = {"messages": [{"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "", "reasoning_content": " \n", "tool_calls": self.CALL}]}
        none = {"messages": [{"role": "user", "content": "Hello"},
                             {"role": "assistant", "content": "", "tool_calls": self.CALL}]}
        self.assertEqual(F.openai_to_messages(pad)[0], F.openai_to_messages(none)[0])

    def test_openai_real_reasoning_kept(self):
        req = {"messages": [{"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "x", "reasoning_content": "because"}]}
        self.assertEqual(F.openai_to_messages(req)[0][1].get("reasoning_content"), "because")

    def test_anthropic_blank_thinking_is_absent(self):
        msgs = [{"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": " "}, {"type": "text", "text": "ok"}]}]
        out = F.anthropic_to_messages({"messages": msgs})[0]
        self.assertNotIn("reasoning_content", out[-1])

    def test_reasoning_given(self):
        for v, want in ((" ", False), ("", False), (None, False), ("\n\t", False), ("x", True), (" y ", True)):
            self.assertEqual(F.reasoning_given(v), want, repr(v))


if __name__ == "__main__":
    unittest.main()
