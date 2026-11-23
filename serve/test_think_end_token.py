"""Token mode (Project Maya 4dd2892): with DeepSeek's end-of-thinking token, only that token ends the reasoning; the
text `</think>` written by the model - in prose or quoted - stays reasoning.    python -m unittest serve.test_think_end_token"""
import unittest

from serve.frontend import OutputParser


def run(parser, pieces):
    """pieces: str = model text; True = the end-of-thinking token (its text is </think>)."""
    evs = []
    for p in pieces:
        if p is True:
            parser.end_tokens += 1
            p = "</think>"
        evs += parser.feed(p)
    evs += parser.finish() if hasattr(parser, "finish") else []
    return "".join(e.text or "" for e in evs if e.kind == "reasoning"), "".join(e.text or "" for e in evs if e.kind == "content")


class ThinkEndToken(unittest.TestCase):
    def tok(self):
        p = OutputParser(thinking=True)
        p.end_tokens = 0
        return p

    def test_prose_tag_stays_reasoning(self):
        r, c = run(self.tok(), ["The tag ", "</think>", " closes it, so I", " answer now.", True, "\n\nHi there."])
        self.assertIn("The tag </think> closes it", r)
        self.assertEqual(c.strip(), "Hi there.")

    def test_quoted_tag_stays_reasoning(self):
        r, c = run(self.tok(), ["Use `", "</think>", "` here.", True, "\n\nDone."])
        self.assertIn("`</think>`", r)
        self.assertEqual(c.strip(), "Done.")

    def test_real_token_ends(self):
        r, c = run(self.tok(), ["thinking", True, "\n\nanswer"])
        self.assertEqual(r.strip(), "thinking")
        self.assertEqual(c.strip(), "answer")

    def test_text_mode_unchanged(self):
        p = OutputParser(thinking=True)          # end_tokens None: the #537 text rule as before
        r, c = run(p, ["thinking", "</think>", "\n\nanswer"])
        self.assertEqual(c.strip(), "answer")


class TokenModeSelected(unittest.TestCase):
    """11-15: Service picks token mode when </think> is ONE token, even when it encodes to itself with and without
    special parsing (DeepSeek-V4.1's 128822 is user-defined).  The old `plain != ids` test left DeepSeek in text mode,
    and the #537 quote rule took the real end after a quote for a quoted tag: '..."</think>You're welcome' came back
    as reasoning with an empty answer (a greeting survey: 2 of 186 replies)."""

    def service(self, end_ids):
        from pathlib import Path
        from serve.frontend import THINK_END, ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service

        class Tok(ByteTokenizer):
            def encode(self, text, parse_special=False, plain=()):
                return list(end_ids) if text == THINK_END else super().encode(text, parse_special=parse_special,
                                                                               plain=plain)
        tok = Tok()
        root = Path(__file__).resolve().parents[1]
        return Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(root / "serve/chat_template.jinja"))

    def test_one_dedicated_token_is_token_mode(self):
        self.assertEqual(self.service([999]).think_end_id, 999)

    def test_tag_spelled_with_several_tokens_keeps_text_mode(self):
        self.assertIsNone(self.service([1, 2, 3]).think_end_id)

    def test_end_after_a_quote_is_the_end_in_token_mode(self):
        r, c = run(ThinkEndToken.tok(None), ['Maybe "You\'re welcome."', True, "You're welcome!"])
        self.assertEqual(c.strip(), "You're welcome!")
        self.assertNotIn("</think>", r)


if __name__ == "__main__":
    unittest.main()
