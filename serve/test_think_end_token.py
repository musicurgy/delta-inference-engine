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


if __name__ == "__main__":
    unittest.main()
