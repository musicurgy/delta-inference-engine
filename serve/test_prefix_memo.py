"""serve/test_prefix_memo.py - the agent-turn costs outside the engine (brainstorm_1107 idea 1, session_gap.md): the
tokenizer's prefix memo gives exactly a whole encode's ids, and the literal-mark pattern matches as the alternation did.

    python -m unittest serve.test_prefix_memo -v
"""
from __future__ import annotations

import contextlib
import io
import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend as F  # noqa: E402

SPECIALS = ["<think>", "</think>", "<tool_call>", "<|im_start|>", "<|im_end|>", "<|im_end|>x", "<|endoftext|>"]
WORDS = ["the", "think", "plain", "thin", "<", "</", "<|", "|>", "im_", "end", " ", "\n", "  ", "漢字", "🙂", "é",
         "0123", "x", "-", "<thi", "</think", "<|im_e"]


def tokenizer(memo=True):
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    types = [1] * 256
    rules = ["t h", "th e", "th i", "thi n", "thin k", "p l", "pl a", "pla i", "plai n", "< /", "< |", "| >"]
    for rule in rules:
        left, right = rule.split(" ")
        tokens.append(left + right)
        types.append(1)
    tokens += SPECIALS
    types += [4, 4, 4, 3, 3, 3, 3]
    tok = ST.Tokenizer(tokens, rules, types)
    tok.PREFIX_MEMO_MIN = 0                     # the test texts are short
    if not memo:
        tok.PREFIX_MEMO = 0
    return tok


def chunk(rng, n):
    return "".join(rng.choice(WORDS + SPECIALS) for _ in range(n))


class PrefixMemo(unittest.TestCase):
    def test_same_ids_as_a_whole_encode(self):
        rng = random.Random(1107)
        tok, ref = tokenizer(), tokenizer(memo=False)
        for conv in range(30):
            text = chunk(rng, 40)
            for turn in range(25):
                r = rng.random()
                if r < 0.6:                                    # the conversation grows
                    text += chunk(rng, rng.randrange(1, 30))
                elif r < 0.75:                                 # cut anywhere (inside a literal too), then grown
                    text = text[:rng.randrange(len(text) + 1)] + chunk(rng, rng.randrange(0, 10))
                elif r < 0.9:                                  # a change in the middle (a cleared tool result)
                    i = rng.randrange(len(text) + 1)
                    text = text[:i] + chunk(rng, 3) + text[i + rng.randrange(0, 20):]
                else:                                          # the header changes (an effort change)
                    text = chunk(rng, 2) + text[rng.randrange(0, 5):]
                spans = ()
                if rng.random() < 0.4:                         # plain spans, kept turn to turn or not
                    lit = [m.span() for m in re.finditer(r"</?think>|<\|im_end\|>", text)]
                    spans = [s for s in lit if rng.random() < 0.5] if rng.random() < 0.5 else lit
                    if rng.random() < 0.2 and text:
                        a = rng.randrange(len(text))
                        spans = list(spans) + [(a, a + rng.randrange(0, 12))]
                    if rng.random() < 0.5:
                        spans = tuple(spans)
                for ps in (True, False):
                    with self.subTest(conv=conv, turn=turn, parse_special=ps):
                        self.assertEqual(tok.encode(text, ps, plain=spans), ref.encode(text, ps, plain=spans))

    def test_a_longer_text_tokenizes_only_its_new_part(self):
        tok = tokenizer()
        seen = []
        plain = tok._encode_plain
        tok._encode_plain = lambda s: (seen.append(len(s)), plain(s))[1]
        head = "<|im_start|>system\n" + "the plain think " * 200 + "<|im_end|>\n<|im_start|>user\nthe question<|im_end|>\n"
        first = tok.encode(head + "<|im_start|>assistant\n<think>", True)
        seen.clear()
        more = tok.encode(head + "<|im_start|>assistant\n</think>a call<|im_end|>\n<|im_start|>tool\nresult<|im_end|>"
                          "\n<|im_start|>assistant\n<think>", True)
        self.assertLess(sum(seen), 80)                # the 3K characters before the shared turn are not read again
        self.assertEqual(more, tokenizer(memo=False).encode(head + "<|im_start|>assistant\n</think>a call<|im_end|>\n"
                                                            "<|im_start|>tool\nresult<|im_end|>\n<|im_start|>assistant\n"
                                                            "<think>", True))
        self.assertEqual(first, tok.encode(head + "<|im_start|>assistant\n<think>", True))   # still remembered

    def test_spans_that_differ_before_the_cut_are_not_reused(self):
        tok, ref = tokenizer(), tokenizer(memo=False)
        text = "a </think> b " * 50 + "<|im_end|>tail"
        i = text.index("</think>")
        tok.encode(text, True, plain=[(i, i + 8)])
        for spans in ([], [(i, i + 8), (i + 13, i + 21)], [(i + 13, i + 21)]):
            self.assertEqual(tok.encode(text + " more", True, plain=spans), ref.encode(text + " more", True, plain=spans))

    def test_memo_is_bounded_and_short_texts_are_not_remembered(self):
        tok = tokenizer()
        for k in range(10):
            tok.encode(f"conversation {k} " * 20, True)
        self.assertEqual(len(tok._memo), tok.PREFIX_MEMO)
        tok2 = tokenizer()
        tok2.PREFIX_MEMO_MIN = 4096
        tok2.encode("short <think> text", True)
        self.assertEqual(tok2._memo, ())

    def test_a_subclass_with_its_own_matching_is_not_bypassed(self):
        class Own(ST.Tokenizer):
            calls = 0

            def _encode_matching(self, text, pat, plain=()):
                Own.calls += 1
                return super()._encode_matching(text, pat, plain)
        base = tokenizer()
        own = Own(base.tokens, [" ".join(k) for k in base.ranks], base.token_types)
        own.PREFIX_MEMO_MIN = 0
        own.encode("the plain think " * 10, True)
        own.encode("the plain think " * 11, True)
        self.assertEqual(Own.calls, 2)

    def test_check_mode_catches_a_wrong_reuse(self):
        tok = tokenizer()
        tok.CHECK = True
        text = "the plain <think> think " * 30
        good = tokenizer(memo=False).encode(text + "more", True)
        tok.encode(text, True)
        pat, old, ids, ends, counts, spans = tok._memo[0]
        ids[0] = (ids[0] + 1) % 256                          # a memo gone wrong
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = tok.encode(text + "more", True)
        self.assertEqual(got, good)
        self.assertIn("tokenize check", err.getvalue())

    def test_common_prefix(self):
        rng = random.Random(5)
        for _ in range(300):
            a = "".join(rng.choice("ab") for _ in range(rng.randrange(0, 70000 if rng.random() < 0.05 else 200)))
            b = a[:rng.randrange(len(a) + 1)] + "".join(rng.choice("abc") for _ in range(rng.randrange(0, 50)))
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            self.assertEqual(ST.common_prefix(a, b), n)


class MarkPattern(unittest.TestCase):
    """unmark_think_literals' pattern: a class of ranges finds exactly what the alternation found."""

    def test_same_matches_as_the_alternation(self):
        rng = random.Random(3)
        for n in (0, 1, 7, 1300):
            tags = F.literal_tags([f"<|c{i}|>" for i in range(n)])
            marks = {v: k for k, v in tags.items()}
            subsets = [marks, {m: marks[m] for m in rng.sample(list(marks), max(1, len(marks) // 3))}]
            noise = list("ab<|>-]^\\[") + [chr(F.CONTROL_MARK0 - 1), chr(0xF0E00), chr(0xF0E0B)]
            for mk in subsets:
                alternation = re.compile("|".join(map(re.escape, mk)))
                ranges = F._marks_re(mk)
                for _ in range(100):
                    text = "".join(rng.choice(list(marks) + noise) for _ in range(200))
                    self.assertEqual([m.span() for m in ranges.finditer(text)],
                                     [m.span() for m in alternation.finditer(text)])

    def test_marks_longer_than_one_character_keep_the_alternation(self):
        p = F._marks_re({"ab": "x", "c": "y"})
        self.assertEqual([m.group(0) for m in p.finditer("abcab")], ["ab", "c", "ab"])

    def test_unmark_round_trip_with_control_tokens(self):
        tags = F.literal_tags([f"<|c{i}|>" for i in range(50)])
        text = "say <|c3|> and </think> then <|c49|>."
        marked = F._mark(text, tags)
        prompt, spans = F.unmark_think_literals(marked, tags)
        self.assertEqual(prompt, text)
        self.assertEqual([prompt[a:b] for a, b in spans], ["<|c3|>", "</think>", "<|c49|>"])


if __name__ == "__main__":
    unittest.main()
