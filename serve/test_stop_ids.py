"""serve/test_stop_ids.py - only a one-token end marker is a stop id.

Service took the ids of "<|im_end|>" and "<|endoftext|>" as its stop ids whatever the tokenizer made of them.  A
tokenizer without them as single tokens (DeepSeek-V4.1's cuts them into '<', '|', 'im', '_end', '|', '>') made every
piece a stop id: a server run without the DeepSeek launcher ended the reply in front of the first "|" or "im"
("Nerazim" came out as "Neraz").  A tokenizer that has both as one token each (Qwen's, the tests' byte tokenizer) keeps
exactly the stop ids it had.  After upstream Xolotl / Niko1221/Strata 43f32b30.

    python -m unittest serve.test_stop_ids -v
"""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import DEEPSEEK_EOS, IM_END, ByteTokenizer, MockEngine, Service, end_token_ids, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EOS = 900


class SplitTokenizer(ByteTokenizer):
    """The byte tokenizer without Qwen's end markers as specials: they come out as one id per byte, as DeepSeek's
    tokenizer cuts them into pieces; its own end token is EOS."""

    def __init__(self, eos=EOS):
        super().__init__()
        self.special_ids = {} if eos is None else {"eos": eos}

    def encode(self, text, parse_special=False, plain=()):
        return list(text.encode("utf-8"))


class FolderTokenizer(SplitTokenizer):
    """What server.py builds from a DeepSeek tokenizer folder (vocab.json, merges.txt, token_type.json): no
    special_ids at all, Qwen's markers in pieces, and its end-of-sentence marker one control token (id 1)."""

    def __init__(self, special_ids=None):
        super().__init__(eos=None)
        self.special_ids = dict(special_ids or {})

    def encode(self, text, parse_special=False, plain=()):
        if parse_special and text == DEEPSEEK_EOS:
            return [1]
        return list(text.encode("utf-8"))


class EndTokenIds(unittest.TestCase):
    def test_a_tokenizer_folder_without_special_ids_ends_on_deepseeks_eos(self):
        # the reviewer's case: ST.Tokenizer(tokens, merges, types) has special_ids {}, so an "eos" lookup alone left
        # no stop id at all; the real DeepSeek-1017 folder gives {1}, the launcher's id
        tok = FolderTokenizer()
        self.assertEqual(tok.special_ids, {})
        self.assertEqual(end_token_ids(tok), {1})
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.stop_ids, {1})

    def test_the_gguf_key_names_the_end_token(self):
        # strata_tokenizer's from_gguf fills special_ids with GGUF key names, not "eos"
        self.assertEqual(end_token_ids(FolderTokenizer({"tokenizer.ggml.eos_token_id": 7})), {7})

    def test_one_token_markers_keep_the_stop_ids_they_had(self):
        tok = ByteTokenizer()
        old = set(tok.encode(IM_END, parse_special=True) + tok.encode("<|endoftext|>", parse_special=True))
        self.assertEqual(len(old), 2)
        self.assertEqual(end_token_ids(tok), old)
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.stop_ids, old)

    def test_pieces_of_a_marker_are_not_stop_ids(self):
        tok = SplitTokenizer()
        pieces = set(tok.encode(IM_END, parse_special=True) + tok.encode("<|endoftext|>", parse_special=True))
        ids = end_token_ids(tok)
        self.assertEqual(ids, {EOS})
        self.assertFalse(ids & pieces)
        for piece in "<|>imendoftext_":
            self.assertNotIn(ord(piece), ids, piece)
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.stop_ids, {EOS})

    def test_no_whole_marker_and_no_eos_means_no_stop_id(self):
        self.assertEqual(end_token_ids(SplitTokenizer(eos=None)), set())


class Routes(unittest.TestCase):
    """The text of a normal reply is the same through the server (the stop ids of a one-token-marker tokenizer)."""

    def test_a_reply_ends_on_its_end_token_and_comes_out_whole(self):
        tok = ByteTokenizer()
        script = "Nerazim | a > b, the end of the text"
        svc = Service(MockEngine(tok, script), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "go"}], "reasoning_effort": "none"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(d["choices"][0]["message"]["content"], script)
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
