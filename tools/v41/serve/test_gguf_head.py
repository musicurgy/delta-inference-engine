"""gguf_head: the type table against the engine's block_geometry, and type 43's two layouts (Unsloth's MXFP8 Engram
tables vs mainline ggml's F8_E4M3) told apart from a tensor's room in the file (R-18).

    python tools/v41/serve/test_gguf_head.py      (no GPU, no model files)
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_head as H  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, align=32):
    """a GGUF v3 file: general.alignment + `tensors` = [(name, dims, type, payload bytes)], payloads aligned"""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    head += _str("general.alignment") + struct.pack("<II", 4, align)
    data, offs = b"", []
    for _, _, _, payload in tensors:
        data += b"\0" * (-len(data) % align)
        offs.append(len(data))
        data += payload
    for (name, dims, typ, _), off in zip(tensors, offs):
        head += _str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", typ, off)
    head += b"\0" * (-len(head) % align)
    with open(path, "wb") as f:
        f.write(head + data)


class TypeTable(unittest.TestCase):
    def test_known_types_match_block_geometry(self):
        with open(os.path.join(REPO, "include", "strata", "artifact", "gguf_reader.hpp"), encoding="utf-8") as f:
            src = f.read()
        body = src[src.index("inline bool block_geometry("):]
        body = body[:body.index("default:")]
        ids = {int(x) for x in re.findall(r"case (\d+):", body)}
        self.assertEqual(ids, set(H.KNOWN_TYPES))

    def test_ik_and_newer_ids_unknown(self):
        for t in (15, 28, 34, 35, 41, 44, 144, 145, 149, 168):
            self.assertNotIn(t, H.KNOWN_TYPES)
        for t in (16, 20, 29, 42, 43, 143):
            self.assertIn(t, H.KNOWN_TYPES)


class Layout43(unittest.TestCase):
    def test_pure_rule(self):
        self.assertEqual(H.layout43([256, 100], 256 * 100 // 32 * 33), "mxfp8")
        self.assertEqual(H.layout43([256, 100], 256 * 100), "f8_e4m3")
        self.assertEqual(H.layout43([256, 100], 256 * 100 + 31), "f8_e4m3")   # (alignment padding)
        self.assertEqual(H.layout43([256, 100], 256 * 100 - 1), "unknown")
        self.assertEqual(H.layout43([100, 4], 400), "f8_e4m3")   # (rows not whole MXFP8 blocks)

    def _file(self, tensors):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.gguf")
        write_gguf(p, tensors)
        self.addCleanup(lambda: (os.remove(p), os.rmdir(d)))
        return p

    def test_mxfp8_table_last_and_followed(self):
        n = 256 * 64
        mx = b"\1" * (n // 32 * 33)
        p = self._file([("output_norm.weight", [256], 0, b"\0" * 1024), ("per_layer_token_embd.weight", [256, 64], 43, mx)])
        self.assertEqual(H.type43_layout(p, "per_layer_token_embd.weight"), "mxfp8")
        p = self._file([("per_layer_token_embd.weight", [256, 64], 43, mx), ("output_norm.weight", [256], 0, b"\0" * 1024)])
        self.assertEqual(H.type43_layout(p, "per_layer_token_embd.weight"), "mxfp8")

    def test_plain_f8_table(self):
        n = 256 * 64
        p = self._file([("per_layer_token_embd.weight", [256, 64], 43, b"\1" * n), ("output_norm.weight", [256], 0, b"\0" * 1024)])
        self.assertEqual(H.type43_layout(p, "per_layer_token_embd.weight"), "f8_e4m3")
        p = self._file([("output_norm.weight", [256], 0, b"\0" * 1024), ("per_layer_token_embd.weight", [256, 64], 43, b"\1" * n)])
        self.assertEqual(H.type43_layout(p, "per_layer_token_embd.weight"), "f8_e4m3")

    def test_absent_and_header_unchanged(self):
        p = self._file([("output_norm.weight", [256], 0, b"\0" * 1024)])
        self.assertIsNone(H.type43_layout(p, "per_layer_token_embd.weight"))
        meta, tensors = H.read_header(p)
        self.assertEqual(meta["general.alignment"], 32)
        self.assertEqual(tensors, [("output_norm.weight", [256], 0, 0)])
        self.assertEqual(H.tensor_types(p), {"output_norm.weight": 0})


if __name__ == "__main__":
    unittest.main()
