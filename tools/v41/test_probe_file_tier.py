"""tools/v41/test_probe_file_tier.py - probe_file_tier.py: the GGUF header walk, the expert table, Ds41FileBlobs'
part / span arithmetic, and (Windows) that the unbuffered bounce path returns the file's own bytes in every mode, on
a small synthetic split GGUF in a temp folder.  No model file, no GPU.

    .venv/Scripts/python.exe -m unittest tools/v41/test_probe_file_tier.py
"""
import ctypes
import os
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_file_tier as P  # noqa: E402


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, extra_kv=True, alignment=32, seed=0):
    """tensors: [(name, shape, type)] with F16 / Q8_0 types; returns {name: (abs offset, bytes)}."""
    rng = random.Random(seed)
    kv = [_s("general.alignment") + struct.pack("<II", 4, alignment)]
    if extra_kv:
        kv.append(_s("general.name") + struct.pack("<I", 8) + _s("probe test"))
        kv.append(_s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3) + _s("a") + _s("bb") + _s("ccc"))
        kv.append(_s("x.f32s") + struct.pack("<IIQ", 9, 6, 2) + struct.pack("<ff", 1.0, 2.0))
    infos, sizes, off = [], [], 0
    for name, shape, t in tensors:
        n = P.row_bytes(t, shape[0])
        for d in shape[1:]:
            n *= d
        infos.append(_s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape) +
                     struct.pack("<IQ", t, off))
        sizes.append((off, n))
        off = (off + n + alignment - 1) // alignment * alignment
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + b"".join(infos)
    ds = (len(head) + alignment - 1) // alignment * alignment
    body = bytearray(off)
    for (o, n) in sizes:
        body[o:o + n] = bytes(rng.getrandbits(8) for _ in range(n))
    with open(path, "wb") as f:
        f.write(head + b"\0" * (ds - len(head)) + bytes(body))
    return {t[0]: (ds + o, n) for t, (o, n) in zip(tensors, sizes)}


def model(d):
    """Two shards: layer 0's experts + a dense tensor in shard 1, layer 1's in shard 2 (F16 and Q8_0)."""
    a = os.path.join(d, "m-00001-of-00002.gguf")
    b = os.path.join(d, "m-00002-of-00002.gguf")
    la = write_gguf(a, [("token_embd.weight", [64, 10], 1),
                        ("blk.0.ffn_gate_exps.weight", [256, 24, 6], 1),
                        ("blk.0.ffn_up_exps.weight", [256, 24, 6], 1),
                        ("blk.0.ffn_down_exps.weight", [24 * 32 // 24 * 24, 256, 6], 8)], seed=1)
    lb = write_gguf(b, [("blk.1.ffn_gate_exps.weight", [256, 24, 6], 8),
                        ("blk.1.ffn_up_exps.weight", [256, 24, 6], 8),
                        ("blk.1.ffn_down_exps.weight", [32, 256, 6], 8)], extra_kv=False, alignment=64, seed=2)
    return a, b, la, lb


class Pure(unittest.TestCase):
    def test_split_paths(self):
        self.assertEqual(P.split_paths("x/m-00002-of-00003.gguf"),
                         ["x/m-00001-of-00003.gguf", "x/m-00002-of-00003.gguf", "x/m-00003-of-00003.gguf"])
        self.assertEqual(P.split_paths("x/one.gguf"), ["x/one.gguf"])

    def test_part_range_tiles_the_blob(self):
        for total in (4096, 12345, 3 * 1048576 + 17):
            for parts in (1, 3, 8):
                rs = [P.part_range(total, p, parts) for p in range(parts)]
                self.assertEqual(rs[0][0], 0)
                self.assertEqual(rs[-1][1], total)
                for (lo, hi), (lo2, _) in zip(rs, rs[1:]):
                    self.assertEqual(hi, lo2)
                    self.assertEqual(hi % 4096, 0)

    def test_part_reads_cover_the_spans(self):
        spans = [(0, 10000, 5000), (0, 90000, 5000), (1, 7, 6000)]
        got = []
        for p in range(8):
            got += P.part_reads(spans, p, 8)
        self.assertEqual(sum(r[2] for r in got), 16000)
        at = 0
        for r in sorted(got, key=lambda r: r[3]):
            self.assertEqual(r[3], at)
            at += r[2]

    def test_header_and_table(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, la, lb = model(d)
            hs = []
            for p in (a, b):
                data = open(p, "rb").read()
                hs.append(P.Header(lambda off, n, data=data: data[off:off + n], len(data)))
            self.assertEqual(hs[0].alignment, 32)
            self.assertEqual(hs[1].alignment, 64)
            tab = P.expert_table(hs)
            self.assertEqual(sorted(tab), [0, 1])
            si, off, sl, ne = tab[0]["gate"][0]
            self.assertEqual((si, ne, sl), (0, 6, 256 * 2 * 24))
            self.assertEqual(off, la["blk.0.ffn_gate_exps.weight"][0])
            si, off, sl, ne = tab[1]["down"][0]
            self.assertEqual((si, sl), (1, 34 * 256))
            self.assertEqual(off, lb["blk.1.ffn_down_exps.weight"][0])
            blobs = P.pick_blobs(tab, 20, 3)
            self.assertEqual(len(blobs), 20)
            for spans in blobs:
                self.assertEqual(len(spans), 3)


@unittest.skipUnless(sys.platform == "win32", "the unbuffered path is Win32")
class Win(unittest.TestCase):
    def test_unbuffered_header_matches(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, _, _ = model(d)
            for p in (a, b):
                data = open(p, "rb").read()
                ref = P.Header(lambda off, n: data[off:off + n], len(data))
                h = P.unbuffered_header(p)
                self.assertEqual(h.tensors, ref.tensors)
                self.assertEqual(h.data_start, ref.data_start)

    def test_read_direct_returns_the_file_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a, _, _, _ = model(d)
            data = open(a, "rb").read()
            h = P._open(a, True)
            r = P.Reader(64 * 1024)
            dst = ctypes.create_string_buffer(32 * 1024)
            try:
                rng = random.Random(5)
                for _ in range(200):
                    n = rng.randrange(1, 30000)
                    off = rng.randrange(0, len(data) - n)
                    r.read_direct(h, off, n, ctypes.addressof(dst))
                    self.assertEqual(dst.raw[:n], data[off:off + n])
            finally:
                r.close()
                P.K.CloseHandle(h)

    def test_every_mode_reads_the_blobs(self):
        with tempfile.TemporaryDirectory() as d:
            a, _, _, _ = model(d)
            for mode in ("unmapped", "mapped", "buffered", "remap", "remap-touch"):
                res = P.main([a, "--blobs", "16", "--threads", "4", "--parts", "3", "--modes", mode])
                self.assertEqual(len(res), 1)
                self.assertGreater(res[0]["reads"], 0)
                self.assertGreater(res[0]["gb"], 0)


if __name__ == "__main__":
    unittest.main()
