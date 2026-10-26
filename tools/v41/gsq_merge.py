#!/usr/bin/env python3
"""gsq_merge.py - our model file with the routed experts swapped for ISTA-DASLab's GSQ-RCO ones.

ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF uses llama.cpp's DeepSeek-V4.1 metadata keys (ours reads the
`deepseek41.config` JSON and our Engram keys) and its own dense types (Q5_K / BF16, ours Q8_0 / F16 / F32 from the FP8
release).  The experts are ~95% of the bytes and the part GSQ + RCO optimise, so this keeps EVERYTHING of ours (metadata,
tokenizer, dense tensors, Engram projections) and takes only the 120 `blk.N.ffn_{gate,up,down}_exps.weight` tensors from
their shards (IQ1_M / IQ2_XXS / IQ2_XS per layer; gate and up always share a type).  Our loader takes any uniform
expert type per role, so no engine change is needed to load it.

    gsq_merge.py --ours models/ds4-q2-slim.gguf --gsq <shards>/...-IQ2_XXS-00001-of-00004.gguf [...more shards]
                 --out models/ds41-gsq-iq2_xxs.gguf [--dry-run]
    (gsq_stream_merge.py does the same without downloading the shards first)
"""
import argparse
import os
import struct
import sys
import time

# GGML type -> (block elements, block bytes)
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56),
         30: (1, 2)}
VSIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Gguf:
    """header of one GGUF file: the raw KV bytes, the tensor infos, the data start"""

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        head = f.read(64 << 20)   # every header here fits (ours ~20 MB with the Engram arrays)
        self.f = f
        o = 0

        def rd(fmt):
            nonlocal o
            v = struct.unpack_from(fmt, head, o)
            o += struct.calcsize(fmt)
            return v[0]

        def skip_str():
            nonlocal o
            n = rd("<Q")
            s = head[o:o + n]
            o += n
            return s

        def skip_val(t):
            nonlocal o
            if t == 8:
                return skip_str()
            if t == 9:
                at, n = rd("<I"), rd("<Q")
                if at == 8:
                    for _ in range(n):
                        skip_str()
                else:
                    o += VSIZE[at] * n
                return None
            v = struct.unpack_from({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
                                    11: "<q", 12: "<d"}[t], head, o)[0]
            o += VSIZE[t]
            return v

        assert head[:4] == b"GGUF", path
        o = 4
        self.version, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
        kv0 = o
        self.alignment = 32
        for _ in range(nkv):
            k = skip_str().decode()
            t = rd("<I")
            v = skip_val(t)
            if k == "general.alignment":
                self.alignment = int(v)
        self.nkv, self.kv_raw = nkv, head[kv0:o]
        self.tensors = []   # (name, dims, type, offset)
        for _ in range(nt):
            name = skip_str().decode()
            nd = rd("<I")
            dims = [rd("<Q") for _ in range(nd)]
            t, off = rd("<I"), rd("<Q")
            self.tensors.append((name, dims, t, off))
        a = self.alignment
        self.data0 = (o + a - 1) // a * a

    def nbytes(self, dims, t):
        n = 1
        for d in dims:
            n *= d
        be, bb = TYPES[t]
        assert n % be == 0
        return n // be * bb


def is_expert(name):
    return name.startswith("blk.") and name.endswith("_exps.weight") and "_shexp" not in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gsq", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ours = Gguf(a.ours)
    gsq = {}
    for p in a.gsq:
        g = Gguf(p)
        for (n, d, t, off) in g.tensors:
            if is_expert(n):
                gsq[n] = (g, d, t, off)
    plan = []   # (name, dims, type, source Gguf, source offset)
    swapped = 0
    for (n, d, t, off) in ours.tensors:
        if is_expert(n):
            if n not in gsq:
                sys.exit(f"GSQ shards lack {n}")
            g, gd, gt, goff = gsq[n]
            if gd != d:
                sys.exit(f"{n}: GSQ shape {gd}, ours {d}")
            plan.append((n, gd, gt, g, goff))
            swapped += 1
        else:
            plan.append((n, d, t, ours, off))
    al = ours.alignment
    out_infos, cur = [], 0
    for (n, d, t, src, soff) in plan:
        out_infos.append((n, d, t, cur))
        cur = (cur + ours.nbytes(d, t) + al - 1) // al * al
    total = cur
    old = sum(ours.nbytes(d, t) for (n, d, t, _) in ours.tensors)
    print(f"{swapped} expert tensors swapped; tensor data {old / 1e9:.1f} GB -> {total / 1e9:.1f} GB "
          f"(alignment {al}); types: " + ", ".join(sorted({f'{n.split(".")[1]}:{t}' for (n, d, t, _, _) in plan if is_expert(n) and 'gate' in n},
                                                         key=lambda s: int(s.split(':')[0]))))
    if a.dry_run:
        return
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ours.version, len(out_infos), ours.nkv) + ours.kv_raw)
    for (n, d, t, off) in out_infos:
        nb = n.encode()
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d)
        hdr += struct.pack("<IQ", t, off)
    pad = (len(hdr) + al - 1) // al * al - len(hdr)
    hdr += b"\0" * pad
    tmp = a.out + ".part"
    t0, done = time.time(), 0
    with open(tmp, "wb") as w:
        w.write(hdr)
        data0 = len(hdr)
        for (n, d, t, src, soff), (_, _, _, ooff) in zip(plan, out_infos):
            nb = ours.nbytes(d, t)
            assert w.tell() == data0 + ooff, n
            src.f.seek(src.data0 + soff)
            left = nb
            while left:
                buf = src.f.read(min(left, 64 << 20))
                if not buf:
                    sys.exit(f"{n}: short read in {src.path}")
                w.write(buf)
                left -= len(buf)
            done += nb
            w.write(b"\0" * ((nb + al - 1) // al * al - nb))
            if is_expert(n) and "down" in n:
                el = time.time() - t0
                print(f"  {n}: {done / 1e9:.1f} / {total / 1e9:.1f} GB, {done / 1e6 / max(el, 1e-3):.0f} MB/s", flush=True)
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e9:.1f} GB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
