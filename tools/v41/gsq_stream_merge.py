#!/usr/bin/env python3
"""gsq_stream_merge.py - gsq_merge.py without storing the shards: each GSQ-RCO shard is STREAMED from Hugging Face, every
byte goes through SHA-256 (checked against the repo's LFS hash at the end), and only the routed-expert tensors are
written - straight into the merged file, at their offsets.  Our own tensors (metadata, dense, Engram projections) are
copied from our file first.

Why: it needs half the disk of downloading the shards first (~130-160 GB instead of twice that), and a damaged
download cannot slip through (the merged file is only renamed into place when every shard's hash matches).

    gsq_stream_merge.py --ours models/DeepSeek-V4.1-Flash-1017-Q2.gguf --build IQ2_XS --out models/ds41-gsq-iq2_xs.gguf
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsq_merge as M  # noqa: E402

REPO = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"


def url(name):
    return f"https://huggingface.co/{REPO}/resolve/main/{name}"


def get(name, start, end=None):
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    return urllib.request.urlopen(urllib.request.Request(url(name), headers={"Range": rng}), timeout=120)


class Head(M.Gguf):
    """a shard's header from its first bytes over HTTP (M.Gguf parses a file; same parser, bytes from a range GET)"""

    def __init__(self, name):
        data = get(name, 0, (64 << 20) - 1).read()
        tmp = os.path.join(os.environ.get("TEMP", "."), "gsq_head_" + name)
        open(tmp, "wb").write(data)
        super().__init__(tmp)
        self.f.close()
        os.remove(tmp)
        self.name = name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--build", required=True, help="IQ2_XXS / IQ2_XS / IQ1_M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60))
    shards = sorted(s["rfilename"] for s in meta["siblings"] if f"-{a.build}-0000" in s["rfilename"])
    want = {s["rfilename"]: (s["lfs"]["sha256"], s["size"]) for s in meta["siblings"] if s["rfilename"] in shards}
    heads = [Head(n) for n in shards]
    ours = M.Gguf(a.ours)
    src = {}   # expert tensor name -> (shard index, absolute byte range in the shard)
    for i, h in enumerate(heads):
        for (n, d, t, off) in h.tensors:
            if M.is_expert(n):
                src[n] = (i, d, t, h.data0 + off)
    al = ours.alignment
    infos, cur, jobs = [], 0, [[] for _ in heads]   # jobs[shard] = [(shard start, nbytes, output data offset)]
    for (n, d, t, off) in ours.tensors:
        if ".engram_embd." in n:   # the Engram tables stay in the main GGUF (the engine's --engram reads them)
            continue
        if M.is_expert(n):
            i, gd, gt, s0 = src[n]
            assert gd == d, n
            t = gt
            jobs[i].append((s0, ours.nbytes(d, t), cur))
        infos.append((n, d, t, cur, off if not M.is_expert(n) else None))
        cur = (cur + ours.nbytes(d, t) + al - 1) // al * al
    total = cur
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ours.version, len(infos), ours.nkv) + ours.kv_raw)
    for (n, d, t, off, _) in infos:
        nb = n.encode()
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d)
        hdr += struct.pack("<IQ", t, off)
    hdr += b"\0" * ((len(hdr) + al - 1) // al * al - len(hdr))
    D = len(hdr)
    print(f"{a.build}: {len(shards)} shards, {sum(len(j) for j in jobs)} expert tensors; output {(D + total) / 1e9:.1f} GB",
          flush=True)
    tmp = a.out + ".part"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    with open(tmp, "wb") as w:   # header + our tensors; the expert regions are filled by the streams below
        w.write(hdr)
        w.truncate(D + total)
        for (n, d, t, off, sof) in infos:
            if sof is None:
                continue
            ours.f.seek(ours.data0 + sof)
            w.seek(D + off)
            left = ours.nbytes(d, t)
            while left:
                b = ours.f.read(min(left, 64 << 20))
                w.write(b)
                left -= len(b)
    print(f"  our tensors written ({time.time() - t0:.0f} s)", flush=True)
    done = [0] * len(heads)
    res = {}

    def stream(i):
        name = shards[i]
        h = hashlib.sha256()
        pos, size = 0, want[name][1]
        js = sorted(jobs[i])
        with open(tmp, "r+b") as w:
            tries = 0
            while pos < size:
                try:
                    r = get(name, pos)
                    while pos < size:
                        b = r.read(16 << 20)
                        if not b:
                            raise IOError("stream ended early")
                        h.update(b)
                        e = pos + len(b)
                        for (s0, nb, oo) in js:   # the expert bytes inside [pos, e)
                            lo, hi = max(s0, pos), min(s0 + nb, e)
                            if lo < hi:
                                w.seek(D + oo + (lo - s0))
                                w.write(b[lo - pos:hi - pos])
                        pos = e
                        done[i] = pos
                except Exception as ex:   # resume from pos (the hash state carries on)
                    tries += 1
                    print(f"  {name}: {ex!r} at {pos}, retry {tries}", flush=True)
                    if tries > 8:
                        res[name] = "FAILED"
                        return
                    time.sleep(10)
        res[name] = h.hexdigest()

    ths = [threading.Thread(target=stream, args=(i,)) for i in range(len(heads))]
    for th in ths:
        th.start()
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        el = time.time() - t0
        print(f"  {sum(done) / 1e9:.1f} / {sum(v[1] for v in want.values()) / 1e9:.1f} GB streamed, "
              f"{sum(done) / 1e6 / el:.0f} MB/s", flush=True)
    ok = all(res.get(n) == want[n][0] for n in shards)
    for n in shards:
        print(f"  {'OK ' if res.get(n) == want[n][0] else 'BAD'} {n} {str(res.get(n))[:12]} (want {want[n][0][:12]})")
    if not ok:
        sys.exit(f"{a.build}: hash mismatch - {tmp} kept, not renamed")
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e9:.1f} GB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
