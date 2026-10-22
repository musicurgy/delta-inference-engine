#!/usr/bin/env python3
"""gguf_drop_tensors.py - copy a GGUF without the tensors whose name contains a pattern (default ".engram_embd.").

The ds4 Q2 GGUF carries DeepSeek-V4.1's Engram tables inline (189 GiB of its 366 GB); the engine reads them from the
--engram directory instead and skips the inline copy, so the slim file serves identically from a fast drive.  The
metadata (key/value section) is copied byte for byte; tensor infos are rewritten with new offsets on the file's own
alignment; tensor bytes are streamed unchanged in file order.

    gguf_drop_tensors.py SRC.gguf DST.gguf [--drop .engram_embd.]
"""
import argparse
import os
import shutil
import struct
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", default=".engram_embd.")
    a = ap.parse_args()
    f = open(a.src, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rstr():
        (n,) = rd("Q")
        return f.read(n).decode("utf-8")

    magic = f.read(4)
    if magic != b"GGUF":
        sys.exit("not a GGUF")
    (ver,) = rd("I")
    nt, nkv = rd("QQ")
    kv_start = f.tell()
    align = 32

    def skip_val(t):
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t in (10, 11, 12):
            f.read(8)
        elif t == 8:
            rstr()
        elif t == 9:
            (et,) = rd("I")
            (n,) = rd("Q")
            for _ in range(n):
                skip_val(et)
        else:
            sys.exit(f"unknown kv type {t}")

    for _ in range(nkv):
        k = rstr()
        (t,) = rd("I")
        if k == "general.alignment" and t == 4:
            (align,) = rd("I")
        else:
            skip_val(t)
    kv_end = f.tell()
    infos = []
    for _ in range(nt):
        name = rstr()
        (nd,) = rd("I")
        dims = rd("Q" * nd)
        (ty,) = rd("I")
        (off,) = rd("Q")
        infos.append([name, dims, ty, off])
    hdr_end = f.tell()
    data0 = (hdr_end + align - 1) // align * align
    size = os.path.getsize(a.src)
    order = sorted(range(nt), key=lambda i: infos[i][3])
    span = {}
    for j, i in enumerate(order):
        nxt = infos[order[j + 1]][3] if j + 1 < nt else size - data0
        span[i] = nxt - infos[i][3]
    keep = [i for i in order if a.drop not in infos[i][0]]
    dropped = nt - len(keep)
    new_off, cur = {}, 0
    for i in keep:
        new_off[i] = cur
        cur += (span[i] + align - 1) // align * align
    total = cur
    # the new header: magic, version, counts, the kv section verbatim, the kept tensor infos (original order)
    f.seek(kv_start)
    kv_bytes = f.read(kv_end - kv_start)
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ver, len(keep), nkv) + kv_bytes)
    keep_set = set(keep)
    for i in range(nt):
        if i not in keep_set:
            continue
        name, dims, ty, _ = infos[i]
        nb = name.encode("utf-8")
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        hdr += struct.pack("<IQ", ty, new_off[i])
    pad = (-len(hdr)) % align
    hdr += b"\0" * pad
    need = len(hdr) + total
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.dst))).free
    print(f"{nt} tensors, dropping {dropped} ({a.drop}); output {need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free", flush=True)
    if need > free - 2 * 2**30:
        sys.exit("not enough free space on the destination")
    tmp = a.dst + ".part"
    t0 = time.time()
    with open(tmp, "wb") as o:
        o.write(hdr)
        done = 0
        next_print = 8 << 30
        buf = 64 << 20
        for i in keep:
            f.seek(data0 + infos[i][3])
            left = span[i]
            while left > 0:
                b = f.read(min(buf, left))
                if not b:
                    sys.exit("short read")
                o.write(b)
                left -= len(b)
                done += len(b)
            o.write(b"\0" * ((-span[i]) % align))
            if done >= next_print:
                next_print += 8 << 30
                print(f"  {done / 2**30:.0f} GiB, {done / 2**20 / (time.time() - t0):.0f} MiB/s", flush=True)
    os.replace(tmp, a.dst)
    print(f"wrote {a.dst}: {os.path.getsize(a.dst) / 2**30:.1f} GiB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
