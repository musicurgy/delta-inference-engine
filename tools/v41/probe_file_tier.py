"""probe_file_tier.py - the file tier's read path on this PC, mapped vs unmapped (STRATA_FILE_TIER_UNMAP).

Reads N random routed-expert blobs of a DeepSeek-V4.1 GGUF the way the engine's file tier does - Ds41FileBlobs::read_part
(tools/v41/engine/ds41_moe.cpp): each blob [gate | up | down] cut into P parts (4 KiB-aligned cuts), T threads pull
parts, each part is read per tensor span as ONE 4 KiB-aligned unbuffered read (CreateFileW FILE_FLAG_NO_BUFFERING |
FILE_FLAG_OVERLAPPED, waited with a per-thread event, as ds41_sys::xfer) into the thread's aligned bounce, then
memcpy'd out of it into the blob's slot (read_direct).  So the GB/s it prints is what the engine's readers can get,
bounce copy included (--no-copy drops the copy: the drive + filesystem alone).

Every mode reads the same blobs and only changes what else holds the shard files:
  unmapped     nothing (what STRATA_FILE_TIER_UNMAP=1 aims for)
  mapped       a whole-file view of every shard held during the run (its first page touched) - the engine today:
               Ds41Dense's and Ds41DSpark's GgufModel keep the shards mapped for the whole run
  buffered     a buffered handle on every shard held during the run, with one 4 KiB buffered read done
               (Ds41FileBlobs keeps one per shard beside the O_DIRECT one)
  remap        map + touch 64 MiB of every shard, unmap and close BEFORE the run (a mapping released at load: does
               the section linger?)
  remap-touch  as remap, then open + close one unbuffered handle per shard (ds41_unbuffered_touch: the purge the
               engine does after releasing) before the run
The engine must NOT be running (its mappings would make every mode 'mapped'), nor any GPU benchmark reading the drive.
Read-only: the files are opened GENERIC_READ and never written.  Windows only (the effect is NTFS's).

    .venv/Scripts/python.exe -I tools/v41/probe_file_tier.py <DS41_MODELS>/<model>-00001-of-00005.gguf
        [--blobs 400] [--threads 32] [--parts 8] [--modes unmapped,mapped,buffered,remap,remap-touch,unmapped]
        [--repeat 1] [--seed 1] [--no-copy] [--json out.json]
"""
import argparse
import ctypes
import json
import os
import random
import re
import struct
import sys
import threading
import time

ALIGN = 4096

# ggml type -> (elements per block, bytes per block): include/strata/artifact/gguf_reader.hpp block_geometry
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66),
         17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         29: (256, 56), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 40: (64, 36), 39: (32, 17), 42: (64, 18),
         43: (32, 33), 143: (128, 28)}
META_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\b")
ROLES = ("gate", "up", "down")


def split_paths(path):
    """Every shard of the model `path` belongs to (strata::gguf_split_paths' naming: -0000k-of-0000N.gguf)."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    return [path[:m.start()] + "-%05d-of-%05d.gguf" % (i, n) for i in range(1, n + 1)]


def row_bytes(t, n0):
    be, bb = BLOCK[t]
    if n0 % be:
        raise ValueError("row of %d elements is not whole blocks of type %d" % (n0, t))
    return n0 // be * bb


class Header:
    """A GGUF v3 header parsed out of `read(off, n) -> bytes` (the caller decides how the file is read)."""

    def __init__(self, read, size):
        self.read, self.size = read, size
        self.buf, self.pos = b"", 0
        self.alignment = 32
        self.tensors = []  # (name, shape, type, offset)
        self._parse()

    def _need(self, n):
        while self.pos + n > len(self.buf):
            want = max(len(self.buf) * 2, 1 << 20, self.pos + n)
            want = min(want, self.size)
            if want <= len(self.buf):
                raise ValueError("GGUF header runs past the end of the file")
            self.buf = self.read(0, want)

    def _u(self, fmt):
        n = struct.calcsize(fmt)
        self._need(n)
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += n
        return v

    def _str(self):
        n = self._u("<Q")
        self._need(n)
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", "replace")

    def _skip_value(self, t, key=None):
        if t == 8:
            s = self._str()
            return s
        if t == 9:
            et, cnt = self._u("<I"), self._u("<Q")
            for _ in range(cnt):
                self._skip_value(et)
            return None
        if t not in META_SIZE:
            raise ValueError("unknown GGUF metadata type %d" % t)
        fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[META_SIZE[t]]
        return self._u(fmt)

    def _parse(self):
        if self._u("<I") != 0x46554747:
            raise ValueError("not a GGUF file")
        if self._u("<I") != 3:
            raise ValueError("not GGUF v3")
        n_t, n_kv = self._u("<Q"), self._u("<Q")
        for _ in range(n_kv):
            key = self._str()
            t = self._u("<I")
            v = self._skip_value(t)
            if key == "general.alignment" and isinstance(v, int) and v:
                self.alignment = v
        for _ in range(n_t):
            name = self._str()
            nd = self._u("<I")
            shape = [self._u("<Q") for _ in range(nd)]
            t = self._u("<I")
            off = self._u("<Q")
            self.tensors.append((name, shape, t, off))
        a = self.alignment
        self.data_start = min((self.pos + a - 1) // a * a, self.size)


def expert_table(headers):
    """{layer: {role: [(shard, abs_offset, slice_bytes, n_expert)]}} from every shard's 3-D *_exps tensors."""
    tab = {}
    for si, h in enumerate(headers):
        for name, shape, t, off in h.tensors:
            m = EXPS.match(name)
            if not m or len(shape) != 3 or t not in BLOCK:
                continue
            sl = row_bytes(t, shape[0]) * shape[1]
            tab.setdefault(int(m.group(1)), {}).setdefault(m.group(2), []).append(
                (si, h.data_start + off, sl, shape[2]))
    return {l: r for l, r in tab.items() if all(x in r for x in ROLES)}


def pick_blobs(tab, n, seed):
    """n random (layer, expert) blobs as [(shard, offset, bytes)] x 3 spans (gate, up, down)."""
    rng = random.Random(seed)
    layers = sorted(tab)
    out = []
    while len(out) < n:
        l = rng.choice(layers)
        ne = min(x[3] for r in ROLES for x in tab[l][r])
        e = rng.randrange(ne)
        spans = []
        for r in ROLES:
            si, base, sl, _ = tab[l][r][0]
            spans.append((si, base + e * sl, sl))
        out.append(spans)
    return out


def part_range(total, part, nparts):
    """Ds41FileBlobs::part_range: [lo, hi) of the blob that part `part` of `nparts` covers."""
    lo = (total * part // nparts) & ~(ALIGN - 1)
    hi = total if part + 1 == nparts else (total * (part + 1) // nparts) & ~(ALIGN - 1)
    return lo, hi


def part_reads(spans, part, nparts):
    """The (shard, file offset, bytes, offset in the blob) reads read_part issues for one part."""
    total = sum(s[2] for s in spans)
    lo, hi = part_range(total, part, nparts)
    out, at = [], 0
    for si, off, n in spans:
        a, b = max(lo, at), min(hi, at + n)
        if a < b:
            out.append((si, off + (a - at), b - a, a))
        at += n
    return out


def aligned(off, n):
    """read_direct's aligned span [a0, a1) around [off, off + n)."""
    a0 = off & ~(ALIGN - 1)
    a1 = (off + n + ALIGN - 1) & ~(ALIGN - 1)
    return a0, a1


# ---------------------------------------------------------------- Win32
if sys.platform == "win32":
    from ctypes import wintypes as W

    K = ctypes.WinDLL("kernel32", use_last_error=True)
    GENERIC_READ = 0x80000000
    SHARE_ALL = 0x1 | 0x2 | 0x4
    OPEN_EXISTING = 3
    ATTR_NORMAL = 0x80
    FLAG_OVERLAPPED = 0x40000000
    FLAG_NO_BUFFERING = 0x20000000
    FLAG_RANDOM = 0x10000000
    PAGE_READONLY = 0x02
    FILE_MAP_READ = 0x0004
    MEM_COMMIT_RESERVE = 0x1000 | 0x2000
    MEM_RELEASE = 0x8000
    PAGE_READWRITE = 0x04
    ERROR_IO_PENDING = 997
    INVALID = ctypes.c_void_p(-1).value

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p), ("Offset", W.DWORD),
                    ("OffsetHigh", W.DWORD), ("hEvent", W.HANDLE)]

    K.CreateFileW.restype = W.HANDLE
    K.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
    K.ReadFile.argtypes = [W.HANDLE, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD), ctypes.POINTER(OVERLAPPED)]
    K.GetOverlappedResult.argtypes = [W.HANDLE, ctypes.POINTER(OVERLAPPED), ctypes.POINTER(W.DWORD), W.BOOL]
    K.CreateEventW.restype = W.HANDLE
    K.CreateEventW.argtypes = [ctypes.c_void_p, W.BOOL, W.BOOL, W.LPCWSTR]
    K.ResetEvent.argtypes = [W.HANDLE]
    K.CloseHandle.argtypes = [W.HANDLE]
    K.CreateFileMappingW.restype = W.HANDLE
    K.CreateFileMappingW.argtypes = [W.HANDLE, ctypes.c_void_p, W.DWORD, W.DWORD, W.DWORD, W.LPCWSTR]
    K.MapViewOfFile.restype = ctypes.c_void_p
    K.MapViewOfFile.argtypes = [W.HANDLE, W.DWORD, W.DWORD, W.DWORD, ctypes.c_size_t]
    K.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    K.VirtualAlloc.restype = ctypes.c_void_p
    K.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, W.DWORD, W.DWORD]
    K.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, W.DWORD]
    K.GetFileSizeEx.argtypes = [W.HANDLE, ctypes.POINTER(ctypes.c_longlong)]


def _open(path, unbuffered):
    if sys.platform != "win32":
        raise SystemExit("probe_file_tier.py: Windows only (the effect measured is NTFS's)")
    flags = ATTR_NORMAL | FLAG_OVERLAPPED | (FLAG_NO_BUFFERING if unbuffered else FLAG_RANDOM)
    h = K.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, flags, None)
    if h is None or h == INVALID:
        raise OSError(ctypes.get_last_error(), "cannot open " + path)
    return h


def _size(h):
    v = ctypes.c_longlong(0)
    K.GetFileSizeEx(h, ctypes.byref(v))
    return v.value


class Reader:
    """One thread's event + aligned bounce (VirtualAlloc: page aligned), ds41_sys::xfer's overlapped read."""

    def __init__(self, bounce_bytes):
        self.ev = K.CreateEventW(None, True, False, None)
        self.n = (bounce_bytes + ALIGN - 1) & ~(ALIGN - 1)
        self.p = K.VirtualAlloc(None, self.n, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not self.p:
            raise MemoryError("VirtualAlloc of the bounce failed")

    def close(self):
        K.VirtualFree(self.p, 0, MEM_RELEASE)
        K.CloseHandle(self.ev)

    def pread(self, h, dst, n, off):
        """n bytes at off into dst (an address); returns bytes read."""
        done = 0
        while done < n:
            want = min(n - done, 1 << 30)
            ov = OVERLAPPED()
            at = off + done
            ov.Offset, ov.OffsetHigh, ov.hEvent = at & 0xFFFFFFFF, at >> 32, self.ev
            K.ResetEvent(self.ev)
            got = W.DWORD(0)
            ok = K.ReadFile(h, dst + done, want, None, ctypes.byref(ov))
            if not ok:
                if ctypes.get_last_error() != ERROR_IO_PENDING:
                    break
                ok = K.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), True)
            else:
                ok = K.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), False)
            if not ok:
                break
            done += got.value
            if got.value < want:
                break
        return done

    def read_direct(self, h, off, n, dst, copy=True):
        """Ds41FileBlobs::read_direct: the aligned span into the bounce, the range memcpy'd to dst."""
        a0, a1 = aligned(off, n)
        if a1 - a0 > self.n:
            raise ValueError("bounce too small")
        got = self.pread(h, self.p, a1 - a0, a0)
        if got < off + n - a0:
            raise OSError("short unbuffered read at %d (%d of %d)" % (a0, got, a1 - a0))
        if copy:
            ctypes.memmove(dst, self.p + (off - a0), n)


def unbuffered_header(path):
    """The GGUF header read through an unbuffered handle (a buffered read would itself create the cached section)."""
    h = _open(path, True)
    try:
        size = _size(h)
        r = Reader(1 << 20)
        try:
            def read(off, n):
                n = min(n, size)
                a1 = (n + ALIGN - 1) & ~(ALIGN - 1)
                if a1 > r.n:
                    r.close()
                    r.__init__(a1)
                got = r.pread(h, r.p, a1, 0)
                return ctypes.string_at(r.p, min(got, n))
            return Header(read, size)
        finally:
            r.close()
    finally:
        K.CloseHandle(h)


# ---------------------------------------------------------------- the other holders of the files (per mode)
class Holders:
    def __init__(self, paths, mode):
        self.paths, self.mode = paths, mode
        self.views, self.maps, self.handles = [], [], []

    def _map(self, path, touch_bytes):
        h = _open(path, False)
        m = K.CreateFileMappingW(h, None, PAGE_READONLY, 0, 0, None)
        if not m:
            K.CloseHandle(h)
            raise OSError(ctypes.get_last_error(), "CreateFileMapping " + path)
        v = K.MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0)
        if not v:
            K.CloseHandle(m)
            K.CloseHandle(h)
            raise OSError(ctypes.get_last_error(), "MapViewOfFile " + path)
        s = 0
        for o in range(0, min(touch_bytes, _size(h)), ALIGN):
            s += ctypes.c_ubyte.from_address(v + o).value
        return h, m, v

    def __enter__(self):
        if self.mode == "mapped":
            for p in self.paths:
                h, m, v = self._map(p, ALIGN)
                self.handles.append(h)
                self.maps.append(m)
                self.views.append(v)
        elif self.mode == "buffered":
            for p in self.paths:
                h = _open(p, False)
                buf = ctypes.create_string_buffer(ALIGN)
                r = Reader(ALIGN)
                try:
                    ov = OVERLAPPED()
                    ov.hEvent = r.ev
                    got = W.DWORD(0)
                    if not K.ReadFile(h, ctypes.addressof(buf), ALIGN, None, ctypes.byref(ov)):
                        K.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), True)
                finally:
                    r.close()
                self.handles.append(h)
        elif self.mode in ("remap", "remap-touch"):
            for p in self.paths:
                h, m, v = self._map(p, 64 << 20)
                K.UnmapViewOfFile(v)
                K.CloseHandle(m)
                K.CloseHandle(h)
            if self.mode == "remap-touch":
                for p in self.paths:
                    K.CloseHandle(_open(p, True))
        elif self.mode != "unmapped":
            raise SystemExit("unknown mode " + self.mode)
        return self

    def __exit__(self, *exc):
        for v in self.views:
            K.UnmapViewOfFile(v)
        for m in self.maps:
            K.CloseHandle(m)
        for h in self.handles:
            K.CloseHandle(h)
        self.views, self.maps, self.handles = [], [], []


def run(handles, blobs, threads, parts, copy):
    """T threads over blobs x parts (read_part); returns (bytes, seconds, per-read ms list)."""
    work = [(b, p) for b in range(len(blobs)) for p in range(parts)]
    maxspan = max(s[2] for spans in blobs for s in spans)
    blob_bytes = max(sum(s[2] for s in spans) for spans in blobs)
    nxt = [0]
    lock = threading.Lock()
    stats = []
    errors = []

    def worker():
        r = Reader(maxspan + 2 * ALIGN)
        dst = ctypes.create_string_buffer(blob_bytes)
        base = ctypes.addressof(dst)
        nbytes, ms = 0, []
        try:
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(work):
                    break
                b, p = work[i]
                for si, off, n, at in part_reads(blobs[b], p, parts):
                    t0 = time.perf_counter()
                    r.read_direct(handles[si], off, n, base + at, copy)
                    ms.append((time.perf_counter() - t0) * 1e3)
                    nbytes += n
        except Exception as e:  # noqa: BLE001 - reported by the caller
            errors.append(e)
        finally:
            r.close()
            with lock:
                stats.append((nbytes, ms))

    th = [threading.Thread(target=worker) for _ in range(threads)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return sum(s[0] for s in stats), dt, [x for s in stats for x in s[1]]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="the GGUF (any shard of a split one)")
    ap.add_argument("--blobs", type=int, default=400, help="random expert blobs per mode (default 400, ~1-3 GB)")
    ap.add_argument("--threads", type=int, default=32, help="reader threads (STRATA_WAVE_THREADS, default 32)")
    ap.add_argument("--parts", type=int, default=8, help="parts per blob (STRATA_WAVE_PARTS, default 8)")
    ap.add_argument("--modes", default="unmapped,mapped,buffered,remap,remap-touch,unmapped")
    ap.add_argument("--repeat", type=int, default=1, help="run the mode list this many times")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-copy", action="store_true", help="skip the bounce -> slot memcpy")
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args(argv)
    paths = split_paths(os.path.abspath(a.model))
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit("missing shard " + p)
    headers = [unbuffered_header(p) for p in paths]
    tab = expert_table(headers)
    if not tab:
        raise SystemExit("no blk.N.ffn_{gate,up,down}_exps tensors in " + a.model)
    blobs = pick_blobs(tab, a.blobs, a.seed)
    total = sum(s[2] for spans in blobs for s in spans)
    print("%d shard(s), %d expert layers; %d blobs, %.2f GB a mode; %d threads, %d parts%s"
          % (len(paths), len(tab), len(blobs), total / 1e9, a.threads, a.parts, ", no copy" if a.no_copy else ""))
    print("(the engine and any benchmark on this drive must not be running)")
    res = []
    for _ in range(a.repeat):
        for mode in a.modes.split(","):
            with Holders(paths, mode):
                handles = [_open(p, True) for p in paths]
                try:
                    nb, dt, ms = run(handles, blobs, a.threads, a.parts, not a.no_copy)
                finally:
                    for h in handles:
                        K.CloseHandle(h)
            row = {"mode": mode, "gb": nb / 1e9, "s": dt, "gbps": nb / dt / 1e9, "reads": len(ms),
                   "ms_p50": pct(ms, 0.5), "ms_p99": pct(ms, 0.99), "ms_max": max(ms) if ms else 0.0}
            res.append(row)
            print("%-12s %6.2f GB/s   %.2f GB in %.2f s   %d reads, ms p50 %.2f p99 %.2f max %.1f"
                  % (mode, row["gbps"], row["gb"], dt, row["reads"], row["ms_p50"], row["ms_p99"], row["ms_max"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "threads": a.threads, "parts": a.parts, "blobs": a.blobs,
                       "copy": not a.no_copy, "results": res}, f, indent=1)
    return res


if __name__ == "__main__":
    main()
