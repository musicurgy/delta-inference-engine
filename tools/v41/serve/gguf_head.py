"""gguf_head.py - read a GGUF file's header only: the metadata keys and the tensor table (name, dims, type, offset),
without any third-party module.  The launcher uses it to see which type a split model stores its Engram tables in.

    from gguf_head import tensor_types
    tensor_types("model-00001-of-00007.gguf")  -> {"token_embd.weight": 8, "per_layer_token_embd.weight": 14, ...}
"""
import struct

# GGUF value types (gguf.h)
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
GGML_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
                   11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
                   19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
                   27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4", 43: "MXFP8"}


class _R:
    def __init__(self, f):
        self.f = f

    def u(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, b)[0]

    def s(self):
        n = self.u("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.s()
        if t == ARRAY:
            at = self.u("<I")
            n = self.u("<Q")
            if at in _SCALARS and n > 1 << 16:   # a big numeric array (a tokenizer's scores): skip it
                self.f.seek(n * struct.calcsize(_SCALARS[at]), 1)
                return None
            return [self.value(at) for _ in range(n)]
        return self.u(_SCALARS[t])


def read_header(path):
    """(metadata dict, tensors list of (name, dims, ggml type, offset)) from the file's header"""
    meta, tensors, _ = _read_header(path)
    return meta, tensors


def _read_header(path):
    """read_header + the byte where the header ends (the data section starts at the next general.alignment)"""
    with open(path, "rb") as f:
        r = _R(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.u("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} is too old")
        n_tensors = r.u("<Q")
        n_kv = r.u("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.s()
            t = r.u("<I")
            meta[key] = r.value(t)
        tensors = []
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u("<I")
            dims = [r.u("<Q") for _ in range(nd)]
            typ = r.u("<I")
            off = r.u("<Q")
            tensors.append((name, dims, typ, off))
        end = f.tell()
    return meta, tensors, end


# GGUF type ids the engine has a block layout for (block_geometry in include/strata/artifact/gguf_reader.hpp; checked by
# test_gguf_head.py).  Any other id - ik_llama's 144+ (IQK / trellis types), Q8_K, F64, TQ1_0 / TQ2_0, Q1_0 - the engine
# refuses by name (ds41_geometry.hpp ds41_type_known).
KNOWN_TYPES = frozenset({0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                         29, 30, 39, 40, 42, 43, 143})


def layout43(dims, room):
    """what a type-43 tensor holds, from its dims and the bytes it has in the file (up to the next tensor or the end):
    "mxfp8" (Unsloth's Engram tables: one E8M0 scale + 32 E4M3 bytes per 32 values - what the engine reads),
    "f8_e4m3" (mainline ggml's GGML_TYPE_F8_E4M3 = 43, llama.cpp v0.9.0+: one byte a value, no block scale) or
    "unknown" (truncated).  The same rule as ds41_geometry.hpp ds41_type43_layout."""
    n = 1
    for d in dims:
        n *= d
    if dims and dims[0] % 32 == 0 and room >= n // 32 * 33:
        return "mxfp8"
    if room >= n:
        return "f8_e4m3"
    return "unknown"


def type43_layout(path, name):
    """layout43 of tensor `name` over the shards of the model `path` belongs to, or None when no shard holds it"""
    import os
    for p in split_paths(path):
        if not os.path.exists(p):
            continue
        meta, tensors, end = _read_header(p)
        for tn, dims, _, off in tensors:
            if tn != name:
                continue
            align = meta.get("general.alignment") or 32
            payload = os.path.getsize(p) - (end + align - 1) // align * align
            nxt = min((o for _, _, _, o in tensors if o > off), default=payload)
            return layout43(dims, min(nxt, payload) - off)
    return None


def split_paths(path):
    """every shard of a split model (<stem>-NNNNN-of-MMMMM.gguf) in order, or [path] for a single file"""
    import re
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    return [path[:m.start(1)] + f"{i:05d}" + path[m.end(1):] for i in range(1, n + 1)]


def tensor_types(path):
    """{tensor name: ggml type id} over every shard of the model (the first shard of a split may hold no tensors)"""
    out = {}
    import os
    for p in split_paths(path):
        if os.path.exists(p):
            out.update({name: typ for name, _, typ, _ in read_header(p)[1]})
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        tt = tensor_types(p)
        print(p, "-", len(tt), "tensors over", len(split_paths(p)), "shard(s)")
        for name, typ in tt.items():
            if "embd" in name or name.startswith("output") or "engram" in name:
                print(f"  {name:<36} {GGML_TYPE_NAMES.get(typ, typ)}")


# bytes a block / elements a block of the ggml types a routed-expert tensor can have (for expert_gib below)
_BLOCK = {0: (4, 1), 1: (2, 1), 30: (2, 1), 2: (18, 32), 8: (34, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256),
          13: (176, 256), 14: (210, 256), 16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
          21: (110, 256), 22: (82, 256), 23: (136, 256), 39: (17, 32)}


def expert_gib(path):
    """the routed experts' size in GiB over every shard (from the tensor table: dims x the type's block size), or None
    when a type is unknown"""
    total = 0
    for p in split_paths(path):
        _, tensors = read_header(p)
        for name, dims, typ, _ in tensors:
            if "_exps" not in name and "experts" not in name:
                continue
            if typ not in _BLOCK:
                return None
            n = 1
            for d in dims:
                n *= d
            bb, be = _BLOCK[typ]
            total += n // be * bb
    return total / 1073741824.0
