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
    return meta, tensors


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
