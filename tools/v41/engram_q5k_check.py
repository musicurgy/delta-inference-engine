"""Engram Q5_K rows, bit for bit: the engine's reader against gguf-py's reference dequantizer.

    python -I tools/v41/engram_q5k_check.py DUMP TABLE_SHARD.gguf

DUMP is test_ds41_engram_src --dump's output: records of (u32 slot, u64 row of the slot's table, u64 row of the combined
per_layer_token_embd tensor, dim f32).  Each record's row is decoded from the shard's own Q5_K block with
gguf.quants.dequantize and compared as raw float32 bits.  Exit 0 and "IDENTICAL" when every row matches.
"""
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize


def main() -> int:
    dump, shard = sys.argv[1], sys.argv[2]
    t = next(x for x in GGUFReader(shard).tensors if x.name == "per_layer_token_embd.weight")
    if t.tensor_type != GGMLQuantizationType.Q5_K:
        print(f"{shard}: per_layer_token_embd.weight is {t.tensor_type.name}, not Q5_K")
        return 2
    dim = int(t.shape[0])
    blocks = np.asarray(t.data).reshape(int(t.shape[1]), -1)   # one row = dim / 256 blocks of 176 bytes
    rec = np.dtype([("slot", "<u4"), ("row", "<u8"), ("abs", "<u8"), ("v", "<f4", (dim,))])
    recs = np.fromfile(dump, dtype=rec)
    bad = 0
    for r in recs:
        ref = dequantize(np.ascontiguousarray(blocks[int(r["abs"])]), GGMLQuantizationType.Q5_K).astype(np.float32).reshape(-1)
        if not np.array_equal(ref.view(np.uint32), np.ascontiguousarray(r["v"]).view(np.uint32)):
            bad += 1
            if bad <= 5:
                d = np.abs(ref - r["v"]).max()
                print(f"DIFFERS: slot {int(r['slot'])} row {int(r['row'])} (abs {int(r['abs'])}): max |diff| {d:g}")
    slots = sorted(set(int(s) for s in recs["slot"]))
    print(f"{len(recs)} rows ({len(set(int(a) for a in recs['abs']))} distinct, slots {slots}), {bad} differ from gguf-py's Q5_K dequant")
    print("IDENTICAL" if bad == 0 and len(recs) > 0 else "DIFFERENT")
    return 0 if bad == 0 and len(recs) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
