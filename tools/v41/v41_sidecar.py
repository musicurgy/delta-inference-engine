"""v41_sidecar.py - the DeepSeek-V4.1-Flash MTP heads + vision tower as a sidecar GGUF beside V41-GSQ.gguf (CPU only).

    python3 tools/v41/v41_sidecar.py --out DIR/V41-MTP-Vision.gguf [--mtp 0,1,2] [--no-vision]
    python3 tools/v41/v41_sidecar.py --out DIR/V41-1017-Vision.gguf --mtp ""       (the vision tower alone: shards 1-2)
    python3 tools/v41/v41_sidecar.py --verify DIR/V41-MTP-Vision.gguf [--sample 4]

The main GGUF leaves out mtp.* / vision.* / aligner.* / image_* (v41_gguf.py); this file holds them under their HF
names, so the 152 GiB model never has to be rewritten:
  - MTP routed experts (FP4 e2m1 + E8M0 per 32 in the checkpoint) -> MXFP4, a lossless byte repack (same value table,
    same scale byte; only the nibble order differs), stacked per role like the main file:
    mtp.N.ffn.experts.{w1,w2,w3}.weight  ggml [in, out, n_expert]
  - the vision tower's 2-D weights (BF16 in the checkpoint) -> BF16, the same bytes
  - every other 2-D weight (FP8 e4m3 + E8M0 per 32x32, or BF16) with a row length divisible by 32 -> Q8_0
  - norms, biases, router, mHC, the rest -> F32 (the router stays F32: its top-k is what the draft depends on)
Metadata: the vision config + image token id from config.json (the engine side reads them).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

from v41_gguf import MAIN, URL, Source  # noqa: E402

Q8 = gguf.GGMLQuantizationType.Q8_0
MX = gguf.GGMLQuantizationType.MXFP4
BF = gguf.GGMLQuantizationType.BF16
ROLES = ("w1", "w2", "w3")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def fp4_to_mxfp4(src, name):
    """[R, C/2] FP4 bytes (low nibble = even element) + [R, C/32] E8M0 -> [R, C/32 * 17] MXFP4 blocks
    (ggml block_mxfp4: e, then qs[j] = elem j | elem j+16 << 4)."""
    dt, (R, Ch), _, _ = src.info(name)
    assert dt == "I8", (name, dt)
    sn = name.removesuffix(".weight") + ".scale"
    sdt, sshape, _, _ = src.info(sn)
    nb = Ch * 2 // 32
    assert sdt == "F8_E8M0" and sshape == [R, nb], (sn, sdt, sshape)
    b = np.frombuffer(src.read(name), np.uint8).reshape(R, nb, 16)
    c = np.empty((R, nb, 32), np.uint8)
    c[..., 0::2], c[..., 1::2] = b & 15, b >> 4
    out = np.empty((R, nb, 17), np.uint8)
    out[..., 0] = np.frombuffer(src.read(sn), np.uint8).reshape(R, nb)
    out[..., 1:] = c[..., :16] | (c[..., 16:] << 4)
    return out.reshape(R, nb * 17)


def plan(src, mtps, vision):
    """[(gguf name, kind, payload)] in write order.  kind: 'mx' (stacked experts), 'q8', 'bf16', 'f32'."""
    wm = src.wm
    out = []
    for m in mtps:
        p = f"mtp.{m}."
        names = sorted(n for n in wm if n.startswith(p) and ".experts." not in n and not n.endswith(".scale"))
        assert names, f"no {p}* tensors in the checkpoint index"
        for n in names:
            out.append((n, None, n))
        ne = 1 + max(int(n.split(".")[4]) for n in wm if n.startswith(p + "ffn.experts."))
        for r in ROLES:
            out.append((f"{p}ffn.experts.{r}.weight", "mx", [f"{p}ffn.experts.{e}.{r}.weight" for e in range(ne)]))
    if vision:
        for n in sorted(wm):
            if n.startswith(("vision.", "aligner.", "image_")):
                # the checkpoint's BF16 matrices as they are: Q8_0's rounding compounds through the 32 ViT layers (worst
                # span row cosine 0.996 vs the checkpoint, test_ds41_vision.py), and the encoder upcasts at load anyway,
                # so lossless costs only file size (0.52 -> 0.97 GB)
                dt, shape, _, _ = src.info(n)
                out.append((n, "bf16" if dt == "BF16" and len(shape) == 2 else None, n))
    res = []
    for name, kind, pay in out:
        if kind is None:
            dt, shape, _, _ = src.info(pay)
            kind = "q8" if len(shape) == 2 and shape[1] % 32 == 0 and shape[0] >= 32 and dt != "F32" else "f32"
        res.append((name, kind, pay))
    return res


def tensor_info(src, name, kind, pay):
    """(byte or value shape, numpy dtype, nbytes, raw ggml type)"""
    if kind == "mx":
        dt, (R, Ch), _, _ = src.info(pay[0])
        shape = (len(pay), R, Ch * 2 // 32 * 17)
        return shape, np.uint8, int(np.prod(shape)), MX
    dt, shape, _, _ = src.info(pay)
    if kind == "q8":
        if dt == "I8":
            raise ValueError(f"{name}: FP4 outside the expert stacks")
        R, C = shape
        shape = (R, C // 32 * 34)
        return shape, np.uint8, R * C // 32 * 34, Q8
    if kind == "bf16":
        R, C = shape
        return (R, C * 2), np.uint8, R * C * 2, BF
    return tuple(shape), np.float32, int(np.prod(shape)) * 4, None


def tensor_data(src, name, kind, pay):
    if kind == "mx":
        return np.stack([fp4_to_mxfp4(src, n) for n in pay])
    if kind == "bf16":
        _, (R, C), _, _ = src.info(pay)
        return np.frombuffer(src.read(pay), np.uint8).reshape(R, C * 2)
    v = src.f32(pay)
    if kind == "q8":
        return quants.quantize(v.astype(np.float32), Q8)
    return np.ascontiguousarray(v, np.float32)


def write(args):
    src = Source()
    cfg = json.load(open(os.path.join(MAIN, "config.json")))
    mtps = [int(x) for x in args.mtp.split(",")] if args.mtp else []
    pl = plan(src, mtps, not args.no_vision)
    w = gguf.GGUFWriter(args.out + ".tmp", "deepseek41-sidecar")
    what = " + ".join(p for p, on in (("MTP heads", mtps), ("vision tower", not args.no_vision)) if on)
    w.add_name(os.environ.get("V41_SRC_NAME", "DeepSeek V4.1 Flash UNCENSORED") + f" - {what} (Strata v41 sidecar)")
    w.add_string("deepseek41.sidecar.source", URL)
    w.add_array("deepseek41.sidecar.mtp_blocks", mtps)
    w.add_string("deepseek41.sidecar.mtp_experts_type", "MXFP4 (lossless repack of the checkpoint FP4)")
    if not args.no_vision:
        w.add_string("deepseek41.sidecar.vision_config", json.dumps(cfg.get("vision_config", {})))
        w.add_uint32("deepseek41.sidecar.image_token_id", int(cfg.get("image_token_id", 0)))
    total = 0
    for name, kind, pay in pl:
        shape, dt, nb, raw = tensor_info(src, name, kind, pay)
        w.add_tensor_info(name, shape, np.dtype(dt), nb, raw_dtype=raw)
        total += nb
    log(f"{len(pl)} tensors, {total / 1e9:.2f} GB -> {args.out}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, (name, kind, pay) in enumerate(pl):
        a = tensor_data(src, name, kind, pay)
        w.write_tensor_data(a)
        if kind == "mx" or i % 50 == 0:
            log(f"  [{i + 1}/{len(pl)}] {name} {kind} {a.nbytes / 1e6:.1f} MB")
    w.close()
    os.replace(args.out + ".tmp", args.out)
    log("done")


def verify(args):
    """Experts: dequantized MXFP4 == the checkpoint's FP4 values exactly.  BF16: the checkpoint's bytes.  Q8_0: relative
    RMS error per tensor."""
    src = Source()
    r = gguf.GGUFReader(args.verify)
    rng = np.random.default_rng(0)
    worst, n_mx, n_q8, n_bf = 0.0, 0, 0, 0
    for t in r.tensors:
        if t.tensor_type == MX:
            ne = int(t.shape[2])
            for e in rng.choice(ne, min(args.sample, ne), replace=False):
                hf = t.name.replace("experts.", f"experts.{e}.")
                ref = src.f32(hf)
                got = quants.dequantize(np.asarray(t.data[e]), MX).reshape(ref.shape)
                if not np.array_equal(got, ref):
                    raise SystemExit(f"MISMATCH {hf}: max abs {np.abs(got - ref).max()}")
                n_mx += 1
        elif t.tensor_type == Q8:
            ref = src.f32(t.name)
            got = quants.dequantize(np.asarray(t.data), Q8).reshape(ref.shape)
            rel = float(np.sqrt(np.mean((got - ref) ** 2)) / max(np.sqrt(np.mean(ref ** 2)), 1e-30))
            worst = max(worst, rel)
            n_q8 += 1
        elif t.tensor_type == BF:
            if np.asarray(t.data).tobytes() != src.read(t.name):
                raise SystemExit(f"MISMATCH {t.name}: BF16 bytes differ from the checkpoint")
            n_bf += 1
    print(f"verify: {n_mx} MXFP4 experts bit-exact vs FP4; {n_bf} BF16 tensors byte-identical; {n_q8} Q8_0 tensors, "
          f"worst relative RMS error {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--mtp", default="0,1,2")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--verify")
    ap.add_argument("--sample", type=int, default=4)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    elif args.out:
        write(args)
    else:
        ap.error("--out or --verify")


if __name__ == "__main__":
    main()
