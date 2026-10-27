#!/usr/bin/env python3
"""hf_verify.py - check downloaded files against the SHA-256 the Hugging Face repo publishes for them, and mark each one
that matches with NAME.ok (setup-ds41-models.bat skips a file that has its .ok; v41_sidecar.py reads only marked shards).

    hf_verify.py --repo deepseek-ai/DeepSeek-V4.1-Flash-1017 --dir models\\engram model-00044-of-00048.safetensors ...

Exit 1 when a file is missing or differs (a damaged or partial download: delete it and run the setup again).
"""
import argparse
import hashlib
import os
import sys
import time

from huggingface_hub import HfApi


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    info = HfApi().model_info(a.repo, files_metadata=True)
    want = {s.rfilename: (s.lfs.sha256 if getattr(s, "lfs", None) else None) for s in info.siblings}
    bad = 0
    for name in a.files:
        path = os.path.join(a.dir, name)
        if os.path.exists(path + ".ok"):
            print(f"  ok (checked before)  {name}")
            continue
        if not os.path.exists(path):
            print(f"  MISSING  {path}")
            bad += 1
            continue
        if want.get(name) is None:
            print(f"  NO HASH  {name}: the repo publishes no SHA-256 for it")
            bad += 1
            continue
        t0 = time.time()
        got = sha256(path)
        if got != want[name]:
            print(f"  DIFFERS  {name}: {got[:12]} (the repo's: {want[name][:12]}) - delete it and run the setup again")
            bad += 1
            continue
        open(path + ".ok", "w", encoding="ascii").write(got + "\n")
        print(f"  ok  {name}  ({os.path.getsize(path) / 1e9:.1f} GB checked in {time.time() - t0:.0f} s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
