#!/usr/bin/env python3
r"""make_release.py - the prebuilt-engine zips for the GitHub Releases page (first-start.bat downloads them).

    make_release.py [OUT_DIR]     (default: dist\)

Zips build-ds41-portable\, build-ds41-compat-avx\ and build-ds41-compat\ (build-ds41.bat portable / compat-avx /
compat) - only what the engine needs: its .exe and .dll files and cpu-level.txt, under the folder's name, plus the
license texts the binaries carry (LICENSE.txt: this project's MIT license; THIRD-PARTY-NOTICES.txt: ggml's MIT license
and NVIDIA's CUDA runtime DLLs) - and prints each zip's size and SHA-256.  Upload all three to a release; get_engine.py
takes them from .../releases/latest/download/."""
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NVIDIA_NOTE = ("cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll: NVIDIA CUDA runtime libraries, Copyright NVIDIA\n"
               "Corporation, redistributed unmodified under the NVIDIA CUDA Toolkit End User License Agreement\n"
               "(https://docs.nvidia.com/cuda/eula/).\n")


def notices():
    """(LICENSE.txt, THIRD-PARTY-NOTICES.txt) texts for the zips, with Windows line endings"""
    def read(*parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
            return f.read().replace("\r\n", "\n")
    ggml = os.path.join("third_party", "llama.cpp", "LICENSE")
    if not os.path.exists(os.path.join(ROOT, ggml)):
        ggml = os.path.join("third_party", "ggml", "LICENSE")
    third = ("The engine (ds41_generate.exe and the tools beside it) is built with ggml / llama.cpp:\n\n"
             + read(ggml).rstrip("\n") + "\n\n" + NVIDIA_NOTE)
    return read("LICENSE").replace("\n", "\r\n"), third.replace("\n", "\r\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist")
    os.makedirs(out, exist_ok=True)
    made = 0
    lic, third = notices()
    for folder in ("build-ds41-portable", "build-ds41-compat-avx", "build-ds41-compat"):
        src = os.path.join(ROOT, folder)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".exe", ".dll")) or f == "cpu-level.txt") \
            if os.path.isdir(src) else []
        if "ds41_generate.exe" not in files or "cpu-level.txt" not in files:
            print(f"skip {folder}: no ds41_generate.exe / cpu-level.txt (build it with build-ds41.bat "
                  f"{folder.split('-')[-1]})")
            continue
        zpath = os.path.join(out, folder + ".zip")
        with zipfile.ZipFile(zpath + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                z.write(os.path.join(src, f), f"{folder}/{f}")
            z.writestr(f"{folder}/LICENSE.txt", lic)
            z.writestr(f"{folder}/THIRD-PARTY-NOTICES.txt", third)
        os.replace(zpath + ".part", zpath)
        h = hashlib.sha256()
        with open(zpath, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        print(f"{zpath}: {os.path.getsize(zpath) / 1e6:.0f} MB, {len(files) + 2} files, sha256 {h.hexdigest()}")
        made += 1
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
