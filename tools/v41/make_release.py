#!/usr/bin/env python3
r"""make_release.py - the prebuilt-engine zips for the GitHub Releases page (first-start.bat downloads them).

    make_release.py [OUT_DIR]     (default: dist\)

Zips build-ds41-portable\, build-ds41-compat-avx\ and build-ds41-compat\ (build-ds41.bat portable / compat-avx /
compat) - only what the engine needs: its .exe and .dll files and cpu-level.txt, under the folder's name - and prints
each zip's size and SHA-256.  Upload all three to a release; get_engine.py takes them from
.../releases/latest/download/."""
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist")
    os.makedirs(out, exist_ok=True)
    made = 0
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
        os.replace(zpath + ".part", zpath)
        h = hashlib.sha256()
        with open(zpath, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        print(f"{zpath}: {os.path.getsize(zpath) / 1e6:.0f} MB, {len(files)} files, sha256 {h.hexdigest()}")
        made += 1
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
