#!/usr/bin/env python3
r"""get_engine.py - download the prebuilt engine for this CPU from the project's GitHub Releases page, so no compiler,
CUDA Toolkit or Git is needed (first-start.bat offers it before building).

    get_engine.py            build-ds41-portable.zip (CPUs with AVX2), build-ds41-compat-avx.zip (AVX, no AVX2) or
                             build-ds41-compat.zip (older) -> unpacked into this folder under the same name
    get_engine.py --url      only prints where it would download from (exit 2 if it does not know)

Where from, first that is set: DS41_RELEASE_URL; the first line of tools/v41/serve/release-url.txt that is not a comment
or the placeholder; this clone's `origin` remote on GitHub (.../OWNER/REPO/releases/latest/download/).  The zip is
checked against the release's SHA256SUMS.txt before it is unpacked (a release without one: said, not checked).
Exit 0 done, 1 failed (says why), 2 no release location known."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_level  # noqa: E402  (same folder)

ROOT = cpu_level.ROOT
URL_FILE = os.path.join(ROOT, "tools", "v41", "serve", "release-url.txt")


def release_base():
    v = os.environ.get("DS41_RELEASE_URL", "").strip()
    if v:
        return v.rstrip("/") + "/"
    try:
        with open(URL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "OWNER/REPO" not in line:
                    return line.rstrip("/") + "/"
    except OSError:
        pass
    try:
        r = subprocess.run(["git", "-C", ROOT, "remote", "get-url", "origin"], capture_output=True, text=True, timeout=10)
        m = re.match(r"(?:https://github\.com/|git@github\.com:)([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", r.stdout.strip())
        if r.returncode == 0 and m:
            return f"https://github.com/{m.group(1)}/{m.group(2)}/releases/latest/download/"
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def published_sha256(base, name):
    """the zip's SHA-256 from the release's SHA256SUMS.txt ("<hex> *<name>" lines), or None when the release has none
    (0.9 had none) or it cannot be read"""
    try:
        req = urllib.request.Request(base + "SHA256SUMS.txt", headers={"User-Agent": "ds41-first-start"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read(1 << 16).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower()
    return None


def download(url, dst):
    """the file at url into dst; returns its SHA-256"""
    req = urllib.request.Request(url, headers={"User-Agent": "ds41-first-start"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, t0, last = 0, time.time(), 0.0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            got += len(b)
            if time.time() - last > 2:
                last = time.time()
                pct = f"{100 * got / total:5.1f}%" if total else ""
                print(f"\r  {got / 1e6:8.0f} MB {pct}  {got / 1e6 / max(1e-3, last - t0):6.1f} MB/s", end="", flush=True)
    print()
    if total and got != total:
        raise OSError(f"the download stopped at {got} of {total} bytes")
    return h.hexdigest()


def main():
    base = release_base()
    if base is None:
        print("No release location known (no GitHub 'origin' remote, no tools\\v41\\serve\\release-url.txt).")
        return 2
    name = cpu_level.release_zip()
    folder = name[:-4]
    url = base + name
    if sys.argv[1:] == ["--url"]:
        print(url)
        return 0
    tmp = os.path.join(ROOT, name + ".part")
    print(f"Downloading the prebuilt engine: {url}")
    want = published_sha256(base, name)
    try:
        got = download(url, tmp)
    except (urllib.error.URLError, OSError) as e:
        code = getattr(e, "code", None)
        print(f"  failed: {e}" + ("  (no such file on the Releases page yet)" if code == 404 else ""))
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1
    try:
        if want is None:
            print("  (the release publishes no SHA256SUMS.txt line for it: not checked)")
        elif got != want:
            raise zipfile.BadZipFile(f"its SHA-256 {got[:16]}... is not the release's {want[:16]}... (SHA256SUMS.txt)")
        else:
            print("  SHA-256 checked against the release's SHA256SUMS.txt")
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"damaged member {bad}")
            names = z.namelist()
            if f"{folder}/ds41_generate.exe" not in names:
                raise zipfile.BadZipFile(f"no {folder}/ds41_generate.exe in it")
            for n in names:   # only plain files inside that one folder
                if not n.startswith(folder + "/") or ".." in n.split("/") or ":" in n or n.startswith("/"):
                    raise zipfile.BadZipFile(f"unexpected entry {n!r}")
            dst = os.path.join(ROOT, folder)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            z.extractall(ROOT)
    except zipfile.BadZipFile as e:
        print(f"  the downloaded file is not a usable engine zip: {e}")
        return 1
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"  unpacked into {folder}\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
