#!/usr/bin/env python3
"""drive_check.py - where should the model files go?  first-start.bat runs it once, before the first big download.

  drive_check.py --pick FOLDER   list this PC's drives (free space; NVMe / SATA / USB; SSD or hard disk), check the page
                                 file, measure the download speed for a few seconds, and let the user choose the model
                                 folder.  A new choice is written into ds41-paths.bat (DS41_MODELS).  Advice only: it
                                 never stops the setup.
  drive_check.py --info FOLDER   the same facts, no question (DS41_MODELS is set outside ds41-paths.bat)
No administrator rights needed (PowerShell's Get-Disk / Get-PhysicalDisk read the bus and media type).
DS41_NO_SPEED_PROBE=1 skips the download test.
"""
from __future__ import annotations

import os
import re
import shutil
import string
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PATHS_BAT = os.path.join(ROOT, "ds41-paths.bat")
GB = 1e9
NEED_DEFAULT_GB = 385          # the default model while it downloads (setup_space.py)
NEED_SMALLEST_GB = 250         # Unsloth UD-IQ1_S with the drafter
SIZES = [("the default model", 375), ("Unsloth UD-IQ2_XXS", 309), ("Unsloth UD-IQ1_S, the smallest", 241)]
PROBE_URL = ("{endpoint}/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF/resolve/main/"
             "DeepSeek-V4.1-Flash-1017-Q2.gguf")
NEW_FOLDER = "DeepSeek-V4.1"   # on another drive: X:\DeepSeek-V4.1

_PS = r"""
$pd = @{}
Get-PhysicalDisk | ForEach-Object { $pd[[string]$_.DeviceId] = $_ }
Get-Partition | Where-Object { $_.DriveLetter } | ForEach-Object {
  $d = Get-Disk -Number $_.DiskNumber
  $p = $pd[[string]$_.DiskNumber]
  $m = if ($p) { $p.MediaType } else { '' }
  'DRIVE|{0}|{1}|{2}|{3}' -f $_.DriveLetter, $d.BusType, $m, $d.FriendlyName
}
'PAGEFILE|' + (Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile
"""


# ------------------------------------------------------------------------------------------------ facts
def disk_types() -> tuple[dict, bool | None]:
    """{letter: (bus, media, name)} and whether Windows manages the page file, from one PowerShell call (no admin);
    ({}, None) when it cannot be read"""
    if os.name != "nt":
        return {}, None
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                            _PS], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}, None
    return parse_ps(r.stdout)


def parse_ps(text: str) -> tuple[dict, bool | None]:
    types, auto = {}, None
    for line in text.splitlines():
        f = line.strip().split("|")
        if f[0] == "DRIVE" and len(f) >= 5 and f[1]:
            types[f[1].upper()] = (f[2], f[3], "|".join(f[4:]).strip())
        elif f[0] == "PAGEFILE" and len(f) == 2:
            auto = f[1].strip().lower() == "true"
    return types, auto


def drive_letters() -> list[tuple[str, str]]:
    """[(letter, "fixed" | "removable" | "network" | "other")] of the drives Windows shows"""
    if os.name != "nt":
        return []
    import ctypes
    k = ctypes.windll.kernel32
    mask = k.GetLogicalDrives()
    out = []
    for i, letter in enumerate(string.ascii_uppercase):
        if mask & (1 << i):
            t = k.GetDriveTypeW(f"{letter}:\\")
            out.append((letter, {2: "removable", 3: "fixed", 4: "network"}.get(t, "other")))
    return out


def memory() -> tuple[int, int] | None:
    """(RAM bytes, commit limit bytes = RAM + page file), or None"""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                    ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]
    m = MS()
    m.len = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return m.total, m.tpage


def collect(types: dict, letters: list) -> list[dict]:
    """one dict per drive with room for files: letter, kind, bus, media, name, free, total, rank (higher = faster)"""
    out = []
    for letter, kind in letters:
        if kind not in ("fixed", "removable"):
            continue
        try:
            u = shutil.disk_usage(f"{letter}:\\")
        except OSError:                     # an empty card reader, a locked drive
            continue
        bus, media, name = types.get(letter, ("", "", ""))
        out.append({"letter": letter, "kind": kind, "bus": bus, "media": media, "name": name, "free": u.free,
                    "total": u.total, "rank": rank(bus, media, kind)})
    return out


def rank(bus: str, media: str, kind: str = "fixed") -> int:
    """4 NVMe, 3 SSD on SATA/SAS, 2 unknown (RAID, Storage Spaces...), 1 USB or hard disk"""
    b, m = (bus or "").lower(), (media or "").lower()
    if b == "usb" or kind == "removable" or m == "hdd":
        return 1
    if b == "nvme":
        return 4
    if m == "ssd":
        return 3
    return 2


def kind_words(d: dict) -> str:
    b, m = d["bus"], (d["media"] or "").upper()
    media = {"SSD": "SSD", "HDD": "hard disk"}.get(m, "")
    if b == "NVMe":
        return "NVMe SSD"
    if b == "USB" or d["kind"] == "removable":
        return f"USB {media}".strip()
    if b:
        return f"{b} {media}".strip()
    return media or "drive"


def speed_words(d: dict) -> str:
    return {4: "fastest", 3: "slower start than NVMe", 2: "type unknown",
            1: "slow: model loads and answers suffer"}[d["rank"]]


def recommend(drives: list[dict], current_letter: str | None, need_gb: float = NEED_DEFAULT_GB) -> dict | None:
    """the drive to suggest: the fastest with room for the default model (more free space breaks a tie); the current
    folder's drive when it is as good as the best"""
    fits = [d for d in drives if d["free"] >= need_gb * GB]
    if not fits:
        fits = [d for d in drives if d["free"] >= NEED_SMALLEST_GB * GB]
    if not fits:
        return None
    best = max(fits, key=lambda d: (d["rank"], d["free"]))
    cur = next((d for d in fits if d["letter"] == current_letter), None)
    return cur if cur is not None and cur["rank"] >= best["rank"] else best


def probe_speed(seconds: float = 6.0, url: str | None = None) -> float | None:
    """the download speed from Hugging Face now, in bytes/s: the start of the model file for a few seconds (the
    bytes are thrown away).  None when it cannot be measured"""
    if os.environ.get("DS41_NO_SPEED_PROBE") == "1":
        return None
    import urllib.request
    url = url or os.environ.get("DS41_PROBE_URL") or PROBE_URL.format(
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/"))
    req = urllib.request.Request(url, headers={"Range": "bytes=0-2000000000", "User-Agent": "delta-setup"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            t0 = time.time()
            n = 0
            while time.time() - t0 < seconds:
                b = r.read(1 << 20)
                if not b:
                    break
                n += len(b)
            dt = time.time() - t0
    except Exception:                       # noqa: BLE001  offline, a proxy: no estimate
        return None
    return n / dt if dt > 0.5 and n > (1 << 20) else None


def duration(s: float) -> str:
    m = round(s / 60)
    if m < 60:
        return f"{max(1, m)} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min" if m else f"{h} h"


# ------------------------------------------------------------------------------------------------ ds41-paths.bat
LINE = re.compile(r'^if not defined DS41_MODELS set "DS41_MODELS=[^"\r\n]*"', re.M)


def write_models_folder(folder: str, bat: str = PATHS_BAT) -> bool:
    """set DS41_MODELS's default line in ds41-paths.bat to folder (forward slashes, as its example); the file keeps its
    CRLF line ends.  -> False when the line is not there (an edited file: the user is told what to set)"""
    try:
        with open(bat, "rb") as f:
            text = f.read().decode("utf-8")
    except OSError:
        return False
    if not LINE.search(text):
        return False
    value = folder.replace("\\", "/").rstrip("/")
    new = LINE.sub(lambda m: f'if not defined DS41_MODELS set "DS41_MODELS={value}"', text, count=1)
    with open(bat, "wb") as f:
        f.write(new.encode("utf-8"))
    return True


def has_files(folder: str) -> bool:
    """a folder that already holds model files or a download in progress: the setup goes on there"""
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if name.endswith((".gguf", ".incomplete")) or name in ("unsloth", "exl3", "custom", "deepseek", "engram"):
            return True
        if name == ".cache" and any(f.endswith(".incomplete") for _, _, fs in os.walk(p) for f in fs):
            return True
    return False


# ------------------------------------------------------------------------------------------------ the question
def report(folder: str, drives: list[dict], auto_pf, mem, rate, out=print) -> None:
    letter = os.path.splitdrive(os.path.abspath(folder))[0][:1].upper() or None
    out("\n  Where the model goes matters: the model is read from the drive at every start (and, with less than 128 GB")
    out("  of RAM, on every word).  An internal NVMe drive is best.  The files need 240-400 GB depending on the model.\n")
    if drives:
        for i, d in enumerate(drives, 1):
            mark = []
            if d["letter"] == letter:
                mark.append("<- now")
            if d["free"] < NEED_SMALLEST_GB * GB:
                mark.append("too full")
            name = f"  ({d['name']})" if d["name"] else ""
            out(f"   {i}. {d['letter']}:    {kind_words(d):<16} {d['free'] / GB:>6,.0f} GB  {speed_words(d)}"
                f"{'  ' + ', '.join(mark) if mark else ''}{name}")
    for line in memory_advice(mem, auto_pf):
        out(line)
    if rate:
        mbs = rate / 1e6
        parts = [f"{what} ({gb} GB) about {duration(gb * GB / rate)}" for what, gb in SIZES]
        out(f"\n  Download speed now: about {mbs:.0f} MB/s.  At that speed: " + "; ".join(parts) + ".")
        out("  The PC stays awake while it downloads; a download that stops is started again by itself.")


def memory_advice(mem, auto_pf) -> list[str]:
    """the page-file lines (docs/DETAILS.md: a tiny page file makes the graphics card's allocations fail)"""
    if not mem:
        return []
    ram, commit = mem
    pf = max(0, commit - ram)
    if pf < 4 * 2**30:
        return [f"\n  Page file: {pf / 2**30:.0f} GB - too small.  With the model loaded Windows can refuse memory (\"out of",
                "  memory\" although memory is free).  Set a fixed page file of 65536 MB: Settings > System > About >",
                "  Advanced system settings > Performance: Settings > Advanced > Virtual memory: Change > Custom size",
                "  (the same initial and maximum size), then restart Windows."]
    if auto_pf:
        return [f"\n  Page file: managed by Windows (now {pf / 2**30:.0f} GB).  Usually fine.  If the start ever fails with",
                "  \"out of memory\" although memory is free, set a fixed 65536 MB page file (docs/DETAILS.md, \"out of memory\")."]
    return [f"\n  Page file: {pf / 2**30:.0f} GB, fixed - fine."]


def ask(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def choose(folder: str, drives: list[dict], answer=ask, out=print) -> str:
    """the user's folder: Enter = the recommended one, a number = that drive, or a typed folder.  -> the folder"""
    cur = os.path.abspath(folder)
    letter = os.path.splitdrive(cur)[0][:1].upper()
    rec = recommend(drives, letter)
    if rec is None:
        out("\n  No drive has room for a model (240 GB at least).  Free some space, or attach a bigger drive, then start")
        out("  again.  (The setup goes on and stops before the download if it does not fit.)")
        return cur

    def folder_on(d):
        return cur if d["letter"] == letter else f"{d['letter']}:\\{NEW_FOLDER}"
    n = drives.index(rec) + 1
    out(f"\n  Recommended: {folder_on(rec)}")
    if rec["rank"] <= 1:
        out("  (It is the only drive with room, and it is slow.  Freeing space on an internal SSD is worth it.)")
    while True:
        a = answer(f"  Enter = {n}, a drive number (1-{len(drives)}), or a folder (like D:\\{NEW_FOLDER}): ")
        if a is None:                       # no console to answer: nothing changes
            return cur
        if a == "":
            return folder_on(rec)
        if a.isdigit() and 1 <= int(a) <= len(drives):
            d = drives[int(a) - 1]
            if d["free"] < NEED_SMALLEST_GB * GB:
                out(f"  {d['letter']}: has {d['free'] / GB:.0f} GB free - too little.  Pick another.")
                continue
            return folder_on(d)
        p = a.strip('"')
        if re.match(r"^[A-Za-z]:[\\/]", p):
            if not os.path.isdir(p[:3]):
                out(f"  There is no drive {p[:2]}.  Try again.")
                continue
            if any(ord(c) > 127 for c in p):
                out("  Please use a folder name with plain letters (A-Z, digits, - and _).")
                continue
            try:
                free = shutil.disk_usage(p[:3]).free
            except OSError:
                free = None
            if free is not None and free < NEED_SMALLEST_GB * GB:
                out(f"  Note: {p[:2]} has {free / GB:.0f} GB free; the setup stops before the download if the model "
                    "does not fit.")
            return os.path.normpath(p)
        out("  Please type a number from the list, or a full folder like D:\\" + NEW_FOLDER + ".")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("--pick", "--info"):
        print(__doc__)
        return 2
    folder = argv[1]
    types, auto_pf = disk_types()
    drives = collect(types, drive_letters())
    keep = argv[0] == "--info" or has_files(folder)
    rate = probe_speed()
    report(folder, drives, auto_pf, memory(), rate)
    if argv[0] == "--info":
        print(f"\n  The model folder is {folder} (DS41_MODELS, set outside ds41-paths.bat).")
        return 0
    if keep:
        print(f"\n  The model folder {folder} already holds files: the setup goes on there.")
        return 0
    if not drives:
        return 0
    new = choose(folder, drives)
    if os.path.normcase(os.path.abspath(new)) == os.path.normcase(os.path.abspath(folder)):
        print(f"  The model files go to {new}.")
        return 0
    try:
        os.makedirs(new, exist_ok=True)
    except OSError as e:
        print(f"  {new} cannot be made ({e}): the model files go to {folder}.")
        return 0
    if write_models_folder(new):
        print(f"  Saved: the model files go to {new} (DS41_MODELS in ds41-paths.bat).")
    else:
        print(f"  ds41-paths.bat was changed by hand: set DS41_MODELS={new} in it yourself; for now the files go to "
              f"{folder}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
