#!/usr/bin/env python3
"""ds41_local.py - turn a serve config into this machine's config (the Windows launchers and bench scripts).

    ds41_local.py IN.json OUT.json     writes OUT with the settings below filled in, prints what it chose

* ${XO_ROOT} (this repository) and ${DS41_MODELS} (the folder with the model files, default: models/ in this repository;
  set it in ds41-paths.bat) are expanded in every string, as is any other %VAR% / ${VAR} the environment has.
* "--arena-gib" "auto": the pinned/locked RAM arena that holds the routed experts the GPU cache cannot.  It is sized
  from the installed RAM: total - DS41_RAM_RESERVE_GIB (default 38 GiB: the engine's own ~24 GiB outside the arena -
  dense half, KV, prefill buffers, Engram hot set - plus ~14 GiB for Windows and other programs), capped at 140 GiB
  (15,102 of the 15,360 experts; more buys nothing).  192 GB -> 140 GiB, 128 GB -> ~89 GiB: the other experts are
  read from the NVMe drive on demand (O_DIRECT), so less RAM costs speed, not correctness.  DS41_ARENA_GIB=N overrides.
"""
import ctypes
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARENA_CAP_GIB = 140


def total_ram_gib():
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                        ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]
        m = MS()
        m.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def arena_gib():
    if os.environ.get("DS41_ARENA_GIB"):
        return float(os.environ["DS41_ARENA_GIB"])
    reserve = float(os.environ.get("DS41_RAM_RESERVE_GIB", "38"))
    return max(16.0, min(ARENA_CAP_GIB, int(total_ram_gib() - reserve)))


def expand(cfg):
    """The config with this machine's paths and arena size (a new dict; IN is not changed)."""
    os.environ.setdefault("XO_ROOT", ROOT.replace("\\", "/"))
    os.environ.setdefault("DS41_MODELS", os.path.join(ROOT, "models").replace("\\", "/"))
    # ${DS41_BUILD}: this PC's own build (build-ds41.bat) if there is one, else the portable one (build-ds41.bat portable)
    native = os.path.exists(os.path.join(ROOT, "build-ds41", "ds41_generate.exe"))
    os.environ.setdefault("DS41_BUILD", "build-ds41" if native else "build-ds41-portable")

    def walk(v):
        if isinstance(v, str):
            return os.path.expandvars(v).replace("\\", "/") if ("$" in v or "%" in v) else v
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    out = walk(cfg)
    if out.get("log"):   # the server opens it for appending; a fresh clone has no logs/ folder
        os.makedirs(os.path.dirname(os.path.abspath(out["log"])), exist_ok=True)
    a = out.get("args", [])
    for i, x in enumerate(a[:-1]):
        if x == "--arena-gib" and a[i + 1] == "auto":
            a[i + 1] = f"{arena_gib():g}"
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cfg = expand(json.load(open(sys.argv[1], encoding="utf-8")))
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
    json.dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
    a = cfg.get("args", [])
    gib = a[a.index("--arena-gib") + 1] if "--arena-gib" in a else "?"
    missing = [p for p in [cfg.get("exe")] + [a[i + 1] for i, x in enumerate(a[:-1]) if x in ("-m", "--rank")]
               if p and not os.path.exists(p)]
    print(f"ds41_local: {total_ram_gib():.0f} GiB RAM -> expert arena {gib} GiB; models in {os.environ['DS41_MODELS']}")
    for p in missing:
        print(f"ds41_local: MISSING {p}  (see README.md, 'Setup')")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
