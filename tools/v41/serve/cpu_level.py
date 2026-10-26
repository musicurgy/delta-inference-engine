#!/usr/bin/env python3
r"""cpu_level.py - which engine build can run on this CPU.

    cpu_level.py                 prints this CPU's level: avx512, avx2, avx or sse
    cpu_level.py --pick-build    prints the engine build folder to use (exit 1 with an explanation if none fits)
    cpu_level.py --check         the same, silent when one fits
    cpu_level.py --release-zip   the prebuilt engine's file name for this CPU (build-ds41-portable.zip or -compat.zip)

The engine comes in three builds (build-ds41.bat): this PC's own (build-ds41\, built for the CPU it was built on),
the portable one (build-ds41-portable\, any CPU with AVX2: Intel 2013+ Haswell, every AMD Ryzen) and the one for older
CPUs (build-ds41-compat\, any x86-64 CPU with SSE4.2: Intel 2008+, AMD 2011+; slower).  Each records the level it needs
in cpu-level.txt; a build without one is taken as this PC's own (older builds) or as its folder's floor.  An AVX2 build
on a CPU without AVX2 stops at its first AVX2 instruction (Windows: "illegal instruction", exit code 0xC000001D)."""
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RANK = {"sse": 0, "avx": 1, "avx2": 2, "avx512": 3}
# folder -> the level it needs when it has no cpu-level.txt
BUILDS = [("build-ds41", None), ("build-ds41-portable", "avx2"), ("build-ds41-compat", "sse")]


def level():
    """this CPU's level from the OS (no AVX instruction is run to find out); DS41_CPU_LEVEL overrides it (tests)"""
    forced = os.environ.get("DS41_CPU_LEVEL", "")
    if forced in RANK:
        return forced
    if os.name == "nt":
        k = ctypes.windll.kernel32
        # PF_AVX512F_INSTRUCTIONS_AVAILABLE 41, PF_AVX2_INSTRUCTIONS_AVAILABLE 40, PF_AVX_INSTRUCTIONS_AVAILABLE 39: true
        # only when the OS also saves the wider registers.  The engine's AVX2 code also uses FMA and F16C, which every
        # AVX2 CPU has.
        if k.IsProcessorFeaturePresent(41):
            return "avx512"
        if k.IsProcessorFeaturePresent(40):
            return "avx2"
        return "avx" if k.IsProcessorFeaturePresent(39) else "sse"
    flags = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        return "avx2"   # unknown: assume the common case
    if "avx512f" in flags:
        return "avx512"
    if {"avx2", "fma", "f16c"} <= flags:
        return "avx2"
    return "avx" if "avx" in flags else "sse"


def needs(folder, floor):
    """the level a build folder needs, or None if it has no engine"""
    d = os.path.join(ROOT, folder)
    if not os.path.exists(os.path.join(d, "ds41_generate.exe")) and not os.path.exists(os.path.join(d, "ds41_generate")):
        return None
    try:
        with open(os.path.join(d, "cpu-level.txt"), encoding="utf-8") as f:
            v = f.read().strip()
        if v in RANK:
            return v
    except OSError:
        pass
    return floor or "sse"


def pick_build():
    """(folder, None) of the first build this CPU can run, or (None, why)"""
    cpu = level()
    found = []
    for folder, floor in BUILDS:
        n = needs(folder, floor)
        if n is None:
            continue
        found.append(f"{folder} (needs {n})")
        if RANK[cpu] >= RANK[n]:
            return folder, None
    if not found:
        return None, "no engine build here yet"
    return None, (f"this CPU supports {cpu.upper()} only, and the engine builds here need more: {', '.join(found)}.\n"
                  f"Use the build for older CPUs: put the build-ds41-compat folder from the Releases page here, or run\n"
                  f"build-ds41.bat (it builds for this CPU).")


def main():
    a = sys.argv[1:]
    if a in (["--pick-build"], ["--check"]):
        folder, why = pick_build()
        if folder is None:
            print(why)
            return 1
        if a == ["--pick-build"]:
            print(folder)
        return 0
    if a == ["--release-zip"]:
        print("build-ds41-portable.zip" if RANK[level()] >= RANK["avx2"] else "build-ds41-compat.zip")
        return 0
    print(level())
    return 0


if __name__ == "__main__":
    sys.exit(main())
