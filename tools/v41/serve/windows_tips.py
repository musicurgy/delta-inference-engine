#!/usr/bin/env python3
r"""windows_tips.py - Windows settings that make the engine faster, checked at every start (read only: it changes
nothing).  Prints one line per setting that is off, pointing at README.md, "Windows settings for speed".  Measured on the
RTX 3090 PC: the first three together +12% writing speed (21.65 -> 24.25 tok/s), the RAM speed +3%.

    windows_tips.py        prints the tips (exit 0 always)       DS41_NO_TIPS=1: prints nothing"""
import os
import subprocess
import sys

# the default power plans that park cores and slow the clock ramp (High performance / Ultimate / a custom plan: fine)
SLOW_PLANS = {"381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced", "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver"}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def tips():
    out = []
    plan = run(["powercfg", "/getactivescheme"]).lower()
    for guid, name in SLOW_PLANS.items():
        if guid in plan:
            out.append(f"power plan is {name}: switch to High performance (powercfg.cpl)")
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers") as k:
            hags = winreg.QueryValueEx(k, "HwSchMode")[0]
    except OSError:
        hags = None
    # 2 = on, 1 = off; no value = never toggled: the default, on in Windows 11 (build 22000+), off in Windows 10
    win11 = sys.getwindowsversion().build >= 22000
    if hags == 1 or (hags is None and not win11):
        out.append("hardware-accelerated GPU scheduling is off: Settings > System > Display > Graphics settings")
    if "SeLockMemoryPrivilege" not in run(["whoami", "/priv"]):
        out.append("no 'Lock pages in memory' right (large pages for the expert RAM): secpol.msc, see README")
    ram = run(["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_PhysicalMemory | ForEach-Object { \"$($_.Speed) $($_.ConfiguredClockSpeed)\" }"])
    pairs = [tuple(int(v) for v in ln.split()) for ln in ram.splitlines() if len(ln.split()) == 2 and ln.replace(" ", "").isdigit()]
    if pairs and all(c and r and c < r for r, c in pairs):
        out.append(f"the RAM runs at {pairs[0][1]} MT/s, below its rated {pairs[0][0]}: raise it in the BIOS (XMP / EXPO "
                   f"or manually)")
    return out


def main():
    if os.name != "nt" or os.environ.get("DS41_NO_TIPS") == "1":
        return 0
    t = tips()
    if t:
        print("Faster with these Windows settings (README.md, 'Windows settings for speed'; DS41_NO_TIPS=1 hides this):")
        for x in t:
            print("  - " + x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
