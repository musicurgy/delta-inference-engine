#!/usr/bin/env python3
r"""windows_tips.py - Windows settings that make the engine faster, checked at every start (read only: it changes
nothing).  Prints one line per setting that is off, pointing at README.md, "Windows settings for speed".  Measured on the
RTX 3090 PC: the first three together +12% writing speed (21.65 -> 24.25 tok/s), the RAM speed +3%.

    windows_tips.py        prints the tips (exit 0 always)       DS41_NO_TIPS=1: prints nothing
The same tips, each with how to fix it, and a page-file check go to local/windows-health.json: the chat page's Setup tab
shows them as its "This PC's health" card (setup_panel.py's catalog)."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# how to fix each tip (README.md, "Windows settings for speed"), by the start of its text
HOW = [
    ("power plan", "Win+R, powercfg.cpl, choose High performance (under \"Show additional plans\" if hidden). "
                   "Part of the +12% the first three settings gave together."),
    ("hardware-accelerated", "Settings > System > Display > Graphics settings: turn on hardware-accelerated GPU "
                             "scheduling, then restart Windows."),
    ("no 'Lock pages", "Win+R, secpol.msc > Local Policies > User Rights Assignment > Lock pages in memory > add your "
                       "user name; then sign out and in. (Windows Home has no secpol.msc.)"),
    ("the RAM runs", "In the BIOS turn on XMP / EXPO (or set the rated speed by hand), then check stability: +3% "
                     "measured."),
]
HEALTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", "windows-health.json")


def page_file():
    """the page-file tip, or None (docs/DETAILS.md: under ~4 GB the graphics card's allocations can fail)"""
    try:
        import drive_check
        mem = drive_check.memory()
    except Exception:                      # noqa: BLE001
        return None
    if not mem or mem[1] - mem[0] >= 4 * 2**30:
        return None
    return {"text": f"the page file is {(mem[1] - mem[0]) / 2**30:.0f} GB: Windows can refuse memory with the model loaded",
            "how": "Settings > System > About > Advanced system settings > Performance: Settings > Advanced > Virtual "
                   "memory: Change > Custom size, 65536 MB initial and maximum; then restart Windows."}


def health(t=None):
    """[{"text", "how"}] for the page: the tips with their fixes, and the page file"""
    out = []
    for x in (tips() if t is None else t):
        how = next((h for k, h in HOW if x.startswith(k)), "README.md, \"Windows settings for speed\".")
        out.append({"text": x, "how": how})
    pf = page_file()
    if pf:
        out.append(pf)
    return out


def save_health(items):
    try:
        os.makedirs(os.path.dirname(HEALTH), exist_ok=True)
        with open(HEALTH, "w", encoding="utf-8") as f:
            json.dump({"checked": int(time.time()), "items": items}, f, indent=1)
    except OSError:
        pass


def main():
    if os.name != "nt" or os.environ.get("DS41_NO_TIPS") == "1":
        return 0
    t = tips()
    save_health(health(t))
    if t:
        print("Faster with these Windows settings (README.md, 'Windows settings for speed'; DS41_NO_TIPS=1 hides this):")
        for x in t:
            print("  - " + x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
