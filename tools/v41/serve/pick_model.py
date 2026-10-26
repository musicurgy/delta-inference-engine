#!/usr/bin/env python3
"""pick_model.py - choose the model "Start DeepSeek.bat" loads ("Choose Model.bat" runs it).

Lists the models in the model folder (DS41_MODELS, ds41-paths.bat) - the default ds4 Q2 GGUF and any GSQ-RCO builds -
with their size, speed and agreement with DeepSeek's API (README.md), and the GSQ builds not made yet, which it can make
(tools/v41/gsq_stream_merge.py: ISTA-DASLab's experts streamed from Hugging Face, every shard SHA-256-checked).  The
choice goes into model-choice.txt in this folder (one line: a file name in the model folder, or a full path); the
launcher reads it (ds41_local.py).  Delete that file, or pick the default, to go back to the ds4 Q2 GGUF.

Every model reads its Engram tables from the ds4 Q2 GGUF, so that file stays in any case.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHOICE = os.path.join(ROOT, "model-choice.txt")
Q2 = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
SLIM = "ds4-q2-slim.gguf"   # an install from before 2026-10-27: the same model without its inline Engram tables
GB = 1e9
# file -> (name, GB without Engram, writing tok/s at 192 / 128 GB RAM, top-1 agreement with DeepSeek's API, GSQ build)
KNOWN = {
    Q2: ("ds4 Q2 (the default)", 163, "41 / 33", "91.5%", None),
    SLIM: ("ds4 Q2 (the default)", 163, "41 / 33", "91.5%", None),
    "ds41-gsq-iq2_xs.gguf": ("GSQ IQ2_XS", 159, "38 / 26", "90.0%", "IQ2_XS"),
    "ds41-gsq-iq2_xxs.gguf": ("GSQ IQ2_XXS", 142, "41 / 32", "87.8%", "IQ2_XXS"),
    "ds41-gsq-iq1_m.gguf": ("GSQ IQ1_M", 125, "42 / 37.5", "85.5%", "IQ1_M"),
}


def models_dir():
    return os.environ.get("DS41_MODELS", os.path.join(ROOT, "models")).replace("/", os.sep).rstrip(os.sep)


def current():
    """the file model-choice.txt names, if it exists - else the default"""
    try:
        with open(CHOICE, encoding="utf-8") as f:
            name = f.readline().strip().strip('"')
    except OSError:
        return None
    return name or None


def main():
    m = models_dir()
    if not os.path.isdir(m):
        sys.exit(f"No model folder at {m} yet - run Start DeepSeek.bat first: it downloads the model.")
    base = Q2 if os.path.exists(os.path.join(m, Q2)) else SLIM if os.path.exists(os.path.join(m, SLIM)) else None
    files = sorted(f for f in os.listdir(m) if f.lower().endswith(".gguf") and not f.upper().startswith("V41-1017-MTP")
                   and os.path.getsize(os.path.join(m, f)) > 20 * GB)
    order = list(KNOWN)
    files.sort(key=lambda f: (order.index(f) if f in KNOWN else len(order), f))   # the default first, then by size
    cur = current()
    cur_file = os.path.basename(cur) if cur else base
    print(f"Model folder: {m}\n")
    print("     model                       size*    writing tok/s     agreement with   file")
    print("                                          (192 / 128 GB)    DeepSeek's API")
    menu = []
    for f in files:
        name, gb, tps, agr, _ = KNOWN.get(f, ("other GGUF (untested here)", os.path.getsize(os.path.join(m, f)) / GB, "?", "?", None))
        mark = "  <- now" if f == cur_file else ""
        menu.append(("use", f))
        print(f"  {len(menu)}. {name:<27} {gb:>4.0f} GB   {tps:<17} {agr:<16} {f}{mark}")
    if cur and not os.path.exists(cur if os.path.isabs(cur) else os.path.join(m, cur)):
        print(f"\n  (model-choice.txt names {cur}, which is not there: the launcher uses the default)")
    todo = [(f, k) for f, k in KNOWN.items() if k[4] and f not in files]
    if todo and base:
        print("\n  Not made yet (built from ISTA-DASLab's GSQ-RCO files, downloaded once; 20-30 min at 100 MB/s):")
        for f, k in todo:
            menu.append(("build", f))
            print(f"  {len(menu)}. build {k[0]:<21} {k[1]:>4} GB   {k[2]:<17} {k[3]}")
    print("\n  * without the Engram tables, which every model reads from the ds4 Q2 GGUF (keep that file).")
    print("  On 192 GB of RAM the default writes as fast as any and agrees best; on 128 GB IQ1_M writes fastest.\n")
    if not menu:
        sys.exit("No model in the folder yet - run Start DeepSeek.bat first: it downloads the default one.")
    pick = input(f"Number to use (1-{len(menu)}), Enter = keep the current one: ").strip()
    if not pick:
        print("Nothing changed.")
        return 0
    if not pick.isdigit() or not 1 <= int(pick) <= len(menu):
        sys.exit(f"'{pick}' is not on the list - nothing changed.")
    what, f = menu[int(pick) - 1]
    if what == "build":
        name, gb, _, _, build = KNOWN[f]
        free = shutil.disk_usage(m).free
        if free < (gb + 5) * GB:
            sys.exit(f"Not enough space for {name}: it needs ~{gb + 5} GB free in {m}, there are {free / GB:.0f} GB.")
        print(f"Building {f} (~{gb} GB) - it can be stopped and started again from the beginning.")
        py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        r = subprocess.call([py if os.path.exists(py) else sys.executable,
                             os.path.join(ROOT, "tools", "v41", "gsq_stream_merge.py"),
                             "--ours", os.path.join(m, base), "--build", build, "--out", os.path.join(m, f)])
        if r != 0 or not os.path.exists(os.path.join(m, f)):
            sys.exit("The build stopped (above) - nothing changed.")
    with open(CHOICE, "w", encoding="utf-8") as fo:
        fo.write(f + "\n")
    print(f"Saved: Start DeepSeek.bat now loads {f} (in model-choice.txt).  Restart the server if it is running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
