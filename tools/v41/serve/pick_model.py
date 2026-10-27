#!/usr/bin/env python3
"""pick_model.py - choose the model "Start DeepSeek.bat" loads ("Choose Model.bat" runs it).

Lists the models in the model folder (DS41_MODELS, ds41-paths.bat) and the ones you can add:
  * the default: nazeshinjite's ds4 Q2 GGUF (DeepSeek's exact Engram tables inline);
  * ISTA-DASLab's GSQ-RCO quants (IQ2_XS / IQ2_XXS / IQ1_M), built here from their files by
    tools/v41/gsq_stream_merge.py - they read their Engram from the ds4 Q2 GGUF, so that file stays;
  * Unsloth's quants (UD-IQ1_S ... UD-Q8_K_XL), downloaded into unsloth/<quant>/ as they are (every file SHA-256
    checked).  Their files carry their own Engram tables (IQ4_NL up to UD-IQ3_S - measured as good as DeepSeek's exact
    FP8 tables on UD-IQ1_S -, exact from UD-Q3_K_XL), which the launcher reads (ds41_local.py).
The choice goes into model-choice.txt in this folder (one line: a path in the model folder, or a full path); the launcher
reads it.  Delete that file, or pick the default, to go back to the ds4 Q2 GGUF.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHOICE = os.path.join(ROOT, "model-choice.txt")
Q2 = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
SLIM = "ds4-q2-slim.gguf"   # an install from before 2026-10-27: the same model without its inline Engram tables
UNSLOTH = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
GB = 1e9
# (menu name, GB on disk, writing tok/s at 192 / 128 GB RAM, top-1 agreement with DeepSeek's API, kind, source)
DEFAULT = ("ds4 Q2 (the default)", 366, "41 / 33", "91.5%")
GSQ = {   # file -> name, GB without Engram, speed, agreement, ISTA build name
    "ds41-gsq-iq2_xs.gguf": ("GSQ IQ2_XS", 159, "38 / 26", "90.0%", "IQ2_XS"),
    "ds41-gsq-iq2_xxs.gguf": ("GSQ IQ2_XXS", 142, "41 / 32", "87.8%", "IQ2_XXS"),
    "ds41-gsq-iq1_m.gguf": ("GSQ IQ1_M", 125, "42 / 37.5", "85.5%", "IQ1_M"),
}
UNSLOTH_Q = {   # quant -> GB to download (its Engram included), speed, agreement, its Engram tables
    "UD-IQ1_S": (233, "40 / 37.5", "89.3%", "IQ4_NL"),   # measured 2026-10-26 (rounds 167, 170)
    "UD-IQ1_M": (250, "-", "-", "IQ4_NL"),
    "UD-IQ2_XXS": (301, "-", "-", "IQ4_NL"),
    "UD-Q2_K_XL": (335, "-", "-", "IQ4_NL"),
    "UD-IQ3_XXS": (395, "-", "-", "IQ4_NL"),
    "UD-IQ3_S": (424, "-", "-", "IQ4_NL"),
    "UD-Q3_K_XL": (485, "-", "-", "exact"),
    "UD-Q4_K_XL": (501, "-", "-", "exact"),
    "UD-Q8_K_XL": (508, "-", "-", "exact"),
}


def models_dir():
    return os.environ.get("DS41_MODELS", os.path.join(ROOT, "models")).replace("/", os.sep).rstrip(os.sep)


def current():
    """the path model-choice.txt names (relative to the model folder, '/' separators), or None"""
    try:
        with open(CHOICE, encoding="utf-8") as f:
            name = f.readline().strip().strip('"')
    except OSError:
        return None
    return name.replace("\\", "/") or None


def unsloth_first(m, q):
    """the first shard of an Unsloth quant in the model folder (relative path), or None"""
    d = os.path.join(m, "unsloth", q)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.endswith(".gguf") and "-00001-of-" in f and os.path.exists(os.path.join(d, f + ".ok")):
            return f"unsloth/{q}/{f}"
    return None


def download_unsloth(m, q):
    """hf download unsloth/... --include q/* into <models>/unsloth, then every shard's SHA-256 (hf_verify.py)"""
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(UNSLOTH) if f.startswith(q + "/") and f.endswith(".gguf")]
    if not files:
        sys.exit(f"{UNSLOTH} has no {q} files")
    hf = os.path.join(ROOT, ".venv", "Scripts", "hf.exe")
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    dst = os.path.join(m, "unsloth")
    os.makedirs(dst, exist_ok=True)
    r = subprocess.call([hf if os.path.exists(hf) else "hf", "download", UNSLOTH, "--include", q + "/*", "--local-dir", dst])
    if r != 0:
        sys.exit("The download stopped (above) - start Choose Model.bat again to go on where it was.")
    r = subprocess.call([py if os.path.exists(py) else sys.executable, os.path.join(ROOT, "tools", "v41", "hf_verify.py"),
                         "--repo", UNSLOTH, "--dir", dst] + files)
    if r != 0:
        sys.exit("A file differs from its published SHA-256 (above): delete it and start Choose Model.bat again.")


def main():
    m = models_dir()
    if not os.path.isdir(m):
        sys.exit(f"No model folder at {m} yet - run Start DeepSeek.bat first: it sets up the default model.")
    base = Q2 if os.path.exists(os.path.join(m, Q2)) else SLIM if os.path.exists(os.path.join(m, SLIM)) else None
    cur = current()
    now = cur if cur and os.path.exists(cur if os.path.isabs(cur) else os.path.join(m, cur)) else base
    print(f"Model folder: {m}\n")
    print("     model                          disk     writing tok/s    agreement with")
    print("                                             (192 / 128 GB)   DeepSeek's API")
    menu = []

    def row(kind, path, name, gb, tps, agr, note=""):
        menu.append((kind, path))
        mark = "  <- now" if kind == "use" and path == now else ""
        print(f"  {len(menu):>2}. {name:<28} {gb:>4} GB   {tps:<16} {agr:<8}{note}{mark}")

    if base:
        row("use", base, *DEFAULT)
    for f, (name, gb, tps, agr, _) in GSQ.items():
        if os.path.exists(os.path.join(m, f)):
            row("use", f, name, gb, tps, agr, "  (+ the default's Engram)")
    for q, (gb, tps, agr, eng) in UNSLOTH_Q.items():
        p = unsloth_first(m, q)
        if p:
            row("use", p, "Unsloth " + q, gb, tps, agr)
    free = shutil.disk_usage(m).free

    def room(gb):   # what a build / download says when it would not fit (picking it is refused below)
        return "" if free >= (gb + 5) * GB else f"  - too big for this drive ({free / GB:.0f} GB free)"

    gsq_todo = [(f, k) for f, k in GSQ.items() if not os.path.exists(os.path.join(m, f))]
    if gsq_todo and base:
        print("\n  Build (ISTA-DASLab's GSQ-RCO experts into a copy of the default; 20-30 min at 100 MB/s):")
        for f, (name, gb, tps, agr, _) in gsq_todo:
            row("build", f, "build " + name, gb, tps, agr, room(gb))
    uns_todo = [q for q in UNSLOTH_Q if not unsloth_first(m, q)]
    if uns_todo:
        print("\n  Download (Unsloth's GGUFs as they are, SHA-256 checked; 100 MB/s = ~40-90 min):")
        for q in uns_todo:
            gb, tps, agr, eng = UNSLOTH_Q[q]
            row("download", q, "download Unsloth " + q, gb, tps, agr, f"  ({eng} Engram){room(gb)}")
    print("\n  Agreement: teacher-forced top-1 with DeepSeek's own model, general text, exact routing.  On 192 GB of RAM")
    print("  the default writes as fast as any and agrees best; on 128 GB a smaller file writes faster (more of it fits).")
    print("  A model bigger than your RAM reads the rest from the SSD on every token: slower.\n")
    if not menu:
        sys.exit("No model yet - run Start DeepSeek.bat first: it downloads the default one.")
    pick = input(f"Number to use (1-{len(menu)}), Enter = keep the current one: ").strip()
    if not pick:
        print("Nothing changed.")
        return 0
    if not pick.isdigit() or not 1 <= int(pick) <= len(menu):
        sys.exit(f"'{pick}' is not on the list - nothing changed.")
    kind, f = menu[int(pick) - 1]
    if kind == "build":
        name, gb, _, _, build = GSQ[f]
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
    elif kind == "download":
        gb = UNSLOTH_Q[f][0]
        free = shutil.disk_usage(m).free
        if free < (gb + 5) * GB:
            sys.exit(f"Not enough space for Unsloth {f}: it needs ~{gb + 5} GB free in {m}, there are {free / GB:.0f} GB.")
        print(f"Downloading Unsloth {f} (~{gb} GB) into {os.path.join(m, 'unsloth', f)} - if it stops, start again: it resumes.")
        download_unsloth(m, f)
        f = unsloth_first(m, f)
        if not f:
            sys.exit("The download finished but its first file is not there - nothing changed.")
    with open(CHOICE, "w", encoding="utf-8") as fo:
        fo.write(f + "\n")
    print(f"Saved: Start DeepSeek.bat now loads {f} (in model-choice.txt).  Restart the server if it is running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
