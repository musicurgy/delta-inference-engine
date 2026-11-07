#!/usr/bin/env python3
"""pick_model.py - choose the model "Start DeepSeek.bat" loads ("Choose Model.bat" runs it).

Lists the models in the model folder (DS41_MODELS, ds41-paths.bat) and the ones you can add:
  * the default: nazeshinjite's ds4 Q2 GGUF (DeepSeek's exact Engram tables inline);
  * ISTA-DASLab's GSQ-RCO quants (IQ2_XS / IQ2_XXS / IQ1_M), built here from their files by
    tools/v41/gsq_stream_merge.py - they read their Engram from the ds4 Q2 GGUF, so that file stays;
  * Unsloth's quants (UD-IQ1_S ... UD-Q8_K_XL), downloaded into unsloth/<quant>/ as they are (every file SHA-256
    checked).  Their files carry their own Engram tables (IQ4_NL / Q5_K - UD-IQ3_XXS's Q6_K ones are not read, it uses
    the default's -, measured as good as DeepSeek's exact
    FP8 tables on UD-IQ1_S -, exact from UD-Q3_K_XL), which the launcher reads (ds41_local.py).
  * the quality mode: bertholomus' EXL3 2.5 bpw routed experts (exllamav3's trellis format) on the default's dense layers
    and Engram, downloaded as they are into exl3/<repo name>/ (every shard SHA-256 checked); the launcher gives the
    engine --exl3 <that folder>.  It agrees more with DeepSeek's model than the default and writes slower.
The choice goes into model-choice.txt in this folder (one line: a path in the model folder, or a full path); the launcher
reads it.  Delete that file, or pick the default, to go back to the ds4 Q2 GGUF.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # no usage reports to Hugging Face (downloads only)
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHOICE = os.path.join(ROOT, "model-choice.txt")
Q2 = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
SLIM = "ds4-q2-slim.gguf"   # an install from before 2026-10-27: the same model without its inline Engram tables
UNSLOTH = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
# the revision our numbers were measured on (2026-10-20): Unsloth re-made UD-IQ1_S / UD-IQ1_M / UD-Q2_K_XL on 10-28 with
# the same sizes and new content - pinned until the new files are measured
UNSLOTH_REV = "60ad3651"
# rounds 323 / 325: the re-made UD-IQ1_S is closer to DeepSeek than the pinned one (89.1 / 91.5% vs 87.5 / 90.2% at bias 0.28), same size
# 0.94: the re-made UD-Q2_K_XL too (round 309: a tie with the pinned one - general 95.08 vs 95.25%, code + maths 96.80 vs
# 96.60%, KL lower) - new downloads get Unsloth's latest; an existing install is not re-downloaded
UNSLOTH_REV_OF = {"UD-IQ1_S": "8a6ec9cf", "UD-Q2_K_XL": "8a6ec9cf"}
GB = 1e9
# Writing tok/s through the server at 192 / 128 / 96 / 64 GB of RAM (the 96 / 64 GB columns: 2026-10-28/29, a PC with
# that much RAM simulated by locking the rest away, so neither Windows' file cache nor the pagefile could use it; "-" =
# not measured).  The model to recommend on a PC with less than LOW_RAM_GB of RAM:
LOW_RAM_GB = 110
LOW_RAM_PICK = "UD-IQ1_S"
# ... and on one with HIGH_RAM_GB or more (0.93's final cards: UD-IQ2_XXS writes about as fast as the default - 41.1 vs
# 40.7 tok/s greedy, 39.7 vs 40.3 at the shipped sampling - agrees more, 91.1 / 94.5% vs 89.9 / 93.1% at the shipped bias, and is 65 GB smaller; at 128 GB it writes 5% slower,
# 34.0 vs 35.9 tok/s, and the answers are worth that: 57 vs 52 of 99 on the benchmark)
HIGH_RAM_GB = 120
HIGH_RAM_PICK = "UD-IQ2_XXS"
# (menu name, GB on disk, writing tok/s, top-1 agreement with DeepSeek's API on general text: at the shipped route bias 0.28 for
# the default and Unsloth (rounds 314-325), exact routing for GSQ and EXL3, which were not measured at 0.28)
DEFAULT = ("ds4 Q2 (the default)", 366, "41 / 36 / 27 / 20", "89.9%")   # 96 / 64: rounds 234 (reserve 28 + read-ahead 8)
GSQ = {   # file -> name, GB without Engram, speed, agreement, ISTA build name
    "ds41-gsq-iq2_xs.gguf": ("GSQ IQ2_XS", 159, "38 / 26 / - / -", "90.0%", "IQ2_XS"),
    "ds41-gsq-iq2_xxs.gguf": ("GSQ IQ2_XXS", 142, "41 / 32 / - / -", "87.8%", "IQ2_XXS"),
    "ds41-gsq-iq1_m.gguf": ("GSQ IQ1_M", 125, "42 / 37.5 / 34 / 28", "85.5%", "IQ1_M"),
}
UNSLOTH_Q = {   # quant -> GB to download (its Engram included), speed, agreement, its Engram tables
    "UD-IQ1_S": (233, "41 / 41 / 36 / 31", "89.1%", "IQ4_NL"),   # 41.0 / 40.6 / 36.5 / 30.6 tok/s (round 325); 89.1 / 91.5 / 95.3%, rev 8a6ec9cf
    "UD-IQ1_M": (250, "-", "-", "IQ4_NL"),
    "UD-IQ2_XXS": (301, "41 / 34 / 29 / 21", "91.1%", "Q5_K"),    # 41.1 / 34.0 / 29.3 / 21.0 tok/s (round 317); 91.1 / 94.5 / 95.9% at bias 0.28
    "UD-Q2_K_XL": (335, "31 / 27 / 23 / 15", "92.5%", "Q5_K"),    # 31.3 / 26.6 / 22.9 / 15.2 tok/s; 92.5 / 95.3 / 97.0% - the quality pick
    "UD-IQ3_XXS": (395, "28 / 23 / 18 / 12", "92.7%", "the default's"),   # 28.2 / 22.8 / 18.3 / 11.7 tok/s; 92.7 / 95.9 / 97.4%; its Q6_K Engram tables are not read (the default's)
    "UD-IQ3_S": (424, "-", "-", "IQ4_NL"),
    "UD-Q3_K_XL": (485, "-", "-", "exact"),
    "UD-Q4_K_XL": (501, "-", "-", "exact"),
    "UD-Q8_K_XL": (508, "-", "-", "exact"),
}
TAG = {"UD-Q2_K_XL": " (quality)", "UD-IQ3_XXS": " (most exact)", "UD-IQ2_XXS": " (128 GB and up)"}   # the picks the menu text below explains
EXL3 = {   # folder in exl3/ -> name, GB to download, speed, agreement, Hugging Face repo
    "DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw": ("EXL3 2.5 bpw", 185, "25 / - / - / -", "94.4%",
                                             "bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw"),   # round 238
}
EXL3_MIN_RAM_GB = 180   # its experts are 159 GiB: below ~192 GB of RAM most of them would come from the SSD (not measured)


def ram_gb():
    """this PC's RAM in GB (DS41_RAM_GB overrides: tests), or None"""
    v = os.environ.get("DS41_RAM_GB", "")
    if v.isdigit():
        return int(v)
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                    ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]
    m = MS()
    m.len = ctypes.sizeof(MS)
    return round(m.total / 2**30) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else None


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


def exl3_ready(m, d):
    """an EXL3 folder in exl3/ whose every shard has its .ok (hf_verify.py), relative path, or None"""
    p = os.path.join(m, "exl3", d)
    if not os.path.isdir(p):
        return None
    shards = [f for f in os.listdir(p) if f.endswith(".safetensors")]
    if not shards or not os.path.exists(os.path.join(p, "config.json")):
        return None
    if not all(os.path.exists(os.path.join(p, f + ".ok")) for f in shards):
        return None
    return f"exl3/{d}"


def _download(hf, args, name):
    """hf download ARGS through hf_get.py: the PC stays awake, one progress line, a stalled or failed download is
    started again (it resumes).  -> its exit code"""
    import hf_get
    return hf_get.run(args, hf_cmd=hf, label=name)


def _done_note(t0, name):
    """a Windows notification when a download took long (someone may have left it running)"""
    if time.time() - t0 > 120:
        import hf_get
        hf_get.notify(f"DeepSeek: {name} is ready", "Downloaded and checked. Start DeepSeek to use it.")


def download_exl3(m, d):
    """hf download REPO --local-dir <models>/exl3/<d>, then every shard's SHA-256 (hf_verify.py)"""
    from huggingface_hub import HfApi
    name, repo = EXL3[d][0], EXL3[d][4]
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".safetensors")]
    hf = os.path.join(ROOT, ".venv", "Scripts", "hf.exe")
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    dst = os.path.join(m, "exl3", d)
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    r = _download([hf if os.path.exists(hf) else "hf"], [repo, "--local-dir", dst], name)
    if r != 0:
        sys.exit("The download stopped (above) - start Choose Model.bat again to go on where it was.")
    r = subprocess.call([py if os.path.exists(py) else sys.executable, os.path.join(ROOT, "tools", "v41", "hf_verify.py"),
                         "--repo", repo, "--dir", dst] + files)
    if r != 0:
        sys.exit("A file differs from its published SHA-256 (above): delete it and start Choose Model.bat again.")
    _done_note(t0, name)


def download_unsloth(m, q):
    """hf download unsloth/... --include q/* into <models>/unsloth, then every shard's SHA-256 (hf_verify.py)"""
    from huggingface_hub import HfApi
    rev = UNSLOTH_REV_OF.get(q, UNSLOTH_REV)
    files = [f for f in HfApi().list_repo_files(UNSLOTH, revision=rev) if f.startswith(q + "/") and f.endswith(".gguf")]
    if not files:
        sys.exit(f"{UNSLOTH} has no {q} files")
    hf = os.path.join(ROOT, ".venv", "Scripts", "hf.exe")
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    dst = os.path.join(m, "unsloth")
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    r = _download([hf if os.path.exists(hf) else "hf"], [UNSLOTH, "--revision", rev, "--include", q + "/*",
                                                         "--local-dir", dst], "Unsloth " + q)
    if r != 0:
        sys.exit("The download stopped (above) - start Choose Model.bat again to go on where it was.")
    r = subprocess.call([py if os.path.exists(py) else sys.executable, os.path.join(ROOT, "tools", "v41", "hf_verify.py"),
                         "--repo", UNSLOTH, "--revision", rev, "--dir", dst] + files)
    if r != 0:
        sys.exit("A file differs from its published SHA-256 (above): delete it and start Choose Model.bat again.")
    _done_note(t0, "Unsloth " + q)


def get(q):
    """--get QUANT (first-start's low-RAM offer): download an Unsloth quant, SHA-256 check it, make it the choice"""
    if q not in UNSLOTH_Q:
        sys.exit(f"pick_model.py --get: {q} is not one of {', '.join(UNSLOTH_Q)}")
    m = models_dir()
    os.makedirs(m, exist_ok=True)
    f = unsloth_first(m, q)
    if not f:
        gb = UNSLOTH_Q[q][0]
        free = shutil.disk_usage(m).free
        if free < (gb + 20) * GB:   # + the drafter, which the first start sets up next (~16 GB at its peak)
            sys.exit(f"Not enough space for Unsloth {q}: it needs ~{gb + 20} GB free in {m}, there are {free / GB:.0f} GB.")
        print(f"Downloading Unsloth {q} (~{gb} GB) into {os.path.join(m, 'unsloth', q)} - if it stops, start again: it resumes.")
        download_unsloth(m, q)
        f = unsloth_first(m, q)
        if not f:
            sys.exit("The download finished but its first file is not there.")
    with open(CHOICE, "w", encoding="utf-8") as fo:
        fo.write(f + "\n")
    print(f"Saved: Start DeepSeek.bat loads {f} (model-choice.txt).")
    return 0


def custom_help(m):
    print(f"""
  Your own model: make the folder {os.path.join(m, "custom")} and copy it there, then start again.
    - a GGUF of DeepSeek-V4.1-Flash-1017 with all 384 experts: one .gguf file, or a split model's files together
      (a split model carries its own Engram tables; one .gguf file needs the default model installed as well)
    - or a folder of EXL3 experts (with its config.json), which also needs the default model
  Start DeepSeek.bat then offers it here, and the Setup card on the chat page lists it.  A file this engine cannot
  run is refused with the reason.
""")


def first():
    """--first: no model yet.  Exit 0 = a model is chosen and there, 10 = the default (setup-ds41-models.bat gets it),
    11 = none: the user brings their own."""
    import setup_panel as SP
    m = models_dir()
    os.makedirs(m, exist_ok=True)
    ram = ram_gb()
    rec = SP.recommended(ram)
    picks = [("default", DEFAULT[0], DEFAULT[1], DEFAULT[2], DEFAULT[3], "")]
    for q in ("UD-IQ1_S", "UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS"):
        gb, tps, agr, eng = UNSLOTH_Q[q]
        # a model that reads the default's Engram tables needs the default as well: the disk column counts both
        picks.append((q, "Unsloth " + q + TAG.get(q, ""), gb + (DEFAULT[1] if eng == "the default's" else 0), tps, agr, ""))
    print("\n  Which model should this PC use?" + (f"  ({ram} GB of RAM)" if ram else ""))
    print("  Speeds are from an RTX 3090.  You can change this at any time with Choose Model.bat or the chat page.\n")
    print("       model                                     disk     writing tok/s at             agreement with")
    print("                                                          192 / 128 / 96 / 64 GB RAM   DeepSeek's API")
    best = 1
    for i, (key, name, gb, tps, agr, _) in enumerate(picks, 1):
        is_rec = key == rec
        best = i if is_rec else best
        print(f"   {i}. {name:<39} {gb:>4} GB   {tps:<28} {agr:<8}{'  <- recommended for this PC' if is_rec else ''}")
    print(f"   {len(picks) + 1}. None: I will use my own model\n")
    while True:
        try:
            a = input(f"  Number (1-{len(picks) + 1}), Enter = {best}: ").strip() or str(best)
        except EOFError:
            a = str(best)
        if a.isdigit() and 1 <= int(a) <= len(picks) + 1:
            break
        print("  Please type one of the numbers.")
    if int(a) == len(picks) + 1:
        for c in SP.custom_models(m):
            if c["ok"] and not c["needs_default"]:
                with open(CHOICE, "w", encoding="utf-8") as f:
                    f.write(c["id"] + "\n")
                print(f"  Using your own model: {c['name']}")
                return 0
        custom_help(m)
        return 11
    key = picks[int(a) - 1][0]
    if key == "default":
        return 10
    rc = get(key)
    # exit 10 keeps first-start's default download going: this model reads the default's Engram tables
    return 10 if rc == 0 and UNSLOTH_Q[key][3] == "the default's" else rc


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--get":
        return get(sys.argv[2])
    if sys.argv[1:] == ["--first"]:
        return first()
    m = models_dir()
    if not os.path.isdir(m):
        sys.exit(f"No model folder at {m} yet - run Start DeepSeek.bat first: it sets up the default model.")
    base = Q2 if os.path.exists(os.path.join(m, Q2)) else SLIM if os.path.exists(os.path.join(m, SLIM)) else None
    cur = current()
    now = cur if cur and os.path.exists(cur if os.path.isabs(cur) else os.path.join(m, cur)) else base
    ram = ram_gb()
    low = ram is not None and ram < LOW_RAM_GB
    high = ram is not None and ram >= HIGH_RAM_GB
    print(f"Model folder: {m}" + (f"   (this PC: {ram} GB of RAM)" if ram else "") + "\n")
    print("     model                                     disk     writing tok/s at             agreement with")
    print("                                                        192 / 128 / 96 / 64 GB RAM   DeepSeek's API")
    menu = []

    def row(kind, path, name, gb, tps, agr, note=""):
        menu.append((kind, path))
        mark = "  <- now" if kind == "use" and path == now else ""
        rec = "  <- recommended for this PC" if (low and LOW_RAM_PICK in name) or (high and HIGH_RAM_PICK in name) else ""
        print(f"  {len(menu):>2}. {name:<39} {gb:>4} GB   {tps:<28} {agr:<8}{note}{mark}{rec}")

    if base:
        row("use", base, *DEFAULT)
    for f, (name, gb, tps, agr, _) in GSQ.items():
        if os.path.exists(os.path.join(m, f)):
            row("use", f, name, gb, tps, agr, "  (+ the default's Engram)")
    for q, (gb, tps, agr, eng) in UNSLOTH_Q.items():
        p = unsloth_first(m, q)
        if p:
            row("use", p, "Unsloth " + q + TAG.get(q, ""), gb, tps, agr)
    for d, (name, gb, tps, agr, _) in EXL3.items():
        p = exl3_ready(m, d)
        if p and base:
            row("use", p, name, gb, tps, agr, "  (+ the default)")
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
            row("download", q, "download Unsloth " + q + TAG.get(q, ""), gb, tps, agr, f"  ({eng} Engram){room(gb)}")
    x3_todo = [d for d in EXL3 if not exl3_ready(m, d)]
    if x3_todo and base:
        print("\n  Download EXL3 2.5 bpw (exllamav3's format on the default's dense layers; 100 MB/s = ~30 min; 192 GB of RAM")
        print("  and an internal NVMe drive).  Unsloth UD-Q2_K_XL above agrees more AND writes faster - prefer it:")
        for d in x3_todo:
            name, gb, tps, agr, _ = EXL3[d]
            small = ram is not None and ram < EXL3_MIN_RAM_GB
            row("download", d, "download " + name.split(" (")[0], gb, tps, agr,
                ("  - needs ~192 GB of RAM" if small else "") + room(gb))
    print("\n  Agreement: teacher-forced top-1 with DeepSeek's own model, general text, as shipped; exact routing for GSQ and")
    print("  EXL3 (code + maths in the README).  From 120 GB of RAM: Unsloth UD-IQ2_XXS is the pick - on 192 GB it writes about")
    print("  as fast as the default (41.1 vs 40.7 tok/s greedy, 39.7 vs 40.3 at the shipped sampling), on 128 GB 5% slower (34")
    print("  vs 36), and it agrees more (91.1%, and 94.5% on code + maths against the default's 89.9 / 93.1%), from 65 GB less")
    print("  disk; Unsloth UD-Q2_K_XL is the")
    print("  quality pick (31 tok/s, 92.5% / 95.3%) - it agrees more than EXL3 2.5 bpw and writes ~25% faster; UD-IQ3_XXS is")
    print("  the most exact (28 tok/s, 92.7% / 95.9%), UD-Q2_K_XL the faster of the two.  With less")
    print("  RAM a smaller file writes faster (more of it fits; the rest is read from the SSD on every token): under")
    print("  110 GB, Unsloth UD-IQ1_S, 36 / 31 tok/s at 96 / 64 GB against the default's 27 / 20, about one point lower")
    print("  agreement.  (Measured on an RTX 3090.)\n")
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
    elif kind == "download" and f in EXL3:
        name, gb = EXL3[f][0], EXL3[f][1]
        free = shutil.disk_usage(m).free
        if free < (gb + 5) * GB:
            sys.exit(f"Not enough space for {name}: it needs ~{gb + 5} GB free in {m}, there are {free / GB:.0f} GB.")
        print(f"Downloading {name} (~{gb} GB) into {os.path.join(m, 'exl3', f)} - if it stops, start again: it resumes.")
        download_exl3(m, f)
        f = exl3_ready(m, f)
        if not f:
            sys.exit("The download finished but a shard is not checked - nothing changed.")
    elif kind == "download":
        gb = UNSLOTH_Q[f][0]
        if UNSLOTH_Q[f][3] == "the default's" and base is None:
            sys.exit(f"Unsloth {f} reads its Engram tables from the default (ds4 Q2 GGUF), which is not in {m}: "
                     "download the default first - nothing changed.")
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
