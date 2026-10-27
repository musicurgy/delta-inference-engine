#!/usr/bin/env python3
"""ds41_local.py - turn a serve config into this machine's config (the Windows launchers and bench scripts).

    ds41_local.py IN.json OUT.json     writes OUT with the settings below filled in, prints what it chose

* ${XO_ROOT} (this repository) and ${DS41_MODELS} (the folder with the model files, default: models/ in this repository;
  set it in ds41-paths.bat) are expanded in every string, as is any other %VAR% / ${VAR} the environment has.
* ${DS41_GGUF} / ${DS41_ENGRAM}: the main model and where the engine reads its Engram tables - both the ds4 Q2 GGUF
  (it carries them), or in an install from before 2026-10-27 the slim copy and DeepSeek's shards in engram/.  Set
  DS41_GGUF, or pick a model with Choose Model.bat (model-choice.txt), to run a GSQ or Unsloth model: an Unsloth GGUF
  reads its own Engram tables, a GSQ build the ds4 Q2 GGUF's.
* "--arena-gib" "auto": the pinned/locked RAM arena that holds the routed experts the GPU cache cannot.  It is sized
  from the installed RAM: total - DS41_RAM_RESERVE_GIB (default 38 GiB: the engine's own ~24 GiB outside the arena -
  dense half, KV, prefill buffers, Engram hot set - plus ~14 GiB for Windows and other programs), capped at 140 GiB
  (15,102 of the 15,360 experts; more buys nothing).  192 GB -> 140 GiB, 128 GB -> ~89 GiB: the other experts are
  read from the NVMe drive on demand (O_DIRECT), so less RAM costs speed, not correctness.  DS41_ARENA_GIB=N overrides.
* "model_name" / "display_name": named after the model file the engine loads (deepseek-v4.1-flash-1017-ds4-q2 /
  -gsq-iq2_xs / -ud-iq1_s ...; "DeepSeek V4.1 Flash · Unsloth UD-IQ1_S"), so /v1/models and the chat page say which
  one runs.  A model_name of your own in the config is kept.  The server answers any model name a client sends.
* "vision" (pictures, off by default): DS41_VISION, else vision-choice.txt: "cpu", "gpu" or "off" (default).
  Set up (setup's pictures step: DeepSeek-V4.1-Flash-1017-Vision.gguf in the model folder, PyTorch in .venv; DeepSeek's
  image code ships in tools/v41/vision/deepseek), the server runs tools/v41/vision/ds41_vision.py beside the engine and the
  engine gets --vision.  "cpu" takes no VRAM; "gpu" encodes on the engine's card, its VRAM taken from the expert cache.
  On but not set up: one line saying what is missing, and the model starts text only.
"""
import ctypes
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_level  # noqa: E402  (same folder)

ROOT =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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


def env_gib(name, default):
    """A GiB setting from the environment: a finite number >= 0, else a clear message (not a traceback)."""
    v = os.environ.get(name, "")
    if not v:
        return default
    try:
        f = float(v)
    except ValueError:
        f = float("nan")
    if not (f >= 0 and f != float("inf")):
        sys.exit(f"ds41_local: {name}={v!r} is not a number of GiB (ds41-paths.bat)")
    return f


def arena_gib():
    fixed = env_gib("DS41_ARENA_GIB", None)
    if fixed is not None:
        if fixed < 1:
            sys.exit(f"ds41_local: DS41_ARENA_GIB={fixed:g} - the expert arena needs at least 1 GiB")
        return fixed
    reserve = env_gib("DS41_RAM_RESERVE_GIB", 38.0)
    return max(16.0, min(ARENA_CAP_GIB, int(total_ram_gib() - reserve)))


def expand(cfg):
    """The config with this machine's paths and arena size (a new dict; IN is not changed)."""
    os.environ.setdefault("XO_ROOT", ROOT.replace("\\", "/"))
    os.environ.setdefault("DS41_MODELS", os.path.join(ROOT, "models").replace("\\", "/"))
    m = os.environ["DS41_MODELS"].replace("\\", "/").rstrip("/")
    full = m + "/DeepSeek-V4.1-Flash-1017-Q2.gguf"
    old = not os.path.exists(full) and os.path.exists(m + "/ds4-q2-slim.gguf") and \
        os.path.exists(m + "/engram/model-00048-of-00048.safetensors")
    # model-choice.txt (Choose Model.bat, pick_model.py): another model to load - a file in the model folder or a full
    # path.  DS41_GGUF in the environment wins over it.
    if "DS41_GGUF" not in os.environ:
        try:
            with open(os.path.join(ROOT, "model-choice.txt"), encoding="utf-8") as f:
                pick = f.readline().strip().strip('"')
        except OSError:
            pick = ""
        if pick:
            p = (pick if os.path.isabs(pick) else m + "/" + pick).replace("\\", "/")
            if os.path.exists(p):
                os.environ["DS41_GGUF"] = p
            else:
                print(f"ds41_local: model-choice.txt names {pick}, which is not in {m} - loading the default")
    os.environ.setdefault("DS41_GGUF", m + "/ds4-q2-slim.gguf" if old else full)
    if "DS41_ENGRAM" not in os.environ:
        # Engram: a split GGUF (Unsloth's, llama.cpp mainline's) carries its own tables - read them (IQ4_NL up to
        # UD-IQ3_S: measured as good as DeepSeek's exact FP8 tables, 89.3 vs 89.0% top-1 on UD-IQ1_S, and no second file
        # on another drive).  A GSQ build has none: the ds4 Q2 GGUF's (or an old install's shards).
        g = os.environ["DS41_GGUF"]
        own = "-00001-of-" in os.path.basename(g)
        os.environ["DS41_ENGRAM"] = g if own else m + "/engram" if old else full
    # ${DS41_BUILD}: the first build this CPU can run - this PC's own (build-ds41.bat), the portable one (AVX2), the one
    # for older CPUs (cpu_level.py)
    if "DS41_BUILD" not in os.environ:
        folder, why = cpu_level.pick_build()
        if folder is None:
            sys.exit(f"ds41_local: {why}")
        os.environ["DS41_BUILD"] = folder

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
    # DS41_GPU (ds41-paths.bat: the card with the most VRAM unless set): the server starts the engine on the config's
    # "gpu" (it sets CUDA_VISIBLE_DEVICES from it, nvidia-smi's numbering), not on the launcher's environment
    g = os.environ.get("DS41_GPU", "").strip()
    if g:
        if not g.isdigit():
            sys.exit(f"ds41_local: DS41_GPU={g!r} is not a GPU number (nvidia-smi's index; ds41-paths.bat)")
        out["gpu"] = int(g)
    # DS41_VRAM_MARGIN=GiB (Start DeepSeek.bat sets it for small cards): the VRAM kept free beside the expert cache
    margin = env_gib("DS41_VRAM_MARGIN", None)
    if margin is not None:
        if "--vram-margin" in a[:-1]:
            a[a.index("--vram-margin") + 1] = f"{margin:g}"
        else:
            a += ["--vram-margin", f"{margin:g}"]
    vision = vision_section(m)
    if vision:
        out["vision"] = vision
        if "--vision" not in a:   # here, not by the server: it would add --vram-reserve-mib, which this engine refuses
            a.append("--vision")
    return out


VISION_FILE = "DeepSeek-V4.1-Flash-1017-Vision.gguf"   # nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF
VISION_DEFAULT = "off"   # off by default (the user, 0.92); "cpu" takes no VRAM, decode unchanged (round 190)


def vision_choice():
    v = os.environ.get("DS41_VISION", "").strip().lower()
    if not v:
        try:
            with open(os.path.join(ROOT, "vision-choice.txt"), encoding="utf-8") as f:
                v = f.readline().strip().lower()
        except OSError:
            v = ""
    return v or VISION_DEFAULT


def torch_build():
    """'cuda', 'cpu' or None (no PyTorch in this Python) - read from torch/version.py, without importing torch (~2 s)."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        with open(os.path.join(list(spec.submodule_search_locations)[0], "version.py"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    return "cpu" if re.search(r"^cuda\s*(?::[^=]*)?=\s*None", text, re.M) else "cuda"


def vision_section(m):
    """The server's "vision" section for the pictures choice, or None (off, or not set up: said in one line)."""
    mode = vision_choice()
    if mode in ("off", "0", "no", "none", "false"):
        return None
    if mode not in ("cpu", "gpu"):
        print(f"ds41_local: pictures {mode!r} (DS41_VISION / vision-choice.txt): use cpu, gpu or off - text only")
        return None
    side, ref = m + "/" + VISION_FILE, ROOT.replace("\\", "/") + "/tools/v41/vision/deepseek"
    missing = [os.path.basename(p) for p in (side, ref + "/vision.py") if not os.path.exists(p)]
    build = torch_build()
    if build is None:
        missing.append("PyTorch")
    elif mode == "gpu" and build != "cuda":
        missing.append("PyTorch with CUDA (this one is the CPU build: choose cpu, or install the CUDA build)")
    if missing:
        print(f"ds41_local: pictures are on ({mode}) but not set up - missing {', '.join(missing)}: text only.  "
              f"setup-ds41-models.bat pictures sets them up")
        return None
    v = {"exe": ROOT.replace("\\", "/") + "/tools/v41/vision/ds41_vision.py", "mmproj": side,
         "model": os.environ["DS41_GGUF"], "reference_dir": ref, "gpu": mode == "gpu"}
    if mode == "gpu":   # DS41_VISION_DTYPE=f32: the CPU's exact output, ~1.1 GB more VRAM
        v["dtype"] = os.environ.get("DS41_VISION_DTYPE", "").strip().lower() or "bf16"
    else:   # i5-13600K: 6 threads 8.1 s for a 1024x701 photo, 14 threads 9.2 s (the E-cores slow it down)
        v["threads"] = max(1, min(6, os.cpu_count() or 1))
    return v


DEFAULT_NAME = "deepseek-v4.1-flash-1017-ds4-q2"   # the shipped config's model_name


def model_names(path):
    """(model id, display name) for the model file the engine loads"""
    b = os.path.basename(path)
    low = b.lower()
    if low in ("deepseek-v4.1-flash-1017-q2.gguf", "ds4-q2-slim.gguf"):
        tag, label = "ds4-q2", "ds4 Q2"
    elif low.startswith("ds41-gsq-") and low.endswith(".gguf"):          # a GSQ build: ds41-gsq-iq2_xs.gguf
        tag = "gsq-" + low[len("ds41-gsq-"):-len(".gguf")]
        label = "GSQ " + tag[4:].upper()
    elif re.search(r"UD-[A-Z0-9_]+", b):                                  # Unsloth: ...-UD-IQ1_S-00001-of-00005.gguf
        q = re.search(r"UD-[A-Z0-9_]+", b).group(0)
        tag, label = q.lower(), "Unsloth " + q
    else:                                                                 # any other GGUF: its own name
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", b[:-5] if low.endswith(".gguf") else b)
        tag, label = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-"), stem
    return f"deepseek-v4.1-flash-1017-{tag}", f"DeepSeek V4.1 Flash · {label}"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        with open(sys.argv[1], encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f"ds41_local: cannot read {sys.argv[1]}: {e}")
    cfg = expand(raw)
    args = cfg.get("args", [])
    if "-m" in args[:-1]:
        mid, shown = model_names(args[args.index("-m") + 1])
        if cfg.get("model_name", DEFAULT_NAME) == DEFAULT_NAME:
            cfg["model_name"] = mid
        cfg.setdefault("display_name", shown)
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    a = cfg.get("args", [])
    gib = a[a.index("--arena-gib") + 1] if "--arena-gib" in a else "?"
    # every file / folder the engine and server open at start (a missing one failed later, with a vaguer message)
    paths = [cfg.get("exe"), cfg.get("tokenizer")] + [a[i + 1] for i, x in enumerate(a[:-1]) if x in ("-m", "--rank")]
    paths += [d for i, x in enumerate(a[:-1]) if x == "--engram" for d in a[i + 1].split(";")]   # a list of folders
    missing = list(dict.fromkeys(p for p in paths if p and not os.path.exists(p)))   # (the model may be its Engram too)
    model = a[a.index("-m") + 1] if "-m" in a else "?"
    print(f"ds41_local: {total_ram_gib():.0f} GiB RAM -> expert arena {gib} GiB; model {os.path.basename(model)} "
          f"(Choose Model.bat picks another); models in {os.environ['DS41_MODELS']}")
    if cfg.get("vision"):
        print(f"ds41_local: pictures on ({'on the GPU' if cfg['vision']['gpu'] else 'on the CPU'})")
    for p in missing:
        print(f"ds41_local: MISSING {p}  (see README.md, 'Setup')")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
