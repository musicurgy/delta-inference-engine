#!/usr/bin/env python3
"""ds41_local.py - turn a serve config into this machine's config (the Windows launchers and bench scripts).

    ds41_local.py IN.json OUT.json     writes OUT with the settings below filled in, prints what it chose

* ${XO_ROOT} (this repository) and ${DS41_MODELS} (the folder with the model files, default: models/ in this repository;
  set it in ds41-paths.bat) are expanded in every string, as is any other %VAR% / ${VAR} the environment has.
* ${DS41_GGUF} / ${DS41_ENGRAM}: the main model and where the engine reads its Engram tables - both the ds4 Q2 GGUF
  (it carries them), or in an install from before 2026-10-27 the slim copy and DeepSeek's shards in engram/.  Set
  DS41_GGUF, or pick a model with Choose Model.bat (model-choice.txt), to run a GSQ or Unsloth model: an Unsloth GGUF
  reads its own Engram tables, a GSQ build the ds4 Q2 GGUF's.
* ${DS41_EXL3}: a folder of EXL3 routed experts (the quality mode, bertholomus' 2.5 bpw: Choose Model.bat downloads it
  into exl3/ and names the folder in model-choice.txt).  The engine then runs the default GGUF's dense layers and Engram
  with these experts (--exl3 FOLDER, added to the args here).
* "--arena-gib" "auto": the pinned/locked RAM arena that holds the routed experts the GPU cache cannot.  It is sized
  from the installed RAM: total - DS41_RAM_RESERVE_GIB (default 28 GiB: the engine's own ~12-14 GiB outside the
  arena - dense half, KV, prefill buffers, Engram hot set; measured - plus ~14 GiB for Windows and other programs;
  a 64 GB PC gets about a 35 GiB arena, a 96 GB PC about 67), capped at 140 GiB
  (15,102 of the 15,360 experts; more buys nothing).  192 GB -> 140 GiB, 128 GB -> ~99 GiB: the other experts are
  read from the NVMe drive on demand (O_DIRECT), so less RAM costs speed, not correctness.  DS41_ARENA_GIB=N overrides.
* "model_name" / "display_name": named after the model file the engine loads (deepseek-v4.1-flash-1017-ds4-q2 /
  -gsq-iq2_xs / -ud-iq1_s ...; "DeepSeek V4.1 Flash · Unsloth UD-IQ1_S"), so /v1/models and the chat page say which
  one runs.  A model_name of your own in the config is kept.  The server answers any model name a client sends.
* "vision" (pictures, on by default since 0.94): DS41_VISION, else vision-choice.txt:
    on (default)      the second card when the Setup card's multi-GPU switch is on and there is one, else the CPU
    cpu               the CPU (no VRAM)
    gpu:N             card N (nvidia-smi's number)
    main (old: gpu)   the engine's card: the fastest, but its ~1.5 GiB come out of the expert cache
    off               text only
  Set up (setup's pictures step: DeepSeek-V4.1-Flash-1017-Vision.gguf in the model folder, PyTorch in .venv - the CUDA
  build only when the choice is a card; DeepSeek's image code ships in tools/v41/vision/deepseek), the server runs
  tools/v41/vision/ds41_vision.py beside the engine and the engine gets --vision.  A card without PyTorch's CUDA build:
  the CPU, said in one line.  On the multi-GPU helper's card the helper holds VISION_GPU_GIB less.  On but not set up:
  one line saying what is missing, and the model starts text only.
  "ds41_local.py --vision-setup" prints what the launchers need: the plan (off / cpu / cuda) and the PyTorch there is.
"""
import ctypes
import importlib.util
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_level  # noqa: E402  (same folder)
import ckpt  # noqa: E402  (tools/v41/ckpt.py: the checkpoint table, data/ds41/checkpoints.json)

ROOT =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARENA_CAP_GIB = 140
KEEP_ALL_RAM_GIB = 80                                             # below: the drafter keeps fewer experts (96,64,32)
# the checkpoint's row in data/ds41/checkpoints.json (DS41_CHECKPOINT picks another than the default): its file names
try:
    CK = ckpt.row()
except ckpt.TableError as _e:
    sys.exit(f"ds41_local: {_e}")
_DE = CK.get("derived") or {}
Q2_FILE = ((CK.get("models") or {}).get("ds4_q2") or {}).get("file") or ""
SLIM_FILE = ((CK.get("models") or {}).get("ds4_q2") or {}).get("slim") or ""


def keep_values(de):
    """(small, big) STRATA_DSPARK_KEEP values for a checkpoint row's "derived" block: <keep file>:96,64,32 (below
    KEEP_ALL_RAM_GIB) and <keep file>:112,80,48 (Start DeepSeek.bat's value on cards of 22 GB+); (None, None) when the
    row has no keep file (keep null: the drafter keeps all its experts)."""
    k = (de or {}).get("keep")
    return (f"{k}:96,64,32", f"{k}:112,80,48") if k else (None, None)


DSPARK_KEEP_SMALL, DSPARK_KEEP_BIG = keep_values(_DE)
# engine settings for this model, carried in the config's "env" (the server starts the engine with them - this script's
# own environment does not reach it)
ENGINE_ENV = {}
BIG_ARENA_CAP_GIB, BIG_RAM_RESERVE_GIB = 158, 20.0   # files with bigger experts (below: measured)


def page_settings():
    """delta-settings.json: the chat page's Setup card (setup_panel.py)"""
    try:
        with open(os.path.join(ROOT, "delta-settings.json"), encoding="utf-8-sig") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def page_access(cfg, s):
    """the chat page's Setup tab: "Use from your phone" listens on every network (0.0.0.0) with an API key - the key
    already in the config, else the one the switch made (delta-settings.json, not in git); off: the config's own host
    (this PC only by default).  "auto_pause_mib" (0 = off) goes into the config as well."""
    if s.get("phone") and (cfg.get("api_key") or s.get("phone_key")):
        if not cfg.get("api_key"):
            cfg["api_key"] = s["phone_key"]
        if str(cfg.get("host") or "127.0.0.1") in ("127.0.0.1", "localhost", "::1"):
            cfg["host"] = "0.0.0.0"
        print(f"ds41_local: Setup tab: reachable from other devices on this network (port {os.environ.get('DS41_PORT', '8080')}, "
              "API key required)")
    if isinstance(s.get("auto_pause_mib"), int) and s["auto_pause_mib"] > 0:
        cfg["auto_pause_mib"] = s["auto_pause_mib"]


# Below LOW_RAM_GIB the 1-bit Unsloth files run at route bias 0.10 instead of the config's 0.28: fewer of their experts
# fit the RAM, and 0.28 then costs agreement (measured, real 64 / 96 GB, release engine: UD-IQ1_S 88.8 / 89.6% top-1 at
# 0.10 vs 86.6 / 87.9% at 0.28; UD-IQ1_M 89.9 / 90.0% vs 87.7 / 89.6%), while 0.10 still writes faster than UD-IQ2_XXS at
# 0.28 (UD-IQ1_S 39.9 / 44.9 vs 35.3 / 40.9 tok/s at the default sampling).  UD-IQ2_XXS and the others keep 0.28.
LOW_RAM_GIB = 110
LOW_RAM_BIAS = "0.10"


def low_ram_bias(gguf, ram_gib):
    """LOW_RAM_BIAS for an Unsloth UD-IQ1_S / UD-IQ1_M file on a PC under LOW_RAM_GIB, else "" (the config's bias)"""
    name = os.path.basename(str(gguf or "")).upper()
    # rounded like pick_model.ram_gb(), so the Setup card's default and the launcher agree right at the line
    if ram_gib is not None and round(ram_gib) < LOW_RAM_GIB and re.search(r"UD-IQ1_[SM]\b", name):
        return LOW_RAM_BIAS
    return ""


def route_bias_choice(settings, gguf, ram_gib):
    """(bias, why) from the Setup card's switch: Quality = 0.10, Most exact = 0, Fast = "" (the config's 0.28).  A switch
    the user never set (no "mode_chosen"; before 0.95 the page saved "fast" on every Apply) takes this PC's default:
    LOW_RAM_BIAS for a 1-bit file under LOW_RAM_GIB, else the config's.  bias "" = the config's."""
    mode = (settings or {}).get("mode")
    if mode == "balanced":
        return "0.10", "Setup card: quality (route bias 0.10)"
    if mode == "exact":
        return "0", "Setup card: most exact"
    if mode == "fast" and (settings or {}).get("mode_chosen"):
        return "", ""
    rb = low_ram_bias(gguf, ram_gib)
    return rb, (f"{ram_gib:.0f} GiB of RAM, a 1-bit model: route bias {rb} (the Setup card's Fast or DS41_ROUTE_BIAS "
                "override it)" if rb else "")


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


def drafter_keep(ram_gib, env, de=None):
    """STRATA_DSPARK_KEEP for the engine, or None (as the launcher left it): the smaller counts (DSPARK_KEEP_SMALL,
    96,64,32) below KEEP_ALL_RAM_GIB when the launcher left the variable unset or empty, or at its 22 GB+ value, and the
    drafter runs.  None too when the checkpoint row (`de`, default this checkpoint's "derived") has no keep file."""
    small, big = keep_values(de) if de is not None else (DSPARK_KEEP_SMALL, DSPARK_KEEP_BIG)
    if not small:
        return None
    if ram_gib < KEEP_ALL_RAM_GIB and env.get("STRATA_SERVE_DSPARK") and env.get("STRATA_DSPARK_KEEP", "").strip() in ("", big):
        return small
    return None


PROMPT_VARS = ("STRATA_WIDE_PROMPT", "STRATA_WIDE_PROMPT_MIN", "STRATA_BM_PREFILL_READAHEAD", "STRATA_ENGRAM_READAHEAD")
# (new, old): switches renamed after the 0.95 release candidates.  The old name still counts when the new one is unset
# (here: with_new_names; the engine: ds41_env.hpp), so saved settings keep their effect.
RENAMED = (("STRATA_WIDE_PROMPT", "STRATA_D11_WIDE_PROMPT"), ("STRATA_WIDE_PROMPT_MIN", "STRATA_D11_WIDE_MIN"),
           ("STRATA_PROMPT_SWEEP_PB", "STRATA_P3_SWEEP_PB"), ("STRATA_PROMPT_SWEEP_FIT", "STRATA_P3_SWEEP_FIT"))


def with_new_names(env):
    """`env` (a new dict) with each old name's value also under its new name where the new name is unset."""
    out = dict(env)
    for new, old in RENAMED:
        if new not in out and old in out:
            out[new] = out[old]
    return out


def prompt_profile(arena, experts, env):
    """(engine env, launcher line) for the two prompt-speed settings, chosen by whether the routed experts fit in the RAM
    arena (no file tier) or not (some are read from the SSD).  Measured at 192 GB (UD-IQ2_XXS, no file tier) and 64 GB
    (UD-IQ1_S, file tier), prompt tok/s vs 0.95-rc with both on:
      file tier:    4K +3%, 8K +6.5%, 32K +40%  -> wide prompt from 6144 rows + prefill read-ahead
      no file tier: 4K -3%, 8K -3.5%, 32K +18%; read-ahead alone 4K -3%, 32K +5%; wide alone 32K +20%
                    -> wide prompt from 16384 rows only, no read-ahead
    `arena` / `experts` in GiB (None: unknown - the file-tier profile, the launcher's earlier default).  A variable the
    user set (in `env`, under its new or old name) is kept: not in the returned env."""
    env = with_new_names(env)
    if arena is None or experts is None:
        tier, why = True, "the experts' size or the arena is unknown"
    elif arena < experts:
        tier, why = True, f"routed experts {experts:.1f} GiB > arena {arena:g} GiB: a file tier"
    else:
        tier, why = False, f"routed experts {experts:.1f} GiB fit the arena {arena:g} GiB: no file tier"
    # 0.95: + STRATA_ENGRAM_READAHEAD=1 on the file-tier profile (measured at a real 128 / 96 GB: prompts +1-2.5%, exact)
    want = {"STRATA_WIDE_PROMPT": "1", "STRATA_WIDE_PROMPT_MIN": "6144", "STRATA_BM_PREFILL_READAHEAD": "1",
            "STRATA_ENGRAM_READAHEAD": "1"} if tier \
        else {"STRATA_WIDE_PROMPT": "1", "STRATA_WIDE_PROMPT_MIN": "16384"}
    out = {k: v for k, v in want.items() if k not in env}
    own = [f"{k}={env[k]}" for k in PROMPT_VARS if k in env]
    name = "wide prompt from 6144 tokens + prefill read-ahead" if tier else "wide prompt from 16384 tokens, no read-ahead"
    line = f"ds41_local: prompt profile: {name} ({why})" + (f"; yours kept: {', '.join(own)}" if own else "")
    return out, line


def file_tier(arena, experts):
    """prompt_profile's choice: True = the file-tier profile (some experts read from the SSD, or the sizes unknown)."""
    return arena is None or experts is None or arena < experts


# STRATA_PROMPT_SWEEP_PB's default: the --prompt-batch the file-tier profile asks for when its batches are wide.  A
# 29.8K-token prompt at 16384 is two full sweeps of the file tier; at 32768 it is one where the VRAM allows - the engine
# takes only the width that fits (STRATA_PROMPT_SWEEP_FIT's counted budget, then a probe).  0 = the config's (16384):
# widths of 24-32K have not been checked for identical output against 16384 yet (only 16384 vs 8192 has, measured),
# so a wider value is experimental and may change the text written.
PROMPT_SWEEP_PB = 0


def sweep_fit_env(cur, pb, env):
    """The engine env a raised prompt batch brings: STRATA_PROMPT_SWEEP_FIT=1 (the counted wide budget, which keeps the
    prestage room) when `pb` is above the config's `cur`, unless the user set STRATA_PROMPT_SWEEP_FIT (or its old
    name); else {}."""
    env = with_new_names(env)
    if pb is None or cur is None or pb <= cur or "STRATA_PROMPT_SWEEP_FIT" in env:
        return {}
    out = {"STRATA_PROMPT_SWEEP_FIT": "1"}
    if "STRATA_PROMPT_SWEEP_SLACK_GIB" not in env:
        out["STRATA_PROMPT_SWEEP_SLACK_GIB"] = "0.3"   # the measured setting
    return out


def env_int(env, name):
    """A whole-number setting (>= 0) from `env`, None when unset or empty; a clear message (not a traceback) otherwise."""
    v = str(env.get(name, "")).strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        f = float("nan")
    if not (0 <= f < float("inf")) or f != int(f):
        sys.exit(f"ds41_local: {name}={v!r} is not a whole number of tokens (>= 0)")
    return int(f)


def prompt_batch(args, tier, wide, capped, env):
    """The --prompt-batch for this PC: (value or None to leave the config's, launcher line or None).

    * DS41_PROMPT_BATCH=N (or STRATA_PROMPT_BATCH, ds41-engine.sh's name for it, which Windows did not read before
      11-13): N, raised or lowered, over the config and every rule below - the sweep knob (16384 / 24576 / 32768).
      The engine still takes only what fits the free VRAM.  0 = the chunk path (no layer-major batch).
    * Else, on the file-tier profile with wide batches (`tier`, `wide`), and not for files whose bigger experts made
      expand() cap the batch (`capped`): STRATA_PROMPT_SWEEP_PB (default PROMPT_SWEEP_PB, 0 = off; wider is experimental)
      raises the config's value, never lowers it.
    A raised width also turns on the engine's counted budget (sweep_fit_env).
    `args` is the expanded engine args (read only); `env` the environment."""
    env = with_new_names(env)
    i = args.index("--prompt-batch") + 1 if "--prompt-batch" in args[:-1] else None
    cur = int(float(args[i])) if i is not None else None
    for name in ("DS41_PROMPT_BATCH", "STRATA_PROMPT_BATCH"):
        n = env_int(env, name)
        if n is not None:
            return n, f"ds41_local: prompt batch {n} ({name}; the config's {cur if cur is not None else 'none'})"
    if not (tier and wide) or capped or cur is None:
        return None, None
    n = env_int(env, "STRATA_PROMPT_SWEEP_PB")
    # 0.95: 32768 by default at <= 64 GB of RAM (measured at a real 64 GB: 32K prompts +13.4%, 40K +12.0%, writing flat;
    # 128 GB only +2-4% and 96 GB ~0, so the larger PCs keep the config's width)
    n = (32768 if total_ram_gib() <= 72 else PROMPT_SWEEP_PB) if n is None else n
    if n <= cur:
        return None, None
    return n, (f"ds41_local: prompt batch {n} (file tier with wide batches: a long prompt in fewer sweeps where the VRAM "
               f"allows; STRATA_PROMPT_SWEEP_PB=0 keeps the config's {cur})")


def big_expert_margins(gguf_name, env):
    """0.95: the VRAM margins for a file whose experts are bigger than the default's (MXFP4 / IQ3_S / Q3_K / 4-bit rows):
    (DS41_VRAM_MARGIN to set or None, the engine's --prompt-margin or None).  Their prompt path needs headroom the
    default's does not (UD-IQ3_XXS: a 4096-token batch at 2 GiB ran out of VRAM; 2048 at 3 GiB ran clean, measured) -
    but only while a prompt batch runs.  Kept free the whole time it cost ~170 expert slots in every decode (12.25 GiB
    instead of ~14.7).  So by default the headroom goes to the prompt path alone (--prompt-margin: its keep-cache borrow
    takes it from the cache and the regrow gives it back) and the slots keep the config's margin.  STRATA_BIG_BORROW=0
    restores the earlier rule (the margin for both); a DS41_VRAM_MARGIN of the user's wins as before (nothing set here)."""
    vm = "3" if "UD-IQ3_XXS" in os.path.basename(gguf_name) else "2"
    if "DS41_VRAM_MARGIN" in env:
        return None, None
    if (env.get("STRATA_BIG_BORROW") or "").strip() == "0":
        return vm, None
    return None, vm


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


def out_bias(args):
    """The --route-bias the expanded args carry (for the startup line)."""
    return args[args.index("--route-bias") + 1] if "--route-bias" in args else "-"


def arena_gib():
    fixed = env_gib("DS41_ARENA_GIB", None)
    if fixed is not None:
        if fixed < 1:
            sys.exit(f"ds41_local: DS41_ARENA_GIB={fixed:g} - the expert arena needs at least 1 GiB")
        return fixed
    reserve = env_gib("DS41_RAM_RESERVE_GIB", 28.0)
    return max(16.0, min(ARENA_CAP_GIB, int(total_ram_gib() - reserve)))


def expand(cfg):
    """The config with this machine's paths and arena size (a new dict; IN is not changed)."""
    os.environ.setdefault("XO_ROOT", ROOT.replace("\\", "/"))
    os.environ.setdefault("DS41_MODELS", os.path.join(ROOT, "models").replace("\\", "/"))
    m = os.environ["DS41_MODELS"].replace("\\", "/").rstrip("/")
    # the checkpoint's ds4 Q2 GGUF ("" when the checkpoint has none: a model must then be picked)
    full = m + "/" + Q2_FILE if Q2_FILE else ""
    old = not os.path.exists(full) and bool(SLIM_FILE) and os.path.exists(m + "/" + SLIM_FILE) and \
        os.path.exists(m + "/" + _DE.get("engram_dir", "engram") + "/model-00048-of-00048.safetensors")
    # the config's ${DS41_RANK}: the checkpoint's own rank file ("" when it has none)
    os.environ.setdefault("DS41_RANK", (os.environ["XO_ROOT"] + "/" + _DE["rank"]) if _DE.get("rank") else "")
    # model-choice.txt (Choose Model.bat, pick_model.py): another model to load - a file in the model folder or a full
    # path.  DS41_GGUF in the environment wins over it.
    if "DS41_GGUF" not in os.environ:
        try:
            with open(os.path.join(ROOT, "model-choice.txt"), encoding="utf-8-sig") as f:
                pick = f.readline().strip().strip('"')
        except OSError:
            pick = ""
        if pick:
            p = (pick if os.path.isabs(pick) else m + "/" + pick).replace("\\", "/")
            if os.path.isdir(p):   # an EXL3 folder: its experts on the default GGUF
                if os.path.exists(p + "/config.json"):
                    os.environ.setdefault("DS41_EXL3", p)
                else:
                    print(f"ds41_local: model-choice.txt names {pick}, a folder without config.json - loading the default")
            elif os.path.exists(p):
                os.environ["DS41_GGUF"] = p
            else:
                print(f"ds41_local: model-choice.txt names {pick}, which is not in {m} - loading the default")
    if "DS41_GGUF" not in os.environ and not old and not full:
        sys.exit(f"ds41_local: checkpoint {CK['id']} has no default model file - pick one with Choose Model.bat")
    os.environ.setdefault("DS41_GGUF", m + "/" + SLIM_FILE if old else full)
    if "DS41_ENGRAM" not in os.environ:
        # Engram: a split GGUF (Unsloth's, llama.cpp mainline's) carries its own tables - read them (IQ4_NL up to
        # UD-IQ3_S: measured as good as DeepSeek's exact FP8 tables, 89.3 vs 89.0% top-1 on UD-IQ1_S, and no second file
        # on another drive).  A GSQ build has none: the ds4 Q2 GGUF's (or an old install's shards).
        g = os.environ["DS41_GGUF"]
        own = "-00001-of-" in os.path.basename(g)
        t = None
        if own:
            # the engine's Engram row reader handles Q5_K (13), IQ4_NL (20) and MXFP8 (43) tables - Unsloth's Dynamic
            # V3.1 UD-IQ2_XXS / UD-Q2_K_XL (rev 60ad3651) store theirs as Q5_K: their own tables, measured as good as the
            # default's exact FP8 ones (93.14 vs 92.94% top-1, the same speed).  Any other type takes the default's FP8
            # tables - the default model must be installed then
            try:
                import gguf_head
                tt = gguf_head.tensor_types(g)   # the tables: per_layer_token_embd (Xolotl's name) or blk.N.engram_kv (llama.cpp's)
                t = tt.get("per_layer_token_embd.weight")
                if t is None:
                    t = next((v for k, v in tt.items() if k.endswith(".engram_kv.weight")), None)
            except Exception as e:   # unreadable header: let the engine decide
                print(f"ds41_local: could not read {os.path.basename(g)}'s header ({e}); trying its own Engram tables")
                t = 20
            if t == 43:   # id 43 is also mainline ggml's F8_E4M3 (one byte a value): the engine reads only MXFP8 under it
                try:
                    if gguf_head.type43_layout(g, "per_layer_token_embd.weight") == "f8_e4m3":
                        t = "F8_E4M3 (type 43, one byte a value)"
                except Exception:   # unreadable: as before, the engine decides
                    pass
            if t is not None and t not in (13, 20, 43):
                alt = m + "/" + _DE.get("engram_dir", "engram") if old else full
                if os.path.exists(alt):
                    print(f"ds41_local: {os.path.basename(g)} stores its Engram tables as "
                          f"{gguf_head.GGML_TYPE_NAMES.get(t, t)} - using the default model's exact FP8 tables ({alt})")
                    own = False
                else:
                    sys.exit(f"ds41_local: {os.path.basename(g)} stores its Engram tables as "
                             f"{gguf_head.GGML_TYPE_NAMES.get(t, t)}, which this build cannot read, and the default model "
                             f"(its Engram tables) is not in {m}.  Install the default with Start DeepSeek.bat first.")
        os.environ["DS41_ENGRAM"] = g if own else m + "/" + _DE.get("engram_dir", "engram") if old else full
        if own and t != 43:   # the Engram hot set holds FP8 rows: a file with other tables would only log "SKIPPED"
            ENGINE_ENV["STRATA_ENGRAM_HOT"] = "0"   # "0" = no hot set (ds41_engram.cpp)
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
    # the checkpoint's tokenizer export, when it has its own folder (the default's is the config's tokenizer/)
    tok = _DE.get("tokenizer_dir") or "tokenizer"
    if tok != "tokenizer" and out.get("tokenizer") == m + "/tokenizer":
        out["tokenizer"] = m + "/" + tok
    if out.get("log"):   # the server opens it for appending; a fresh clone has no logs/ folder
        os.makedirs(os.path.dirname(os.path.abspath(out["log"])), exist_ok=True)
    a = out.get("args", [])
    if "--rank" in a[:-1] and not a[a.index("--rank") + 1].strip():   # a checkpoint without a rank file yet
        i = a.index("--rank")
        del a[i:i + 2]
        print(f"ds41_local: checkpoint {CK['id']} has no rank file yet (data/ds41/checkpoints.json): the engine starts "
              "without --rank")
    for i, x in enumerate(a[:-1]):
        if x == "--arena-gib" and a[i + 1] == "auto":
            a[i + 1] = f"{arena_gib():g}"
    # the route bias: the config's 0.28 on every PC; DS41_ROUTE_BIAS overrides it (0 = exact routing, 0.10 = the
    # high-agreement setting: 91.8 / 92.8% next-token agreement with DeepSeek's API vs 90.2 / 92.4% at 0.28, for about
    # 18% of writing speed at 192 GB and 12-26% on 96 / 64 GB PCs - measured 2026-10-30 / 31)
    rb = os.environ.get("DS41_ROUTE_BIAS", "").strip()
    if not rb:      # the chat page's Setup card (Fast / Quality / Most exact), else this PC's default
        rb, why = route_bias_choice(page_settings(), os.environ.get("DS41_GGUF", ""), total_ram_gib())
        if why:
            print(f"ds41_local: {why}")
    for i, x in enumerate(a[:-1]):
        if x == "--route-bias":
            if rb:
                try:
                    float(rb)
                except ValueError:
                    sys.exit(f"ds41_local: DS41_ROUTE_BIAS={rb!r} is not a number (ds41-paths.bat)")
                a[i + 1] = rb
                # 0 means exact routing: the prompt path's own bias (--route-bias-file) goes to 0 as well
                if float(rb) == 0 and "--route-bias-file" in a[:-1]:
                    a[a.index("--route-bias-file") + 1] = "0"
                    print("ds41_local: DS41_ROUTE_BIAS=0 - exact routing (--route-bias 0, --route-bias-file 0)")
    x3 = os.environ.get("DS41_EXL3", "").strip().replace("\\", "/")
    if x3 and "--exl3" not in a:
        a += ["--exl3", x3]
    # Bigger experts than the default's (MXFP4 / IQ3_S / Q3_K / 4-bit rows: Unsloth UD-Q2_K_XL and up): the batch
    # prompt path sizes its buffers and the kept cache for the default's ~10 MB experts and ran out of VRAM on
    # UD-Q2_K_XL (measured: a 7K-token batch, then the cache regrow / c_bounce out of memory) - cap the prompt batch
    # at 4096 tokens and keep 2 GiB of VRAM free unless the user set DS41_VRAM_MARGIN
    big_experts = False   # (prompt_batch: the file-tier sweep width does not raise such a file's cap)
    try:
        import gguf_head
        big = {39, 21, 11, 12, 20, 23}   # MXFP4, IQ3_S, Q3_K, Q4_K, IQ4_NL, IQ4_XS
        ex = {v for k, v in gguf_head.tensor_types(os.environ["DS41_GGUF"]).items() if "_exps" in k or "experts" in k}
        if ex & big:
            big_experts = True
            names = sorted(gguf_head.GGML_TYPE_NAMES.get(t, str(t)) for t in ex & big)
            # UD-IQ3_XXS (the largest menu file): prompt batch 2048 and 3 GiB free - its first card at 4096 / 2 ran out
            # of VRAM in the prompt path (measured); 2048 / 3 ran clean at 27.8 tok/s (measured)
            pb_cap = 2048 if "UD-IQ3_XXS" in os.path.basename(os.environ["DS41_GGUF"]) else 4096
            if "--prompt-batch" in a[:-1] and int(float(a[a.index("--prompt-batch") + 1])) > pb_cap:
                a[a.index("--prompt-batch") + 1] = str(pb_cap)
            # 0.95: the headroom goes to the prompt path only (big_expert_margins; STRATA_BIG_BORROW=0 = the earlier rule)
            vm, pm = big_expert_margins(os.environ["DS41_GGUF"], os.environ)
            if vm is not None:
                os.environ["DS41_VRAM_MARGIN"] = vm
            if pm is not None:
                if "--prompt-margin" in a[:-1]:
                    a[a.index("--prompt-margin") + 1] = pm
                else:
                    a += ["--prompt-margin", pm]
            # their experts outgrow the 140 GiB arena (UD-Q2_K_XL: ~177 GiB): on 192 GB of RAM the arena goes to
            # 158 GiB with 20 GiB kept back instead of 28 (measured: half the file tier, +2% +- 1.9%; ran clean on an
            # idle PC - other programs get ~8 GB).  Only where it is larger than the usual arena; DS41_ARENA_GIB /
            # DS41_RAM_RESERVE_GIB still win.  192 GB-class PCs only: below that the usual 28 GiB reserve stays.
            if total_ram_gib() >= 170 and "DS41_ARENA_GIB" not in os.environ and "DS41_RAM_RESERVE_GIB" not in os.environ and "--arena-gib" in a[:-1]:
                i = a.index("--arena-gib")
                bigger = min(BIG_ARENA_CAP_GIB, int(total_ram_gib() - BIG_RAM_RESERVE_GIB))
                if bigger > float(a[i + 1]):
                    a[i + 1] = f"{bigger:g}"
                    names.append(f"arena {bigger:g} GiB")
            print(f"ds41_local: {os.path.basename(os.environ['DS41_GGUF'])} has "
                  f"{', '.join(n for n in names if not n.startswith('arena'))} experts (bigger than the "
                  f"default's): prompt batch {pb_cap}, " +
                  (f"{pm} GiB of VRAM kept free for prompt batches (borrowed from the expert cache)" if pm is not None
                   else f"{os.environ.get('DS41_VRAM_MARGIN')} GiB of VRAM kept free") +
                  "".join(f", {n}" for n in names if n.startswith("arena")))
        else:
            # experts a little over the usual arena (Unsloth UD-IQ2_XXS: 146 GiB vs 140): on 192 GB the arena covers
            # them all, up to 150 GiB with 20 GiB kept back (measured: +2.5%, the file tier gone)
            # only split files (Unsloth's): ds4 Q2's 142.4 GiB at 147 measured -2.6% vs 140 (measured: 39.68 vs 40.72) -
            # its residency changes the text it writes, which drafts worse
            eg = gguf_head.expert_gib(os.environ["DS41_GGUF"]) if "-00001-of-" in os.path.basename(os.environ["DS41_GGUF"]) else None
            if eg and total_ram_gib() >= 170 and "DS41_ARENA_GIB" not in os.environ and "DS41_RAM_RESERVE_GIB" not in os.environ                     and "--arena-gib" in a[:-1] and eg > float(a[a.index("--arena-gib") + 1]):
                i = a.index("--arena-gib")
                cover = min(150, math.ceil(eg) + 4, int(total_ram_gib() - BIG_RAM_RESERVE_GIB))
                if cover > float(a[i + 1]):
                    a[i + 1] = f"{cover:g}"
                    print(f"ds41_local: {os.path.basename(os.environ['DS41_GGUF'])}'s experts are {eg:.1f} GiB: "
                          f"arena {cover:g} GiB")
        # Unsloth UD-IQ2_XXS: the dense half re-quantized to q5_K instead of q6_K - measured: +1.2% at greedy, the same
        # agreement with DeepSeek's API (91.07 / 94.50 vs 90.80 / 94.38%).  Only when the config's q6_K is in force.
        if "UD-IQ2_XXS" in os.path.basename(os.environ["DS41_GGUF"]) and os.environ.get("STRATA_DENSE_REQUANT", "q6_K") == "q6_K":
            ENGINE_ENV["STRATA_DENSE_REQUANT"] = "q5_K"
    except Exception as e:
        print(f"ds41_local: expert-type check skipped ({e})")
    # STRATA_CHUNK_CPU (Start DeepSeek.bat: 3) helps short prompts only with every expert in RAM: on 96 / 64 GB the CPU pool
    # also serves the file tier and prompts read ~3.5% slower at k=3 (B2's lowk, 11-03) - off below 192 GB-class RAM
    if total_ram_gib() < 170 and os.environ.get("STRATA_CHUNK_CPU", "") not in ("", "0"):
        ENGINE_ENV["STRATA_CHUNK_CPU"] = "0"
    # Start DeepSeek.bat sets the larger drafter counts (<keep>:112,80,48) on a card of 22 GB or more (measured, at
    # 192 GB: ~+3% writing).  The VRAM they take comes out of the expert cache, and on a 64 GB PC the cache's misses
    # come from the SSD: the placement simulator prices 194 fewer slots at -5.7% there and -0.8% at 96 GB.  Below
    # KEEP_ALL_RAM_GIB: the smaller counts (<keep>:96,64,32), until a 64 GB run decides.  A row without a keep file:
    # nothing set (the drafter keeps all its experts).
    keep = drafter_keep(total_ram_gib(), os.environ)
    if keep:
        ENGINE_ENV["STRATA_DSPARK_KEEP"] = keep
    # the prompt-speed profile: by whether the routed experts fit the arena (prompt_profile).  EXL3 experts are not
    # in the GGUF: their size is unknown here (the file-tier profile)
    try:
        arena = float(a[a.index("--arena-gib") + 1]) if "--arena-gib" in a[:-1] else None
    except ValueError:
        arena = None
    experts = None
    if not x3:
        try:
            import gguf_head
            experts = gguf_head.expert_gib(os.environ["DS41_GGUF"])
        except Exception:
            experts = None
    pe, line = prompt_profile(arena, experts, os.environ)
    ENGINE_ENV.update(pe)
    print(line)
    # the prompt batch width: DS41_PROMPT_BATCH, else the file-tier profile's wider batch (prompt_batch)
    w = with_new_names(os.environ).get("STRATA_WIDE_PROMPT", pe.get("STRATA_WIDE_PROMPT",
                                       with_new_names(out.get("env") or {}).get("STRATA_WIDE_PROMPT", "")))
    pb, pb_line = prompt_batch(a, file_tier(arena, experts), str(w).strip() not in ("", "0"), big_experts, os.environ)
    pb_cur = int(float(a[a.index("--prompt-batch") + 1])) if "--prompt-batch" in a[:-1] else None
    ENGINE_ENV.update(sweep_fit_env(pb_cur, pb, {**(out.get("env") or {}), **os.environ}))
    if pb is not None:
        if "--prompt-batch" in a[:-1]:
            a[a.index("--prompt-batch") + 1] = str(pb)
        else:
            a += ["--prompt-batch", str(pb)]
    if pb_line:
        print(pb_line)
    if ENGINE_ENV:
        out["env"] = {**(out.get("env") or {}), **ENGINE_ENV}
    # DS41_GPU (ds41-paths.bat: the card with the most VRAM unless set): the server starts the engine on the config's
    # "gpu" (it sets CUDA_VISIBLE_DEVICES from it, nvidia-smi's numbering), not on the launcher's environment
    g = os.environ.get("DS41_GPU", "").strip()
    if g:
        if not g.isdigit():
            sys.exit(f"ds41_local: DS41_GPU={g!r} is not a GPU number (nvidia-smi's index; ds41-paths.bat)")
        out["gpu"] = int(g)
    # DS41_VRAM_MARGIN=GiB (the user's, or set above for files with bigger experts): the VRAM kept free beside the expert cache
    margin = env_gib("DS41_VRAM_MARGIN", None)
    if margin is not None:
        if "--vram-margin" in a[:-1]:
            a[a.index("--vram-margin") + 1] = f"{margin:g}"
        else:
            a += ["--vram-margin", f"{margin:g}"]
    vision = vision_section(m, out.get("gpu"))
    # the chat page's multi-GPU switch: the helper on the second card (serve_ds41.py starts it beside the engine), with
    # room left for the picture encoder and read-aloud when they share its card
    plan = helper_plan(out, a)
    voice = voice_section(m, out.get("gpu"), bool(plan))
    plan = share_card(plan, vision, voice)
    hf =os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", "helper.json")
    if plan:
        os.makedirs(os.path.dirname(hf), exist_ok=True)
        with open(hf, "w", encoding="utf-8") as f:
            json.dump(plan, f)
        out["env"] = {**(out.get("env") or {}), "STRATA_DS41_REMOTE": plan["name"]}
        print(f"ds41_local: multi-GPU on: {plan['gpu_name']} holds experts ({plan['gib']:g} GiB)")
    elif os.path.exists(hf):
        os.remove(hf)
    if vision:
        out["vision"] = vision
        if "--vision" not in a:   # here, not by the server: it would add --vram-reserve-mib, which this engine refuses
            a.append("--vision")
    if voice:
        out["voice"] = voice
    return out


def share_card(plan, vision, voice=None):
    """The helper's plan with VISION_GPU_GIB fewer experts when the picture encoder runs on its card, VOICE_GPU_GIB fewer
    when read-aloud does (else unchanged)."""
    if not plan:
        return plan
    less = VISION_GPU_GIB if vision and vision.get("cuda_device") == plan["gpu"] else 0
    tts_env = ((voice or {}).get("tts") or {}).get("env") or {}
    if tts_env.get("CUDA_VISIBLE_DEVICES") == str(plan["gpu"]):
        less += VOICE_GPU_GIB
    if less:
        plan = {**plan, "gib": plan["gib"] - less, "args": list(plan["args"])}
        plan["args"][plan["args"].index("--gib") + 1] = f"{plan['gib']:g}"
    return plan


def helper_plan(out, a):
    """The Setup card's multi-GPU switch -> what ds41_helper needs, or None (off, no second card, or a card under 20 GB:
    the small-card profile has no room for the slots the helper's misses go through)."""
    if not page_settings().get("multi_gpu") or "--rank" not in a[:-1] or "-m" not in a[:-1]:
        return None
    import setup_panel
    gpus = setup_panel.list_gpus()
    h = setup_panel.helper_gpu(gpus, out.get("gpu"))
    me = next((g for g in gpus if g["index"] == out.get("gpu")), None)
    if not h or not me or me["mib"] < 20000:
        return None
    args = ["-m", a[a.index("-m") + 1], "--rank", a[a.index("--rank") + 1], "--device", "0",
            "--skip-gib", f"{me['mib'] / 1024 - 9.5:g}", "--gib", f"{h['mib'] / 1024 - 2.5:g}", "--name", "ds41_remote"]
    if "--exl3" in a[:-1]:
        args += ["--exl3", a[a.index("--exl3") + 1]]
    return {"exe": os.path.join(os.path.dirname(out["exe"]), "ds41_helper.exe"), "args": args, "gpu": h["index"],
            "gpu_name": h["name"], "gib": h["mib"] / 1024 - 2.5, "name": "ds41_remote"}


VISION_FILE = _DE.get("vision") or ""   # 1017: DeepSeek-V4.1-Flash-1017-Vision.gguf (nazeshinjite/...-1017-ds4-GGUF)
# on by default since 0.94 (the user, 11-07; off in 0.92 / 0.93), never on the engine's card unless asked: the CPU takes
# no VRAM (decode unchanged, measured); a second card reads a picture ~25x faster (RTX 3060, bf16: 0.41 vs 11.5 s on
# 6 CPU threads for a 1024x701 picture, 11-07)
VISION_DEFAULT = "on"
VISION_OFF = ("off", "0", "no", "none", "false", "disable", "disabled")
# what the encoder holds on a card: 1.4 GiB after its warm-up (bf16, a 1,001-token picture) + the CUDA context, 1.5 GiB
# measured on an RTX 3060 (11-07); the multi-GPU helper keeps this much fewer experts when they share its card
VISION_GPU_GIB = 2.0
# read-aloud (Kokoro, onnxruntime CUDA) on the helper's card: "about 1.5 GB" (voice_section's note); without this the
# helper's 2.5 GiB of headroom was all that was left for it
VOICE_GPU_GIB = 1.5


def vision_choice():
    v = os.environ.get("DS41_VISION", "").strip().lower()
    if not v:
        try:
            with open(os.path.join(ROOT, "vision-choice.txt"), encoding="utf-8-sig") as f:
                v = f.readline().strip().lower()
        except OSError:
            v = ""
    return v or VISION_DEFAULT


def vision_place(mode, engine_gpu=None, gpus=None):
    """Where pictures are read: ("off" | "cpu" | "gpu", the card's nvidia-smi number or None for the engine's own,
    a note for the startup line); (None, None, why) for a choice that is not one.  "on" / "auto": the multi-GPU
    helper's card when the Setup card's switch is on and such a card is there, else the CPU - never the engine's card
    by default (it would take the experts' VRAM)."""
    mode = (mode or VISION_DEFAULT).strip().lower()
    if mode in VISION_OFF:
        return "off", None, ""
    if engine_gpu is None:
        g = os.environ.get("DS41_GPU", "").strip()
        engine_gpu = int(g) if g.isdigit() else None
    if mode in ("on", "auto", "yes", "true", "1"):
        if not page_settings().get("multi_gpu"):
            return "cpu", None, ""
        import setup_panel
        h = setup_panel.helper_gpu(setup_panel.list_gpus() if gpus is None else gpus, engine_gpu)
        if h is None:
            return "cpu", None, "multi-GPU is on, but no second card was found"
        return "gpu", h["index"], f"on {h['name']}, GPU {h['index']}: the multi-GPU card"
    if mode == "cpu":
        return "cpu", None, ""
    main_card = "on the engine's card: its VRAM comes out of the expert cache"
    if mode in ("main", "gpu"):
        return "gpu", None, main_card
    m = re.fullmatch(r"(?:gpu|cuda)\s*[:=]?\s*(\d+)", mode)
    if m:
        n = int(m.group(1))
        if engine_gpu is not None and n == engine_gpu:
            return "gpu", None, main_card
        if gpus is None:
            import setup_panel
            gpus = setup_panel.list_gpus()
        if gpus and not any(x["index"] == n for x in gpus):
            return "cpu", None, f"there is no GPU {n} (nvidia-smi's numbers)"
        return "gpu", n, f"on GPU {n}"
    return None, None, f"pictures {mode!r} (DS41_VISION / vision-choice.txt): use on, cpu, gpu:N, main or off"


def vision_plan():
    """For the launchers (--vision-setup): "<plan> <torch>" - plan "off", "cpu" or "cuda" (pictures on a card: setup
    installs PyTorch's CUDA build, a 2.0 GB download), torch the PyTorch in this Python: "none", "cpu" or "cuda"."""
    where = vision_place(vision_choice())[0]
    return f"{'cuda' if where == 'gpu' else where or 'off'} {torch_build() or 'none'}"


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


def vision_section(m, engine_gpu=None):
    """The server's "vision" section for the pictures choice, or None (off, or not set up: said in one line)."""
    where, dev, note = vision_place(vision_choice(), engine_gpu)
    if where is None:
        print(f"ds41_local: {note} - text only")
        return None
    if where == "off":
        return None
    side, ref = m + "/" + VISION_FILE, ROOT.replace("\\", "/") + "/tools/v41/vision/deepseek"
    missing = [os.path.basename(p) for p in (side, ref + "/vision.py") if not os.path.exists(p)]
    if not VISION_FILE:   # a checkpoint without a vision file (data/ds41/checkpoints.json) yet
        missing.insert(0, f"a vision file for checkpoint {CK['id']}")
    build = torch_build()
    if build is None:
        missing.append("PyTorch")
    if missing:
        print(f"ds41_local: pictures are on but not set up - missing {', '.join(missing)}: text only.  "
              f"Enable Pictures.bat sets them up (Disable Pictures.bat turns them off)")
        return None
    if where == "gpu" and build != "cuda":
        print(f"ds41_local: pictures would run {note}, but this PyTorch is the CPU build: they run on the CPU.  "
              f"Enable Pictures.bat installs the CUDA build (a 2.0 GB download)")
        where, note = "cpu", ""
    elif note and where == "cpu":
        print(f"ds41_local: pictures on the CPU ({note})")
    v = {"exe": ROOT.replace("\\", "/") + "/tools/v41/vision/ds41_vision.py", "mmproj": side,
         "model": os.environ["DS41_GGUF"], "reference_dir": ref, "gpu": where == "gpu"}
    if where == "gpu":   # DS41_VISION_DTYPE=f32: the CPU's exact output, ~1.1 GB more VRAM
        v["dtype"] = os.environ.get("DS41_VISION_DTYPE", "").strip().lower() or "bf16"
        if dev is not None:   # the server starts the encoder with CUDA_VISIBLE_DEVICES=dev (server.vision_env)
            v["cuda_device"] = dev
        v["where"] = note
    else:   # i5-13600K: 6 threads 8.1 s for a 1024x701 photo, 14 threads 9.2 s (the E-cores slow it down)
        v["threads"] = max(1, min(6, os.cpu_count() or 1))
        if vision_lazy():
            # 1107 review: the CPU encoder holds 3.9 GB of RAM (4.3 GB peak; PyTorch alone 1.5 GB), taken from the
            # ~14 GiB the RAM reserve leaves Windows on a 64 / 96 GB PC.  There it starts with the first picture (a few
            # seconds, once) and stops after VISION_LAZY_IDLE_S without one; serve_ds41 skips lazy mode's VRAM step
            # (this engine has no VRAM command)
            v.update({"lazy": True, "lazy_idle_s": VISION_LAZY_IDLE_S, "lazy_reserve_mib": 0})
            print(f"ds41_local: pictures on the CPU, started with the first picture ({total_ram_gib():.0f} GiB of RAM)")
    return v


# 170 (1107 vision review, was 110): a 128 GB PC's arena is total - 28 = ~100 GiB, not capped, so it has the same
# headroom as a 64 GB one; only from ~168 GiB does the 140 GiB cap leave spare RAM for a resident encoder.
VISION_LAZY_BELOW_GIB, VISION_LAZY_IDLE_S = 170, 600


def vision_lazy():
    """A CPU encoder that starts with the first picture: under VISION_LAZY_BELOW_GIB of RAM, or DS41_VISION_LAZY=1
    (=0: resident whatever the RAM)."""
    v = os.environ.get("DS41_VISION_LAZY", "").strip()
    if v in ("0", "1"):
        return v == "1"
    return total_ram_gib() < VISION_LAZY_BELOW_GIB


# Voice (off by default): dictation and read-aloud, serve/speech.py run by sherpa-onnx, started on first use and stopped
# after VOICE_IDLE_S idle.  Enable Voice.bat (setup-ds41-models.bat voice cpu|gpu) downloads the models into
# <models>/speech and writes voice-choice.txt.
VOICE_ASR = "parakeet-tdt-0.6b-v3-int8"     # csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (CC-BY-4.0), 671 MB
VOICE_TTS = "kokoro-multi-lang-v1_0"        # csukuangfj/kokoro-multi-lang-v1_0 (Apache-2.0), 401 MB
VOICE_VAD = "silero_vad.onnx"               # Silero VAD (MIT), 0.6 MB: cuts long recordings at pauses
VOICE_IDLE_S = 600
# i5-13600K, efficiency cores, the model idle (11-07): dictation 0.66 s for a 5 s clip (4 threads); read-aloud 2.6 s
# for 5.1 s of speech at 4 threads, 1.7 s at 8.  Any core, 4 threads: 0.29 s and 1.0 s - but those are the expert
# threads' cores.  While the model writes, its expert pool uses every core, the efficiency cores too: read-aloud then
# took 16 s for 5.9 s of speech (6 threads), dictation 1-10 s.
VOICE_THREADS = {"asr": 4, "tts": 6}


def voice_choice():
    v = os.environ.get("DS41_VOICE", "").strip().lower()
    if not v:
        try:
            with open(os.path.join(ROOT, "voice-choice.txt"), encoding="utf-8-sig") as f:
                v = f.readline().strip().lower()
        except OSError:
            v = ""
    return v or "off"


def sherpa_cuda():
    """True when the installed sherpa-onnx is its CUDA build (it carries onnxruntime's CUDA provider)."""
    spec = importlib.util.find_spec("sherpa_onnx")
    if spec is None or not spec.submodule_search_locations:
        return False
    lib = os.path.join(list(spec.submodule_search_locations)[0], "lib")
    return any(os.path.exists(os.path.join(lib, n))
               for n in ("onnxruntime_providers_cuda.dll", "libonnxruntime_providers_cuda.so"))


def voice_section(m, engine_gpu=None, helper_on=False):
    """The server's "voice" section (dictation + read-aloud), or None (off, or not set up: said in one line).
    "cpu": the efficiency cores of a hybrid Intel CPU (any core elsewhere); "gpu" / "gpu:N": read-aloud on a second
    NVIDIA card (dictation stays on the CPU: the int8 Parakeet is faster there - 0.66 s vs 0.87 s for a 5 s clip, 11-07;
    Kokoro on a 3060 0.47 s vs 2.6 s on the efficiency cores for 5.1 s of speech)."""
    mode = voice_choice()
    if mode in ("off", "0", "no", "none", "false"):
        return None
    if mode != "cpu" and not re.fullmatch(r"gpu(:\d+)?", mode):
        print(f"ds41_local: voice {mode!r} (DS41_VOICE / voice-choice.txt): use cpu, gpu or off - voice off")
        return None
    d = os.path.join(m, "speech")
    need = {"asr": os.path.join(d, VOICE_ASR, "encoder.int8.onnx"), "tts": os.path.join(d, VOICE_TTS, "model.onnx")}
    missing = [os.path.basename(os.path.dirname(p)) for p in need.values() if not os.path.exists(p)]
    if importlib.util.find_spec("sherpa_onnx") is None:
        missing.append("sherpa-onnx (the speech runtime)")
    if missing:
        print(f"ds41_local: voice is on but not set up - missing {', '.join(missing)}: voice off.  "
              f"Enable Voice.bat sets it up")
        return None
    env, device = {}, "cpu"
    if mode.startswith("gpu"):
        gpu = int(mode[4:]) if ":" in mode else None
        if gpu is None:
            import setup_panel
            h = setup_panel.helper_gpu(setup_panel.list_gpus(), engine_gpu)
            gpu = h["index"] if h else None
        if gpu is None or gpu == engine_gpu:
            print("ds41_local: voice on a second card was asked for, but there is none - voice on the CPU")
        elif not sherpa_cuda():
            print("ds41_local: voice on a second card needs sherpa-onnx's CUDA build (Enable Voice.bat gpu) - "
                  "voice on the CPU")
        else:
            # nvidia-smi's numbering, as the helper and the picture encoder use (not CUDA's fastest-first)
            env, device = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}, "cuda"
            if helper_on:
                print(f"ds41_local: voice shares GPU {gpu} with the multi-GPU helper (about 1.5 GB of it)")
    py = sys.executable.replace("\\", "/")
    script = ROOT.replace("\\", "/") + "/serve/speech.py"
    out = {"lazy_idle_s": VOICE_IDLE_S}
    for kind, port, folder in (("asr", 5110, VOICE_ASR), ("tts", 5210, VOICE_TTS)):
        args = ["-I", script, "--kind", kind, "--port", "{port}", "--model", os.path.join(d, folder).replace("\\", "/"),
                "--threads", str(VOICE_THREADS[kind]), "--device", device if kind == "tts" else "cpu",
                "--cpus", f"e{VOICE_THREADS[kind]}"]
        if kind == "asr" and os.path.exists(os.path.join(d, VOICE_VAD)):
            args += ["--vad", os.path.join(d, VOICE_VAD).replace("\\", "/")]
        out[kind] = {"cmd": py, "args": args, "port": port, "start_timeout_s": 180, "lazy_reserve_mib": 0,
                     "show_log": True, **({"env": env} if env and kind == "tts" else {})}
    return out


DEFAULT_NAME = "deepseek-v4.1-flash-1017-ds4-q2"   # the shipped config's model_name
ID_PREFIX = CK.get("model_id_prefix") or "deepseek-v4.1-flash"   # 1017: deepseek-v4.1-flash-1017


def model_names(path):
    """(model id, display name) for the model file the engine loads"""
    b = os.path.basename(path)
    low = b.lower()
    if low and low in (Q2_FILE.lower(), SLIM_FILE.lower()):
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
    return f"{ID_PREFIX}-{tag}", f"DeepSeek V4.1 Flash · {label}"


def derived_in_use(cfg):
    """{role: path} of the checkpoint-derived files this config gives the engine and the server (R-2): the rank file,
    the drafter and its expert counts (KEEP), the Engram hot set, Engram tables in their own folder, the vision file,
    EXL3 experts, the tokenizer export.  The engine's environment is this process's (the launcher's) plus the config's
    "env" (ENGINE_ENV above wins)."""
    a = cfg.get("args", [])
    env = {**os.environ, **(cfg.get("env") or {})}
    root = cfg.get("cwd") or ROOT

    def at(p):
        return p if os.path.isabs(p) else os.path.join(root, p)
    files = {}
    if "--rank" in a[:-1]:
        files["rank"] = at(a[a.index("--rank") + 1])
    if env.get("STRATA_SERVE_DSPARK", "").strip():
        files["drafter"] = env["STRATA_SERVE_DSPARK"]
        keep = env.get("STRATA_DSPARK_KEEP", "").strip()
        if keep:   # <counts file>:<K>[,<K1>,<K2>] - a Windows path has a ':' of its own
            p, _, ks = keep.rpartition(":")
            files["keep"] = at(p if p and re.fullmatch(r"[\d,]+", ks) else keep)
    hot = env.get("STRATA_ENGRAM_HOT", "").strip()
    if hot and hot != "0":
        files["engram_hot"] = hot
    model = a[a.index("-m") + 1] if "-m" in a[:-1] else ""
    for i, x in enumerate(a[:-1]):
        if x == "--engram":
            for d in a[i + 1].split(";"):
                if d and d != model and os.path.isdir(d):
                    files["engram"] = d
    if cfg.get("vision") and cfg["vision"].get("mmproj"):
        files["vision"] = cfg["vision"]["mmproj"]
    if "--exl3" in a[:-1]:
        files["exl3"] = a[a.index("--exl3") + 1]
    if cfg.get("tokenizer"):
        files["tokenizer"] = cfg["tokenizer"]
    return files


def checkpoint_report(cfg):
    """R-2: which checkpoint the model is (its stamp, file name or GGUF metadata; data/ds41/checkpoints.json), one line
    on it, and a WARNING line for every derived file in use that was built from another checkpoint - and for a model
    of another checkpoint than the launcher's file names (DS41_CHECKPOINT).  The config gets "checkpoint",
    "model_key" (R-6: the main model's files + fingerprint - what the server keys the state learned from the model's
    weights by: the learned expert profile, the reasoning memory) and "load_key" (the model key plus the drafter and
    EXL3 files: what the load-time estimate depends on).  A drafter made again, or a start with or without it, changes
    "load_key" only.  Warnings only: nothing is refused here."""
    a = cfg.get("args", [])
    if "-m" not in a[:-1]:
        return
    model = a[a.index("-m") + 1]
    files = derived_in_use(cfg)
    try:
        mid, mfp, how = ckpt.which(model)
        _, warns = ckpt.check(model, files)
    except Exception as e:  # noqa: BLE001  (a broken stamp or table must not stop a start)
        print(f"ds41_local: checkpoint check skipped ({e})")
        return
    if mid is None:
        print(f"ds41_local: checkpoint of {os.path.basename(model)}: unknown (no stamp, and neither its name nor its "
              f"metadata names one) - derived files not checked")
    else:
        label = (ckpt.load()["checkpoints"].get(mid) or {}).get("label", mid)
        print(f"ds41_local: checkpoint {mid} ({label}, fingerprint {mfp or '?'}, by the model's {how})")
        if mid != CK["id"]:
            warns.insert(0, f"the model is checkpoint {mid}, but the launcher's file names are checkpoint {CK['id']}'s "
                            f"(DS41_CHECKPOINT / data/ds41/checkpoints.json 'default')")
    for w in warns:
        print(f"ds41_local: WARNING - {w}.  Rebuild it from this checkpoint (setup-ds41-models.bat) or pick the "
              f"matching model (Choose Model.bat): mixing checkpoints costs speed and drafts, never correctness")
    cfg["checkpoint"] = {"id": mid, "fingerprint": mfp, "how": how, "files": CK["id"]}
    try:
        cfg["model_key"] = ckpt.model_key([model], mfp)
        cfg["load_key"] = ckpt.model_key([model, files.get("drafter"), files.get("exl3")], mfp)
    except OSError as e:
        print(f"ds41_local: model key skipped ({e})")


# R-6: state that outlives a start and depends on the model's weights, keyed by the model key ("model_key" above:
# the main model only) - or, for the load-time estimate, by "load_key" (the model key + the drafter and EXL3 files).
#   on disk, ours:   <config>.last-start.json (the load-time estimate)            -> deleted here when load_key changes
#                    the config's "expert_profile_save" (opt-in, learned tier)    -> the server ignores and deletes it
#                                                                                    (<save>.model-key differs)
#   on disk, derived (rank, KEEP, drafter, vision, Engram hot set, tokenizer)     -> stamps + checkpoint_report()
#   the user's own:  "slot_save_path" session files                             -> left alone: the engine refuses a
#                                                                                    file saved with another model
#                                                                                    (conversation_file.cpp fingerprint)
#   in the engine process only (die with it; a model change is a new engine process): conversation cache slots,
#   snapshots + snapshot stack, keep-cache, the parked conversation, DSpark seed rows; in the server process only:
#   reasoning memory (bound to the key: serve/reasoning_memory.py), stored responses, the tokenizer's prefix memo,
#   the vision encoder's per-start temp folder
MODEL_KEY_FILE = "model-key.json"


def model_state(cfg, out_path):
    """R-6: the load key (the model key + drafter / EXL3 files) against the last start's (<config folder>/
    model-key.json); when it changed, the persisted state above that is ours - the load-time estimate - is cleared.
    The first start with this file only records the keys."""
    key = cfg.get("model_key")
    lkey = cfg.get("load_key") or key
    if not key:
        return
    d = os.path.dirname(os.path.abspath(out_path))
    path = os.path.join(d, MODEL_KEY_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        prev, prev_l = rec.get("model_key"), rec.get("load_key") or rec.get("model_key")
    except (OSError, ValueError, AttributeError):
        prev = prev_l = None
    if prev == key and prev_l == lkey:
        return
    if prev_l and prev_l != lkey:
        cleared = []
        last = os.path.splitext(os.path.abspath(out_path))[0] + ".last-start.json"
        if os.path.exists(last):
            try:
                os.remove(last)
                cleared.append(os.path.basename(last))
            except OSError:
                pass
        what = (f"the model changed since the last start (key {prev} -> {key})" if prev != key else
                f"the drafter / EXL3 files changed since the last start (load key {prev_l} -> {lkey})")
        print(f"ds41_local: {what}" + (f": cleared {', '.join(cleared)}" if cleared else ""))
    a = cfg.get("args", [])
    try:
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump({"model_key": key, "load_key": lkey,
                       "model": os.path.basename(a[a.index("-m") + 1]) if "-m" in a[:-1] else None,
                       "checkpoint": (cfg.get("checkpoint") or {}).get("id")}, f, indent=1)
        os.replace(path + ".part", path)
    except OSError:
        pass


def main():
    if sys.argv[1:] == ["--vision-setup"]:   # first-start.bat, setup-ds41-models.bat: "<plan> <torch>"
        print(vision_plan())
        return
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        with open(sys.argv[1], encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f"ds41_local: cannot read {sys.argv[1]}: {e}")
    cfg = expand(raw)
    page_access(cfg, page_settings())
    args = cfg.get("args", [])
    if "-m" in args[:-1]:
        mid, shown = model_names(args[args.index("-m") + 1])
        if "--exl3" in args[:-1]:   # EXL3 experts on the default GGUF: named after the experts
            b = re.search(r"EXL3-([0-9.]+)bpw", args[args.index("--exl3") + 1], re.I)
            bpw = b.group(1) if b else ""
            mid = f"{ID_PREFIX}-exl3-{bpw}bpw" if bpw else f"{ID_PREFIX}-exl3"
            shown = f"DeepSeek V4.1 Flash · EXL3 {bpw} bpw".replace("  ", " ")
        if cfg.get("model_name", DEFAULT_NAME) == DEFAULT_NAME:
            cfg["model_name"] = mid
        cfg.setdefault("display_name", shown)
    checkpoint_report(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
    model_state(cfg, sys.argv[2])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    a = cfg.get("args", [])
    gib = a[a.index("--arena-gib") + 1] if "--arena-gib" in a else "?"
    # every file / folder the engine and server open at start (a missing one failed later, with a vaguer message)
    paths = [cfg.get("exe"), cfg.get("tokenizer")] + [a[i + 1] for i, x in enumerate(a[:-1])
                                                      if x in ("-m", "--rank", "--exl3")]
    paths += [d for i, x in enumerate(a[:-1]) if x == "--engram" for d in a[i + 1].split(";")]   # a list of folders
    missing = list(dict.fromkeys(p for p in paths if p and not os.path.exists(p)))   # (the model may be its Engram too)
    model = a[a.index("-m") + 1] if "-m" in a else "?"
    x3 = a[a.index("--exl3") + 1] if "--exl3" in a[:-1] else ""
    print(f"ds41_local: {total_ram_gib():.0f} GiB RAM -> expert arena {gib} GiB, route bias {out_bias(args)}; model {os.path.basename(model)}"
          + (f" with the EXL3 experts in {os.path.basename(x3)}" if x3 else "")
          + f" (Choose Model.bat picks another); models in {os.environ['DS41_MODELS']}")
    # under ~110 GB of RAM the ds4 Q2 file reads most experts from the SSD (measured, an RTX 3090): one line on the
    # model that runs better there - UD-IQ1_M at route bias 0.10, the pick under 110 GB since 0.95
    if total_ram_gib() < 102 and os.path.basename(model) and os.path.basename(model) in (Q2_FILE, SLIM_FILE):
        print("ds41_local: tip - with this much RAM, Unsloth's UD-IQ1_M writes faster than the ds4 Q2 file (0.95: 40 / 35 "
              "tok/s at 96 / 64 GB; ds4 Q2 27 / 20 with 0.93) and agrees more (90.0 / 89.9 vs 89.9%): Choose Model.bat, "
              "'download Unsloth UD-IQ1_M'")
    if cfg.get("vision"):
        where = (cfg["vision"].get("where") or "on the GPU") if cfg["vision"]["gpu"] else "on the CPU"
        if cfg["vision"].get("lazy"):
            where += ", loaded with the first picture"
        print(f"ds41_local: pictures on ({where}); Disable Pictures.bat turns them off")
    if cfg.get("voice"):
        env = cfg["voice"]["tts"].get("env") or {}
        print(f"ds41_local: voice on (read-aloud on {'GPU ' + env['CUDA_VISIBLE_DEVICES'] if env else 'the CPU'}, "
              f"dictation on the CPU; "
              f"the speech models start with the first use)")
    for p in missing:
        print(f"ds41_local: MISSING {p}  (see README.md, 'Setup')")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
