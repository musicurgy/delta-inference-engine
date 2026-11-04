#!/usr/bin/env python3
"""setup_panel.py - what the web page's Setup card and the first start share: the models this PC can load, the one setting
file, the check of a custom model, and the rollback after a start that failed.  Standard library only.

  delta-settings.json (repository folder, not in git): {"mode": "fast|balanced|exact", "multi_gpu": true|false, "pending": ...}
  model-choice.txt stays the model's one source, as for Choose Model.bat.
  python setup_panel.py --rollback    after a failed start: put the model back that last started (exit 0 = put back)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_head  # noqa: E402
import pick_model as P  # noqa: E402

ROOT = P.ROOT
SETTINGS = os.path.join(ROOT, "delta-settings.json")
MODES = {"fast": "0.28", "balanced": "0.10", "exact": "0"}     # the route bias of each choice on the page
CUSTOM = "custom"                                              # the folder inside the model folder
RESTART_EXIT = 75                                              # serve_ds41.py's exit code: Start DeepSeek.bat starts it again


def load_settings() -> dict:
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, SETTINGS)


def check_gguf(path: str):
    """(True, "") when the engine can load this GGUF as the model, else (False, why, in plain words)."""
    if os.path.basename(path).find("-of-") >= 0 and "-00001-of-" not in os.path.basename(path):
        return False, "this is not the first file of a split model (pick the one named -00001-of-)"
    try:
        meta, _ = gguf_head.read_header(path)
    except (OSError, ValueError, KeyError, UnicodeError, MemoryError) as e:
        return False, f"it cannot be read as a GGUF file ({e})"
    if meta.get("general.architecture") != "deepseek41":
        return False, f"it is a {meta.get('general.architecture') or 'different'} model, not DeepSeek-V4.1-Flash"
    experts = meta.get("deepseek41.expert_count", meta.get("deepseek41.n_routed_experts"))
    if experts is not None and experts != 384:
        return False, f"it has {experts} experts per layer; this engine runs the full 384 (pruned models are not supported)"
    return True, ""


def check_exl3(path: str):
    """An EXL3 folder: exllamav3's own format, on the default GGUF's dense layers."""
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, ValueError):
        return False, "it has no readable config.json"
    if "quantization_config" not in c and "quantization" not in c:
        return False, "its config.json names no EXL3 quantization"
    return True, ""


def custom_models(m: str) -> list:
    """Everything in <models>/custom: [{id, name, ok, why, needs_default}].  id = what goes in model-choice.txt."""
    d = os.path.join(m, CUSTOM)
    out = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        p = os.path.join(d, n)
        if os.path.isdir(p):
            if not os.path.exists(os.path.join(p, "config.json")):
                continue
            ok, why = check_exl3(p)
            out.append({"id": f"{CUSTOM}/{n}", "name": n, "ok": ok, "why": why, "needs_default": True})
        elif n.lower().endswith(".gguf") and ("-of-" not in n or "-00001-of-" in n):
            ok, why = check_gguf(p)
            out.append({"id": f"{CUSTOM}/{n}", "name": n, "ok": ok, "why": why, "needs_default": "-00001-of-" not in n})
    return out


def _speed(tps: str, ram) -> str:
    """the writing speed column for this PC's RAM ('40 / 36 / 27 / 20' at 192 / 128 / 96 / 64 GB)"""
    cols = [x.strip() for x in tps.split("/")]
    if len(cols) != 4 or ram is None:
        return ""
    v = cols[0 if ram >= 170 else 1 if ram >= 110 else 2 if ram >= 80 else 3]
    return "" if v == "-" else f"{v} tok/s"


def recommended(ram) -> str:
    """the name pick_model recommends for this RAM"""
    if ram is not None and ram < P.LOW_RAM_GB:
        return P.LOW_RAM_PICK
    if ram is not None and ram >= P.HIGH_RAM_GB:
        return P.HIGH_RAM_PICK
    return "default"


def catalog(m=None) -> dict:
    """The model list for the page: installed ones can be picked, the others are shown with what they take."""
    m = m or P.models_dir()
    ram = P.ram_gb()
    rec = recommended(ram)
    base = P.Q2 if os.path.exists(os.path.join(m, P.Q2)) else P.SLIM if os.path.exists(os.path.join(m, P.SLIM)) else None
    cur = P.current()
    now = cur if cur and os.path.exists(cur if os.path.isabs(cur) else os.path.join(m, cur)) else base
    rows = []

    def add(id_, name, gb, tps, agr, installed, is_rec=False, needs_default=False):
        rows.append({"id": id_, "name": name, "gb": gb, "speed": _speed(tps, ram), "agreement": agr, "installed": installed,
                     "current": bool(id_) and id_ == now, "recommended": is_rec,
                     "blocked": "needs the default model installed" if needs_default and not base else ""})

    add(base or "", P.DEFAULT[0], P.DEFAULT[1], P.DEFAULT[2], P.DEFAULT[3], bool(base), rec == "default")
    for q, (gb, tps, agr, eng) in P.UNSLOTH_Q.items():
        p = P.unsloth_first(m, q)
        if p or agr != "-":        # the unmeasured files stay on Choose Model.bat's list
            add(p or q, "Unsloth " + q + P.TAG.get(q, ""), gb, tps, agr, bool(p), rec == q, eng == "the default's")
    for f, (name, gb, tps, agr, _) in P.GSQ.items():
        if os.path.exists(os.path.join(m, f)):
            add(f, name, gb, tps, agr, True, False, True)
    for d, (name, gb, tps, agr, _) in P.EXL3.items():
        p = P.exl3_ready(m, d)
        if p:
            add(p, name, gb, tps, agr, True, False, True)
    custom = custom_models(m)
    for c in custom:
        if c["ok"]:
            rows.append({"id": c["id"], "name": c["name"] + " (your own)", "gb": None, "speed": "", "agreement": "", "installed": True,
                         "current": c["id"] == now, "recommended": False,
                         "blocked": "needs the default model installed" if c["needs_default"] and not base else ""})
    s = load_settings()
    gpus = list_gpus()
    return {"ram_gb": ram, "models": rows, "refused": [c for c in custom if not c["ok"]], "custom_folder": os.path.join(m, CUSTOM),
            "mode": s.get("mode") if s.get("mode") in MODES else "fast", "gpus": gpus, "multi_gpu": bool(s.get("multi_gpu")),
            "helper_gpu": helper_gpu(gpus), "bias_override": "DS41_ROUTE_BIAS" in os.environ}


def list_gpus() -> list:
    """[{index, name, mib}] from nvidia-smi, in PCI bus order; [] without it"""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,display_active", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in o.splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) >= 3 and a[0].isdigit() and a[2].isdigit():
            out.append({"index": int(a[0]), "name": a[1], "mib": int(a[2]), "display": a[-1].lower() == "enabled"})
    return out


def helper_gpu(gpus, engine_gpu=None):
    """the second card for the helper: the largest one that is not the engine's, 8 GB or more, a card without a display
    before one with; None if there is none"""
    if engine_gpu is None:
        g = os.environ.get("DS41_GPU", "").strip()
        engine_gpu = int(g) if g.isdigit() else (max(gpus, key=lambda x: x["mib"])["index"] if gpus else 0)
    rest = [x for x in gpus if x["index"] != engine_gpu and x["mib"] >= 7800]
    return max(rest, key=lambda x: (x["mib"], not x.get("display"))) if rest else None


def apply(req: dict) -> dict:
    """What the page's Apply sends: {model, mode, multi_gpu}.  Checks everything first, writes only if all is right."""
    m = P.models_dir()
    cat = catalog(m)
    s = load_settings()
    model = req.get("model")
    if model is not None:
        row = next((r for r in cat["models"] if r["id"] == model), None)
        if row is None or not row["installed"]:
            raise ValueError("that model is not installed")
        if row["blocked"]:
            raise ValueError(f"{row['name']} {row['blocked']}")
    mode = req.get("mode")
    if mode is not None and mode not in MODES:
        raise ValueError("mode must be fast, balanced or exact")
    if req.get("multi_gpu") and not cat["helper_gpu"]:
        raise ValueError("no second NVIDIA card with 8 GB or more was found")
    if model is not None:
        was = P.current()
        s["pending"] = {"model": was or ""}
        with open(P.CHOICE, "w", encoding="utf-8") as f:
            f.write(model + "\n")
    if mode is not None:
        s["mode"] = mode
    if "multi_gpu" in req:
        s["multi_gpu"] = bool(req["multi_gpu"])
    save_settings(s)
    return s


def started() -> None:
    """The model is loaded: the choice stays, whatever a later start does."""
    s = load_settings()
    if s.pop("pending", None) is not None:
        save_settings(s)


def rollback() -> bool:
    """After a start that failed: the model before the page's last change, once.  False = nothing to put back."""
    s = load_settings()
    p = s.pop("pending", None)
    if p is None:
        return False
    if p.get("model"):
        with open(P.CHOICE, "w", encoding="utf-8") as f:
            f.write(p["model"] + "\n")
    elif os.path.exists(P.CHOICE):
        os.remove(P.CHOICE)
    save_settings(s)
    return True


if __name__ == "__main__":
    if sys.argv[1:] == ["--rollback"]:
        ok = rollback()
        if ok:
            print("That model did not start: going back to the one that did.")
        sys.exit(0 if ok else 1)
    print(json.dumps(catalog(), indent=1))
