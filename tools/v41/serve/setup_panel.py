#!/usr/bin/env python3
"""setup_panel.py - what the web page's Setup card and the first start share: the models this PC can load, the one setting
file, the check of a custom model, and the rollback after a start that failed.  Standard library only.

  delta-settings.json (repository folder, not in git): {"mode": "fast|balanced|exact", "multi_gpu": true|false, "pending": ...,
    "phone": true|false, "phone_key": "<the API key the phone switch made>", "auto_pause_mib": N}
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
            "helper_gpu": helper_gpu(gpus), "bias_override": "DS41_ROUTE_BIAS" in os.environ, "health": pc_health(),
            "phone": phone_state(s), "auto_pause_mib": int(s.get("auto_pause_mib") or 0)}


# ---- use from other devices (a phone): the page's switch.  On: ds41_local.py writes "host": "0.0.0.0" and the API key
# into this PC's config (the server refuses to listen beyond this PC without a key); the page shows the address and the
# key as a QR code.  The Windows firewall rule is offered separately and only runs with Windows' own UAC prompt.
LOCAL_CONFIG = os.path.join(HERE, "local", "ds41-windows.json")
AUTO_PAUSE_CHOICES = (0, 512, 1024, 2048, 4096)


def port() -> int:
    p = os.environ.get("DS41_PORT", "").strip()
    return int(p) if p.isdigit() else 8080


def lan_ips() -> list:
    """this PC's IPv4 addresses on its networks, the default route's first (sends nothing)"""
    import socket
    first, ips = None, set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            first = sk.getsockname()[0]
    except OSError:
        pass
    try:
        ips = {i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))   # noqa: E731
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def rule_name(p=None) -> str:
    return f"Delta DeepSeek {p or port()}"


def firewall_rule(p=None):
    """True / False: the inbound rule for the port is there (read only, no admin); None: cannot tell (not Windows)"""
    if os.name != "nt":
        return None
    try:
        r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name(p)}"],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and rule_name(p) in r.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def firewall_command(add=True, p=None) -> str:
    """the PowerShell line the page shows (and runs elevated, after Windows' UAC prompt, when the user asks)"""
    p = p or port()
    if add:
        return (f'New-NetFirewallRule -DisplayName "{rule_name(p)}" -Direction Inbound -Protocol TCP -LocalPort {p} '
                f'-Action Allow -Profile Private')
    return f'Remove-NetFirewallRule -DisplayName "{rule_name(p)}"'


def change_firewall(add=True) -> bool:
    """run firewall_command elevated: Windows shows its UAC prompt and the user says yes or no there.  Never silent.
    -> whether the rule is there afterwards"""
    if os.name != "nt":
        raise ValueError("the firewall rule is for Windows")
    cmd = firewall_command(add).replace("'", "''")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        f"Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList "
                        f"'-NoProfile','-Command','{cmd}'"], capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        pass
    return bool(firewall_rule())


def _local_config() -> dict:
    try:
        with open(LOCAL_CONFIG, encoding="utf-8") as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def phone_state(s=None) -> dict:
    """for the page: is this server reachable from other devices now, at which addresses, with which key"""
    s = load_settings() if s is None else s
    c = _local_config()
    host = str(c.get("host") or "127.0.0.1")
    key = c.get("api_key")
    key = key[0] if isinstance(key, list) and key else key.split(",")[0].strip() if isinstance(key, str) else ""
    on = host not in ("127.0.0.1", "localhost", "::1", "")
    p = port()
    return {"on": on, "wanted": bool(s.get("phone")), "own_config": on and not s.get("phone"), "port": p,
            "addresses": lan_ips() if on else [], "key": key if on else "", "firewall": firewall_rule(p) if on else None,
            "firewall_add": firewall_command(True, p), "firewall_remove": firewall_command(False, p)}


def pc_health():
    """what windows_tips.py found at this start (local/windows-health.json): [{"text", "how"}], [] = all fine, None =
    not checked (DS41_NO_TIPS, another launcher)"""
    try:
        with open(os.path.join(HERE, "local", "windows-health.json"), encoding="utf-8") as f:
            items = json.load(f).get("items")
        return [x for x in items if isinstance(x, dict) and x.get("text")] if isinstance(items, list) else None
    except (OSError, ValueError, AttributeError):
        return None


def list_gpus() -> list:
    """[{index, name, mib, display, cc, lanes}] from nvidia-smi, in PCI bus order (cc: compute capability as 8.6, lanes:
    the PCIe link's current width; None when not read); [] without it"""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,pcie.link.width.current,display_active",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in o.splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) >= 3 and a[0].isdigit() and a[2].isdigit():
            cc = lanes = None
            if len(a) >= 6:
                try:
                    cc = float(a[3])
                except ValueError:
                    pass
                lanes = int(a[4]) if a[4].isdigit() else None
            out.append({"index": int(a[0]), "name": a[1], "mib": int(a[2]), "display": a[-1].lower() == "enabled",
                        "cc": cc, "lanes": lanes})
    return out


def helper_gpu(gpus, engine_gpu=None):
    """the second card for the helper: the largest one that is not the engine's, 8 GB or more, RTX 20 or newer (compute
    7.5: the CUDA 13 builds - the engine zips, PyTorch cu130 for pictures - have no code for older cards), a card without
    a display before one with, then the wider PCIe link (two 3060s here: x1 and x4; Windows reports display_active
    Disabled for the card driving the screen too, so the link is what tells them apart); None if there is none"""
    if engine_gpu is None:
        g = os.environ.get("DS41_GPU", "").strip()
        engine_gpu = int(g) if g.isdigit() else (max(gpus, key=lambda x: x["mib"])["index"] if gpus else 0)
    rest = [x for x in gpus if x["index"] != engine_gpu and x["mib"] >= 7800 and (x.get("cc") or 7.5) >= 7.5]
    return max(rest, key=lambda x: (x["mib"], not x.get("display"), x.get("lanes") or 0)) if rest else None


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
        raise ValueError("no second NVIDIA card (RTX 20 or newer, 8 GB or more) was found")
    if "auto_pause_mib" in req and req["auto_pause_mib"] not in AUTO_PAUSE_CHOICES:
        raise ValueError(f"auto_pause_mib must be one of {', '.join(map(str, AUTO_PAUSE_CHOICES))}")
    if model is not None:
        was = P.current()
        s["pending"] = {"model": was or ""}
        with open(P.CHOICE, "w", encoding="utf-8") as f:
            f.write(model + "\n")
    if mode is not None:
        s["mode"] = mode
    if "multi_gpu" in req:
        s["multi_gpu"] = bool(req["multi_gpu"])
    if "phone" in req:
        s["phone"] = bool(req["phone"])
        if s["phone"] and not s.get("phone_key"):
            import secrets
            s["phone_key"] = secrets.token_urlsafe(24)       # 32 characters; kept for the next time it is switched on
    if "auto_pause_mib" in req:
        s["auto_pause_mib"] = int(req["auto_pause_mib"])
    save_settings(s)
    return s


def needs_restart(req: dict) -> bool:
    """an Apply that changes how the server starts (model, mode, cards, phone access); auto-pause applies at once"""
    return any(k in req for k in ("model", "mode", "multi_gpu", "phone"))


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
    if not isinstance(p, dict):     # a hand-edited delta-settings.json: nothing to put back, but it starts
        save_settings(s)
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
