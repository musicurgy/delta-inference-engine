#!/usr/bin/env python3
"""Tests for setup_panel.py and pick_model.py --first (CPU, no model).  Run: python tools/v41/serve/test_setup_panel.py"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:300]))
    if not ok:
        FAILS.append(name)


def gguf(path, kv):
    """a GGUF with only metadata"""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = b""
    for k, v in kv.items():
        body += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<I", 4) + struct.pack("<I", v))
    Path(path).write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body)


tmp = tempfile.mkdtemp()
models = os.path.join(tmp, "models")
os.makedirs(models + "/custom")
os.environ["DS41_MODELS"] = models
os.environ["DS41_RAM_GB"] = "192"
os.environ["DS41_GPU"] = "0"
import pick_model as P  # noqa: E402
import setup_panel as SP  # noqa: E402
P.ROOT = tmp
P.CHOICE = SP.P.CHOICE = os.path.join(tmp, "model-choice.txt")
SP.SETTINGS = os.path.join(tmp, "delta-settings.json")
SP.list_gpus = lambda: [{"index": 0, "name": "RTX 3090", "mib": 24576}, {"index": 1, "name": "RTX 3060", "mib": 12288}]

# the recommendation follows the RAM
check("192 GB -> UD-IQ2_XXS", SP.recommended(192) == "UD-IQ2_XXS")
check("128 GB -> the default", SP.recommended(128) == "default")
check("96 GB -> UD-IQ1_S", SP.recommended(96) == "UD-IQ1_S")

# the custom check
good = models + "/custom/mine.gguf"
gguf(good, {"general.architecture": "deepseek41", "deepseek41.expert_count": 384})
gguf(models + "/custom/split-00001-of-00003.gguf", {"general.architecture": "deepseek41", "deepseek41.expert_count": 384})
gguf(models + "/custom/split-00002-of-00003.gguf", {"general.architecture": "deepseek41"})
gguf(models + "/custom/llama.gguf", {"general.architecture": "qwen3"})
gguf(models + "/custom/pruned.gguf", {"general.architecture": "deepseek41", "deepseek41.expert_count": 192})
Path(models + "/custom/junk.gguf").write_bytes(b"not a model")
c = {x["name"]: x for x in SP.custom_models(models)}
check("a DeepSeek GGUF passes", c["mine.gguf"]["ok"] and c["mine.gguf"]["needs_default"], c.get("mine.gguf"))
check("a split model's first file passes, on its own", c["split-00001-of-00003.gguf"]["ok"] and not c["split-00001-of-00003.gguf"]["needs_default"])
check("later shards are not listed", "split-00002-of-00003.gguf" not in c)
check("another architecture is refused", not c["llama.gguf"]["ok"] and "qwen3" in c["llama.gguf"]["why"], c["llama.gguf"])
check("a pruned model is refused", not c["pruned.gguf"]["ok"] and "384" in c["pruned.gguf"]["why"])
check("a broken file is refused", not c["junk.gguf"]["ok"])

# the list, without the default installed
cat = SP.catalog(models)
ids = {m["id"]: m for m in cat["models"]}
check("the catalog is JSON", bool(json.dumps(cat)))
check("custom models are listed", "custom/mine.gguf" in ids)
check("a custom one-file model needs the default", "default" in ids["custom/mine.gguf"]["blocked"], ids["custom/mine.gguf"])
check("a custom split model does not", ids["custom/split-00001-of-00003.gguf"]["blocked"] == "")
check("the second card is found", cat["helper_gpu"] and cat["helper_gpu"]["index"] == 1, cat["helper_gpu"])
check("UD-IQ2_XXS is shown as recommended here", any(m["recommended"] and "UD-IQ2_XXS" in m["name"] for m in cat["models"]))

# apply: refused changes write nothing
for bad, why in (({"model": "nope.gguf"}, "unknown model"), ({"model": "custom/mine.gguf"}, "blocked"), ({"mode": "turbo"}, "bad mode")):
    try:
        SP.apply(bad)
        check(f"refused: {why}", False)
    except ValueError:
        check(f"refused: {why}", not os.path.exists(P.CHOICE) and not os.path.exists(SP.SETTINGS))

# apply, then a failed start goes back; a good start keeps it
Path(P.CHOICE).write_text("old.gguf\n", encoding="utf-8")
SP.apply({"model": "custom/split-00001-of-00003.gguf", "mode": "exact", "multi_gpu": True})
s = SP.load_settings()
check("apply writes the choice", Path(P.CHOICE).read_text().strip() == "custom/split-00001-of-00003.gguf" and s["mode"] == "exact" and s["multi_gpu"])
check("rollback restores the model", SP.rollback() and Path(P.CHOICE).read_text().strip() == "old.gguf")
check("rollback happens once", not SP.rollback())
SP.apply({"model": "custom/split-00001-of-00003.gguf"})
SP.started()
check("a start that worked keeps the choice", not SP.rollback() and "split" in Path(P.CHOICE).read_text())
check("the mode survives", SP.load_settings().get("mode") == "exact")

# the first start's picker, as a user answers it
env = {**os.environ, "DS41_MODELS": os.path.join(tmp, "m2"), "DS41_RAM_GB": "128"}
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="1\n", capture_output=True, text=True, env=env)
check("--first: the default exits 10", p.returncode == 10 and "recommended" in p.stdout, (p.returncode, p.stdout[-300:], p.stderr[-300:]))
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="5\n", capture_output=True, text=True, env=env)
check("--first: none explains and exits 11", p.returncode == 11 and "custom" in p.stdout, (p.returncode, p.stdout[-300:], p.stderr[-300:]))
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="x\n9\n\n", capture_output=True, text=True, env=env)
check("--first: bad answers ask again, Enter takes the recommended", p.returncode == 10, (p.returncode, p.stdout[-300:]))

print("all passed" if not FAILS else f"FAILED {len(FAILS)}")
sys.exit(1 if FAILS else 0)
