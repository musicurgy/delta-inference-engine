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

# the recommendation: UD-IQ2_XXS from 110 GB up and when the RAM is not known, UD-IQ1_M below (at bias 0.10)
for ram in (192, 128, 115, 110, None):
    check(f"{ram} GB -> UD-IQ2_XXS", SP.recommended(ram) == "UD-IQ2_XXS")
for ram in (109, 96, 64):
    check(f"{ram} GB -> UD-IQ1_M", SP.recommended(ram) == "UD-IQ1_M")
check("the menu tags say which RAM each pick is for", P.TAG.get("UD-IQ2_XXS") == " (110 GB and up)" and P.TAG.get("UD-IQ1_M") == " (under 110 GB)")
check("the 1-bit files are still offered", "UD-IQ1_S" in P.UNSLOTH_Q and "UD-IQ1_M" in P.UNSLOTH_Q)

# the Speed / Quality switch: this PC's default, the measured notes, a chosen Fast stays Fast
IQ1M = "unsloth/UD-IQ1_M/x-UD-IQ1_M-00001-of-00005.gguf"
check("64 GB + UD-IQ1_M defaults to Quality", SP.default_mode(IQ1M, 64) == "balanced")
check("96 GB + UD-IQ1_S defaults to Quality", SP.default_mode("x-UD-IQ1_S-00001-of-00005.gguf", 96) == "balanced")
check("128 GB + UD-IQ1_M and 64 GB + UD-IQ2_XXS default to Fast",
      SP.default_mode(IQ1M, 128) == "fast" and SP.default_mode("x-UD-IQ2_XXS-00001-of-00006.gguf", 64) == "fast")
n = SP.mode_notes(IQ1M, 64)
check("notes at 64 GB, UD-IQ1_M", n["balanced"] == "89.9% agreement, 35 tok/s" and n["fast"] == "87.7% agreement, 42 tok/s"
      and n["exact"] == "91.2% agreement, 16 tok/s", n)
check("a cell that was not measured stays empty", SP.mode_notes(IQ1M, 192)["balanced"] == "", SP.mode_notes(IQ1M, 192))
check("an own model has no notes", SP.mode_notes("custom/mine.gguf", 64) == {"fast": "", "balanced": "", "exact": ""})
check("own-model rows carry the switch's default", all("default_mode" in r and "modes" in r for r in SP.catalog(models)["models"]))
check("an old page's saved fast is not a choice", SP.chosen_mode({"mode": "fast"}) is None)
check("a chosen fast is", SP.chosen_mode({"mode": "fast", "mode_chosen": True}) == "fast")
check("quality / exact are always choices", SP.chosen_mode({"mode": "balanced"}) == "balanced" and SP.chosen_mode({"mode": "exact"}) == "exact")

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
check("a custom one-file model needs the ds4 Q2 file", "ds4 Q2" in ids["custom/mine.gguf"]["blocked"], ids["custom/mine.gguf"])
check("a custom split model does not", ids["custom/split-00001-of-00003.gguf"]["blocked"] == "")
check("the second card is found", cat["helper_gpu"] and cat["helper_gpu"]["index"] == 1, cat["helper_gpu"])
# a card older than RTX 20 is never the second card (no code for it in the CUDA 13 builds); unknown cc still counts
pascal = [{"index": 0, "name": "RTX 3090", "mib": 24576, "cc": 8.6}, {"index": 1, "name": "Tesla P40", "mib": 24576, "cc": 6.1},
          {"index": 2, "name": "RTX 2080 Ti", "mib": 11264, "cc": 7.5}]
check("a Pascal card is passed over", (SP.helper_gpu(pascal, 0) or {}).get("index") == 2, SP.helper_gpu(pascal, 0))
check("only a Pascal card: none", SP.helper_gpu(pascal[:2], 0) is None)
twins = [{"index": 0, "name": "RTX 3090", "mib": 24576, "lanes": 16}, {"index": 1, "name": "RTX 3060", "mib": 12288, "lanes": 1},
         {"index": 2, "name": "RTX 3060", "mib": 12288, "lanes": 4}]
check("two equal cards: the wider link", (SP.helper_gpu(twins, 0) or {}).get("index") == 2, SP.helper_gpu(twins, 0))
check("no cc read: the card counts", (SP.helper_gpu([pascal[0], {"index": 1, "name": "x", "mib": 12288}], 0) or {}).get("index") == 1)
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
SP.save_settings({**SP.load_settings(), "pending": "old.gguf"})      # hand-edited: not an object
check("a malformed pending is dropped, not a traceback", not SP.rollback() and "pending" not in SP.load_settings())

# the first start's picker, as a user answers it
env = {**os.environ, "DS41_MODELS": os.path.join(tmp, "m2"), "DS41_RAM_GB": "115"}
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="1\n", capture_output=True, text=True, env=env)
check("--first: the default exits 10", p.returncode == 10 and "recommended" in p.stdout, (p.returncode, p.stdout[-300:], p.stderr[-300:]))
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="7\n", capture_output=True, text=True, env=env)
check("--first: the quality picks are offered", "UD-Q2_K_XL" in p.stdout and "UD-IQ3_XXS (most exact)" in p.stdout, p.stdout[-300:])
check("--first: none explains and exits 11", p.returncode == 11 and "custom" in p.stdout, (p.returncode, p.stdout[-300:], p.stderr[-300:]))
p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="x\n9\n1\n", capture_output=True, text=True, env=env)
check("--first: bad answers ask again", p.returncode == 10 and p.stdout.count("Please type one of the numbers") == 2,
      (p.returncode, p.stdout[-300:]))
# UD-IQ2_XXS is the recommended row (and Enter's) from 110 GB up, UD-IQ1_M below - checked on the menu, without downloading it
for ram, want in (("115", "UD-IQ2_XXS"), ("96", "UD-IQ1_M"), ("64", "UD-IQ1_M"), ("192", "UD-IQ2_XXS")):
    p = subprocess.run([sys.executable, str(HERE / "pick_model.py"), "--first"], input="7\n", capture_output=True, text=True,
                       env={**env, "DS41_RAM_GB": ram})
    rec = [l for l in p.stdout.splitlines() if "<- recommended" in l]
    check(f"--first at {ram} GB: {want} is the recommended row and Enter's",
          len(rec) == 1 and want in rec[0] and f"Enter = {rec[0].split('.')[0].strip()}" in p.stdout, p.stdout[-400:])

print("all passed" if not FAILS else f"FAILED {len(FAILS)}")
sys.exit(1 if FAILS else 0)
