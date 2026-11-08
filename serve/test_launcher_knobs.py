"""The engine settings the launchers start DeepSeek V4.1 with: Start DeepSeek.bat (the Windows app) is the reference;
tools/v41/serve/ds41-engine.sh (Linux) and the manual Windows configs strata-ds41*-win.json (docs/WINDOWS.md) carry
the same serving switches, and differ only where listed below, on purpose.

    python -m unittest serve.test_launcher_knobs -v
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "tools" / "v41" / "serve"

# exact serving switches: the same everywhere
SHARED = ("STRATA_SERVE_STEP_MAX", "STRATA_SERVE_PASS_BIAS", "STRATA_SERVE_LOOKUP", "STRATA_SERVE_DSPARK_EVICT",
          "STRATA_ENC_BOUND", "STRATA_DENSE_REQUANT", "STRATA_SERVE_ROLLBACK", "STRATA_SERVE_BATCH_MIN",
          "STRATA_PASS_CAP_MIN", "STRATA_SERVE_REWIND1", "STRATA_SERVE_CACHE_SLOTS", "STRATA_SERVE_SNAP_STACK",
          "STRATA_SERVE_KEEP_CACHE", "STRATA_SERVE_KEEP_LRU", "STRATA_SERVE_KEEP_MAX", "STRATA_DSPARK_SAMPLE",
          "STRATA_DSPARK_SAMPLE_T")

# different on purpose in ds41-engine.sh / the -win.json configs (they run the GSQ model with its Q2_K MTP sidecar):
ON_PURPOSE = {
    "STRATA_DSPARK_DEPTH": "the Q2_K sidecar's depth 4 (the launcher's 5 was measured with the IQ2_XXS drafter)",
    "STRATA_DSPARK_CONF": "the Q2_K sidecar keeps the 0.6 confidence stop (the launcher: the product rule, CUM 0.25)",
    "STRATA_DSPARK_CUM": "see STRATA_DSPARK_CONF",
    "STRATA_DSPARK_KEEP": "the sidecar's 64 most-used experts (the launcher picks by VRAM / RAM)",
    "STRATA_PRESTAGE_GIB": "1.5 with --prompt-batch 12288 (measured together); the launcher: 0 with 16384",
    "STRATA_TIER_BATCH_PCIE": "0.75 tuned on the -win configs' 128 GB test PC; the launcher 0.5",
    "STRATA_SERVE_DSPARK": "a path",
    "STRATA_NO_LARGEPAGES": "4 KiB pages pin faster on Linux; the launcher pins with STRATA_ARENA_PIN_GIB",
}
# set by the launcher only: Windows-specific, sized per PC, or kernel / prefetch switches not measured on the GSQ setup
LAUNCHER_ONLY = {"STRATA_ARENA_PIN_GIB", "STRATA_TEXT_LOG", "STRATA_PCIE_PINNED_ONLY", "STRATA_BALANCE",
                 "STRATA_SHEXP_SPLIT", "STRATA_PASS_GRAPHS", "STRATA_D3_DMA_STREAM", "STRATA_PRED_ROUTER",
                 "STRATA_ENGRAM_HOT", "STRATA_POOL_TASKS", "STRATA_ENGRAM_ASYNC", "STRATA_ENGRAM_DIRECT",
                 "STRATA_DENSE_ASYNC", "STRATA_PASS_NOSYNC", "STRATA_D3_PF_ASYNC", "STRATA_D3_SPEC",
                 "STRATA_CHUNK_CPU", "STRATA_CHUNK_CPU_BIG", "STRATA_CHUNK_CPU_MAXN", "STRATA_ENGRAM_THREADS",
                 "STRATA_KGATHER", "STRATA_D3_PF_NOISYOR", "STRATA_BIAS_REFRESH", "DS41_PRIORITY", "DS41_MODELS_W"}


def launcher() -> dict:
    """Start DeepSeek.bat's values (the first unconditional `set`; `if not defined X set X=...` counts too)."""
    out = {}
    for line in (ROOT / "Start DeepSeek.bat").read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*(?:if not defined \w+ )?set "?((?:STRATA|DS41)_\w+)=([^"]*)"?\s*$', line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return out


def script() -> dict:
    """ds41-engine.sh's defaults: export X="${X:-v}" / "${X-v}"."""
    text = (SERVE / "ds41-engine.sh").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r'export ((?:STRATA|DS41)_\w+)="\$\{\1:?-([^}]*)\}"', text)}


class Knobs(unittest.TestCase):
    def test_every_launcher_setting_is_classified(self):
        unknown = set(launcher()) - set(SHARED) - set(ON_PURPOSE) - LAUNCHER_ONLY
        self.assertFalse(unknown, f"new in Start DeepSeek.bat: put each in SHARED, ON_PURPOSE or LAUNCHER_ONLY")

    def test_the_linux_script_matches(self):
        ref, sh = launcher(), script()
        for k in SHARED:
            self.assertEqual(sh.get(k), ref.get(k), f"ds41-engine.sh {k}")

    def test_the_windows_configs_match(self):
        ref = launcher()
        for p in sorted(SERVE.glob("strata-ds41*-win.json")):
            env = json.loads(p.read_text(encoding="utf-8")).get("env") or {}
            for k in SHARED:
                self.assertEqual(env.get(k), ref.get(k), f"{p.name} {k}")


if __name__ == "__main__":
    unittest.main()
