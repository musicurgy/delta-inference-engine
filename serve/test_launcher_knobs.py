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
          "STRATA_DSPARK_SAMPLE_T", "STRATA_SINK_REJECT", "STRATA_DSPARK_EAGER", "STRATA_SNAP_TAIL",
          "STRATA_LOOKUP_YIELD", "STRATA_DSPARK_LGATE")

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
                 "STRATA_SHEXP_SPLIT", "STRATA_PASS_GRAPHS", "STRATA_D3_DMA_STREAM", "STRATA_D3_DMA_EARLY",
                 "STRATA_PRED_ROUTER",
                 "STRATA_ENGRAM_HOT", "STRATA_POOL_TASKS", "STRATA_ENGRAM_ASYNC", "STRATA_ENGRAM_DIRECT",
                 "STRATA_DENSE_ASYNC", "STRATA_PASS_NOSYNC", "STRATA_D3_PF_ASYNC", "STRATA_D3_SPEC",
                 "STRATA_CHUNK_CPU", "STRATA_CHUNK_CPU_BIG", "STRATA_CHUNK_CPU_MAXN", "STRATA_ENGRAM_THREADS",
                 "STRATA_KGATHER", "STRATA_D3_PF_NOISYOR", "STRATA_BIAS_REFRESH", "DS41_PRIORITY", "DS41_MODELS_W",
                 "STRATA_DS41_HOST_PIN",    # Windows executor-thread pin, measured only through the launcher
                 "STRATA_FILE_TIER_UNMAP",  # unmapped SSD reads, measured on Windows only
                 # 0.95 decode kernel switches, measured only through the launcher on UD-IQ2_XXS
                 "STRATA_ZC_PRED", "STRATA_FUSE", "STRATA_X3", "STRATA_FOLDS",
                 "STRATA_DSPARK_GATE",      # replaces the CONF / CUM stop, which the Q2_K sidecar keeps on purpose
                 "STRATA_KEEP_FIX_GIB",     # sized with the launcher's prestage 0 / prompt batch 16384
                 "STRATA_SERVE_SEED"}   # server-side (serve/server.py); the launcher writes its default 0


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

    def test_random_seed_stays_off_in_the_launcher(self):
        """The per-request random seed ships off for now: a fresh start writes the same sampled replies as 0.94.
        Turning STRATA_SERVE_SEED on here is a deliberate change to that, made together with this test."""
        self.assertEqual(launcher().get("STRATA_SERVE_SEED"), "0")

    def test_095_defaults_on_and_overridable(self):
        """0.95's defaults are on in the launcher, and each only `if not defined`: 'set NAME=0' before starting turns
        it off (the engine reads 0 as off for each switch; STRATA_KEEP_FIX_GIB is a size: 4.3 is the old reserve)."""
        want = {"STRATA_SINK_REJECT": "85000", "STRATA_DSPARK_EAGER": "1", "STRATA_D3_DMA_EARLY": "1",
                "STRATA_SNAP_TAIL": "1024", "STRATA_ZC_PRED": "1", "STRATA_FUSE": "1", "STRATA_X3": "1",
                "STRATA_FOLDS": "1", "STRATA_DSPARK_GATE": "1", "STRATA_LOOKUP_YIELD": "1",
                "STRATA_KEEP_FIX_GIB": "2.8"}
        ref = launcher()
        text = (ROOT / "Start DeepSeek.bat").read_text(encoding="utf-8")
        for k, v in want.items():
            self.assertEqual(ref.get(k), v, k)
            self.assertIn(f"if not defined {k} set {k}={v}", text, k)
            code = [ln for ln in text.splitlines() if not ln.lstrip().lower().startswith("rem")]
            self.assertEqual(sum(bool(re.search(rf"\bset \"?{k}=", ln)) for ln in code), 1,
                             f"{k}: set once, only if not defined")

    def test_opt_in_switches_stay_off(self):
        """Measured, but not defaults: CPU_SHARE_FIRST changes the text, DROP_MISS / EXPERTS_USED_DECODE drop
        experts (quality under evaluation), the rest showed no gain.  None of them is set by any launcher or config."""
        off = ("STRATA_CPU_SHARE_FIRST", "STRATA_EXPERTS_DROP_MISS", "STRATA_EXPERTS_USED_DECODE", "STRATA_X3B",
               "STRATA_GATEUP_FUSE", "STRATA_QFUSE", "STRATA_PROMPT_FUSED", "STRATA_SHARED_PREFIX", "STRATA_GPU_TIER_STATS",
               "STRATA_CPU_Q8_1", "STRATA_IDLE_PREFETCH", "STRATA_X3B_SKIP_ENGRAM")
        sh = script()
        for k in off:
            self.assertNotIn(k, launcher(), f"Start DeepSeek.bat sets {k}")
            self.assertNotIn(k, sh, f"ds41-engine.sh sets {k}")
            for p in sorted(SERVE.glob("strata-ds41*-win.json")):
                env = json.loads(p.read_text(encoding="utf-8")).get("env") or {}
                self.assertNotIn(k, env, f"{p.name} sets {k}")

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
