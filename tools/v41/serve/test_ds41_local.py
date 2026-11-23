"""ds41_local.prompt_profile: the prompt-speed settings by whether the routed experts fit the RAM arena.

    python tools/v41/serve/test_ds41_local.py      (no GPU, no model files)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds41_local as L  # noqa: E402


class PromptProfile(unittest.TestCase):
    def test_file_tier_wide_from_6144_and_read_ahead(self):   # 64 GB, UD-IQ1_S: arena 36 GiB
        env, line = L.prompt_profile(36.0, 120.5, {})
        self.assertEqual(env, {"STRATA_WIDE_PROMPT": "1", "STRATA_WIDE_PROMPT_MIN": "6144",
                               "STRATA_BM_PREFILL_READAHEAD": "1", "STRATA_ENGRAM_READAHEAD": "1"})
        self.assertIn("read-ahead", line)
        self.assertIn("file tier", line)

    def test_no_file_tier_wide_from_16384_no_read_ahead(self):   # 192 GB, UD-IQ2_XXS: arena 150 >= 146.1 GiB
        env, line = L.prompt_profile(150.0, 146.1, {})
        self.assertEqual(env, {"STRATA_WIDE_PROMPT": "1", "STRATA_WIDE_PROMPT_MIN": "16384"})
        self.assertIn("no file tier", line)
        self.assertIn("no read-ahead", line)

    def test_equal_sizes_count_as_fitting(self):
        self.assertNotIn("STRATA_BM_PREFILL_READAHEAD", L.prompt_profile(146.0, 146.0, {})[0])

    def test_unknown_size_takes_the_file_tier_profile(self):   # EXL3 experts, or an unreadable header
        self.assertEqual(L.prompt_profile(140.0, None, {})[0]["STRATA_BM_PREFILL_READAHEAD"], "1")
        self.assertEqual(L.prompt_profile(None, 146.1, {})[0]["STRATA_WIDE_PROMPT_MIN"], "6144")

    def test_the_users_own_values_are_kept(self):
        user = {"STRATA_BM_PREFILL_READAHEAD": "0", "STRATA_WIDE_PROMPT_MIN": "8192"}
        env, line = L.prompt_profile(36.0, 120.5, user)
        self.assertEqual(env, {"STRATA_WIDE_PROMPT": "1", "STRATA_ENGRAM_READAHEAD": "1"})   # only what the user did not set
        self.assertIn("yours kept: STRATA_WIDE_PROMPT_MIN=8192, STRATA_BM_PREFILL_READAHEAD=0", line)
        env, _ = L.prompt_profile(150.0, 146.1, {"STRATA_WIDE_PROMPT": "0"})
        self.assertEqual(env, {"STRATA_WIDE_PROMPT_MIN": "16384"})

    def test_the_launcher_no_longer_sets_them(self):
        bat = os.path.join(L.ROOT, "Start DeepSeek.bat")
        with open(bat, "rb") as f:
            text = f.read()
        for name in L.PROMPT_VARS:
            self.assertNotIn(b"\r\nset " + name.encode() + b"=", text)
        self.assertNotIn(b"\n", text.replace(b"\r\n", b""))   # still CRLF


class PromptBatch(unittest.TestCase):   # the prompt batch width on Windows
    ARGS = ["--arena-gib", "36", "--prompt-batch", "16384", "--rank", "x"]

    def test_sweep_default_is_off_until_the_md5_gate(self):
        self.assertEqual(L.PROMPT_SWEEP_PB, 0)
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {}), (None, None))

    def test_file_tier_wide_raises_when_asked(self):
        pb, line = L.prompt_batch(self.ARGS, True, True, False, {"STRATA_PROMPT_SWEEP_PB": "32768"})
        self.assertEqual(pb, 32768)
        self.assertIn("STRATA_PROMPT_SWEEP_PB=0", line)

    def test_a_raised_width_turns_on_the_counted_budget(self):
        self.assertEqual(L.sweep_fit_env(16384, 32768, {}), {"STRATA_PROMPT_SWEEP_FIT": "1",
                                                             "STRATA_PROMPT_SWEEP_SLACK_GIB": "0.3"})
        self.assertEqual(L.sweep_fit_env(16384, 32768, {"STRATA_PROMPT_SWEEP_FIT": "0"}), {})   # the user's kept
        self.assertEqual(L.sweep_fit_env(16384, 8192, {}), {})    # lowered: the old width rule
        self.assertEqual(L.sweep_fit_env(16384, 16384, {}), {})
        self.assertEqual(L.sweep_fit_env(16384, None, {}), {})
        self.assertEqual(L.sweep_fit_env(None, 32768, {}), {})

    def test_no_file_tier_keeps_16384(self):
        self.assertEqual(L.prompt_batch(self.ARGS, False, True, False, {}), (None, None))

    def test_not_wide_or_big_experts_keep_the_config(self):
        self.assertEqual(L.prompt_batch(self.ARGS, True, False, False, {}), (None, None))
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, True, {}), (None, None))

    def test_sweep_knob_sets_the_width_and_zero_turns_it_off(self):
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"STRATA_PROMPT_SWEEP_PB": "24576"})[0], 24576)
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"STRATA_PROMPT_SWEEP_PB": "0"}), (None, None))
        # it only raises: a smaller value leaves the config's
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"STRATA_PROMPT_SWEEP_PB": "8192"}), (None, None))

    def test_ds41_prompt_batch_wins_both_ways(self):
        for env in ({"DS41_PROMPT_BATCH": "32768"}, {"STRATA_PROMPT_BATCH": "32768"}):
            pb, line = L.prompt_batch(self.ARGS, False, False, True, env)   # no file tier, not wide, big experts
            self.assertEqual(pb, 32768)
            self.assertIn(next(iter(env)), line)
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"DS41_PROMPT_BATCH": "8192"})[0], 8192)
        both = {"DS41_PROMPT_BATCH": "24576", "STRATA_PROMPT_BATCH": "16384"}
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, both)[0], 24576)
        self.assertEqual(L.prompt_batch(["--rank", "x"], False, False, False, {"DS41_PROMPT_BATCH": "4096"})[0], 4096)

    def test_bad_values_stop_with_a_message(self):
        for v in ("abc", "-1", "1.5", "inf", "nan"):
            with self.assertRaises(SystemExit):
                L.prompt_batch(self.ARGS, True, True, False, {"DS41_PROMPT_BATCH": v})
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"DS41_PROMPT_BATCH": " "}), (None, None))
        self.assertEqual(L.prompt_batch(self.ARGS, True, True, False, {"DS41_PROMPT_BATCH": " ", "STRATA_PROMPT_SWEEP_PB": "32768"})[0], 32768)

    def test_file_tier_matches_the_profile(self):
        for arena, experts in ((36.0, 120.5), (150.0, 146.1), (146.0, 146.0), (None, 1.0), (1.0, None)):
            self.assertEqual(L.file_tier(arena, experts),
                             "STRATA_BM_PREFILL_READAHEAD" in L.prompt_profile(arena, experts, {})[0])


class BigExpertMargins(unittest.TestCase):   # f096: bigger experts' headroom for the prompt path only
    Q3 = "D:/models/unsloth/UD-IQ3_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00001-of-00007.gguf"
    Q2KXL = "D:/models/unsloth/UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf"

    def test_default_borrows_for_the_prompt_path(self):
        self.assertEqual(L.big_expert_margins(self.Q3, {}), (None, "3"))
        self.assertEqual(L.big_expert_margins(self.Q2KXL, {}), (None, "2"))

    def test_switch_restores_the_095_rule(self):
        self.assertEqual(L.big_expert_margins(self.Q3, {"STRATA_BIG_BORROW": "0"}), ("3", None))
        self.assertEqual(L.big_expert_margins(self.Q2KXL, {"STRATA_BIG_BORROW": " 0 "}), ("2", None))
        self.assertEqual(L.big_expert_margins(self.Q3, {"STRATA_BIG_BORROW": "1"}), (None, "3"))

    def test_the_users_margin_wins(self):
        for env in ({"DS41_VRAM_MARGIN": "2.5"}, {"DS41_VRAM_MARGIN": "2.5", "STRATA_BIG_BORROW": "0"}):
            self.assertEqual(L.big_expert_margins(self.Q3, env), (None, None))

    def test_the_engine_reads_the_prompt_margin_only_on_the_prompt_path(self):
        src = open(os.path.join(L.ROOT, "tools", "v41", "engine", "ds41_generate.cpp"), encoding="utf-8").read()
        self.assertIn('k == "--prompt-margin"', src)
        # the slot budgets (load, seed) still read --vram-margin; the borrow / batch width / sweep read prompt_margin()
        self.assertEqual(src.count("a.prompt_margin()"), 4)
        self.assertEqual(src.count("((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0"), 2)


class RenamedSwitches(unittest.TestCase):   # the 0.95-rc names still count, the new name wins
    def test_old_names_are_the_users(self):
        env, line = L.prompt_profile(150.0, 146.1, {"STRATA_D11_WIDE_PROMPT": "0"})
        self.assertEqual(env, {"STRATA_WIDE_PROMPT_MIN": "16384"})
        self.assertIn("yours kept: STRATA_WIDE_PROMPT=0", line)
        self.assertEqual(L.prompt_batch(PromptBatch.ARGS, True, True, False, {"STRATA_P3_SWEEP_PB": "24576"})[0], 24576)
        self.assertEqual(L.sweep_fit_env(16384, 32768, {"STRATA_P3_SWEEP_FIT": "0"}), {})

    def test_the_new_name_wins(self):
        env = L.with_new_names({"STRATA_D11_WIDE_MIN": "8192", "STRATA_WIDE_PROMPT_MIN": "4096"})
        self.assertEqual(env["STRATA_WIDE_PROMPT_MIN"], "4096")
        self.assertEqual(L.with_new_names({"STRATA_D11_WIDE_MIN": "8192"})["STRATA_WIDE_PROMPT_MIN"], "8192")


class DrafterKeep(unittest.TestCase):   # the drafter's expert counts from the checkpoint row's keep file
    ON = {"STRATA_SERVE_DSPARK": "d.gguf"}

    def test_a_row_with_keep(self):
        de = {"keep": "k.bin"}
        self.assertEqual(L.keep_values(de), ("k.bin:96,64,32", "k.bin:112,80,48"))
        self.assertEqual(L.drafter_keep(63.8, self.ON, de), "k.bin:96,64,32")
        self.assertEqual(L.drafter_keep(63.8, {**self.ON, "STRATA_DSPARK_KEEP": "k.bin:112,80,48"}, de), "k.bin:96,64,32")
        self.assertIsNone(L.drafter_keep(95.8, self.ON, de))

    def test_a_row_with_keep_null_sets_nothing(self):
        """keep null (a day-0 row whose counts are not measured yet): no "None:96,64,32" for the engine to open"""
        for de in ({"keep": None}, {"keep": ""}, {}):
            self.assertEqual(L.keep_values(de), (None, None))
            self.assertIsNone(L.drafter_keep(63.8, self.ON, de))
            self.assertIsNone(L.drafter_keep(63.8, {**self.ON, "STRATA_DSPARK_KEEP": ""}, de))
            self.assertIsNone(L.drafter_keep(63.8, {**self.ON, "STRATA_DSPARK_KEEP": "None:112,80,48"}, de))


class LowRamBias(unittest.TestCase):
    def test_1bit_files_under_110_gib_get_010(self):
        for f in ("D:/models/unsloth/UD-IQ1_S/DeepSeek-V4.1-Flash-UD-IQ1_S-00001-of-00005.gguf", "x-ud-iq1_m-00001-of-00006.gguf"):
            for ram in (62.9, 95.9, 109.4):
                self.assertEqual(L.low_ram_bias(f, ram), "0.10", (f, ram))

    def test_others_keep_the_config(self):
        self.assertEqual(L.low_ram_bias("x-UD-IQ1_S-00001-of-00005.gguf", 127.9), "")
        self.assertEqual(L.low_ram_bias("x-UD-IQ1_S-00001-of-00005.gguf", 109.6), "")   # the page rounds it to 110 too
        self.assertEqual(L.low_ram_bias("x-UD-IQ1_S-00001-of-00005.gguf", None), "")
        for f in ("x-UD-IQ2_XXS-00001-of-00007.gguf", "x-UD-IQ3_XXS-00001-of-00009.gguf", "ds41-q2.gguf", "ds41-gsq-iq1_m.gguf", ""):
            self.assertEqual(L.low_ram_bias(f, 63.9), "", f)


class SetupSwitch(unittest.TestCase):
    IQ1M = "x-UD-IQ1_M-00001-of-00005.gguf"
    IQ2 = "x-UD-IQ2_XXS-00001-of-00006.gguf"

    def test_quality_and_exact_on_every_pc(self):
        for f in (self.IQ1M, self.IQ2):
            for ram in (63.9, 191.9):
                self.assertEqual(L.route_bias_choice({"mode": "balanced"}, f, ram)[0], "0.10")
                self.assertEqual(L.route_bias_choice({"mode": "exact"}, f, ram)[0], "0")

    def test_chosen_fast_is_fast_on_a_small_pc(self):
        self.assertEqual(L.route_bias_choice({"mode": "fast", "mode_chosen": True}, self.IQ1M, 63.9)[0], "")

    def test_unchosen_switch_takes_the_pc_default(self):
        for s in ({}, {"mode": "fast"}, None):   # "fast" without mode_chosen: what pre-0.95 pages saved on every Apply
            self.assertEqual(L.route_bias_choice(s, self.IQ1M, 63.9)[0], "0.10")
            self.assertEqual(L.route_bias_choice(s, self.IQ1M, 127.9)[0], "")
            self.assertEqual(L.route_bias_choice(s, self.IQ2, 63.9)[0], "")


if __name__ == "__main__":
    unittest.main()
