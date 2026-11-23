# Models and quality

[Back to the README](../../README.md)

**`Choose Model.bat`** (or the chat page's Setup tab) switches between the files below.  The first start picks
**Unsloth UD-IQ2_XXS** from 110 GB of RAM (route bias 0.28) and **Unsloth UD-IQ1_M** below (route bias 0.10).

## Quality

How often the top choice matches DeepSeek's own API, on the same prompts:

| Model | Routing | General | Code + maths | Agentic |
|---|---|---|---|---|
| **Unsloth UD-IQ2_XXS (the pick from 110 GB)** | **Route bias 0.28, as shipped** | **91.1%** | **94.5%** | **95.9%** |
| Unsloth UD-IQ2_XXS | Exact routing | 93.1% | 95.9% | 96.5% |
| Unsloth UD-IQ1_M (the pick under 110 GB) | Route bias 0.10, as shipped there | 89.9-90.0% (64 / 96 GB) | - | - |
| ds4 Q2 | Route bias 0.28, as shipped | 89.9% | 93.1% | 95.1% |
| ds4 Q2 | Route bias 0.10 | 91.8% | 92.8% | - |
| ds4 Q2 | Exact routing | 91.5% | 93.0% | 96.1% |

The UD-IQ2_XXS row is at 192 GB; with less RAM the bias costs more (see "Agreement by RAM size" below).  Every model
is in the tables under "Other quants".

- **Want more agreement?** Use the chat page's Setup tab: **Quality** (route bias 0.10) or **Most exact** (0); each
  choice shows its measured agreement and speed for your model at your RAM.  `DS41_ROUTE_BIAS` in `ds41-paths.bat`
  overrides the switch.  On UD-IQ2_XXS (0.95): 0.10 gives 92.5 / 91.9 / 91.6 / 90.1% at 192 / 128 / 96 / 64 GB for
  36.5 / 37.0 / 35.4 / 28.9 tok/s; exact routing 93.1% everywhere for 32.5 / 26.0 / 20.0 / 11.5 tok/s.  (ds4 Q2 at 0.10,
  0.93: 34 instead of 41 tok/s at 192 GB, 12-26% slower at 96 / 64 GB.)  The startup
  line shows the bias in use.
- **Exact routing for answers only (0.94, opt-in):** `set STRATA_ANSWER_ROUTE_BIAS=0` the same way keeps the shipped
  bias while the model thinks and routes exactly in the answer after it (or give another bias, such as `0.10`).  It
  is off by default: it cost 10-13% of writing speed in agent sessions, and its quality gain is not measured yet.
- Route bias prefers experts that are already on the GPU, so fewer have to come from RAM: exact routing is about 25% slower at
  192 GB (0.95: 32.5 vs 43.7 tok/s) and much slower on smaller PCs (11.5 vs 36.2 at 64 GB).  It follows the GPU's experts as they change during an answer
  (`STRATA_BIAS_REFRESH=1` in the launcher), which made it both faster and closer to DeepSeek's own model (at 0.3:
  89.3% / 92.7% with the bias fixed at the start of each answer, 89.8-90.0% / 92.9% with the refresh).  The test moves
  by about 0.2 points between identical runs.
- On agentic, tool-calling text (53 transcripts with tools, tool calls and tool results, 46K tokens) the bias costs the
  same as on prose: on UD-IQ2_XXS exact routing agrees with DeepSeek's own model on 96.48% of next
  tokens, bias 0.28 on 95.93% (one run each, about 0.15 points between identical runs).  The tool-call syntax itself
  costs nothing; the cost is in the argument values and the prose around the calls.  There is no collapse at this
  setting (reports of collapse come from a bias of 1.0, which this launcher does not use).
- `--kv-fp4` (the compressed KV cache, on by default) is exact: the logits are bit-identical with and without it.
- For exact routing pick **Most exact** in the Setup tab or set `DS41_ROUTE_BIAS=0` (both set `--route-bias-file` to 0
  too), or per request with `"strata_tune": {"route_bias": 0, "route_bias_file": 0}`.  Setting `--route-bias` and
  `--route-bias-file` to 0 in `tools/v41/serve/ds41-windows.json` still works, but only while the switch is on Fast and
  the PC has 110 GB or more: the switch, `DS41_ROUTE_BIAS` and the under-110 GB default override the config.

## Agreement by RAM size (0.95)

- Agreement with DeepSeek's own model (general text, the shipped route bias): UD-IQ2_XXS 91.1% at 192 GB and 90.5%
  at 128 GB, 89.5% at 96 GB and 87.8% at 64 GB; UD-IQ1_M 89.4% and UD-IQ1_S 89.1% at 192 GB; at bias 0.10 UD-IQ1_S
  89.6% at 96 GB and 88.8% at 64 GB, UD-IQ1_M 90.0% and 89.9%.  It dips on smaller PCs because the
  route bias picks experts that are already loaded more often when fewer of them fit; with exact routing a model agrees
  the same at any RAM size (UD-IQ2_XXS 93.1%, UD-IQ1_M 91.2%, UD-IQ1_S 90.0%).
- **UD-IQ2_XXS from 110 GB of RAM up, UD-IQ1_M below.**  On 128 / 192 GB the 1-bit files write faster at about two
  points less agreement; on 64 / 96 GB UD-IQ1_M at bias 0.10 agrees more than UD-IQ2_XXS at 0.28 (89.9 / 90.0% vs
  87.8 / 89.5%) at its speed, and UD-IQ1_S at 0.10 beats it on both counts (88.8 / 89.6%, 10-13% faster at the default
  sampling).  `Choose Model.bat` downloads any of them.  New prompts take
  longer on small PCs, because the experts that do not fit are read from the SSD as they are needed.

## Other quants: Unsloth, ISTA-DASLab's GSQ-RCO, EXL3

Besides the ds4 Q2 GGUF, the engine runs the two other most-downloaded quant families of 1017, and EXL3 experts.  Some
of them read their dense layers or Engram tables from the ds4 Q2 GGUF (the first default), so keep that file for them.
**`Choose Model.bat`** lists the models in your model folder and the ones you can add, builds or downloads the one you
pick and saves your choice (`model-choice.txt`); the next start loads it.

- **ISTA-DASLab's GSQ-RCO** ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)):
  IQ2_XS, IQ2_XXS and IQ1_M.  A script builds each one into the engine's layout: the ds4 Q2 GGUF's dense weights and
  metadata with GSQ's routed experts, streamed from Hugging Face (every shard SHA-256-checked; 20-30 min at 100 MB/s).
  They read their Engram tables from the ds4 Q2 GGUF, so keep it.  By hand:
  `.venv\Scripts\python tools\v41\gsq_stream_merge.py --ours models\DeepSeek-V4.1-Flash-1017-Q2.gguf --build IQ2_XS --out models\ds41-gsq-iq2_xs.gguf`
  (`--build IQ2_XXS` or `IQ1_M` for the others), then put the file's name in `model-choice.txt`.
- **Unsloth** ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)):
  UD-IQ1_S to UD-Q8_K_XL, downloaded as they are into `models\unsloth\` (every file SHA-256-checked).  They carry their
  own Engram tables (IQ4_NL or Q5_K - on UD-IQ1_S as good as DeepSeek's exact ones: 89.3 vs 89.0% -, DeepSeek's
  exact FP8 from UD-Q3_K_XL), so they need no other file - except UD-IQ3_XXS, whose Q6_K tables the engine does not
  read: it uses the ds4 Q2 GGUF's, so keep that file for it (Choose Model refuses it without).  To install another
  Unsloth quant, pick it when the first start asks which model to use, or later in `Choose Model.bat`, then start
  `Start DeepSeek.bat` again.
- **The quality mode: Unsloth UD-Q2_K_XL** (Choose Model, 335 GB): 92.5% / 95.3% agreement at the shipped bias at 31 tok/s
  on 192 GB of RAM (0.93) (95.0% / 96.8% exact routing, more than EXL3 2.5 bpw's 94.4 / 95.6) and ~25% faster than EXL3.  Unsloth
  UD-IQ3_XXS (395 GB + the ds4 Q2 file) is the most exact, 92.7% / 95.9% at 27.1 tok/s (0.95); UD-Q2_K_XL stays the faster quality pick.  On 192 GB the launcher gives files with
  experts this big a 158 GiB arena (20 GiB of RAM kept back instead of 28: close other big programs).  **The default (from
  110 GB of RAM; UD-IQ1_M below): Unsloth UD-IQ2_XXS** (301 GB): 91.1% / 94.5% - about a point more than the ds4 Q2 GGUF on general text and on
  code and maths - and about as fast (0.93: 41.1 vs 40.7 tok/s greedy, 39.7 vs 40.3 at the default sampling).
  UD-Q2_K_XL and UD-IQ2_XXS: measured on revision 60ad3651 (Unsloth's files of 10-20; the re-made UD-Q2_K_XL tied them in our test), the one Choose Model
  downloads (UD-IQ1_S and UD-IQ1_M: 8a6ec9cf, the files measured here; UD-IQ1_S's re-made files agree 1.3-1.6 points more than the older ones at the same size; re-download
  to get them), with the ds4 Q2 file's exact FP8 Engram tables.  They carry their own (Q5_K, in the last file), which
  the engine reads: as good as the FP8 ones (UD-IQ2_XXS 93.1 vs 92.9% top-1, the same speed), so they need no other
  file.
- **EXL3 2.5 bpw** ([bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw)):
  routed experts in exllamav3's trellis format (EXL3, 2.5 bits a weight), run by the engine's own EXL3 kernels on the
  GPU and CPU, with the ds4 Q2 file's dense layers and Engram tables (keep it).  Downloaded as they are into
  `models\exl3\` (185 GB, every shard SHA-256-checked).  It agrees more with DeepSeek's own model than the ds4 Q2 file
  (94.4% / 95.6% vs 91.5 / 93.0%) and **writes slower: 25 vs 41 tok/s** on 192 GB of RAM - its experts take
  more GPU time each, and 159 GiB of them do not all fit the 140 GiB arena, so a few are read from the drive as it
  writes.  Keep `models\exl3\` on an **internal NVMe** drive: from a USB drive it wrote as fast, but new prompts read
  ~30% slower and the first load (140 GiB into RAM) took ~6.5 min.  Not measured with less than 192 GB of RAM; there,
  most of its experts would come from the drive.  By hand: put
  `exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw` in `model-choice.txt`, or set `DS41_EXL3` to the folder.

Measured here at route bias 0.28 (what ships from 110 GB of RAM; under 110 GB the 1-bit files run at 0.10 - see
"Agreement by RAM size"), top-1 agreement with DeepSeek's own model (teacher-forced, every token, at 192 GB), and
writing speed through the server, greedy: 0.95 where it says so, else 0.93 (0.95 is faster on every file it measured):

| | download | general | code + maths | agentic | writing, 192 / 128 / 96 / 64 GB RAM |
|---|---|---|---|---|---|
| ds4 Q2 | 366 GB | 89.9% | 93.1% | 95.1% | 41 / 36 / 27 / 20 tok/s (0.93) |
| Unsloth UD-IQ1_S | 233 GB | 89.1% | 91.5% | 95.3% | 49.8 / 48.9 / 48.5 / 45.0 tok/s (0.95; at 0.10 as shipped under 110 GB: 43.9 / 40.3 at 96 / 64) |
| Unsloth UD-IQ1_M | 250 GB | 89.4% | - | - | 48.1 / 46.6 / 46.2 / 41.5 tok/s (0.95; at 0.10 as shipped under 110 GB: 40.2 / 35.2 at 96 / 64) |
| **Unsloth UD-IQ2_XXS** (the pick from 110 GB) | 301 GB | **91.1%** | **94.5%** | **95.9%** | 44.6 / 43.7 / 41.6 / 36.2 tok/s (0.95) |
| **Unsloth UD-Q2_K_XL** (the quality mode) | 335 GB | **92.5%** | **95.3%** | **97.0%** | 31 / 27 / 23 / 15 tok/s (0.93) |
| Unsloth UD-IQ3_XXS | 395 GB + the ds4 Q2 file | **92.7%** | **95.9%** | **97.4%** | 27.1 / 25.8 / 23.2 / 18.7 tok/s (0.95) |

Two hosts of the full model agreed with each other on 98.3% and 98.9% of the same tokens.  Exact routing, what each file
reaches with no route bias (slower):

| | download | general | code + maths | writing at the shipped bias, 192 / 128 GB RAM (0.93 unless marked) | notes |
|---|---|---|---|---|---|
| ds4 Q2 | 366 GB | 91.5% | 93.0% | 41 / 36 tok/s | DeepSeek's exact Engram tables |
| GSQ IQ2_XS | 159 GB + the ds4 Q2 file | 90.0% | 92.9% | 38 / 26 tok/s | answered our task questions as well as the ds4 Q2 file |
| GSQ IQ2_XXS | 142 GB + the ds4 Q2 file | 87.8% | 89.5% | 41 / 32 tok/s | fewer knowledge questions right; long prompts slower |
| GSQ IQ1_M | 125 GB + the ds4 Q2 file | 85.5% | - | 42 / 37.5 tok/s | long prompts slower (no fast prompt kernel for IQ1_M) |
| Unsloth UD-IQ1_S | 233 GB | 90.0% | - | 50 / 49 tok/s (0.95) | faster on small PCs (0.95, bias 0.10: 40 tok/s at 64 GB) |
| Unsloth UD-IQ1_S (the older files) | 233 GB | 89.3% | 91.1% | 40 / 37.5 tok/s | GSQ IQ1_M's speed with 4 points more agreement |
| Unsloth UD-IQ1_M | 250 GB | 91.2% | - | 48 / 47 tok/s (0.95) | the pick under 110 GB (0.95, bias 0.10: 35 tok/s, 89.9% at 64 GB) |
| **Unsloth UD-IQ2_XXS** (the pick from 110 GB) | 301 GB | **93.1%** | **95.9%** | 41 / 34 tok/s | +1.6 points general, +3 on code and maths; about as fast as ds4 Q2 (39.7 vs 40.3 at the default sampling) |
| **Unsloth UD-Q2_K_XL** (the quality mode) | 335 GB | **95.0%** | **96.8%** | 31 / - tok/s | the faster quality pick; ~25% faster than EXL3 2.5 |
| Unsloth UD-IQ3_XXS | 395 GB + the ds4 Q2 file | **95.5%** | **97.6%** | 28 / - tok/s | the most exact measured here; reads the ds4 Q2 file's Engram tables |
| Unsloth UD-IQ3_S ... UD-Q8_K_XL | 424-508 GB | - | - | - | load and run (every expert type checked on real weights); not measured |
| EXL3 2.5 bpw | 185 GB + the ds4 Q2 file | 94.4% | 95.6% | 25 / - tok/s | below UD-Q2_K_XL on both counts; 192 GB of RAM and an internal NVMe |

On 192 GB of RAM (0.95) the 1-bit files write fastest (UD-IQ1_S 49.8, UD-IQ1_M 48.1 tok/s) at about two points less
agreement than UD-IQ2_XXS (44.6 tok/s), which agrees best of the fast files.  UD-Q2_K_XL agrees more, about 21% slower
than the ds4 Q2 GGUF (0.93: 32.2 vs 40.7 tok/s, greedy), and UD-IQ3_XXS is the most exact (92.7% / 95.9% at the
shipped bias) at 27.1 tok/s (0.95).
Speed falls once a model's experts (its file without the Engram table) outgrow the RAM (~150-160 GB of them fit at
192 GB): the rest is read from the SSD on every token.  So on 192 GB the Unsloth quants above
UD-IQ2_XXS (UD-Q2_K_XL: ~190 GB of experts) are slower (above UD-IQ3_XXS: not measured yet) - in exchange UD-Q4_K_XL /
Q8_K_XL keep DeepSeek's own MXFP4 experts.  With 0.93, on 128 GB the smaller files wrote fastest (more of their experts
fit in RAM): Unsloth UD-IQ1_S 41 and GSQ IQ1_M 37.5 vs 36 tok/s for ds4 Q2 - UD-IQ1_S at about a point less agreement than
ds4 Q2, IQ1_M at about 6 (exact routing).  UD-IQ1_S also reads long prompts faster than ds4 Q2 (a 16K-token prompt at ~655 vs ~580
tok/s) and loaded in 25 s vs ~60 s here.  Checked
against ISTA's files as of 2026-10-25: the merged experts are byte-identical to theirs; their norms, router and router
bias are bit-identical to the ds4 Q2 file's (GSQ-RCO only changes the routed experts).  Unsloth's Engram hash constants
(token map, primes, multipliers) are identical to the ds4 Q2 file's.
