# Delta Inference Engine

**DeepSeek-V4.1-Flash on one NVIDIA GPU, natively on Windows.**

Delta runs all 552 billion parameters of **DeepSeek-V4.1-Flash-1017** (2-bit experts) on one
consumer PC, natively on Windows. It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` with up to 1M tokens
of context.

This is a Windows port of [Xolotl](https://github.com/Indras-Mirror/Xolotl), which is built on
[Strata](https://github.com/Niko1221/Strata). Upstream targets Linux; its README is
[README.xolotl.md](README.xolotl.md).

## Quick start

**Which download?** Just this repository: **Code > Download ZIP** (or `git clone`). You do not need any of the
`build-ds41-*.zip` files on the Releases page - `Start DeepSeek.bat` downloads the right engine for your CPU by
itself.  (Only if you set it up by hand: almost every PC wants `build-ds41-portable.zip`; the two `compat` zips are for
CPUs from before 2013.)

1. Get this folder (**Code > Download ZIP**, or
   `git clone -c core.longpaths=true https://github.com/musicurgy/delta-inference-engine C:\AI\Delta`) somewhere with
   a short path, and check "What you need" below: an NVIDIA RTX card with 12 GB+ (24 GB is fastest), 64 GB+ of RAM (128 GB+ recommended),
   ~320 GB free on a fast NVMe drive for the default model (its speed matters - see below).
2. Double-click **`Start DeepSeek.bat`**. The first time, it sets up everything that is missing, step by step, and
   asks before anything big:
   - Python and its packages (no Python 3.10-3.12 on the PC: it offers to install python.org's Python 3.11 with
     Windows' own winget, for your Windows user only),
   - the engine: the prebuilt one for your CPU from the
     [Releases page](https://github.com/musicurgy/delta-inference-engine/releases) (under 1 GB, no compiler needed) - or it builds
     it, if you have Visual Studio Build Tools, the CUDA Toolkit and Git,
   - the model: it asks which one you want and marks the one that suits your RAM, then downloads it from Hugging Face
     (233-395 GB, checked against the published SHA-256; it checks the free space first; resumable - just start it
     again if it stops).
3. Then the server starts and the chat page opens in your browser. Apps use `http://127.0.0.1:8080/v1` (OpenAI) or
   `/v1/messages` (Anthropic), any API key.  (Another port: `DS41_PORT` in `ds41-paths.bat`.)

Every later start: double-click `Start DeepSeek.bat` again - it finds everything in place and starts the server
(ready in about a minute). To run another quant (see "Other quants" below), double-click **`Choose Model.bat`**.
Everything else in this folder is the engine's source, upstream's files and tools - you do not need to open them.

**Updating:** a git clone: `git pull` in this folder (or double-click `UPDATE.bat`, which does that). A ZIP copy:
download the new ZIP and unpack it over this folder, or next to it with `models\` moved across. Then double-click
`Start DeepSeek.bat`: if the new version needs a new engine or new Python packages, it gets them first. Your model
files stay as they are.

**Something wrong?** Double-click **`Start DeepSeek (debug).bat`** instead. It checks the PC and the install (a
PASS / FAIL list), starts DeepSeek in a second window with everything it prints saved, and when you close that window
it packs it all into `logs\debug-<time>.zip`. Attach that zip to an issue with a line on what went wrong. It holds no
prompts or answers, and your Windows user name is replaced by `<user>`.

## What's new in 0.93

- **A model picker and a Setup tab.** The first start offers the model that suits your RAM, and the Setup tab in the
  chat page changes the model and the settings without a batch file or a text editor.
- **A new default from 128 GB of RAM up:** Unsloth's UD-IQ2_XXS, a 301 GB download. It writes 41 tok/s at 192 GB and
  34 at 128 GB, and got 57 of 99 hard questions right against 52 for the old default, ds4 Q2.
- **64 and 96 GB of RAM are supported.** The first start offers Unsloth's UD-IQ1_S there: 36 tok/s at 96 GB, 31 at 64 GB.
- **More models at full speed:** Unsloth's UD-Q2_K_XL and UD-IQ3_XXS for better quality, in `Choose Model.bat`.
- **A second GPU** can run a helper, a switch on the Setup tab: about 7% more writing speed with an RTX 3060 beside the 3090.
- **Built in public.** The release zips come from GitHub's own runners, each with a build attestation.

The full list is on the [Releases page](https://github.com/musicurgy/delta-inference-engine/releases).

## Speed

These numbers are from an RTX 3090 with an NVMe SSD, through the server:

| UD-IQ2_XXS (the default from 128 GB up) | 192 GB RAM | 128 GB RAM |
|---|---|---|
| Writing (6 chats, up to 1,500 tokens each, thinking on), greedy | **41 tok/s** | **34 tok/s** |
| Writing, the default sampling (temperature 1, top-p 0.95) | **40 tok/s** | - |
| Reading a long prompt (25-64k tokens) | **620-740 tok/s** | **470-600 tok/s** |

The 128 GB column is the same PC with the expert memory a 128 GB PC gets.  Greedy means temperature 0 in the
request; the default sampling is what a client gets when it sends no temperature.  Sampling accepts fewer of the
drafter's guesses, so it writes a little slower (ds4 Q2 40.7 -> 40.3 tok/s, UD-IQ2_XXS 41.1 -> 39.7).  0.93 narrowed
that gap from ~12%: the drafter's guesses are now
sampled instead of taken as its top pick (`STRATA_DSPARK_SAMPLE` in `Start DeepSeek.bat`); the answers'
distribution is unchanged.

**New in 0.93: from 128 GB of RAM up the default is Unsloth's UD-IQ2_XXS** (first start offers it on 120 GB or more; the
ds4 Q2 GGUF stays one answer away).  On 192 GB it writes about as fast as the ds4 Q2 GGUF - 41.1 vs 40.7 tok/s greedy, 39.7 vs
40.3 at the default sampling - and on 128 GB about 5% slower (34.0 vs 35.9 tok/s, greedy).  It agrees with DeepSeek's own model more often: 91.1% vs 89.9% on general text, 94.5%
vs 93.1% on code and maths at the settings that ship (below: "Other quants").

With less RAM, more of the model's experts are read from the SSD as they are needed. That is slower, most of all on
long prompts, but the output is the same.

### 96 GB and 64 GB of RAM

Measured on the same PC with the RAM a 96 or 64 GB PC would not have locked away (so neither Windows' file cache nor
the pagefile could use it). Writing speed through the server; agreement = how often the next token matches DeepSeek's
own model:

| Model | Agreement | 192 GB | 128 GB | **96 GB** | **64 GB** |
|---|---|---|---|---|---|
| **Unsloth UD-IQ2_XXS** (the default from 128 GB up) | 91.1% | 41 | 34 | 29 | 21 |
| ds4 Q2 | 89.9% | 41 | 36 | 27 | 20 |
| **Unsloth UD-IQ1_S** | 89.1% | 41 | 41 | **36** | **31** |
| Unsloth UD-Q2_K_XL | 92.5% | 31 | 27 | 23 | 15 |
| Unsloth UD-IQ3_XXS | 92.7% | 28 | 23 | 18 | 12 |
| GSQ IQ1_M (exact routing) | 85.5% | 42 | 37.5 | **34** | **28** |

- **Under ~110 GB of RAM, use Unsloth's UD-IQ1_S**: about a point less agreement than ds4 Q2 and much faster.  New prompts
  take longer than on 192 GB: UD-IQ1_S reads a ~180-token prompt in ~4.3 s at 96 GB and ~8.3 s at 64 GB. The first start offers
  UD-IQ1_S instead of the default on such a PC; `Choose Model.bat` downloads it on an existing install.
- ds4 Q2's 96 / 64 GB numbers are this release's launcher as it ships: the expert memory it gives such a PC
  (61 and 31 GiB; it was 51 and 21) and the SSD read-ahead (`STRATA_D3_SPEC=8`, which reads the experts the next
  layers will need while the current ones run - exact, +22% at 64 GB).  Before those two changes the same rows were
  22 and 16.  The SSD matters most here: everything was read from a PCIe 4.0 NVMe (~3.5 GiB/s).
- Below 56 GB of RAM it will probably not start.

On the 192 GB PC, a new 6k-token prompt takes about 15 s and a 25k-token prompt about 41 s (UD-IQ2_XXS). Writing is fastest on
maths and code (~48 and ~45 tok/s) and ~35 tok/s on plain prose; counting the prompt and the thinking, a client sees
~38 tok/s.

### A second GPU

The Setup tab in the chat page has a Multi-GPU switch: a helper on a second card of 8 GB or more holds experts the
main card has no room for.  The main card needs 20 GB or more for this switch; the engine itself runs from 12 GB.  With
an RTX 3060 12 GB in an x4 slot beside the 3090 it adds about 7% writing speed.  It does not split the model across cards, so it
does not scale: for two or more big cards use [Project Maya](https://github.com/mw00/project-maya), which runs
DeepSeek in tensor parallel.

## Windows settings for speed

Optional, about 15% faster.  `Start DeepSeek.bat` checks these at every start and names any that is off; they need
administrator rights, the BIOS or a restart, so it does not change them itself.  Measured on the RTX 3090 PC: the first
three together +12% writing speed (21.7 -> 24.3 tok/s at the time), the RAM speed another +3%.

1. **Power plan: High performance.** Win+R, `powercfg.cpl`, select High performance (under "Show additional plans" if
   it is hidden).  It stops CPU core parking and the PCIe link's power saving.
2. **Hardware-accelerated GPU scheduling: on.** Settings > System > Display > Graphics settings, turn it on, restart.
3. **"Lock pages in memory" for your account**: 2 MB pages for the ~140 GB of experts in RAM.  Win+R, `secpol.msc`,
   Local Policies > User Rights Assignment > "Lock pages in memory" > Add User or Group, type your user name, Check
   Names, OK; then sign out and in again.  (Windows Home has no `secpol.msc`; the engine then uses normal pages.)
4. **The RAM at its rated speed.** Boards often run four large sticks below their rating: set XMP / EXPO, or the speed
   by hand, in the BIOS (here four 48 GB DDR5 sticks went from 4000 to 5600 MT/s: +3%).  Check stability afterwards.
5. Optional: NVIDIA Control Panel > Manage 3D settings > Power management mode > Prefer maximum performance.
6. **A Microsoft Defender exclusion for the model folder and the engine.**  Windows Security > Virus & threat protection >
   Manage settings > Exclusions: add the `models\` folder (or wherever the model files are) and `ds41_generate.exe`.  Every
   read of an expert from the SSD otherwise passes through the scanner: measured here, expert-sized random reads from
   the model drive went from 5.35 to 6.43 GB/s (+20%), which is model loading on every PC and the writing speed on 64
   and 96 GB PCs, where part of the model is read from the SSD while it writes.

While it writes, a busy browser or game on the same PC can slow the CPU side down a lot; the engine already runs above
normal priority.

## Agents, chats and tools

- **Conversation cache:** the server keeps up to 8 conversations in memory (about 20 MB per 10k tokens). An agent that
  switches between conversations, or makes a side request between turns, resumes each one in a second or two instead
  of reading its whole prompt again (measured: 16-18 s per switch before, 1-3 s now; a 9-turn session alternating two
  ~8k-token conversations: 95 s before, 49 s now).
- **Follow-up messages** only read what is new. In thinking mode, the next turn no longer re-reads the history. A
  follow-up of a few hundred to 2,000 new tokens (a tool result, a pasted file) keeps most of the GPU's expert cache
  instead of emptying and refilling it, which saves 0.6-0.9 s per turn.
- **Tool calls** (OpenAI `tools`, including a forced `tool_choice`: `"required"` or a named function) and **JSON
  output** (`response_format` `json_object` / `json_schema`) work through the OpenAI-compatible API. Requests sent at
  the same time are answered one after another.
- **Pictures** are off by default. Double-click **`Enable Pictures.bat`** once, then start DeepSeek again: it downloads
  DeepSeek's vision encoder (0.97 GB, checked) and PyTorch's CPU build (124 MB). Then attach, drop or paste a picture
  in the chat page, or send one through the OpenAI (`image_url`) or Anthropic (`image`) API. The encoder runs on the
  CPU and takes no VRAM. On an i5-13600K a 1024x701 photo takes ~8 s to read and its first answer ~19 s; writing speed
  stays the same. To turn pictures off again, put `off` in `vision-choice.txt`.

## Privacy

- The server listens on this PC only (`127.0.0.1:8080`). To reach it from other devices, add `"host": "0.0.0.0"`
  AND `"api_key": "<a long secret>"` to `tools/v41/serve/ds41-windows.json` (clients send the key as their API key;
  without one, anyone on your network could use the model, and the server warns about it).
- Nothing from your prompts or answers is written to disk unless you turn it on: conversations live in memory and
  are gone when the server stops. Opt-in only: the plain-text watch log (`STRATA_TEXT_LOG=<a file path>`), the
  request monitor page (`"api_monitor": true` in `tools/v41/serve/ds41-windows.json`; memory only) and, in the
  built-in chat page, "Keep chats in this browser" (Settings; off = the chat lives in the tab only).

## Quality

How often the top choice matches DeepSeek's own API, on the same prompts:

| Model | Routing | General | Code + maths | Agentic |
|---|---|---|---|---|
| **Unsloth UD-IQ2_XXS (the default from 128 GB up)** | **Route bias 0.28, as shipped** | **91.1%** | **94.5%** | **95.9%** |
| Unsloth UD-IQ2_XXS | Exact routing | 92.5% | 95.9% | 96.5% |
| ds4 Q2 | Route bias 0.28, as shipped | 89.9% | 93.1% | 95.1% |
| ds4 Q2 | Route bias 0.10 | 91.8% | 92.8% | - |
| ds4 Q2 | Exact routing | 91.5% | 93.0% | 96.1% |

Every model is in the tables under "Other quants".

- **Want the most agreement?** Set `DS41_ROUTE_BIAS=0` in `ds41-paths.bat` for exact routing: +1.4 points on UD-IQ2_XXS,
  about 25% slower.  `0.10` costs about 18% of writing speed at 192 GB of RAM on ds4 Q2 (34 instead of 41 tok/s) and
  12-26% on 96 / 64 GB PCs; it has not been measured on UD-IQ2_XXS.  The startup line shows the bias in use.
- Route bias prefers experts that are already on the GPU, so fewer have to come from RAM: about 25% faster than exact
  routing at 192 GB, more on smaller PCs.  It follows the GPU's experts as they change during an answer
  (`STRATA_BIAS_REFRESH=1` in the launcher), which made it both faster and closer to DeepSeek's own model (at 0.3:
  89.3% / 92.7% with the bias fixed at the start of each answer, 89.8-90.0% / 92.9% with the refresh).  The test moves
  by about 0.2 points between identical runs.
- On agentic, tool-calling text (53 transcripts with tools, tool calls and tool results, 46K tokens) the bias costs the
  same as on prose: on ds4 Q2 exact routing agrees with DeepSeek's own model on 96.05% of next tokens, bias 0.28 on
  95.10%; inside the tool-call JSON itself the cost is 0.4 points, in the prose between calls 2.9.  There is no
  collapse at this setting (reports of collapse come from a bias of 1.0, which this launcher does not use).
- `--kv-fp4` (the compressed KV cache, on by default) is exact: the logits are bit-identical with and without it.
- For exact routing, set `--route-bias` and `--route-bias-file` to `0` in `tools/v41/serve/ds41-windows.json`, or
  per request with `"strata_tune": {"route_bias": 0, "route_bias_file": 0}` - about 25% slower.

## Other quants: Unsloth (the default from 128 GB up and the quality mode), ISTA-DASLab's GSQ-RCO, EXL3

Besides the ds4 Q2 GGUF, the engine runs the two other most-downloaded quant families of 1017, and EXL3 experts.  In
this section "the default" means the ds4 Q2 file, the first default, whose Engram tables some of these quants read.
**`Choose Model.bat`** lists the models in your model folder and the ones you can add, builds or downloads the one you
pick and saves your choice (`model-choice.txt`); the next start loads it.

- **ISTA-DASLab's GSQ-RCO** ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)):
  IQ2_XS, IQ2_XXS and IQ1_M.  A script builds each one into the engine's layout: the default file's dense weights and
  metadata with GSQ's routed experts, streamed from Hugging Face (every shard SHA-256-checked; 20-30 min at 100 MB/s).
  They read their Engram tables from the default file, so keep it.  By hand:
  `.venv\Scripts\python tools\v41\gsq_stream_merge.py --ours models\DeepSeek-V4.1-Flash-1017-Q2.gguf --build IQ2_XS --out models\ds41-gsq-iq2_xs.gguf`
  (`--build IQ2_XXS` or `IQ1_M` for the others), then put the file's name in `model-choice.txt`.
- **Unsloth** ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)):
  UD-IQ1_S to UD-Q8_K_XL, downloaded as they are into `models\unsloth\` (every file SHA-256-checked).  They carry their
  own Engram tables (IQ4_NL or Q5_K - on UD-IQ1_S as good as DeepSeek's exact ones: 89.3 vs 89.0% -, DeepSeek's
  exact FP8 from UD-Q3_K_XL), so they need no other file - except UD-IQ3_XXS, whose Q6_K tables the engine does not
  read: it uses the default's, so keep the default for it (Choose Model refuses it without).  To install an Unsloth
  quant INSTEAD of the default: answer N when `Start DeepSeek.bat` offers the default model's download, run
  `Choose Model.bat`, pick the download, then start `Start DeepSeek.bat` again.
- **The quality mode: Unsloth UD-Q2_K_XL** (Choose Model, 335 GB): 92.5% / 95.3% agreement at the shipped bias at 31 tok/s
  on 192 GB of RAM (95.0% / 96.8% exact routing, more than EXL3 2.5 bpw's 94.4 / 95.6) and ~25% faster than EXL3.  Unsloth
  UD-IQ3_XXS (395 GB + the default) is the most exact, 92.7% / 95.9% at 28 tok/s; UD-Q2_K_XL stays the faster quality pick.  On 192 GB the launcher gives files with
  experts this big a 158 GiB arena (20 GiB of RAM kept back instead of 28: close other big programs).  **The default from
  128 GB up: Unsloth UD-IQ2_XXS** (301 GB): 91.1% / 94.5% - about a point more than the ds4 Q2 GGUF on general text and on
  code and maths - and about as fast (41.1 vs 40.7 tok/s greedy, 39.7 vs 40.3 at the default sampling).
  UD-Q2_K_XL and UD-IQ2_XXS: measured on revision 60ad3651 (Unsloth's files of 10-20; the re-made UD-Q2_K_XL tied them in our test), the one Choose Model
  downloads (UD-IQ1_S: 8a6ec9cf, whose re-made files agree 1.3-1.6 points more than the older ones at the same size; re-download
  to get them), with the default model's exact FP8 Engram tables.  They carry their own (Q5_K, in the last file), which
  the engine reads: as good as the FP8 ones (UD-IQ2_XXS 93.1 vs 92.9% top-1, the same speed), so they need no other
  file.
- **EXL3 2.5 bpw** ([bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw)):
  routed experts in exllamav3's trellis format (EXL3, 2.5 bits a weight), run by the engine's own EXL3 kernels on the
  GPU and CPU, with the default file's dense layers and Engram tables (keep it).  Downloaded as they are into
  `models\exl3\` (185 GB, every shard SHA-256-checked).  It agrees more with DeepSeek's own model than the default
  (94.4% / 95.6% vs 91.5 / 93.0%) and **writes slower: 25 vs 41 tok/s** on 192 GB of RAM - its experts take
  more GPU time each, and 159 GiB of them do not all fit the 140 GiB arena, so a few are read from the drive as it
  writes.  Keep `models\exl3\` on an **internal NVMe** drive: from a USB drive it wrote as fast, but new prompts read
  ~30% slower and the first load (140 GiB into RAM) took ~6.5 min.  Not measured with less than 192 GB of RAM; there,
  most of its experts would come from the drive.  By hand: put
  `exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw` in `model-choice.txt`, or set `DS41_EXL3` to the folder.

Measured here at the settings that ship (route bias 0.28), top-1 agreement with DeepSeek's own model (teacher-forced,
every token), and writing speed through the server:

| | download | general | code + maths | agentic | writing, 192 / 128 / 96 / 64 GB RAM |
|---|---|---|---|---|---|
| ds4 Q2 | 366 GB | 89.9% | 93.1% | 95.1% | 41 / 36 / 27 / 20 tok/s |
| Unsloth UD-IQ1_S | 233 GB | 89.1% | 91.5% | 95.3% | 41 / 41 / 36 / 31 tok/s |
| **Unsloth UD-IQ2_XXS** (the default from 128 GB up) | 301 GB | **91.1%** | **94.5%** | **95.9%** | 41 / 34 / 29 / 21 tok/s |
| **Unsloth UD-Q2_K_XL** (the quality mode) | 335 GB | **92.5%** | **95.3%** | **97.0%** | 31 / 27 / 23 / 15 tok/s |
| Unsloth UD-IQ3_XXS | 395 GB + the default | **92.7%** | **95.9%** | **97.4%** | 28 / 23 / 18 / 12 tok/s |

Two hosts of the full model agreed with each other on 98.3% and 98.9% of the same tokens.  Exact routing, what each file
reaches with no route bias (slower):

| | download | general | code + maths | writing, 192 / 128 GB RAM | notes |
|---|---|---|---|---|---|
| ds4 Q2 | 366 GB | 91.5% | 93.0% | 41 / 36 tok/s | DeepSeek's exact Engram tables |
| GSQ IQ2_XS | 159 GB + the default | 90.0% | 92.9% | 38 / 26 tok/s | answered our task questions as well as the default |
| GSQ IQ2_XXS | 142 GB + the default | 87.8% | 89.5% | 41 / 32 tok/s | fewer knowledge questions right; long prompts slower |
| GSQ IQ1_M | 125 GB + the default | 85.5% | - | 42 / 37.5 tok/s | long prompts slower (no fast prompt kernel for IQ1_M) |
| Unsloth UD-IQ1_S (the older files) | 233 GB | 89.3% | 91.1% | 40 / 37.5 tok/s | the pick under 110 GB: IQ1_M's speed, 4 points more agreement |
| **Unsloth UD-IQ2_XXS** (the default from 128 GB up) | 301 GB | **92.5%** | **95.9%** | 41 / 34 tok/s | +1 point general, +3 on code and maths; about as fast as ds4 Q2 (39.7 vs 40.3 at the default sampling) |
| **Unsloth UD-Q2_K_XL** (the quality mode) | 335 GB | **95.0%** | **96.8%** | 31 / - tok/s | the faster quality pick; ~25% faster than EXL3 2.5 |
| Unsloth UD-IQ3_XXS | 395 GB + the default | **95.5%** | **97.6%** | 28 / - tok/s | the most exact measured here; reads the default's Engram tables |
| Unsloth UD-IQ1_M, UD-IQ3_S ... UD-Q8_K_XL | 250-508 GB | - | - | - | load and run (every expert type checked on real weights); not measured |
| EXL3 2.5 bpw | 185 GB + the default | 94.4% | 95.6% | 25 / - tok/s | below UD-Q2_K_XL on both counts; 192 GB of RAM and an internal NVMe |

On 192 GB of RAM ds4 Q2, UD-IQ2_XXS and UD-IQ1_S write about as fast as anything at the default sampling, and of those
UD-IQ2_XXS agrees best with DeepSeek.  UD-Q2_K_XL agrees more, about 21% slower than the ds4 Q2 GGUF (32.2 vs 40.7 tok/s,
greedy), and UD-IQ3_XXS is the most exact (92.7% / 95.9% at the shipped bias) at 28 tok/s.
Speed falls once a model's experts (its file without the Engram table) outgrow the RAM (~150-160 GB of them fit at
192 GB): the rest is read from the SSD on every token.  So on 192 GB the Unsloth quants above
UD-IQ2_XXS (UD-Q2_K_XL: ~190 GB of experts) are slower (above UD-IQ3_XXS: not measured yet) - in exchange UD-Q4_K_XL /
Q8_K_XL keep DeepSeek's own MXFP4 experts.  On 128 GB the smaller files write fastest (more of their experts fit in
RAM): Unsloth UD-IQ1_S 41 and GSQ IQ1_M 37.5 vs 36 tok/s for ds4 Q2 - UD-IQ1_S at about a point less agreement than
ds4 Q2, IQ1_M at about 6 (exact routing).  UD-IQ1_S also reads long prompts faster than ds4 Q2 (a 16K-token prompt at ~655 vs ~580
tok/s) and loaded in 25 s vs ~60 s here.  Checked
against ISTA's files as of 2026-10-25: the merged experts are byte-identical to theirs; their norms, router and router
bias are bit-identical to the default file's (GSQ-RCO only changes the routed experts).  Unsloth's Engram hash constants
(token map, primes, multipliers) are identical to the default file's.

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA RTX card with 12 GB of VRAM or more, and driver 580 or newer.  24 GB (RTX 3090 / 4090, or a 32 GB 5090)
  is the fast path; with less, fewer experts fit on the card.  Measured on the 3090 with the expert budget of a smaller
  card (the default model, 192 GB RAM): 24 GB 41 tok/s, 23 GB (a monitor on the card) 39, 16 GB 31, 12 GB 17.  The
  launcher adapts by itself: under 20 GB it pins at most 48 GiB of RAM for the card; under 14 GB it pins none and runs
  without the drafter (a real 12 GB RTX 3060 ran out of memory with them and answers without).  Below 12 GB it
  probably does not fit (~9 GB is fixed: dense weights, KV, buffers).  RTX 20 (Titan RTX, Quadro RTX): the engine is
  built for it, not tested here.  GTX 10 series (Pascal) and older cards cannot run it, whatever the driver: CUDA 13
  supports compute capability 7.5 and newer only.  `Start DeepSeek.bat` says so before downloading anything.
- 64 GB of RAM or more, 128 GB or more recommended.  Below 56 GB it will probably not start.
- An x86-64 CPU.  AVX2 (Intel from 2013 / Haswell on, any AMD Ryzen, Threadripper or EPYC) is the fast path; newer
  instructions (AVX-512, AVX-VNNI) are used where the CPU has them.  Older CPUs get an engine built for them, picked
  automatically: with AVX but not AVX2 (Sandy / Ivy Bridge Xeons) 38 tok/s, with neither (e.g. Westmere Xeons) 17 -
  measured on this PC with the newer kernels switched off, against 40 with AVX2.
- **A fast NVMe SSD - its speed matters.**  The model is read from it while it runs, not just at the start: the Engram
  tables (~190 GB of the file) for every new piece of text, and the experts that do not fit in RAM on every token.
  Measured on this PC, the same model on a PCIe 4.0 NVMe (Corsair MP600 PRO, 5.4 GB/s) against a nearly full system
  SSD (0.8 GB/s): it starts in about 1 minute instead of 4-5, reads a new 4K-token prompt ~40% faster (168 vs 120
  tok/s) and writes 6-9% faster.  With 128 GB of RAM, more experts come from the SSD, so it matters more.  Use a PCIe
  4.0 or 5.0 NVMe with room to spare (a nearly full SSD slows down) - not a SATA SSD, a hard disk or a USB drive.
  The model files take 233-395 GB: about 320 GB free for the default, UD-IQ2_XXS (a GSQ build adds its 125-159 GB).
- Python 3.10, 3.11 or 3.12 (3.11 recommended) - `Start DeepSeek.bat` offers to install it.
- To build (not needed with the prebuilt engine from the Releases page):
  - Visual Studio 2019 or 2022 (Build Tools are enough) with "Desktop development with C++";
  - the CUDA Toolkit 13.0;
  - Git.

## Prebuilt engine: built by CI

The three `build-ds41-*.zip` files on the Releases page are built on GitHub's public Windows runners by
[`.github/workflows/release.yml`](.github/workflows/release.yml), not on anyone's PC.  A version tag starts it; the
log of every build is public under [Actions](https://github.com/musicurgy/delta-inference-engine/actions/workflows/release.yml).
Each zip is what `build-ds41.bat portable` / `compat-avx` / `compat` builds (CUDA Toolkit 13.0.2 from NVIDIA's
SHA-256-checked archives, archs 75;86;89;120a), packed by `tools/v41/make_release.py`.  (Releases up to 0.92 were
built here by hand.)

`Start DeepSeek.bat` checks the zip it downloads against the release's `SHA256SUMS.txt` by itself.  To check one by
hand:

- **Checksum:** compare `certutil -hashfile build-ds41-portable.zip SHA256` (or `Get-FileHash build-ds41-portable.zip`
  in PowerShell) with the zip's line in `SHA256SUMS.txt`.
- **Attestation** (that the zip came from this repository's workflow, at which commit), with the
  [GitHub CLI](https://cli.github.com/):
  ```bat
  gh attestation verify build-ds41-portable.zip --repo musicurgy/delta-inference-engine
  ```
  It prints the workflow run and commit the zip was built from; a zip that was not built there fails.

## Setup by hand (what `Start DeepSeek.bat` does for you)

1. **Python environment**, in this folder:
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"
   ```
2. **Build** with `build-ds41.bat` (10-25 minutes the first time). It builds for this PC's CPU and GPU (on a CPU
   without AVX2, for that CPU); `build-ds41.bat portable` builds one for any AVX2 CPU and RTX 20, 30, 40 and 50 cards,
   to copy to other PCs; `build-ds41.bat compat-avx` one for CPUs with AVX but not AVX2, `build-ds41.bat compat` one
   for any x86-64 CPU (SSE4.2).  The launcher takes the first of `build-ds41`, `build-ds41-portable`,
   `build-ds41-compat-avx`, `build-ds41-compat` that this CPU can run.
3. **Model folder:** `DS41_MODELS` in `ds41-paths.bat`, on your fastest NVMe drive (default: `models\` here).
4. **Model files:** `setup-ds41-models.bat` for ds4 Q2, or `Choose Model.bat` for the others (resumable; every large file is checked against its published SHA-256):
   - the ds4 Q2 GGUF from
     [nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF)
     (366 GB): the model, and DeepSeek's Engram tables inline (byte for byte the official ones), which the engine
     reads straight from it;
   - the tokenizer, config and shard index from
     [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (a few MB);
   - the speculative-decoding drafter (the biggest single speed-up), built from DeepSeek's own MTP heads in the same
     repository (three shards, 7.9 GB, removed afterwards) and shrunk to 4.2 GB. The drafter only proposes tokens;
     the main model checks every one.
5. **Start** with `Start DeepSeek.bat`. Use the model name `/v1/models` shows (`deepseek-v4.1-flash-1017-ds4-q2` for
   ds4 Q2, `deepseek-v4.1-flash-1017-ud-iq2_xxs` for UD-IQ2_XXS; the server answers any name) with any
   OpenAI-compatible client.

## Settings

- **One setting file:** `Start DeepSeek.bat` sets everything. Each line in it carries a short note.
- **Paths:** they live in `ds41-paths.bat`.
- **Model:** the one the first start or `Choose Model.bat` picked (saved in `model-choice.txt`).
- **RAM:** the memory used for experts is sized from your installed RAM at every start: total minus 28 GiB, at most
  140 GiB (on 192 GB-class PCs: 150 for UD-IQ2_XXS, and 158 with 20 GiB kept back for files with bigger experts).
  `DS41_ARENA_GIB` or `DS41_RAM_RESERVE_GIB` in `ds41-paths.bat` override it.
- **The first minutes after a reboot are slower** (measured: ~13 instead of ~25 tok/s on a cold start): the model's
  Engram tables are read from the SSD until Windows has cached the rows your text uses. It speeds up by itself as
  you use it. An optional Engram hot set (`engram-hot-2g.bin` in the model folder) keeps the most-used rows in RAM
  from the start; build it with `ds41_engram_hot.exe` - the header of `tools/v41/engine/ds41_engram_hot.cpp` explains
  how.

## Credits

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) and their
  contributors: the engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): the tensor library.
- nazeshinjite: the ds4 Q2 GGUF.
- ISTA-DASLab: the GSQ-RCO quants.
- DeepSeek: the model, and the MTP heads the drafter is built from.
- [Project Maya](https://github.com/mw00/project-maya) and [antirez/ds4](https://github.com/antirez/ds4): server fixes
  ported in 0.93 (the end of thinking by its token, TCP_NODELAY; a blank replayed reasoning read as absent); the
  EXL3 CPU / CUDA kernels started from Project Maya's port of exllamav3's (MIT).

MIT license: [LICENSE](LICENSE).
