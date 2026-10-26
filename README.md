# DeepSeek-V4.1-Flash on one NVIDIA GPU, natively on Windows

Runs all 552 billion parameters of **DeepSeek-V4.1-Flash-1017** (DeepSeek's official weights, 2-bit experts) on one
consumer PC, natively on Windows. It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` with up to 1M tokens
of context.

This is a Windows port of [Xolotl](https://github.com/Indras-Mirror/Xolotl), which is built on
[Strata](https://github.com/Niko1221/Strata). Upstream targets Linux; its README is
[README.xolotl.md](README.xolotl.md).

## Quick start

1. Get this folder (**Code > Download ZIP**, or `git clone -c core.longpaths=true <URL> C:\AI\DeepSeek`) somewhere with
   a short path, and check "What you need" below: an NVIDIA RTX card with 16 GB+ (24 GB is fastest), 128 GB+ of RAM,
   ~560 GB free on an NVMe drive.
2. Double-click **`Start DeepSeek.bat`**. The first time, it sets up everything that is missing, step by step, and
   asks before anything big:
   - Python and its packages (no Python 3.10-3.12 on the PC: it offers to install python.org's Python 3.11 with
     Windows' own winget, for your Windows user only),
   - the engine: the prebuilt one for your CPU from the Releases page (under 1 GB, no compiler needed) - or it builds
     it, if you have Visual Studio Build Tools, the CUDA Toolkit and Git,
   - the model files from Hugging Face (~565 GB of downloads, checked against their published SHA-256; it checks the
     free space first; resumable - just start it again if it stops).
3. Then the server starts and the chat page opens in your browser. Apps use `http://127.0.0.1:8080/v1` (OpenAI) or
   `/v1/messages` (Anthropic), any API key.

Every later start: double-click `Start DeepSeek.bat` again - it finds everything in place and starts the server
(ready in about a minute). Everything else in this folder is the engine's source, upstream's files and tools - you do
not need to open them.

## Speed

These numbers are from an RTX 3090 with an NVMe SSD, through the server:

| | 192 GB RAM | 128 GB RAM |
|---|---|---|
| Writing (6 chats, up to 1,500 tokens each, thinking on) | **41 tok/s** | **33 tok/s** |
| Reading a long prompt (25-64k tokens) | **610 tok/s** | **410 tok/s** |

The 128 GB column is the same PC with the expert memory a 128 GB PC gets.

With less RAM, more of the model's experts are read from the SSD as they are needed. That is slower, most of all on
long prompts, but the output is the same.

On the 192 GB PC, a new 6k-token prompt takes about 18 s and a 25k-token prompt about 42 s. Writing is fastest on
maths and code (~48 and ~45 tok/s) and ~35 tok/s on plain prose; counting the prompt and the thinking, a client sees
~39 tok/s.

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

| | General | Code + maths |
|---|---|---|
| Exact routing | 91.4% | 92.1% |
| Route bias 0.28 (the default) | 90.2% | 92.4% |

- Route bias prefers experts that are already on the GPU. It is roughly 20% faster. It follows the GPU's experts as
  they change during an answer (`STRATA_BIAS_REFRESH=1` in the launcher), which made it both faster and closer to
  DeepSeek's own model (at 0.3: 89.3% / 92.7% with the bias fixed at the start of each answer, 89.8-90.0% / 92.9% with
  the refresh).  The test moves by about 0.2 points between identical runs.
- For exact routing, set `--route-bias` and `--route-bias-file` to `0` in `tools/v41/serve/ds41-windows.json`.

## Other quants: ISTA-DASLab's GSQ-RCO

The IQ2_XS, IQ2_XXS and IQ1_M builds of
[ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
run in this engine. It reads its own file layout, so a script builds one: it keeps the default file's dense weights,
Engram and metadata and takes GSQ's routed experts, streamed from Hugging Face (every shard SHA-256-checked; the file
is only written into place when all match):

```bat
.venv\Scripts\python tools\v41\gsq_stream_merge.py --ours models\ds4-q2-slim.gguf --build IQ2_XS --out models\ds41-gsq-iq2_xs.gguf
```

(`--build IQ2_XXS` or `IQ1_M` for the others; 20-30 min each at 100 MB/s.)  Then point `"-m"` in
`tools/v41/serve/ds41-windows.json` at the new file.  Measured here, exact routing, against DeepSeek's API:

| | file | general | code + maths | writing, 192 / 128 GB RAM | notes |
|---|---|---|---|---|---|
| default (ds4 Q2) | 163 GB | 91.5% | 93.0% | 41 / 33 tok/s | |
| GSQ IQ2_XS | 159 GB | 90.0% | 92.9% | 38 / 26 tok/s | answered our task questions as well as the default |
| GSQ IQ2_XXS | 142 GB | 87.8% | 89.5% | 41 / 32 tok/s | fewer knowledge questions right; long prompts slower (IQ1_M layers) |
| GSQ IQ1_M | 125 GB | 85.5% | - | 42 / 37.5 tok/s | the smallest; long prompts slower (no fast prompt kernel for IQ1_M) |

On 192 GB of RAM they write about as fast as the default, so the default stays (it agrees best with DeepSeek's API).  On
128 GB IQ1_M writes fastest (more of its experts fit in RAM: 37.5 vs 33 tok/s), at 6 points less agreement; the smaller
files also suit a PC with less disk.  Checked against ISTA's files as of 2026-10-25: the
merged experts are byte-identical to theirs and every expert's format matches their published allocation; their
norms, router and router bias are bit-identical to the default file's (GSQ-RCO only changes the routed experts).

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA RTX card with 16 GB of VRAM or more, and driver 580 or newer.  24 GB (RTX 3090 / 4090, or a 32 GB 5090)
  is the fast path; with less, fewer experts fit on the card.  Measured on the 3090 with the expert budget of a smaller
  card (the default model, 192 GB RAM): 24 GB 41 tok/s, 23 GB (a monitor on the card) 39, 16 GB 32.  12 GB is not
  supported yet: on our 12 GB test card (an RTX 3060) the model loaded but ran out of memory at the first request
  (~9 GB is fixed: dense weights, KV, buffers).  RTX 20 (Titan RTX, Quadro RTX): the engine is built for it, not
  tested here.
- 128 GB of RAM or more.
- An x86-64 CPU.  AVX2 (Intel from 2013 / Haswell on, any AMD Ryzen, Threadripper or EPYC) is the fast path; newer
  instructions (AVX-512, AVX-VNNI) are used where the CPU has them.  A CPU without AVX2 (e.g. Sandy / Ivy Bridge or
  Westmere Xeons) gets the engine built for older CPUs, picked automatically: it runs, slower on the CPU side
  (measured on this PC with the AVX2 kernels switched off: 17 instead of 40 tok/s).
- An NVMe SSD with ~560 GB free during setup and ~365 GB after.
- Python 3.10, 3.11 or 3.12 (3.11 recommended) - `Start DeepSeek.bat` offers to install it.
- To build (not needed with the prebuilt engine from the Releases page):
  - Visual Studio 2019 or 2022 (Build Tools are enough) with "Desktop development with C++";
  - the CUDA Toolkit 13.0;
  - Git.

## Setup by hand (what `Start DeepSeek.bat` does for you)

1. **Python environment**, in this folder:
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"
   ```
2. **Build** with `build-ds41.bat` (10-25 minutes the first time). It builds for this PC's CPU and GPU (on a CPU
   without AVX2, for that CPU); `build-ds41.bat portable` builds one for any AVX2 CPU and RTX 20, 30, 40 and 50 cards,
   to copy to other PCs, and `build-ds41.bat compat` one for any x86-64 CPU (SSE4.2).  The launcher takes the first of
   `build-ds41`, `build-ds41-portable`, `build-ds41-compat` that this CPU can run.
3. **Model folder:** `DS41_MODELS` in `ds41-paths.bat`, on your fastest NVMe drive (default: `models\` here).
4. **Model files:** `setup-ds41-models.bat` (resumable; every large file is checked against its published SHA-256):
   - the ds4 Q2 GGUF from
     [nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF)
     (366 GB), slimmed to 163 GB; the download is then removed;
   - the two Engram shards, the tokenizer and config from
     [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (~203 GB);
   - the speculative-decoding drafter (the biggest single speed-up), built from DeepSeek's own MTP heads in the same
     repository (three shards, 7.9 GB, removed afterwards) and shrunk to 4.2 GB. The drafter only proposes tokens;
     the main model checks every one.
5. **Start** with `Start DeepSeek.bat`. Use model `deepseek-v4.1-flash-1017-ds4-q2` with any OpenAI-compatible client.

## Settings

- **One setting file:** `Start DeepSeek.bat` sets everything. Each line in it carries a short note.
- **Paths:** they live in `ds41-paths.bat`.
- **RAM:** the memory used for experts is sized from your installed RAM at every start: total minus 38 GiB, at most
  140 GiB. `DS41_ARENA_GIB` or `DS41_RAM_RESERVE_GIB` in `ds41-paths.bat` override it.
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

MIT license: [LICENSE](LICENSE).
