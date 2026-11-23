# Delta Inference Engine

**DeepSeek-V4.1-Flash on one NVIDIA GPU, natively on Windows.**

Delta runs all 552 billion parameters of **DeepSeek-V4.1-Flash-1017** on one consumer PC: one NVIDIA RTX card plus 64 GB of RAM or more,
natively on Windows (not WSL).  Double-click `Start DeepSeek.bat`: it installs what it needs, recommends the model file
that suits your RAM, and opens a chat page plus OpenAI- and Anthropic-compatible APIs at `http://127.0.0.1:8080`, with
up to 1M tokens of context.  Free, MIT license.

This is a Windows port of [Xolotl](https://github.com/Indras-Mirror/Xolotl), which is built on
[Strata](https://github.com/Niko1221/Strata).  Upstream targets Linux; its README is [README.xolotl.md](README.xolotl.md).

| RAM | Writing speed (RTX 3090, 0.95) | Model it picks | Agreement with DeepSeek's own model |
|---|---|---|---|
| 192 GB | 44.6 tok/s | Unsloth UD-IQ2_XXS | 91.1% |
| 128 GB | 43.7 tok/s | Unsloth UD-IQ2_XXS | 90.5% |
| 96 GB | 40.2 tok/s | Unsloth UD-IQ1_M | 90.0% |
| 64 GB | 35.2 tok/s | Unsloth UD-IQ1_M | 89.9% |

Greedy, whole chats with thinking on.  Agreement = how often the top next token matches DeepSeek's own API on general
text (two hosts of the full model agree with each other on 98.3-98.9%).  More in "Speed" and "Quality" below.

## Quick start

**Get Delta only from [github.com/musicurgy/delta-inference-engine](https://github.com/musicurgy/delta-inference-engine).**
It is free and never asks for payment, a login or an account; it downloads models only from the Hugging Face
repositories named in its own scripts, and its engine only from this repository's Releases page.

1. Get this folder: **Code > Download ZIP**, or
   `git clone -c core.longpaths=true https://github.com/musicurgy/delta-inference-engine C:\AI\Delta`, somewhere with a
   short path.  You do not need the `build-ds41-*.zip` files on the Releases page: the first start downloads the right
   engine by itself.  Check "What you need" below first.
2. Double-click **`Start DeepSeek.bat`**.  If Windows shows "Windows protected your PC" (the files are not
   code-signed): More info, then Run anyway.  The first time it sets up what is missing, step by step, and asks
   before anything big: Python (it offers to install 3.11), the prebuilt engine for your CPU, a folder for the model
   (it lists your drives and their speed), and the model (233-395 GB from Hugging Face, SHA-256 checked).  You can
   leave the download alone: the PC stays awake and a stalled download restarts by itself.
3. The chat page opens at once and shows the model loading (about a minute on later starts).  Apps use
   `http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages` (Anthropic) with any API key.

**Updating:** a git clone: `UPDATE.bat` (or `git pull`).  A ZIP copy: unpack the new ZIP over this folder.  With a ZIP,
keep your `ds41-paths.bat` (it holds your model folder): copy it aside before unpacking and back after, or re-enter
your model folder when the first start asks where the model files go.  Then `Start DeepSeek.bat` gets a new engine or
packages if the version needs them; your model files stay.

**Coming from 0.94 with under 110 GB of RAM?**  0.95 recommends UD-IQ1_M there (closer to DeepSeek on small PCs);
`Choose Model.bat` downloads it.  Your current model keeps working.

**Something wrong?** Double-click `Start DeepSeek (debug).bat`: it checks the PC and the install, and packs a log zip
(no prompts or answers, your user name replaced) to attach to an issue.

More: [installing, updating and setting up by hand](docs/delta/SETUP.md).

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA RTX card with 12 GB of VRAM or more (RTX 20 or newer), driver 580 or newer.  24 GB (RTX 3090 / 4090, or a
  32 GB 5090) is the fast path; with less, fewer experts fit on the card (measured with 0.93: 16 GB 31 tok/s, 12 GB
  17, against 41 on 24 GB).  GTX 10 and older cannot run it.
- **64 GB of RAM or more, 128 GB or more recommended.**  Below 56 GB it will probably not start.
- **A fast NVMe SSD with room to spare** - the model is read from it while it runs.  About 320 GB free for UD-IQ2_XXS
  (110 GB of RAM and up), about 270 GB for UD-IQ1_M (under 110 GB).  Not a SATA SSD, hard disk or USB drive.
- An x86-64 CPU (AVX2 is the fast path; older CPUs get an engine built for them).
- Python 3.10 to 3.14 - `Start DeepSeek.bat` offers to install it.

Smaller cards, older CPUs and why the SSD matters: [docs/delta/SPEED.md](docs/delta/SPEED.md).

## What's new in 0.95

- **Up to 72% faster on 64 GB PCs** (UD-IQ2_XXS, against 0.93): 36.2 tok/s at 64 GB (21 before), 41.6 at 96 GB (29),
  43.7 at 128 GB (34), about 45 at 192 GB.  (Under 110 GB the recommended UD-IQ1_M writes 35.2 / 40.2, closer to DeepSeek.)  Experts read from the SSD are no longer memory-mapped, and prompts go
  through in wider batches.
- **Smaller PCs agree more with DeepSeek:** under 110 GB the first start offers UD-IQ1_M, run at route bias 0.10:
  89.9 / 90.0% agreement at 64 / 96 GB against 87.8 / 89.5% for UD-IQ2_XXS, at about the same speed (35.0 / 40.5
  against 35.3 / 40.9 tok/s at the default sampling).
- **A Fast / Quality / Most exact switch** in the chat page's Setup tab, showing the measured agreement and speed of
  each choice for your model at your RAM.
- **Faster writing:** +6.7% at the default sampling and +9% greedy against 0.94 (192 GB).  A learned draft gate (on by
  default) adds about 2% to sampled replies with the same output distribution.
- **Coding agents:** turns write 7.5% faster and start 8-15% sooner; 16 conversations stay cached, so agents with
  sub-agents resume in under a second; an edited message or a new question about the same long document starts in
  about 3 s; Claude Code compacts instead of stopping on a context overflow.
- **Long prompts:** a 32K-token prompt reads 19-43% faster at 96-192 GB and 12-26% faster at 64 GB (in our A/Bs).
- **Safer:** control tokens typed inside a message stay text (no forged turns or tool calls), more origin checks, and
  no single request can stall the server.
- **Quality:** a 99-question bake-off averaged 56 of 99 over four runs, against 58 for 0.94 (one run varies by about 2).

Every change, with its numbers and switches: [docs/delta/CHANGES.md](docs/delta/CHANGES.md).  Earlier versions: the
[Releases page](https://github.com/musicurgy/delta-inference-engine/releases).

## Speed

Writing speed through the server on the 0.95 release engine, RTX 3090, six chats of up to 1,500 tokens with thinking
on, greedy, counting whole chats.  Below 192 GB the rest of the PC's RAM was locked away, so Windows could not use it.

| RAM | UD-IQ2_XXS | UD-IQ1_M | UD-IQ1_S | UD-IQ3_XXS |
|---|---|---|---|---|
| 192 GB | 44.6 tok/s | 48.1 | 49.8 | 27.1 |
| 128 GB | 43.7 | 46.6 | 48.9 | 25.8 |
| 96 GB | 41.6 | 40.2 (46.2 at bias 0.28) | 43.9 (48.5 at 0.28) | 23.2 |
| 64 GB | 36.2 | 35.2 (41.5 at bias 0.28) | 40.3 (45.0 at 0.28) | 18.7 |

The 1-bit files run at route bias 0.10 under 110 GB of RAM (the launcher sets it); every other cell is at 0.28.  At the
default sampling (temperature 1) UD-IQ2_XXS writes 43.4 / 42.7 / 40.9 / 35.3 tok/s at 192 / 128 / 96 / 64 GB.  With
less RAM more experts are read from the SSD, so new prompts take longer, but the output is the same.  A second NVIDIA
card can hold extra experts for a few percent more (the Setup tab's Multi-GPU switch).

Earlier versions, prompt speeds, smaller cards, a second GPU: [docs/delta/SPEED.md](docs/delta/SPEED.md).

## Quality

How often the top choice matches DeepSeek's own API on the same text (teacher-forced, general text; this is agreement,
not task accuracy).  Two hosts of the full model agree with each other on 98.3% and 98.9%.

| Model | RAM | As shipped | Exact routing (slower) |
|---|---|---|---|
| UD-IQ2_XXS (bias 0.28) | 192 / 128 GB | 91.1% / 90.5% | 93.1% |
| UD-IQ2_XXS (bias 0.28) | 96 / 64 GB | 89.5% / 87.8% | 93.1% |
| UD-IQ1_M (bias 0.10) | 96 / 64 GB | 90.0% / 89.9% | 91.2% |

The route bias prefers experts that are already loaded, which is faster; it costs more agreement when less of the
model fits in RAM, which is why small PCs run a smaller file at a lower bias.  The chat page's Setup tab switches
between **Fast / Quality / Most exact** (route bias 0.28 / 0.10 / 0) and shows the measured numbers for your PC;
`DS41_ROUTE_BIAS` in `ds41-paths.bat` overrides it.  On a 99-question bake-off (maths, knowledge, reasoning, coding)
0.95 scored 55, 53, 56 and 61 of 99 (average 56; one run varies by about 2), 0.94 58.

Code, maths and agentic scores, every model file and how to pick one: [docs/delta/MODELS.md](docs/delta/MODELS.md).

## Apps, agents and the chat page

The server is at `http://127.0.0.1:8080`; any API key works unless you set one, and it answers under any model name.
Give apps the real context size (1,000,000), or they may compact early.  The chat page's **Setup > Connect your tools**
shows the exact settings for each app, with Copy and Test buttons.

- **Claude Code**, in the Command Prompt you start it from:
  ```bat
  set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
  set ANTHROPIC_AUTH_TOKEN=local
  set ANTHROPIC_MODEL=deepseek-v4.1-flash-1017
  set CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000
  claude
  ```
- **OpenCode, Codex CLI, Cline, Aider, Continue, Open WebUI** and other OpenAI-compatible apps: base URL
  `http://127.0.0.1:8080/v1`, model `deepseek-v4.1-flash-1017`.
- Tool calls, JSON output, pictures (OpenAI `image_url`, Anthropic `image`), documents (PDF, Word, Excel, PowerPoint)
  and optional voice (`/v1/audio/...`) all work through the APIs.  Requests are answered one at a time.

**In the chat page:** pictures (paste, drop, screenshot, camera), documents, optional voice (`Enable Voice.bat`:
dictation, read-aloud, hands-free talk), Quick / Think / Code / Write modes, chat history (off by default), **Pause**
to free the GPU for a game (and an optional auto-pause), and use from your phone with a QR code.

Configs for each app, the conversation cache, pictures, documents and voice in detail: [docs/delta/APPS.md](docs/delta/APPS.md).

## Privacy

- The server listens on this PC only (`127.0.0.1:8080`).  The Setup tab's "Use from your phone" switch opens it to
  your network with an API key; without a key the server refuses to listen beyond this PC.
- Nothing from your prompts or answers is written to disk unless you turn it on (chat history is kept in your
  browser, never on the server).  Downloads send no telemetry.
- More (opening it to your network by hand, the opt-in logs): [privacy in detail](docs/delta/SETUP.md#privacy-in-detail).

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

## Settings

`Start DeepSeek.bat` sets everything, with a short note on each line; paths live in `ds41-paths.bat`, the model in
`model-choice.txt` (`Choose Model.bat` changes it), the route bias in the chat page's Setup tab.  The memory for
experts is sized from your installed RAM at every start.  More: [settings in detail](docs/delta/SETUP.md#settings)
(RAM, port, the first minutes after a reboot) and [every switch](docs/DETAILS.md).

## Setup by hand

What `Start DeepSeek.bat` does for you, step by step: [docs/delta/SETUP.md](docs/delta/SETUP.md#setup-by-hand).  In short:

1. **Python environment**, in this folder: `py -3.11 -m venv .venv`, then
   `.venv\Scripts\pip install -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"`.
2. **Build** with `build-ds41.bat` (10-25 minutes; Visual Studio Build Tools, the CUDA Toolkit 13.0 and Git), or use
   the prebuilt engine from the Releases page.
3. **Model files** with `Choose Model.bat` (or `setup-ds41-models.bat` for ds4 Q2), including the drafter
   (`setup-ds41-models.bat drafter-imx` makes it).

Also there: [checking a release zip](docs/delta/SETUP.md#prebuilt-engine-built-by-ci) and
[uninstalling](docs/delta/SETUP.md#uninstall).

## Credits

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) and their
  contributors: the engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): the tensor library.
- DeepSeek: the model, and the MTP heads the drafter is built from.
- [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF): the UD quants; nazeshinjite: the ds4 Q2
  GGUF; ISTA-DASLab: the GSQ-RCO quants.
- [Project Maya](https://github.com/mw00/project-maya) and [antirez/ds4](https://github.com/antirez/ds4): ported fixes
  and test cases, and the start of the EXL3 kernels; Basalt (jesdga95/basalt): the ideas behind several decode switches.
- Voice and documents: see [docs/delta/APPS.md](docs/delta/APPS.md#credits).

MIT license: [LICENSE](LICENSE).
