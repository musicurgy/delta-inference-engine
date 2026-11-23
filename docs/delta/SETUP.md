# Installing, updating and setting up by hand

[Back to the README](../../README.md)

## The first start in detail

**Which download?** Just this repository: **Code > Download ZIP** (or `git clone`). You do not need any of the
`build-ds41-*.zip` files on the Releases page - `Start DeepSeek.bat` downloads the right engine for your CPU by
itself.  (Only if you set it up by hand: almost every PC wants `build-ds41-portable.zip`; the two `compat` zips are for
CPUs from before 2013.)

**Beware of fakes.** Get Delta only from [github.com/musicurgy/delta-inference-engine](https://github.com/musicurgy/delta-inference-engine).
It downloads models only from the Hugging Face repositories written into its own scripts (the model ones are named in
[MODELS.md](MODELS.md) and in `Choose Model.bat`), and its engine only from this repository's Releases page.  It is free and never
asks for payment, a login or an account.  A download that wants payment or a login, or a model these docs do not
name, is not from this project.

1. Get this folder (**Code > Download ZIP**, or
   `git clone -c core.longpaths=true https://github.com/musicurgy/delta-inference-engine C:\AI\Delta`) somewhere with
   a short path, and check "What you need" (below, or the README): an NVIDIA RTX card with 12 GB+ (24 GB is fastest), 64 GB+ of RAM (128 GB+ recommended),
   ~320 GB free on a fast NVMe drive for the default model (its speed matters - see below).
2. Double-click **`Start DeepSeek.bat`**. The first time, it sets up everything that is missing, step by step, and
   asks before anything big.  Windows may first show a blue "Windows protected your PC" box, because these files are
   not code-signed: if you got them from this repository, choose More info, then Run anyway.
   - Python and its packages (no Python 3.10-3.14 on the PC: it offers to install python.org's Python 3.11 with
     Windows' own winget, for your Windows user only),
   - the engine: the prebuilt one for your CPU from the
     [Releases page](https://github.com/musicurgy/delta-inference-engine/releases) (under 1 GB, no compiler needed) - or it builds
     it, if you have Visual Studio Build Tools, the CUDA Toolkit and Git,
   - where the model goes: it lists your drives with their free space and type (NVMe is best; USB and hard disks are
     slow), checks Windows' page file, measures your download speed and says how long the download takes; press
     Enter for the recommended folder, or type another (it is saved as `DS41_MODELS` in `ds41-paths.bat`),
   - the model: it asks which one you want and marks the one that suits your RAM, then downloads it from Hugging Face
     (233-395 GB, checked against the published SHA-256; it checks the free space first).  You can leave it alone:
     the PC stays awake while it downloads, a download that stops moving is started again by itself, one line shows
     the progress and the time left, and a Windows notification says when it is done.  (Laptop lid or power button:
     it still sleeps.  If it stops for good, start again: it goes on where it was.)
3. Then the server starts and the chat page opens in your browser at once: it shows the start's steps and how long
   the last start took, and a question typed meanwhile is answered when the model is ready.  Apps use
   `http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages` (Anthropic), any API key.  (Another port: `DS41_PORT` in
   `ds41-paths.bat`.)

**Every day in the chat page:**
- **Pause / Resume** (top right): Pause frees the model's RAM and the graphics card for games or other programs;
  Resume, or simply the next message or app request, loads it again (about a minute).  Setup > "Pause by itself when
  a game needs the graphics card" (off by default) does it for you while DeepSeek is idle - never during an answer.
- **Use from your phone** (Setup tab): one switch makes an API key, listens on your home network and restarts; scan
  the QR code with the phone (the page opens with the key in it), allow Windows Firewall when asked, and "Add to Home
  Screen" makes it an app.  Off: this PC only again.
- **Chats** (the list button beside the message box): turn on "Save chats in this browser" to keep a chat list with
  search, pins, rename, delete and export / import (JSON).  Off by default: then each chat lives in its tab only.
  "New temporary chat" makes one that is never saved.
- **Modes** above the message box: Quick (no thinking), Think (the default), Code and Write - starting points, not
  tuned presets (Sampling changes them); under them, how long answers in that mode took to start on this PC.
- **Answer-ready notifications** (Setup > Settings, off by default): a Windows notification when an answer finishes
  while the tab is in the background (on `http://127.0.0.1` only; browsers refuse it on a LAN address).
- **Connect an app** (Setup > Connect your tools): the exact settings for Claude Code, OpenCode, Cline / Aider /
  Open WebUI and others, with Copy buttons and a Test button.

Every later start: double-click `Start DeepSeek.bat` again - it finds everything in place and starts the server
(ready in about a minute). To run another quant (see [MODELS.md](MODELS.md#other-quants-unsloth-ista-daslabs-gsq-rco-exl3)), double-click **`Choose Model.bat`**.
Everything else in this folder is the engine's source, upstream's files and tools - you do not need to open them.

**Updating:** a git clone: `git pull` in this folder (or double-click `UPDATE.bat`, which does that). A ZIP copy:
download the new ZIP and unpack it over this folder, or next to it with `models\` moved across. Then double-click
`Start DeepSeek.bat`: if the new version needs a new engine or new Python packages, it gets them first. Your model
files stay as they are. With a ZIP, keep your `ds41-paths.bat` (it holds your model folder): copy it aside before
unpacking and back after, or re-enter your model folder when the first start asks where the model files go.

**Updating from 0.94 to 0.95:** the first start downloads the 0.95 engine (SHA-256 checked) and updates the Python
packages. Your model and drafter stay. If you start DeepSeek some other way than `Start DeepSeek.bat`, see "What's new
in 0.95" for the new switches.

**Something wrong?** Double-click **`Start DeepSeek (debug).bat`** instead. It checks the PC and the install (a
PASS / FAIL list), starts DeepSeek in a second window with everything it prints saved, and when you close that window
it packs it all into `logs\debug-<time>.zip`. Attach that zip to an issue with a line on what went wrong. It holds no
prompts or answers, and your Windows user name is replaced by `<user>`.

## Settings

- **One setting file:** `Start DeepSeek.bat` sets everything. Each line in it carries a short note.  The route bias
  is also set by the chat page's Setup tab (Fast / Quality / Most exact, 0.28 / 0.10 / 0; the default depends on your
  RAM and model), and `DS41_ROUTE_BIAS` overrides both.
- **Paths:** they live in `ds41-paths.bat`.
- **Model:** the one the first start or `Choose Model.bat` picked (saved in `model-choice.txt`).
- **RAM:** the memory used for experts is sized from your installed RAM at every start: total minus 28 GiB, at most
  140 GiB (on 192 GB-class PCs: 150 for UD-IQ2_XXS, and 158 with 20 GiB kept back for files with bigger experts).
  `DS41_ARENA_GIB` or `DS41_RAM_RESERVE_GIB` in `ds41-paths.bat` override it.

## Privacy in detail

- The server listens on this PC only (`127.0.0.1:8080`). To reach it from other devices, use the Setup tab's "Use
  from your phone" switch (it makes the API key, kept in `delta-settings.json` on this PC), or add `"host": "0.0.0.0"`
  AND `"api_key": "<a long secret>"` to `tools/v41/serve/ds41-windows.json` (clients send the key as their API key).
  Without a key the server refuses to listen beyond this PC.
- Nothing from your prompts or answers is written to disk unless you turn it on: conversations live in memory and
  are gone when the server stops. Opt-in only: the plain-text watch log (`STRATA_TEXT_LOG=<a file path>`), the
  request monitor page (`"api_monitor": true` in `tools/v41/serve/ds41-windows.json`; memory only) and, in the
  built-in chat page, "Save chats in this browser" (the Chats list or Sampling; kept in the browser's own storage on
  this PC, never on the server; off = the chat lives in the tab only).

## What you need, in full

- Windows 10 or 11, 64-bit.
- An NVIDIA RTX card with 12 GB of VRAM or more, and driver 580 or newer.  24 GB (RTX 3090 / 4090, or a 32 GB 5090)
  is the fast path; with less, fewer experts fit on the card.  Measured with 0.93 on the 3090 with the expert budget of
  a smaller card (the default model, 192 GB RAM): 24 GB 41 tok/s, 23 GB (a monitor on the card) 39, 16 GB 31, 12 GB 17
  (0.94 changes which drafter experts each card keeps, so these may differ slightly).  The
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
  The model files take 233-395 GB: about 320 GB free for UD-IQ2_XXS, 270 GB for UD-IQ1_M (a GSQ build adds its 125-159 GB).
- Python 3.10 to 3.14 (3.11 recommended) - `Start DeepSeek.bat` offers to install it.
- To build (not needed with the prebuilt engine from the Releases page):
  - Visual Studio 2019 or 2022 (Build Tools are enough) with "Desktop development with C++";
  - the CUDA Toolkit 13.0;
  - Git.

## Prebuilt engine: built by CI

The three `build-ds41-*.zip` files on the Releases page are built on GitHub's public Windows runners by
[`.github/workflows/release.yml`](../../.github/workflows/release.yml), not on anyone's PC.  A version tag starts it; the
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

## Setup by hand

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
     the main model checks every one.  From 0.94 it is shrunk with an importance matrix
     (`tools\v41\serve\drafter-imatrix.bin`): the same size, 0.6% more of its guesses accepted (greedy).  A 0.93
     install keeps its drafter; `setup-ds41-models.bat drafter-imx` makes the new one (it downloads the 7.9 GB once).
5. **Start** with `Start DeepSeek.bat`. Use the model name `/v1/models` shows (`deepseek-v4.1-flash-1017-ds4-q2` for
   ds4 Q2, `deepseek-v4.1-flash-1017-ud-iq2_xxs` for UD-IQ2_XXS; the server answers any name) with any
   OpenAI-compatible client.

## Uninstall

Delta installs nothing system-wide except Python, and only if the first start installed it for you.  To remove it:

1. Stop DeepSeek (close its window).
2. If you used "Use from your phone" and allowed it through Windows Firewall: in the chat page, Setup tab, press
   **Remove the rule** - or run this in an administrator Command Prompt (`8080` is your `DS41_PORT`, if you changed it):
   ```bat
   netsh advfirewall firewall delete rule name="Delta DeepSeek 8080"
   ```
   If Windows' own firewall prompt asked instead and you allowed it, remove "Python" under Windows Security >
   Firewall & network protection > Allow an app through firewall.
3. Delete the model folder: `DS41_MODELS` in `ds41-paths.bat` (by default `models\` inside this folder, so step 4
   deletes it too).
4. Delete this folder.  That removes the engine, the Python environment (`.venv\`), your settings and the logs.

Also, if you want:
- Python 3.11, if the first start installed it: Settings > Apps, uninstall "Python 3.11".
- Download caches in your Windows user folder: `%USERPROFILE%\.cache\huggingface` and `%LOCALAPPDATA%\pip\Cache`.
- Saved chats live in your browser: delete them in the Chats list first, or clear the browser's data for
  `127.0.0.1`.
- Anything you changed yourself under "Windows settings for speed" (power plan, Lock pages in memory, Defender
  exclusions) stays until you change it back.
