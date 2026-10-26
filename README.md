# DeepSeek-V4.1-Flash on one NVIDIA GPU, natively on Windows

Runs all 552 billion parameters of **DeepSeek-V4.1-Flash-1017** (DeepSeek's official weights, 2-bit experts) on one
consumer PC, natively on Windows. It serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` with up to 1M tokens
of context.

This is a Windows port of [Xolotl](https://github.com/Indras-Mirror/Xolotl), which is built on
[Strata](https://github.com/Niko1221/Strata). Upstream targets Linux; its README is
[README.xolotl.md](README.xolotl.md).

## Speed

These numbers are from an RTX 3090 with an NVMe SSD, through the server:

| | 192 GB RAM | 128 GB RAM |
|---|---|---|
| Writing (6 chats, up to 1,500 tokens each, thinking on) | **42 tok/s** | **29 tok/s** * |
| Reading a long prompt (51-64k tokens) | **610 tok/s** | **119 tok/s** * |

\* Measured with an earlier version (the 192 GB figures were 36 and 408 tok/s then).

With less RAM, more of the model's experts are read from the SSD as they are needed. That is slower, most of all on
long prompts, but the output is the same.

On the 192 GB PC, a new 6k-token prompt takes about 15 s and a 25k-token prompt about 41 s. Writing is fastest on
maths and code (~48 and ~45 tok/s) and ~35 tok/s on plain prose; counting the prompt and the thinking, a client sees
~39 tok/s.

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
- We also tested ISTA-DASLab's GSQ-RCO quants
  ([DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF))
  in this engine. IQ2_XS answered our test questions as well as the default Q2 build, and IQ2_XXS got fewer of the
  knowledge questions right. Neither was faster here, so the default stays.

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA RTX 30, 40 or 50 series card with 24 GB of VRAM. Smaller cards are untested.
- 128 GB of RAM or more.
- An x86-64 CPU with AVX2: Intel from 2013 (Haswell) on, or any AMD Ryzen, Threadripper or EPYC. Newer instructions
  (AVX-512, AVX-VNNI) are used where the CPU has them.
- An NVMe SSD with ~730 GB free during setup and ~360 GB after.
- To build:
  - Visual Studio 2019 or 2022 with "Desktop development with C++";
  - the CUDA Toolkit 13.0;
  - Python 3.11 or newer;
  - Git.

## Setup

1. **Python environment**, in this folder:
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt tokenizers==0.23.3 "huggingface_hub[cli]==2.2.0"
   ```
2. **Build** with `build-ds41.bat`.
   - It builds for this PC's CPU and GPU.
   - `build-ds41.bat portable` builds a version for any AVX2 CPU and RTX 30, 40 and 50 cards, which you can copy to
     other PCs.
   - The first build takes about 25 minutes.
3. **Model folder.** Set `DS41_MODELS` in `ds41-paths.bat` to a folder on your fastest NVMe drive. The default is
   `models\` in this folder.
4. **Download and prepare the model** with `setup-ds41-models.bat`. You can run it again; it resumes.
   - The ds4 Q2 GGUF from
     [nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF)
     (366 GB) is slimmed to 163 GB. You can delete the download afterwards.
   - The two Engram shards and the tokenizer come from
     [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (~190 GB).
5. **Speculative-decoding drafter** (recommended: the biggest single speed-up).
   - Save a Strata v41 sidecar GGUF (DeepSeek-V4.1's MTP heads) as `V41-1017-MTP.gguf` in the model folder. One is
     published with
     [IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored).
   - Run `setup-ds41-models.bat` again to convert it.
   - The drafter only proposes tokens; the main model checks every one.
6. **Start** with `Start DeepSeek.bat`.
   - It is ready in about a minute.
   - Use model `deepseek-v4.1-flash-1017-ds4-q2` with any OpenAI-compatible client.

## Settings

- **One setting file:** `Start DeepSeek.bat` sets everything. Each line in it carries a short note.
- **Paths:** they live in `ds41-paths.bat`.
- **RAM:** the memory used for experts is sized from your installed RAM at every start: total minus 38 GiB, at most
  140 GiB. `DS41_ARENA_GIB` or `DS41_RAM_RESERVE_GIB` in `ds41-paths.bat` override it.
- **Optional Engram hot set:** an `engram-hot-2g.bin` in the model folder keeps the most-used Engram rows in RAM.
  Build it with `ds41_engram_hot.exe`; the header of `tools/v41/engine/ds41_engram_hot.cpp` explains how.

## Credits

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) and their
  contributors: the engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): the tensor library.
- nazeshinjite: the ds4 Q2 GGUF.
- DeepSeek: the model.

MIT license: [LICENSE](LICENSE).
