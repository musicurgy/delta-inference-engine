# Apps, agents and the chat page

[Back to the README](../../README.md)

## Every day in the chat page

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

## Connecting apps and agents

Start DeepSeek first.  The server is at `http://127.0.0.1:8080`, any API key
works unless you set one, and it answers under any model name.  Give the app the real context size, or it may
compact your conversation early.

- **Claude Code**, in the Command Prompt you start it from:

  ```bat
  set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
  set ANTHROPIC_AUTH_TOKEN=local
  set ANTHROPIC_MODEL=deepseek-v4.1-flash-1017
  set CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000
  claude
  ```

  Without the last `set`, Claude Code assumes 200K tokens and compacts there.
  Claude Code's WebSearch works once you give the server a search service (`"web_tools"`, off by default; see
  [docs/DETAILS.md](../DETAILS.md#web-search-and-web-fetch-off-by-default-094)).
- **OpenCode**, in `opencode.jsonc` (in your project, or `~/.config/opencode/`):

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "delta": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Delta (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
        "models": {
          "deepseek-v4.1-flash-1017": {
            "name": "DeepSeek V4.1 Flash",
            "limit": { "context": 1000000, "output": 32768 }
          }
        }
      }
    }
  }
  ```

- **Cline, Aider, Continue, Open WebUI and other OpenAI-compatible apps:** base URL `http://127.0.0.1:8080/v1`, any API
  key, model name `deepseek-v4.1-flash-1017`.  Where the app asks for a context window, enter `1000000`.
- **Codex CLI:** see [the Responses API](../DETAILS.md#the-responses-api-and-codex-cli).

- **Conversation cache:** the server keeps up to 16 conversations in memory (about 20 MB per 10k tokens; 8 before
  0.95; `DS41_CACHE_SLOTS` sets it).  Each keeps its last 1024 window rows, so an edited last message or a new
  question about the same long document starts in about 3 s instead of re-reading it (0.95). An agent that
  switches between conversations, or makes a side request between turns, resumes each one in a second or two instead
  of reading its whole prompt again (measured: 16-18 s per switch before, 1-3 s now; a 9-turn session alternating two
  ~8k-token conversations: 95 s before, 49 s now).
- **Follow-up messages** only read what is new. In thinking mode, the next turn no longer re-reads the history. A
  follow-up of a few hundred to 2,000 new tokens (a tool result, a pasted file) keeps most of the GPU's expert cache
  instead of emptying and refilling it, which saves 0.6-0.9 s per turn.
- **Tool calls** (OpenAI `tools`, including a forced `tool_choice`: `"required"` or a named function) and **JSON
  output** (`response_format` `json_object` / `json_schema`) work through the OpenAI-compatible API. Requests sent at
  the same time are answered one after another.
- **Pictures** are on. The first start downloads DeepSeek's vision encoder (0.97 GB, checked) and PyTorch (124 MB).
  In the chat page, attach, drop or paste a picture, or use the screen and camera buttons; apps send one through the
  OpenAI (`image_url`) or Anthropic (`image`) API. Pictures never take VRAM from the model's card:
  - **One graphics card:** pictures are read on the CPU. A 1024x701 photo takes ~8-11 s on an i5-13600K; writing
    speed was within run-to-run spread of text-only in our one measurement (0.92).  On PCs with less than about
    170 GB of RAM the picture reader starts with the first picture (about 7 s once, longer while the model is
    writing) and stops after 10 idle minutes, so its ~4 GB of RAM is free the rest of the time.
  - **If the picture reader cannot start** (a broken PyTorch, say), the model still starts and answers text; a
    picture then gets an error that says why.
  - **Multi-GPU on** (the chat page's Setup card): pictures are read on the second card, in under half a second. This
    needs PyTorch's CUDA build, a 2.0 GB download that the next start offers; the second card keeps ~2 GB less for
    the model's experts.
  - **Faster, smaller pictures:** an app can send `"detail": "low"` with a picture (OpenAI's `image_url`): at most 256
    tokens instead of up to 1,024, read about twice as fast. Small text may become hard to read.
  - **Turn them off:** double-click **`Disable Pictures.bat`** and start DeepSeek again. **`Enable Pictures.bat`** turns
    them back on. To choose where pictures are read, put one word in `vision-choice.txt`: `on` (the default), `cpu`,
    `gpu:1` (a card by its `nvidia-smi` number), `main` (the model's own card: fastest, but ~1.5 GB of its VRAM) or
    `off`.
- **Documents**: attach or drop a PDF, Word, Excel or PowerPoint file (or text, Markdown, CSV, HTML) in the chat page.
  Its text goes into the message, and the page shows how many tokens it is and how long the first read takes
  (~650 tokens/s; follow-ups reuse it) before you send. Through the API: Anthropic `document` blocks, OpenAI `file`
  parts and Responses `input_file` parts (the file itself, base64; URLs and file ids are not fetched). Scanned PDFs
  have no text to read; old `.doc` / `.xls` / `.ppt` files need saving as the new formats first.
- **Voice** is off by default. Double-click **`Enable Voice.bat`** once, then start DeepSeek again: it downloads the
  speech models (1.1 GB, checked) and sherpa-onnx (28 MB). Then the chat page has a **microphone** (hold to talk, or
  click to start and again to stop; the text lands in the message box), a **speaker** under every answer, and a
  **Talk** button for a hands-free conversation (a short pause sends; talking over the answer interrupts it). Any
  OpenAI app can use `/v1/audio/transcriptions` and `/v1/audio/speech`. The speech models start with their first use
  (a few seconds) and stop after 10 idle minutes. They run on the CPU's efficiency cores: a 5 s clip is written out in
  ~0.7 s and an answer is spoken about twice as fast as it plays - while the model is not writing. While it writes,
  they share the CPU with it and are much slower. `Enable Voice.bat gpu` (from a command window) puts read-aloud on a
  second NVIDIA card instead (~2 GB more to download). The microphone works on `http://127.0.0.1` / `localhost`;
  from another device the page needs HTTPS. To turn voice off again, put `off` in `voice-choice.txt`.

## Credits

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) and their
  contributors: the engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): the tensor library.
- nazeshinjite: the ds4 Q2 GGUF.
- ISTA-DASLab: the GSQ-RCO quants.
- DeepSeek: the model, and the MTP heads the drafter is built from.
- Voice (optional): [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0); NVIDIA's
  [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0), in sherpa-onnx's int8
  conversion; [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad (Apache-2.0);
  [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
- Documents: [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) / PDFium (Apache-2.0 / BSD-3),
  python-docx, openpyxl and python-pptx (MIT).
- [Project Maya](https://github.com/mw00/project-maya) and [antirez/ds4](https://github.com/antirez/ds4): server fixes
  ported in 0.93 (the end of thinking by its token, TCP_NODELAY; a blank replayed reasoning read as absent); the
  EXL3 CPU / CUDA kernels started from Project Maya's port of exllamav3's (MIT).
