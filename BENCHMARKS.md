# Benchmarks

DeepSeek-V4.1-Flash-1017 running locally on one RTX 3090, compared with DeepSeek's own model served over an API.
Every number below was measured. Nothing is estimated or copied from elsewhere.

## Answer quality: 99 hard questions

| Set | Source | Questions | Local UD-IQ2_XXS (0.93) | Local ds4 Q2 (0.92) | DeepSeek V4.1 Flash (API) | Qwen3.8 Flash-Next (API) |
|---|---|---|---|---|---|---|
| Maths | AIME / HMMT 2026, recent arXiv maths (MathArena) | 14 | 10 | 10 | 10 | 8 |
| Knowledge | SimpleQA Verified | 40 | 12 | 9 | 13 | 7 |
| Reasoning | BIG-Bench Extra Hard | 30 | 25 | 23 | 25 | 21 |
| Coding | AtCoder, Sep-Oct 2026 | 10 | 10 | 10 | 10 | 8 |
| Coding, hard | AtCoder AGC 078 / ARC 230 (hard pool rated 2,393-3,827) | 5 | 0 | 0 | 1 | 0 |
| **Total** | | **99** | **57** | **52** | **59** | **44** |

### Method
- **Budget:** 43,691 output tokens per question for every model.
- **Settings:** temperature 0, thinking on, one attempt per question, no tools or search.
- **Grader:** one fixed grader for every model.
  - Maths answers are compared symbolically with sympy.
  - Knowledge and reasoning answers are matched against the official key, with no AI judge.
  - Code passes only if it is right on every official sample and on 40 brute-force-checked random cases, and finishes 2 large cases within the time limit.
- **Forced:** on the local server, thinking stops at 32,768 tokens, and the model must then answer within the remaining 10,923. The API runs have no such stop: an answer still unfinished at 43,691 tokens scores 0.
- **Dates:** questions frozen 2026-10-21 and 2026-10-22. Maths comes from Feb-Jun 2026; coding from AtCoder contests held 2026-09-05 to 2026-10-17. API runs on 2026-10-22, with the answers that hit the earlier cap rerun at 43,691 on 2026-10-27/28. Local ds4 Q2 runs on 2026-10-27/28, UD-IQ2_XXS on 2026-11-03/04, at route bias 0.28.

## Next-token agreement with DeepSeek's API

The share of positions where the local build picks the same next token as DeepSeek's own API, on the same text. Higher is closer to the original model. For scale, two hosts of the full model agreed 98.3% and 98.9%.

At the settings that ship (route bias 0.28):

| Build | General | Code + maths | Agentic |
|---|---|---|---|
| UD-IQ2_XXS (default from 128 GB up) | 91.1% | 94.5% | 95.9% |
| ds4 Q2 | 89.9% | 93.1% | 95.1% |
| UD-Q2_K_XL | 92.5% | 95.3% | 97.0% |
| UD-IQ3_XXS | 92.7% | 95.9% | 97.4% |
| UD-IQ1_S | 89.1% | 91.5% | 95.3% |

Exact routing (no route bias, slower):

| Build | General | Code + maths |
|---|---|---|
| ds4 Q2 | 91.5% | 93.0% |
| GSQ IQ2_XS | 90.0% | 92.9% |
| GSQ IQ2_XXS | 87.8% | 89.5% |
| GSQ IQ1_M | 85.5% | - |
| UD-IQ1_S (older files) | 89.3% | 91.1% |
| UD-IQ2_XXS | 92.5% | 95.9% |
| UD-Q2_K_XL | 95.0% | 96.8% |
| UD-IQ3_XXS | 95.5% | 97.6% |
| EXL3 2.9 bpw | 95.1% | 97.2% |
| EXL3 2.5 bpw | 94.4% | 95.6% |

## Speed

Decode speed on one RTX 3090. The measurement is `bench_server`: 6 chats of up to 1,500 tokens each, with thinking on, through the release's own server.
- The 128, 96 and 64 GB columns are the same PC with the spare RAM locked away (so neither Windows' file cache nor the pagefile could use it).
- Greedy, route bias 0.28, this release's launcher.

| Build | 192 GB | 128 GB | 96 GB | 64 GB |
|---|---|---|---|---|
| UD-IQ2_XXS | 40.4 | 34.0 | 29.3 | 21.0 |
| ds4 Q2 | 40.0 | 35.9 | 27 | 20 |
| UD-IQ1_S | 41.0 | 40.6 | 36.5 | 30.6 |
| UD-Q2_K_XL | 31.3 | 26.6 | 22.9 | 15.2 |
| UD-IQ3_XXS | 28.2 | 22.8 | 18.3 | 11.7 |
| EXL3 2.5 bpw (opt-in) | 25.4 | - | - | - |

Long prompts on UD-IQ2_XXS, a new prompt with nothing cached:

| Prompt tokens | 192 GB | 128 GB |
|---|---|---|
| ~1,300 | 13 s (100 tok/s) | 16 s (80 tok/s) |
| ~6,300 | 15 s (410 tok/s) | 20 s (320 tok/s) |
| ~25,500 | 41 s (620 tok/s) | 54 s (470 tok/s) |
| ~64,000 | 87 s (740 tok/s) | 107 s (600 tok/s) |

### Same PC, same file: llama.cpp

Mainline llama.cpp (commit 36a0e2f of 2026-10-29, built with CUDA for the 3090) and this engine, on the same PC, the same Unsloth UD-IQ1_S model file, the same 6 chats of up to 1,500 tokens with thinking on, full RAM, two interleaved runs each. llama.cpp was run the usual way for a model this size: `-ngl 99 -ot exps=CPU -t 14 -c 8192` (the routed experts in RAM on the CPU, everything else on the card). This comparison ran llama.cpp WITHOUT its speculative decoding: llama.cpp does support a DSpark drafter for DeepSeek V4.1 (`--spec-type draft-dspark`, the drafter Unsloth ships), and a re-run with it is in progress; this row will be updated.

| Engine | Decode tok/s (server) | Decode tok/s (what the client sees) | Prompt tok/s |
|---|---|---|---|
| llama.cpp mainline | 7.7 | 7.4 | ~35 |
| this release | 37.5 | 34 | ~45 |

The difference is where the experts run: llama.cpp computes every routed expert on the CPU for every token, which is bound by RAM bandwidth; this engine keeps most expert reads on the card (a VRAM cache plus a pinned RAM arena over PCIe) and drafts about 3 tokens per pass.
