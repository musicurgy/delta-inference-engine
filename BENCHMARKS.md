# Benchmarks

DeepSeek-V4.1-Flash-1017 running locally on one RTX 3090, compared with DeepSeek's own model served over an API.
Every number below was measured. Nothing is estimated or copied from elsewhere.

## Answer quality: 99 hard questions

| Set | Source | Questions | Local (default ds4 Q2) | DeepSeek V4.1 Flash (API) | Qwen3.8 Flash-Next (API) |
|---|---|---|---|---|---|
| Maths | AIME / HMMT 2026, recent arXiv maths (MathArena) | 14 | 10 | 10 | 8 |
| Knowledge | SimpleQA Verified | 40 | 9 | 13 | 7 |
| Reasoning | BIG-Bench Extra Hard | 30 | 23 | 25 | 21 |
| Coding | AtCoder, Sep-Oct 2026 | 10 | 10 | 10 | 8 |
| Coding, hard | AtCoder AGC 078 / ARC 230 (hard pool rated 2,393-3,827) | 5 | 0 | 1 | 0 |
| **Total** | | **99** | **52** | **59** | **44** |

### Method
- **Budget:** 43,691 output tokens per question for every model.
- **Settings:** temperature 0, thinking on, one attempt per question, no tools or search.
- **Grader:** one fixed grader for every model.
  - Maths answers are compared symbolically with sympy.
  - Knowledge and reasoning answers are matched against the official key, with no AI judge.
  - Code passes only if it is right on every official sample and on 40 brute-force-checked random cases, and finishes 2 large cases within the time limit.
- **Forced:** on the local server, thinking stops at 32,768 tokens, and the model must then answer within the remaining 10,923. The API runs have no such stop: an answer still unfinished at 43,691 tokens scores 0.
- **Dates:** questions frozen 2026-10-21 and 2026-10-22. Maths comes from Feb-Jun 2026; coding from AtCoder contests held 2026-09-05 to 2026-10-17. API runs on 2026-10-22, with the answers that hit the earlier cap rerun at 43,691 on 2026-10-27/28. Local runs on 2026-10-27/28.

## Next-token agreement with DeepSeek's API

The share of positions where the local build picks the same next token as DeepSeek's own API, on the same text. Higher is closer to the original model.

| Build | General | Code + maths |
|---|---|---|
| ds4 Q2 (default) | 91.5% | 93.0% |
| GSQ IQ2_XS | 90.0% | 92.9% |
| GSQ IQ2_XXS | 87.8% | 89.5% |
| GSQ IQ1_M | 85.5% | - |
| UD-IQ1_S | 89.3% | 91.1% |
| EXL3 2.9 bpw | 95.1% | 97.2% |
| EXL3 2.5 bpw | 94.4% | 95.6% |

## Speed

Decode speed on one RTX 3090. The measurement is `bench_server`: 6 chats of up to 1,500 tokens each, with thinking on, through the release's own server.
- The 96 and 64 GB rows are the same PC with the spare RAM locked away (so neither Windows' file cache nor the pagefile could use it); the 128 GB row is an earlier measurement with the arena sized for 128 GB.
- The ds4 Q2 rows use the default model file and this release's launcher.

| Build | RAM | Decode tok/s |
|---|---|---|
| ds4 Q2 (default) | 192 GB | 41 |
| ds4 Q2 (default) | 128 GB | 33 |
| ds4 Q2 (default) | 96 GB | 27 |
| ds4 Q2 (default) | 64 GB | 20 |
| UD-IQ1_S | 96 GB | 30 |
| UD-IQ1_S | 64 GB | 25.5 |
| EXL3 2.5 bpw (opt-in quality mode) | 192 GB | 25.4 |
