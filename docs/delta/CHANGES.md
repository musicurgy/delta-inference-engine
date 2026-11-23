# What's new in 0.95, in full

[Back to the README](../../README.md)

The 192 GB figures below were measured on the 0.95 release build against 0.94 as published, both through their own
`Start DeepSeek.bat`, one after the other on the same PC: RTX 3090, 192 GB of RAM, UD-IQ2_XXS. The per-change 64-128 GB
figures in the bullets come from single A/B tests during the 0.95 work; the speeds by RAM size (the first bullet, and
"Speed by RAM size" in [SPEED.md](SPEED.md)) were measured on the release build.

- **Up to 72% faster on 64 GB PCs.** UD-IQ2_XXS writing whole chats (greedy) on the release build: **36.2 tok/s at 64 GB**
  against 21 for 0.93 (+72%), **41.6 at 96 GB** against 29 (+43%), **43.7 at 128 GB** against 34 (+29%), and about 45 at
  192 GB (44.6).  The baselines are 0.93's own measurements (0.94 was only measured at 192 GB).  The full table is under
  "Speed by RAM size".
- **Smaller PCs agree more with DeepSeek.**  Under 110 GB of RAM the first start offers UD-IQ1_M, and the launcher now
  runs the 1-bit files there at route bias 0.10 instead of 0.28.  On a 64 / 96 GB PC UD-IQ1_M agrees with DeepSeek more
  than UD-IQ2_XXS at 0.28 (89.9 / 90.0% vs 87.8 / 89.5% on general text) at the same speed (35.0 / 40.5 vs 35.3 / 40.9
  tok/s at the default sampling); UD-IQ1_S at 0.10 is the faster choice (39.9 / 44.9 tok/s, 88.8 / 89.6%).  From 110 GB
  up UD-IQ2_XXS stays the pick, at 0.28.  The chat page's Setup card shows it as a Fast / Quality / Most exact switch
  (route bias 0.28 / 0.10 / 0) on every PC: Quality is the default for a 1-bit file under 110 GB, Fast elsewhere, and
  each choice shows its measured agreement and speed for that model at this PC's RAM.  `DS41_ROUTE_BIAS` still
  overrides it.
- **Faster writing.** Plain chat at the default sampling (temperature 1, top-p 0.95; six chats with thinking on, two
  seeds): 42.0 tok/s while writing, against 39.5 for 0.94 (+6.3%); counting whole chats 43.4 against 40.7 (+6.7%).
  This test gives each chat a longer instruction than the test behind the 0.94 figure in the history table in [SPEED.md](SPEED.md), so
  compare 0.95 with 0.94 within this line, not across the two.
- **The same test at greedy** (temperature 0, as in the speed tables): **about 45 tok/s - 44.6 measured on the final
  build** counting whole chats, against 40.9 for 0.94 (+9%).
- **Learned draft gate (new, on by default):** a small model on the drafter's own confidence, uncertainty and top-2
  margin decides how many drafted tokens each pass checks, with the accept coins drawn first, so the output
  distribution is unchanged.  It acts only on sampled replies (temperature above 0, the default): +2.0% on chat,
  +2.3% on agent turns; greedy replies are unchanged.  Four 99-question bake-offs with it
  averaged 56 (55, 53, 56, 61) against 58 without it (56, 60) - inside the run-to-run spread of about 2 points.
  `set STRATA_DSPARK_LGATE=0` turns it off.
- **64-128 GB PCs read prompts faster again:** at 64 GB a long prompt goes through in fewer, wider batches - 32K-40K
  token prompts +12-13% (measured at a real 64 GB) - and on every PC that reads experts from the SSD the Engram rows
  are read ahead in blocks (+1-2.5% on prompts at 96 / 128 GB, the same output).
- **Opt-in fast mode, 47 tok/s:** `set STRATA_EXPERTS_DROP_MISS=5` skips a low-ranked expert when it is not on the
  graphics card. The greedy test above gives 47.4 tok/s counting whole chats (+9% over 0.95's default, +16% over
  0.94). It changes the answers: in two 99-question bake-offs it averaged 54 against 58 for the default, so it trades
  some quality for speed and stays off unless you set it.
- **Agent loops write faster and start sooner.** On a coding agent's short turns (greedy, 28 turns): 49.5 tok/s against
  46.1 (+7.5%), first token after 4.5 s instead of 4.9 s (-8%). On turns that write ~500 tokens: 39.2 against 36.5
  tok/s (+7.5%), first token after 4.6 s instead of 5.4 s (-15%).
- **Much faster on 64-128 GB PCs:** experts read from the SSD are no longer memory-mapped. In our A/Bs writing got
  +29% / +10% / +7% faster and 32K-token prompts +26% / +11% / +17% at 64 / 96 / 128 GB (not re-measured on the final
  build).
- **Long prompts read faster:** wide prompt batches, a faster expert top-k and a prefill read-ahead where experts sit
  on the SSD: a 32K prompt +19% at 192 GB, +23% at 128 GB, +43% at 96 GB (not re-measured on the final build).
- **Edits and revisits resume:** each cached conversation keeps its last 1024 window rows, so an edited last message or
  a new question about the same long document starts in about 3 s instead of re-reading 8-32K tokens.
- **The first answer comes sooner:** the drafter loads while the model starts.
- **Fewer garbled tokens:** a sampled token that turns into an "attention sink" (after which the model writes noise) is
  replaced by the next candidate.
- **Agents and APIs:** 16 conversations kept in RAM (agents with sub-agents resume in under a second), loop guard over
  the last 12 tool calls, Claude Code compacts on a context overflow, thinking a client dropped is put back, DeepSeek's
  own API paths, and fixes for streaming, batch slots, stop tokens and the Windows console.
- **Quality:** 99-question bake-off (maths, knowledge, reasoning, coding), as shipped (learned draft gate on), four runs with different seeds: 55, 53, 56 and 61 out of 99, average 56; without the gate 56 and 60 (average 58), 0.94 58.  One run varies by about 2 points.
- **Same input, same answer?** At temperature 0 a reply can differ slightly from one run to the next: the draft gate
  sizes each pass from live timings, and a different pass width rounds the sums differently.  The answers are as good,
  just not byte-identical between runs; `set STRATA_DSPARK_GATE=0` restores repeatable greedy
  output at a small speed cost.

All of these are on by default; [docs/DETAILS.md](../DETAILS.md) lists each switch and how to turn it off. Opt-in (off unless you
`set` them before starting):
- `STRATA_EXPERTS_DROP_MISS=5`, an **experimental fast mode**: it skips a low-ranked expert when it is not on the
  graphics card. +6.8% writing in our A/B at 192 GB, but it changes the answers: two 99-question runs averaged 54
  against 58 without it. Use it only if you want speed over answer quality.
- `STRATA_X3B=1`: launches the next layer's work ahead. Not recommended: in our last test a pass still had to be
  rescued by its watchdog, and it was not faster.
- `STRATA_IDLE_PREFETCH=1`: copies experts ahead only while the bus is idle. Not recommended: it was 6-13% slower in
  our tests.

What was new in 0.94 (pictures on by default, documents, voice, pause, chat history, use from your phone, Claude Code's
helper calls, bounded prompt replay, a safer server) and in 0.93 (a model picker and Setup tab, Unsloth's UD-IQ2_XXS as
the default from 128 GB, 64 and 96 GB of RAM supported, the second-GPU helper, release zips built by GitHub), and the
full list for every version, are on the [Releases page](https://github.com/musicurgy/delta-inference-engine/releases).
