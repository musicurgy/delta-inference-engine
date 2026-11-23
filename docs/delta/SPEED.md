# Speed in detail

[Back to the README](../../README.md)

## Speed by RAM size (0.95)

Writing speed through the server on the 0.95 release engine, RTX 3090: six chats of up to 1,500 tokens each with
thinking on, greedy (temperature 0), counting whole chats.  Below 192 GB the rest of the PC's RAM was locked away, so
neither Windows' file cache nor the page file could use it, and the launcher set the PC up for that much RAM.

| RAM | UD-IQ2_XXS (the pick from 110 GB) | UD-IQ1_M (the pick below 110 GB) | UD-IQ1_S | UD-IQ3_XXS |
|---|---|---|---|---|
| 192 GB | 43.7 tok/s (44.6 on the final-build test) | 48.1 | 49.8 | 27.1 |
| 128 GB | 43.7 | 46.6 | 48.9 | 25.8 |
| 96 GB | 41.6 | 40.2 (46.2 at bias 0.28) | 43.9 (48.5 at 0.28) | 23.2 |
| 64 GB | 36.2 | 35.2 (41.5 at bias 0.28) | 40.3 (45.0 at 0.28) | 18.7 |

The 1-bit files run at route bias 0.10 below 110 GB of RAM (the launcher sets it), every other cell at 0.28.

- At the default sampling (temperature 1, learned draft gate on): UD-IQ2_XXS 43.4 tok/s at 192 GB, 42.7 at 128 GB,
  40.9 at 96 GB and 35.3 at 64 GB; at bias 0.10 UD-IQ1_M 40.5 at 96 GB and 35.0 at 64 GB, UD-IQ1_S 44.9 and 39.9.
- Download sizes: UD-IQ2_XXS 301 GB, UD-IQ1_M 250 GB, UD-IQ1_S 233 GB, UD-IQ3_XXS 395 GB (it also needs the ds4 Q2 file
  for its Engram tables).
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
- Below 56 GB of RAM it will probably not start.

Measured with 0.93 on the 192 GB PC (0.95 reads a 32K prompt 19-43% faster, depending on RAM): a new 6k-token prompt takes about 15 s and a 25k-token prompt about 41 s (UD-IQ2_XXS). Writing is fastest on
maths and code (~48 and ~45 tok/s) and ~35 tok/s on plain prose; counting the prompt and the thinking, a client sees
~38 tok/s.

## Earlier versions

These numbers are from an RTX 3090 with an NVMe SSD, through the server.  In these docs a speed was measured with
0.93 (or earlier) unless it says 0.94 or 0.95.  0.94 changed the drafter, bounded the prompt replay and turned pictures
on; 0.95 reads SSD experts without memory-mapping, reads prompts in wider batches and adds the learned draft gate
(sampled replies only), so the 0.93 / 0.94 figures here are lower than 0.95's - see "Speed by RAM size" above.

0.95 against 0.94, the same run (192 GB, the `--long` chat test, two seeds): greedy about 45 tok/s (44.6 on the
final build) against 40.9; at the default sampling 43.4 against 40.7 counting whole chats (42.0 against 39.5 while
writing).  0.94 against 0.93, the same evening (0.94's own test):

| UD-IQ2_XXS (history) | 192 GB RAM, 0.94 | 192 GB RAM, 0.93 | 128 GB RAM, 0.93 |
|---|---|---|---|
| Writing (6 chats, up to 1,500 tokens each, thinking on), greedy | **41.7 tok/s** | 40.8 tok/s (41.1 on its release card) | 34 tok/s |
| Writing, the default sampling (temperature 1, top-p 0.95) | **41.3 tok/s** | 39.2 tok/s (39.7 on its release card) | - |
| Reading a long prompt (~60K tokens, fresh) | **811 tok/s** (860 at ~30K) | 747 tok/s the same evening (up to 740 on its release card) | up to 600 tok/s |

The two 192 GB writing rows were measured the same evening on the same PC, one version after the other, with the
graphics card cooled down before each run.  Speeds on one PC vary by about 1% from day to day, which is why 0.93's
own release card is a little higher than its run here.  The 128 GB column is the same PC with the expert memory a 128 GB PC gets.  Greedy means temperature 0 in the
request; the default sampling is what a client gets when it sends no temperature.  Sampling accepts fewer of the
drafter's guesses, so it writes a little slower (0.93: ds4 Q2 40.7 -> 40.3 tok/s, UD-IQ2_XXS 41.1 -> 39.7; 0.94: UD-IQ2_XXS 41.7 -> 41.3).  0.93 narrowed
that gap from ~12%: the drafter's guesses are now
sampled instead of taken as its top pick (`STRATA_DSPARK_SAMPLE` in `Start DeepSeek.bat`); the answers'
distribution is unchanged.

**0.93 made Unsloth's UD-IQ2_XXS the default from 128 GB of RAM** (since 0.95 it is the pick from 110 GB, and UD-IQ1_M
below; the ds4 Q2 GGUF stays one answer away).  Measured with 0.93: on 192 GB it writes about as fast as the ds4 Q2 GGUF - 41.1 vs 40.7 tok/s greedy, 39.7 vs
40.3 at the default sampling - and on 128 GB about 5% slower (34.0 vs 35.9 tok/s, greedy).  It agrees with DeepSeek's own model more often: 91.1% vs 89.9% on general text, 94.5%
vs 93.1% on code and maths at the settings that ship ([MODELS.md](MODELS.md#other-quants-unsloth-ista-daslabs-gsq-rco-exl3)).

With less RAM, more of the model's experts are read from the SSD as they are needed. That is slower, most of all on
long prompts, but the output is the same.

## A second GPU

The Setup tab in the chat page has a Multi-GPU switch: a helper on a second card of 8 GB or more holds experts the
main card has no room for.  The main card needs 20 GB or more for this switch; the engine itself runs from 12 GB.  With
an RTX 3060 12 GB in an x4 slot beside the 3090 it adds a few percent: 0.94 on UD-IQ2_XXS writes 42.9 tok/s greedy
and 42.2 at the default sampling, against 41.7 / 41.3 on the 3090 alone (the same evening).  The second card must be an RTX 20
or newer; of two equal cards, the one on the wider PCIe link is used.  If the helper stops or cannot start, the model
still starts and runs on one card (the server window says so).  It does not split the model across cards, so it
does not scale: for two or more big cards use [Project Maya](https://github.com/mw00/project-maya), which runs
DeepSeek in tensor parallel.

## Smaller graphics cards and older CPUs

- An NVIDIA RTX card with 12 GB of VRAM or more, and driver 580 or newer.  24 GB (RTX 3090 / 4090, or a 32 GB 5090)
  is the fast path; with less, fewer experts fit on the card.  Measured with 0.93 (when the 24 GB card wrote 41 tok/s; 0.95 writes about 45) on the 3090 with the expert budget of
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

## Why the SSD matters

- **A fast NVMe SSD - its speed matters.**  The model is read from it while it runs, not just at the start: the Engram
  tables (~190 GB of the file) for every new piece of text, and the experts that do not fit in RAM on every token.
  Measured on this PC, the same model on a PCIe 4.0 NVMe (Corsair MP600 PRO, 5.4 GB/s) against a nearly full system
  SSD (0.8 GB/s): it starts in about 1 minute instead of 4-5, reads a new 4K-token prompt ~40% faster (168 vs 120
  tok/s) and writes 6-9% faster.  With 128 GB of RAM, more experts come from the SSD, so it matters more.  Use a PCIe
  4.0 or 5.0 NVMe with room to spare (a nearly full SSD slows down) - not a SATA SSD, a hard disk or a USB drive.
  The model files take 233-395 GB: about 320 GB free for UD-IQ2_XXS, 270 GB for UD-IQ1_M (a GSQ build adds its 125-159 GB).

## The first minutes after a reboot

The first minutes after a reboot are slower (measured on an early build: about half speed, ~13 instead of ~25
tok/s, on a cold start): the model's
  Engram tables are read from the SSD until Windows has cached the rows your text uses. It speeds up by itself as
  you use it. An optional Engram hot set (`engram-hot-2g.bin` in the model folder) keeps the most-used rows in RAM
  from the start; build it with `ds41_engram_hot.exe` - the header of `tools/v41/engine/ds41_engram_hot.cpp` explains
  how.
