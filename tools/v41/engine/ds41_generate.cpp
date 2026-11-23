// tools/v41/engine/ds41_generate.cpp - the Strata DeepSeek-V4.1-Flash engine end to end: Ds41Dense (attention,
// Engram, router, shared expert; CUDA or CPU ggml backend) + Ds41MoeTier (routed experts with per-expert formats:
// VRAM cache, pinned RAM arena, file tier, CPU pool, PCIe share) + sampling.  Token ids in, token ids out (text is
// the server's job: tools/v41/serve/serve_ds41.py).
//
//   ds41_generate -m V41-GSQ.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu] [--exl3 DIR]
//                 [--engram DIR:DIR:DIR | --engram MODEL.gguf] [--engram-hot hot.bin] [--slots auto|N] [--vram-margin GiB] [--arena-gib 60] [--rank r.bin]
//                 [--vram-lru] [--arena-adapt] [--pcie 0.25] [--tier-batch-pcie 1.0] [--prefill-chunk N] [--prompt-batch B] [--chunk-mmq] [--ctx N]
//                 [--temp 0] [--seed 1] [--stop id,id] [--ppl] [--no-actquant] [--kv-fp4] [--serve] [--quiet]
//                 [--enc-skip|--no-enc-skip] [--enc-replay N] [--dump-last F] [--gpus 0,1] [--gpu-split L] [--gpu-margin 1.5]
// --gpus A,B,.. (the expert layer split, opt-in): the dense half, KV and head on card A; each further card holds the
// routed-expert tier (VRAM cache + its RAM arena share) of a contiguous range of layers (by free VRAM, or --gpu-split
// = the first layer of each further card's range).  Without it: one card, as before.
// --enc-skip (layer-major batches only; DEFAULT for one-shot runs, --no-enc-skip = full path): prompt tokens before the last --enc-replay (default 2560 = 128 * 20) stop
// after layer 20's persistent-KV writes; only the last 2560 tokens run layers 21-39.  Same last-token logits / decode.
//
// --serve speaks Strata's engine line protocol (~/AI/Strata/serve/server.py StrataEngine, as glm_generate):
//   in:  GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. seed=..] <ids csv> | STOP | QUIT
//        GENI <max_new> [k=v ..] <image embeddings file> <ids csv>   (GEN with vision; see --image-token)
//   out: READY <ctx> stop, RESUME <reused>, PP <read> <total> <ms> <tok/s>, T <id>, DONE ..., ERR <text>
// Conversation reuse: a request that extends what the state has seen continues; one that shares the last prompt
// restores the snapshot taken at its end (Ds41Dense::snapshot: every other cache is position-indexed); the same prompt
// again (regenerate) restores it and samples from the saved last-row logits without a prefill.  VRAM: a multi-token
// prompt runs with the expert cache closed (its buffers need that VRAM) and the decode after it re-seeds the cache from
// the measured free memory (~0.6 s), as a one-shot run; the decode biases (--route-bias / --route-bias-file) apply only
// while decoding.
//
// Load the real model only through tools/ds4/memguard.sh (RAM cap + the shared GPU lock).
#include "ds41_dense.hpp"
#include "ds41_moe.hpp"
#include "ds41_posix.hpp"
#include "ds41_env.hpp"
#include "ds41_unmap.hpp"
#include "ds41_lookup.hpp"
#if defined(DS41_HAVE_DSPARK)
#include "ds41_dspark.hpp"
#include "ds41_lgate.hpp"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <array>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, rank, dump_logits, dump_last, dump_topk, kl_ref;
    std::string score_cases, score_out;   // --score-cases / --score-out: teacher-forced continuations (official_score.py)
    std::string build_mirror;      // --build-mirror F [--mirror-gib N]: write the blob mirror (coldest experts by --rank) and exit
    std::string tier_replay;       // --tier-replay F [--tier-replay-reps N]: replay a STRATA_TIER_RECORD file through the
    int tier_replay_reps = 3;      //   tier after the seed (exact old-vs-new kernel check + timing, no decode) and exit
    double mirror_gib = 85.0;
    std::string exl3;             // --exl3 DIR: the routed experts from an EXL3 checkpoint (LibertAIDAI's; Delta). --experts cpu: CPU tier only
    std::string prune;            // --prune F: "layer expert" lines (coldest first - v41_prune_list.py), soft or hard pruned
    float prune_penalty = 0.0f;   // --prune-penalty X: selection bias -X on the listed experts (0 = masked out of routing)
    float route_bias_file = 0.6f;    // --route-bias-file X: selection penalty -X on file-tier experts (neither VRAM nor RAM arena);
                                     // DEFAULT 0.6 (Mal: max decode w/o too much loss): 21.1 vs 15.1 tok/s, code ppl 2.67->2.87, agentic 6.88->6.39
    int   bias_prompt = 0;          // --bias-prompt N: quality proxy - apply route bias during the prompt, treating the first N
                                    // ranked experts as VRAM-resident (what the deferred seed will cache)
    float route_bias = 1.0f;      // --route-bias X: selection bonus +X on the experts resident in VRAM at the seed (DEFAULT 1.0, Mal 2026-10-11:
                                  // ~4x decode; 4096-tok ppl code 2.38 -> 2.67, agentic 15.1 -> 6.9; --route-bias 0 = exact routing)
    std::vector<std::string> engram;
    std::string engram_hot;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;            // 0 = auto (free VRAM after the dense half, minus --vram-margin)
    // defaults = the measured best decode config on the 4090 (2026-10-11): --pcie 0.75 --pf-b 1.43 --vram-lru
    double vram_margin_gib = 1.0, pcie = 0.75, arena_gib = 60.0, pf_b = 1.43;
    // 0.95 --prompt-margin GiB: the VRAM the PROMPT path keeps free (the keep-cache borrow, the batch width, the sweep
    // budget) when it differs from --vram-margin, which then sizes only the expert slots the decode keeps.  < 0 (not
    // given) = --vram-margin everywhere, as before.  The launcher gives it for files with bigger experts (UD-IQ3_XXS:
    // 3 GiB for the prompt path, the config's 0.6 for the slots) - see ds41_local.big_expert_margins.
    double prompt_margin_gib = -1.0;
    double prompt_margin() const { return prompt_margin_gib >= 0.0 ? prompt_margin_gib : vram_margin_gib; }
    double tier_batch_pcie = -1.0;   // < 0 = STRATA_TIER_BATCH_PCIE / 1.0
    int threads = 0, prefill_chunk = 0, prompt_batch = 8192;   // prompt_batch: layer-major batch B (0 = the chunk path)
    int ppl_step = 0;                // --ppl-step M: --ppl scores positions 1.. through step() (M = 1) or pass_v (M > 1)
    int64_t ctx = 0;
    float temp = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool ppl = false, serve = false, quiet = false, vram_lru = true, arena_adapt = false, chunk_mmq = true;
    bool act_quant = true, chunk_prefetch = true, kv_fp4 = false;
    int  enc_skip = -1;           // -1 = default (on for one-shot layer-major runs), 1 = --enc-skip, 0 = --no-enc-skip; only run the tokens before the replay window through layers 21-39
    int  enc_replay = 2560;       // --enc-replay N: the replay window (window_size * (n_layer - cand_layer) = 128*20)
    float skip_miss = 0.0f, skip_file = 0.0f;
    double chunk_prefetch_gib = 0.0;
    int image_token = 129264;     // --image-token N: the model's <|image|> id (each image is one run of it in the prompt)
    bool vision = false;
    std::vector<int> gpus;        // --gpus: CUDA devices, the dense card first (empty / one = one card)
    std::vector<int> gpu_split;   // --gpu-split: the first layer of each further card's range (empty = by free VRAM)
    double gpu_margin_gib = 1.5;  // --gpu-margin: VRAM a further card keeps free (its prompt-chunk buffers)
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

// The image-embedding file of a GENI line: one or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then
// n * n_embd float32) in prompt order (serve/server.py combined_embeddings_path; generate.cpp reads the same).
static bool read_sve1(const std::string & path, std::vector<std::vector<float>> & rows, std::vector<int64_t> & counts,
                      int64_t n_embd, int64_t max_rows, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open the embeddings file"; return false; }
    int32_t hdr[5];
    int64_t total = 0;
    for (;;) {
        f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
        if (f.gcount() == 0 && f.eof()) break;
        if (f.gcount() != sizeof(hdr)) { err = "short embeddings header"; return false; }
        if (hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || hdr[4] != (int32_t) n_embd
            || (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[1] > max_rows - total) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            return false;
        }
        const int64_t n = hdr[1];
        total += n;
        std::vector<float> r((size_t) (n * n_embd));
        if (f.read(reinterpret_cast<char *>(r.data()), (std::streamsize) (r.size() * 4)).gcount()
                != (std::streamsize) (r.size() * 4)) { err = "short embeddings file"; return false; }
        rows.push_back(std::move(r));
        counts.push_back(n);
    }
    if (rows.empty()) { err = "the embeddings file has no records"; return false; }
    return true;
}

// Map records onto image-token runs, using the row counts to split adjacent images within the same run.
static bool map_vision(const std::vector<int> & ids, int image_token, const std::vector<std::vector<float>> & rows,
                       const std::vector<int64_t> & counts, std::vector<float> & vis_rows,
                       std::vector<int64_t> & vis_pos, std::string & err) {
    size_t k = 0;
    for (size_t i = 0; i < ids.size(); ) {
        if (ids[i] != image_token) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == image_token) ++j;
        const int64_t run = (int64_t) (j - i);
        if (k >= rows.size()) { err = "the prompt has more images than the embeddings file"; return false; }
        int64_t matched = 0;
        while (matched < run && k < rows.size()) {
            if (counts[k] > run - matched) { err = "an image span does not match its embeddings"; return false; }
            matched += counts[k];
            vis_rows.insert(vis_rows.end(), rows[k].begin(), rows[k].end());
            ++k;
        }
        if (matched != run) { err = "an image span is missing embeddings"; return false; }
        for (size_t t = i; t < j; ++t) vis_pos.push_back((int64_t) t);
        i = j;
    }
    if (k != rows.size()) { err = "the embeddings file has more images than the prompt"; return false; }
    return true;
}

bool parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--engram-hot") a.engram_hot = next();
#ifdef _WIN32
        // the Windows path-list separator: a drive letter's ':' must not split "D:\a;E:\b"
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ';')) a.engram.push_back(d); }
#else
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ':')) a.engram.push_back(d); }
#endif
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--prompt-margin") a.prompt_margin_gib = std::max(0.0, std::atof(next().c_str()));
        else if (k == "--gpus") a.gpus = parse_csv(next());
        else if (k == "--gpu-split") a.gpu_split = parse_csv(next());
        else if (k == "--gpu-margin") a.gpu_margin_gib = std::atof(next().c_str());
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--rank") a.rank = next();
        else if (k == "--build-mirror") a.build_mirror = next();
        else if (k == "--tier-replay") a.tier_replay = next();
        else if (k == "--tier-replay-reps") a.tier_replay_reps = std::atoi(next().c_str());
        else if (k == "--check-mirror") { a.build_mirror = next(); a.mirror_gib = -1; }
        else if (k == "--mirror-gib") a.mirror_gib = std::atof(next().c_str());
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--no-vram-lru") a.vram_lru = false;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--tier-batch-pcie") a.tier_batch_pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());
        // prompt experts on MMQ (int8 tensor cores; PTQ1_0 through its lossless Q2_0 transcode) - DEFAULT since
        // 2026-10-12: 8K prompt 219-220 -> 244-248 tok/s, 4096 code ppl 2.3799 -> 2.3790, KL 7.6e-3, deterministic
        // (8K --dump-last md5 ee31507a0e88; the MMVQ path's was 4d229102872a).  --no-chunk-mmq = the MMVQ kernels.
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--no-chunk-mmq") a.chunk_mmq = false;
        else if (k == "--no-chunk-prefetch") a.chunk_prefetch = false;
        else if (k == "--chunk-prefetch") a.chunk_prefetch_gib = std::atof(next().c_str());
        else if (k == "--skip-miss") a.skip_miss = (float) std::atof(next().c_str());
        else if (k == "--skip-file") a.skip_file = (float) std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--ppl-step") a.ppl_step = std::atoi(next().c_str());
        else if (k == "--no-actquant") a.act_quant = false;
        else if (k == "--kv-fp4") a.kv_fp4 = true;
        else if (k == "--serve") a.serve = true;
        else if (k == "--exl3") a.exl3 = next();
        else if (k == "--prune") a.prune = next();
        else if (k == "--prune-penalty") a.prune_penalty = (float) std::atof(next().c_str());
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--bias-prompt") a.bias_prompt = std::atoi(next().c_str());
        else if (k == "--route-bias-file") a.route_bias_file = (float) std::atof(next().c_str());
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-topk") a.dump_topk = next();
        else if (k == "--kl-ref") a.kl_ref = next();
        else if (k == "--score-cases") a.score_cases = next();
        else if (k == "--score-out") a.score_out = next();
        else if (k == "--dump-last") a.dump_last = next();
        else if (k == "--enc-skip") a.enc_skip = 1;
        else if (k == "--no-enc-skip") a.enc_skip = 0;
        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
        else if (k == "--image-token") a.image_token = std::atoi(next().c_str());
        else if (k == "--vision") a.vision = true;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty() || !a.build_mirror.empty() ||
                                !a.tier_replay.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// STRATA_DS41_HOST_PIN=1 (opt-in, OFF by default until an A/B shows it; upstream b87654e6, PR #131 --host-core;
// BRAINSTORM X10): the executor thread - the one that launches the graphs and spins on the CPU pool and its flags - is
// placed on the core the pool leaves free for it (strata::kernels::cpu::planned_host_core(): the pool's workers are
// physical_cores(true), i.e. every physical core but that one), as the Qwen path does once per session
// (src/core/session.cpp SessionLoopScratch).  Unpinned, Windows may run the spinning host on a worker's core or its
// SMT sibling (the Qwen pool measured 26.9 vs 36.3 GB/s), or on an E-core.  pin_current_thread() is the Windows CPU
// Set selection (SetThreadSelectedCpuSets: soft, reversible, inside any hard affinity) / Linux sched affinity; the
// destructor restores the previous placement.  Placement only: no result changes.  Pinned after load (serve: before
// READY; one-shot: before the prompt), so the loader's threads and the pool keep the affinity they were created with.
struct Ds41HostPin {
    strata::kernels::cpu::ThreadAffinity previous;
    bool done = false;
    ~Ds41HostPin() { strata::kernels::cpu::restore_thread_affinity(previous); }
    void pin() {
        if (done) return;
        done = true;
        const char * e = std::getenv("STRATA_DS41_HOST_PIN");
        if (!e || !e[0] || e[0] == '0') return;
        const int core = strata::kernels::cpu::planned_host_core();
        if (core < 0) { std::fprintf(stderr, "ds41_generate: STRATA_DS41_HOST_PIN: no CPU topology, the executor stays unpinned\n"); return; }
        previous = strata::kernels::cpu::pin_current_thread(core);
        if (previous.valid) std::fprintf(stderr, "ds41_generate: executor pinned to logical processor %d (STRATA_DS41_HOST_PIN)\n", core);
    }
};

// The K best ids of a logits row, best first: idx[0..K) (idx may be longer).  The reference is the partial_sort below
// over all n ids (comparator logits[x] > logits[y]).  STRATA_FAST_TOPK (default on, =0 always the reference): a
// two-pass pick instead - the K-th largest of >= 2K block maxima is a lower bound t on the K-th largest logit (K
// blocks each hold an element >= t), so the K best are among {i : logits[i] >= t} (typically a few hundred ids of
// 129,280), and only those are sorted.  It is used ONLY when its answer is unique: no NaN, the K best strictly
// decreasing and strictly above the (K+1)-th.  Then any correct partial_sort returns exactly that order, so the ids,
// the sampled tokens and the output are bit-identical; a tie or a NaN anywhere that matters takes the reference path
// (whose tie order is the library's).  Saves the 129,280-entry index vector and the heap walk over it on every
// verified row (sample_dist) and every sampled token (sample_p).
void top_candidates(const float * logits, int n, int K, std::vector<int> & idx) {
    static const bool fast = !(ds41_env("STRATA_FAST_TOPK", "STRATA_P3_HOST_TOPK") && ds41_env("STRATA_FAST_TOPK", "STRATA_P3_HOST_TOPK")[0] == '0');
    if (fast && K >= 1 && n >= 8 * K) {
        thread_local std::vector<float> bm;
        thread_local std::vector<int> cand;
        const int bs = n / (2 * K);    // >= 4
        const int nb = n / bs;         // full blocks only: nb >= 2K; the tail is still scanned below
        bm.resize((size_t) nb);
        for (int b = 0; b < nb; ++b) {
            const float * r = logits + (size_t) b * (size_t) bs;
            float m = -INFINITY;       // a NaN never wins (it is caught in the scan below)
            for (int j = 0; j < bs; ++j) m = r[j] > m ? r[j] : m;
            bm[(size_t) b] = m;
        }
        std::nth_element(bm.begin(), bm.begin() + (K - 1), bm.end(), [](float a, float b) { return a > b; });
        const float t = bm[(size_t) (K - 1)];
        cand.clear();
        bool nan = false;
        for (int i = 0; i < n; ++i) {
            const float v = logits[i];
            if (v >= t) cand.push_back(i);
            else if (v != v) { nan = true; break; }
        }
        const int C = (int) cand.size();
        if (!nan && C >= K && C <= 64 * K) {
            const int S = std::min(C, K + 1);
            std::partial_sort(cand.begin(), cand.begin() + S, cand.end(), [&](int x, int y) { return logits[x] > logits[y]; });
            bool unique = true;
            for (int j = 0; j + 1 < S; ++j)
                if (!(logits[cand[(size_t) j]] > logits[cand[(size_t) j + 1]])) { unique = false; break; }
            if (unique) {
                idx.assign(cand.begin(), cand.begin() + K);
                return;
            }
        }
    }
    idx.resize((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best); temp 0 = argmax
int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx;
    top_candidates(logits, n_vocab, K, idx);
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

// the distribution sample_p draws from, as (token, probability) pairs summing to 1 (temp 0: the argmax alone) - the
// trunk's p for --serve's speculative sampling (same candidates, same weights as sample_p)
void sample_dist(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p,
                 std::vector<std::pair<int, double>> & out) {
    out.clear();
    if (temp <= 0.0f) { out.push_back({ (int) (std::max_element(logits, logits + n_vocab) - logits), 1.0 }); return; }
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx;
    top_candidates(logits, n_vocab, K, idx);
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    for (int i = 0; i < n; ++i) out.push_back({ idx[(size_t) i], p[(size_t) i] / z });
}

// Speculative sampling with a DETERMINISTIC draft d (DSpark drafts by argmax: q = a point mass on d).  Leviathan /
// Chen: accept with min(1, p(d) / q(d)) = p(d); on rejection sample from norm(max(0, p - q)) = p without d.  The
// emitted token is then distributed exactly as p - the same as plain sampling.  true = d accepted; false = `out` is
// the replacement token.
bool spec_accept(const std::vector<std::pair<int, double>> & dist, int d, std::mt19937_64 & rng, int & out) {
    double pd = 0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (pd > 0 && u(rng) < pd) return true;
    double r = u(rng) * (1.0 - pd);
    int last = -1;
    for (const auto & c : dist) {
        if (c.first == d) continue;
        last = c.first;
        if ((r -= c.second) <= 0) { out = c.first; return false; }
    }
    out = last >= 0 ? last : d;   // rounding: the last candidate (d only if it was the sole candidate - then pd was 1)
    return false;
}

// Speculative sampling with a SAMPLED draft d ~ q (STRATA_DSPARK_SAMPLE: Ds41DSpark::step_sampled; q(x) = cdf[x] -
// cdf[x - 1] over the vocabulary, the last token taking the rest up to 1, exactly the intervals the device's inverse-CDF
// draw used): accept with min(1, p(d) / q(d)); on rejection sample from norm(max(0, p - q)).  The emitted token is
// distributed exactly as p (Leviathan / Chen), the same as plain sampling; a draft that matches a flat p is kept with
// probability sum min(p, q) instead of the argmax draft's p(d).
bool spec_accept_q(const std::vector<std::pair<int, double>> & dist, int d, const float * cdf, int64_t V,
                   std::mt19937_64 & rng, int & out) {
    auto q = [&](int x) -> double {
        if (x < 0 || x >= V) return 0.0;
        const double hi = x == V - 1 ? 1.0 : std::min(1.0, (double) cdf[x]);
        const double lo = x == 0 ? 0.0 : std::min(1.0, (double) cdf[x - 1]);
        return std::max(0.0, hi - lo);
    };
    double pd = 0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (pd > 0 && u(rng) * q(d) < pd) return true;
    double z = 0;
    for (const auto & c : dist) z += std::max(0.0, c.second - q(c.first));
    double r = u(rng) * (z > 0 ? z : 1.0);
    int last = d;
    for (const auto & c : dist) {
        const double w = z > 0 ? std::max(0.0, c.second - q(c.first)) : c.second;   // z = 0: p <= q everywhere (p = q up to rounding)
        if (w <= 0) continue;
        last = c.first;
        if ((r -= w) <= 0) break;
    }
    out = last;
    return false;
}

// Attention-sink watch (--serve; spark-ds41f's sink rejection v2, experiments/2026-10-19-sink-rejection-v2).  V4.1 Flash
// sometimes turns a sampled token into an attention sink: column 2455 of the residual runs away from layer 20 on and
// the next-token distribution becomes noise.  Ds41Dense::sink_value reads max over the 4 hc streams of |x[2455]|
// after the last layer for each logits row.  Both switches default off (no readback, no change to anything):
//   STRATA_SINK_TRACE=1           one stderr line per verify row / decode step / prompt's last row:
//                                 "sink: rq R pos P tok T L20 <after layer 20> Llast <after the last layer> <tag>"
//                                 (tag e = the row's token was kept, x = a rejected draft, sunk = held token replaced,
//                                 prompt = the last prompt row; -1 = not captured)
//   STRATA_SINK_REJECT=<thr>      the correction: each pass's last sampled token is held back (not emitted) until the
//                                 next verify pass has run it as row 0; a row over thr replaces it by the next
//                                 candidate of the draw that produced it (STRATA_SINK_REJECT_TRIES, default 3, per
//                                 position; past them it is kept: "fallthrough"); a draft whose row is over thr cuts
//                                 the verification there (the row before it draws without that draft).  A stop /
//                                 max-tokens / end-of-context token is emitted unchecked (nothing verifies it).
struct SinkOpts { bool trace = false; double thr = 0.0; int tries = 3; };
const SinkOpts & sink_opts() {
    static const SinkOpts o = [] {
        SinkOpts c;
        const char * tr = std::getenv("STRATA_SINK_TRACE");
        c.trace = tr && tr[0] && tr[0] != '0';
        if (const char * e = std::getenv("STRATA_SINK_REJECT")) c.thr = std::max(0.0, std::atof(e));
        if (const char * e = std::getenv("STRATA_SINK_REJECT_TRIES")) c.tries = std::max(0, std::atoi(e));
        return c;
    }();
    return o;
}

// the next candidate of a draw from `row` (the logits it was drawn from) once the tokens in `excl` are taken out:
// temp 0 = the best remaining logit; else a draw from sample_dist's distribution without them (sampling without
// replacement).  When the sampler's own set (top-k / top-p / min-p) holds nothing else - a confident sunk token is the
// whole set under the server's default top_p 0.95 (measured: every "kept after 0 tries" fallthrough and every draft
// cut that never fired) - the best remaining logit, *outside = true.  -1 = no token left at all.
int sink_alt(const float * row, int nv, float temp, int top_k, float top_p, float min_p, const std::vector<int> & excl,
             std::mt19937_64 & rng, bool * outside = nullptr, const std::vector<uint8_t> * ban = nullptr) {
    // excluded: the tried tokens, and (ban) every control token and every token whose text spells a control marker
    auto out = [&](int x) {
        if (ban && x >= 0 && x < (int) ban->size() && (*ban)[(size_t) x]) return true;
        return std::find(excl.begin(), excl.end(), x) != excl.end();
    };
    auto best_left = [&]() {
        int best = -1;
        for (int i = 0; i < nv; ++i)
            if (!out(i) && (best < 0 || row[i] > row[best])) best = i;
        return best;
    };
    if (outside) *outside = false;
    if (temp <= 0.0f) return best_left();
    std::vector<std::pair<int, double>> dist;
    sample_dist(row, nv, temp, top_k, top_p, min_p, dist);
    double z = 0;
    for (const auto & c : dist) if (!out(c.first)) z += c.second;
    if (!(z > 0)) {
        if (outside) *outside = true;
        return best_left();
    }
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    int last = -1;
    for (const auto & c : dist) {
        if (out(c.first)) continue;
        last = c.first;
        if ((r -= c.second) <= 0) return c.first;
    }
    return last;
}

// STRATA_SINK_REJECT: the tokens the correction never touches.  1 = a control / special token (the GGUF's
// tokenizer.ggml.token_type is not NORMAL (1) or BYTE (6): <think>, </think>, EOS, BOS, the DSML / tool / role markers,
// ...): never rejected, never cut, never a replacement.  2 = a normal token whose text contains a control token's text
// (3+ characters): never a replacement.  Without the GGUF's tokenizer arrays: ids >= 128000 (DeepSeek V4.1's added
// tokens start there) and 0 / 1 / 2 (BOS / EOS / pad).  The caller adds its stop ids and the think ids.
std::vector<uint8_t> sink_special_set(const std::string & model, int nv) {
    std::vector<uint8_t> ban((size_t) std::max(nv, 0), 0);
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(model.c_str(), gp);
    bool typed = false;
    if (g) {
        const int64_t kt = gguf_find_key(g, "tokenizer.ggml.token_type");
        const int64_t ks = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (kt >= 0 && gguf_get_arr_type(g, kt) == GGUF_TYPE_INT32) {
            const size_t n = std::min(gguf_get_arr_n(g, kt), ban.size());
            const int32_t * ty = (const int32_t *) gguf_get_arr_data(g, kt);
            for (size_t i = 0; i < n; ++i) ban[i] = ty[i] != 1 && ty[i] != 6 ? 1 : 0;
            typed = true;
            if (ks >= 0 && gguf_get_arr_type(g, ks) == GGUF_TYPE_STRING) {
                const size_t nt = std::min(gguf_get_arr_n(g, ks), ban.size());
                std::vector<std::string> marks;
                for (size_t i = 0; i < nt; ++i)
                    if (ban[i] == 1) {
                        std::string m = gguf_get_arr_str(g, ks, i);
                        if (m.size() >= 3) marks.push_back(std::move(m));
                    }
                for (size_t i = 0; i < nt; ++i) {
                    if (ban[i]) continue;
                    const std::string tx = gguf_get_arr_str(g, ks, i);
                    if (tx.size() < 3) continue;
                    for (const auto & m : marks)
                        if (tx.find(m) != std::string::npos) { ban[i] = 2; break; }
                }
            }
        }
        gguf_free(g);
    }
    if (!typed) {
        for (int i = 128000; i < nv; ++i) ban[(size_t) i] = 1;
        for (int i = 0; i < 3 && i < nv; ++i) ban[(size_t) i] = 1;
    }
    long c1 = 0, c2 = 0;
    for (uint8_t b : ban) { c1 += b == 1; c2 += b == 2; }
    std::fprintf(stderr, "sink: %ld control tokens exempt, %ld more never a replacement (%s)\n", c1, c2,
                 typed ? "the GGUF's token types" : "fallback: ids >= 128000 and 0-2");
    return ban;
}
// q(x) of a sampled draft row from its cumulative distribution (as spec_accept_q's)
double draft_q(const float * cdf, int64_t V, int x) {
    if (x < 0 || x >= V) return 0.0;
    const double hi = x == V - 1 ? 1.0 : std::min(1.0, (double) cdf[x]);
    const double lo = x == 0 ? 0.0 : std::min(1.0, (double) cdf[x - 1]);
    return std::max(0.0, hi - lo);
}

// 0.95: online calibration of DSpark's confidence head for the serve loop's draft gate (STRATA_DSPARK_GATE /
// STRATA_DSPARK_CALIB).  Binned counts per (draft slot, raw-probability bin) with a prior: the calibrated probability is
// the raw sigmoid plus the bin's mean residual (label - raw) shrunk by a pseudo-count of 4, so it starts as the identity
// and moves only where the data says (spark-ds41f 0746 found the raw head ~3x optimistic at the tail slots).  Labels are
// the acceptance probability of a VERIFIED draft given the drafts before it - min(1, p(d) / q(d)) for a sampled draft,
// p(d) for an argmax one (the expectation of the sampled accept outcome, so less noisy); a draft never verified is not
// an observation.  Counts halve past 512 per bin, so the estimate follows a drifting head.
struct DraftCalib {
    static constexpr int kPos = 5, kBins = 8;
    double n[kPos][kBins] = {}, sy[kPos][kBins] = {}, sr[kPos][kBins] = {};
    double mean[kPos] = { 0.75, 0.75, 0.75, 0.75, 0.75 };   // per-slot EMA of the label (a draw-free survival estimate)
    long obs = 0;
    static double raw(float c) { return 1.0 / (1.0 + std::exp(-(double) c)); }
    static int bin(double r) {
        static const double e[kBins - 1] = { 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97 };
        int b = 0;
        while (b < kBins - 1 && r >= e[b]) ++b;
        return b;
    }
    double prob(int j, float c) const {
        const double r = raw(c);
        const int p = std::max(0, std::min(kPos - 1, j)), b = bin(r);
        return std::max(0.001, std::min(0.999, r + (sy[p][b] - sr[p][b]) / (n[p][b] + 4.0)));
    }
    void add(int j, float c, double y) {
        if (j < 0 || j >= kPos) return;
        const double r = raw(c);
        const int b = bin(r);
        n[j][b] += 1.0; sy[j][b] += y; sr[j][b] += r;
        if (n[j][b] > 512.0) { n[j][b] *= 0.5; sy[j][b] *= 0.5; sr[j][b] *= 0.5; }
        mean[j] += (y - mean[j]) / 64.0;
        ++obs;
    }
};

// 0.95: the gate's price of a verify row.  lambda = EMA (1/16) of tokens per ms over real passes (tokens emitted per
// pass / the pass's ms, as pico-dgx-ds4's SpecRate); deltaC = the marginal ms of one more row, the slope of a decayed
// least-squares fit of pass ms on pass rows plus a fixed prior of 8 pseudo-passes on 52.6 + 6.8 w (the 3090's ~6.8 ms
// per extra row, ~73 ms a pass), clamped to [1, 40] ms.  An outlier pass (a keep-cache regrow, a page-in) is clipped
// to 4x the running mean first.
struct GateCost {
    double tok = 3.0, ms = 80.0;
    double s0 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    void add(int rows, int tokens, double pms) {
        if (!(pms > 0)) return;
        pms = std::min(pms, 4.0 * ms);
        tok += (tokens - tok) / 16.0;
        ms += (pms - ms) / 16.0;
        const double d = 0.97, x = (double) rows;
        s0 = d * s0 + 1.0; sx = d * sx + x; sy = d * sy + pms; sxx = d * sxx + x * x; sxy = d * sxy + x * pms;
    }
    double lambda() const { return tok / std::max(1.0, ms); }
    double dc() const {
        // prior: 4 pseudo-passes at 2 rows (66.2 ms), 4 at 7 rows (100.2 ms)
        const double S = s0 + 8.0, X = sx + 4.0 * 2 + 4.0 * 7, Y = sy + 4.0 * 66.2 + 4.0 * 100.2,
                     XX = sxx + 4.0 * 4 + 4.0 * 49, XY = sxy + 4.0 * 2 * 66.2 + 4.0 * 7 * 100.2;
        const double den = S * XX - X * X;
        const double b = den > 1e-9 ? (S * XY - X * Y) / den : 6.8;
        return std::max(1.0, std::min(40.0, b));
    }
};

double log_softmax_at(const float * lg, int nv, int tok) {
    const float mx = *std::max_element(lg, lg + nv);
    double z = 0;
    for (int i = 0; i < nv; ++i) z += std::exp((double) lg[i] - mx);
    return (double) lg[tok] - mx - std::log(z);
}

}  // namespace

#if defined(_WIN32)
// DS41_CRASH_TRACE=1 (Windows): print the faulting stack (needs the PDB next to the exe for names / lines)
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
static LONG WINAPI ds41_veh(EXCEPTION_POINTERS * ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != 0xC0000374 && code != STATUS_STACK_BUFFER_OVERRUN &&
        code != EXCEPTION_ILLEGAL_INSTRUCTION && code != 0xC0000409)
        return EXCEPTION_CONTINUE_SEARCH;
    static volatile LONG once = 0;
    if (InterlockedExchange(&once, 1)) return EXCEPTION_CONTINUE_SEARCH;
    const HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
    char exe_dir[MAX_PATH] = "";
    GetModuleFileNameA(nullptr, exe_dir, MAX_PATH);
    if (char * sl = std::strrchr(exe_dir, '\\')) *sl = 0;
    SymInitialize(proc, exe_dir, TRUE);
    std::fprintf(stderr, "ds41 crash: exception 0x%08lx at %p, thread %lu\n", (unsigned long) code,
                 ep->ExceptionRecord->ExceptionAddress, (unsigned long) GetCurrentThreadId());
    void * fr[62];
    const USHORT n = RtlCaptureStackBackTrace(0, 62, fr, nullptr);   // the handler's own frames, then the faulting stack
    alignas(SYMBOL_INFO) char sb[sizeof(SYMBOL_INFO) + 512];
    for (USHORT i = 0; i < n; ++i) {
        const DWORD64 pc = (DWORD64) fr[i];
        SYMBOL_INFO * s = (SYMBOL_INFO *) sb;
        s->SizeOfStruct = sizeof(SYMBOL_INFO); s->MaxNameLen = 511;
        DWORD64 d = 0; DWORD dl = 0; IMAGEHLP_LINE64 ln{}; ln.SizeOfStruct = sizeof ln;
        const bool hs = SymFromAddr(proc, pc, &d, s);
        const bool hl = SymGetLineFromAddr64(proc, pc, &dl, &ln);
        char mod[MAX_PATH] = "?";
        HMODULE hm = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) pc, &hm))
            GetModuleFileNameA(hm, mod, MAX_PATH);
        const char * base = std::strrchr(mod, '\\');
        std::fprintf(stderr, "  #%02u %s+0x%llx %s  %s:%lu\n", (unsigned) i, base ? base + 1 : mod,
                     (unsigned long long) (pc - (DWORD64) hm), hs ? s->Name : "?",
                     hl ? ln.FileName : "", hl ? (unsigned long) ln.LineNumber : 0ul);
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int main(int argc, char ** argv) {
    // STRATA_X3B: eager CUDA module loading, set before CUDA starts (11-18).  CUDA 12.2+ loads a kernel
    // lazily at its first launch, and that load may need the context to synchronize (CUDA's "Lazy Loading" notes: a
    // kernel spin-waiting on work in another stream can deadlock with it).  An armed X3B gate is exactly that: the gate
    // (spin kernel or stream wait) holds the dense stream for work the host issues afterwards.  A kernel launched for
    // the first time inside the window - the release kernel itself at the run's first gate, a tier kernel first used
    // after a batch prompt - then waited for the gate, and the gate for it, until the watchdog opened it ("the tier's
    // sum had NOT landed, the release kernel had NOT run").  EAGER loads every kernel at start-up.  Only with
    // STRATA_X3B=1, and never over a CUDA_MODULE_LOADING the caller set.
    if (const char * xb = std::getenv("STRATA_X3B"); xb && xb[0] == '1' && !std::getenv("CUDA_MODULE_LOADING")) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif
        std::fprintf(stderr, "ds41_generate: STRATA_X3B: CUDA_MODULE_LOADING=EAGER (no lazy kernel load inside an armed gate)\n");
    }
#if defined(_WIN32)
    if (std::getenv("DS41_CRASH_TRACE")) AddVectoredExceptionHandler(1, ds41_veh);
    // An abort() the engine does not route through ds41_fatal_exit (ggml's GGML_ABORT / CUDA_CHECK,
    // a library assert) ends without the abort message box and without the Windows Error Reporting pass, which on a
    // process with 100+ GB pinned takes minutes.  The exit code stays 3 (SIGABRT's default on Windows).
    // STRATA_FATAL_ABORT=1 keeps the CRT default (a WER dump for a debugger).
    if (!(ds41_env("STRATA_FATAL_ABORT", "STRATA_P3_HOST_ABORT") && ds41_env("STRATA_FATAL_ABORT", "STRATA_P3_HOST_ABORT")[0] == '1'))
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    // DS41_PRIORITY=abovenormal|high: the engine raises its own priority class, so other programs cannot pause the
    // busy-waiting CPU pool (one paused worker stalls the whole pool - the "half-speed" runs).  The test scripts set it
    // from outside; this lets a launcher set just the variable (a child process does not inherit above-normal).
    if (const char * pr = std::getenv("DS41_PRIORITY")) {
        const DWORD cls = _stricmp(pr, "high") == 0 ? HIGH_PRIORITY_CLASS
                        : _stricmp(pr, "abovenormal") == 0 ? ABOVE_NORMAL_PRIORITY_CLASS : 0;
        if (cls && SetPriorityClass(GetCurrentProcess(), cls)) std::fprintf(stderr, "ds41_generate: priority class %s\n", pr);
    }
    // STRATA_TIMER_MS=N (Maya #358): the system timer at N ms while the engine runs.  Since Windows 10 2004 the
    // resolution is per process, so another program's 1 ms does not reach us: a sleep_for / timed wait rounds up to the
    // 15.6 ms default tick.  timeBeginPeriod lives in winmm, taken at run time (no import library to link).
    if (const char * tm = std::getenv("STRATA_TIMER_MS")) {
        const UINT ms = (UINT) std::max(1, std::atoi(tm));
        using Tbp = UINT (WINAPI *)(UINT);
        if (HMODULE wm = LoadLibraryA("winmm.dll"))
            if (Tbp tbp = (Tbp) (void *) GetProcAddress(wm, "timeBeginPeriod"))
                std::fprintf(stderr, "ds41_generate: timer %u ms (%s)\n", ms, tbp(ms) == 0 ? "ok" : "refused");
    }
    // DS41_QOS=high: opt out of Windows power throttling (EcoQoS) for the whole process.  A windowless engine counts as
    // background work while the user is busy in another program, and on a hybrid CPU that can mean low-importance
    // scheduling (E-cores) whatever the priority class (measured: one-shots at half speed - CPU pool, drafter and
    // Engram all 2x slower - from the moment Firefox was in use, at above-normal priority).
    if (const char * q = std::getenv("DS41_QOS"); q && _stricmp(q, "high") == 0) {
        PROCESS_POWER_THROTTLING_STATE st {};
        st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;   // controlled ...
        st.StateMask = 0;                                            // ... and off
        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof st))
            std::fprintf(stderr, "ds41_generate: power throttling off (DS41_QOS=high)\n");
        else
            std::fprintf(stderr, "ds41_generate: DS41_QOS=high refused (error %lu)\n", (unsigned long) GetLastError());
    }
#endif
    strata::v41::ds41_cuda_schedule_from_env();   // STRATA_CUDA_SPIN=1 (before any CUDA context exists)
    if (!std::getenv("DS41_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: see tools/v41/engine/ds41_generate.cpp\n");
        return 2;
    }
    Ds41HostPin host_pin;   // STRATA_DS41_HOST_PIN: pinned after load (see Ds41HostPin); restored when main returns
    if (!a.build_mirror.empty()) {   // the second-drive blob mirror (STRATA_BLOB_MIRROR): coldest experts first
        std::string err;
        if (a.mirror_gib < 0) {   // --check-mirror F: open it with every blob compared against the model file
            ds41_sys::setenv("STRATA_BLOB_MIRROR", a.build_mirror.c_str(), 1);
            ds41_sys::setenv("STRATA_BLOB_MIRROR_CHECK", "0", 1);
            Ds41FileBlobs chk(a.model);
            if (!chk.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
            return chk.mirrored() > 0 ? 0 : 1;
        }
        Ds41FileBlobs src(a.model);
        if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
        if (a.rank.empty()) { std::fprintf(stderr, "ds41_generate: --build-mirror needs --rank\n"); return 2; }
        std::vector<std::pair<int32_t, int32_t>> order = ds41_read_ranked(a.rank, err);
        if (order.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        std::reverse(order.begin(), order.end());
        if (!src.build_mirror(a.build_mirror, order, (uint64_t) (a.mirror_gib * 1073741824.0), err)) {
            std::fprintf(stderr, "ds41_generate: build-mirror: %s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        prompt.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) prompt.data(), (std::streamsize) (prompt.size() * 4));
    }
    std::mt19937_64 rng(a.seed);

    // ---- dense half
    ggml_backend_t be = nullptr;
    if (a.backend == "cuda" && !a.gpus.empty()) {   // --gpus: the dense half on the first card listed (the n-th GPU device)
        for (size_t i = 0, g = 0; i < ggml_backend_dev_count() && !be; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if ((int) g++ == a.gpus[0]) be = ggml_backend_dev_init(d, nullptr);
        }
    } else {
        be = a.backend == "cuda" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                                 : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!be) { std::fprintf(stderr, "ds41_generate: no %s backend\n", a.backend.c_str()); return 1; }
    if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, a.threads > 0 ? a.threads : 8);
    int64_t ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 64;
    ctx = std::max<int64_t>(4096, (ctx + 255) / 256 * 256);
    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
    // the layer-major batch B: --prompt-batch, capped to the prompt length for a one-shot run (the state tensors hold
    // B columns, so a short prompt must not pay for the flag's full size; --serve sizes for the largest request)
    const int PB = a.prompt_batch > 0 ? (a.serve ? a.prompt_batch : (int) std::min<int64_t>(a.prompt_batch, (int64_t) prompt.size())) : 0;
    // the dense sub-batch size S: --prefill-chunk, or 512 with a layer-major prompt batch and no chunk size given
    const int S = a.prefill_chunk > 0 ? a.prefill_chunk : (a.prompt_batch > 0 ? 512 : 1);
    dc.max_pass = S;
    dc.prompt_batch = PB;
    // --serve: elastic state columns (STRATA_ELASTIC_COLS=0 = the fixed prompt_batch columns): decode keeps S columns and
    // the ~0.31 MiB a column of a big prompt batch (2.6 GiB at 8192) goes to VRAM expert slots; a layer-major batch
    // grows them (the VRAM cache is released for it anyway) and gives them back.  s11: 2.6 GiB of slots = 10-20% of a
    // verify pass (one-shot, margin 1.0 / 2.3 / 3.6: python 131 / 136 / 145 ms, code 89 / 93 / 107 ms).
    dc.elastic_cols = a.serve && PB > 1 && !(std::getenv("STRATA_ELASTIC_COLS") && std::getenv("STRATA_ELASTIC_COLS")[0] == '0');
    // encoder-only prompt prefill (enc-skip): tokens before the last `enc_replay` need their layers past layer 20 only
    // through the per-layer 128-token window, so for a long prompt they can stop after the KV-writing part of layer
    // 20 (the layer-major batch path only; the chunk path has no per-layer column split, --ppl/--dump-logits need
    // every row's logits, which the skipped rows do not have).
    // default ON (byte-identical last-row logits vs the full path, 8192 tokens, 2026-10-11) for layer-major runs, --serve
    // included (a continuation / snapshot restore only reads per-position caches + the last window of every layer,
    // which the replay window computes in full)
    bool enc_skip = a.enc_skip == 1 || (a.enc_skip < 0 && PB > 1);
    if (enc_skip && a.enc_replay < 1) { std::fprintf(stderr, "ds41_generate: --enc-replay must be >= 1\n"); return 2; }
    if (enc_skip && (a.ppl || !a.dump_logits.empty())) {
        if (a.enc_skip == 1) std::fprintf(stderr, "note: --enc-skip disabled (--ppl/--dump-logits need every row's logits, the old full path)\n");
        enc_skip = false;
    }
    if (enc_skip && PB <= 1) { std::fprintf(stderr, "ds41_generate: --enc-skip needs the layer-major prompt path (--prompt-batch >= 2)\n"); return 2; }
    int64_t ER = enc_skip ? (int64_t) a.enc_replay : 0;
    // STRATA_ENC_BOUND=N (0 = off; CHANGES OUTPUT): bounded replay, DeepSeek's own design.  Only a prompt's last N tokens
    // run the layers past the last kv source, and those layers' windows stop where the replay starts.  N stays at least
    // one window, so decode never reads a row these layers did not write.  The scorer applies the same limit.
    const char * enc_bound_env = std::getenv("STRATA_ENC_BOUND");
    int64_t enc_bound = enc_bound_env ? std::max<int64_t>(0, std::atoll(enc_bound_env)) : 0;
    dc.win_floor = enc_bound > 0;
    dc.engram_dirs = a.engram;
    dc.engram_hot = a.engram_hot;
    dc.vision = a.vision;
    // STRATA_SNAP_TAIL=<rows> (serve; default 0 = off): each conversation snapshot also keeps the last <rows> window rows
    // past the ring (host memory), so a request sharing a prefix that ends up to that far before the snapshot - an
    // edited last message, a new question about the same document - resumes there instead of re-reading (see
    // Ds41Dense::snap_tail_resume and the serve loop's snap-tail branch).
    if (a.serve) {
        const char * st = std::getenv("STRATA_SNAP_TAIL");
        dc.snap_tail = st ? std::max(0, std::atoi(st)) : 0;
    }
    dc.engram_hot = a.engram_hot;
    dc.act_quant = a.act_quant;
    // --kv-fp4 / STRATA_KV_FP4=1: the compressed KV + indexer keys packed to their exact FP4 grids (Ds41DenseConfig::kv_fp4)
    dc.kv_fp4 = a.kv_fp4 || (std::getenv("STRATA_KV_FP4") && std::getenv("STRATA_KV_FP4")[0] == '1');
    if (dc.kv_fp4) std::fprintf(stderr, "kv: compressed KV NVFP4 + indexer keys MXFP4 (--kv-fp4, exact)\n");
    Ds41Dense dense;
    std::string err;
    const double t_load = now_ms();
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds41_generate: dense init: %s\n", err.c_str()); return 1; }
    const Ds41Geometry & G = dense.geom();
    const int n_layer = (int) dense.n_layer(), n_embd = (int) G.n_embd, top_k = (int) G.n_used, nv = (int) G.vocab;
    // fewer experts per token (ds41_geometry.hpp ds41_experts_used): top_k = the router's k on every pass
    // (STRATA_EXPERTS_USED, already in G.n_used); dec_k = the decode / verify passes' k (STRATA_EXPERTS_USED_DECODE,
    // <= top_k: the router still picks top_k, the pass keeps the first dec_k - the highest-scoring - and renormalises)
    const int top_k_model = G.n_used_model > 0 ? (int) G.n_used_model : top_k;
    const int dec_k = std::min(top_k, (int) ds41_experts_used_decode(top_k_model));
    // enc-skip's layer 20: the LAST kv source (every compressed-KV / index-key write is owned by a kv source, and the
    // layers above it only read that owner's state, so the cut belongs there).  Also the window reach R = swa *
    // (n_layer - cut_layer) is what --enc-replay defaults to.
    int CL = 0;
    for (int64_t s : G.kv_sources) CL = std::max<int>(CL, (int) s);
    // exactness needs every later layer's window reach to stay inside the replay: reach = swa * (n_layer - CL)
    const int64_t reach = (int64_t) G.n_swa * (int64_t) (n_layer - CL);
    if (enc_bound > 0) {
        enc_bound = std::max<int64_t>(enc_bound, (int64_t) G.n_swa);
        if (enc_skip) ER = enc_bound;
        std::fprintf(stderr, "enc-bound: layers past %d run on the last %lld tokens of a prompt (changes output)\n", CL, (long long) enc_bound);
    }
    if (enc_skip && ER < reach && enc_bound == 0) {
        std::fprintf(stderr, "note: --enc-replay %lld < the window reach %lld (swa %lld x %d layers past %d); raised\n",
                     (long long) ER, (long long) reach, (long long) G.n_swa, n_layer - CL, CL);
        ER = reach;
    }
    if (enc_skip) std::fprintf(stderr, "enc-skip: cut at layer %d (last kv source), replay window %lld (reach %lld)\n", CL, (long long) ER, (long long) reach);
    if (a.engram.empty()) std::fprintf(stderr, "WARNING: no --engram (the shard folder, or a GGUF with the tables inline): the Engram add is skipped (not the real model)\n");

    // ---- routed experts
    Ds41FileBlobs src(a.model);
    if (!a.exl3.empty()) src.set_exl3(a.exl3);   // Delta: the EXL3 checkpoint's experts (its config.json must say exl3)
    if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    double slot_gib = 0;
    if (a.experts != "cpu" && !ggml_backend_is_cpu(be)) {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        fr = ds41_vram_quant(fr, "load: expert slots");   // STRATA_VRAM_QUANT_MIB
        slot_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        std::fprintf(stderr, "VRAM: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f)\n",
                     fr / 1073741824.0, slot_gib, a.vram_margin_gib);
        if (a.prompt_margin_gib >= 0.0)
            std::fprintf(stderr, "VRAM: the prompt path keeps %.2f GiB free (--prompt-margin): its batches borrow it from the "
                                 "expert cache and give it back\n", a.prompt_margin_gib);
        ds41_vram_decision("load: expert slots", "slot budget %.3f GiB", slot_gib);
    }
    Ds41MoeTier tier;
    Ds41MoeConfig mc;
    mc.cpu_only = a.experts == "cpu" || ggml_backend_is_cpu(be);
    mc.slots = a.slots > 0 ? a.slots : TG.n_layers * TG.n_experts;
    mc.slot_gib = a.slots > 0 ? 0.0 : slot_gib;
    // auto slots (one per expert, capped by slot_gib): a uniform-type file (e.g. a DwarfStar Q2 GGUF) would otherwise open
    // its cache at init with every expert's slot; open it at the seed, sized to slot_gib, as a mixed-tier file does
    if (mc.slot_gib > 0) mc.defer_cache = true;
    mc.pcie_frac = a.pcie;
    mc.tier_batch_pcie = a.tier_batch_pcie;
    mc.pf_b = a.pf_b;   // predicted prefetch (Ds41Dense::predict -> tier.prefetch), experts per layer
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prefetch = a.chunk_prefetch;
    mc.chunk_prefetch_gib = a.chunk_prefetch_gib;
    mc.skip_miss = a.skip_miss;
    mc.skip_file = a.skip_file;
    // soft / hard pruning (the GLM recipe, as tools/ds4/ds4_generate): listed experts take no VRAM slot and no arena space;
    // with --prune-penalty X they keep a selection bias -X (still chosen when the router prefers them by more than X, then
    // read from the file tier), without it they are never routed
    std::vector<char> pruned;
    int64_t n_pruned = 0;
    if (!a.prune.empty()) {
        pruned.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        std::FILE * pf = std::fopen(a.prune.c_str(), "r");
        if (!pf) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.prune.c_str()); return 2; }
        long l, e;
        while (std::fscanf(pf, "%ld %ld", &l, &e) == 2)
            if (l >= 0 && l < TG.n_layers && e >= 0 && e < TG.n_experts && !pruned[(size_t) (l * TG.n_experts + e)]) {
                pruned[(size_t) (l * TG.n_experts + e)] = 1;
                ++n_pruned;
            }
        std::fclose(pf);
        std::fprintf(stderr, "prune: %lld (layer, expert) pairs %s (%s)\n", (long long) n_pruned,
                     a.prune_penalty > 0 ? ("penalised -" + std::to_string(a.prune_penalty)).c_str() : "masked out of routing", a.prune.c_str());
        mc.exclude = pruned;
    }
    // --gpus A,B[,..] (the expert layer split): the dense half, every KV cache and the head stay on card A (the backend
    // above); each further card holds a whole tier - VRAM expert cache + its share of the RAM arena - for a contiguous
    // range of routed layers.  Those layers take the tier's host paths (run / run_multi / run_chunk: the activation in,
    // the sum out through the host; no peer access needed); card A's layers run exactly as on one card.
    const bool multi = a.gpus.size() > 1;
    if (multi && mc.cpu_only) { std::fprintf(stderr, "ds41_generate: --gpus needs the GPU tier\n"); return 2; }
    std::vector<int> layer_card((size_t) n_layer, 0);   // index into a.gpus (0 = the dense card)
    std::vector<std::unique_ptr<Ds41MoeTier>> xtier;    // xtier[c - 1]: card c's tier
    std::vector<double> card_gib(a.gpus.size(), 0.0), card_share(a.gpus.size(), 0.0);
    if (multi) {
        const int NC = (int) a.gpus.size();
        card_gib[0] = slot_gib;
        for (int c = 1; c < NC; ++c) {
            strata::v41::ds41_cuda_schedule_from_env(a.gpus[(size_t) c]);   // STRATA_CUDA_SPIN, before its context
            std::string name;
            size_t fr = 0, tot = 0;
            if (a.gpus[(size_t) c] == a.gpus[0] || !strata::v41::ds41_cuda_device_info(a.gpus[(size_t) c], name, fr, tot)) {
                std::fprintf(stderr, "ds41_generate: --gpus: no usable CUDA device %d\n", a.gpus[(size_t) c]);
                return 2;
            }
            fr = ds41_vram_quant(fr, "load: further card slots");
            card_gib[(size_t) c] = std::max(0.0, (double) fr / 1073741824.0 - a.gpu_margin_gib);
            ds41_vram_decision("load: further card slots", "card %d slot budget %.3f GiB", c, card_gib[(size_t) c]);
            std::fprintf(stderr, "gpus: card %d = CUDA %d (%s): %.2f of %.2f GiB free -> %.2f GiB of expert slots (margin %.2f)\n",
                         c, a.gpus[(size_t) c], name.c_str(), fr / 1073741824.0, tot / 1073741824.0, card_gib[(size_t) c], a.gpu_margin_gib);
            const int pa = strata::v41::ds41_cuda_can_peer(a.gpus[0], a.gpus[(size_t) c]);
            std::fprintf(stderr, "gpus: card %d peer access with card 0: %s (the sums go through the host either way)\n", c,
                         pa > 0 ? "yes" : pa == 0 ? "no" : "unknown");
        }
        std::vector<int> rl;   // the routed layers, in order
        for (int l = 0; l < n_layer; ++l) if (TG.routed(l)) rl.push_back(l);
        if (!a.gpu_split.empty()) {   // explicit: the first layer of each further card's range
            if ((int) a.gpu_split.size() != NC - 1) { std::fprintf(stderr, "ds41_generate: --gpu-split needs %d layer(s) for %d cards\n", NC - 1, NC); return 2; }
            for (int c = 0; c + 1 < NC; ++c)
                if (a.gpu_split[(size_t) c] <= (c ? a.gpu_split[(size_t) c - 1] : 0) || a.gpu_split[(size_t) c] >= n_layer) {
                    std::fprintf(stderr, "ds41_generate: --gpu-split must increase, inside 1..%d\n", n_layer - 1); return 2;
                }
            for (int l = 0; l < n_layer; ++l)
                for (int c = 0; c + 1 < NC; ++c) if (l >= a.gpu_split[(size_t) c]) layer_card[(size_t) l] = c + 1;
        }
        // STRATA_SPLIT_HOT=1 (with --rank; off by default): split by expert hotness - the boundaries that let the cards'
        // slots hold the most of the global ranked head (the experts one card with all the slots would cache), less
        // STRATA_SPLIT_LAYER_COST (default 0.005) of that head per layer on a further card (each costs a crossing).
        // Card A's estimate is less STRATA_SPLIT_A_RESERVE GiB (default 2: the prompt / DSpark buffers).  Rounds
        // 212-213 (3090 + x1 3060): it chose 22 next to the share split's 23 and measured the same (30.85 vs 30.73
        // tok/s); 12 (a routes-file coverage optimum) 25.26, 30 30.33 - so the share split stays the default.
        bool split_done = !a.gpu_split.empty();
        if (!split_done && !a.rank.empty() && std::getenv("STRATA_SPLIT_HOT") && std::getenv("STRATA_SPLIT_HOT")[0] == '1') {
            std::string rerr;
            const std::vector<std::pair<int32_t, int32_t>> order = ds41_read_ranked(a.rank, rerr);
            const char * rv = std::getenv("STRATA_SPLIT_A_RESERVE");
            const char * lv = std::getenv("STRATA_SPLIT_LAYER_COST");
            std::vector<double> budget(card_gib);
            budget[0] = std::max(0.0, budget[0] - (rv ? std::atof(rv) : 2.0));
            double all = 0;
            for (double g : budget) all += g * 1073741824.0;
            std::vector<int> ri((size_t) n_layer, -1);   // routed layer -> its index in rl
            for (size_t i = 0; i < rl.size(); ++i) ri[(size_t) rl[i]] = (int) i;
            std::vector<std::pair<int, double>> head;    // (rl index, bytes) of the global ranked head that fits `all`
            for (const auto & le : order) {
                if (le.first < 0 || le.first >= n_layer || ri[(size_t) le.first] < 0) continue;
                const double b = (double) ((src.expert_bytes(le.first, le.second) + 4095) & ~(size_t) 4095);
                if (all < b) break;
                all -= b;
                head.emplace_back(ri[(size_t) le.first], b);
            }
            const int L = (int) rl.size();
            if (!head.empty() && L >= NC) {
                const double lcost = (lv ? std::atof(lv) : 0.005) * (double) head.size();
                // cov(c, i, j): the head experts of routed layers [i, j) that card c's budget holds (the range's head
                // comes first in its own ranking, so it is that prefix)
                auto cov = [&](int c, int i, int j) {
                    double left = budget[(size_t) c] * 1073741824.0;
                    int n = 0;
                    for (const auto & h : head)
                        if (h.first >= i && h.first < j) { if (left < h.second) break; left -= h.second; ++n; }
                    return (double) n - (c ? lcost * (double) (j - i) : 0.0);
                };
                // best[c][j]: routed layers [0, j) on cards 0..c, each a non-empty range; ties keep more layers early
                const double NEG = -1e300;
                std::vector<std::vector<double>> best((size_t) NC, std::vector<double>((size_t) L + 1, NEG));
                std::vector<std::vector<int>> from((size_t) NC, std::vector<int>((size_t) L + 1, -1));
                for (int j = 1; j <= L; ++j) best[0][(size_t) j] = cov(0, 0, j);
                for (int c = 1; c < NC; ++c)
                    for (int j = c + 1; j <= L; ++j)
                        for (int i = c; i < j; ++i) {
                            if (best[(size_t) c - 1][(size_t) i] <= NEG) continue;
                            const double v = best[(size_t) c - 1][(size_t) i] + cov(c, i, j);
                            if (v >= best[(size_t) c][(size_t) j]) { best[(size_t) c][(size_t) j] = v; from[(size_t) c][(size_t) j] = i; }
                        }
                std::vector<int> start((size_t) NC, 0);
                for (int c = NC - 1, j = L; c > 0; --c) { start[(size_t) c] = from[(size_t) c][(size_t) j]; j = start[(size_t) c]; }
                for (int c = 0; c < NC; ++c)
                    for (int i = start[(size_t) c]; i < (c + 1 < NC ? start[(size_t) c + 1] : L); ++i) layer_card[(size_t) rl[(size_t) i]] = c;
                double held = 0;
                for (int c = 0; c < NC; ++c) held += cov(c, start[(size_t) c], c + 1 < NC ? start[(size_t) c + 1] : L) +
                                                     (c ? lcost * (double) ((c + 1 < NC ? start[(size_t) c + 1] : L) - start[(size_t) c]) : 0.0);
                std::fprintf(stderr, "gpus: split by expert hotness: the cards hold %.0f of the %zu-expert ranked head (%.1f%%)\n", held, head.size(), 100.0 * held / (double) head.size());
                split_done = true;
            }
        }
        if (!split_done) {   // by free VRAM: the slots each card can give, routed layers dealt out in proportion
            double sum = 0;
            for (double g : card_gib) sum += g;
            if (sum <= 0) { std::fprintf(stderr, "ds41_generate: --gpus: no free VRAM for experts on any card\n"); return 2; }
            double cum = 0;
            size_t i = 0;
            for (int c = 0; c < NC; ++c) {
                cum += card_gib[(size_t) c];
                const size_t end = c + 1 == NC ? rl.size() : (size_t) std::llround(cum / sum * (double) rl.size());
                for (; i < end && i < rl.size(); ++i) layer_card[(size_t) rl[i]] = c;
            }
        }
        for (int l : rl) card_share[(size_t) layer_card[(size_t) l]] += 1.0 / (double) rl.size();
        for (int c = 0; c < NC; ++c) {
            int lo = -1, hi = -1;
            for (int l : rl) if (layer_card[(size_t) l] == c) { if (lo < 0) lo = l; hi = l; }
            std::fprintf(stderr, "gpus: card %d (CUDA %d): layers %d-%d (%.0f%% of the routed layers)\n", c, a.gpus[(size_t) c],
                         lo, hi, 100.0 * card_share[(size_t) c]);
        }
        // the RAM arena is split by the same shares (the RAM total is unchanged)
        mc.device = a.gpus[0];
    }
    const Ds41MoeConfig mc_all = mc;   // the further cards' tiers start from the whole-model config
    if (multi) {
        mc.arena_gib = a.arena_gib * card_share[0];
        mc.max_arena_gib = mc.arena_gib;
        // WDDM: with a second card's context in the process, ~45-50 GiB of pinned host memory IN ALL is what works; past
        // that the further card cannot allocate anything (measured: card A 45 GiB + card 1 1 GiB ran; 45 + 11,
        // 64 + 1 and 71 + 1 did not, portable or not).  STRATA_MULTI_PIN_GIB overrides card A's 44 GiB.
        const char * mp = std::getenv("STRATA_MULTI_PIN_GIB");
        mc.pin_cap_gib = mp ? std::max(1.0, std::atof(mp)) : 44.0;
    }
    if (!tier.init(TG, &src, mc, err)) { std::fprintf(stderr, "ds41_generate: tier init: %s\n", err.c_str()); return 1; }
    for (size_t c = 1; multi && c < a.gpus.size(); ++c) {
        Ds41MoeConfig xc = mc_all;
        xc.device = a.gpus[c];
        xc.shared_pool = tier.shared_pool();   // one set of CPU workers: the tiers run one layer at a time
        xc.arena_gib = a.arena_gib * card_share[c];
        xc.max_arena_gib = xc.arena_gib;
        // STRATA_SPLIT_PIN_GIB (default 1): a further card's pin.  WDDM charges a card's pinned host memory to that card's
        // budget (see card A's cap above): with 1 GiB it ran.  Its arena misses go through staging copies.
        const char * xp = std::getenv("STRATA_SPLIT_PIN_GIB");
        xc.pin_cap_gib = xp ? std::max(1.0, std::atof(xp)) : 1.0;
        xc.slot_gib = card_gib[c];
        xc.sum_on_card = !std::getenv("STRATA_SPLIT_HOST_SUM");   // the sums on the card (STRATA_SPLIT_HOST_SUM=1: the host's)
        xc.defer_cache = true;
        xtier.push_back(std::make_unique<Ds41MoeTier>());
        if (!xtier.back()->init(TG, &src, xc, err)) { std::fprintf(stderr, "ds41_generate: card %zu tier init: %s\n", c, err.c_str()); return 1; }
    }
    // STRATA_FILE_TIER_UNMAP=1 (ds41_unmap.hpp; default off): the arena is filled and no reader runs yet -
    // close the file tier's buffered shard handles (every later read goes O_DIRECT, the same bytes) and touch the shards
    // unbuffered.  Ds41Dense released its mapping at init; each DSpark init releases its own and touches them again.
    if (ds41_p3_unmap()) {
        std::string why;
        const bool ok = src.drop_buffered(why);
        std::fprintf(stderr, "unmap: file tier %s (%s)\n", ok ? "unbuffered only" : "NOT changed", why.c_str());
    }
    // the tier that owns layer l, and whether it is on another card than the dense half
    auto T = [&](int64_t l) -> Ds41MoeTier & {
        const int c = multi && l >= 0 && l < n_layer ? layer_card[(size_t) l] : 0;
        return c ? *xtier[(size_t) c - 1] : tier;
    };
    auto remote = [&](int64_t l) { return multi && l >= 0 && l < n_layer && layer_card[(size_t) l] != 0; };
    auto each_tier = [&](const std::function<void(Ds41MoeTier &)> & f) { f(tier); for (auto & x : xtier) f(*x); };
    auto stats_all = [&]() { Ds41MoeStats s = tier.stats(); for (auto & x : xtier) s.add(x->stats()); return s; };
    auto resident_all = [&]() { int64_t r = tier.resident(); for (auto & x : xtier) r += x->resident(); return r; };
    // STRATA_EXPERTS_USED_DECODE: run_k = the experts per token the tier routes in the CURRENT phase - top_k for a
    // prompt (feed, feed_steps, pass_m, the DSpark seed passes), dec_k from the start of a request's decode (set by
    // set_run_k at the phase changes).  step / pass_v take the router's top_k ids per row and keep_k cuts them to
    // run_k in place (stride top_k -> run_k), so the tier reads run_k per row.  Unset: run_k == top_k, no-op.
    int run_k = top_k;
    bool km_dec = false;   // STRATA_EXPERTS_DROP_MISS: the current phase is a decode (set_run_k(dec_k, true))
    auto set_run_k = [&](int k, bool dec = false) {
        km_dec = dec;
        k = std::max(1, std::min(k, top_k));
        if (k == run_k) return;
        run_k = k;
        each_tier([&](Ds41MoeTier & t) { t.set_active_top_k(k); });
    };
    // keep the first run_k of each row's top_k (the router writes them in descending selection score - argsort_top_k
    // and ggml_dsv4_route alike), weights renormalised as the gate's own top-run_k would be: w_i * sum(all) / sum(kept)
    // = sc_i / sum(kept sc) * scale when norm_topk (V4.1), else the raw weights (no normalisation to redo)
    auto keep_k = [&](int * ids, float * w, int m) {
        if (run_k >= top_k) return;
        for (int t = 0; t < m; ++t) {
            const int * si = ids + (size_t) t * top_k;
            const float * sw = w + (size_t) t * top_k;
            double s_all = 0.0, s_keep = 0.0;
            for (int j = 0; j < top_k; ++j) { s_all += sw[j]; if (j < run_k) s_keep += sw[j]; }
            const float f = (G.norm_topk && s_keep > 0.0) ? (float) (s_all / s_keep) : 1.0f;
            int * di = ids + (size_t) t * run_k;   // in place: row t's destination never passes its unread source
            float * dw = w + (size_t) t * run_k;
            for (int j = 0; j < run_k; ++j) { const int e = si[j]; const float x = sw[j]; di[j] = e; dw[j] = x * f; }
        }
    };
    // STRATA_EXPERTS_DROP_MISS=<min_rank> (k5m = 6, k4m = 5; unset / 0 = off; decode and verify passes only - step and
    // pass_v while set_run_k(dec_k, true) is in force, never a prompt pass).  After keep_k, the entry at router rank
    // r >= min_rank (1-based: the row's order, descending selection score) is dropped when its expert would not be
    // served from VRAM by this layer's tier call - Ds41MoeTier::vram_hit_now: its VRAM cache slot or this layer's
    // prefetch staging, the lookup the call classifies with, at the moment of the call - AND every entry of that expert
    // in the pass sits at rank >= min_rank (otherwise another row fetches it anyway and the drop would save nothing).
    // Each row's kept weights are renormalised as keep_k does (w_i * sum(row) / sum(kept) under norm_topk).  The tier
    // reads a uniform K per call, so the call's K becomes the pass's largest kept count (set_active_top_k for that call
    // only) and a shorter row is padded with a weight-0 copy of one of its kept entries - a VRAM hit when the row has
    // one (the batched call serves a distinct expert once, so a pad costs one hit-row product).  A one-row step is
    // never padded.  STRATA_EXPERTS_DROP_MISS_STATS=1: a "kmiss:" line every 256 passes and at the end.
    static const int km_rank = [] { const char * v = std::getenv("STRATA_EXPERTS_DROP_MISS"); return v ? std::max(0, std::atoi(v)) : 0; }();
    static const bool km_stats = [] { const char * v = std::getenv("STRATA_EXPERTS_DROP_MISS_STATS"); return v && v[0] == '1'; }();
    if (km_rank > 0)
        std::fprintf(stderr, "experts: decode drops VRAM-missing experts at router rank >= %d (STRATA_EXPERTS_DROP_MISS, changes output)%s\n",
                     std::max(2, km_rank), km_rank < 2 ? " - clamped to 2 (rank 1 always stays)" : "");
    std::vector<int> km_ids;
    std::vector<float> km_w;
    std::vector<char> km_miss, km_drop;
    struct KmStats { int64_t passes = 0, calls = 0, ent = 0, drop = 0, miss = 0, avoid = 0, ent_r[16] = {}, drop_r[16] = {}; } kms;
    auto km_report = [&](const char * when) {
        if (!km_stats || kms.passes == 0) return;
        const double P = (double) kms.passes;
        std::string br;
        for (int j = 0; j < std::min(top_k, 16); ++j) {
            char b[48];
            std::snprintf(b, sizeof(b), " r%d %.1f%%", j + 1, 100.0 * kms.drop_r[j] / std::max<int64_t>(1, kms.ent_r[j]));
            br += b;
        }
        std::fprintf(stderr, "kmiss (%s): rank >= %d, %lld passes: entries dropped %.1f/pass (%.2f%% of %lld), distinct misses avoided "
                     "%.1f/pass (%.1f%% of %.1f/pass); dropped by rank:%s\n", when, std::max(2, km_rank), (long long) kms.passes,
                     kms.drop / P, 100.0 * kms.drop / std::max<int64_t>(1, kms.ent), (long long) kms.ent, kms.avoid / P,
                     100.0 * kms.avoid / std::max<int64_t>(1, kms.miss), kms.miss / P, br.c_str());
    };
    // the call's K back to run_k on every way out of the layer (a failed call too: the next request's set_run_k
    // returns early when run_k does not change, so it would not restore it)
    struct KmGuard {
        Ds41MoeTier * t;
        int k;
        ~KmGuard() { if (t) t->set_active_top_k(k); }
    };
    auto km_pass = [&]() {   // one decode step / verify pass begins
        if (km_rank <= 0 || !km_dec) return;
        ++kms.passes;
        if (km_stats && kms.passes % 256 == 0) km_report("so far");
    };
    // returns the call's K (0: nothing dropped - the tier reads ids / w at run_k as before); km_ids / km_w hold the rows
    // `multi`: the call is a run_multi / run_multi_to_dev (it adopts prefetch_next's early staging before classifying)
    auto drop_miss = [&](int l, const int * ids, const float * w, int m, bool multi) -> int {
        if (km_rank <= 0 || !km_dec) return 0;
        const int K = run_k, r0 = std::max(2, km_rank) - 1;   // r0: the first droppable 0-based rank
        if (r0 >= K) return 0;
        const int NE = m * K;
        km_miss.assign((size_t) NE, 0);
        km_drop.assign((size_t) NE, 0);
        Ds41MoeTier & tl = T(l);
        for (int j = 0; j < NE; ++j) km_miss[(size_t) j] = ids[j] >= 0 && !tl.vram_hit_now(l, ids[j], multi);
        // distinct misses: droppable when every entry of the expert in the pass is at rank >= r0
        int nmiss = 0, navoid = 0, ndrop = 0;
        for (int j = 0; j < NE; ++j) {
            if (!km_miss[(size_t) j]) continue;
            bool first = true, all_tail = true;
            for (int i = 0; i < NE; ++i) {
                if (ids[i] != ids[j]) continue;
                if (i < j) first = false;
                if (i % K < r0) all_tail = false;
            }
            if (!first) continue;
            ++nmiss;
            if (!all_tail) continue;
            ++navoid;
            for (int i = j; i < NE; ++i)
                if (ids[i] == ids[j]) { km_drop[(size_t) i] = 1; ++ndrop; }
        }
        ++kms.calls;
        kms.ent += NE;
        kms.drop += ndrop;
        kms.miss += nmiss;
        kms.avoid += navoid;
        if (km_stats)
            for (int j = 0; j < NE; ++j) { ++kms.ent_r[(j % K) & 15]; if (km_drop[(size_t) j]) ++kms.drop_r[(j % K) & 15]; }
        if (ndrop == 0) return 0;
        int KC = 0;
        for (int t = 0; t < m; ++t) {
            int c = 0;
            for (int k = 0; k < K; ++k) c += !km_drop[(size_t) t * K + k];
            KC = std::max(KC, c);
        }
        km_ids.assign((size_t) m * KC, 0);
        km_w.assign((size_t) m * KC, 0.0f);
        for (int t = 0; t < m; ++t) {
            const int * si = ids + (size_t) t * K;
            const float * sw = w + (size_t) t * K;
            const char * dr = km_drop.data() + (size_t) t * K;
            const char * mi = km_miss.data() + (size_t) t * K;
            double s_all = 0.0, s_keep = 0.0;
            int pad = -1;
            for (int k = 0; k < K; ++k) {
                s_all += sw[k];
                if (dr[k]) continue;
                s_keep += sw[k];
                if (pad < 0 || (mi[pad] && !mi[k])) pad = k;   // the first kept VRAM hit, else the first kept entry
            }
            const float f = (G.norm_topk && s_keep > 0.0) ? (float) (s_all / s_keep) : 1.0f;
            int * di = km_ids.data() + (size_t) t * KC;
            float * dw = km_w.data() + (size_t) t * KC;
            int c = 0;
            for (int k = 0; k < K; ++k)
                if (!dr[k]) { di[c] = si[k]; dw[c] = sw[k] * f; ++c; }
            for (; c < KC; ++c) { di[c] = si[pad]; dw[c] = 0.0f; }   // weight 0: adds nothing to the row's sum
        }
        return KC;
    };
    // the predicted ranks a prefetch may DMA (the decode step's prefetch, D3's union): with fewer experts per token
    // the prediction's ranks >= k are experts the pass will probably not route (the 6th of a 5-expert pass) -
    // STRATA_EXPERTS_PF_WIN=n overrides (0 = all 16, the old behaviour); default k when k < the model's, else 16
    static const int pf_win_env = std::getenv("STRATA_EXPERTS_PF_WIN") ? std::atoi(std::getenv("STRATA_EXPERTS_PF_WIN")) : -1;
    auto pf_win = [&]() -> int {
        if (pf_win_env == 0) return 16;
        if (pf_win_env > 0) return std::min(16, pf_win_env);
        return run_k < top_k_model ? run_k : 16;
    };
    dense.set_oom_hook([&]() { return tier.release_prestage(); });   // a prompt pass that does not fit: prestage gives way
    // defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the cache opens into
    // it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots.  --serve does this
    // per request: a multi-token prompt closes the cache (release_cache), the decode after it re-seeds (~0.6 s)
    // (--tier-replay: no prompt will run, so the VRAM cache seeds now - the replay exercises the hit class too)
    const bool defer_vram = !mc.cpu_only && (a.prefill_chunk > 1 || a.prompt_batch > 1) && a.tier_replay.empty();
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!mc.cpu_only) {
        if (!a.rank.empty()) {
            ranked = ds41_read_ranked(a.rank, err);
            if (ranked.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        } else {
            for (int64_t l = 0; l < TG.n_layers; ++l)
                for (int64_t e = 0; e < TG.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        }
        const double t0 = now_ms();
        std::vector<std::vector<std::pair<int32_t, int32_t>>> rk(multi ? a.gpus.size() : 0);
        if (multi) {   // each card's tier gets its own layers' share of the ranking; card A keeps `ranked`
            for (const auto & le : ranked)
                if (le.first >= 0 && le.first < n_layer) rk[(size_t) layer_card[(size_t) le.first]].push_back(le);
            ranked.swap(rk[0]);
        }
        // the further cards' tiers: arena, then the VRAM cache.  After card A by default, so card A pins its arena share
        // first (WDDM takes ~45-50 GiB pinned in all with two cards); STRATA_SPLIT_SEED_FIRST=1: before it (their cache
        // allocated before card A pins).  Measured: card A then pinned 76 GiB and its own next cudaMalloc failed - the budget
        // is the whole process's, so it does not help; kept for tests on other drivers)
        auto seed_extra = [&]() -> bool {
            for (size_t c = 1; c < a.gpus.size(); ++c) {
                const double tc = now_ms();
                Ds41MoeTier & xt = *xtier[c - 1];
                if (!xt.build_arena_ranked(rk[c], err)) {
                    std::fprintf(stderr, "ds41_generate: card %zu arena: %s\n", c, err.c_str());
                    return false;
                }
                // the cache is sized from the card's free VRAM NOW (after its context and pinning), and stepped down
                // 0.5 GiB at a time when one big cudaMalloc is refused: WDDM can refuse it with the memory reported
                // free (measured: 9.48 GiB failed on the 3060 with 10.99 GiB free).  The arena is already built, so a
                // retry only re-opens the cache; below 1 GiB the card runs with its cache off (arena + file tier).
                std::string nm;
                size_t fb = 0, tb = 0;
                double slot = card_gib[c];
                if (ds41_cuda_device_info(a.gpus[c], nm, fb, tb)) {
                    slot = std::min(slot, (double) ds41_vram_quant(fb, "seed: further card cache") / 1073741824.0 - a.gpu_margin_gib);
                    ds41_vram_decision("seed: further card cache", "card %zu cache %.3f GiB", c, slot);
                }
                bool seeded = false;
                for (; slot >= 1.0 && !seeded; slot -= 0.5) {
                    xt.set_cache_enabled(true);
                    xt.set_slot_gib(slot);
                    seeded = xt.seed_from_ranked(rk[c], err);
                    if (!seeded) std::fprintf(stderr, "gpus: card %zu cache of %.2f GiB refused (%s) - trying 0.5 GiB less\n", c, slot, err.c_str());
                }
                if (!seeded) {
                    std::fprintf(stderr, "gpus: card %zu: no VRAM cache (the arena + file tier serve its layers)\n", c);
                    xt.set_cache_enabled(false);
                    if (!xt.seed_from_ranked(rk[c], err)) {
                        std::fprintf(stderr, "ds41_generate: card %zu seed: %s\n", c, err.c_str());
                        return false;
                    }
                }
                std::fprintf(stderr, "gpus: card %zu tier: %lld experts in VRAM, %lld in the RAM arena (%.1f GiB), %lld file tier; seeded in %.1f s\n",
                             c, (long long) xtier[c - 1]->resident(), (long long) xtier[c - 1]->arena_experts(),
                             xtier[c - 1]->arena_gib(), (long long) xtier[c - 1]->file_tier(), (now_ms() - tc) / 1000.0);
            }
            return true;
        };
        static const bool seed_first = std::getenv("STRATA_SPLIT_SEED_FIRST") && std::getenv("STRATA_SPLIT_SEED_FIRST")[0] == '1';
        if (multi && seed_first && !seed_extra()) return 1;
        if (defer_vram ? !tier.build_arena_ranked(ranked, err) : !tier.seed_from_ranked(ranked, err)) {
            std::fprintf(stderr, "ds41_generate: seed: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "tier: %lld experts in VRAM, %lld in the RAM arena (%.1f GiB), %lld file tier; seeded in %.1f s\n",
                     (long long) tier.resident(), (long long) tier.arena_experts(), tier.arena_gib(), (long long) tier.file_tier(),
                     (now_ms() - t0) / 1000.0);
        if (multi && !seed_first && !seed_extra()) return 1;
    }
    // STRATA_ROUTE_TRACE=<file> (measurement only, both modes): the routing trace tools/v41/sim/route_sim.py and
    // place_sim.py replay.  Opened here so the server's passes are in it too; each decode start adds an
    // 'S' snapshot of the VRAM / arena sets (trace_snapshot), which is where the replay's state restarts.
    if (const char * rt = std::getenv("STRATA_ROUTE_TRACE")) tier.trace_open(rt);
    // --tier-replay F: the kernel replay bench.  A STRATA_TIER_RECORD file holds real verify-pass tier
    // calls (ids, weights, activations, the outputs the recording build produced); replay them through THIS build's
    // tier right after the seed - bitwise comparison of the outputs (the per-token sums are in double in a fixed
    // order, so a correct kernel change gives identical bytes whatever tier served each expert) and the time per
    // call - and exit.  No prompt, no decode: one model load instead of one per one-shot.
    if (!a.tier_replay.empty()) {
        const bool ok = tier.tier_replay(a.tier_replay, a.tier_replay_reps, err);
        if (!ok) std::fprintf(stderr, "ds41_generate: --tier-replay: %s\n", err.c_str());
        return ok ? 0 : 1;
    }
    // the selection-only bias per layer: the prune penalty / mask, + route_bias on the experts resident in VRAM now
    std::vector<char> proxy_res;   // --bias-prompt: the would-be VRAM set (first N ranked) until the real seed
    if (a.bias_prompt > 0) {
        proxy_res.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        int got = 0;
        for (const auto & le : ranked) {
            if (got >= a.bias_prompt) break;
            char & r = proxy_res[(size_t) ((int64_t) le.first * TG.n_experts + le.second)];
            if (!r) { r = 1; ++got; }
        }
    }
    auto resident_now = [&](int64_t l, int64_t e) {
        return proxy_res.empty() ? T(l).resident(l, e) : proxy_res[(size_t) (l * TG.n_experts + e)] != 0;
    };
    // bias_now: route-bias / file penalty are DECODE levers - before the VRAM seed (the prompt) only the prune part
    // applies, unless --bias-prompt measures them (the prompt's file tier is prefetched; a penalty there buys nothing)
    bool bias_now = a.bias_prompt > 0 || !defer_vram;
    // STRATA_KEEP_LAZY=K (0 = off): the kept cache's tail comes back only once a turn has written K tokens, not after every batch.
    // The refill took 340-430 ms a turn (a measured log), and a short turn's next batch would push the same slots out again.
    static const int keep_lazy = std::getenv("STRATA_KEEP_LAZY") ? std::max(0, std::atoi(std::getenv("STRATA_KEEP_LAZY"))) : 0;
    bool regrow_due = false;
    bool bias_set = false;   // a nonzero bias is in the dense half now (--serve turns it off again for the next prompt)
    // Live decode knobs: the settings an A/B most often flips, read from the environment once here and
    // changeable per request through the serve protocol (GEN ... d3_spec=N d3_spec_ahead=N d3_pf=N pf_p=X
    // bias_refresh=N bias_top=N; the server passes a request's "strata_tune" fields), so two settings can alternate
    // inside ONE loaded server (an interleaved A/B).  All exact except bias_top (A9, lossy
    // like the bias itself): 0 = the flat bias on every resident expert (today); N = only the N experts the
    // NEXT layer's prediction ranks highest, if resident, get the bias - the rest of routing is exact (arXiv
    // 2610.34721's bounded steer; the per-layer upload happens one layer ahead in pass_v).
    struct LiveKnobs {
        int d3_spec, d3_spec_ahead, d3_pf, bias_refresh, bias_top;
        float pf_pmin;
    } knobs {
        std::getenv("STRATA_D3_SPEC") ? std::atoi(std::getenv("STRATA_D3_SPEC")) : 0,
        std::getenv("STRATA_D3_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_D3_SPEC_AHEAD")) : 2,
        std::getenv("STRATA_D3_PF") ? std::atoi(std::getenv("STRATA_D3_PF")) : 1,
        std::getenv("STRATA_BIAS_REFRESH") ? std::max(0, std::atoi(std::getenv("STRATA_BIAS_REFRESH"))) : 0,
        std::getenv("STRATA_BIAS_TOP") ? std::max(0, std::atoi(std::getenv("STRATA_BIAS_TOP"))) : 0,
        std::getenv("STRATA_D3_PF_P") ? (float) std::atof(std::getenv("STRATA_D3_PF_P")) : 0.0f,
    };
    std::vector<std::vector<float>> rb_up((size_t) TG.n_layers);   // the bias each layer has on the card (changed_only)
    // changed_only (STRATA_BIAS_REFRESH): upload just the layers whose bias differs from what the card has - between two
    // verify passes only a few experts change residency, and every set_route_bias is a synchronous upload
    auto apply_bias = [&](bool changed_only = false) {
        const bool want = n_pruned > 0 || (bias_now && (a.route_bias != 0.0f || a.route_bias_file != 0.0f));
        if (!want && !bias_set) return;
        if (changed_only && knobs.bias_top > 0) return;   // A9: pass_v keeps each layer's bounded bias current itself
        bias_set = want;
        std::vector<float> rb((size_t) TG.n_experts);
        for (int64_t l = 0; l < TG.n_layers; ++l) {
            for (int64_t e = 0; e < TG.n_experts; ++e)
                rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) (l * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (bias_now && a.route_bias != 0.0f && !mc.cpu_only && knobs.bias_top == 0 && resident_now(l, e)) ? a.route_bias
                               : (bias_now && a.route_bias_file != 0.0f && !resident_now(l, e) && !T(l).in_arena(l, e)) ? -a.route_bias_file : 0.0f;
            if (changed_only && rb_up[(size_t) l] == rb) continue;
            dense.set_route_bias((int) l, rb.data());
            rb_up[(size_t) l] = rb;
        }
    };
    apply_bias();   // the prune part now (the prompt routes with it); route_bias again once the VRAM cache is seeded
    ds41_heapcheck("loaded");
    std::fprintf(stderr, "loaded in %.1f s: %d layers, ctx %lld, prefill chunk %d, experts %s, %zu classes\n",
                 (now_ms() - t_load) / 1000.0, n_layer, (long long) ctx, a.prefill_chunk, tier.mode(), TG.cls_gu.size());

    // ---- the forward pieces
    std::vector<int> ids1((size_t) top_k);
    std::vector<float> w1((size_t) top_k), routed1((size_t) n_embd), logits1((size_t) nv);
    // --serve's short-turn passes (STRATA_SERVE_PASS_M tokens per dense pass; 1 = single decode steps)
    const int pass_max = std::max(1, std::getenv("STRATA_SERVE_PASS_M") ? std::atoi(std::getenv("STRATA_SERVE_PASS_M")) : 16);
    std::vector<float> crouted_m((size_t) std::max(pass_max, 17) * n_embd);   // 17: pass_v's DSpark seeding / verify passes
    std::string ferr;
    // STRATA_DEC_PROF=1: per-stage wall time of the decode step (begin / predict+prefetch / attn_router / tier / finish)
    static const bool dprof = std::getenv("STRATA_DEC_PROF") != nullptr;
    // decode: finish(l) and predict(l+1) as one graph (identical results; STRATA_NO_FUSE_PREDICT=1 = separate graphs)
    static const bool fuse_predict = std::getenv("STRATA_NO_FUSE_PREDICT") == nullptr;
    // STRATA_PF_OVERLAP=1: the fused prediction's readback no longer blocks before the next layer's attention launches
    // (pinned buffer + event); the host waits for it only after that attention is queued, so the GPU never idles between
    // finish(l) and attn(l+1) and the prefetch DMAs overlap the attention.  Same graphs, same predictions: exact.
    static const bool pf_overlap = std::getenv("STRATA_PF_OVERLAP") && std::getenv("STRATA_PF_OVERLAP")[0] == '1';
    // STRATA_ADVISE_AHEAD=k (0 = off): at layer l, predict layer l+k's routing from the current state and hint its
    // top STRATA_ADVISE_TOP (8) file-tier experts to the kernel (page-cache readahead; the SSD is idle in decode)
    const int advise_ahead = std::getenv("STRATA_ADVISE_AHEAD") ? std::atoi(std::getenv("STRATA_ADVISE_AHEAD")) : 0;
    const int advise_top = std::getenv("STRATA_ADVISE_TOP") ? std::atoi(std::getenv("STRATA_ADVISE_TOP")) : 8;
    // STRATA_FILE_PRED=1 (measurement only, slower): at layer l, predict layers l+1..l+4 from the current state and count
    // how many of the file-tier misses each layer actually routes were in that prediction's top 4 / 8 / 16 - the hit
    // rate a speculative O_DIRECT read k layers ahead would get.  Printed after the decode line.
    // speculative file-tier reads (decode, OPT-IN): finish(l)'s fused prediction graph also predicts layers
    // l+2..l+1+SPEC_AHEAD; the file-tier experts among each prediction's top STRATA_SPEC_TOP are read ahead
    // (tier.spec_read).  Measured 2026-10-12 (p256 x 256, --slots 1500, tokens identical): 61-81% of the file misses
    // come from the stash and the tier gets ~1.8 ms/token faster, but the bigger finish graph (+1.5 ms) and issuing
    // (+1.3 ms) eat it - net 24.97-25.10 vs 25.56 tok/s off.  Default 0 = off; try STRATA_SPEC_TOP=4.
    const int spec_top = std::getenv("STRATA_SPEC_TOP") ? std::atoi(std::getenv("STRATA_SPEC_TOP")) : 0;
    const int spec_ahead = std::getenv("STRATA_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_SPEC_AHEAD")) : 2;
    // STRATA_SPEC_L1=1: also read the next layer's predicted file-tier experts (one layer of lead: mostly late)
    const bool spec_l1 = std::getenv("STRATA_SPEC_L1") && std::getenv("STRATA_SPEC_L1")[0] == '1';
    // STRATA_DUMP_MAIN=F (measurement only, slower): per decode step append {int32 token, int32 pos, 3 x n_embd f32} to F
    // = the step's input token and DeepSeek model.py's DSpark main_hidden: the HC-mean of the state ENTERING target
    // layers 37/38/39 ("the MTP head reads the attention input of its target layers, not their output") = the output
    // of layers 36/37/38 - for the offline DSpark acceptance test (tools/v41/dspark_accept.py).  The one-shot run then
    // feeds the last 128 prompt tokens (DSpark's attention window) as steps too.
    const char * dump_main_path = std::getenv("STRATA_DUMP_MAIN");
    std::FILE * dump_main = dump_main_path ? std::fopen(dump_main_path, "wb") : nullptr;
    const int dm_layers[3] = { 36, 37, 38 };   // outputs of these = inputs of the targets 37 / 38 / 39
    std::vector<float> dm_x, dm_pre, dm_row;
    int dm_tok = 0, dm_pos = 0;
    const bool file_pred = std::getenv("STRATA_FILE_PRED") != nullptr;
    constexpr int kFP = 4;
    std::vector<int> fp_ids(file_pred ? (size_t) kFP * (size_t) n_layer * 16 : 0, -1);
    long long fp_miss[kFP + 1] = {0}, fp_hit[kFP + 1][3] = {{0}}, fp_cand[kFP + 1][3] = {{0}};
    // STRATA_PRED_LOG=F (measurement only): per layer of every decode step / verify pass, the routing the prefetch was
    // given (16 ranked ids per pass row, predicted before the attention; -1 = no prediction that layer) next to the
    // routing the layer then chose, each id with its tier at that moment (0 VRAM-resident, 1 RAM arena, 2 file).
    // int16 records: 'R', layer, m, K, pos & 0x7fff, pos >> 15, pred[m*16], act[m*K], tier(pred), tier(act).  Appends;
    // scored offline (per-layer precision, budgets, temporal / draft-row ensembles).
    std::FILE * plog = std::getenv("STRATA_PRED_LOG") ? std::fopen(std::getenv("STRATA_PRED_LOG"), "ab") : nullptr;
    std::vector<int16_t> pl_buf;
    std::vector<int> pl_pred;    // the prediction handed to this layer's prefetch (pv_pred is reused by the spec reads)
    bool pl_have = false;
    // STRATA_PRED_LOG2=F + STRATA_PRED_AHEAD=k (default 1; measurement only): the same records, but `pred` is layer l's
    // routing predicted k layers EARLIER (at the start of layer l-k, from the state entering it) - what a prefetch
    // issued k layers ahead would get: its copies would have k more layers of time to land (measured: a 2nd / 3rd copy
    // started at layer l's own attention lands too late).  Decode steps (predict) and verify passes (predict_multi).
    std::FILE * plog2 = std::getenv("STRATA_PRED_LOG2") ? std::fopen(std::getenv("STRATA_PRED_LOG2"), "ab") : nullptr;
    const int pred_ahead = std::getenv("STRATA_PRED_AHEAD") ? std::max(1, std::atoi(std::getenv("STRATA_PRED_AHEAD"))) : 1;
    std::vector<std::vector<int>> pl_ahead((size_t) n_layer);   // [layer]: its k-ahead prediction (16 per pass row)
    auto plog_rec = [&](std::FILE * plf, int l, int m, int pos, const int * pred, const int * act) {
        auto tier_of = [&](int e) -> int16_t { return (int16_t) (e < 0 ? -1 : T(l).resident(l, e) ? 0 : T(l).in_arena(l, e) ? 1 : 2); };
        pl_buf.assign({ (int16_t) 'R', (int16_t) l, (int16_t) m, (int16_t) run_k, (int16_t) (pos & 0x7fff), (int16_t) (pos >> 15) });
        for (int i = 0; i < m * 16; ++i) pl_buf.push_back((int16_t) (pred ? pred[i] : -1));
        for (int i = 0; i < m * run_k; ++i) pl_buf.push_back((int16_t) act[i]);   // (keep_k's stride)
        for (int i = 0; i < m * 16; ++i) pl_buf.push_back(pred ? tier_of(pred[i]) : (int16_t) -1);
        for (int i = 0; i < m * run_k; ++i) pl_buf.push_back(tier_of(act[i]));
        std::fwrite(pl_buf.data(), 2, pl_buf.size(), plf);
        if (l == n_layer - 1) std::fflush(plf);
    };
    // the k-ahead prediction for layer l + pred_ahead, made at the start of layer l (n = the pass rows; 1 = a step)
    auto pl_ahead_make = [&](int l, int n) {
        if (l == 0) for (auto & v : pl_ahead) v.clear();
        if (l + pred_ahead >= n_layer) return;
        std::vector<int> & v = pl_ahead[(size_t) (l + pred_ahead)];
        v.resize((size_t) n * 16);
        const bool ok = n == 1 ? dense.predict(l + pred_ahead, v.data(), 16) : dense.predict_multi(l + pred_ahead, v.data());
        if (!ok) v.clear();
    };
    // decode: the tier sums the routed experts on the card straight into the dense half's routed tensor (same bits as
    // the host sum; no parts readback / routed upload per layer).  STRATA_DEC_HOSTSUM=1 = the host round trip
    // the routed tensor's device address: re-read after every dense.set_columns (elastic columns reallocate it)
    float * rdev1 = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? dense.routed_dev() : nullptr;
    double dp[6] = {0, 0, 0, 0, 0, 0};
    long dp_n = 0;
    auto step = [&](int tok, int pos) -> bool {   // one token through every layer; logits via dense.logits
        double ta = dprof ? now_ms() : 0;
        km_pass();   // STRATA_EXPERTS_DROP_MISS_STATS
        if (dump_main) {
            dm_tok = tok;
            dm_pos = pos;
            dm_row.assign((size_t) 3 * n_embd, 0.0f);
        }
        if (!dense.begin(&tok, 1, pos)) { ferr = dense.last_error(); return false; }
        if (dprof) { const double t = now_ms(); dp[0] += t - ta; ta = t; }
        int32_t top16[16];
        bool have_pred = false;
        bool pred_pending = false;   // STRATA_PF_OVERLAP: finish_predict_launch issued, predict_wait not yet
        for (int l = 0; l < n_layer; ++l) {
            if (plog2) pl_ahead_make(l, 1);   // STRATA_PRED_LOG2: layer l + k's routing, predicted here
            if (file_pred) {
                for (int k = 1; k <= kFP; ++k) {
                    int * const pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) (l + k < n_layer ? l + k : 0)) * 16;
                    if (l + k < n_layer && !dense.predict(l + k, pk, 16)) std::fill(pk, pk + 16, -1);
                    if (l + k < n_layer)   // file-tier candidates = the reads a speculative fetch would issue
                        for (int i = 0; i < 16; ++i)
                            if (pk[i] >= 0 && !T(l + k).resident(l + k, pk[i]) && !T(l + k).in_arena(l + k, pk[i]))
                                for (int c = 0; c < 3; ++c) fp_cand[k][c] += i < (4 << c);
                }
            }
            if (advise_ahead > 0 && l + advise_ahead < n_layer) {   // file-tier readahead for layer l+k's predicted experts
                int ta[16];
                if (dense.predict(l + advise_ahead, ta, 16)) T(l + advise_ahead).advise_file(l + advise_ahead, ta, advise_top);
            }
            const float * fn = nullptr;
            ggml_tensor * fn_t = nullptr;   // the same activation on the card: the tier quantizes it there
            bool attn_done = false;
            if (pred_pending) {   // STRATA_PF_OVERLAP: this layer's attention starts first, then the prefetch is issued
                if (!dense.attn_router_launch(l)) { ferr = dense.last_error(); return false; }
                static const bool pf_spin = std::getenv("STRATA_PF_SPIN") && std::getenv("STRATA_PF_SPIN")[0] == '1';
                if (pf_spin) strata::v41::ds41_cuda_event_spin(dense.predict_event());
                if (!(pf_spin ? dense.predict_take(top16, 16) : dense.predict_wait(top16, 16))) { ferr = dense.last_error(); return false; }
                T(l).prefetch(l, top16, pf_win());   // its DMAs overlap the attention already on the card
                pl_have = true;
                pred_pending = false;
                have_pred = false;
                if (dprof) { const double t = now_ms(); dp[1] += t - ta; ta = t; }
                if (!dense.attn_router_collect(l, ids1.data(), w1.data(), &fn, &fn_t)) { ferr = dense.last_error(); return false; }
                attn_done = true;
            } else if (mc.pf_b > 0 && !mc.cpu_only) {   // the DMAs run under this layer's attention
                int t16[16];
                if (have_pred) {                  // predicted by the previous layer's finish_predict (same state)
                    T(l).prefetch(l, top16, pf_win());
                    pl_have = true;
                } else if (dense.predict(l, t16, 16)) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    T(l).prefetch(l, top16, pf_win());
                    pl_have = true;
                }
                have_pred = false;
            }
            if (!attn_done) {
                if (dprof) { const double t = now_ms(); dp[1] += t - ta; ta = t; }
                if (!dense.attn_router(l, ids1.data(), w1.data(), &fn, &fn_t, true)) { ferr = dense.last_error(); return false; }
            }
            keep_k(ids1.data(), w1.data(), 1);   // STRATA_EXPERTS_USED_DECODE (no-op otherwise)
            // STRATA_EXPERTS_DROP_MISS: the tier call below reads s_ids / s_w at s_k (a run / run_to_dev: no early staging)
            const int s_k = drop_miss(l, ids1.data(), w1.data(), 1, false);
            const int * s_ids = s_k ? km_ids.data() : ids1.data();
            const float * s_w = s_k ? km_w.data() : w1.data();
            if (plog) plog_rec(plog, l, 1, pos, pl_have ? top16 : nullptr, ids1.data());
            if (plog2 && !pl_ahead[(size_t) l].empty()) plog_rec(plog2, l, 1, pos, pl_ahead[(size_t) l].data(), ids1.data());
            pl_have = false;
            if (dprof) { const double t = now_ms(); dp[2] += t - ta; ta = t; }
            if (file_pred) {
                for (int j = 0; j < run_k; ++j) {
                    const int e = ids1[(size_t) j];
                    if (e < 0 || T(l).resident(l, e) || T(l).in_arena(l, e)) continue;
                    ++fp_miss[0];
                    for (int k = 1; k <= kFP; ++k) {
                        if (l - k < 0) continue;   // nothing predicted this layer k layers ago
                        const int * pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) l) * 16;
                        ++fp_miss[k];
                        int at = 16;
                        for (int i = 0; i < 16; ++i) if (pk[i] == e) { at = i; break; }
                        if (at < 4) ++fp_hit[k][0];
                        if (at < 8) ++fp_hit[k][1];
                        if (at < 16) ++fp_hit[k][2];
                    }
                }
            }
            if (!dense.shexp_launch(l)) { ferr = dense.last_error(); return false; }   // STRATA_SHEXP_SPLIT
            float * const rd1 = remote(l) ? nullptr : rdev1;   // --gpus: another card's layer sums through the host
            KmGuard s_kg { s_k ? &T(l) : nullptr, run_k };
            if (s_k) T(l).set_active_top_k(s_k);
            const bool tok = rd1 ? tier.run_to_dev(l, s_ids, s_w, fn, rd1, fn_t ? fn_t->data : nullptr)
                                 : T(l).run(l, s_ids, s_w, fn, routed1.data());
            if (s_k) T(l).set_active_top_k(run_k);
            if (!tok) { ferr = "tier.run failed"; return false; }
            if (void * ev = T(l).take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC: order on the card
            if (dprof) { const double t = now_ms(); dp[3] += t - ta; ta = t; }
            const float * rsum = rd1 ? nullptr : routed1.data();   // null: already in the routed tensor
            if (fuse_predict && mc.pf_b > 0 && !mc.cpu_only && l + 1 < n_layer) {
                int t16[16];
                if (spec_top > 0 && spec_ahead > 0) {
                    int ahead[16 * 8];
                    const int na = std::min(spec_ahead, 8);
                    if (!dense.finish_predict_ahead(l, rsum, l + 1, t16, 16, na, ahead)) { ferr = dense.last_error(); return false; }
                    if (dprof) { const double t = now_ms(); dp[4] += t - ta; ta = t; }
                    if (spec_l1) T(l + 1).spec_read(l + 1, t16, std::min(spec_top, 16));
                    for (int a = 0; a < na && l + 2 + a < n_layer; ++a)   // (no layer past the last: their rows are -1 anyway)
                        T(l + 2 + a).spec_read(l + 2 + a, ahead + a * 16, std::min(spec_top, 16));
                    if (dprof) { const double t = now_ms(); dp[5] += t - ta; ta = t; }
                } else if (pf_overlap) {   // launch only: the next layer waits for the prediction after its attention starts
                    if (!dense.finish_predict_launch(l, rsum, l + 1)) { ferr = dense.last_error(); return false; }
                    pred_pending = true;
                } else if (!dense.finish_predict(l, rsum, l + 1, t16, 16)) { ferr = dense.last_error(); return false; }
                if (!pred_pending) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    have_pred = true;
                }
            } else if (!dense.finish(l, rsum)) { ferr = dense.last_error(); return false; }
            if (dprof) { const double t = now_ms(); dp[4] += t - ta; ta = t; }
            for (int j = 0; dump_main && j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                dense.read_state(dm_x, dm_pre);   // [n_embd, HC] for the one token
                const size_t HC = dm_x.size() / (size_t) n_embd;
                for (size_t h = 0; h < HC; ++h)
                    for (int64_t d = 0; d < n_embd; ++d) dm_row[(size_t) j * n_embd + d] += dm_x[h * n_embd + d] / (float) HC;
                if (j == 2) {
                    std::fwrite(&dm_tok, 4, 1, dump_main);
                    std::fwrite(&dm_pos, 4, 1, dump_main);
                    std::fwrite(dm_row.data(), 4, dm_row.size(), dump_main);
                }
            }
        }
        if (dprof && ++dp_n % 64 == 0)
            std::fprintf(stderr, "decode step ms/token: begin %.2f predict %.2f attn_router %.2f tier %.2f finish %.2f spec %.2f (%ld steps)\n",
                         dp[0] / dp_n, dp[1] / dp_n, dp[2] / dp_n, dp[3] / dp_n, dp[4] / dp_n, dp[5] / dp_n, dp_n);
        return true;
    };
    // a few tokens in ONE dense pass (the dense weights - ~8 GB - read once for m tokens, not per token), the routed
    // experts token by token from the VRAM cache (run_to_dev into the pass's routed rows): --serve's short turns.
    // Row-batched matmuls: the same model, not the same bits as m single steps.
    std::vector<int> mids;
    std::vector<float> mw;
    auto pass_m = [&](const int * toks, int m, int pos) -> bool {
        set_run_k(top_k);   // a prompt pass (feed_steps): the router's top_k rows as they are
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        if (!dense.begin(toks, m, pos)) { ferr = dense.last_error(); return false; }
        for (int l = 0; l < n_layer; ++l) {
            const float * fn = nullptr;
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            float * const rdl = remote(l) ? nullptr : rdev1;   // --gpus: another card's layer through the host
            for (int t = 0; t < m; ++t) {
                const bool ok = rdl ? tier.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                      fn + (size_t) t * n_embd, rdl + (size_t) t * n_embd)
                                    : T(l).run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                               fn + (size_t) t * n_embd, routed1.data());
                if (!ok) { ferr = "tier.run failed"; return false; }
                if (void * ev = T(l).take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC
                if (!rdl) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
            }
            if (!dense.finish(l, rdl ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
        }
        return true;
    };
    // pass_m + DSpark's main_hidden: vh[t * 3D + j * D ..] = the HC-mean of the state leaving layer dm_layers[j] (=
    // entering the DSpark target layer 37 + j) for pass token t (the STRATA_DUMP_MAIN math, token-major read_state);
    // vlog = the pass's m rows of logits: a view of the dense's pinned logits rows (STRATA_PINNED_STAGE, valid until
    // the next pass_v) or of vlog_own; data() / begin() / end() as the vector it replaced.
    struct LogitRows {
        float * p = nullptr;
        size_t n = 0;
        float * data() const { return p; }
        float * begin() const { return p; }
        float * end() const { return p + n; }
    };
    std::vector<float> vh, vlog_own, vx, vpre;
    LogitRows vlog;
    std::vector<int> pv_pred, pv_rank;                 // D3 prefetch: per-token predictions, the ranked union
    std::vector<int> pv_next, pv_rank_next;            // STRATA_D3_PF_EARLY: the next layer's, made one layer early
    std::vector<std::pair<int, int>> pv_score;
    std::vector<std::pair<int, float>> pv_noq;         // STRATA_D3_PF_NOISYOR: (expert, prod over rows of 1 - p)
    double pv_ms[4] = {0, 0, 0, 0};   // pass_v split: dense (begin/attn_router/finish), tier, taps (read_state), logits
    double pv_rm = 0;   // --gpus: the wall of the other cards' layers' tier calls (part of pv_ms[1])
    long pv_n = 0, pv_tok = 0;
    auto pass_v = [&](const int * toks, int m, int pos) -> bool {
        // D3 (the batched per-class verify tier) is the default: correct (bias-off verify-vs-prefill gate, near-ties
        // only) and faster (verify 167.6 -> ~100 ms/pass).  STRATA_TIER_BATCH=0 = the per-token run_to_dev path.
        static const bool no_tier_batch = [] { const char* v = std::getenv("STRATA_TIER_BATCH"); return v && v[0] == '0'; }();
        double tq = now_ms();
        auto lap = [&](int i) { const double t = now_ms(); pv_ms[i] += t - tq; tq = t; };
        ++pv_n; pv_tok += m;
        km_pass();   // STRATA_EXPERTS_DROP_MISS_STATS
        // STRATA_PASS_LOG=<file>: one line per pass (rows, wall / dense / tier ms, the tier's hits / cpu / pcie / file
        // for the pass) - the cost model behind batched decoding: what a pass costs per pass, per row and per miss
        static std::FILE * pass_log = std::getenv("STRATA_PASS_LOG") ? std::fopen(std::getenv("STRATA_PASS_LOG"), "w") : nullptr;
        // STRATA_ROUTE_TRACE: the same per-pass numbers as a 'V' record of the trace - {m, wall us, dense
        // us, tier us, hits, cpu, pcie, file, spec hits, spec issued, prefetch useful, prefetch issued, trace us} -
        // the last being the time the trace's own extra predictions took ('R' records below: in the wall and the
        // dense time, never the tier's; the replay subtracts it).  --gpus (multigpu): the pass log line also carries
        // the other cards' tier wall and their hits / cpu / pcie / file.
        const bool tracing = tier.tracing();
        double tr_extra_ms = 0;
        const double pl_t0 = tq, pl_d0 = pv_ms[0], pl_t1 = pv_ms[1], pl_r0 = pv_rm;
        const Ds41MoeStats pl_s0 = (pass_log || tracing) ? tier.stats() : Ds41MoeStats{};
        auto xstats = [&]() { Ds41MoeStats s; for (auto & x : xtier) s.add(x->stats()); return s; };
        const Ds41MoeStats pl_x0 = pass_log && multi ? xstats() : Ds41MoeStats{};
        struct PassLogLine {
            std::FILE * f; int m; double t0, d0, k0, r0; const double * ms; const double * rm; const Ds41MoeStats * s0;
            Ds41MoeTier * tr; const Ds41MoeStats * x0; const std::function<Ds41MoeStats()> xs;
            bool trace; const double * extra;
            ~PassLogLine() {
                if (!f && !trace) return;
                const Ds41MoeStats s = tr->stats();
                if (f) {
                    std::fprintf(f, "%d %.3f %.3f %.3f %lld %lld %lld %lld", m, now_ms() - t0, ms[0] - d0, ms[1] - k0,
                                 (long long) (s.hits - s0->hits), (long long) (s.cpu - s0->cpu), (long long) (s.pcie - s0->pcie),
                                 (long long) (s.file_tier - s0->file_tier));
                    if (xs) {   // --gpus: + the other cards' tier wall (ms) and their hits / cpu / pcie / file
                        const Ds41MoeStats x = xs();
                        std::fprintf(f, " %.3f %lld %lld %lld %lld", *rm - r0, (long long) (x.hits - x0->hits),
                                     (long long) (x.cpu - x0->cpu), (long long) (x.pcie - x0->pcie),
                                     (long long) (x.file_tier - x0->file_tier));
                    }
                    std::fputc('\n', f);
                    std::fflush(f);
                }
                if (trace) {
                    const int32_t v[13] = { m, (int32_t) ((now_ms() - t0) * 1000.0), (int32_t) ((ms[0] - d0) * 1000.0),
                                            (int32_t) ((ms[1] - k0) * 1000.0), (int32_t) (s.hits - s0->hits), (int32_t) (s.cpu - s0->cpu),
                                            (int32_t) (s.pcie - s0->pcie), (int32_t) (s.file_tier - s0->file_tier),
                                            (int32_t) (s.spec_hits - s0->spec_hits), (int32_t) (s.spec_issued - s0->spec_issued),
                                            (int32_t) (s.prefetched_useful - s0->prefetched_useful),
                                            (int32_t) (s.prefetch_issued - s0->prefetch_issued), (int32_t) (*extra * 1000.0) };
                    tr->trace_rec('V', v, 13);
                }
            }
        } pass_log_line { pass_log, m, pl_t0, pl_d0, pl_t1, pl_r0, pv_ms, &pv_rm, &pl_s0, &tier, &pl_x0,
                          multi ? std::function<Ds41MoeStats()>(xstats) : std::function<Ds41MoeStats()>(), tracing, &tr_extra_ms };
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        vh.assign((size_t) m * 3 * n_embd, 0.0f);
        if (!dense.begin(toks, m, pos)) { ferr = dense.last_error(); return false; }
        // D3 prefetch (STRATA_D3_PF = slots per layer, default 1, 0 = off): before each layer's attention, predict
        // every pass token's routing (predict_multi), rank the union (score 16 - rank, summed over tokens) and DMA the
        // likely misses into the prefetch slots on the prefetch stream - they land under the attention.  PCIe is the
        // wall (nsys: ~100 expert DMAs per verify pass, 1.4 s of a 4.2 s decode), so a wrong guess costs a full blob:
        // docs prompt, same text: slots 0/1/2/3/6 -> verify 104.4/101.7/104.5/111.1/124.1 ms (precision 40% at 1).
        // Its row cap stays 8 (the old kMaxMulti, i.e. the passes it was measured on): since kMaxMulti 16 a 9-16-row
        // pass is ONE tier call too, but it ran without the prefetch before and still does (a separate experiment).
        const int d3_pf = knobs.d3_pf;   // live knob (per request)
        // STRATA_D3_PF_EARLY=K (default 0 = off): K experts of the NEXT layer, ranked as this layer's
        // (noisy-OR when on) from a prediction made one layer early in the same graph (predict_multi2: no extra sync),
        // copied into a second staging right after this layer's tier call has queued its demand copies - they have the
        // rest of this layer and the next attention to land (measured: a 2nd / 3rd copy started at a layer's own
        // attention lands too late).  The next layer's own STRATA_D3_PF prefetch appends to them.
        static const int d3_pf_early = std::getenv("STRATA_D3_PF_EARLY") ? std::max(0, std::atoi(std::getenv("STRATA_D3_PF_EARLY"))) : 0;
        bool have_next = false;
        constexpr int kD3PfRows = 8;
        static_assert(kD3PfRows <= Ds41MoeTier::kMaxMulti, "the prefetch is consumed by ONE tier call per layer");
        // (STRATA_D3_PF=0 + STRATA_D3_PF_EARLY: the one prefetch copy per layer moved a layer earlier)
        const bool pf_now = rdev1 && !no_tier_batch && (d3_pf > 0 || d3_pf_early > 0) && m <= kD3PfRows;
        // STRATA_IDLE_PREFETCH=1 (Ds41MoeTier::idle_*, ds41_moe.cpp struct IdlePf): armed for this pass
        // (one tier call per layer), planned from the D3 spec prediction STRATA_IDLE_PF_AHEAD layers ahead (below) and
        // refreshed by each layer's own prediction (prefetch_multi); the guard drops the pass's plans on every return
        const bool ipf = tier.idle_pf_on() && rdev1 && !no_tier_batch && m <= Ds41MoeTier::kMaxMulti;
        const bool ipf_spec = ipf && tier.idle_pf_ahead() > 0;   // plans ahead need the extra prediction
        struct IpfPass {
            Ds41MoeTier & t;
            bool on;
            ~IpfPass() { if (on) t.idle_pass(false); }
        } ipf_pass { tier, ipf };
        if (ipf) tier.idle_pass(true);
        // STRATA_D3_PF_ASYNC=1: the prediction's readback waits on its own event while the layer's attention is already
        // queued behind it (predict_multi_launch, attn_router_launch, predict_multi_wait, attn_router_collect), so the
        // host's ranking and the prefetch DMA overlap the attention instead of preceding it.  The same graphs and
        // predictions; only the DMA starts a little later.
        static const bool pf_async = std::getenv("STRATA_D3_PF_ASYNC") && std::getenv("STRATA_D3_PF_ASYNC")[0] == '1';
        // STRATA_D3_PF_NOISYOR=1 (default off): rank the union by P(at least one pass row routes it) =
        // 1 - prod over the rows of (1 - p(its rank in that row)), p = how often a row routes its rank-r prediction
        // (the launcher's own log: 95 89 80 69 57 44 32 23 17 13 10 8 7 5 5 4 %), instead of the summed
        // 16 - rank, which puts rank 5 in four rows (~44% each) ahead of rank 0 in one (95%).  Held out: prefetch
        // precision 61% -> 77% at the same one slot.  STRATA_D3_PF_P=x: only candidates with P >= x go to the prefetch
        // (then STRATA_D3_PF can be raised: P >= 0.8 x 2 slots = 82%).  STRATA_D3_PF_PTAB=F: per-layer p rows, 16 per
        // line, '#' comments instead of the pooled ones.
        static const bool pf_noisyor = std::getenv("STRATA_D3_PF_NOISYOR") && std::getenv("STRATA_D3_PF_NOISYOR")[0] == '1';
        const float pf_pmin = knobs.pf_pmin;   // live knob (per request)
        // STRATA_DMA_TRACE=1 (measurement only): host ms this pass spends waiting for the prefetch prediction and for
        // the router (the attention + router graph and its readback), and in the tier calls; with the tier's own
        // counters (Ds41MoeTier::dma_trace_take) one "dma:" line every STRATA_DMA_TRACE_EVERY passes (default 32)
        static const bool dma_tr = ds41_dma_trace_on();
        const double dtr_t0 = dma_tr ? now_ms() : 0.0;
        double dtr_pred = 0.0, dtr_router = 0.0, dtr_tier = 0.0;
        static const std::vector<std::vector<float>> pf_ptab = [] {
            std::vector<std::vector<float>> t;
            const char * fn = std::getenv("STRATA_D3_PF_PTAB");
            std::FILE * f = fn ? std::fopen(fn, "r") : nullptr;
            if (fn && !f) std::fprintf(stderr, "STRATA_D3_PF_PTAB: cannot open %s - the pooled table\n", fn);
            char line[1024];
            while (f && std::fgets(line, sizeof(line), f)) {
                if (line[0] == '#') continue;
                std::vector<float> row;
                char * s = line, * end = nullptr;
                for (float v = std::strtof(s, &end); end != s && row.size() < 16; v = std::strtof(s = end, &end)) row.push_back(v);
                if (row.size() == 16) t.push_back(row);
            }
            if (f) { std::fclose(f); std::fprintf(stderr, "STRATA_D3_PF_PTAB: %zu layer rows from %s\n", t.size(), fn); }
            return t;
        }();
        auto rank_union = [&](int layer) {   // pv_pred (16 per pass token) -> pv_rank: the union, by summed 16 - rank
            const int pfw = pf_win();   // fewer experts per token: only the ranks the pass may route (16 = all)
            if (pf_noisyor) {
                static const float pooled[16] = { 0.950f, 0.886f, 0.798f, 0.690f, 0.564f, 0.433f, 0.313f, 0.228f,
                                                  0.170f, 0.132f, 0.102f, 0.083f, 0.066f, 0.053f, 0.046f, 0.038f };
                const float * p = layer >= 0 && (size_t) layer < pf_ptab.size() ? pf_ptab[(size_t) layer].data() : pooled;
                pv_noq.clear();
                for (int t = 0; t < m; ++t)
                    for (int i = 0; i < pfw; ++i) {
                        const int e = pv_pred[(size_t) t * 16 + i];
                        if (e < 0) continue;
                        auto it = std::find_if(pv_noq.begin(), pv_noq.end(), [&](const std::pair<int, float> & q) { return q.first == e; });
                        if (it == pv_noq.end()) pv_noq.push_back({ e, 1.0f - p[i] }); else it->second *= 1.0f - p[i];
                    }
                std::stable_sort(pv_noq.begin(), pv_noq.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) { return a.second < b.second; });
                pv_rank.clear();
                for (const auto & q : pv_noq) if (1.0f - q.second >= pf_pmin) pv_rank.push_back(q.first);
                return;
            }
            pv_score.clear();
            for (int t = 0; t < m; ++t)
                for (int i = 0; i < pfw; ++i) {
                    const int e = pv_pred[(size_t) t * 16 + i];
                    if (e < 0) continue;
                    auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & p) { return p.first == e; });
                    if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                }
            std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
            pv_rank.clear();
            for (const auto & p : pv_score) pv_rank.push_back(p.first);
        };
        // STRATA_X3B=1 (opt-in; Ds41Dense::x3b_prelaunch): on the D3_PF_ASYNC path, while the host
        // waits for layer l's router readback, finish(l) + layer l + 1's prediction and attention are launched behind a
        // device-side gate that opens when layer l's routed sum lands (x3b_release after the tier call) - their host
        // launch cost leaves the sum -> next dense kernel gap.  The same graphs in the same stream order.  Not with the
        // per-layer extras that put other dense work between the tier call and layer l + 1's launches (the A9 bounded
        // bias, the route trace, STRATA_PRED_LOG2) nor at the drafter's tap layers (read_state after finish).  The
        // file-tier D3 spec prediction (the launcher's STRATA_D3_SPEC=8 on a PC with experts on the SSD) is the one
        // such extra X3B queues itself, in its place (11-17; x3b_prelaunch's spec_il, x3b_spec_wait).
        // STRATA_X3B_STATS=1: the "x3b:" line (X3B on or off).
        const bool x3b = dense.x3b_on() && pf_now && pf_async && knobs.bias_top == 0 && !tracing && !plog2;
        const bool x3b_st = dense.x3b_stats_on();
        bool x3b_ahead = false;    // layer l's prediction + attention were launched during layer l - 1
        bool x3b_spec_pre = false; // ... and its D3 spec prediction (x3b_spec_wait collects its rows)
        bool x3b_gated = false;    // the gate is armed and not yet released (the guard opens it on an error return)
        int x3b_npre = 0;
        double x3b_pre_ms = 0.0, x3b_blk_ms = 0.0;
        // the armed window's host phases (arm, launches done, readback in, tier call returned): a window that ran long
        // is printed, and named in the error when its gate timed out
        double x3b_w0 = 0.0, x3b_w1 = 0.0, x3b_w2 = 0.0;
        std::string x3b_slow;
        struct X3bGuard {
            Ds41Dense & d;
            bool & armed;
            ~X3bGuard() { if (armed) { d.x3b_abort(); armed = false; } }
        } x3b_guard { dense, x3b_gated };
        for (int l = 0; l < n_layer; ++l) {
            if (plog2) pl_ahead_make(l, m);   // STRATA_PRED_LOG2: layer l + k's routing, predicted here
            // STRATA_D3_L3WARM=K: the next K predicted experts (not in VRAM, not prefetched to the card) are read into the
            // CPU's cache under the attention, for the tier's pool (Ds41MoeTier::warm_multi)
            static const int d3_l3warm = std::getenv("STRATA_D3_L3WARM") ? std::atoi(std::getenv("STRATA_D3_L3WARM")) : 0;
            if (pf_now && !pf_async && !remote(l)) {   // (--gpus: another card's layers go without the D3 prefetch)
                pv_pred.resize((size_t) m * 16);
                const bool two = d3_pf_early > 0 && l + 1 < n_layer;   // STRATA_D3_PF_EARLY: the next layer's too
                if (two) pv_next.resize((size_t) m * 16);
                const double tpw = dma_tr ? now_ms() : 0.0;
                const bool pred_ok = two ? dense.predict_multi2(l, pv_pred.data(), pv_next.data()) : dense.predict_multi(l, pv_pred.data());
                if (dma_tr) dtr_pred += now_ms() - tpw;
                if (pred_ok) {
                    rank_union(l);
                    tier.prefetch_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_pf);
                    if (d3_l3warm > 0) tier.warm_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_l3warm);
                    if (plog) { pl_pred = pv_pred; pl_have = true; }
                    if (two) {   // ranked the same way; sent after this layer's tier call (below)
                        std::swap(pv_pred, pv_next);
                        rank_union(l + 1);
                        pv_rank_next = pv_rank;
                        std::swap(pv_pred, pv_next);
                        have_next = true;
                    }
                }
            }
            // A9 (bias_top=N): the bounded cache-aware bias.  The bias for layer l + 1 is built here, one
            // layer ahead, from the router's own prediction for l + 1 made on the state entering l (as the trace's
            // 'R' records and predict_multi2 do): only the N highest-ranked predicted experts that are resident get
            // the route bias; every other expert routes exactly (the file penalty stays as apply_bias sets it).
            // One prediction and one small upload per layer (~1.5% of a pass); the upload lands before layer l + 1's
            // attention is launched, on both the async and the sync prefetch paths.  Lossy like the flat bias;
            // arXiv 2610.34721 measured this shape at +1.7-2.2 points over the flat bias at the same latency.
            if (knobs.bias_top > 0 && bias_now && a.route_bias != 0.0f && rdev1 && !no_tier_batch && m <= Ds41MoeTier::kMaxMulti &&
                l + 1 < n_layer && !remote(l + 1)) {
                pv_pred.resize((size_t) m * 16);
                std::vector<float> & rb = rb_up[(size_t) (l + 1)];
                rb.assign((size_t) TG.n_experts, 0.0f);
                for (int64_t e = 0; e < TG.n_experts; ++e)   // the base: the prune mask and the file penalty, as apply_bias
                    rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) ((l + 1) * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                                   : (a.route_bias_file != 0.0f && !resident_now(l + 1, e) && !T(l + 1).in_arena(l + 1, e)) ? -a.route_bias_file : 0.0f;
                if (dense.predict_multi(l + 1, pv_pred.data())) {
                    rank_union(l + 1);
                    int given = 0;
                    for (int e : pv_rank) {
                        if (given >= knobs.bias_top) break;
                        if (e < 0 || e >= TG.n_experts || !resident_now(l + 1, e)) continue;
                        rb[(size_t) e] = a.route_bias;
                        ++given;
                    }
                }
                dense.set_route_bias((int) (l + 1), rb.data());
            }
            // STRATA_ROUTE_TRACE: the ranked prediction for layers l + a, a in STRATA_ROUTE_TRACE_AHEAD
            // (default "0,2": the prefetch's own layer and the read-ahead's), as an 'R' record {l, l + a, n, n x
            // {expert, score}} - score = P(routed) x 1e6 under noisy-OR (every candidate, before STRATA_D3_PF_P),
            // else the summed 16 - rank - so place_sim.py can act on the prediction the engine would have had at
            // any prefetch / read-ahead setting.  These predict_multi calls are the trace's own: their time goes to
            // the 'V' record's last field, and the real prefetch / read-ahead below predict again as they always did.
            static const std::vector<int> tr_aheads = [] {
                std::vector<int> v;
                const char * s = std::getenv("STRATA_ROUTE_TRACE_AHEAD");
                for (const char * p = s ? s : "0,2"; *p; ) {
                    char * e = nullptr;
                    const long a = std::strtol(p, &e, 10);
                    if (e == p) break;
                    v.push_back((int) a);
                    p = *e ? e + 1 : e;
                }
                return v;
            }();
            if (tracing && rdev1 && !no_tier_batch && m <= Ds41MoeTier::kMaxMulti) {
                const double t_tr = now_ms();
                pv_pred.resize((size_t) m * 16);
                std::vector<int32_t> rec;
                for (int a : tr_aheads) {
                    if (a < 0 || l + a >= n_layer || !dense.predict_multi(l + a, pv_pred.data())) continue;
                    rank_union(l + a);
                    rec.assign({ l, l + a, 0 });
                    if (pf_noisyor)
                        for (const auto & q : pv_noq) { rec.push_back(q.first); rec.push_back((int32_t) ((1.0f - q.second) * 1e6f)); }
                    else
                        for (const auto & q : pv_score) { rec.push_back(q.first); rec.push_back(q.second); }
                    rec[2] = (int32_t) ((rec.size() - 3) / 2);
                    tier.trace_rec('R', rec.data(), (int) rec.size());
                }
                tr_extra_ms += now_ms() - t_tr;
            }
            // D3 speculative NVMe reads (STRATA_D3_SPEC=n, default 0 = off; STRATA_D3_SPEC_AHEAD=k, default 2): the
            // pass's routing for layer l+k predicted from the state entering l (predict_multi, bias included), its
            // union ranked as above, and the NVMe-tier experts among the top n read into the stash by background
            // readers while layers l..l+k-1 run - the verify pass otherwise waits on them (nsys x0.5: pthread_join
            // 7.9 ms/token on the decode thread).  A wrong guess costs SSD bandwidth only.
            const int d3_spec = knobs.d3_spec, d3_spec_ahead = knobs.d3_spec_ahead;   // live knobs (per request)
            // (A7: the launcher can set STRATA_D3_SPEC=8 for every PC - with no file tier (the arena holds
            // every expert: 128 GB and up) the extra prediction per layer buys nothing, so it is skipped)
            // (the layer whose routing it predicts, or -1: none at layer L - x3b_prelaunch asks it for layer l + 1)
            // STRATA_IDLE_PREFETCH: the same prediction feeds its plans (with no file tier it runs for them alone, at
            // STRATA_IDLE_PF_AHEAD; with one, at the read-ahead's STRATA_D3_SPEC_AHEAD, shared)
            const bool spec_ft = rdev1 && !no_tier_batch && d3_spec > 0 && m <= Ds41MoeTier::kMaxMulti && tier.file_tier() > 0;
            const int sp_ahead = spec_ft || !ipf_spec ? d3_spec_ahead : tier.idle_pf_ahead();
            auto spec_layer = [&](int L) {
                return (spec_ft || ipf_spec) && L + sp_ahead < n_layer ? L + sp_ahead : -1;
            };
            if (x3b_spec_pre || spec_layer(l) >= 0) {
                pv_pred.resize((size_t) m * 16);
                // STRATA_X3B: the very prediction, queued in its place behind layer l - 1's gate - its rows only
                const bool spec_ok = x3b_spec_pre ? dense.x3b_spec_wait(pv_pred.data()) : dense.predict_multi(l + sp_ahead, pv_pred.data());
                if (x3b_spec_pre && !dense.x3b_check()) { ferr = dense.last_error() + x3b_slow; return false; }
                x3b_spec_pre = false;
                if (spec_ok) {
                    rank_union(l + sp_ahead);
                    if (spec_ft)
                        T(l + sp_ahead).spec_read(l + sp_ahead, pv_rank.data(), std::min(d3_spec, (int) pv_rank.size()), true);
                    if (ipf_spec && !remote(l + sp_ahead)) tier.idle_plan(l + sp_ahead, pv_rank.data(), (int) pv_rank.size());
                }
            }
            // STRATA_GPU_TIER_STATS=1 (measurement only): the residency a device plan queued behind this
            // layer's router would read - taken before the attention + router graph goes out, as such a plan would be
            if (Ds41MoeTier::gpu_tier_stats_on() && rdev1 && !no_tier_batch && !remote(l))
                tier.gpu_tier_snap(l, m <= Ds41MoeTier::kMaxMulti && run_k == top_k);
            const float * fn = nullptr;
            bool attn_done = false;
            const bool ahead_l = x3b_ahead;   // STRATA_X3B: launched during layer l - 1 (behind the gate it released)
            x3b_ahead = false;
            bool pm_launched = false;
            if (pf_now && pf_async && !remote(l)) {
                if (ahead_l) {
                    pm_launched = true;
                } else {
                    const double tl0 = x3b_st ? now_ms() : 0.0;
                    pm_launched = dense.predict_multi_launch(l);
                    if (x3b_st) x3b_blk_ms += now_ms() - tl0;
                }
            }
            if (pm_launched) {
                pv_pred.resize((size_t) m * 16);
                bool launched = ahead_l;
                if (!ahead_l) {
                    const double tl0 = x3b_st ? now_ms() : 0.0;
                    launched = dense.attn_router_launch(l);   // false: no cached pass graph - attn_router below
                    if (x3b_st) x3b_blk_ms += now_ms() - tl0;
                    if (launched && dma_tr)   // STRATA_DMA_TRACE: "router done" on the card, for the router -> tier round trip
                        if (void * rev = tier.dma_trace_router_event()) dense.record_event(rev);
                }
                const double tpw = dma_tr ? now_ms() : 0.0;
                const bool pred_ok = dense.predict_multi_wait(pv_pred.data());
                if (dma_tr) dtr_pred += now_ms() - tpw;
                // STRATA_X3B: this prediction came out behind layer l - 1's gate - a spin gate that gave up (or a host
                // sync that opened it early) means layer l - 1's finish ran before its routed sum: fail the pass
                if (ahead_l && !dense.x3b_check()) { ferr = dense.last_error() + x3b_slow; return false; }
                if (pred_ok) {
                    rank_union(l);
                    tier.prefetch_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_pf);
                    if (plog) { pl_pred = pv_pred; pl_have = true; }
                }
                if (launched) {
                    const double trw = dma_tr ? now_ms() : 0.0;
                    const bool x3b_try = x3b && l + 1 < n_layer && !remote(l + 1) && l != dm_layers[0] && l != dm_layers[1] &&
                                         l != dm_layers[2];
                    if (x3b_try) {
                        // the readbacks + an event (not a stream sync), then finish(l) + layer l + 1's launches behind
                        // the gate while the card still runs this attention and the host waits for its readback
                        if (!dense.attn_router_collect_issue(l)) { ferr = dense.last_error(); return false; }
                        const double tp0 = now_ms();
                        const int pr = dense.x3b_prelaunch(l, spec_layer(l + 1));
                        if (pr != 0) x3b_gated = true;   // (on -1 the guard opens whatever was armed)
                        if (pr < 0) { ferr = dense.last_error(); return false; }
                        x3b_spec_pre = pr == 2;          // layer l + 1's spec prediction is queued too
                        if (pr > 0) {
                            ++x3b_npre;
                            x3b_w0 = tp0;
                            x3b_w1 = now_ms();
                            if (x3b_st) x3b_pre_ms += x3b_w1 - tp0;
                            dense.x3b_note(l, 2);   // (strata::v41::X3B_PH_READBACK)
                        }
                        if (!dense.attn_router_collect_wait(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
                        if (pr > 0) { x3b_w2 = now_ms(); dense.x3b_note(l, 3); }   // (X3B_PH_TIER)
                    } else if (!dense.attn_router_collect(l, mids.data(), mw.data(), &fn)) {
                        ferr = dense.last_error(); return false;
                    }
                    if (dma_tr) dtr_router += now_ms() - trw;
                    attn_done = true;
                }
            }
            if (!attn_done) {
                const double trw = dma_tr ? now_ms() : 0.0;
                if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
                if (dma_tr) dtr_router += now_ms() - trw;
            }
            keep_k(mids.data(), mw.data(), m);   // STRATA_EXPERTS_USED_DECODE (no-op otherwise): rows now run_k apart
            // STRATA_EXPERTS_DROP_MISS: the tier calls below read v_ids / v_w, rows v_k apart (run_k when nothing dropped).
            // The per-token fallback (STRATA_TIER_BATCH=0) is a run / run_to_dev: no early staging there
            const int v_dk = drop_miss(l, mids.data(), mw.data(), m, !no_tier_batch && rdev1 != nullptr);
            const int v_k = v_dk ? v_dk : run_k;
            const int * v_ids = v_dk ? km_ids.data() : mids.data();
            const float * v_w = v_dk ? km_w.data() : mw.data();
            KmGuard v_kg { v_dk ? &T(l) : nullptr, run_k };
            if (v_dk) T(l).set_active_top_k(v_dk);
            if (plog) plog_rec(plog, l, m, pos, pl_have ? pl_pred.data() : nullptr, mids.data());
            if (plog2 && !pl_ahead[(size_t) l].empty()) plog_rec(plog2, l, m, pos, pl_ahead[(size_t) l].data(), mids.data());
            pl_have = false;
            lap(0);
            float * const rdl = remote(l) ? nullptr : rdev1;   // --gpus: another card's layer through the host
            if (rdl && !no_tier_batch) {
                // D3: ONE batched per-class tier call for the pass's m tokens (the union of their experts is served
                // once), not m run_to_dev calls.  Split at Ds41MoeTier::kMaxMulti (16: a depth-4 pass and a 16-row
                // lookup verify / prompt pass are one call each).
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                const double ttw = dma_tr ? now_ms() : 0.0;
                for (int off = 0; off < m; off += kMM) {
                    const int chunk = std::min(kMM, m - off);
                    if (!tier.run_multi_to_dev(l, chunk, v_ids + (size_t) off * v_k, v_w + (size_t) off * v_k,
                                               fn + (size_t) off * n_embd, rdev1 + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi_to_dev failed"; return false;
                    }
                }
                if (v_dk) T(l).set_active_top_k(run_k);
                if (dma_tr) dtr_tier += now_ms() - ttw;
                void * sum_ev = tier.take_sum_event();
                if (x3b_st && sum_ev) dense.x3b_mark_sum(l, sum_ev);   // STRATA_X3B_STATS: the sum landed
                if (x3b_gated) {
                    // STRATA_X3B: the gate opens behind the sum (null: the tier synced it on the host) - finish(l) and
                    // layer l + 1's prediction + attention, already queued, start the moment it lands
                    const double w3 = now_ms();
                    dense.x3b_note(l, 4);   // (X3B_PH_RELEASE)
                    const bool rel_ok = dense.x3b_release(sum_ev, l);
                    const double w4 = now_ms();
                    dense.x3b_note(l, 5);   // (X3B_PH_RELEASED)
                    // a long armed window: which host phase held it - the release call included (11-17: it was measured
                    // only up to the call, so a call stuck in the driver showed no slow window)
                    if (w4 - x3b_w0 > 100.0) {
                        char b[256];
                        std::snprintf(b, sizeof(b), " [layer %d armed %.1f ms: launches %.1f, readback wait %.1f, tier call %.1f, release call %.1f]",
                                      l, w4 - x3b_w0, x3b_w1 - x3b_w0, x3b_w2 - x3b_w1, w3 - x3b_w2, w4 - w3);
                        x3b_slow = b;
                        std::fprintf(stderr, "x3b: slow window%s\n", b);
                    }
                    if (!rel_ok) { x3b_gated = false; ferr = dense.last_error() + x3b_slow; return false; }
                    x3b_gated = false;
                    x3b_ahead = true;
                    if (dma_tr)   // the pre-launched attention's "router done" (attn_router_launch(l + 1) recorded it there)
                        if (void * rev = tier.dma_trace_router_event()) dense.record_event(rev);
                } else {
                    if (sum_ev) dense.wait_event(sum_ev);   // STRATA_PASS_NOSYNC: order on the card
                    if (x3b_st) dense.x3b_mark_go(l);       // the dense stream's next start: its next launch
                }
                // STRATA_D3_PF_EARLY: the next layer's copies start now - behind this layer's demand copies, ahead of
                // the rest of this layer and the next attention
                if (have_next && !remote(l + 1)) tier.prefetch_next(l + 1, pv_rank_next.data(), (int) pv_rank_next.size(), d3_pf_early);
                have_next = false;
            } else if (rdev1 && !no_tier_batch) {   // --gpus: another card's layer - one batched call, sums to the host
                const double tr0 = now_ms();
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int chunk = std::min(kMM, m - off);
                    if (!T(l).run_multi(l, chunk, v_ids + (size_t) off * v_k, v_w + (size_t) off * v_k,
                                        fn + (size_t) off * n_embd, crouted_m.data() + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi failed"; return false;
                    }
                }
                if (v_dk) T(l).set_active_top_k(run_k);
                pv_rm += now_ms() - tr0;
                have_next = false;
            } else {
                for (int t = 0; t < m; ++t) {
                    const bool ok = rdl ? tier.run_to_dev(l, v_ids + (size_t) t * v_k, v_w + (size_t) t * v_k,
                                                          fn + (size_t) t * n_embd, rdl + (size_t) t * n_embd)
                                        : T(l).run(l, v_ids + (size_t) t * v_k, v_w + (size_t) t * v_k,
                                                   fn + (size_t) t * n_embd, routed1.data());
                    if (!ok) { ferr = "tier.run failed"; return false; }
                    if (void * ev = T(l).take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC
                    if (!rdl) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
                }
            }
            if (v_dk) T(l).set_active_top_k(run_k);   // (the per-token path; the batched ones restored it already)
            lap(1);
            if (!x3b_ahead) {   // (STRATA_X3B: finish(l) is already queued behind the gate)
                const double tl0 = x3b_st ? now_ms() : 0.0;
                if (!dense.finish(l, rdl ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
                if (x3b_st) x3b_blk_ms += now_ms() - tl0;
            }
            if (x3b && (x3b_gated || dense.x3b_armed())) {   // invariant: a layer's gate is released within the layer
                ferr = "STRATA_X3B: layer " + std::to_string(l) + " ended with its gate still armed";
                return false;
            }
            lap(0);
            for (int j = 0; j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                if (!dense.read_state(vx, vpre)) { ferr = dense.last_error(); return false; }
                const size_t HC = vx.size() / ((size_t) m * n_embd);
                for (int t = 0; t < m; ++t)
                    for (size_t h = 0; h < HC; ++h) {
                        const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                        float * dst = vh.data() + (size_t) t * 3 * n_embd + (size_t) j * n_embd;
                        for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                    }
                lap(2);
            }
        }
        vlog.n = (size_t) m * nv;
        vlog.p = dense.logits_host(vlog.n);
        if (!vlog.p) { vlog_own.resize(vlog.n); vlog.p = vlog_own.data(); }
        if (!dense.logits(0, m, vlog.data())) { ferr = dense.last_error(); return false; }
        lap(3);
        if (x3b_st) dense.x3b_stats_pass(m, x3b_npre, x3b_pre_ms, x3b_blk_ms);   // STRATA_X3B_STATS: the "x3b:" line
        if (dma_tr) {   // STRATA_DMA_TRACE: the "dma:" line (per-pass means over the last N passes)
            static const int every = [] { const char * e = std::getenv("STRATA_DMA_TRACE_EVERY"); return e && *e ? std::max(1, std::atoi(e)) : 32; }();
            static long np_ = 0, nrow_ = 0;
            static double wall_ = 0, pred_ = 0, router_ = 0, tier_ = 0;
            ++np_; nrow_ += m; wall_ += now_ms() - dtr_t0; pred_ += dtr_pred; router_ += dtr_router; tier_ += dtr_tier;
            Ds41DmaTrace d;
            if (np_ >= every && tier.dma_trace_take(d)) {
                const double P = (double) np_;
                std::fprintf(stderr, "dma: %ld passes, %.2f rows/pass, %.1f ms/pass | per pass: calls %.1f hits %.1f pcie %.2f file %.2f "
                             "cpu %.2f, demand %.1f MB | prefetch %.2f issued %.2f used (%.0f%%) %.1f MB | gpu wait: prefetch %.2f ms "
                             "demand %.2f ms (%lld pairs, %lld lost) | router->tier gap %.3f ms/call (%lld) | host: pred wait %.2f "
                             "router wait %.2f tier %.2f ms, issue %.3f ms/call | early %lld/%lld calls\n",
                             np_, nrow_ / P, wall_ / P, d.calls / P, d.hits / P, d.pcie / P, d.file / P, d.cpu / P,
                             d.demand_bytes / 1e6 / P, d.pf_issued / P, d.pf_used / P,
                             d.pf_issued > 0 ? 100.0 * (double) d.pf_used / (double) d.pf_issued : 0.0, d.pf_bytes / 1e6 / P,
                             d.wait_pf_ms / P, d.wait_dn_ms / P, (long long) d.waits, (long long) d.waits_lost,
                             d.router_gaps > 0 ? d.router_gap_ms / (double) d.router_gaps : 0.0, (long long) d.router_gaps,
                             pred_ / P, router_ / P, tier_ / P, d.calls > 0 ? d.issue_ms / (double) d.calls : 0.0,
                             (long long) d.early, (long long) d.calls);
                np_ = 0; nrow_ = 0; wall_ = pred_ = router_ = tier_ = 0.0;
            }
        }
        return true;
    };
    const int NP = std::max(1, a.prefill_chunk);
    const int SB = std::max(1, S);                 // the dense sub-batch size
    std::vector<int> cids((size_t) NP * top_k);
    std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), clog;
    const bool routed_host = std::getenv("STRATA_ROUTED_HOST") != nullptr;
    // STRATA_WIDE_PROMPT=1 (opt-in, default off; Keisawaakira/Xolotl's STRATA_WIDE_PROMPT + the prefill-fast deep
    // read's "smaller per-column buffers" and "balanced pieces"): wide layer-major prompt batches.  A batch column costs
    // ~0.36 MiB of VRAM instead of ~0.72: the dense half keeps the FFN's hc state in x_state (ds41_dense wide_prompt) and
    // the MoE input on the card, and the tier reads it there and holds only the buffers it uses (run_chunk_xdev: no
    // c_x / c_out, MMVQ scratch only if a launch needs it, MMQ scratch by the largest expert run).  So --prompt-batch
    // can be ~2x wider and a long prompt sweeps each layer's experts fewer times.  A batch is one tier call per layer
    // (STRATA_CHUNK_BUDGET_GB, when set, splits it into balanced pieces); a batch that is not the last ends on a
    // sub-batch boundary and before the enc-skip replay window.  Same kernels and sums per token.
    // STRATA_CHUNK_CPU keeps its host rows: a batch whose calls may use the CPU pool (<= STRATA_CHUNK_CPU_MAXN tokens)
    // still copies fn to the host, for the pool only.
    bool wide = ds41_env("STRATA_WIDE_PROMPT", "STRATA_D11_WIDE_PROMPT") && std::atoi(ds41_env("STRATA_WIDE_PROMPT", "STRATA_D11_WIDE_PROMPT")) != 0;
    if (wide && (routed_host || mc.cpu_only || multi || !tier.chunk_xdev_ok())) {
        std::fprintf(stderr, "ds41_generate: STRATA_WIDE_PROMPT needs one GPU tier on the dense card (no --gpus, no "
                             "STRATA_ROUTED_HOST, not --experts cpu) - wide prompt batches off\n");
        wide = false;
    }
    // STRATA_WIDE_PROMPT_MIN (rows, default 6144): wide applies only to a FEED of at least this many new rows (the serve
    // request's dh, the one-shot prompt); a shorter feed takes the classic path exactly as with the mode unset (hap its
    // own tensor, classic pb_cap constants, run_chunk_dev, the same batch cuts).  Measured at 192 GB: wide was -16% at
    // 1K and -3% at 4K / 8K, +20% at 32K; at 64 GB +24% at 8K.  The decision is per feed, not per batch, because the
    // serve loop sizes pb_cap with the wide or classic constants before feed() runs, and a wide feed's short last
    // batch must stay on the wide columns it was sized for.  0 = every feed wide (the mode as it was).
    const int wide_min = [] {
        const char * e = ds41_env("STRATA_WIDE_PROMPT_MIN", "STRATA_D11_WIDE_MIN");
        return e && *e ? std::max(0, std::atoi(e)) : 6144;
    }();
    auto wide_for = [&](int rows) { return wide && rows >= wide_min; };
    bool wide_now = false;   // this feed's choice (feed() sets it from its row count)
    if (wide && PB > 1) std::fprintf(stderr, "wide prompt: on for feeds of >= %d rows (batch columns ~0.36 MiB, one tier call per "
                                             "layer and batch%s)\n",
                                     wide_min, tier.chunk_all_mmq() ? "" : "; a class off MMQ keeps the MMVQ scratch");
    // STRATA_BM_PREFILL_READAHEAD=1 (default off; exact): the layer-major batch guesses layer l+1's routing right after
    // layer l's attention (Ds41Dense::guess_route) and hands it to l+1's tier, which starts reading the guessed file-tier
    // experts while layer l's experts run (Ds41MoeTier::prefill_readahead; Big-MoE-Small-Box warm_prefill.py).  Only
    // batches of >= STRATA_BM_PREFILL_READAHEAD_MIN rows (default 1024) guess.
    const bool ra_on = Ds41MoeTier::prefill_readahead_on();
    const int ra_min = Ds41MoeTier::prefill_readahead_min_rows();
    std::vector<int32_t> ra_ids;
    std::vector<float> brouted;                    // the layer-major path's routed sums (n_embd * B)
    if (PB > 0 && !wide) brouted.resize((size_t) n_embd * (size_t) PB);   // (wide: always the device path)
    // a single run_chunk call streams layer l's experts once for its n tokens; the card's chunk buffers (dominated by
    // c_parts = n * top_k * n_embd * 4) must fit ~1 GiB, so a larger batch is run in pieces (each streams once again)
    // budget 3 GB by default (STRATA_CHUNK_BUDGET_GB): at B=8192 the old 1 GB cap split every layer into 2 pieces and each
    // piece re-uploaded the whole layer over PCIe (1.97M vs 0.98M expert copies); the prompt has ~12 GiB of VRAM free
    // (the VRAM expert cache is only seeded after the prompt, from the memory measured then)
    const double chunk_budget = std::getenv("STRATA_CHUNK_BUDGET_GB") ? std::atof(std::getenv("STRATA_CHUNK_BUDGET_GB")) * 1e9 : 3.0e9;
    const int piece = (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) n_embd * 4 * (top_k + 3)));
    // wide: the batch width was sized for its whole tier call (the serve cap counts the tier's columns), so one call
    // per layer unless STRATA_CHUNK_BUDGET_GB is set; then the cap is the budget over a wide call's real per-column
    // bytes (c_parts n_embd * top_k floats + the entry arrays), and the batch is cut into EQUAL pieces
    // (ceil(B / ceil(B / cap))) instead of full pieces plus a sliver that re-streams the layer for a few columns.
    const int piece_wide_max = std::getenv("STRATA_CHUNK_BUDGET_GB")
        ? (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) top_k * (n_embd * 4 + 12)))
        : (1 << 30);
    auto piece_for = [&](int B) -> int {
        if (!wide_now) return piece;
        const int np = (int) (((int64_t) B + piece_wide_max - 1) / piece_wide_max);
        return (int) (((int64_t) B + np - 1) / std::max(1, np));
    };
    // the prompt ids[0, n) at positions pos0..: last_logits = the last row; all_logits (if given) every row
    std::vector<float> last_logits((size_t) nv);
    // the deferred VRAM seed after a prompt: the prompt's buffers go back, the cache opens into the free space, and the
    // decode biases (route bias + file penalty) switch on.  Returns the free GiB measured (for the log).
    double seed_free = 0, seed_gib = 0, seed_s = 0;
    // STRATA_DSPARK=<sidecar> (one-shot, greedy): DSpark speculative decoding - the checkpoint's mtp.0-2 draft stack
    // (ds41_dspark) drafts STRATA_DSPARK_DEPTH (default 4, <= 5) tokens, one dense pass verifies them (pass_v).  The
    // drafter loads here, before the VRAM expert cache measures the free memory, so the cache sizes around it.
    // --serve: STRATA_SERVE_DSPARK=<sidecar> (opt-in) - speculative SAMPLING (spec_accept), same output distribution
    const char * dsp_path = a.serve ? std::getenv("STRATA_SERVE_DSPARK") : std::getenv("STRATA_DSPARK");
    // an empty value means off - the fork's launcher sets it empty when the drafter file is missing (upstream #53)
    if (dsp_path && !*dsp_path) dsp_path = nullptr;
    if (a.serve && dsp_path && (!defer_vram || S < 17)) {
        std::fprintf(stderr, "serve: STRATA_SERVE_DSPARK needs the GPU layer-major prompt path - DSpark off\n");
        dsp_path = nullptr;
    }
#if defined(DS41_HAVE_DSPARK)
    std::unique_ptr<Ds41DSpark> dsp;
    // STRATA_DSPARK_PARK (A7): the drafter a long batch evicted, its device memory freed and its weights kept in host
    // RAM (Ds41DSpark::park) - open_cache() brings it back with one upload instead of a GGUF re-parse.  `dsp` is null
    // meanwhile, exactly as after the old dsp.reset(), so every "if (dsp)" behaves as before.
    std::unique_ptr<Ds41DSpark> dsp_parked;
    // the drafter from its file (open_cache, or STRATA_DSPARK_EAGER at server start); false = a one-shot run's failure
    auto load_drafter = [&]() -> bool {
        if (dsp_path && !dsp) {
            Ds41DSparkConfig sc;
            sc.backend = be;
            sc.sidecar = dsp_path;
            sc.main_gguf = a.model;
            sc.max_append = 16;
            // STRATA_DSPARK_HEAD_ROWS=1 (default off): the head's Markov chain runs over the first DEPTH slots
            // only (STRATA_DSPARK_DEPTH, the same default and clamp as the serve and one-shot loops; neither reads a draft
            // or confidence past it) - Ds41DSparkConfig::head_rows
            if (std::getenv("STRATA_DSPARK_HEAD_ROWS") && std::getenv("STRATA_DSPARK_HEAD_ROWS")[0] == '1')
                sc.head_rows = std::max(1, std::min(5, std::getenv("STRATA_DSPARK_DEPTH") ? std::atoi(std::getenv("STRATA_DSPARK_DEPTH")) : 4));
            if (!std::getenv("STRATA_DSPARK_OWN_HEAD")) sc.shared_head = dense.weight("output.weight");   // -0.7 GB
            dsp = std::make_unique<Ds41DSpark>();
            const double ti = now_ms();
            if (!dsp->init(sc, err)) {
                if (!a.serve) { err = "DSpark init: " + err; return false; }
                // serve: a drafter that will not load must not fail every request (each one re-ran this and returned
                // "ERR VRAM seed failed"): decode without drafts for the run (upstream #53)
                std::fprintf(stderr, "serve: DSpark disabled (%s) - decoding without drafts\n", err.c_str());
                dsp.reset();
                dsp_path = nullptr;
                err.clear();
            } else {
                std::fprintf(stderr, "\nDSpark: %s, %.2f GB of weights on the card (%.1f s)", dsp_path, dsp->weight_bytes() / 1e9,
                             (now_ms() - ti) / 1000.0);
                src.touch_unbuffered();   // STRATA_FILE_TIER_UNMAP: its load mapped the trunk (released now); no-op otherwise
            }
        }
        return true;
    };
#endif
    auto open_cache = [&]() -> bool {
        dense.release_pass();
#if defined(DS41_HAVE_DSPARK)
        if (dsp_path && !dsp && dsp_parked) {
            const double tu = now_ms();
            std::string perr;
            if (dsp_parked->unpark(perr)) {
                dsp = std::move(dsp_parked);
                std::fprintf(stderr, "\nDSpark: %.2f GB of weights back on the card from %s host RAM (%.2f s, STRATA_DSPARK_PARK)",
                             dsp->weight_bytes() / 1e9, dsp->host_pinned() ? "pinned" : "pageable", (now_ms() - tu) / 1000.0);
            } else {   // destroy it (its partial device buffers too), then load from the file as before
                std::fprintf(stderr, "serve: DSpark unpark failed (%s) - reloading from the file\n", perr.c_str());
                dsp_parked.reset();
            }
        }
        if (!load_drafter()) return false;
#endif
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        fr = ds41_vram_quant(fr, "seed: expert cache");   // the cache's slot count (STRATA_VRAM_QUANT_MIB)
        seed_free = fr / 1073741824.0;
        seed_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        // A slot budget of 0 is the tier's "NO LIMIT" (--slots given: the ranked head, up to `slots` experts).  The
        // AUTO path passed it too whenever the card had no room (`free <= margin` clamps seed_gib to 0) - and "no
        // limit" there asked for the WHOLE model's experts: s13 run8 hit exactly that (142.38 GiB requested, 0.53
        // GiB free -> "VRAM seed failed" -> HTTP 400).  With no usable room the cache stays SHUT for this seed and
        // the arena + file tier serve every miss: a slower decode, never a failed request.
        const double min_gib = 1.0;   // below this the admission traffic is not worth the slots
        const bool explicit_slots = a.slots > 0;
        const bool cache_room = explicit_slots || seed_gib >= min_gib;
        ds41_vram_decision("seed: expert cache", "cache %.3f GiB (room %d, explicit slots %d)", seed_gib, (int) cache_room,
                           (int) explicit_slots);
        tier.set_cache_enabled(cache_room);
        tier.set_slot_gib(explicit_slots ? 0.0 : seed_gib);
        if (!cache_room) std::fprintf(stderr, "serve: VRAM %.2f GiB free (margin %.2f) - no room for the expert cache, "
                                             "the arena + file tier serve the decode\n", seed_free, a.vram_margin_gib);
        const double ts = now_ms();
        if (!tier.seed_from_ranked(ranked, err)) {
            // the cache's own allocation is what fails on a busy card (the arena is host RAM and is built first):
            // one more try with the cache off, so the request still runs
            if (cache_room) {
                std::fprintf(stderr, "serve: cache seed failed (%s) - seeding again with the VRAM cache off\n", err.c_str());
                tier.set_cache_enabled(false);
                if (!tier.seed_from_ranked(ranked, err)) return false;
            } else return false;
        }
        seed_s = (now_ms() - ts) / 1000.0;
        proxy_res.clear();   // the real VRAM set from here on
        bias_now = true;     // decode: route bias + file penalty on
        apply_bias();
        return true;
    };
    double fd_dense = 0, fd_tier = 0, fd_logits = 0;   // prompt wall split (printed with the prompt line)
    bool enc_reported = false;
    // ---- env-gated per-layer dumps (determinism bisect): STRATA_LAYER_DUMP=<dir> writes, for every layer of the
    // sequential chunk path, the routed experts' input (fn, the dense half's output) and their weighted sum, plus an
    // index source layer's top-k ids.  Two runs' files diff at the first layer where a run-to-run difference appears,
    // and fn-vs-routed tells whether it is the MoE tier or the attention / indexer half.
    const char * dump_dir = std::getenv("STRATA_LAYER_DUMP");
    auto dumpf = [&](const char * name, const void * p, size_t bytes) {
        std::ofstream f(std::string(dump_dir) + "/" + name, std::ios::binary);
        f.write((const char *) p, (std::streamsize) bytes);
    };
    // tail capture (STRATA_SERVE_DUMP_TAIL, the quality scorer): the last `tail_keep` rows of a feed's logits into
    // tail_logits (row n - tail_keep first) - a teacher-forced continuation's rows without the whole prompt's (a 23K-token
    // prompt's full logits would be 12 GB).  0 = off.
    int tail_keep = 0;
    std::vector<float> tail_logits;
    auto capture_tail = [&](int n, int row0, int nrows, int dense_row0) -> bool {   // rows [row0, row0+nrows) of this feed
        if (tail_keep <= 0) return true;
        const int t0 = n - tail_keep, lo = std::max(row0, t0), hi = row0 + nrows;
        if (lo >= hi) return true;
        if (tail_logits.size() != (size_t) tail_keep * (size_t) nv) tail_logits.assign((size_t) tail_keep * (size_t) nv, 0.0f);
        if (!dense.logits(dense_row0 + (lo - row0), hi - lo, tail_logits.data() + (size_t) (lo - t0) * nv)) { ferr = dense.last_error(); return false; }
        return true;
    };
    int bcap_n = 0, bcap_got = 0;  // serve DSpark seed captured from a layer-major batch (see the layer loop below)
    std::vector<float> bcap_vh;    // [bcap_n][3 * n_embd], as pass_v's vh
    // the layer-major batch width for THIS turn: PB, clamped per request by the VRAM actually free (the --serve loop
    // sets it; a one-shot run keeps PB)
    int pb_cap = PB;
    // STRATA_VRAM_TRACE=1: every allocation-relevant step prints the free VRAM (debugging a shared card)
    const bool vram_trace = std::getenv("STRATA_VRAM_TRACE") != nullptr;
    auto vram_log = [&](const char * where, int a) {
        if (!vram_trace) return;
        size_t tf = 0, tt = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
        std::fprintf(stderr, "vram trace: %s %d free %.2f GiB\n", where, a, tf / 1073741824.0);
    };
    // The serve stdin reader can receive STOP while a long layer-major prompt is running (upstream 6e5ba8d: a client
    // that hung up kept the engine reading for minutes).  Poll only between completed graph / tier operations; a
    // one-shot run has no callback and keeps the existing path.
    std::function<bool()> prefill_stop;
    bool prefill_cancelled = false;
    auto stop_prefill = [&]() -> bool {
        if (prefill_cancelled || (prefill_stop && prefill_stop())) {
            prefill_cancelled = true;
            ferr = "cancelled";
            return true;
        }
        return false;
    };
    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,
                    std::vector<float> * all_logits) -> bool {
        set_run_k(top_k);   // STRATA_EXPERTS_USED_DECODE: a prompt batch routes the router's top_k
        // STRATA_ENC_BOUND: the floor of the state this read continues stays (the serve loop sets -1 for a request that
        // continues nothing, and restores a saved state's floor with it); only a batch with a skipped part sets a new one.
        // (Until the 0.94 fix the floor was reset here, so a request that read no batch kept another conversation's.)
        bool floor_set = false;
        // STRATA_WIDE_PROMPT_MIN: wide or classic for the whole feed (see wide_min); a classic feed's host routed rows are
        // sized as at startup with the mode unset (only read when the tier cannot write to the card)
        wide_now = wide_for(n);
        if (!wide_now && PB > 0 && brouted.size() < (size_t) n_embd * (size_t) PB) brouted.resize((size_t) n_embd * (size_t) PB);
        for (int c0 = 0; c0 < n; ) {
            if (stop_prefill()) return false;
            if (PB > 1 && n - c0 >= 2) {
                // ---- layer-major: up to B tokens of every layer, the experts run once per layer over the whole batch
                int B = std::min(pb_cap, n - c0);
                // wide: a batch that is not the last ends on a sub-batch boundary (its last columns would otherwise be
                // past the enc-skip cut, which rounds down to one, and run layers 21-39 - another sweep of their
                // experts) and before the replay window (else this batch and the last both run layers 21-39); a retry
                // halves to a sub-batch boundary too
                auto halve = [&](int b) { return wide_now ? std::max(SB, b / 2 / SB * SB) : std::max(SB, b / 2); };
                if (wide_now && B < n - c0) {
                    B = std::max(SB, B / SB * SB);
                    if (enc_skip && n - c0 - B < ER && pb_cap >= ER + SB) {
                        const int b2 = (int) ((n - c0 - ER) / SB * SB);
                        if (b2 >= SB) B = b2;
                    }
                }
                // a DSpark batch seed is read off the final layer-major batch; a 1-token remainder would run on the
                // chunk path and capture nothing (prompt = batch + 1: "DSpark batch seed captured nothing", upstream #53)
                if (bcap_n > 0 && n - c0 - B == 1 && B > 2) --B;
                // The batch must FIT: its state columns (~0.31 MiB a column) and its graph (~0.17 MiB a column,
                // 8192 -> 1424 MiB) come out of whatever is left after everything else sharing the card (a
                // desktop's STT server, DSpark's 2.9 GB, the dense half).  A batch that does not fit is retried at
                // HALF the width (twice the sweeps over the file tier, the same answer) down to SB, so a tight card
                // makes a prompt slower instead of failing the request (s13 run8: "prefill failed: graph allocation
                // failed" on a 21K prompt with Whisper's 2.5 GB resident).
                for (;;) {
                    // a wide feed's columns keep hap in x_state, a classic feed's give it its own tensor: a feed whose
                    // choice differs from the columns' reallocates them (same width or wider; graphs rebuilt lazily)
                    if (B > dense.columns() || dense.hap_in_state() != wide_now) {
                        if (dense.set_columns(std::max(B, dense.columns()), wide_now)) { if (rdev1) rdev1 = dense.routed_dev(); }
                        else if (B > SB) {
                            std::fprintf(stderr, "ds41_generate: %d state columns did not fit (%s) - retrying at %d\n",
                                         B, dense.last_error().c_str(), halve(B));
                            B = halve(B);
                            continue;
                        } else { ferr = dense.last_error(); return false; }
                    }
                    ds41_heapcheck("pre-begin_batch", B);
                    if (dense.begin_batch(ids + c0, B, pos0 + c0)) {
                        ds41_heapcheck("begin_batch", B);
                        if (c0 + B < n) dense.engram_next(ids + c0 + B, std::min(B, n - c0 - B), pos0 + c0 + B);
                        break;
                    }
                    if (B > SB) {
                        std::fprintf(stderr, "ds41_generate: batch of %d did not fit (%s) - retrying at %d\n",
                                     B, dense.last_error().c_str(), halve(B));
                        B = halve(B);
                        continue;
                    }
                    ferr = dense.last_error();
                    return false;
                }
                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20
                // are never read for them), rounded DOWN to a sub-batch boundary so a straddling sub-batch still runs
                // the full path (more work, never wrong); the last sub-batches holding the final token are always full.
                int cut = 0;
                if (enc_skip) {
                    const int64_t gpos = (int64_t) pos0 + c0;
                    // STRATA_SERVE_DUMP_TAIL: the captured tail runs every layer too (it used to only under ENC_BOUND: without it a
                    // tail longer than --enc-replay captured rows that never ran layers 21-39).  0 when not dumping.
                    const int64_t ta = (int64_t) pos0 + n - ER - tail_keep - gpos;
                    cut = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                    cut = (cut / SB) * SB;
                    if (enc_bound > 0 && !floor_set && cut < B) {   // the first column past the skip: the replay's start
                        floor_set = true;
                        if (gpos + cut > pos0) dense.set_window_floor(gpos + cut, B, (int) gpos);
                    }
                    if (!enc_reported) { enc_reported = true; std::fprintf(stderr, "enc-skip: first batch B=%d, cut=%d (phase A %d cols, phase B %d)\n", B, cut, cut, B - cut); }
                }
                // wide: whether this batch's tier calls need fn on the host - only for the STRATA_CHUNK_CPU pool, which
                // takes calls of at most STRATA_CHUNK_CPU_MAXN tokens; a layer's calls cover columns [d0, B) in balanced
                // pieces (piece_for), d0 = 0 or the enc-skip cut, so its smallest call is the last piece of B or B - cut
                auto min_call = [&](int w) {
                    const int p = piece_for(w);
                    return w - (w - 1) / p * p;
                };
                const bool b_fn_host = !wide_now || tier.chunk_cpu_wants(cut > 0 && cut < B ? std::min(min_call(B), min_call(B - cut))
                                                                                          : min_call(B));
                dense.set_batch_fn_host(b_fn_host);
                double t0 = now_ms();
                for (int l = 0; l < n_layer; ++l) {
                    if (stop_prefill()) return false;
                    if (l < 4) vram_log("batch layer", l);
                    // layer 20 is the last kv source: for phase A only its persistent KV writes run; layers 21-39 run
                    // nothing at all for phase A (their output columns are never read, and no later layer reads state
                    // a phase-A position would leave past layer 20)
                    const int d0 = (cut > 0 && l >= CL) ? cut : 0;
                    if (cut > 0 && l == CL) {
                        for (int s = 0; s < cut; s += SB) {
                            const int sn = std::min(SB, cut - s);
                            if (!dense.attn_router_span_kv(l, s, sn)) { ferr = dense.last_error(); return false; }
                        }
                    }
                    for (int s = d0; s < B; s += SB) {   // sub-batches in order: the layer's KV state advances as today
                        const int sn = std::min(SB, B - s);
                        if (!dense.attn_router_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                    ds41_heapcheck("attn", l);
                    // STRATA_BM_PREFILL_READAHEAD: layer l+1's columns are [nd0, B) (enc-skip: d0 grows at CL, never
                    // shrinks, so they are columns this layer's attention just wrote); a failed guess only means no read-ahead
                    if (ra_on && l + 1 < n_layer) {
                        const int nl = l + 1, nd0 = (cut > 0 && nl >= CL) ? cut : 0, rows = B - nd0;
                        if (rows >= ra_min && nd0 >= d0) {
                            ra_ids.resize((size_t) rows * (size_t) top_k);
                            if (dense.guess_route(l, nl, nd0, rows, top_k, ra_ids.data()))
                                T(nl).prefill_readahead(nl, ra_ids.data(), rows, top_k);
                        }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    // GPU: the tier writes the routed sums straight into the dense half's routed tensor (no 2 x n*D*4
                    // host round trip per layer); CPU / STRATA_ROUTED_HOST=1: through the host as before
                    float * rdev = (routed_host || remote(l)) ? nullptr : dense.routed_dev();   // --gpus: another card's via the host
                    if (l < 2) vram_log("pre-tier", l);
                    const int bpiece = piece_for(B - d0);   // wide: balanced pieces of this layer's columns; else `piece`
                    for (int s = d0; s < B; s += bpiece) {   // the experts for the dense rows (in pieces if the buffers cap)
                        const int pn = std::min(bpiece, B - s);
                        const bool ok = (wide_now && rdev) ? T(l).run_chunk_xdev(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                                             dense.batch_wts() + (size_t) s * top_k,
                                                                             b_fn_host ? dense.batch_fn() + (size_t) s * n_embd : nullptr,
                                                                             dense.batch_fn_dev() + (size_t) s * n_embd,
                                                                             rdev + (size_t) s * n_embd)
                                      : rdev ? T(l).run_chunk_dev(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                                  dense.batch_wts() + (size_t) s * top_k,
                                                                  dense.batch_fn() + (size_t) s * n_embd,
                                                                  rdev + (size_t) s * n_embd)
                                             : T(l).run_chunk(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                              dense.batch_wts() + (size_t) s * top_k,
                                                              dense.batch_fn() + (size_t) s * n_embd,
                                                              brouted.data() + (size_t) s * n_embd);
                        if (!ok) { ferr = "tier run refused"; return false; }
                    }
                    ds41_heapcheck("tier", l);
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!rdev && !dense.upload_routed(brouted.data(), B)) { ferr = dense.last_error(); return false; }
                    for (int s = d0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.finish_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                    // serve DSpark seed (bcap_n > 0): the main_hidden rows of the batch's last bcap_n tokens - pass_v's
                    // HC-mean of the state leaving dm_layers[j] - read here, so the tail need not re-run as passes
                    if (bcap_n > 0 && c0 + B == n) {
                        for (int j = 0; j < 3; ++j) {
                            if (l != dm_layers[j]) continue;
                            const int cn = std::min(bcap_n, B);
                            bcap_got = cn;
                            if (!dense.read_state_cols(B - cn, cn, vx)) { ferr = dense.last_error(); return false; }
                            const size_t HC = vx.size() / ((size_t) cn * n_embd);
                            for (int t = 0; t < cn; ++t) {
                                float * dst = bcap_vh.data() + (size_t) (bcap_n - cn + t) * 3 * n_embd + (size_t) j * n_embd;
                                for (size_t h = 0; h < HC; ++h) {
                                    const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                                    for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                                }
                            }
                        }
                    }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)
                    all_logits->resize((size_t) (c0 + B) * (size_t) nv);
                    for (int s = 0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.logits(s, sn, all_logits->data() + (size_t) (c0 + s) * nv)) { ferr = dense.last_error(); return false; }
                    }
                    if (c0 + B == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + B == n && !dense.logits(B - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                for (int s = 0; tail_keep > 0 && s < B; s += SB)   // one sub-batch at a time, as above
                    if (!capture_tail(n, c0 + s, std::min(SB, B - s), s)) return false;
                fd_logits += now_ms() - tl;
                if (vram_trace) {   // the batch's peak is its last layers' (the pass graph only grows within a prompt)
                    size_t tf = 0, tt = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                    std::fprintf(stderr, "vram trace: batch at %d (B %d) done: free %.2f GiB; %s\n", pos0 + c0, B,
                                 tf / 1073741824.0, dense.vram_ledger().c_str());
                }
                dense.set_batch_fn_host(true);   // the default for every other path
                c0 += B;
            } else {
                // ---- the sequential chunk path (unchanged): one pass of up to NP tokens through every layer
                const int m = std::min(NP, n - c0);
                double t0 = now_ms();
                if (!dense.begin(ids + c0, m, pos0 + c0)) { ferr = dense.last_error(); return false; }
                for (int l = 0; l < n_layer; ++l) {
                    if (stop_prefill()) return false;
                    const float * fn = nullptr;
                    if (!dense.attn_router(l, cids.data(), cw.data(), &fn)) { ferr = dense.last_error(); return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_fn.f32", l);
                        dumpf(nb, fn, (size_t) m * (size_t) n_embd * 4);
                        std::vector<int> sel; std::vector<float> smask; int K = 0;
                        if (dense.read_sel(l, sel, smask, &K) && K > 0 && !sel.empty()) {
                            std::snprintf(nb, sizeof(nb), "L%02d_sel.i32", l);
                            dumpf(nb, sel.data(), sel.size() * 4);
                        }
                    }
                    ds41_heapcheck("attn", l);
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    const bool ok = m == 1 ? T(l).run(l, cids.data(), cw.data(), fn, crouted.data())
                                           : T(l).run_chunk(l, m, cids.data(), cw.data(), fn, crouted.data());
                    if (!ok) { ferr = "tier run refused"; return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_routed.f32", l);
                        dumpf(nb, crouted.data(), (size_t) m * (size_t) n_embd * 4);
                    }
                    ds41_heapcheck("tier", l);
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!dense.finish(l, crouted.data())) { ferr = dense.last_error(); return false; }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {
                    all_logits->resize((size_t) (c0 + m) * (size_t) nv);
                    if (!dense.logits(0, m, all_logits->data() + (size_t) c0 * nv)) { ferr = dense.last_error(); return false; }
                    if (c0 + m == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + m == n && !dense.logits(m - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                if (!capture_tail(n, c0, m, 0)) return false;
                fd_logits += now_ms() - tl;
                c0 += m;
            }
            progress(c0);
        }
        if (dc.elastic_cols) {   // back to S columns
            if (!dense.set_columns(0)) { ferr = dense.last_error(); return false; }
            if (rdev1) rdev1 = dense.routed_dev();
        }
        // (fixed columns - one-shot, STRATA_ELASTIC_COLS=0 - keep a wide feed's hap-in-x_state columns for the passes
        // after it, exact, as the mode always did; the next classic feed's first batch gives hap its own tensor again)
        each_tier([](Ds41MoeTier & t) { t.release_chunk(); });
        return true;
    };

    // ---- serve
    if (a.serve) {
        std::mutex qm;
        std::condition_variable qc;
        std::deque<std::string> q;
        bool eof = false;
        std::thread([&]() {
            std::string ln;
            while (std::getline(std::cin, ln)) {
                // a CR before the newline (a client writing CRLF where the CRT does not translate it) would turn STOP / QUIT
                // into "ERR unsupported command" and end the csv with a bad id
                while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ')) ln.pop_back();
                std::lock_guard<std::mutex> lk(qm);
                q.push_back(ln);
                qc.notify_all();
            }
            std::lock_guard<std::mutex> lk(qm);
            eof = true;
            qc.notify_all();
        }).detach();
        auto next_line = [&](std::string & out) -> bool {
            std::unique_lock<std::mutex> lk(qm);
            qc.wait(lk, [&] { return !q.empty() || eof; });
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        auto stop_pending = [&]() -> bool {
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end(); ++it)
                if (*it == "STOP") { q.erase(it); return true; }
            return false;
        };
        prefill_stop = stop_pending;
        auto is_stop = [&](int t) { return std::find(a.stop.begin(), a.stop.end(), t) != a.stop.end() || t == (int) G.eos; };
        const int ctxv = (int) ctx;
        std::vector<int> hist;
        std::vector<std::array<uint64_t, 3>> hist_vis;   // the live history's pictures (see "picture-aware reuse" below)
        int snap_len = -1;
        std::vector<float> snap_logits;   // the last prompt row's logits at the snapshot (a repeated prompt samples from them)
        // s6: DSpark's seed rows (main_hidden) of the snapshot's prompt tail - a repeated prompt (regenerate / a client
        // retry) feeds no new tokens, so without them DSpark had nothing to seed from and the whole answer decoded
        // without drafts (measured: 21.9 vs 31.8 tok/s on the same 1200-token edit).  cur_seed = this prompt's tail.
        std::vector<float> snap_seed, cur_seed;
        int snap_seed_n = 0, cur_seed_n = 0;
        int n_req = 0;
        // STRATA_SERVE_PARK (default on): an agent harness sends short UNRELATED requests between the turns of its main
        // conversation (titles, summaries).  Each wrote its own positions from 0 over the main conversation's caches, so
        // the next main turn re-read the whole prompt (2026-10-15 harness: a 454-token side call -> the 20,736-token
        // system prompt again, 69.5 s).  Now the main conversation is parked (Ds41Dense::park: the snapshot rings + the
        // rows the side request can write, to host memory) and unparked when a request continues it.
        // STRATA_SERVE_PARK_MIN: only conversations of >= this many tokens (default 2048); STRATA_SERVE_PARK_MAX: the
        // most positions a side request may write (prompt + max tokens, default 65536 ~ 1 GB of host memory).
        const bool park_on = !(std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0');
        const int park_min = std::getenv("STRATA_SERVE_PARK_MIN") ? std::atoi(std::getenv("STRATA_SERVE_PARK_MIN")) : 2048;
        const int64_t park_max = std::getenv("STRATA_SERVE_PARK_MAX") ? std::atoll(std::getenv("STRATA_SERVE_PARK_MAX")) : 65536;
        std::vector<int> park_hist;       // the parked conversation's tokens (its snapshot length)
        std::vector<float> park_logits;
        int64_t park_rows = 0;            // positions saved: a request writing past this invalidates the park
        // STRATA_SERVE_CACHE_SLOTS=N (default 0 = the single park above; the user's agent workloads, 2026-10-24): keep up
        // to N WHOLE conversations (Ds41Dense::save_conv: the snapshot rings + every row up to their length, ~16 KB per
        // position) in host memory, least recently used out first, within STRATA_SERVE_CACHE_GIB (default 8).  A coding
        // agent alternates big conversations (main task, sub-agents, side calls); park keeps only one and only for small
        // side requests, so every switch between two big ones re-read the whole prompt (a 20K prompt ~35 s; a restore
        // copies ~0.3 GB back in tens of ms).  Before a request runs, the saved conversation it continues best (all of
        // it, or up to 16 positions short of its end - see below) replaces the live one, which is saved first; a request
        // that continues nothing saves the live one before overwriting it.  Conversations under STRATA_SERVE_PARK_MIN
        // tokens (2048) are not kept.
        struct CachedConv { std::vector<int> toks; std::vector<float> logits; std::shared_ptr<Ds41Dense::SavedConv> st; long used = 0; };
        std::vector<CachedConv> convs;
        const int cache_slots = std::getenv("STRATA_SERVE_CACHE_SLOTS") ? std::max(0, std::atoi(std::getenv("STRATA_SERVE_CACHE_SLOTS"))) : 0;
        const double cache_gib = std::getenv("STRATA_SERVE_CACHE_GIB") ? std::atof(std::getenv("STRATA_SERVE_CACHE_GIB")) : 8.0;
        long conv_clock = 0;
        // STRATA_SHARED_PREFIX=1 (default off: none of this runs): many conversations - an agent's sub-agents,
        // parallel chats, other clients - open with the same long system prompt + tool definitions, and every state kept
        // above sits at the END of a prompt, so a new conversation re-read that block from scratch.  The server sends where
        // the block ends in the rendered prompt (GEN key sys_end=<tokens>); a read that passes it saves a restore point there
        // (Ds41Dense::arm_prefix_capture / take_prefix_capture: the rings at that position, copied while the read runs, + the
        // position-indexed rows below it) into this small pool, keyed by the prefix's token hash, shared by every
        // conversation.  A later request whose prompt starts with a pooled prefix - and continues nothing better - resumes
        // there (the "unrelated request" branch below).  The point is sys_end rounded DOWN to a sub-batch boundary, which a
        // resume needs to re-run exactly what a full read of the new prompt runs (tail_point).  STRATA_SHARED_PREFIX_SLOTS
        // (default 4) / _GIB (default 2) bound the pool, least recently used out first; _MIN (default 1024 tokens): shorter
        // blocks are not kept.  ~16 KB a position, as the conversation cache.
        static const bool sh_on = [] { const char * e = std::getenv("STRATA_SHARED_PREFIX"); return e && e[0] == '1'; }();
        const int sh_slots = std::getenv("STRATA_SHARED_PREFIX_SLOTS") ? std::max(1, std::atoi(std::getenv("STRATA_SHARED_PREFIX_SLOTS"))) : 4;
        const double sh_gib = std::getenv("STRATA_SHARED_PREFIX_GIB") ? std::atof(std::getenv("STRATA_SHARED_PREFIX_GIB")) : 2.0;
        const int64_t sh_min = std::getenv("STRATA_SHARED_PREFIX_MIN") ? std::max<int64_t>(2, std::atoll(std::getenv("STRATA_SHARED_PREFIX_MIN"))) : 1024;
        struct SharedSnap { uint64_t key; std::vector<int> toks; std::shared_ptr<Ds41Dense::SavedConv> st; long used = 0; };
        std::vector<SharedSnap> shared_pool;
        long sh_clock = 0;
        auto tok_hash = [](const int * t, size_t n) {   // FNV-1a over the ids' bytes: the pool's key
            uint64_t h = 1469598103934665603ull;
            const unsigned char * b = (const unsigned char *) t;
            for (size_t q = 0, nb = n * sizeof(int); q < nb; ++q) { h ^= b[q]; h *= 1099511628211ull; }
            return h;
        };
        if (sh_on)
            std::fprintf(stderr, "serve: shared prefix restore points on (STRATA_SHARED_PREFIX: %d slots, %.1f GiB, blocks >= %lld tokens)\n",
                         sh_slots, sh_gib, (long long) sh_min);
        // STRATA_SERVE_SNAP_STACK=K (default 0 = off; the speed card 2026-10-24): the last K snapshots of the LIVE
        // conversation (their rings, ~11 MB each, in host memory).  A thinking request is two engine calls - the thinking,
        // then the answer on the reused prefix - and the second one moves the snapshot to the thinking's end; the next user
        // turn's history drops the thinking and diverges just before the OLD snapshot, which REWIND1 cannot reach (it
        // rewinds the latest one): srv-sc-follow.log re-read 4,746 tokens in 13 s at lcp 4095 vs snap 5101.  A request
        // that diverges before the latest snapshot resumes at the newest stacked one <= lcp (or lcp + 1: one token back,
        // as REWIND1).  An entry is pushed only while every request since resumed at or past it (its position rows are
        // in place); a resume below an entry drops it, an unrelated request or a switch of conversation drops them all.
        struct StackSnap { int pos; std::shared_ptr<Ds41Dense::SavedConv> st; };
        std::vector<StackSnap> snap_stack;
        const int snap_stack_k = std::getenv("STRATA_SERVE_SNAP_STACK") ? std::max(0, std::atoi(std::getenv("STRATA_SERVE_SNAP_STACK"))) : 0;
        auto snap_stack_pick = [&](size_t lcp_) -> int {   // the newest usable entry, or -1
            int best = -1;
            for (size_t i = 0; i < snap_stack.size(); ++i) {
                const int s = snap_stack[i].pos;
                if (s >= 256 && ((size_t) s <= lcp_ || (size_t) s == lcp_ + 1) && (best < 0 || s > snap_stack[(size_t) best].pos))
                    best = (int) i;
            }
            return best;
        };
        auto save_live = [&]() {   // the live conversation (its snapshot) into the cache; least recently used out first
            if (cache_slots <= 0 || snap_len < park_min || hist.size() < (size_t) snap_len) return;
            // a conversation with pictures stays out (the cache matches token ids, and a restore has no picture
            // signatures; 0.94 snapshots image prompts, so this is no longer implied by snap_len)
            if (!hist_vis.empty()) return;
            std::vector<int> toks(hist.begin(), hist.begin() + snap_len);
            // a saved conversation that this one continues is superseded (its rows are a subset of these)
            convs.erase(std::remove_if(convs.begin(), convs.end(), [&](const CachedConv & c) {
                return c.toks.size() <= toks.size() && std::equal(c.toks.begin(), c.toks.end(), toks.begin()); }), convs.end());
            auto st = dense.save_conv(snap_len);
            if (!st) return;
            convs.push_back({ std::move(toks), snap_logits, st, ++conv_clock });
            const size_t cap = (size_t) (cache_gib * 1073741824.0);
            auto bytes = [&] { size_t b = 0; for (const auto & c : convs) b += Ds41Dense::saved_bytes(*c.st); return b; };
            while (!convs.empty() && ((int) convs.size() > cache_slots || bytes() > cap)) {
                auto lru = std::min_element(convs.begin(), convs.end(), [](const CachedConv & x, const CachedConv & y) { return x.used < y.used; });
                std::fprintf(stderr, "serve: cache dropped the %zu-token conversation (least recently used)\n", lru->toks.size());
                convs.erase(lru);
            }
            std::fprintf(stderr, "serve: cached the %d-token conversation (%zu saved, %.2f GiB)\n", snap_len, convs.size(),
                         (double) bytes() / 1073741824.0);
        };
        bool cache_open = !defer_vram;   // deferred: the arena only until the first decode
#if defined(DS41_HAVE_DSPARK)
        // STRATA_SERVE_DSPARK: DSpark's window (main_hidden of the last <= 128 positions) is seeded by running a
        // request's last min(128, new) prompt tokens through pass_v instead of the passes / batch (no extra work), and
        // it carries over when the next request continues this one (dsp_next = the position its ring expects next);
        // any other reuse resets it (lower acceptance for a while, never a different output).
        int dsp_next = -1;
        const int sp_depth_serve = std::getenv("STRATA_DSPARK_DEPTH") ? std::max(1, std::min(5, std::atoi(std::getenv("STRATA_DSPARK_DEPTH")))) : 4;
        long sv_pass = 0, sv_acc = 0;
#endif
        // STRATA_RING_FIX (default on; =0 the earlier behaviour, for A/B runs): the ring-slot fixes A, B and C (an
        // audit of the state rings' slot tags).  live_hi = the highest position written into
        // the LIVE rings (window ring pos % RAW, ratio-2 state ring pos % 2) since they last held only this history, or
        // -1 when nothing above hist.size() - 1 was written.  A DSpark / lookup verify pass writes its rejected drafts
        // (and accepted drafts after a stop token) past the history; those rows outlive the request.  A live rewind
        // (fix A, ROLLBACK, REWIND_GAP) shortens the history but leaves its old rows in the rings, so it raises live_hi
        // to the old hist.size() - 1 first.
        static const bool ringfix = !(ds41_env("STRATA_RING_FIX", "STRATA_D11_RINGFIX") && ds41_env("STRATA_RING_FIX", "STRATA_D11_RINGFIX")[0] == '0');
        int64_t live_hi = -1;
        // STRATA_SERVE_STEP_MAX: new prompt tokens up to which a request runs on the decode side - passes of
        // STRATA_SERVE_PASS_M tokens with the VRAM cache and the decode biases (~40 tok/s) - instead of the exact
        // layer-major batch (~16.8 s + n / 250 tok/s; measured crossover ~765 tokens); 0 = always the batch (every prompt
        // token exact)
        const int step_max = std::getenv("STRATA_SERVE_STEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_STEP_MAX")) : 768;
        // STRATA_SERVE_PASS_BIAS=0: those passes route exactly (the decode biases only while decoding)
        const bool pass_bias = !(std::getenv("STRATA_SERVE_PASS_BIAS") && std::getenv("STRATA_SERVE_PASS_BIAS")[0] == '0');
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) resident_all(),
                    (long long) tier.arena_experts());
#if defined(DS41_HAVE_DSPARK)
        // STRATA_DSPARK_PREFETCH=0 turns it off: the drafter loads on the first request (open_cache), 2.6-4.9 s from a
        // cold file against 1.1-1.4 s when the file is already in the OS cache (measured in-run reloads).
        // Read the file once here, after the arena is pinned, on a detached thread: it only warms the OS cache - no VRAM,
        // no change to any output - so the user's first turn pays the upload alone (measured: 2.8 s of turn 1).
        // STRATA_DSPARK_EAGER=1 (default off): the drafter loads HERE, before READY, instead of inside the first
        // request's open_cache (measured: 2.8 s of turn 1).  open_cache then finds it loaded and measures the free VRAM
        // for the expert cache with the same weights on the card as before.  What changes: the first request's prompt
        // runs with the drafter's ~2.4 GB already on the card (later prompts always did), so a first BATCH turn longer
        // than the narrower width cap takes a narrower batch.  No read-ahead then (the load just read the file).
        static const bool dsp_eager = std::getenv("STRATA_DSPARK_EAGER") && std::getenv("STRATA_DSPARK_EAGER")[0] == '1';
        if (dsp_eager && dsp_path && !dsp) {
            const double te = now_ms();
            load_drafter();   // serve: a drafter that will not load is switched off for the run (never false here)
            std::fprintf(stderr, "\nserve: DSpark loaded at start (STRATA_DSPARK_EAGER, %.1f s)\n", (now_ms() - te) / 1000.0);
        }
        if (dsp_path && !dsp && !(std::getenv("STRATA_DSPARK_PREFETCH") && std::getenv("STRATA_DSPARK_PREFETCH")[0] == '0')) {
            std::thread([p = std::string(dsp_path)]() {
                const double t0 = now_ms();
                FILE * f = std::fopen(p.c_str(), "rb");
                if (!f) return;
                std::vector<char> buf((size_t) 16 << 20);
                size_t n = 0, got;
                while ((got = std::fread(buf.data(), 1, buf.size(), f)) > 0) n += got;
                std::fclose(f);
                std::fprintf(stderr, "serve: DSpark file read ahead, %.2f GB in %.1f s\n", n / 1e9, (now_ms() - t0) / 1000.0);
            }).detach();
        }
#endif
        host_pin.pin();   // STRATA_DS41_HOST_PIN (off by default): this thread runs every request from here on
        std::printf("READY %d stop\n", ctxv);
        std::fflush(stdout);
        std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d)\n", ctxv, NP);
        // per-request decode biases (GEN keys route_bias= / route_bias_file=, from the server's "strata_tune"): a request
        // that names neither gets the --route-bias / --route-bias-file the engine started with, so one server can answer
        // exact-routing requests (both 0) beside the fast default without a restart.  apply_bias() writes zeros over a
        // bias it set earlier, so an exact request never inherits the previous one's.
        const float cfg_route_bias = a.route_bias, cfg_route_bias_file = a.route_bias_file;
        // the smart router's ranks (route_protect= / route_pool=, engines built with STRATA_RB_SMART / _PROTECT / _POOL):
        // a request that names neither gets the env's
        auto env_rank = [](const char * k) { const char * e = std::getenv(k); return e ? std::max(0, std::atoi(e)) : 0; };
        const int cfg_rprot = env_rank("STRATA_RB_PROTECT"), cfg_rpool = env_rank("STRATA_RB_POOL");
        int cur_rprot = cfg_rprot, cur_rpool = cfg_rpool;
        // STRATA_SERVE_REWIND_GAP (tokens; default 16, 0 = off): a request that continues the live conversation but
        // is a FEW tokens shorter than the live history.  That is what an aborted turn leaves behind: the harness
        // cancels the in-flight request (ctrl+x / chat:sendNow) and drops its partial assistant turn, so the next
        // template render differs by a few tokens (DeepSeek's thinking re-render / an empty assistant turn) and
        // lcp sits just under hist.size() - every exact reuse branch above misses and with pool_on the whole
        // conversation is parked as "unrelated" and re-read.  Positional caches make a short rewind safe inside the
        // window ring (RAW - SWA = 16); the forward pass re-feeds from the common prefix and overwrites the stale
        // tail.  Same rule as the partial unpark, which is why 16 is the ceiling.
        static const int rewind_gap = std::getenv("STRATA_SERVE_REWIND_GAP") ? std::atoi(std::getenv("STRATA_SERVE_REWIND_GAP")) : 16;
        std::string line;
        bool previous_vision = false;
        // 0.94 picture-aware reuse: every image span is the same run of image_token, so token ids alone cannot tell two
        // pictures apart.  hist_vis keeps, for the live history, each image run's {start, length, hash of its embedding
        // rows}; a request continues the history only up to the first run whose picture differs (or is missing on
        // either side).  Before, any request after an image prompt re-read the whole conversation (a 40K agent chat with
        // one screenshot: ~60 s a turn).  A conversation restored from the cache or the park has no signatures: its
        // runs never match, so it re-reads from its first picture (safe, only slower).  (hist_vis is declared next to hist.)
        while ((ds41_end_if_gpu_fault(), next_line(line))) {   // (the check: before every wait for a request, on all the paths that `continue`)
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            dense.clear_vision();
            // GENI = GEN with an image-embedding file: GENI <max_new> [k=v ..] <embeddings file> <ids csv> (the file
            // holds one strata-vision record per image, in prompt order; the engine lands them on the image_token runs).
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, emb_file;
            int max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            float req_rb = cfg_route_bias, req_rbf = cfg_route_bias_file;
            bool bias_top_changed = false;   // a bias_top= key: the flat bias must be rebuilt (apply_bias) for the new mode
            bool named_rb = false;           // the request set its own route_bias / route_bias_file (the answer phase leaves it)
            int req_rprot = cfg_rprot, req_rpool = cfg_rpool;
            int req_sys_end = 0;   // STRATA_SHARED_PREFIX: where the prompt's system + tools block ends (the server's sys_end=)
            ss >> w >> max_new;
            while (ss >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) {   // the two positionals: GENI's embeddings file then the csv (GEN: just the csv)
                    if (geni && emb_file.empty()) emb_file = w; else csv = w;
                    continue;
                }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                // the server range-checks these (sampling_keys); the engine does again: a non-finite temperature would
                // turn every probability into NaN (std::uniform_real_distribution(0, NaN) is undefined), an infinite
                // route bias would leave only resident experts selectable
                const float fv = (float) std::atof(v.c_str());
                const bool fin = std::isfinite(fv);
                if (k == "temperature") temp = fin ? std::max(0.0f, fv) : 0.0f;
                else if (k == "top_p") top_p = fin ? fv : 1.0f;
                else if (k == "top_k") tk = std::atoi(v.c_str());
                else if (k == "min_p") min_p = fin ? fv : 0.0f;
                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
                else if (k == "route_bias") { req_rb = fin ? std::min(8.0f, std::max(0.0f, fv)) : 0.0f; named_rb = true; }
                else if (k == "route_bias_file") { req_rbf = fin ? std::min(8.0f, std::max(0.0f, fv)) : 0.0f; named_rb = true; }
                else if (k == "route_protect") req_rprot = std::min(32, std::max(0, std::atoi(v.c_str())));
                else if (k == "route_pool") req_rpool = std::min(32, std::max(0, std::atoi(v.c_str())));
                else if (k == "sys_end") req_sys_end = std::max(0, std::atoi(v.c_str()));
                // the live decode knobs (LiveKnobs above): for this request and the ones after it until changed again
                else if (k == "d3_spec") knobs.d3_spec = std::min(64, std::max(0, std::atoi(v.c_str())));
                else if (k == "d3_spec_ahead") knobs.d3_spec_ahead = std::min(8, std::max(1, std::atoi(v.c_str())));
                else if (k == "d3_pf") knobs.d3_pf = std::min(6, std::max(0, std::atoi(v.c_str())));   // 6 = the prefetch staging slots (kMaxPf)
                else if (k == "pf_p") knobs.pf_pmin = fin ? std::min(1.0f, std::max(0.0f, fv)) : 0.0f;
                else if (k == "bias_refresh") knobs.bias_refresh = std::min(64, std::max(0, std::atoi(v.c_str())));
                else if (k == "bias_top") { const int nb = std::min(64, std::max(0, std::atoi(v.c_str()))); bias_top_changed = nb != knobs.bias_top; knobs.bias_top = nb; }
            }
            // a changed bias is applied again just before the decode: a repeated prompt (regenerate) runs no prompt pass,
            // and the passes are what turn the decode bias on - without this it decoded with the previous request's
            bool bias_changed = false;
            // 0.94 phase-aware route bias (opt-in, STRATA_ANSWER_ROUTE_BIAS=<f>, STRATA_ANSWER_ROUTE_BIAS_FILE=<f>, default the
            // same): the request's bias while the model thinks, this one from </think> on (the answer and its tool calls).
            // A lower answer bias buys quality where the user reads it and costs answer speed (cheaper verify passes are
            // all the bias buys; separate-server runs show it does not change drafter acceptance).  The thinking is
            // unchanged.  A prompt that already ends its thinking (thinking off) starts in the answer phase.
            static const float ans_rb = std::getenv("STRATA_ANSWER_ROUTE_BIAS") ? std::max(0.0f, (float) std::atof(std::getenv("STRATA_ANSWER_ROUTE_BIAS"))) : -1.0f;
            static const float ans_rbf = std::getenv("STRATA_ANSWER_ROUTE_BIAS_FILE") ? std::max(0.0f, (float) std::atof(std::getenv("STRATA_ANSWER_ROUTE_BIAS_FILE"))) : ans_rb;
            static const int think_beg_id = std::getenv("STRATA_THINK_BEGIN_ID") ? std::atoi(std::getenv("STRATA_THINK_BEGIN_ID")) : 128821;
            static const int think_end_id = std::getenv("STRATA_THINK_END_ID") ? std::atoi(std::getenv("STRATA_THINK_END_ID")) : 128822;
            bool in_answer = false;   // set from the prompt once its ids are read (below)
            if (req_rb != a.route_bias || req_rbf != a.route_bias_file || bias_top_changed) {
                a.route_bias = req_rb;
                a.route_bias_file = req_rbf;
                bias_changed = true;
                std::fprintf(stderr, "serve: decode route bias %.2f, file penalty %.2f%s\n", req_rb, req_rbf,
                             (req_rb == 0.0f && req_rbf == 0.0f) ? " (exact routing)" : "");
            }
            auto enter_answer = [&](int tok) {   // called on every emitted token
                if (ans_rb < 0.0f || named_rb || in_answer || tok != think_end_id) return;   // (a request's own bias: kept, e.g. exact)
                in_answer = true;
                if (a.route_bias != ans_rb || a.route_bias_file != ans_rbf) {
                    a.route_bias = ans_rb;
                    a.route_bias_file = ans_rbf;
                    apply_bias();
                    std::fprintf(stderr, "serve: answer phase: route bias %.2f, file penalty %.2f\n", ans_rb, ans_rbf);
                }
            };
            if ((req_rprot != cur_rprot || req_rpool != cur_rpool) && dense.set_route_select(req_rprot, req_rpool)) {
                cur_rprot = req_rprot;
                cur_rpool = req_rpool;
                std::fprintf(stderr, "serve: smart routing: exact top %d always kept, picks inside the exact top %d%s\n",
                             req_rprot, req_rpool, (req_rprot == 0 && req_rpool == 0) ? " (off: the flat bias)" : "");
            }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
            if (ans_rb >= 0.0f && !named_rb) {   // phase-aware bias: a prompt that already ends its thinking starts in the answer phase
                long last_beg = -1, last_end = -1;
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] == think_beg_id) last_beg = (long) i;
                    else if (ids[i] == think_end_id) last_end = (long) i;
                }
                in_answer = last_end > last_beg;
                if (in_answer && (a.route_bias != ans_rb || a.route_bias_file != ans_rbf)) {
                    a.route_bias = ans_rb;
                    a.route_bias_file = ans_rbf;
                    bias_changed = true;
                    std::fprintf(stderr, "serve: answer phase from the start (thinking off): route bias %.2f\n", ans_rb);
                }
            }
            if ((int64_t) ids.size() + std::max(0, max_new) > ctxv) {
                std::printf("ERR prompt (%zu) + max tokens (%d) exceed the context (%d)\n", ids.size(), max_new, ctxv);
                std::fflush(stdout);
                continue;
            }
            // vision: resolve this request's image rows onto its image_token runs and hand them to the dense layers
            std::vector<float> vis_rows;
            std::vector<int64_t> vis_pos;
            const bool is_vision = geni;
            if (is_vision) {
                if (!dense.vision()) { std::printf("ERR image input requires --vision\n"); std::fflush(stdout); continue; }
                std::string verr;
                std::vector<std::vector<float>> recs;
                std::vector<int64_t> counts;
                const int64_t ne = dense.geom().n_embd;
                if (!read_sve1(emb_file, recs, counts, ne, (int64_t) ids.size(), verr)
                    || !map_vision(ids, a.image_token, recs, counts, vis_rows, vis_pos, verr)) {
                    std::printf("ERR %s\n", verr.c_str()); std::fflush(stdout); continue;
                }
                if (vis_rows.empty()) { std::printf("ERR GENI prompt has no image tokens\n"); std::fflush(stdout); continue; }
                // the reuse test below compares each image run's picture (hist_vis), so an image prompt may continue the
                // history; the park and the conversation cache stay text-only.
                if (!previous_vision) save_live();   // a text conversation it overwrites stays cached (STRATA_SERVE_CACHE_SLOTS)
                dense.drop_park();
                park_hist.clear();
                dense.set_vision(vis_rows.data(), vis_pos.data(), (int) vis_pos.size());
            } else {
                dense.clear_vision();
            }
            previous_vision = is_vision;
            // this request's image runs: consecutive vis_pos rows, hashed (FNV-1a over the run's embedding bytes)
            std::vector<std::array<uint64_t, 3>> req_vis;
            if (is_vision) {
                const size_t ne = (size_t) dense.geom().n_embd;
                for (size_t i = 0; i < vis_pos.size(); ) {
                    size_t j = i + 1;
                    while (j < vis_pos.size() && vis_pos[j] == vis_pos[j - 1] + 1) ++j;
                    uint64_t h = 1469598103934665603ull;
                    const unsigned char * b = (const unsigned char *) (vis_rows.data() + i * ne);
                    for (size_t q = 0, nb = (j - i) * ne * sizeof(float); q < nb; ++q) { h ^= b[q]; h *= 1099511628211ull; }
                    req_vis.push_back({ (uint64_t) vis_pos[i], (uint64_t) (j - i), h });
                    i = j;
                }
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            {   // the prefix holds only where every image run inside it is the same picture on both sides
                size_t cut = lcp;
                auto has = [](const std::vector<std::array<uint64_t, 3>> & v, const std::array<uint64_t, 3> & r) {
                    for (auto & x : v) if (x == r) return true;
                    return false;
                };
                for (auto & r : hist_vis) if (r[0] < cut && !has(req_vis, r)) cut = (size_t) r[0];
                for (auto & r : req_vis) if (r[0] < cut && !has(hist_vis, r)) cut = (size_t) r[0];
                // an image_token run inside the prefix that neither side has a signature for (a restored text-only
                // history with image tokens, a GEN request with image tokens): no reuse past its start
                for (size_t i = 0; dense.vision() && i < cut; ++i)
                    if (ids[i] == a.image_token) {
                        bool known = false;
                        for (auto & r : req_vis) if ((uint64_t) i >= r[0] && (uint64_t) i < r[0] + r[1]) { known = true; break; }
                        if (!known) { cut = i; break; }
                    }
                if (cut < lcp) {
                    std::fprintf(stderr, "serve: picture-aware reuse: prefix %zu -> %zu (a picture differs there)\n", lcp, cut);
                    lcp = cut;
                }
            }
            // STRATA_SNAP_TAIL: where a tail resume re-feeds from, for this request (ids.size() tokens) sharing an even
            // prefix lpe with a snapshot, and the window floor it must run with (-1 = keep / none).  The tail restores the
            // window rows of EVERY layer, but the layers past the last kv source (enc-skip's cut) hold exact rows only
            // where a read ran them, so the resume re-runs what a full read of this prompt would run there:
            //  - STRATA_ENC_BOUND (the launcher's 128): a full read runs those layers from F = (n - ER) / SB * SB with
            //    their windows stopping at F (rows below F are masked).  Resume at F (or, when the prefix ends before F,
            //    at the sub-batch boundary <= it, so the feed's own cut lands on F too) with the floor set to F: every
            //    row a full read computes is computed the same way, from the same positional rows.
            //  - plain enc-skip: the final rows of a read depend on the layers past the cut only within its replay (ER >=
            //    the window reach), so a resume re-feeds at least the last ER tokens.
            //  - no enc-skip: every row is exact; resume at the prefix.
            // -1 = no tail resume (a short prompt, or STRATA_SERVE_DUMP_TAIL, which shifts a read's cut by its delta).
            static const bool tail_dump = std::getenv("STRATA_SERVE_DUMP_TAIL") && std::atoi(std::getenv("STRATA_SERVE_DUMP_TAIL")) > 0;
            auto tail_point = [&](size_t lpe, int64_t & floor_out) -> int64_t {
                floor_out = -1;
                const int64_t n_req = (int64_t) ids.size();
                if (enc_skip && enc_bound > 0) {
                    if (tail_dump) return -1;
                    const int64_t f = (n_req - ER) / SB * SB;
                    if (f <= 0) return -1;
                    floor_out = f;
                    return (int64_t) lpe >= f ? f : (int64_t) lpe / SB * SB;
                }
                if (enc_skip) return std::min<int64_t>((int64_t) lpe, (n_req - ER) & ~(int64_t) 1);
                return (int64_t) lpe;
            };
            // STRATA_SHARED_PREFIX: the longest pooled prefix this prompt starts with (leaving at least a token to read).  It
            // is usable when a full read of THIS prompt re-runs everything from it on - tail_point(ps) == ps: with
            // STRATA_ENC_BOUND the read's cut lands on the same sub-batch boundary F, whose floor (sh_fl) the resume sets.
            int sh_pick = -1;
            int64_t sh_ps = 0, sh_fl = -1;
            bool sh_hit = false;
            if (sh_on && !is_vision) {
                for (size_t i = 0; i < shared_pool.size(); ++i) {
                    const auto & e = shared_pool[i];
                    const size_t n = e.toks.size();
                    if (n >= ids.size() || (int64_t) n <= sh_ps || tok_hash(ids.data(), n) != e.key ||
                        !std::equal(e.toks.begin(), e.toks.end(), ids.begin())) continue;
                    int64_t fl = -1;
                    if (tail_point(n, fl) != (int64_t) n) continue;
                    sh_pick = (int) i;
                    sh_ps = (int64_t) n;
                    sh_fl = fl;
                }
            }
            bool unparked = false;
            int reuse_override = -1;   // a partial unpark: resume at the longest common prefix (see below)
            size_t lp_show = 0;        // how far this request actually continues the PARKED conversation (logged)
            if (cache_slots > 0 && !is_vision) {   // STRATA_SERVE_CACHE_SLOTS: a saved conversation it continues better?
                int best = -1;
                size_t best_lp = 0;
                // a resume point inside a saved conversation is exact only within the window ring's 16 spare positions
                // of its end, at an even position (a fresh ratio-2 compressor group) - the rules ROLLBACK / REWIND1 keep;
                // further back the first re-fed queries would attend window rows the ring no longer holds
                for (size_t i = 0; i < convs.size(); ++i) {
                    const auto & t = convs[i].toks;
                    size_t lp = 0;
                    while (lp < t.size() && lp < ids.size() && t[lp] == ids[lp]) ++lp;
                    if (lp < t.size()) lp &= ~(size_t) 1;
                    bool usable = lp == t.size() || (lp + 16 >= t.size() && lp >= (size_t) park_min);
                    // STRATA_SNAP_TAIL: or from the resume point inside the snapshot's kept tail (tail_point)
                    if (!usable && lp >= (size_t) park_min) {
                        int64_t fl = -1;
                        const int64_t tf = dense.saved_tail_from(*convs[i].st), rp = tail_point(lp, fl);
                        if (tf >= 0 && rp >= tf && rp >= park_min && rp > 0) { usable = true; lp = (size_t) rp; }
                    }
                    if (usable && lp > lcp && lp > best_lp) { best = (int) i; best_lp = lp; }
                }
                if (best >= 0 && sh_pick >= 0 && (int64_t) best_lp < sh_ps) best = -1;   // STRATA_SHARED_PREFIX resumes further on
                if (best >= 0) {
                    CachedConv c = std::move(convs[(size_t) best]);
                    convs.erase(convs.begin() + best);
                    save_live();   // the conversation it replaces stays cached
                    if (dense.load_conv(*c.st)) {
                        const bool whole = best_lp == c.toks.size();
                        hist = c.toks;
                        hist_vis.clear();
                        dense.set_history(hist.data(), (int64_t) best_lp, 0);   // Engram's n-grams look back over these
                        if (whole) {   // resumes through restore() at the snapshot, as a full unpark
                            snap_len = (int) c.toks.size();
                            snap_logits = c.logits;
                            snap_seed_n = 0;
                        } else {       // a partial resume (<= 16 back, even): the tail past it is re-fed - from the
                                       // saved snapshot's rings, so copy them into the live ones (a whole resume does
                                       // that through restore() below)
                            dense.restore();
                            // STRATA_SNAP_TAIL: further back than the ring's 16, the window rows come from the tail
                            if (best_lp + 16 < c.toks.size()) {
                                int64_t fl = -1;
                                (void) tail_point(best_lp, fl);   // (best_lp is already its resume point: the floor)
                                if (dense.snap_tail_resume((int64_t) best_lp)) {
                                    if (fl >= 0) dense.set_window_floor(fl);
                                    std::fprintf(stderr, "snap-tail: resumed at %zu of %zu (kept %lld, cached conversation)\n",
                                                 best_lp, c.toks.size(), (long long) dense.snap_tail_kept());
                                } else best_lp = 0;   // (cannot happen: `usable` checked the same tail) - re-read, never inexact
                            }
                            hist.resize(best_lp);
                            snap_len = -1;
                            reuse_override = (int) best_lp;
                        }
                        lcp = best_lp;
                        unparked = true;
#if defined(DS41_HAVE_DSPARK)
                        dsp_next = -1;   // DSpark's window holds the previous conversation's states
#endif
                        std::fprintf(stderr, "serve: cache restored the %zu-token conversation at lcp %zu%s (%zu saved)\n",
                                     c.toks.size(), best_lp, whole ? "" : " (partial: re-feeds from there)", convs.size());
                    } else {
                        hist.clear();
                        snap_len = -1;
                        lcp = 0;
                    }
                }
            }
            if (dense.parked()) {   // does this request continue the parked conversation (better than the live one)?
                size_t lp = 0;
                while (lp < park_hist.size() && lp < ids.size() && park_hist[lp] == ids[lp]) ++lp;
                lp_show = lp;
                // STRATA_SERVE_PARK_PARTIAL (default on): a request that continues MOST of the parked conversation is
                // worth restoring, not only one that continues ALL of it.  A chat template that re-renders its tail -
                // or an EMPTY assistant turn, as the s13 harness produces - leaves lp just under park_hist.size(), and
                // requiring an exact prefix threw away a 21,349-token cache over a 27-token difference: run8/run15 both
                // logged "reuse lcp=27 hist=0 snap=-1 prompt=23531 -> reused=0" and re-read the whole prompt.
                // Positional caches make a partial resume safe: unpark() restores positions [0, park_size) and the
                // forward pass then re-feeds from lp, overwriting the stale tail (that is the documented fix: "resume
                // at the longest common prefix even when it is below the snapshot").
                static const bool park_partial = !(std::getenv("STRATA_SERVE_PARK_PARTIAL") && std::getenv("STRATA_SERVE_PARK_PARTIAL")[0] == '0');
                // D11 ring fix C (AUDITS_AND_PORTS.md "Audit ring-tags" C): as the conversation-cache path above - resume
                // at an even position within the window ring's spare (16) of the park's end, and make the parked
                // snapshot rings the live ones with restore() (unpark() writes back only the *_snap rings and the rows;
                // the live rings were still the side request's).  Off (=0): lp_min = 3/4 of the park, no rounding, no
                // restore().
                size_t lpp = lp;
                if (ringfix && lp < park_hist.size()) lpp = lp & ~(size_t) 1;
                const size_t lp_min = (size_t) std::max<int64_t>(park_min, (int64_t) park_hist.size() * 3 / 4);
                bool lp_ok = ringfix ? (lpp >= (size_t) std::max(0, park_min) && lpp + (size_t) dense.ring_spare() >= park_hist.size())
                                     : lpp >= lp_min;
                // STRATA_SNAP_TAIL (with the ring fixes): or from the resume point inside the parked snapshot's kept tail
                int64_t pfl = -1;
                bool p_tail = false;
                if (ringfix && !lp_ok && lp < park_hist.size() && lpp >= (size_t) std::max(0, park_min)) {
                    const int64_t ptf = dense.parked_tail_from(), rp = tail_point(lpp, pfl);
                    if (ptf >= 0 && rp >= ptf && rp >= park_min && rp > 0) { lp_ok = p_tail = true; lpp = (size_t) rp; }
                }
                if (park_partial && lp < park_hist.size() && lp_ok && lpp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation at lcp %zu (partial: re-feeds from there)\n",
                                 park_hist.size(), lpp);
                    if (ringfix) dense.restore();   // fix C: unpark() re-armed the parked snapshot; its rings go live
                    if (p_tail) {   // STRATA_SNAP_TAIL
                        if (dense.snap_tail_resume((int64_t) lpp)) {
                            if (pfl >= 0) dense.set_window_floor(pfl);
                            dense.set_history(park_hist.data(), (int64_t) lpp, 0);   // Engram: the side request wrote its own
                            std::fprintf(stderr, "snap-tail: resumed at %zu of %zu (kept %lld, parked conversation)\n",
                                         lpp, park_hist.size(), (long long) dense.snap_tail_kept());
                        } else {   // (cannot happen: lp_ok checked the same tail) - re-read rather than resume inexactly
                            lpp = 0;
                        }
                    }
                    hist = park_hist;
                    hist_vis.clear();
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) lpp;
                    lcp = lpp;
                    if (lpp == 0) hist.clear();
                    unparked = true;
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;        // DSpark's window holds the side request's states
#endif
                } else if (lp == park_hist.size() && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation\n", park_hist.size());
                    hist = park_hist;
                    hist_vis.clear();
                    snap_len = (int) park_hist.size();
                    snap_logits = park_logits;
                    snap_seed_n = 0;
                    lcp = lp;
                    unparked = true;   // the live rings are the side request's: resume through restore()
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;     // DSpark's window holds the side request's states
#endif
                } else if ((int64_t) ids.size() + std::max(0, max_new) > park_rows) {
                    dense.drop_park();   // this request writes past the saved rows
                    std::fprintf(stderr, "serve: park dropped (request reaches past %lld positions)\n", (long long) park_rows);
                }
            }
            int reused = 0;
            bool regen = false;   // the snapshot's own prompt again: nothing new to feed
            // STRATA_SERVE_ROLLBACK=K (default 0 = off): a prompt that continues all but the last <= K fed tokens of the
            // live history resumes from the live state at the even position <= lcp, re-feeding the rest, instead of
            // falling back to the prompt's snapshot.  The client re-tokenizes the model's own output: the forced answer
            // after the thinking budget re-renders the thinking + "</think>", and its last 1-2 tokens come back merged
            // differently (measured: lcp 1168 of 1169 fed -> 1,005 tokens re-read in 9.4 s; at 32k thinking ~85 s).
            // Safe as after a rejected DSpark draft: every cache is position-indexed, the window ring has 16 spare
            // slots, and an even position starts a fresh ratio-2 compressor group.  The spare slots also hold the rows a
            // verify pass wrote past the history (rejected drafts: up to 4 DSpark ones, up to 14 with STRATA_SERVE_LOOKUP
            // - and up to 16 across passes), so with STRATA_RING_FIX a live rewind is taken only while it stays
            // inside the ring (fix B below).  K is capped at 8.  The resumed rows are the decode's own (decode
            // routing), as on a plain resume.
            static const size_t rollback_k = (size_t) std::min(8, std::max(0, std::getenv("STRATA_SERVE_ROLLBACK") ?
                                                                              std::atoi(std::getenv("STRATA_SERVE_ROLLBACK")) : 0));
            static const bool rewind1 = std::getenv("STRATA_SERVE_REWIND1") && std::getenv("STRATA_SERVE_REWIND1")[0] == '1';
            // D11 ring fix B (AUDITS_AND_PORTS.md "Audit ring-tags" B): a row written at q evicts q - RAW from the window
            // ring, which a re-fed query needs iff it sits at or below q - (RAW - SWA) - 1.  So a live rewind to rs is
            // exact only if rs >= (the highest position written into the live rings) - ring_spare().  Otherwise the
            // request falls through to the snapshot / snapshot-stack / re-read paths, which are exact.
            const int64_t live_top = live_hi >= 0 ? live_hi : (int64_t) hist.size() - 1;
            auto ring_ok = [&](size_t rs) { return !ringfix || (int64_t) rs >= live_top - dense.ring_spare(); };
            // D11 ring fix A (AUDITS_AND_PORTS.md "Audit ring-tags" A): a live continuation at an odd position reads its
            // ratio-2 group partner (lcp - 1) from state slot (lcp - 1) % 2, which still holds a row the last verify pass
            // wrote past the history (a rejected draft, or an accepted one after a stop).  The decode loop's own rule
            // (dirty && ((P + 1) & 1)) was lost when the request ended: re-feed the token at lcp - 1 (an even position, a
            // fresh group), 1 token of cost.  live_hi >= lcp covers every case the loop's `dirty` flag did.
            const bool a_refeed = ringfix && !unparked && !hist.empty() && lcp == hist.size() && lcp < ids.size() &&
                                  (lcp & 1) && live_hi >= (int64_t) lcp;
            bool live_path = false;   // the live rings were kept (a live continuation / ROLLBACK / REWIND_GAP)
            if (reuse_override >= 0) reused = reuse_override;   // a partial unpark: positions [0, lp) are the parked one's
            else if (!unparked && !hist.empty() && lcp == hist.size() && lcp < ids.size() && (!a_refeed || ring_ok(lcp - 1))) {
                reused = (int) lcp;
                live_path = true;
                if (a_refeed) {
                    reused = (int) lcp - 1;
                    live_hi = std::max(live_hi, (int64_t) hist.size() - 1);   // rows past the shorter history stay
                    hist.resize(lcp - 1);
                    std::fprintf(stderr, "serve: live resume at %zu re-feeds 1 token (state ring held a draft row up to %lld, STRATA_RING_FIX)\n",
                                 lcp, (long long) live_hi);
                }
            }
            else if (!unparked && rollback_k > 0 && lcp < hist.size() && hist.size() - lcp <= rollback_k && lcp >= 2 &&
                     lcp < ids.size() && (snap_len <= 0 || (lcp & ~(size_t) 1) >= (size_t) snap_len) && ring_ok(lcp & ~(size_t) 1)) {
                live_path = true;
                reused = (int) (lcp & ~(size_t) 1);
                std::fprintf(stderr, "serve: rolled back %zu of %zu fed tokens (live resume at %d)\n", hist.size() - (size_t) reused,
                             hist.size(), reused);
                live_hi = std::max(live_hi, (int64_t) hist.size() - 1);   // rows past the shorter history stay in the rings
                hist.resize((size_t) reused);
            } else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len == ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);   // the same prompt again (regenerate): its end state + saved logits
                reused = snap_len;
                last_logits = snap_logits;
                regen = true;
            } else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len < ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else if (rewind1 && !unparked && snap_len > 1 && lcp + 1 == (size_t) snap_len && lcp < ids.size() && dense.restore()) {
                // STRATA_SERVE_REWIND1=1 (upstream 0a8f28f, opt-in): the request matches the snapshot's prompt except its
                // LAST token - DeepSeek's template, in thinking mode, re-renders the previous turn's "<think>" generation
                // prompt as "</think>" once a new user message follows, so every new user message re-read the whole
                // conversation.  Restore the snapshot and re-feed from that token: one token back is exact (the window
                // ring keeps RAW = SWA + 16 positions and the ratio-2 compressor state the last 2; KV / compressed /
                // indexer rows and Engram's history are positional and are overwritten).
                hist.resize(lcp);
                reused = (int) lcp;
                std::fprintf(stderr, "serve: rewound the snapshot one token (%d -> %zu)\n", snap_len, lcp);
                snap_len = -1;
            } else if (rewind_gap > 0 && !unparked && !hist.empty() && lcp > 0 && lcp < hist.size() &&
                       lcp < ids.size() && hist.size() - lcp <= (size_t) rewind_gap &&
                       ring_ok((lcp & ~(size_t) 1) < 1 ? lcp : (lcp & ~(size_t) 1))) {   // fix B: rs (below) inside the ring
                // STRATA_SERVE_REWIND_GAP: the live state continues this prompt but overshoots it by a few tokens
                // (an aborted turn's dropped assistant turn).  Keep the state and re-feed from the common prefix;
                // round down to an even position so the ratio-2 compressor's group partner is re-fed too.
                live_path = true;
                const size_t gap = hist.size() - lcp;
                size_t rs = lcp & ~(size_t) 1;
                if (rs < 1) rs = lcp;
                live_hi = std::max(live_hi, (int64_t) hist.size() - 1);   // rows past the shorter history stay in the rings
                hist.resize(rs);
                snap_len = -1;   // the end state belonged to the dropped turn
                reused = (int) rs;
                std::fprintf(stderr, "serve: rewound the live state to %zu (gap %zu <= %d, STRATA_SERVE_REWIND_GAP)\n",
                             rs, gap, rewind_gap);
            } else if (int si = (snap_stack_k > 0 && !unparked && lcp < ids.size()) ? snap_stack_pick(lcp) : -1;
                       si >= 0 && dense.load_conv(*snap_stack[(size_t) si].st) && dense.restore()) {
                // STRATA_SERVE_SNAP_STACK: an earlier snapshot of this conversation (its rings back; its rows are in place)
                const int s = snap_stack[(size_t) si].pos;
                const int r = (size_t) s <= lcp ? s : (int) lcp;   // s == lcp + 1: one token back, as REWIND1
                std::fprintf(stderr, "serve: resumed the stacked snapshot %d at %d (lcp %zu; the latest was %d)\n", s, r, lcp, snap_len);
                hist.resize((size_t) r);
                reused = r;
                snap_len = -1;
            } else if (int64_t tfl = -1, trs = (!unparked && snap_len > 0 && lcp < (size_t) snap_len && lcp < ids.size() &&
                                                 hist.size() >= (size_t) snap_len) ? tail_point(lcp & ~(size_t) 1, tfl) : -1;
                       trs > 0 && dense.snap_tail_from() >= 0 && trs >= dense.snap_tail_from() && dense.restore() &&
                       dense.snap_tail_resume(trs)) {
                // STRATA_SNAP_TAIL: the request shares the snapshot's prompt up to lcp, further back than the ring's 16
                // spare rows (an edited last message, another question about the same document, a re-sent changed last
                // turn).  restore() + the tail's window rows at the even position rs <= lcp, then re-feed from rs: the
                // compressed KV, indexer keys and Engram history are position-indexed and still hold this prompt's rows
                // below rs (written by the read that made the snapshot; the answer decode wrote only past it).
                const size_t rs = (size_t) trs;
                if (tfl >= 0) dense.set_window_floor(tfl);   // STRATA_ENC_BOUND: the floor a full read of this prompt uses
                std::fprintf(stderr, "snap-tail: resumed at %zu of %d (lcp %zu, kept %lld%s)\n", rs, snap_len, lcp,
                             (long long) dense.snap_tail_kept(), tfl >= 0 ? ", enc-bound floor" : "");
                // Engram's n-grams look back over rows below rs: a side request since an unpark may have written its own
                // tokens there (text-only: set_history would clear a picture run's dead marks, and pictures never park)
                if (hist_vis.empty()) dense.set_history(ids.data(), (int64_t) rs, 0);
                hist.resize(rs);
                reused = (int) rs;
                snap_len = -1;   // rows from rs on are re-fed: the old snapshot no longer describes them
#if defined(DS41_HAVE_DSPARK)
                dsp_next = -1;   // the drafter's window is past rs: it re-seeds from the re-fed tail
#endif
            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK), or with
                // STRATA_SERVE_CACHE_SLOTS save it whole into the cache
                if (cache_slots > 0 && !unparked) save_live();
                const int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                if (cache_slots <= 0 && park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len && rows <= park_max
                    && hist_vis.empty()   // (the park is text-only too, as the cache)
                    && dense.park(rows)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved)\n", snap_len, (long long) rows);
                }
                hist.clear();
                snap_len = -1;
                // STRATA_SHARED_PREFIX: start from the pooled restore point of this prompt's system + tools block.  load_conv
                // writes its rows [0, ps] and rings, restore() makes the rings live; the floor is the one a full read of this
                // prompt uses (sh_fl, -1 = none); Engram's n-grams look back over this prompt's tokens.
                if (sh_pick >= 0 && !unparked) {
                    auto & e = shared_pool[(size_t) sh_pick];
                    if (dense.load_conv(*e.st) && dense.restore()) {
                        dense.set_window_floor(sh_fl);
                        dense.set_history(ids.data(), sh_ps, 0);
                        hist.assign(ids.begin(), ids.begin() + sh_ps);
                        hist_vis.clear();
                        reused = (int) sh_ps;
                        e.used = ++sh_clock;
                        sh_hit = true;
#if defined(DS41_HAVE_DSPARK)
                        dsp_next = -1;   // the drafter's window holds the previous conversation's states
#endif
                        std::fprintf(stderr, "shared prefix: resumed at %lld of %zu (key %016llx, %zu pooled%s)\n", (long long) sh_ps,
                                     ids.size(), (unsigned long long) e.key, shared_pool.size(), sh_fl >= 0 ? ", enc-bound floor" : "");
                    }
                }
            }
            // D11 ring fixes: every other branch replaced the live rings (restore / load_conv / unpark) or re-reads from 0,
            // which rewrites every slot below the new history's end - nothing past the history is left in the rings
            if (!live_path) live_hi = -1;
            // STRATA_ENC_BOUND: a request that continues nothing starts without a window floor (a resumed state brought its
            // own back through restore() / load_conv() / unpark(); a live resume keeps the live one)
            if (enc_bound > 0 && reused == 0) dense.set_window_floor(-1);
            if (snap_stack_k > 0) {   // entries past the resume point are overwritten from it on; another conversation: none
                if (unparked || reused <= 0 || sh_hit) snap_stack.clear();
                else snap_stack.erase(std::remove_if(snap_stack.begin(), snap_stack.end(),
                                                     [&](const StackSnap & e) { return e.pos > reused; }), snap_stack.end());
            }
            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\n", lcp,
                         park_hist.size(), lp_show, hist.size(), snap_len, ids.size(), reused);
            if (reused > 0) { std::printf("RESUME %d\n", reused); std::fflush(stdout); }
            const int total = (int) ids.size();
            // a long prompt runs layer-major without the VRAM cache (its pass / chunk buffers need that VRAM) and with
            // the prompt's routing (no decode bias).  A short one (<= step_max new tokens: a chat / agent turn's tool
            // result) runs on the decode side instead - small dense passes, the cache and the decode biases: the
            // layer-major path streams nearly every expert of every layer whatever the batch size (~18 s for 40 tokens)
            const int delta = total - reused;
            int sp_tail = 0;   // DSpark: the last new prompt tokens, fed by pass_v (seeds the window)
#if defined(DS41_HAVE_DSPARK)
            if (dsp_path) {
                // fix A's 1-token re-feed (reused == dsp_next - 1) keeps the drafter: its step at P rewrites the same
                // slot and tag
                const bool dsp_cont = reused > 0 && (reused == dsp_next || (a_refeed && live_path && reused + 1 == dsp_next));
                if (!dsp_cont && dsp) dsp->reset();
                dsp_next = -1;
                sp_tail = std::min(128, delta);
            }
#endif
            int dh = delta - sp_tail;   // the part the passes / batch feed
            bool by_step = defer_vram && dh >= 1 && dh <= step_max;
            // STRATA_SERVE_BATCH_TAIL (default on, =0 the old way): a turn that runs as a layer-major batch anyway takes
            // its DSpark tail INTO the batch and the seed rows are read off the batch's state (bcap_vh) - the tail as
            // 16-token exact passes was half of every turn (s11 2026-10-15: +273 tokens 33.5 s = batch 17.1 + tail 15.8).
            // DSpark drafts are verified, so only the seed's numerics (batch vs pass kernels) change, never the output.
            static const bool batch_tail = !(std::getenv("STRATA_SERVE_BATCH_TAIL") && std::getenv("STRATA_SERVE_BATCH_TAIL")[0] == '0');
            int bcap = 0;
            if (batch_tail && sp_tail > 0 && !by_step && dh >= 2 && PB > 1) { bcap = sp_tail; dh = delta; }
            // STRATA_SERVE_BATCH_MIN (tokens; default 128, 0 = off): a turn of at least this many new tokens runs as a
            // layer-major batch even when the passes would take it (every turn up to 128 + STEP_MAX = 224 tokens did:
            // the DSpark tail + the step path).  The batch's floor fell with the routed-only reads.  s11, 512K, prompt s
            // (+ 32-token decode) passes / batch: +30 5.0 (1.7) / 6.0 (2.0), +60 6.6 (1.7) / 7.0 (2.0), +90 9.7 (1.0) /
            // 7.7 (1.7), +150 17.6 (1.4) / 9.8 (1.6).  Decode right after a batch starts colder (the VRAM cache is
            // re-seeded), hence the margin above the ~80-token crossover.
            static const int batch_min = std::getenv("STRATA_SERVE_BATCH_MIN") ? std::atoi(std::getenv("STRATA_SERVE_BATCH_MIN")) : 128;
            bool force_batch = false;
            if (batch_min > 0 && batch_tail && sp_tail > 0 && bcap == 0 && delta >= batch_min && PB > 1 && delta >= 2) {
                bcap = sp_tail; dh = delta; force_batch = true; by_step = false;
            }
            (void) force_batch;
            // STRATA_SHARED_PREFIX: a resumed read runs as the layer-major batch a full read of this prompt (>= the pool's
            // minimum, so a batch) runs - the decode-side passes would route with the decode bias
            if (sh_hit && by_step && PB > 1 && delta >= 2) {
                by_step = false;
                if (batch_tail && sp_tail > 0 && bcap == 0) { bcap = sp_tail; dh = delta; }
            }
            if (by_step && !cache_open) {
                if (!open_cache()) { std::printf("ERR VRAM seed failed: %s\n", err.c_str()); std::fflush(stdout); continue; }
                cache_open = true;
            }
            // exact passes (STRATA_SERVE_PASS_BIAS=0): the decode-side prompt tokens route with the model's own
            // selection (no route bias / file penalty) - only the batch shape differs from the exact batch path
            const bool exact_pass = by_step && !pass_bias;
            if (exact_pass && bias_set) {
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            // STRATA_SERVE_KEEP_CACHE=1 (s6, opt-in until measured): a batch turn of at most STRATA_SERVE_KEEP_MAX new
            // tokens (default 4096) gives back only the VRAM its batch needs (the calibrated need(B) below) - the cache's
            // lowest-ranked tail - instead of the whole cache, so the hot experts serve the batch from VRAM (no PCIe) and
            // the decode after it skips the re-seed (regrow_cache refills just the tail).  The prompt still routes exactly.
            // s6 A/B: it never engages with the default --vram-lru (the LRU reorders slots, so "the tail" is not the
            // coldest experts and a batch-time admission could duplicate a saved one) - it needs --no-vram-lru for now.
            bool kept = false;
            if (Ds41MoeTier::keep_cache_mode() && !by_step && dh >= 2 && !(defer_vram && cache_open))
                std::fprintf(stderr, "serve: keep-cache: not applicable (deferred VRAM %d, cache open %d)\n", (int) defer_vram,
                             (int) cache_open);
            if (!by_step && defer_vram && cache_open && dh >= 2) {
                static const int keep_max = std::getenv("STRATA_SERVE_KEEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_KEEP_MAX")) : 4096;
                if (Ds41MoeTier::keep_cache_mode() && dh <= keep_max && !mc.cpu_only) {
                    // STRATA_KEEP_FIX_GIB=x: the fixed part of need(B) for a batch that fits one pass (up to 512 tokens); default 4.3, the
                    // value calibrated on long batches.  A short turn gives up about 520 slots of 1,335 for it (a measured log).  Above 512
                    // tokens the 4.3 stays: the batch width below is sized from what this leaves free.
                    static const double keep_fix = std::getenv("STRATA_KEEP_FIX_GIB") ? std::atof(std::getenv("STRATA_KEEP_FIX_GIB")) : 4.3;
                    const double need = ((dh <= 512 ? keep_fix : 4.3) * 1073741824.0 + 0.95 * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.prompt_margin() * 1073741824.0;   // 0.95: the prompt path's margin
                    std::string kerr;
                    const int64_t g = tier.shrink_cache_for((int64_t) need, kerr);
                    if (g >= 0) {
                        kept = true;
                        std::fprintf(stderr, "serve: cache kept for a %d-token batch (%lld tail slots out, %lld experts stay)\n",
                                     dh, (long long) g, (long long) tier.resident());
                    } else if (!kerr.empty()) {
                        std::fprintf(stderr, "serve: keep-cache: %s - releasing the cache\n", kerr.c_str());
                    }
                }
                if (!kept) {
                    tier.release_cache();
                    cache_open = false;
#if defined(DS41_HAVE_DSPARK)
                    // STRATA_SERVE_DSPARK_EVICT=N (s6, opt-in until measured; 0 = off): a batch of >= N new tokens also
                    // frees DSpark's ~2.9 GB of weights for its duration - the prompt batch is sized from free VRAM
                    // (s6: 8731 columns before DSpark loads, 5434 after), so a long prompt takes fewer expert sweeps.
                    // open_cache() re-creates it for the tail (~1-4 s); the window is re-seeded by the tail as always.
                    static const int dsp_evict = std::getenv("STRATA_SERVE_DSPARK_EVICT") ? std::atoi(std::getenv("STRATA_SERVE_DSPARK_EVICT")) : 0;
                    if (dsp && dsp_evict > 0 && dh >= dsp_evict) {
                        // STRATA_DSPARK_PARK (A7; default 0 until measured): 1 = keep the weights in PAGEABLE host RAM, 2 = in pinned host
                        // RAM (cudaMallocHost; ggml falls back to pageable when refused), 0 (default) = the old free + GGUF re-parse
                        // (1.1-1.4 s a reload with the file in the OS cache, 2.6-4.9 s cold).  The copy is made at the first
                        // eviction (one D2H of weight_bytes(), ~2.9 GB) and kept for the run, only when the OS reports at
                        // least that + STRATA_DSPARK_PARK_FREE_GIB (default 8) GiB available; else the old path.  Pageable is
                        // the mode 1 picks: the arena's auto pin leaves ~4 GiB of the WDDM pinned budget, which the read-ahead
                        // slabs (2 x 768 MiB, pinned at the first wide prompt - this one) and the lazy pinned buffers need.
                        // Exact: unpark() re-creates the same buffers in init()'s order with the same bytes and a cleared
                        // state (Ds41DSpark::unpark), so the free VRAM the cache seeds from and every draft are unchanged.
                        static const int dsp_pin = ds41_env("STRATA_DSPARK_PARK", "STRATA_P3_DSPARKPIN") ? std::atoi(ds41_env("STRATA_DSPARK_PARK", "STRATA_P3_DSPARKPIN")) : 0;
                        static const double dsp_pin_free = ds41_env("STRATA_DSPARK_PARK_FREE_GIB", "STRATA_P3_DSPARKPIN_FREE_GIB") ? std::atof(ds41_env("STRATA_DSPARK_PARK_FREE_GIB", "STRATA_P3_DSPARKPIN_FREE_GIB")) : 8.0;
                        bool parked = false;
                        if (dsp_pin > 0) {
                            const bool first = dsp->host_bytes() == 0;
                            const double tp = now_ms();
                            std::string perr;
                            if (dsp->park(dsp_pin >= 2, (size_t) (std::max(0.0, dsp_pin_free) * 1073741824.0), perr)) {
                                parked = true;
                                if (first)
                                    std::fprintf(stderr, "serve: DSpark weights kept in %s host RAM for the run (%.2f GB, %.2f s; "
                                                         "STRATA_DSPARK_PARK=0 turns it off)\n", dsp->host_pinned() ? "pinned" : "pageable",
                                                 dsp->host_bytes() / 1e9, (now_ms() - tp) / 1000.0);
                                dsp_parked = std::move(dsp);
                            } else {
                                std::fprintf(stderr, "serve: STRATA_DSPARK_PARK: %s - the drafter reloads from the file\n", perr.c_str());
                            }
                        }
                        if (!parked) dsp.reset();
                        std::fprintf(stderr, "serve: DSpark weights freed for a %d-token batch%s\n", dh, parked ? " (kept in host RAM)" : "");
                    }
#endif
                }
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            each_tier([](Ds41MoeTier & t) { t.rebalance_begin_prompt(); t.reset_stats(); });
            set_run_k(top_k);   // STRATA_EXPERTS_USED_DECODE: the prompt routes top_k (the decode below switches)
            if (const char* rl = std::getenv("STRATA_ROUTE_LOG")) {   // a turn marker for the route log (see ds41_moe.cpp)
                if (FILE* f = std::fopen(rl, "a")) { std::fprintf(f, "# turn %d new %d %s\n", n_req, dh, by_step ? "passes" : "batch"); std::fclose(f); }
            }
            const double tp0 = now_ms();
            const long sk_seq_req = dense.sink_seq();   // STRATA_SINK_TRACE: a logits capture made for this prompt
            auto pp = [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            };
            prefill_cancelled = false;
            auto feed_steps = [&]() -> bool {   // the short-turn path: passes of up to pass_max tokens (a lone token: a step)
                // STRATA_PASS_SPLIT_FIXED=1: the tiers know these passes are a prompt's (gpu_run_n_ex)
                struct PromptPassMark {
                    std::function<void(bool)> set;
                    explicit PromptPassMark(std::function<void(bool)> f) : set(std::move(f)) { set(true); }
                    ~PromptPassMark() { set(false); }
                } prompt_mark([&](bool on) { each_tier([on](Ds41MoeTier & t) { t.set_prompt_pass(on); }); });
                int last_rows = 1;
                for (int i = 0; i < dh; ) {
                    if (stop_prefill()) return false;
                    const int m = std::min(pass_max, dh - i);
                    if (!(m == 1 ? step(ids[(size_t) (reused + i)], reused + i) : pass_m(ids.data() + reused + i, m, reused + i))) return false;
                    if (!capture_tail(delta, i, m, 0)) return false;   // this pass's rows, before the next pass replaces them
                    const int before = i;
                    i += m;
                    last_rows = m;
                    if (i / 64 != before / 64 && i < dh) pp(i);
                }
                if (!dense.logits(last_rows - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                pp(dh);
                return true;
            };
            static const int dump_tail = std::getenv("STRATA_SERVE_DUMP_TAIL") ? std::atoi(std::getenv("STRATA_SERVE_DUMP_TAIL")) : 0;
            tail_keep = dump_tail > 0 ? std::min(dump_tail, delta) : 0;
            tail_logits.clear();
            // VRAM-aware batch width for this turn (feed() halves it again if even this does not fit): the batch's
            // state columns + graph are ~0.48 MiB a column, and the card is shared with whatever else the desktop
            // runs.  This is what lets a big --prompt-batch stay OPT-IN: the engine takes what is actually free.
            pb_cap = PB;
            if (!by_step && dh >= 2 && PB > 1 && !mc.cpu_only) {
                size_t frb = 0, totb = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &frb, &totb);
                frb = ds41_vram_quant(frb, "serve: prompt batch width");   // the batch shape (STRATA_VRAM_QUANT_MIB)
                // The layer-major batch's own VRAM, CALIBRATED on four traced points (STRATA_VRAM_TRACE=1, s13
                // run9/run13/run14, a 4 GiB foreign process): free 9.28 GiB with B=8192 failed, free 9.26 with
                // B=5210 passed, free 5.92 with B=2411 failed, a free card with B=8192 passed - all four fit
                //     need(B) ~= 4.3 GiB + 0.95 MiB * B
                // 4.3 GiB fixed = the tier's two 1 GiB stage halves + a ~2.19 GiB graph buffer that is the SAME
                // SIZE at B=2411 and B=8192 (so it is not the piece buffers; identity unconfirmed - see the
                // residuals in V41_START_HERE).  0.95 MiB a column = columns 0.32 + graph 0.17 + part buffers 0.41,
                // measured directly.  Per-tensor arithmetic undercounted the tier 2x and let a 2.19 GiB allocation
                // fail while cudaMemGetInfo still said 6.83 GiB free.  A free card (>= 13 GiB) still gets its full PB.
                // run20 (2026-10-16) fixes the GRAPH term: the pass-graph buffer is B-INDEPENDENT - the B=16384 leg
                // OOM'd on "allocating 1424.31 MiB", the very size this comment records for 8192, so "graph 0.17 MiB
                // a column" is a mis-derivation (1424/8192) and the decomposition is really ~5.7 GiB fixed + 0.78 a
                // column.  Same line, same fit - but it matters: a wide batch's per-column STATE is what starves the
                // fixed graph buffer (16384 = 5.14 GiB of state, 1.03 GiB short), not the graph itself.
                // 2026-10-20: the GRAPH also grows with the CONTEXT.  The indexer's score / mask tensors are
                // [visible compressed rows x queries] (build_attn), and the visible rows are a power of two: a
                // ratio-1 layer past 2^19 rows builds a 9.06 GiB span graph untiled (5.56 with build_attn's indexer
                // tile) instead of ~1.2, and it is built first by
                // the prompt's LAST batch (under enc-skip the only one that runs layers 21-39) - long after this width
                // took the columns.  The line above budgets the graph as its own decomposition says, 2.19 GiB + 0.17
                // MiB a column (4.2 GiB at 12288 columns): room for some growth over the 1.16 GiB a short context
                // builds, none for 9 GiB.  So the width must also fit the batch's memory WITHOUT that allowance plus
                // the graph this prompt's last sub-batch will really build, as the allocator plans it:
                //     need(B) = max(4.3 GiB + 0.95 MiB * B,  2.5 GiB + 0.78 MiB * B + graph)
                // The second term is a 5090 trace (2026-10-20, 12288 columns, 18.74 GiB free at the cap): a 512K
                // prompt's batches took 13.01 GiB with a 1.16 GiB graph - 11.85 GiB besides it = 2.5 GiB + run20's
                // 0.78 MiB a column - and its last batch's 4.56 GiB graph left 2.33 GiB free, as the term says.  Up to
                // the allowance (here every context up to 512K at the full width) the cap is the one it was.
                // STRATA_PB_GRAPH=0: the line alone (diagnostics).
                static const bool pb_graph = !(std::getenv("STRATA_PB_GRAPH") && std::atoi(std::getenv("STRATA_PB_GRAPH")) == 0);
                const double g_need = pb_graph ? (double) dense.span_graph_bytes(total, SB) : 0.0;   // memoized per row count
                // STRATA_WIDE_PROMPT with the cache released and every class on MMQ: a column is the state's ~0.24
                // MiB + the tier's parts ~0.12 (no MMVQ scratch: nothing is VRAM-resident).  Keisawaakira's 5090 trace
                // (60K one-shot at 30720 columns: state 7.30 + tier 7.09 GiB, + a 1.5 GiB lazy prestage) = 4.75 GiB +
                // 0.352 MiB a column, so the graph term's fixed part is the line's 4.3 there, not 2.5; their constants
                // are 0.44 / 0.38 with margin.  NOT recalibrated on the 3090 yet (STRATA_VRAM_TRACE=1).  A kept cache
                // (its resident experts run on MMVQ) or a class off MMQ keeps the calibrated constants.
                const bool wide_cols = wide_for(dh) && !kept && tier.chunk_all_mmq();   // feed(.., dh, ..) decides the same
                const double mib_line = wide_cols ? 0.44 : 0.95, mib_g = wide_cols ? 0.38 : 0.78, gib_g = wide_cols ? 4.3 : 2.5;
                const double room = (double) frb - a.prompt_margin() * 1073741824.0 - 4.3 * 1073741824.0;
                const double room_g = (double) frb - a.prompt_margin() * 1073741824.0 - gib_g * 1073741824.0 - g_need;
                const int fit_line = (int) std::max<int64_t>(2, (int64_t) (room / (mib_line * 1048576.0)));
                const int fit_g = pb_graph ? (int) std::max<int64_t>(2, (int64_t) (room_g / (mib_g * 1048576.0))) : fit_line;
                const int fit = std::min(fit_line, fit_g);
                pb_cap = std::min(PB, std::max(SB, fit));
                if (vram_trace) std::fprintf(stderr, "vram trace: cap free %.2f GiB room %.2f GiB fit %d, graph %.2f GiB fit %d -> pb_cap %d\n",
                                             frb / 1073741824.0, room / 1073741824.0, fit_line, g_need / 1073741824.0, fit_g, pb_cap);
                // STRATA_PROMPT_SWEEP_FIT (default 0 = the 5090 constants above; 1 = this, which the launcher
                // turns on with its STRATA_PROMPT_SWEEP_PB raise): a WIDE batch's width from the bytes this build
                // allocates, not fitted constants.  Counted, nothing measured:
                //     need(B) = state_bytes(B) (Ds41Dense::state_bytes, the column tensors alloc_cols declares, hap in
                //               x_state) + chunk_wide_bytes(B) (c_parts B*K rows + ids / weights) + chunk_mmq_bytes(B)
                //               (the MMQ scratch, 16384 rows up to B = 32768)
                //     fixed   = max(margin, keep) + the two stage halves at their requested size + this context's span
                //               graph (span_graph_bytes) + STRATA_PROMPT_SWEEP_SLACK_GIB (default 1.0: ggml-cuda's pool, the
                //               finish graphs, cuBLAS workspace, allocation alignment - the part not counted)
                //     keep    = STRATA_PROMPT_SWEEP_PRESTAGE (default 1): chunk_keep_bytes - the free VRAM the layer prestage
                //               region (3 GiB over a 3 GiB keep) and the PTQ1_0 transcode copies (1 over 2) need, else
                //               they are skipped for the whole prompt; 0 = the margin only (widest; the prestage may go
                //               off - the tier then says 'chunk prestage: off for this prompt')
                // On the 3090 (D 4096, HC 4, K 6) that is ~0.35 MiB a column against the 0.44 above.  It never goes
                // below the constants' width (that width runs exactly as the default runs it), and it is not used while the
                // STRATA_CHUNK_CPU pool can take calls: there the width alone moves the numerics (measured: 11,560 vs
                // 8192 columns: top-1 93.5%).  Then STRATA_PROMPT_SWEEP_PROBE (default 1) cudaMallocs those very blocks
                // (and frees them) before the batch: if they do not fit, the width steps down by an eighth (not below
                // the constants' width) until they do - a wrong count costs columns, never an out-of-memory abort in
                // the tier (its cudaMallocs abort).  Only the width changes: wide batches without the pool were
                // bit-identical across widths (16384 vs 8192); 24-32K is the md5 gate before any default.
                static const bool p3_fit = [] { const char * e = ds41_env("STRATA_PROMPT_SWEEP_FIT", "STRATA_P3_SWEEP_FIT"); return e && *e && e[0] != '0'; }();
                const bool p3_on = wide_cols && p3_fit && !tier.chunk_cpu_wants(1);
                if (wide_cols && p3_fit && !p3_on) {
                    static bool said_pool = false;
                    if (!said_pool) {
                        said_pool = true;
                        std::fprintf(stderr, "serve: wide prompt budget: off - the STRATA_CHUNK_CPU pool can take prompt calls "
                                             "(the width would move the numerics); the 5090 constants' width\n");
                    }
                }
                if (p3_on) {
                    static const double p3_slack = 1073741824.0 * [] {
                        const char * e = ds41_env("STRATA_PROMPT_SWEEP_SLACK_GIB", "STRATA_P3_SWEEP_SLACK_GIB");
                        return e && *e ? std::max(0.0, std::atof(e)) : 1.0;
                    }();
                    static const bool p3_probe = [] { const char * e = ds41_env("STRATA_PROMPT_SWEEP_PROBE", "STRATA_P3_SWEEP_PROBE"); return !e || !*e || e[0] != '0'; }();
                    static const bool p3_pre = [] { const char * e = ds41_env("STRATA_PROMPT_SWEEP_PRESTAGE", "STRATA_P3_SWEEP_PRESTAGE"); return !e || !*e || e[0] != '0'; }();
                    const double GiB = 1073741824.0, MiB = 1048576.0;
                    const double g3 = g_need > 0 ? g_need : (double) dense.span_graph_bytes(total, SB);
                    const double stage = (double) tier.chunk_stage_bytes(), margin = a.prompt_margin() * GiB;
                    const double keep = p3_pre ? (double) tier.chunk_keep_bytes(PB) : 0.0;
                    const double resv = std::max(margin, keep);
                    auto need = [&](int b) {
                        return (double) dense.state_bytes(b, true) + (double) tier.chunk_wide_bytes(b) + (double) tier.chunk_mmq_bytes(b);
                    };
                    const double avail = (double) frb - resv - stage - g3 - p3_slack;
                    const int b1 = 16384, b2 = 32768;   // the MMQ scratch is flat between them: need() is linear there
                    const double col = (need(b2) - need(b1)) / (double) (b2 - b1);
                    const double col_state = ((double) dense.state_bytes(b2, true) - (double) dense.state_bytes(b1, true)) / (double) (b2 - b1);
                    const double fix = need(b1) - col * (double) b1;
                    const int fit_old = pb_cap;   // the constants' width (above): the floor
                    int fit3 = col > 0 ? (int) std::max(2.0, std::min(1e9, (avail - fix) / col)) : PB;
                    fit3 = std::min(PB, std::max(SB, fit3));
                    while (fit3 > SB && need(fit3) > avail) fit3 = std::max(SB, fit3 - SB);   // past 32768 the MMQ rows grow
                    pb_cap = std::max(fit_old, fit3);
                    std::fprintf(stderr, "serve: wide prompt budget: %.2f GiB free - kept %.2f (margin %.2f, prestage/transcode %.2f) - "
                                         "stage %.2f - graph %.2f - slack %.2f = %.2f GiB for %.0f MiB of MMQ scratch + %.3f MiB a column "
                                         "(state %.3f + tier %.3f) -> %d columns -> prompt batch %d (of %d; floor: the 5090 constants' %d)\n",
                                 frb / GiB, resv / GiB, margin / GiB, keep / GiB, stage / GiB, g3 / GiB, p3_slack / GiB, avail / GiB,
                                 fix / MiB, col / MiB, col_state / MiB, (col - col_state) / MiB, fit3, pb_cap, PB, fit_old);
                    if (p3_probe && pb_cap > fit_old) {
                        std::string perr;
                        const int first = pb_cap;
                        int tries = 0;
                        for (;;) {
                            const size_t st = tier.chunk_stage_bytes() / 2;
                            const std::vector<size_t> blocks = { (size_t) resv, dense.state_bytes(pb_cap, true), tier.chunk_wide_bytes(pb_cap),
                                                                 st, st, tier.chunk_mmq_bytes(pb_cap), (size_t) g3, (size_t) p3_slack };
                            ++tries;
                            std::string e1;
                            if (tier.vram_probe(blocks, e1)) break;
                            perr = e1;
                            if (pb_cap <= fit_old) break;
                            pb_cap = std::max(fit_old, (pb_cap - std::max(SB, pb_cap / 8)) / SB * SB);
                        }
                        if (pb_cap != first || vram_trace)
                            std::fprintf(stderr, "serve: wide prompt probe: %d -> %d columns (%d tries%s%s)\n", first, pb_cap, tries,
                                         perr.empty() ? "" : "; last miss: ", perr.c_str());
                    }
                } else if (fit_g < fit_line && fit_g < PB)
                    std::fprintf(stderr, "serve: a %d-token context's pass graph needs %.2f GiB -> prompt batch %d (of %d)\n",
                                 total, g_need / 1073741824.0, pb_cap, PB);
                else if (pb_cap < PB) std::fprintf(stderr, "serve: VRAM %.2f GiB free -> prompt batch %d (of %d)\n",
                                                   frb / 1073741824.0, pb_cap, PB);
                ds41_vram_decision("serve: prompt batch width", "prompt batch %d (of %d) for a %d-token turn", pb_cap, PB, dh);
            }
            const double tph0 = now_ms();
            const double fd0[3] = { fd_dense, fd_tier, fd_logits };   // this request's batch split (stderr)
            bcap_n = bcap; bcap_got = 0;
            if (bcap > 0) bcap_vh.assign((size_t) bcap * 3 * n_embd, 0.0f);
            dense.reset_span_width();   // a sub-batch narrowed by the last prompt's tight card starts wide again
            // STRATA_SHARED_PREFIX: this read (a layer-major batch) passes the end of its system + tools block, rounded down to a
            // sub-batch boundary, and the pool does not hold that prefix yet: capture the rings there as the read goes by
            int64_t sh_cap = -1;
            if (sh_on && !is_vision && !sh_hit && req_sys_end > 0 && !by_step && PB > 1) {
                const int64_t ps = std::min<int64_t>((int64_t) req_sys_end, total) / SB * SB;
                bool have = false;
                for (auto & e : shared_pool)
                    if ((int64_t) e.toks.size() == ps && std::equal(e.toks.begin(), e.toks.end(), ids.begin())) { have = true; break; }
                if (ps >= sh_min && ps > reused && ps < total && !have) {
                    sh_cap = ps;
                    dense.arm_prefix_capture(ps);
                }
            }
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            bcap_n = 0;
            if (sh_cap > 0) {   // STRATA_SHARED_PREFIX: rows [0, sh_cap] are this read's and still in place - pool them
                std::shared_ptr<Ds41Dense::SavedConv> st = ok ? dense.take_prefix_capture(sh_cap, !enc_skip) : nullptr;
                if (!ok) dense.arm_prefix_capture(-1);
                if (st) {
                    shared_pool.push_back({ tok_hash(ids.data(), (size_t) sh_cap), std::vector<int>(ids.begin(), ids.begin() + sh_cap), st, ++sh_clock });
                    const size_t cap = (size_t) (sh_gib * 1073741824.0);
                    auto bytes = [&] { size_t b = 0; for (const auto & e : shared_pool) b += Ds41Dense::saved_bytes(*e.st); return b; };
                    while (shared_pool.size() > 1 && ((int) shared_pool.size() > sh_slots || bytes() > cap)) {
                        auto lru = std::min_element(shared_pool.begin(), shared_pool.end(), [](const SharedSnap & x, const SharedSnap & y) { return x.used < y.used; });
                        std::fprintf(stderr, "shared prefix: dropped the %zu-token prefix (least recently used)\n", lru->toks.size());
                        shared_pool.erase(lru);
                    }
                    if (bytes() > cap) shared_pool.clear();   // one prefix past the whole budget: not kept
                    std::fprintf(stderr, "shared prefix: saved %lld of %d (system + tools end at %d; %zu pooled, %.2f GiB)\n",
                                 (long long) sh_cap, total, req_sys_end, shared_pool.size(), (double) bytes() / 1073741824.0);
                } else if (ok) {
                    std::fprintf(stderr, "shared prefix: %lld not saved (no sub-batch of this read ended there)\n", (long long) sh_cap);
                }
            }
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
                if (keep_lazy > 0) {
                    regrow_due = true;
                    bias_now = true;
                    apply_bias();
                } else if (tier.regrow_cache(kerr) < 0) {
                    std::fprintf(stderr, "serve: keep-cache regrow: %s - re-seeding\n", kerr.c_str());
                    tier.release_cache();
                    cache_open = false;
                } else {
                    bias_now = true;
                    apply_bias();
                }
            }
            double ph_feed = now_ms() - tph0, ph_open = 0, ph_tail = 0;   // the request's prompt phases (stderr)
            int sp_t = -1;   // DSpark: the first decode token (sampled from the last prompt row), its drafts in sp_ids
            int sp_ids[5] = {0}, sp_tmp[5];
            float sp_conf[5];
#if defined(DS41_HAVE_DSPARK)
            // STRATA_DSPARK_SAMPLE=1 (opt-in, S1, 2026-10-31): at temperature > 0 the drafter SAMPLES its drafts from its
            // own distribution q (step_sampled) and the verifier accepts with min(1, p / q) (spec_accept_q) - the output
            // is still distributed as p, but more drafts survive a flat p than the argmax draft's p(d).  Lookup drafts
            // stay deterministic (spec_accept).  STRATA_DSPARK_SAMPLE_T=f: q's temperature = f x the request's (default 1).
            static const bool dsp_sample = [] { const char * e = std::getenv("STRATA_DSPARK_SAMPLE"); return e && e[0] && e[0] != '0'; }();
            static const float dsp_qt = std::getenv("STRATA_DSPARK_SAMPLE_T") ? std::max(0.05f, (float) std::atof(std::getenv("STRATA_DSPARK_SAMPLE_T"))) : 1.0f;
            bool sp_sampled = false;   // sp_ids came from step_sampled: their q rows are at sp_q
            std::vector<float> sp_cdf;
            const float * sp_q = nullptr;   // the q rows: the drafter's pinned staging (STRATA_PINNED_STAGE) or sp_cdf
            auto dstep = [&](const float * mh, const int * pos, int m, int bonus, int * dids, float * dconf) -> bool {
                if (bonus < 0) return dsp->step(mh, pos, m, bonus, dids, dconf);
                sp_sampled = false;
                if (!dsp_sample || temp <= 0.0f) return dsp->step(mh, pos, m, bonus, dids, dconf);
                float u[5];
                std::uniform_real_distribution<float> U(0.0f, 1.0f);
                for (float & x : u) { x = U(rng); if (!(x < 1.0f)) x = std::nextafter(1.0f, 0.0f); }
                const float * staged = dsp->staged_cdf();
                if (!staged) sp_cdf.resize((size_t) 5 * (size_t) dsp->vocab());
                if (!dsp->step_sampled(mh, pos, m, bonus, dids, dconf, u, 1.0f / (temp * dsp_qt), staged ? nullptr : sp_cdf.data())) return false;
                sp_q = staged ? staged : sp_cdf.data();
                sp_sampled = true;
                return true;
            };
            if (ok && sp_tail > 0) {   // the tail through pass_v (chunks of <= 16), the cache open; routing as the passes
                const double to0 = now_ms();
                if (defer_vram && !cache_open) {
                    if (!open_cache()) { ferr = "VRAM seed failed: " + err; ok = false; }
                    else cache_open = true;
                }
                ph_open = now_ms() - to0;
                if (ok) { bias_now = pass_bias; apply_bias(); }
                const double tt0 = now_ms();
                std::vector<int> spos;
                if (bcap > 0) {   // the seed rows came off the batch (bcap_vh, its last bcap_got tokens): DSpark steps only
                    const int c0s = total - bcap_got;
                    // upstream be8f039: the capture wrote the final batch's rows at the END of bcap_vh (row bcap_n - cn
                    // + t); a final batch shorter than the tail fills only the last bcap_got of its bcap rows, so read
                    // from there, not from row 0 (a short final batch seeded the drafter with zero rows: exact output,
                    // lower acceptance).  bcap: this request's tail width (bcap_n is cleared after the feed)
                    const size_t vh0 = (size_t) (bcap - bcap_got);
                    for (int i0 = c0s; ok && i0 < total; i0 += 16) {
                        if (stop_prefill()) { ok = false; break; }
                        const int m = std::min(16, total - i0);
                        spos.resize((size_t) m);
                        for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                        const bool last = i0 + m == total;
                        if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);   // the batch's last row
                        if (i0 == c0s) cur_seed_n = 0;
                        cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                        std::copy_n(bcap_vh.data() + (vh0 + (size_t) (i0 - c0s)) * 3 * n_embd, (size_t) m * 3 * n_embd,
                                    cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                        cur_seed_n += m;
                        // dsp null: the drafter failed to load at this cache open (off for the run, see open_cache)
                        if (dsp && !dstep(bcap_vh.data() + (vh0 + (size_t) (i0 - c0s)) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                          last ? sp_ids : sp_tmp, sp_conf)) {
                            ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                        }
                    }
                    if (dsp && bcap_got < 1) { ferr = "DSpark batch seed captured nothing"; ok = false; }
                }
                // STRATA_PASS_SPLIT_FIXED=1: these tail passes are the prompt's too (a short serve prompt is ALL tail)
                if (bcap == 0) each_tier([](Ds41MoeTier & t) { t.set_prompt_pass(true); });
                for (int i0 = total - sp_tail; ok && bcap == 0 && i0 < total; i0 += 16) {
                        if (stop_prefill()) { ok = false; break; }
                    const int m = std::min(16, total - i0);
                    if (!pass_v(ids.data() + i0, m, i0)) { ok = false; break; }
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) {
                        std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                        sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    }
                    if (i0 == total - sp_tail) cur_seed_n = 0;
                    cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                    std::copy_n(vh.data(), (size_t) m * 3 * n_embd, cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                    cur_seed_n += m;
                    if (dsp && !dstep(vh.data(), spos.data(), m, last ? sp_t : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                    }
                }
                each_tier([](Ds41MoeTier & t) { t.set_prompt_pass(false); });
                if (ok) { bias_now = true; apply_bias(); }
                ph_tail = now_ms() - tt0;
                pp(delta);
            } else if (ok && regen && dsp && snap_seed_n > 0) {   // a repeated prompt: re-seed DSpark from the snapshot's rows
                const double tt0 = now_ms();
                dsp->reset();
                std::vector<int> spos;
                const int s0 = total - snap_seed_n;
                for (int i0 = s0; i0 < total; i0 += 16) {
                    const int m = std::min(16, total - i0);
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    if (!dstep(snap_seed.data() + (size_t) (i0 - s0) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                               last ? sp_ids : sp_tmp, sp_conf)) {
                        std::fprintf(stderr, "serve: DSpark re-seed: %s (decoding without drafts)\n", dsp->last_error().c_str());
                        dsp->reset();
                        sp_t = -1;
                        break;
                    }
                }
                ph_tail = now_ms() - tt0;
                std::fprintf(stderr, "serve: repeated prompt - DSpark re-seeded from %d saved rows (%.2f s)\n", snap_seed_n, ph_tail / 1000.0);
            }
#endif
            dense.prof_report("turn");
            std::fprintf(stderr, "serve: prompt phases: %d new = %d %s (%.2f s: dense %.2f, experts %.2f, logits %.2f) + DSpark tail %d%s "
                         "(cache open %.2f s, seed %.2f s)\n", delta, dh, by_step ? "passes" : "batch", ph_feed / 1000.0,
                         (fd_dense - fd0[0]) / 1000.0, (fd_tier - fd0[1]) / 1000.0, (fd_logits - fd0[2]) / 1000.0,
                         bcap > 0 ? bcap_got : sp_tail, bcap > 0 ? " from the batch" : " as passes", ph_open / 1000.0, ph_tail / 1000.0);
            if (tail_keep > 0 && ok) {   // the quality scorer's rows: req%03d.tail.f32, [tail_keep x n_vocab], last prompt row last
                if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {
                    char nb[40];
                    std::snprintf(nb, sizeof(nb), "/req%03d.tail.f32", n_req);
                    std::ofstream f(std::string(dd) + nb, std::ios::binary);
                    f.write((const char *) tail_logits.data(), (std::streamsize) (tail_logits.size() * 4));
                }
            }
            tail_keep = 0;
            if (!ok) {
                if (prefill_cancelled) {
                    // No partial prompt can be reused as a valid conversation.
                    snap_seed_n = cur_seed_n = 0;
                    // feed returned before its ordinary elastic-column/chunk cleanup.
                    dense.release_pass();
                    const bool cleaned = !dc.elastic_cols || dense.set_columns(0);
                    if (rdev1) rdev1 = dense.routed_dev();
                    each_tier([](Ds41MoeTier & t) { t.release_chunk(); t.rebalance_tick(true); });
                    if (cleaned) std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d 0 0 0 0 0 0\n",
                                             total, now_ms() - tp0, reused);
                    else std::printf("ERR cancelled prefill cleanup: %s\n", dense.last_error().c_str());
                } else {
                    ds41_end_if_gpu_fault();   // a GPU fault ends the engine now, with this request (see the decode below)
                    // the same cleanup as a cancelled read: feed() returned before its own (wide state columns, the batch's
                    // pass and chunk buffers), and the next decode would seed its VRAM cache around them
                    dense.release_pass();
                    if (dc.elastic_cols && !dense.set_columns(0))
                        std::fprintf(stderr, "serve: failed-read cleanup: %s\n", dense.last_error().c_str());
                    if (rdev1) rdev1 = dense.routed_dev();
                    each_tier([](Ds41MoeTier & t) { t.release_chunk(); });
                    std::printf("ERR prefill failed: %s\n", ferr.c_str());
                }
                std::fflush(stdout);
                // 0.94: a CANCELLED read (Esc in an agent, a client watchdog, an aborted helper request) kept nothing, so the
                // next request re-read the whole conversation from token 0 (70-90 s at 50K, minutes at 200K).  When this
                // read started at or past the latest snapshot, every row below it and the snapshot itself are untouched (the
                // feed writes rows >= reused): restore it now, so the live state, parking and the conversation cache all
                // see the snapshot's state, and keep the history up to it.  Otherwise (an error, or a read that began
                // below the snapshot) nothing is kept, as before.
                if (prefill_cancelled && snap_len > 0 && reused >= snap_len && hist.size() >= (size_t) snap_len && dense.restore()) {
                    hist.resize((size_t) snap_len);
                    live_hi = -1;    // the snapshot's rings are the live ones again
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;   // the drafter's ring re-seeds from the next prompt's tail
#endif
                    std::fprintf(stderr, "serve: cancelled read - the conversation is kept at its %d-token snapshot\n", snap_len);
                } else {
                    hist.clear();
                    snap_len = -1;
                }
                continue;
            }
            hist = ids;
            hist_vis = req_vis;
            // D11 ring fixes: the feed wrote [reused, total); draft rows from an earlier decode above total - 1 stay
            if (!(reused > 0 && live_hi > (int64_t) total - 1)) live_hi = -1;
            // image prompts snapshot too (0.94): hist_vis keeps a text request with the same ids from restoring their rows
            // STRATA_SERVE_SNAP_STACK: the snapshot this one replaces is kept while its rows are in place (this request
            // resumed at or past it)
            if (snap_stack_k > 0 && total > reused && snap_len > 0 && reused >= snap_len) {
                if (auto st = dense.save_conv(0)) {
                    snap_stack.push_back({ snap_len, st });
                    if ((int) snap_stack.size() > snap_stack_k) snap_stack.erase(snap_stack.begin());
                }
            }
            if (total > reused && dense.snapshot(total)) {   // image prompts too: hist_vis guards their reuse
                snap_len = total;
                snap_logits = last_logits;
                snap_seed.swap(cur_seed);   // this prompt's DSpark tail rows (0 rows when DSpark is off / no tail ran)
                snap_seed_n = cur_seed_n;
            }
            cur_seed_n = 0;
            if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {   // exactness checks: request N's last-row logits
                char nb[32];
                std::snprintf(nb, sizeof(nb), "/req%03d.f32", n_req);
                std::ofstream f(std::string(dd) + nb, std::ios::binary);
                f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
            }
            ++n_req;
            const double prompt_ms = now_ms() - tp0;
            if (exact_pass && cache_open) {   // decode: the biases back on, from the VRAM set as it is now
                bias_now = true;
                apply_bias();
            }
            const Ds41MoeStats pst = stats_all();
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused, %s) in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / "
                                 "%lld pcie / %lld file, file wait %.0f ms)\n", total, reused, by_step ? (exact_pass ? "exact passes" : "passes") : "batch", prompt_ms / 1000.0,
                         1000.0 * (total - reused) / std::max(prompt_ms, 1e-9), (long long) pst.hits, (long long) pst.cpu,
                         (long long) pst.pcie, (long long) pst.file_tier, pst.file_ms);
            if (const Ds41ReadFallbacks fb = ds41_read_fallbacks(); fb.total() > 0)   // B1: process totals, only when any
                std::fprintf(stderr, "serve: read fallbacks so far: %lld O_DIRECT -> buffered, %lld pread -> mapping, "
                                     "%lld pinned -> pageable\n", (long long) fb.direct_to_buffered,
                             (long long) fb.pread_to_mmap, (long long) fb.pinned_to_pageable);
            if (defer_vram && !cache_open) {   // decode next: the VRAM cache opens into what the prompt freed
                if (!open_cache()) {
                    std::printf("ERR VRAM seed failed: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                cache_open = true;
                std::fprintf(stderr, "serve: VRAM %.2f GiB free -> %lld experts in VRAM (seed %.2f s)\n", seed_free,
                             (long long) tier.resident(), seed_s);
            }
            if (ok) each_tier([](Ds41MoeTier & t) { t.rebalance_plan(); });
            if (bias_changed) {   // decode: this request's route bias + file penalty (see bias_changed)
                bias_now = true;
                apply_bias();
            }
            each_tier([](Ds41MoeTier & t) { t.reset_stats(); });
            dense.route_meter(true);
            const double td0 = now_ms();
            int gen = 0, pos = total;
            auto lazy_regrow = [&]() {   // STRATA_KEEP_LAZY: between passes, nothing of the tier in flight
                if (!regrow_due || gen < keep_lazy) return;
                regrow_due = false;
                std::string kerr;
                if (tier.regrow_cache(kerr) < 0) {   // mid-answer there is no clean way back; the server restarts a dead engine
                    std::fprintf(stderr, "serve: keep-cache regrow: %s - ending the engine\n", kerr.c_str());
                    std::exit(3);
                }
                apply_bias();
            };
            const char * finish = "length";
            static const SinkOpts & sk = sink_opts();
            auto sk_line = [&](int row, int at, int tok, const char * tag) {   // STRATA_SINK_TRACE
                std::fprintf(stderr, "sink: rq %d pos %d tok %d L%d %.0f Llast %.0f %s\n", n_req, at, tok, dense.sink_layer(),
                             dense.sink_value(row, true), dense.sink_value(row, false), tag);
            };
            if (sk.trace && dense.sink_seq() != sk_seq_req && dense.sink_rows() > 0 && !ids.empty())
                sk_line(dense.sink_rows() - 1, total - 1, ids.back(), "prompt");
            int sk_rej = 0, sk_cut = 0, sk_fall = 0;   // STRATA_SINK_REJECT events of this request
            (void) sk_rej; (void) sk_cut; (void) sk_fall;   // (a build without DSpark has no verify loop)
            set_run_k(dec_k, true);   // STRATA_EXPERTS_USED_DECODE: this request's decode / verify passes route dec_k
            std::vector<float> lg = last_logits;
#if defined(DS41_HAVE_DSPARK)
            // ---- DSpark speculative sampling (as the one-shot loop: P = last fed position, t = the token for P+1,
            // sp_ids = drafts for P+2..; each verify row's distribution is sample_p's, spec_accept keeps it exact)
            long rq_pass = 0, rq_acc = 0;
            tier.trace_snapshot();   // STRATA_ROUTE_TRACE: the VRAM / arena sets this request's decode starts from
            if (!dsp) sp_t = -1;   // the drafter failed to load during this request: the plain decode below (from lg)
            if (sp_t >= 0) {
                int P = total - 1, t = sp_t, prev = ids.back();
                bool dirty = false;
                std::vector<int> vt;
                std::vector<std::pair<int, double>> dist;
                // STRATA_SERVE_LOOKUP=L (s6, opt-in until measured; mab776/Strata-DS4 M11's design): when the context's
                // last G+1 tokens (STRATA_LOOKUP_NGRAM=G, default 3) occurred earlier in this request (prompt + output),
                // the tokens that followed them there are the verify pass's drafts instead of DSpark's - an agent's output
                // copies its context (edits, rewrites, quoted tool results).  spec_accept treats every draft as a
                // deterministic token (accept with p, else resample without it), so the output distribution is unchanged.
                // A match of M tokens drafts M - G + 1 (<= L, <= the pass width).  STRATA_LOOKUP_AGREE (default 1): only
                // when the copy's first token is DSpark's own first draft.  A lookup pass with less than half accepted
                // sits the lookup out 1, 2, 4 .. 64 passes.  DSpark steps on the verified rows either way.
                // kVerify (s6): rows one verify pass may take - 16, as the prompt passes (pass_v splits the tier calls at
                // kMaxMulti - 8 then, 16 now - and the pass buffers hold 17).  A copy is accepted ~96% on edits, so the
                // draft length was the limit at the old kMaxMulti - 2 = 6 (s6: 932/972 accepted in 169 passes, only +3%
                // over DSpark's own).
                constexpr int kVerify = 16;
                static const int lk_max = std::getenv("STRATA_SERVE_LOOKUP") ? std::max(0, std::min(kVerify - 2, std::atoi(std::getenv("STRATA_SERVE_LOOKUP")))) : 0;
                static const int lk_g = std::getenv("STRATA_LOOKUP_NGRAM") ? std::max(1, std::atoi(std::getenv("STRATA_LOOKUP_NGRAM"))) : 3;
                static const bool lk_agree = !(std::getenv("STRATA_LOOKUP_AGREE") && std::getenv("STRATA_LOOKUP_AGREE")[0] == '0');
                // STRATA_LOOKUP_LONG=1 (default off; Basalt 97bf1f168, ds41_lookup.hpp): also the latest places whose last 8
                // tokens match - an old source behind a common short n-gram's newer repeats; the longest copy wins
                static const bool lk_long = lk_max > 0 && std::getenv("STRATA_LOOKUP_LONG") && std::getenv("STRATA_LOOKUP_LONG")[0] == '1';
                long lk_long_n = 0;   // passes whose copy came from the long scan
                int lk_cool = 0, lk_back = 1;
                long lk_pass = 0, lk_dr = 0, lk_acc = 0;
                int dr[kVerify] = {0};
                // ---- (all default off; unset = the previous passes, byte for byte, and no extra work):
                // STRATA_DSPARK_GATE=1: the drafts a pass verifies are chosen by expected tokens per ms - draft j is
                //   verified while survival_j = prod_{i<=j} s_i >= lambda x deltaC (s_i the CALIBRATED acceptance
                //   probability, lambda the measured tokens per ms, deltaC the measured marginal ms of one more row; see
                //   GateCost).  Replaces STRATA_DSPARK_CONF / STRATA_DSPARK_CUM.  s_i depends on the drafts before slot i
                //   only (the confidence reads the previous slot's pick), so whether row i is verified never depends on
                //   its own draw: the output distribution stays exact (same-seed texts move: the width moves the RNG).
                // STRATA_DSPARK_CALIB=1 (implied by GATE): the online calibration (DraftCalib) - alone it feeds the
                //   CONF / CUM cuts the calibrated probability instead of the raw sigmoid.
                // STRATA_LOOKUP_YIELD=1 (with STRATA_SERVE_LOOKUP): a copy is taken only when offered x the request's
                //   decayed copy hit rate (prior 0.5 over 4 tokens) >= DSpark's expected accepted drafts; the offer
                //   starts at 8 tokens and doubles to 16 after a fully accepted copy (back to 8 after any miss); among
                //   the matches the one offering the longest copy wins, then the latest (off: the latest match only).
                //   No cool-down (the hit rate does that).  For sampled drafts DSpark's expectation uses this pass's
                //   confidences only up to the slot whose pick the copy's own rule already depends on (slot 0 - slot 1
                //   with STRATA_LOOKUP_AGREE), the per-slot mean after it: the choice never depends on a draw whose row
                //   it changes the accept rule of (exactness, as the 0.94 q_row0 fix).
                // STRATA_DSPARK_GATE_TRACE=<file>: one TSV line per verified DSpark draft: rq P slot k conf_logit
                //   q_top2_margin log_q(d) p(d) q(d) accepted(1/0/-1 = after the first rejection) label (min(1, p/q)) -
                //   training data for a learned gate.  "nan" where the draft is an argmax one (no q).
                // A summary per request on stderr ("serve: gate: ...") when any of them is set.
                auto env_on = [](const char * k) { const char * e = std::getenv(k); return e && e[0] && e[0] != '0'; };
                static const bool g_gate = env_on("STRATA_DSPARK_GATE");
                // 0.95 STRATA_DSPARK_LGATE=1 (default off: unset = the code paths above, byte for byte): on DSpark passes with
                //   SAMPLED drafts (temperature > 0, STRATA_DSPARK_SAMPLE) the width comes from a learned acceptance predictor
                //   and the acceptance coins drawn BEFORE the pass (ds41_lgate.hpp: the design and the exactness argument);
                //   it replaces STRATA_DSPARK_GATE / CALIB / CONF / CUM there.  Greedy requests, argmax drafts and lookup
                //   passes keep the shipped rules.  STRATA_DSPARK_LGATE_W="b,w0..w8" overrides the predictor's weights.
                // STRATA_DSPARK_TRACE_X=1 (with STRATA_DSPARK_TRACE): extra fields for refitting it - per pass ms / rows /
                //   think flag / the previous resolution row's target entropy and top-1 / the shipped gate's width kg /
                //   lambda / deltaC / the coin cut; per sampled draft the drafter's top-4 mass and, on verified rows,
                //   tv = sum min(p, q) (the draft's exact expected acceptance) and p(the drafter's top-1); ahat with LGATE.
                static const bool g_lgate = env_on("STRATA_DSPARK_LGATE");
                static const bool g_trx = env_on("STRATA_DSPARK_TRACE_X");
                static const bool g_calib = g_gate || env_on("STRATA_DSPARK_CALIB");
                static const bool g_yield = lk_max > 0 && env_on("STRATA_LOOKUP_YIELD");
                static FILE * g_trace = [] { const char * e = std::getenv("STRATA_DSPARK_GATE_TRACE"); return e && *e ? std::fopen(e, "a") : nullptr; }();
                static const bool g_obs = g_calib || g_yield || g_trace != nullptr || g_lgate || g_trx;
                static const bool g_any = g_obs;
                static DraftCalib g_cal;
                static GateCost g_cost;
                static ds41_lgate::Predictor g_lp = [] { ds41_lgate::Predictor p; p.load_env(); return p; }();
                static ds41_lgate::CondTable g_lt;   // 0.95 LGATE: P(a > u | slot, ahat decile), seeded, updated online
                auto g_prob = [&](int j, float c) { return g_calib ? g_cal.prob(j, c) : DraftCalib::raw(c); };
                double g_pd[kVerify] = {0};
                std::vector<std::pair<int, double>> g_dist;
                long g_k = 0, g_lkskip = 0;
                long lg_passes = 0, lg_cuts = 0, lg_cut_acc = 0;   // 0.95 LGATE: coin passes, coin cuts, drafts kept at the cut
                double x_hprev = -1.0, x_p1prev = -1.0;   // 0.95 TRACE_X: the previous resolution row's target entropy / top-1
                int x_think = 0;                          // 0.95 TRACE_X: inside the thinking (from the prompt, then the output)
                if (g_trx) {
                    long lb = -1, le = -1;
                    for (size_t i = 0; i < ids.size(); ++i) {
                        if (ids[i] == think_beg_id) lb = (long) i;
                        else if (ids[i] == think_end_id) le = (long) i;
                    }
                    x_think = lb > le ? 1 : 0;
                }
                double lk_h = 0, lk_o = 0;   // STRATA_LOOKUP_YIELD: decayed copied tokens accepted / offered (this request)
                int lk_offer = 8;
                auto emit = [&](int tok) -> bool {   // false = stop here
                    std::printf("T %d\n", tok);
                    std::fflush(stdout);
                    ++gen;
                    enter_answer(tok);
                    if (is_stop(tok)) { finish = "stop"; return false; }
                    if (max_new > 0 && gen >= max_new) return false;
                    return true;
                };
                // STRATA_SINK_REJECT: t is held (not emitted) until the next pass has checked its row; sk_row is the
                // logits row t was drawn from, sk_excl the tokens that draw may no longer give, sk_tries its rejections
                const bool sk_hold = sk.thr > 0;
                bool held = false;
                std::vector<float> sk_row;
                std::vector<int> sk_excl;
                int sk_tries = 0;
                if (sk_hold) sk_row = last_logits;   // t = sp_t came from the prompt's last row
                // the tokens the correction never touches (sink_special_set + the stop ids + the think ids), once
                static const std::vector<uint8_t> sk_ban = [&] {
                    std::vector<uint8_t> b = sk_hold ? sink_special_set(a.model, nv) : std::vector<uint8_t>();
                    auto mark = [&](int x) { if (x >= 0 && x < (int) b.size()) b[(size_t) x] = 1; };
                    if (!b.empty()) {
                        for (int x : a.stop) mark(x);
                        mark((int) G.eos);
                        mark((int) G.bos);
                        mark(think_beg_id);
                        mark(think_end_id);
                    }
                    return b;
                }();
                auto sk_ctl = [&](int x) { return x >= 0 && x < (int) sk_ban.size() && sk_ban[(size_t) x] == 1; };
                for (;;) {
                    if (!sk_hold) { if (!emit(t)) break; }
                    else if (is_stop(t) || (max_new > 0 && gen + 1 >= max_new) || ctxv - (P + 3) < 0) { emit(t); break; }   // final: unchecked
                    else if (sk_ctl(t)) { if (!emit(t)) break; }   // a control token (</think>, a tool marker ...) is never held
                    else held = true;
                    const int gen_c = gen + (held ? 1 : 0);   // tokens out once t is
                    lazy_regrow();
                    if (stop_pending()) { finish = "cancel"; break; }
                    int k = sp_depth_serve;
                    if (max_new > 0) k = std::min(k, max_new - gen_c);
                    k = std::min(k, ctxv - (P + 3));   // the pass writes P+1 .. P+1+k
                    if (k < 0) break;
                    // STRATA_DSPARK_CONF=<tau>, as the one-shot loop: verify only the leading drafts whose confidence-head
                    // probability is >= tau (the emitted tokens are still the trunk's - only the verify width changes)
                    static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
                    // STRATA_DSPARK_CUM=<c> (Maya's MAYA_DS4_DSPARK_CUM): also stop once the PRODUCT of the confidences -
                    // the chance that this draft and every one before it are kept - falls below c (the same tokens)
                    static const float cum_s = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
                    const int k_bud = k;   // 0.95: the width before any cut (STRATA_LOOKUP_YIELD's DSpark estimate)
                    const double g_thr = (g_gate || g_lgate) ? g_cost.lambda() * g_cost.dc() : 0.0;
                    // 0.95 LGATE: the predicted acceptance of each slot, from features known before that slot's draw (the
                    // head's logit, the drafter's row at the slot, the head's product over the slots before it)
                    const bool lg_on = g_lgate && sp_sampled && sp_q != nullptr;
                    double lg_ahat[kVerify + 1] = { 0.0 };
                    if (lg_on) {
                        const int64_t V = dsp->vocab();
                        double lcp = 0.0;
                        for (int j = 0; j < k && j < ds41_lgate::kPos && sp_ids[j] >= 0; ++j) {
                            const ds41_lgate::SlotFeat f = ds41_lgate::slot_features(sp_q + (size_t) j * (size_t) V, V);
                            lg_ahat[j] = g_lp.predict(j, sp_conf[j], f, lcp);
                            lcp += std::log(std::max(1e-6, DraftCalib::raw(sp_conf[j])));
                        }
                    }
                    if (lg_on) {
                        // no cut here: the coins cut the width after the lookup decision (below), which must not see them
                    } else if (g_gate) {   // 0.95 STRATA_DSPARK_GATE (see above)
                        int kk = 0;
                        double sv = 1.0;
                        while (kk < k && (sv *= g_cal.prob(kk, sp_conf[kk])) >= g_thr) ++kk;
                        k = kk;
                    } else if (g_calib) {   // 0.95 STRATA_DSPARK_CALIB alone: the same cuts on the calibrated probability
                        if (conf_tau_s >= 0) {
                            int kk = 0;
                            while (kk < k && g_cal.prob(kk, sp_conf[kk]) >= (double) conf_tau_s) ++kk;
                            k = kk;
                        }
                        if (cum_s > 0) {
                            int kk = 0;
                            double pr = 1.0;
                            while (kk < k && (pr *= g_cal.prob(kk, sp_conf[kk])) >= (double) cum_s) ++kk;
                            k = kk;
                        }
                    } else {
                    if (conf_tau_s >= 0) {
                        int kk = 0;
                        while (kk < k && 1.0f / (1.0f + std::exp(-sp_conf[kk])) >= conf_tau_s) ++kk;
                        k = kk;
                    }
                    if (cum_s > 0) {
                        int kk = 0;
                        float pr = 1.0f;
                        while (kk < k && (pr *= 1.0f / (1.0f + std::exp(-sp_conf[kk]))) >= cum_s) ++kk;
                        k = kk;
                    }
                    }
                    {   // draft-id bounds guard (Maya 075cbe6): Ds41DSpark marks a slot with no finite logit, and every
                        // slot after it, as id -1 - "no draft" from there; such an id must never reach the verify pass
                        int kk = 0;
                        while (kk < k && sp_ids[kk] >= 0) ++kk;
                        k = kk;
                    }
                    const bool refeed = dirty && ((P + 1) & 1);
                    for (int i = 0; i < k; ++i) dr[i] = sp_ids[i];
                    bool is_lk = false;
                    if (lk_max > 0 && lk_cool > 0) --lk_cool;
                    else if (g_yield) {   // 0.95 STRATA_LOOKUP_YIELD (see above); the sequence as below: hist + t
                        const int n = (int) hist.size();
                        auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
                        int cap = std::min({lk_max, kVerify - 1 - (refeed ? 1 : 0), lk_offer});
                        if (max_new > 0) cap = std::min(cap, max_new - gen_c);   // gen_c: a token held by STRATA_SINK_REJECT counts
                        cap = std::min(cap, ctxv - (P + 3));
                        if (cap >= 2 && n > lk_g + 1) {
                            int bq = -1, bL = 0, seen = 0;
                            int q = n - 1;
                            for (; q >= lk_g && seen < 256; --q) {   // latest first: a tie keeps the latest
                                if (hist[(size_t) q] != t) continue;
                                int m = 1;
                                while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
                                if (m < lk_g + 1) continue;
                                ++seen;
                                if (lk_agree && sq(q + 1) != sp_ids[0]) continue;
                                const int L = std::min({cap, m - lk_g + 1, n - q});
                                if (L > bL) { bL = L; bq = q; if (L >= cap) break; }
                            }
                            // STRATA_LOOKUP_LONG: the 256 newest matches were all looked at and none copies `cap`: the older
                            // places matching 8+ tokens (up to 256 more), the same agreement rule
                            if (lk_long && seen >= 256 && bL < cap && !(lk_agree && sp_ids[0] < 0)) {   // review: no 'agree with anything'
                                int q2 = -1;
                                const int L2 = lookup_long_scan(hist.data(), n, t, q, lk_g, cap, 256, lk_agree ? sp_ids[0] : -1, bL, q2);
                                if (L2 > bL) { bL = L2; bq = q2; ++lk_long_n; }
                            }
                            if (bL >= 2) {
                                // DSpark's expected accepted drafts this pass: sum of the survivals while the gate (if on)
                                // would verify them; sampled drafts use this pass's confidences only for the free slots
                                const int jfree = sp_sampled ? (lk_agree ? 2 : 1) : DraftCalib::kPos;
                                double sv = 1.0, ds = 0.0;
                                for (int j = 0; j < k_bud && j < DraftCalib::kPos; ++j) {
                                    sv *= j < jfree ? (lg_on ? lg_ahat[j] : g_prob(j, sp_conf[j])) : g_cal.mean[j];
                                    if (sv < g_thr) break;
                                    ds += sv;
                                }
                                const double hr = (lk_h + 2.0) / (lk_o + 4.0);
                                if ((double) bL * hr >= ds) {
                                    for (int i = 0; i < bL; ++i) dr[i] = sq(bq + 1 + i);
                                    k = bL;
                                    is_lk = true;
                                } else {   // passed over: the hit rate drifts back toward its prior
                                    lk_h *= 0.97; lk_o *= 0.97;
                                    ++g_lkskip;
                                }
                            }
                        }
                    }
                    else if (lk_max > 0) {
                        // the sequence is hist + t (t = the token for P+1, not in hist yet); find the latest earlier q with
                        // seq[q-G .. q] == seq[n-G .. n] (G+1 tokens, the last being t), then copy seq[q+1 ..]
                        const int n = (int) hist.size();
                        auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
                        int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                        if (max_new > 0) cap = std::min(cap, max_new - gen_c);
                        cap = std::min(cap, ctxv - (P + 3));
                        if (cap >= 2 && n > lk_g + 1) {
                            for (int q = n - 1; q >= lk_g; --q) {
                                if (hist[(size_t) q] != t) continue;
                                int m = 1;
                                while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
                                if (m < lk_g + 1) continue;
                                int L = std::min({cap, m - lk_g + 1, n - q});
                                if (lk_long && L < cap) {   // STRATA_LOOKUP_LONG: the 4 newest older 8-token matches (Basalt's WAYS)
                                    int q2 = -1;
                                    const int L2 = lookup_long_scan(hist.data(), n, t, q - 1, lk_g, cap, 4, lk_agree ? sp_ids[0] : -1, L, q2);
                                    if (L2 > L) { L = L2; q = q2; ++lk_long_n; }
                                }
                                if (L < 2) break;   // the latest match copies too little: not worth a pass
                                if (lk_agree && sq(q + 1) != sp_ids[0]) break;
                                for (int i = 0; i < L; ++i) dr[i] = sq(q + 1 + i);
                                k = L;
                                is_lk = true;
                                break;
                            }
                        }
                    }
                    // 0.95 LGATE coin-first: the coins of this pass's positions drawn now, the width cut on them (ds41_lgate.hpp).
                    // A DSpark pass only (a lookup copy keeps its own rule); position k_cut is resolved from row k_cut with
                    // its own coins after the verify loop.
                    bool lg_coin = false, lg_cut = false, lg_rej = false;
                    double lg_u[kVerify + 1], lg_r[kVerify + 1];
                    if (lg_on && !is_lk && k > 0) {
                        k = std::min(k, ds41_lgate::kPos);
                        std::uniform_real_distribution<double> U(0.0, 1.0);
                        for (int j = 0; j <= k; ++j) { lg_u[j] = U(rng); lg_r[j] = U(rng); }
                        const int kc = ds41_lgate::coin_cut(lg_ahat, lg_u, k, g_thr, g_lt);
                        lg_cut = kc < k;
                        k = kc;
                        lg_coin = true;
                        ++lg_passes;
                        if (lg_cut) ++lg_cuts;
                    }
                    vt.clear();
                    if (refeed) vt.push_back(prev);
                    vt.push_back(t);
                    for (int i = 0; i < k; ++i) vt.push_back(dr[i]);
                    const int r0 = refeed ? 1 : 0, p0 = refeed ? P : P + 1;
                    // STRATA_BIAS_REFRESH=N (default 0 = off; upstream 078896c, on there by default): the route bias /
                    // file penalty rebuilt from the VRAM set as it is NOW every N verify passes.  The set keeps changing
                    // during a decode (vram_lru admissions, arena rebalancing), so the request-start snapshot favours
                    // evicted experts and penalises admitted ones.  Upstream (Q8, Linux): 17-19.5 -> 21-22 tok/s, API
                    // top-1 85.1 -> 86.7%.  Changes outputs, as any residency change does.
                    const int bias_refresh = knobs.bias_refresh;   // live knob (per request)
                    if (bias_refresh > 0 && bias_now && bias_set && rq_pass > 0 && rq_pass % bias_refresh == 0) apply_bias(true);
                    const double g_t0 = g_any ? now_ms() : 0.0;   // 0.95: the pass's ms for lambda / deltaC
                    if (!pass_v(vt.data(), (int) vt.size(), p0)) { finish = "error"; break; }
                    // D11 ring fixes: this pass wrote [p0, p0 + vt.size()) into the live rings, rejected drafts included
                    live_hi = std::max(live_hi, (int64_t) p0 + (int64_t) vt.size() - 1);
                    int acc = 0, tn = -1;
                    if (held) {   // STRATA_SINK_REJECT: row r0 is t's own row
                        const float v0 = dense.sink_value(r0, false);
                        if (v0 > sk.thr) {
                            int alt = -1;
                            bool outside = false;
                            if (sk_tries < sk.tries) {
                                sk_excl.push_back(t);
                                alt = sink_alt(sk_row.data(), nv, temp, tk, top_p, min_p, sk_excl, rng, &outside, &sk_ban);
                            }
                            if (alt >= 0) {
                                if (sk.trace) sk_line(r0, P + 1, t, "sunk");
                                std::fprintf(stderr, "sink: rejected rq %d pos %d tok %d value %.0f -> tok %d (try %d of %d%s)\n",
                                             n_req, P + 1, t, v0, alt, sk_tries + 1, sk.tries,
                                             outside ? ", best logit outside the sampler's set" : "");
                                ++sk_tries;
                                ++sk_rej;
                                t = alt;
                                dirty = true;     // position P+1 and the drafts after it are stale in the live rings
                                sp_ids[0] = -1;   // DSpark's drafts followed the rejected token: the next pass checks t alone
                                held = false;     // decided again at the loop head
                                continue;
                            }
                            std::fprintf(stderr, "sink: fallthrough rq %d pos %d tok %d value %.0f (kept after %d tries%s)\n",
                                         n_req, P + 1, t, v0, sk_tries, sk_tries < sk.tries ? ", no token left" : "");
                            ++sk_fall;
                        }
                        sk_tries = 0;
                        sk_excl.clear();
                        held = false;
                        if (!emit(t)) break;   // not reached: a final token was emitted before the pass
                    }
                    const bool q_on = sp_sampled && !is_lk;   // DSpark's sampled drafts: accept against their q rows
                    // 0.94 exactness fix: with STRATA_LOOKUP_AGREE a lookup pass is taken only when the copy's first token is
                    // DSpark's own first draft - a SAMPLED draw from q.  Accepting that row with p(d) on lookup passes and
                    // min(1, p/q) on DSpark passes picked the rule by the draw itself, so the mix was not exact (the copied
                    // token came out ~q*p instead of p; TV up to ~0.1 when the drafter is unsure; Maya refuses the pair for
                    // this reason).  Row 0 of such a pass is now accepted against q like any DSpark row; the copied rows
                    // after it are deterministic given the context, so p(d) stays exact for them.
                    // (only with the agree gate: without it the pass does not depend on the draw, and p(d) is exact)
                    const bool q_row0 = sp_sampled && is_lk && lk_agree && dr[0] == sp_ids[0];
                    const bool g_obs_pass = g_obs && !is_lk;   // 0.95: DSpark passes only (a copy is not the head's draft)
                    for (; acc < k; ++acc) {
                        sample_dist(vlog.data() + (size_t) (r0 + acc) * nv, nv, temp, tk, top_p, min_p, dist);
                        if (g_obs_pass) {
                            double pd = 0;
                            for (const auto & c : dist) if (c.first == dr[acc]) pd = c.second;
                            g_pd[acc] = pd;
                        }
                        if (sk_hold) {   // STRATA_SINK_REJECT: draft acc's own row (r0 + 1 + acc) is sunk - cut here
                            const float vd = dense.sink_value(r0 + 1 + acc, false);
                            if (vd > sk.thr && !sk_ctl(dr[acc])) {   // a control token is never cut
                                const std::vector<int> ex { dr[acc] };
                                bool outside = false;
                                const int a2 = sink_alt(vlog.data() + (size_t) (r0 + acc) * nv, nv, temp, tk, top_p, min_p, ex, rng,
                                                        &outside, &sk_ban);
                                if (a2 >= 0) {
                                    std::fprintf(stderr, "sink: cut rq %d pos %d draft %d value %.0f -> tok %d%s\n", n_req, P + 2 + acc,
                                                 dr[acc], vd, a2, outside ? " (best logit outside the sampler's set)" : "");
                                    ++sk_cut;
                                    tn = a2;
                                    break;
                                }
                                std::fprintf(stderr, "sink: cut skipped rq %d pos %d draft %d value %.0f (no token left)\n", n_req,
                                             P + 2 + acc, dr[acc], vd);
                            }
                        }
                        const bool qa = q_on || (q_row0 && acc == 0);
                        if (lg_coin ? !ds41_lgate::accept_q_coins(dist, dr[acc], sp_q + (size_t) acc * (size_t) dsp->vocab(), dsp->vocab(),
                                                                  lg_u[acc], lg_r[acc], tn)
                                    : qa ? !spec_accept_q(dist, dr[acc], sp_q + (size_t) acc * (size_t) dsp->vocab(), dsp->vocab(), rng, tn)
                                         : !spec_accept(dist, dr[acc], rng, tn)) break;
                    }
                    if (acc == k && lg_cut) {   // 0.95 LGATE: the cut position from row k with its own coins (not a fresh draw)
                        const int dk = sp_ids[k];
                        const float * qk = sp_q + (size_t) k * (size_t) dsp->vocab();
                        sample_dist(vlog.data() + (size_t) (r0 + k) * nv, nv, temp, tk, top_p, min_p, dist);
                        if (ds41_lgate::accept_q_coins(dist, dk, qk, dsp->vocab(), lg_u[k], lg_r[k], tn)) { tn = dk; ++lg_cut_acc; }
                        else lg_rej = true;
                        double pd = 0.0;   // row k verified draft k too: one more observation for the table
                        for (const auto & c : dist) if (c.first == dk) pd = c.second;
                        const double qd = ds41_lgate::row_q(qk, dsp->vocab(), dk);
                        g_lt.add(k, lg_ahat[k], qd > 0 ? std::min(1.0, pd / qd) : 1.0);
                    } else if (acc == k) tn = sample_p(vlog.data() + (size_t) (r0 + k) * nv, nv, temp, tk, top_p, min_p, rng);
                    if (sk_hold) {   // tn is held next: its draw's row and what that draw excluded
                        sk_row.assign(vlog.data() + (size_t) (r0 + acc) * nv, vlog.data() + (size_t) (r0 + acc + 1) * nv);
                        sk_excl.clear();
                        if (acc < k) sk_excl.push_back(dr[acc]);
                        else if (lg_rej) sk_excl.push_back(sp_ids[k]);   // 0.95 LGATE: draft k was rejected at the cut
                        sk_tries = 0;
                    }
                    if (sk.trace)
                        for (int i = 0; i <= k; ++i) sk_line(r0 + i, P + 1 + i, i == 0 ? t : dr[i - 1], i <= acc ? "e" : "x");
                    if (g_obs_pass && k > 0) {   // 0.95: every verified draft is an observation (no RNG used here)
                        // rows past the first rejection were verified too: the trunk's row i is conditioned on drafts
                        // 0..i-1, so p(d_i) there is exactly the conditional acceptance the survival product needs
                        for (int i = acc + 1; i < k; ++i) {
                            sample_dist(vlog.data() + (size_t) (r0 + i) * nv, nv, temp, tk, top_p, min_p, g_dist);
                            double pd = 0;
                            for (const auto & c : g_dist) if (c.first == dr[i]) pd = c.second;
                            g_pd[i] = pd;
                        }
                        const int64_t V = dsp->vocab();
                        for (int i = 0; i < k; ++i) {
                            const float * cd = sp_sampled ? sp_q + (size_t) i * (size_t) V : nullptr;
                            const double qd = cd ? draft_q(cd, V, dr[i]) : 1.0;
                            const double y = qd > 0 ? std::min(1.0, g_pd[i] / qd) : 1.0;
                            g_cal.add(i, sp_conf[i], y);
                            if (lg_coin) g_lt.add(i, lg_ahat[i], y);   // 0.95 LGATE: the conditional table follows the drafter
                            if (g_trace) {
                                const int hard = i < acc ? 1 : (i == acc ? 0 : -1);
                                if (cd) {
                                    double q1 = -1, q2 = -1, prv = 0;
                                    for (int64_t x = 0; x < V; ++x) {
                                        const double q = std::min(1.0, (double) cd[x]) - prv;
                                        prv = std::min(1.0, (double) cd[x]);
                                        if (q > q1) { q2 = q1; q1 = q; } else if (q > q2) q2 = q;
                                    }
                                    std::fprintf(g_trace, "%d\t%d\t%d\t%d\t%.4f\t%.5f\t%.4f\t%.5f\t%.5f\t%d\t%.5f\n", n_req, P, i, k,
                                                 sp_conf[i], q1 - q2, qd > 0 ? std::log(qd) : -99.0, g_pd[i], qd, hard, y);
                                } else {
                                    std::fprintf(g_trace, "%d\t%d\t%d\t%d\t%.4f\tnan\tnan\t%.5f\tnan\t%d\t%.5f\n", n_req, P, i, k,
                                                 sp_conf[i], g_pd[i], hard, y);
                                }
                            }
                        }
                        if (g_trace) std::fflush(g_trace);
                    }
                    // STRATA_DSPARK_TRACE=<file> (opt-in, data collection only - nothing else reads it, no output change; it
                    // costs a few ms a pass): one JSON line per verify pass with every draft DSpark made (also the ones the
                    // confidence cut left out): its id, the confidence head's sigmoid, and - for sampled drafts - q(draft),
                    // q's top-2 and entropy; for the verified rows the trunk's p(draft) and top-1.  Offline (drafter_misses.py)
                    // the emitted stream gives the truth for the rows that were never verified.
                    static FILE * dtrace = [] { const char * e = std::getenv("STRATA_DSPARK_TRACE"); return e && *e ? std::fopen(e, "a") : nullptr; }();
                    if (dtrace) {
                        const int64_t V = dsp->vocab();
                        std::fprintf(dtrace, "{\"rq\":%d,\"P\":%d,\"t\":%d,\"temp\":%.3f,\"lk\":%d,\"k\":%d,\"acc\":%d,\"tn\":%d,",
                                     n_req, P, t, temp, is_lk ? 1 : 0, k, acc, tn);
                        if (g_trx) {   // 0.95 TRACE_X: per-pass fields (see STRATA_DSPARK_TRACE_X above)
                            if (t == think_beg_id) x_think = 1; else if (t == think_end_id) x_think = 0;
                            const double thr_x = g_cost.lambda() * g_cost.dc();
                            int kg = 0;
                            double svx = 1.0;
                            while (kg < k_bud && kg < DraftCalib::kPos && sp_ids[kg] >= 0 && (svx *= g_cal.prob(kg, sp_conf[kg])) >= thr_x) ++kg;
                            std::fprintf(dtrace, "\"ms\":%.2f,\"rows\":%d,\"th\":%d,\"hprev\":%.4f,\"p1prev\":%.4f,\"kbud\":%d,\"kg\":%d,"
                                                 "\"lam\":%.5f,\"dc\":%.3f,\"lgc\":%d,\"lgcut\":%d,",
                                         now_ms() - g_t0, (int) vt.size(), x_think, x_hprev, x_p1prev, k_bud, kg, g_cost.lambda(),
                                         g_cost.dc(), lg_coin ? 1 : 0, lg_cut ? 1 : 0);
                        }
                        std::fprintf(dtrace, "\"d\":[");
                        std::vector<std::pair<int, double>> md;
                        for (int i = 0; i < sp_depth_serve; ++i) {
                            std::fprintf(dtrace, "%s{\"id\":%d,\"c\":%.4f", i ? "," : "", sp_ids[i], 1.0f / (1.0f + std::exp(-sp_conf[i])));
                            if (sp_sampled) {
                                const float * cd = sp_q + (size_t) i * (size_t) V;
                                int b1 = -1, b2 = -1; double q1 = -1, q2 = -1, H = 0, prev = 0;
                                double t4[4] = { 0, 0, 0, 0 };   // 0.95 TRACE_X: the four largest q
                                for (int64_t x = 0; x < V; ++x) {
                                    const double q = (double) cd[x] - prev; prev = cd[x];
                                    if (q > 1e-12) H -= q * std::log(q);
                                    if (q > q1) { q2 = q1; b2 = b1; q1 = q; b1 = (int) x; } else if (q > q2) { q2 = q; b2 = (int) x; }
                                    if (g_trx && q > t4[3]) {
                                        int p4 = 3;
                                        while (p4 > 0 && q > t4[p4 - 1]) { t4[p4] = t4[p4 - 1]; --p4; }
                                        t4[p4] = q;
                                    }
                                }
                                const int s = sp_ids[i];
                                const double qs = s < 0 || s >= V ? 0.0 : (double) cd[s] - (s > 0 ? (double) cd[s - 1] : 0.0);
                                std::fprintf(dtrace, ",\"q\":%.4f,\"q1\":[%d,%.4f],\"q2\":[%d,%.4f],\"H\":%.3f", qs, b1, q1, b2, q2, H);
                                if (g_trx) {
                                    std::fprintf(dtrace, ",\"m4\":%.4f", t4[0] + t4[1] + t4[2] + t4[3]);
                                    if (lg_on && i < ds41_lgate::kPos) std::fprintf(dtrace, ",\"ah\":%.4f", lg_ahat[i]);
                                    if (i < k && !is_lk) {   // tv: sum min(p, q) = the draft's exact expected acceptance; p(q's top-1)
                                        sample_dist(vlog.data() + (size_t) (r0 + i) * nv, nv, temp, tk, top_p, min_p, md);
                                        double tv = 0.0, pq1 = 0.0;
                                        for (auto & e : md) {
                                            tv += std::min(e.second, ds41_lgate::row_q(cd, V, e.first));
                                            if (e.first == b1) pq1 = e.second;
                                        }
                                        std::fprintf(dtrace, ",\"tv\":%.4f,\"pq1\":%.4f", tv, pq1);
                                    }
                                }
                            }
                            if (i < k && !is_lk) {
                                sample_dist(vlog.data() + (size_t) (r0 + i) * nv, nv, temp, tk, top_p, min_p, md);
                                double pd = 0; int m1 = md.empty() ? -1 : md[0].first; double p1 = md.empty() ? 0 : md[0].second;
                                for (auto & e : md) { if (e.first == dr[i]) pd = e.second; if (e.second > p1) { p1 = e.second; m1 = e.first; } }
                                std::fprintf(dtrace, ",\"p\":%.4f,\"m1\":[%d,%.4f]", pd, m1, p1);
                            }
                            std::fprintf(dtrace, "}");
                        }
                        std::fprintf(dtrace, "]}\n");
                        std::fflush(dtrace);
                        if (g_trx) {   // 0.95 TRACE_X: this pass's resolution row (it drew tn) -> the next line's hprev / p1prev
                            const int rr = std::min(acc, k);
                            sample_dist(vlog.data() + (size_t) (r0 + rr) * nv, nv, temp, tk, top_p, min_p, md);
                            double h = 0.0, p1 = 0.0;
                            for (auto & e : md) { if (e.second > 0) h -= e.second * std::log(e.second); p1 = std::max(p1, e.second); }
                            x_hprev = h; x_p1prev = p1;
                            for (int i = 0; i < acc && i < k; ++i) {
                                if (dr[i] == think_beg_id) x_think = 1; else if (dr[i] == think_end_id) x_think = 0;
                            }
                        }
                    }
                    ++rq_pass; rq_acc += acc;
                    if (tier.tracing()) {   // STRATA_ROUTE_TRACE: 'P' {m, k, acc, refeed, 0} - the server's passes (timing is in 'V')
                        const float pm[5] = { (float) vt.size(), (float) k, (float) acc, refeed ? 1.0f : 0.0f, 0.0f };
                        tier.trace_mark(pm, 5);
                    }
                    if (is_lk) {
                        ++lk_pass; lk_dr += k; lk_acc += acc;
                        if (g_yield) {   // 0.95: the request's decayed hit rate and the offer (8, 16 after a full copy)
                            lk_h = 0.8 * lk_h + acc;
                            lk_o = 0.8 * lk_o + k;
                            lk_offer = acc == k ? std::min(16, lk_offer * 2) : 8;
                        }
                        else if (2 * acc < k) { lk_cool = lk_back; lk_back = std::min(64, lk_back * 2); }
                        else lk_back = 1;
                    }
                    if (g_any) g_k += k;
                    int spos[kVerify + 1];
                    for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
                    int drafts[kVerify];
                    for (int i = 0; i < k; ++i) drafts[i] = dr[i];
                    if (!dstep(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, sp_ids, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); finish = "error"; break;
                    }
                    if (g_any) g_cost.add((int) vt.size(), acc + 1, now_ms() - g_t0);   // 0.95: pass + accept + draft
                    hist.push_back(t);
                    bool more = true;
                    for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); more = emit(drafts[i]); }
                    prev = acc == 0 ? t : drafts[acc - 1];
                    dirty = acc < k;
                    P += 1 + acc;
                    t = tn;
                    each_tier([](Ds41MoeTier & t) { t.rebalance_tick(); });
                    if (!more) break;
                }
                if (lk_pass > 0)
                    std::fprintf(stderr, "serve: lookup %ld/%ld drafts accepted in %ld passes\n", lk_acc, lk_dr, lk_pass);
                if (lk_long && lk_long_n > 0)   // STRATA_LOOKUP_LONG
                    std::fprintf(stderr, "serve: lookup long (8-token) match was the best copy %ld times\n", lk_long_n);
                if (sk_rej + sk_cut + sk_fall > 0)
                    std::fprintf(stderr, "serve: sink: %d rejected, %d drafts cut, %d fallthrough\n", sk_rej, sk_cut, sk_fall);
                if (g_lgate && rq_pass > 0)   // 0.95 LGATE: one line per request, greppable as "lgate:"
                    std::fprintf(stderr, "serve: lgate: %ld coin passes of %ld, %ld cut by the coins, %ld drafts kept at the cut; "
                                         "table %ld obs; weights %s\n", lg_passes, rq_pass, lg_cuts, lg_cut_acc, g_lt.obs,
                                 g_lp.from_env ? "STRATA_DSPARK_LGATE_W" : "built-in");
                if (g_any && rq_pass > 0)   // 0.95: one line per request, greppable as "gate:"
                    std::fprintf(stderr, "serve: gate: %ld passes, %.2f verified drafts/pass, %.2f accepted/pass, %.2f tokens/pass, "
                                         "lambda %.4f tok/ms, dC %.2f ms, thr %.3f (gate %d calib %d yield %d); calib %ld obs, "
                                         "slot means %.2f %.2f %.2f %.2f %.2f; copies %ld taken %ld passed over, hit %.2f, offer %d\n",
                                 rq_pass, (double) g_k / rq_pass, (double) rq_acc / rq_pass, 1.0 + (double) rq_acc / rq_pass,
                                 g_cost.lambda(), g_cost.dc(), g_cost.lambda() * g_cost.dc(), g_gate ? 1 : 0, g_calib ? 1 : 0,
                                 g_yield ? 1 : 0, g_cal.obs, g_cal.mean[0], g_cal.mean[1], g_cal.mean[2], g_cal.mean[3],
                                 g_cal.mean[4], lk_pass, g_lkskip, (lk_h + 2.0) / (lk_o + 4.0), lk_offer);
                dsp_next = P + 1;   // the ring holds every position <= P (the last emitted token is not fed)
                if (std::strcmp(finish, "error") == 0) dsp_next = -1;
                sv_pass += rq_pass; sv_acc += rq_acc;
            }
#endif
            while (sp_t < 0 && (gen < max_new || max_new <= 0)) {
                if (stop_pending()) { finish = "cancel"; break; }
                const int tok = sample_p(lg.data(), nv, temp, tk, top_p, min_p, rng);
                std::printf("T %d\n", tok);
                std::fflush(stdout);
                ++gen;
                enter_answer(tok);
                if (is_stop(tok)) { finish = "stop"; break; }
                if ((max_new > 0 && gen >= max_new) || pos + 1 >= ctxv) break;
                // STRATA_BIAS_REFRESH without a drafter: every N single steps (the verify loop above counts passes) - the live
                // knob (GEN ... bias_refresh=N), which starts from the same variable
                const int step_refresh = knobs.bias_refresh;
                lazy_regrow();
                if (step_refresh > 0 && bias_now && bias_set && gen % step_refresh == 0) apply_bias(true);
                if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { finish = "error"; break; }
                if (sk.trace) sk_line(0, pos, tok, "e");
                if (sk.thr > 0 && dense.sink_value(0, false) > sk.thr)   // no drafter: detected, not corrected
                    std::fprintf(stderr, "sink: flagged rq %d pos %d tok %d value %.0f (no drafter: emitted)\n", n_req, pos, tok,
                                 dense.sink_value(0, false));
                hist.push_back(tok);
                ++pos;
                each_tier([](Ds41MoeTier & t) { t.rebalance_tick(); });
            }
            const double dec_ms = now_ms() - td0;
            const Ds41MoeStats st = stats_all();
            if (std::strcmp(finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                // a GPU fault: end the engine before this request's DONE, so the request that failed is the one that
                // reports the ended engine.  At the loop head (as before) the server may already have sent the next
                // request, which then failed too.  (finish stays "length": the server maps only stop / length / cancel.)
                ds41_end_if_gpu_fault();
                hist.clear();
                snap_len = -1;
                finish = "length";
            }
            // D11 ring fixes (A/B): carry only what lies past the history this request leaves behind
            if (live_hi <= (int64_t) hist.size() - 1) live_hi = -1;
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file "
                                 "(%.0f ms; %lld read ahead)\n",
                         gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), finish,
                         (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier, st.file_ms,
                         (long long) st.spec_hits);
            if (st.remote > 0) std::fprintf(stderr, "serve: experts on the STRATA_DS41_REMOTE helper's card: %lld\n", (long long) st.remote);
            if (vram_trace) {   // who holds the card while decoding (the expert slots are the rest)
                size_t tf = 0, tt = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                double dsp_gib = 0.0;
#if defined(DS41_HAVE_DSPARK)
                if (dsp) dsp_gib = (double) dsp->weight_bytes() / 1073741824.0;
#endif
                std::fprintf(stderr, "vram ledger: %s; DSpark weights %.2f GiB; free %.2f of %.2f GiB\n",
                             dense.vram_ledger().c_str(), dsp_gib, tf / 1073741824.0, tt / 1073741824.0);
            }
            if (const Ds41Dense::RouteMeter rm = dense.route_meter(false); rm.on && rm.slots > 0)
                std::fprintf(stderr, "serve: routing vs exact: %.2f%% of expert picks swapped, %.2f%% of router weight kept (%lld slots)\n",
                             100.0 * (double) rm.swaps / (double) rm.slots, 100.0 * rm.weight_kept, rm.slots);
            if (vram_trace) {   // who holds the card while decoding (the expert slots are the rest)
                size_t tf = 0, tt = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                double dsp_gib = 0.0;
#if defined(DS41_HAVE_DSPARK)
                if (dsp) dsp_gib = (double) dsp->weight_bytes() / 1073741824.0;
#endif
                std::fprintf(stderr, "vram ledger: %s; DSpark weights %.2f GiB; free %.2f of %.2f GiB\n",
                             dense.vram_ledger().c_str(), dsp_gib, tf / 1073741824.0, tt / 1073741824.0);
            }
#if defined(DS41_HAVE_DSPARK)
            if (rq_pass > 0)
                std::fprintf(stderr, "serve: DSpark %ld passes, %.2f tokens/pass (all requests %.2f)\n", rq_pass,
                             1.0 + (double) rq_acc / rq_pass, 1.0 + (double) sv_acc / std::max(1L, sv_pass));
#endif
            each_tier([](Ds41MoeTier & t) { t.rebalance_tick(true); });   // the gap until the next request: retire a large batch for the reader
            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
        }
        for (auto & x : xtier) x->close();
        tier.close();
        // the stdin reader thread (detached) sits in getline holding cin's lock: a normal exit's stream teardown waits
        // on it until the client closes the pipe - leave without the static destructors
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    }

    // ---- --score-cases FILE --score-out FILE: teacher-forced continuations, one model load for many independent cases
    // (e.g. DeepSeek's own API continuations from antirez/ds4's quality fixtures).  Each line
    // of FILE is "prompt ids<TAB>continuation ids" (comma-separated).  Every case runs from position 0 through step() -
    // the decode path with this run's routing (route bias, VRAM cache), as an unrelated serve request does: positional
    // caches are simply overwritten.  For each continuation token the output row is: case, index, token, its log-prob,
    // our argmax and its log-prob, and our top 20 as id:logprob.
    if (!a.score_cases.empty()) {
        std::ifstream cf(a.score_cases);
        std::ofstream of(a.score_out.empty() ? a.score_cases + ".out.tsv" : a.score_out);
        if (!cf || !of) { std::fprintf(stderr, "ds41_generate: --score-cases: cannot open the case / output file\n"); return 2; }
        if (defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: --score-cases: %s\n", err.c_str()); return 1; }
        std::vector<float> lg((size_t) nv);
        std::vector<int32_t> idx((size_t) nv);
        std::string line;
        int ci = 0;
        double nll = 0;
        long ntok = 0;
        const double ts0 = now_ms();
        // STRATA_BIAS_REFRESH=N here: the route bias rebuilt from the live VRAM set every N decode steps - the scored
        // path is the decode the serve refresh applies to, so the quality gate measures what decoding gets
        static const int score_refresh = std::getenv("STRATA_BIAS_REFRESH") ? std::max(0, std::atoi(std::getenv("STRATA_BIAS_REFRESH"))) : 0;
        long score_steps = 0;
        auto refresh_bias = [&]() {
            if (score_refresh > 0 && bias_now && bias_set && ++score_steps % score_refresh == 0) apply_bias(true);
        };
        // STRATA_SCORE_LIKE_SERVE=1 (default off: the scores stay comparable with every earlier round): score what the server
        // runs.  The prompt routes exactly (the server reads a prompt as a layer-major batch, or as exact passes with the
        // launcher's STRATA_SERVE_PASS_BIAS=0) and the decode bias starts with the continuation; STRATA_ENC_BOUND's floor
        // sits where feed() puts it (rounded down to a sub-batch, so 128-639 tokens back, not exactly N; a batch narrowed by
        // a tight card can move it in the server); STRATA_ANSWER_ROUTE_BIAS applies to a prompt that ends its thinking
        // (the fixtures' non-thinking template does).  Off: the prompt routes with the decode bias, the floor is exactly N.
        static const bool like_serve = std::getenv("STRATA_SCORE_LIKE_SERVE") && std::getenv("STRATA_SCORE_LIKE_SERVE")[0] == '1';
        static const float sc_ans_rb = std::getenv("STRATA_ANSWER_ROUTE_BIAS") ? std::max(0.0f, (float) std::atof(std::getenv("STRATA_ANSWER_ROUTE_BIAS"))) : -1.0f;
        static const float sc_ans_rbf = std::getenv("STRATA_ANSWER_ROUTE_BIAS_FILE") ? std::max(0.0f, (float) std::atof(std::getenv("STRATA_ANSWER_ROUTE_BIAS_FILE"))) : sc_ans_rb;
        static const int sc_beg = std::getenv("STRATA_THINK_BEGIN_ID") ? std::atoi(std::getenv("STRATA_THINK_BEGIN_ID")) : 128821;
        static const int sc_end = std::getenv("STRATA_THINK_END_ID") ? std::atoi(std::getenv("STRATA_THINK_END_ID")) : 128822;
        const float cfg_rb = a.route_bias, cfg_rbf = a.route_bias_file;
        if (like_serve) std::fprintf(stderr, "score: STRATA_SCORE_LIKE_SERVE - exact prompt routing, the server's ENC_BOUND floor%s\n",
                                     sc_ans_rb >= 0.0f ? ", answer-phase bias" : "");
        while (std::getline(cf, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            const std::vector<int> pr = parse_csv(line.substr(0, tab));
            const std::vector<int> tg = parse_csv(line.substr(tab + 1));
            if (pr.empty() || tg.empty()) { ++ci; continue; }
            if ((int64_t) (pr.size() + tg.size()) + 1 >= ctx) { std::fprintf(stderr, "case %d exceeds the context\n", ci); ++ci; continue; }
            int pos = 0;
            if (enc_bound > 0) {
                const int64_t np = (int64_t) pr.size();
                int64_t fl = np > enc_bound ? np - enc_bound : -1;
                if (like_serve) { const int64_t c = (np - enc_bound) / SB * SB; fl = np > enc_bound && c > 0 ? c : -1; }
                dense.set_window_floor(fl);
            }
            const bool sc_bias = bias_now;   // the decode bias of this run (open_cache / the start)
            if (like_serve && bias_now) { bias_now = false; apply_bias(); }   // the prompt: exact routing (prune mask only)
            for (int t : pr) {   // the prompt, token by token (its last step's logits predict the first continuation token)
                refresh_bias();
                if (!step(t, pos++)) { std::fprintf(stderr, "case %d: %s\n", ci, ferr.c_str()); return 1; }
            }
            if (like_serve) {   // the continuation decodes: the run's bias, or the answer phase's for a prompt past its thinking
                long lb = -1, le = -1;
                for (size_t q = 0; q < pr.size(); ++q) { if (pr[q] == sc_beg) lb = (long) q; else if (pr[q] == sc_end) le = (long) q; }
                const bool ans = sc_ans_rb >= 0.0f && le > lb;
                a.route_bias = ans ? sc_ans_rb : cfg_rb;
                a.route_bias_file = ans ? sc_ans_rbf : cfg_rbf;
                bias_now = sc_bias;
                apply_bias();
            }
            for (size_t i = 0; i < tg.size(); ++i) {
                if (!dense.logits(0, 1, lg.data())) { std::fprintf(stderr, "case %d: %s\n", ci, dense.last_error().c_str()); return 1; }
                const float mx = *std::max_element(lg.begin(), lg.end());
                double se = 0;
                for (int v = 0; v < nv; ++v) se += std::exp((double) lg[(size_t) v] - (double) mx);
                const double lse = (double) mx + std::log(se);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int32_t x, int32_t y) { return lg[(size_t) x] > lg[(size_t) y]; });
                const int tok = tg[i];
                const double lp = (double) lg[(size_t) tok] - lse;
                nll -= lp;
                ++ntok;
                of << ci << '\t' << i << '\t' << tok << '\t' << lp << '\t' << idx[0] << '\t' << ((double) lg[(size_t) idx[0]] - lse) << '\t';
                for (int k = 0; k < 20; ++k) of << (k ? "," : "") << idx[(size_t) k] << ':' << ((double) lg[(size_t) idx[(size_t) k]] - lse);
                of << '\n';
                if (i + 1 < tg.size()) refresh_bias();
                if (i + 1 < tg.size() && !step(tok, pos++)) { std::fprintf(stderr, "case %d: %s\n", ci, ferr.c_str()); return 1; }
            }
            ++ci;
            if (!a.quiet) std::fprintf(stderr, "\rscored %d cases", ci);
        }
        of.close();
        std::printf("score: %d cases, %ld tokens, mean NLL %.6f, %.1f s\n", ci, ntok, ntok ? nll / ntok : 0.0, (now_ms() - ts0) / 1000.0);
        return 0;
    }

    // ---- one-shot: prompt (+ perplexity), then generate
    if (prompt.empty()) { std::fprintf(stderr, "ds41_generate: empty prompt\n"); return 2; }
    host_pin.pin();   // STRATA_DS41_HOST_PIN (off by default)
    each_tier([](Ds41MoeTier & t) { t.reset_stats(); });
    const double tp0 = now_ms();
    std::vector<float> all;
    const int dm_tail = dump_main ? std::min<int>(128, (int) prompt.size() - 1) : 0;   // DSpark's window, as steps
    bool spec = dsp_path != nullptr;
#if !defined(DS41_HAVE_DSPARK)
    if (spec) { std::fprintf(stderr, "ds41_generate: STRATA_DSPARK: built without ds41_dspark\n"); return 2; }
#endif
    const int spec_depth = std::getenv("STRATA_DSPARK_DEPTH") ? std::atoi(std::getenv("STRATA_DSPARK_DEPTH")) : 4;
    if (spec && (!defer_vram || S < 17 || dump_main || spec_depth < 0 || spec_depth > 5 || std::getenv("STRATA_FORCE_IDS"))) {
        std::fprintf(stderr, "ds41_generate: STRATA_DSPARK needs the GPU layer-major prompt path (S >= 17), depth 1..5, and no "
                             "STRATA_DUMP_MAIN / STRATA_FORCE_IDS\n");
        return 2;
    }
    if (spec && a.temp > 0) std::fprintf(stderr, "note: STRATA_DSPARK decodes greedily (temp ignored)\n");
    // the DSpark window needs per-position main_hidden, which the layer-major prompt does not produce: the last
    // min(128, L-1) prompt tokens run through pass_v (chunks of <= 16) after the VRAM cache opens
    const int sp_tail = spec ? std::min<int>(128, (int) prompt.size() - 1) : 0;
    // --ppl-step M (with --ppl): positions 1.. are scored through the DECODE machinery instead of the prompt path - M = 1
    // one step() per token (gpu_run, the decode tier), M > 1 pass_v over M tokens at a time (gpu_run_n_ex, DSpark's
    // verify pass), teacher-forced - so the knobs that live only there (STRATA_DEFER, STRATA_DEFER_MULTI) reach the KL gate
    const int ppl_tail = (a.ppl && a.ppl_step > 0) ? (int) prompt.size() - 1 : 0;
    if (ppl_tail > 0 && (spec || dm_tail > 0)) {
        std::fprintf(stderr, "ds41_generate: --ppl-step: not with STRATA_DSPARK / STRATA_DUMP_MAIN\n");
        return 2;
    }
    // STRATA_PROFILE_PREFILL=1: nsys --capture-range=cudaProfilerApi captures exactly the one-shot prompt's batch feed
    static const bool prof_prefill = std::getenv("STRATA_PROFILE_PREFILL") != nullptr;
    if (prof_prefill) Ds41MoeTier::profiler(true);
    const bool ok = feed(prompt.data(), (int) prompt.size() - dm_tail - sp_tail - ppl_tail, 0, [&](int done) {
        if (!a.quiet) std::fprintf(stderr, "\rprompt %d/%zu", done, prompt.size());
    }, a.ppl || !a.dump_logits.empty() ? &all : nullptr);
    if (prof_prefill) Ds41MoeTier::profiler(false);
    if (!ok) { std::fprintf(stderr, "\nds41_generate: prefill: %s\n", ferr.c_str()); return 1; }
    if (dm_tail > 0 && defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
    for (int i = (int) prompt.size() - dm_tail; i < (int) prompt.size(); ++i)
        if (!step(prompt[(size_t) i], i) || !dense.logits(0, 1, last_logits.data())) {
            std::fprintf(stderr, "ds41_generate: dump-main prompt step: %s\n", ferr.c_str());
            return 1;
        }
    if (ppl_tail > 0) {
        if (defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        for (int i0 = (int) prompt.size() - ppl_tail; i0 < (int) prompt.size();) {
            const int m = std::min<int>(a.ppl_step, (int) prompt.size() - i0);
            const size_t at = all.size();
            all.resize(at + (size_t) m * (size_t) nv);
            if (a.ppl_step == 1) {
                if (!step(prompt[(size_t) i0], i0) || !dense.logits(0, 1, all.data() + at)) {
                    std::fprintf(stderr, "\nds41_generate: --ppl-step: %s\n", ferr.c_str());
                    return 1;
                }
            } else {
                if (!pass_v(prompt.data() + i0, m, i0)) { std::fprintf(stderr, "\nds41_generate: --ppl-step: %s\n", ferr.c_str()); return 1; }
                std::copy(vlog.begin(), vlog.begin() + (ptrdiff_t) ((size_t) m * (size_t) nv), all.begin() + (ptrdiff_t) at);
            }
            i0 += m;
            if (!a.quiet && (i0 % 256 < m || i0 == (int) prompt.size())) std::fprintf(stderr, "\rscored %d/%zu", i0, prompt.size());
        }
        std::copy(all.end() - nv, all.end(), last_logits.begin());
    }
#if defined(DS41_HAVE_DSPARK)
    int sp_bonus = -1;   // the trunk's token for position L (argmax of the last prompt row) and DSpark's first drafts
    int sp_ids[5] = {0}, sp_tmp[5];
    float sp_conf[5];
    if (spec) {
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        bias_now = false;   // the tail is PROMPT: it routes like the rest of the prompt (no decode biases), as plain decode
        apply_bias();
        std::vector<int> spos;
        for (int i0 = (int) prompt.size() - sp_tail; i0 < (int) prompt.size(); i0 += 16) {
            const int m = std::min<int>(16, (int) prompt.size() - i0);
            if (!pass_v(prompt.data() + i0, m, i0)) { std::fprintf(stderr, "ds41_generate: DSpark seed: %s\n", ferr.c_str()); return 1; }
            spos.resize((size_t) m);
            for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
            const bool last = i0 + m == (int) prompt.size();
            if (last) {
                std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                sp_bonus = (int) (std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
            }
            if (!dsp->step(vh.data(), spos.data(), m, last ? sp_bonus : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark seed step: %s\n", dsp->last_error().c_str());
                return 1;
            }
        }
        bias_now = true;    // decode: route bias + file penalty on (open_cache's state)
        apply_bias();
    }
#endif
    const double prompt_ms = now_ms() - tp0;
    if (defer_vram && dm_tail == 0 && !spec) {   // the prompt is done: its buffers go back, the VRAM expert cache opens into the free space
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "\nVRAM after the prompt: %.2f GiB free -> %.2f GiB of expert slots: %lld experts in VRAM (%.1f s)",
                     seed_free, seed_gib, (long long) tier.resident(), seed_s);
    }
    const Ds41MoeStats pst = stats_all();
    std::fprintf(stderr, "\nprompt: %zu tokens in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / %lld pcie / %lld file)\n",
                 prompt.size(), prompt_ms / 1000.0, 1000.0 * prompt.size() / prompt_ms, (long long) pst.hits,
                 (long long) pst.cpu, (long long) pst.pcie, (long long) pst.file_tier);
    std::fprintf(stderr, "prompt split: dense %.1f s, experts %.1f s (file wait %.1f s), logits %.1f s\n", fd_dense / 1000.0,
                 fd_tier / 1000.0, pst.file_ms / 1000.0, fd_logits / 1000.0);
    if (a.ppl) {
        double nll = 0;
        int n = 0, top1 = 0;
        for (size_t t = 0; t + 1 < prompt.size(); ++t) {
            const float * lg = all.data() + t * (size_t) nv;
            nll -= log_softmax_at(lg, nv, prompt[t + 1]);
            top1 += (int) (std::max_element(lg, lg + nv) - lg) == prompt[t + 1];
            ++n;
        }
        std::printf("PPL %.4f  mean NLL %.4f over %d tokens, top-1 %.3f\n", std::exp(nll / std::max(1, n)), nll / std::max(1, n), n,
                    (double) top1 / std::max(1, n));
    }
    // --dump-topk FILE (with --ppl): per scored position the log-sum-exp and the top 64 (id, logit) - a reference
    // (exact routing, Q8_0 dense) for --kl-ref.  --kl-ref FILE (with --ppl, the same ids): KL(reference || this run) per
    // position over the reference's top 64 plus the remaining mass as one bucket, its mean, and top-1 agreement with
    // the reference.  A fidelity score for lossy knobs: PPL on text this engine generated rewards agreeing with its own
    // earlier outputs (dropping experts scored better than exact routing), not with the model.
    if (a.ppl && (!a.dump_topk.empty() || !a.kl_ref.empty())) {
        constexpr int KT = 64;
        const size_t np = prompt.size() > 0 ? prompt.size() - 1 : 0;
        auto lse_of = [&](const float * lg) {
            const float m = *std::max_element(lg, lg + nv);
            double s = 0;
            for (int64_t i = 0; i < (int64_t) nv; ++i) s += std::exp((double) lg[i] - (double) m);
            return (double) m + std::log(s);
        };
        if (!a.dump_topk.empty()) {
            std::ofstream f(a.dump_topk, std::ios::binary);
            std::vector<int32_t> idx((size_t) nv);
            for (size_t t = 0; t < np; ++t) {
                const float * lg = all.data() + t * (size_t) nv;
                const float lse = (float) lse_of(lg);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + KT, idx.end(), [&](int32_t x, int32_t y) { return lg[x] > lg[y]; });
                f.write((const char *) &lse, 4);
                for (int k = 0; k < KT; ++k) {
                    f.write((const char *) &idx[(size_t) k], 4);
                    f.write((const char *) &lg[idx[(size_t) k]], 4);
                }
            }
            std::printf("top-%d reference written for %zu positions: %s\n", KT, np, a.dump_topk.c_str());
        }
        if (!a.kl_ref.empty()) {
            std::ifstream f(a.kl_ref, std::ios::binary);
            double kl_sum = 0, kl_max = 0;
            int n = 0, agree = 0;
            for (size_t t = 0; t < np; ++t) {
                float lse_r = 0;
                int32_t id[KT];
                float lr[KT];
                if (!f.read((char *) &lse_r, 4)) break;
                for (int k = 0; k < KT; ++k) { f.read((char *) &id[k], 4); f.read((char *) &lr[k], 4); }
                const float * lg = all.data() + t * (size_t) nv;
                const double lse_t = lse_of(lg);
                double kl = 0, pr_in = 0, pt_in = 0;
                for (int k = 0; k < KT; ++k) {
                    const double lpr = (double) lr[k] - (double) lse_r, lpt = (double) lg[id[k]] - lse_t;
                    const double pr = std::exp(lpr);
                    kl += pr * (lpr - lpt);
                    pr_in += pr;
                    pt_in += std::exp(lpt);
                }
                const double pr_out = std::max(0.0, 1.0 - pr_in), pt_out = std::max(1e-12, 1.0 - pt_in);
                if (pr_out > 1e-12) kl += pr_out * (std::log(pr_out) - std::log(pt_out));
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
                agree += (int) (std::max_element(lg, lg + nv) - lg) == id[0];
                ++n;
            }
            std::printf("KL vs reference %.5f (max %.3f)  top-1 agreement %.4f over %d positions\n", kl_sum / std::max(1, n), kl_max,
                        (double) agree / std::max(1, n), n);
        }
    }
    if (!a.dump_logits.empty()) {
        std::ofstream f(a.dump_logits, std::ios::binary);
        f.write((const char *) all.data(), (std::streamsize) (all.size() * 4));
    }
    if (!a.dump_last.empty()) {   // the last row's logits - the one row enc-skip still computes exactly
        std::ofstream f(a.dump_last, std::ios::binary);
        f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
    }
    each_tier([](Ds41MoeTier & t) { t.reset_stats(); });
    // STRATA_PROFILE_DECODE=1: nsys --capture-range=cudaProfilerApi captures exactly the decode
    static const bool prof_decode = std::getenv("STRATA_PROFILE_DECODE") != nullptr;
    // STRATA_PROFILE_FROM=K (DSpark decode): the capture starts at the K-th generated token instead, past the first
    // passes' graph building, so the profile shows the steady state
    static const int prof_from = std::getenv("STRATA_PROFILE_FROM") ? std::atoi(std::getenv("STRATA_PROFILE_FROM")) : 0;
    if (prof_decode && prof_from <= 0) Ds41MoeTier::profiler(true);
    const double td0 = now_ms();
    std::vector<float> lg = last_logits;
    int pos = (int) prompt.size(), gen = 0;
    // STRATA_FORCE_IDS=<file>: decode this token sequence (one id per line - a previous run's stdout) in place of the
    // sampled tokens, so kernel A/Bs whose variants round differently still do the same work (greedy decoding flips on
    // near-ties, and a different text means different experts and a different VRAM hit rate)
    std::vector<int> forced;
    if (const char * fp = std::getenv("STRATA_FORCE_IDS")) {   // lines that are not a bare id (a memguard banner) skipped
        std::ifstream f(fp);
        for (std::string line; std::getline(f, line);) {
            char * end = nullptr;
            const long t = std::strtol(line.c_str(), &end, 10);
            if (end != line.c_str() && *end == '\0') forced.push_back((int) t);
        }
        std::fprintf(stderr, "STRATA_FORCE_IDS: %zu forced tokens from %s\n", forced.size(), fp);
    }
#if defined(DS41_HAVE_DSPARK)
    // ---- DSpark: P = the last fed position, t = the trunk's token for P+1 (not fed yet), d = drafts for P+2..
    // A pass verifies [t, d0..d(k-1)] at P+1..P+1+k; g_i = the argmax of row P+1+i (the trunk's token for P+2+i);
    // a = #leading d_i == g_i; d0..d(a-1) + t' = g_a are emitted, DSpark appends the a+1 accepted rows and drafts
    // after t'.  Every cache is position-indexed (rejected rows are overwritten when re-fed; the window ring is sized
    // for it), except the ratio-2 compressor's 2-slot state ring: a pass writes its LAST 2 tokens' states, so after a
    // rejection a pass that starts at an odd position re-feeds the last accepted token (its group's state then comes
    // from inside the pass).
    // STRATA_ROUTE_TRACE=<file> (measurement only): the verify passes' routing + acceptance, for
    // tools/v41/sim/route_sim.py (session 13: in-pass draft pruning and the VRAM replacement policy, replayed offline).
    // Opened after the seed; this is the decode-start snapshot (the prompt may have re-seeded the cache)
    set_run_k(dec_k, true);   // STRATA_EXPERTS_USED_DECODE: the decode / verify passes from here route dec_k
    tier.trace_snapshot();
    long sp_pass = 0, sp_acc = 0, sp_refeed = 0, sp_full = 0;
    const bool sp_noacc = std::getenv("STRATA_DSPARK_NOACC") != nullptr;
    double sp_vms = 0, sp_dms = 0;
    if (spec) {
        int P = (int) prompt.size() - 1, t = sp_bonus, prev = prompt.back();
        bool dirty = false;   // the last pass rejected something
        std::vector<int> vt;
        auto emit = [&](int tok) -> bool {   // false = stop here
            std::printf("%d\n", tok);
            std::fflush(stdout);
            ++gen;
            if (prof_decode && prof_from > 0 && gen == prof_from) Ds41MoeTier::profiler(true);
            return !(std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos || gen >= a.n_predict);
        };
        // STRATA_DSPARK_CONF=<tau>: verify only the leading drafts whose confidence-head probability (sigmoid of
        // conf) is >= tau - a draft that will be rejected still costs its experts' PCIe / NVMe bytes in the verify
        // pass, while every emitted token is still the trunk's own (lossless: only the verify width changes).
        // STRATA_DSPARK_CONFLOG=<file>: per pass "k acc verify_ms p0 p1 p2 p3 p4" (the offline threshold screen).
        static const float conf_tau = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
        static FILE * conf_log = std::getenv("STRATA_DSPARK_CONFLOG") ? std::fopen(std::getenv("STRATA_DSPARK_CONFLOG"), "w") : nullptr;
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        // STRATA_DSPARK_OPT=1 (overrides CONF): each pass verifies the k that maximises expected tokens per ms -
        // E(k) = 1 + sum_{j<=k} prod_{i<=j} p_i (p_i the confidence head's sigmoid) over the pass's cost at that width,
        // learned online (EMA per width; prior 40 + 18.5 (w - 1) ms, measured on the 3090).  DeepSeek's production
        // scheduler does this across a batch with a profiled throughput curve; this is the single-stream version.
        // Lossless like CONF: only the verify width changes.
        static const bool dsp_opt = std::getenv("STRATA_DSPARK_OPT") && std::getenv("STRATA_DSPARK_OPT")[0] == '1';
        double cost_w[16];
        for (int w = 0; w < 16; ++w) cost_w[w] = 40.0 + 18.5 * (double) std::max(0, w - 1);
        while (emit(t)) {
            int k = std::min(spec_depth, a.n_predict - gen);   // drafts past the budget are never emitted
            if (dsp_opt) {
                const int extra = (dirty && ((P + 1) & 1)) ? 1 : 0;   // a refeed row widens the pass
                double best = -1.0, e = 1.0, run = 1.0;
                int bk = 0;
                for (int kk = 0; kk <= k; ++kk) {
                    if (kk > 0) { run *= sig(sp_conf[kk - 1]); e += run; }
                    const double r = e / cost_w[std::min(15, kk + 1 + extra)];
                    if (r > best) { best = r; bk = kk; }
                }
                k = bk;
            } else if (conf_tau >= 0) {
                int kk = 0;
                while (kk < k && sig(sp_conf[kk]) >= conf_tau) ++kk;
                k = kk;
            }
            // STRATA_DSPARK_CUM=<c>: as the serve loop - stop once the product of the confidences falls below c
            static const float cum = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
            if (cum > 0) {
                int kk = 0;
                float pr = 1.0f;
                while (kk < k && (pr *= sig(sp_conf[kk])) >= cum) ++kk;
                k = kk;
            }
            {   // draft-id bounds guard (Maya 075cbe6), as the serve loop: an id -1 (no finite logit) ends the draft
                int kk = 0;
                while (kk < k && sp_ids[kk] >= 0) ++kk;
                k = kk;
            }
            const bool refeed = dirty && ((P + 1) & 1);
            vt.clear();
            if (refeed) vt.push_back(prev);
            vt.push_back(t);
            for (int i = 0; i < k; ++i) vt.push_back(sp_ids[i]);
            const int r0 = refeed ? 1 : 0, p0 = refeed ? P : P + 1;
            const double tv = now_ms();
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { std::fprintf(stderr, "ds41_generate: verify: %s\n", ferr.c_str()); return 1; }
            sp_vms += now_ms() - tv;
            if (dsp_opt) {   // learn this width's cost
                const int w = std::min(15, (int) vt.size());
                cost_w[w] += 0.2 * ((now_ms() - tv) - cost_w[w]);
            }
            // STRATA_DSPARK_DUMP=<file> (correctness tap): every verify row as {pos, m, row, refeed, nv floats} - row
            // `pos` predicts the token at pos+1, so it lines up with row `pos` of a --dump-logits prefill of the stream
            static FILE * sp_dump = std::getenv("STRATA_DSPARK_DUMP") ? std::fopen(std::getenv("STRATA_DSPARK_DUMP"), "wb") : nullptr;
            if (sp_dump) {
                for (int i = 0; i < (int) vt.size(); ++i) {
                    const int32_t hdr[4] = { p0 + i, (int32_t) vt.size(), i, refeed ? 1 : 0 };
                    std::fwrite(hdr, 4, 4, sp_dump);
                    std::fwrite(vlog.data() + (size_t) i * nv, 4, (size_t) nv, sp_dump);
                }
                std::fflush(sp_dump);
            }
            int g[6];
            for (int i = 0; i <= k; ++i) {
                const float * row = vlog.data() + (size_t) (r0 + i) * nv;
                g[i] = (int) (std::max_element(row, row + nv) - row);
            }
            int acc = 0;
            while (acc < k && sp_ids[acc] == g[acc]) ++acc;
            if (sp_noacc) acc = 0;   // bisect tap: every draft rejected (the stream must stay plain greedy)
            ++sp_pass; sp_acc += acc; sp_refeed += refeed; sp_full += acc == k;
            if (conf_log) {
                // acceptance of EVERY draft against the trunk rows this pass computed (beyond k: unknown, -1)
                std::fprintf(conf_log, "%d %d %.2f", k, acc, now_ms() - tv);
                for (int i = 0; i < spec_depth; ++i) std::fprintf(conf_log, " %.4f", sig(sp_conf[i]));
                std::fprintf(conf_log, "\n");
                std::fflush(conf_log);
            }
            {   // STRATA_ROUTE_TRACE: 'P' {m, k, acc, refeed, verify ms, conf p0..p4}
                float pm[10] = { (float) vt.size(), (float) k, (float) acc, (float) refeed, (float) (now_ms() - tv) };
                for (int i = 0; i < 5; ++i) pm[5 + i] = i < spec_depth ? sig(sp_conf[i]) : 0.0f;
                tier.trace_mark(pm, 10);
            }
            int spos[6];
            for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
            const int drafts[5] = { sp_ids[0], sp_ids[1], sp_ids[2], sp_ids[3], sp_ids[4] };
            const double td = now_ms();
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, g[acc], sp_ids, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark step: %s\n", dsp->last_error().c_str());
                return 1;
            }
            sp_dms += now_ms() - td;
            bool more = true;
            for (int i = 0; i < acc && more; ++i) more = emit(drafts[i]);
            if (!more) break;
            prev = acc == 0 ? t : drafts[acc - 1];
            dirty = acc < k;
            P += 1 + acc;
            t = g[acc];
        }
    }
#endif
    for (; !spec && gen < a.n_predict; ++gen) {
        int tok = sample_p(lg.data(), nv, a.temp, 0, 1.0f, 0.0f, rng);
        if (gen < (int) forced.size()) {
            // STRATA_DEC_MARGIN=1 (gate tap, with STRATA_FORCE_IDS): per step the argmax, its logit and the forced
            // token's logit - a forced stream that leaves the greedy path at a near-tie shows a tiny gap there
            static const bool margin = std::getenv("STRATA_DEC_MARGIN") != nullptr;
            if (margin && tok != forced[(size_t) gen])
                std::fprintf(stderr, "margin: step %d argmax %d (%.4f) forced %d (%.4f) gap %.4f\n", gen, tok, lg[(size_t) tok],
                             forced[(size_t) gen], lg[(size_t) forced[(size_t) gen]], lg[(size_t) tok] - lg[(size_t) forced[(size_t) gen]]);
            tok = forced[(size_t) gen];
        }
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos) { ++gen; break; }
        if (gen + 1 == a.n_predict) { ++gen; break; }
        if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { std::fprintf(stderr, "ds41_generate: decode: %s\n", ferr.c_str()); return 1; }
        ++pos;
    }
    const double dec_ms = now_ms() - td0;
    if (prof_decode) Ds41MoeTier::profiler(false);
    const Ds41MoeStats st = stats_all();
    std::fprintf(stderr, "decode: %d tokens in %.2f s = %.2f tok/s; experts %lld hit / %lld cpu / %lld pcie / %lld file "
                 "(%.1f%% VRAM hits), cpu %.0f ms, file %.0f ms; spec reads %lld issued / %lld used; prefetch %lld issued / "
                 "%lld outside the arena / %lld used\n",
                 gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), (long long) st.hits,
                 (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                 100.0 * st.hits / std::max<int64_t>(1, st.lookups()), st.cpu_ms, st.file_ms,
                 (long long) st.spec_issued, (long long) st.spec_hits, (long long) st.prefetch_issued,
                 (long long) st.prefetch_dummy, (long long) st.prefetched_useful);
    if (const Ds41ReadFallbacks fb = ds41_read_fallbacks(); fb.total() > 0)   // B1: process totals, only when any
        std::fprintf(stderr, "decode: read fallbacks: %lld O_DIRECT -> buffered, %lld pread -> mapping, %lld pinned -> pageable\n",
                     (long long) fb.direct_to_buffered, (long long) fb.pread_to_mmap, (long long) fb.pinned_to_pageable);
    if (st.remote > 0)   // STRATA_DS41_REMOTE: the helper's card (not counted as VRAM hits above)
        std::fprintf(stderr, "decode: experts on the helper's card %lld (%.1f%% of the lookups)\n", (long long) st.remote,
                     100.0 * st.remote / std::max<int64_t>(1, st.lookups()));
    if (spec)
        std::fprintf(stderr, "DSpark: %ld verify passes, %.3f accepted drafts/pass = %.3f tokens/pass, %.1f%% full, %ld re-fed; "
                     "verify %.1f ms/pass, draft %.2f ms/pass (depth %d)\n", sp_pass, (double) sp_acc / std::max(1L, sp_pass),
                     1.0 + (double) sp_acc / std::max(1L, sp_pass), 100.0 * sp_full / std::max(1L, sp_pass), sp_refeed,
                     sp_vms / std::max(1L, sp_pass), sp_dms / std::max(1L, sp_pass), spec_depth);
    if (spec)
        std::fprintf(stderr, "pass_v split (all %ld passes, %.2f tokens each): dense %.1f ms, tier %.1f ms, taps %.1f ms, "
                     "logits %.1f ms per pass\n", pv_n, (double) pv_tok / std::max(1L, pv_n), pv_ms[0] / std::max(1L, pv_n),
                     pv_ms[1] / std::max(1L, pv_n), pv_ms[2] / std::max(1L, pv_n), pv_ms[3] / std::max(1L, pv_n));
    if (spec && multi)
        std::fprintf(stderr, "pass_v split: the other cards' layers %.1f ms of the tier's per pass\n", pv_rm / std::max(1L, pv_n));
    km_report("end");   // STRATA_EXPERTS_DROP_MISS_STATS
    if (dump_main) std::fclose(dump_main);
    if (file_pred) {
        std::fprintf(stderr, "file-pred: %lld file-tier routes", fp_miss[0]);
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " | k=%d: %lld scored, in top4 %.1f%% top8 %.1f%% top16 %.1f%%", k, fp_miss[k],
                         100.0 * fp_hit[k][0] / std::max(1LL, fp_miss[k]), 100.0 * fp_hit[k][1] / std::max(1LL, fp_miss[k]),
                         100.0 * fp_hit[k][2] / std::max(1LL, fp_miss[k]));
        std::fprintf(stderr, "\nfile-pred candidates (file-tier experts in the top 4/8/16, all layers):");
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " k=%d %lld/%lld/%lld", k, fp_cand[k][0], fp_cand[k][1], fp_cand[k][2]);
        std::fprintf(stderr, "\n");
    }
    for (auto & x : xtier) x->close();
    tier.close();
    return 0;
}
