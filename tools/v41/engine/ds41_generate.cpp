// tools/v41/engine/ds41_generate.cpp - the Strata DeepSeek-V4.1-Flash engine end to end: Ds41Dense (attention,
// Engram, router, shared expert; CUDA or CPU ggml backend) + Ds41MoeTier (routed experts with per-expert formats:
// VRAM cache, pinned RAM arena, file tier, CPU pool, PCIe share) + sampling.  Token ids in, token ids out (text is
// the server's job: tools/v41/serve/serve_ds41.py).
//
//   ds41_generate -m V41-GSQ.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                 [--engram DIR:DIR:DIR | --engram MODEL.gguf] [--engram-hot hot.bin] [--slots auto|N] [--vram-margin GiB] [--arena-gib 60] [--rank r.bin]
//                 [--vram-lru] [--arena-adapt] [--pcie 0.25] [--tier-batch-pcie 1.0] [--prefill-chunk N] [--prompt-batch B] [--chunk-mmq] [--ctx N]
//                 [--temp 0] [--seed 1] [--stop id,id] [--ppl] [--no-actquant] [--kv-fp4] [--serve] [--quiet]
//                 [--enc-skip|--no-enc-skip] [--enc-replay N] [--dump-last F]
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
#if defined(DS41_HAVE_DSPARK)
#include "ds41_dspark.hpp"
#endif

#include "ggml.h"
#include "ggml-backend.h"
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
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, rank, dump_logits, dump_last, dump_topk, kl_ref;
    std::string score_cases, score_out;   // --score-cases / --score-out: teacher-forced continuations (official_score.py)
    std::string build_mirror;      // --build-mirror F [--mirror-gib N]: write the blob mirror (coldest experts by --rank) and exit
    double mirror_gib = 85.0;
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
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--rank") a.rank = next();
        else if (k == "--build-mirror") a.build_mirror = next();
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
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty() || !a.build_mirror.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best); temp 0 = argmax
int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
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
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
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
#if defined(_WIN32)
    if (std::getenv("DS41_CRASH_TRACE")) AddVectoredExceptionHandler(1, ds41_veh);
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
    // scheduling (E-cores) whatever the priority class (round 53: one-shots at half speed - CPU pool, drafter and
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
    ggml_backend_t be = a.backend == "cuda" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                                            : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
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
    dc.engram_dirs = a.engram;
    dc.engram_hot = a.engram_hot;
    dc.vision = a.vision;
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
    // enc-skip's layer 20: the LAST kv source (every compressed-KV / index-key write is owned by a kv source, and the
    // layers above it only read that owner's state, so the cut belongs there).  Also the window reach R = swa *
    // (n_layer - cut_layer) is what --enc-replay defaults to.
    int CL = 0;
    for (int64_t s : G.kv_sources) CL = std::max<int>(CL, (int) s);
    // exactness needs every later layer's window reach to stay inside the replay: reach = swa * (n_layer - CL)
    const int64_t reach = (int64_t) G.n_swa * (int64_t) (n_layer - CL);
    if (enc_skip && ER < reach) {
        std::fprintf(stderr, "note: --enc-replay %lld < the window reach %lld (swa %lld x %d layers past %d); raised\n",
                     (long long) ER, (long long) reach, (long long) G.n_swa, n_layer - CL, CL);
        ER = reach;
    }
    if (enc_skip) std::fprintf(stderr, "enc-skip: cut at layer %d (last kv source), replay window %lld (reach %lld)\n", CL, (long long) ER, (long long) reach);
    if (a.engram.empty()) std::fprintf(stderr, "WARNING: no --engram (the shard folder, or a GGUF with the tables inline): the Engram add is skipped (not the real model)\n");

    // ---- routed experts
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    double slot_gib = 0;
    if (a.experts != "cpu" && !ggml_backend_is_cpu(be)) {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        slot_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        std::fprintf(stderr, "VRAM: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f)\n",
                     fr / 1073741824.0, slot_gib, a.vram_margin_gib);
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
    if (!tier.init(TG, &src, mc, err)) { std::fprintf(stderr, "ds41_generate: tier init: %s\n", err.c_str()); return 1; }
    dense.set_oom_hook([&]() { return tier.release_prestage(); });   // a prompt pass that does not fit: prestage gives way
    // defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the cache opens into
    // it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots.  --serve does this
    // per request: a multi-token prompt closes the cache (release_cache), the decode after it re-seeds (~0.6 s)
    const bool defer_vram = !mc.cpu_only && (a.prefill_chunk > 1 || a.prompt_batch > 1);
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
        if (defer_vram ? !tier.build_arena_ranked(ranked, err) : !tier.seed_from_ranked(ranked, err)) {
            std::fprintf(stderr, "ds41_generate: seed: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "tier: %lld experts in VRAM, %lld in the RAM arena (%.1f GiB), %lld file tier; seeded in %.1f s\n",
                     (long long) tier.resident(), (long long) tier.arena_experts(), tier.arena_gib(), (long long) tier.file_tier(),
                     (now_ms() - t0) / 1000.0);
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
        return proxy_res.empty() ? tier.resident(l, e) : proxy_res[(size_t) (l * TG.n_experts + e)] != 0;
    };
    // bias_now: route-bias / file penalty are DECODE levers - before the VRAM seed (the prompt) only the prune part
    // applies, unless --bias-prompt measures them (the prompt's file tier is prefetched; a penalty there buys nothing)
    bool bias_now = a.bias_prompt > 0 || !defer_vram;
    bool bias_set = false;   // a nonzero bias is in the dense half now (--serve turns it off again for the next prompt)
    std::vector<std::vector<float>> rb_up((size_t) TG.n_layers);   // the bias each layer has on the card (changed_only)
    // changed_only (STRATA_BIAS_REFRESH): upload just the layers whose bias differs from what the card has - between two
    // verify passes only a few experts change residency, and every set_route_bias is a synchronous upload
    auto apply_bias = [&](bool changed_only = false) {
        const bool want = n_pruned > 0 || (bias_now && (a.route_bias != 0.0f || a.route_bias_file != 0.0f));
        if (!want && !bias_set) return;
        bias_set = want;
        std::vector<float> rb((size_t) TG.n_experts);
        for (int64_t l = 0; l < TG.n_layers; ++l) {
            for (int64_t e = 0; e < TG.n_experts; ++e)
                rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) (l * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (bias_now && a.route_bias != 0.0f && !mc.cpu_only && resident_now(l, e)) ? a.route_bias
                               : (bias_now && a.route_bias_file != 0.0f && !resident_now(l, e) && !tier.in_arena(l, e)) ? -a.route_bias_file : 0.0f;
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
    // scored offline by bench/win3090/pred_score.py (per-layer precision, budgets, temporal / draft-row ensembles).
    std::FILE * plog = std::getenv("STRATA_PRED_LOG") ? std::fopen(std::getenv("STRATA_PRED_LOG"), "ab") : nullptr;
    std::vector<int16_t> pl_buf;
    std::vector<int> pl_pred;    // the prediction handed to this layer's prefetch (pv_pred is reused by the spec reads)
    bool pl_have = false;
    // STRATA_PRED_LOG2=F + STRATA_PRED_AHEAD=k (default 1; measurement only): the same records, but `pred` is layer l's
    // routing predicted k layers EARLIER (at the start of layer l-k, from the state entering it) - what a prefetch
    // issued k layers ahead would get: its copies would have k more layers of time to land (round 114: a 2nd / 3rd copy
    // started at layer l's own attention lands too late).  Decode steps (predict) and verify passes (predict_multi).
    std::FILE * plog2 = std::getenv("STRATA_PRED_LOG2") ? std::fopen(std::getenv("STRATA_PRED_LOG2"), "ab") : nullptr;
    const int pred_ahead = std::getenv("STRATA_PRED_AHEAD") ? std::max(1, std::atoi(std::getenv("STRATA_PRED_AHEAD"))) : 1;
    std::vector<std::vector<int>> pl_ahead((size_t) n_layer);   // [layer]: its k-ahead prediction (16 per pass row)
    auto plog_rec = [&](std::FILE * plf, int l, int m, int pos, const int * pred, const int * act) {
        auto tier_of = [&](int e) -> int16_t { return (int16_t) (e < 0 ? -1 : tier.resident(l, e) ? 0 : tier.in_arena(l, e) ? 1 : 2); };
        pl_buf.assign({ (int16_t) 'R', (int16_t) l, (int16_t) m, (int16_t) top_k, (int16_t) (pos & 0x7fff), (int16_t) (pos >> 15) });
        for (int i = 0; i < m * 16; ++i) pl_buf.push_back((int16_t) (pred ? pred[i] : -1));
        for (int i = 0; i < m * top_k; ++i) pl_buf.push_back((int16_t) act[i]);
        for (int i = 0; i < m * 16; ++i) pl_buf.push_back(pred ? tier_of(pred[i]) : (int16_t) -1);
        for (int i = 0; i < m * top_k; ++i) pl_buf.push_back(tier_of(act[i]));
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
                            if (pk[i] >= 0 && !tier.resident(l + k, pk[i]) && !tier.in_arena(l + k, pk[i]))
                                for (int c = 0; c < 3; ++c) fp_cand[k][c] += i < (4 << c);
                }
            }
            if (advise_ahead > 0 && l + advise_ahead < n_layer) {   // file-tier readahead for layer l+k's predicted experts
                int ta[16];
                if (dense.predict(l + advise_ahead, ta, 16)) tier.advise_file(l + advise_ahead, ta, advise_top);
            }
            const float * fn = nullptr;
            ggml_tensor * fn_t = nullptr;   // the same activation on the card: the tier quantizes it there
            bool attn_done = false;
            if (pred_pending) {   // STRATA_PF_OVERLAP: this layer's attention starts first, then the prefetch is issued
                if (!dense.attn_router_launch(l)) { ferr = dense.last_error(); return false; }
                static const bool pf_spin = std::getenv("STRATA_PF_SPIN") && std::getenv("STRATA_PF_SPIN")[0] == '1';
                if (pf_spin) strata::v41::ds41_cuda_event_spin(dense.predict_event());
                if (!(pf_spin ? dense.predict_take(top16, 16) : dense.predict_wait(top16, 16))) { ferr = dense.last_error(); return false; }
                tier.prefetch(l, top16, 16);   // its DMAs overlap the attention already on the card
                pl_have = true;
                pred_pending = false;
                have_pred = false;
                if (dprof) { const double t = now_ms(); dp[1] += t - ta; ta = t; }
                if (!dense.attn_router_collect(l, ids1.data(), w1.data(), &fn, &fn_t)) { ferr = dense.last_error(); return false; }
                attn_done = true;
            } else if (mc.pf_b > 0 && !mc.cpu_only) {   // the DMAs run under this layer's attention
                int t16[16];
                if (have_pred) {                  // predicted by the previous layer's finish_predict (same state)
                    tier.prefetch(l, top16, 16);
                    pl_have = true;
                } else if (dense.predict(l, t16, 16)) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    tier.prefetch(l, top16, 16);
                    pl_have = true;
                }
                have_pred = false;
            }
            if (!attn_done) {
                if (dprof) { const double t = now_ms(); dp[1] += t - ta; ta = t; }
                if (!dense.attn_router(l, ids1.data(), w1.data(), &fn, &fn_t, true)) { ferr = dense.last_error(); return false; }
            }
            if (plog) plog_rec(plog, l, 1, pos, pl_have ? top16 : nullptr, ids1.data());
            if (plog2 && !pl_ahead[(size_t) l].empty()) plog_rec(plog2, l, 1, pos, pl_ahead[(size_t) l].data(), ids1.data());
            pl_have = false;
            if (dprof) { const double t = now_ms(); dp[2] += t - ta; ta = t; }
            if (file_pred) {
                for (int j = 0; j < top_k; ++j) {
                    const int e = ids1[(size_t) j];
                    if (e < 0 || tier.resident(l, e) || tier.in_arena(l, e)) continue;
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
            const bool tok = rdev1 ? tier.run_to_dev(l, ids1.data(), w1.data(), fn, rdev1, fn_t ? fn_t->data : nullptr)
                                   : tier.run(l, ids1.data(), w1.data(), fn, routed1.data());
            if (!tok) { ferr = "tier.run failed"; return false; }
            if (void * ev = tier.take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC: order on the card
            if (dprof) { const double t = now_ms(); dp[3] += t - ta; ta = t; }
            const float * rsum = rdev1 ? nullptr : routed1.data();   // null: already in the routed tensor
            if (fuse_predict && mc.pf_b > 0 && !mc.cpu_only && l + 1 < n_layer) {
                int t16[16];
                if (spec_top > 0 && spec_ahead > 0) {
                    int ahead[16 * 8];
                    const int na = std::min(spec_ahead, 8);
                    if (!dense.finish_predict_ahead(l, rsum, l + 1, t16, 16, na, ahead)) { ferr = dense.last_error(); return false; }
                    if (dprof) { const double t = now_ms(); dp[4] += t - ta; ta = t; }
                    if (spec_l1) tier.spec_read(l + 1, t16, std::min(spec_top, 16));
                    for (int a = 0; a < na; ++a) tier.spec_read(l + 2 + a, ahead + a * 16, std::min(spec_top, 16));
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
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        if (!dense.begin(toks, m, pos)) { ferr = dense.last_error(); return false; }
        for (int l = 0; l < n_layer; ++l) {
            const float * fn = nullptr;
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            for (int t = 0; t < m; ++t) {
                const bool ok = rdev1 ? tier.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                        fn + (size_t) t * n_embd, rdev1 + (size_t) t * n_embd)
                                      : tier.run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                 fn + (size_t) t * n_embd, routed1.data());
                if (!ok) { ferr = "tier.run failed"; return false; }
                if (void * ev = tier.take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC
                if (!rdev1) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
            }
            if (!dense.finish(l, rdev1 ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
        }
        return true;
    };
    // pass_m + DSpark's main_hidden: vh[t * 3D + j * D ..] = the HC-mean of the state leaving layer dm_layers[j] (=
    // entering the DSpark target layer 37 + j) for pass token t (the STRATA_DUMP_MAIN math, token-major read_state);
    // vlog = the pass's m rows of logits.
    std::vector<float> vh, vlog, vx, vpre;
    std::vector<int> pv_pred, pv_rank;                 // D3 prefetch: per-token predictions, the ranked union
    std::vector<int> pv_next, pv_rank_next;            // STRATA_D3_PF_EARLY: the next layer's, made one layer early
    std::vector<std::pair<int, int>> pv_score;
    std::vector<std::pair<int, float>> pv_noq;         // STRATA_D3_PF_NOISYOR: (expert, prod over rows of 1 - p)
    double pv_ms[4] = {0, 0, 0, 0};   // pass_v split: dense (begin/attn_router/finish), tier, taps (read_state), logits
    long pv_n = 0, pv_tok = 0;
    auto pass_v = [&](const int * toks, int m, int pos) -> bool {
        // D3 (the batched per-class verify tier) is the default: correct (bias-off verify-vs-prefill gate, near-ties
        // only) and faster (verify 167.6 -> ~100 ms/pass).  STRATA_TIER_BATCH=0 = the per-token run_to_dev path.
        static const bool no_tier_batch = [] { const char* v = std::getenv("STRATA_TIER_BATCH"); return v && v[0] == '0'; }();
        double tq = now_ms();
        auto lap = [&](int i) { const double t = now_ms(); pv_ms[i] += t - tq; tq = t; };
        ++pv_n; pv_tok += m;
        // STRATA_PASS_LOG=<file>: one line per pass (rows, wall / dense / tier ms, the tier's hits / cpu / pcie / file
        // for the pass) - the cost model behind batched decoding: what a pass costs per pass, per row and per miss
        static std::FILE * pass_log = std::getenv("STRATA_PASS_LOG") ? std::fopen(std::getenv("STRATA_PASS_LOG"), "w") : nullptr;
        const double pl_t0 = tq, pl_d0 = pv_ms[0], pl_t1 = pv_ms[1];
        const Ds41MoeStats pl_s0 = pass_log ? tier.stats() : Ds41MoeStats{};
        struct PassLogLine {
            std::FILE * f; int m; double t0, d0, k0; const double * ms; const Ds41MoeStats * s0; Ds41MoeTier * tr;
            ~PassLogLine() {
                if (!f) return;
                const Ds41MoeStats s = tr->stats();
                std::fprintf(f, "%d %.3f %.3f %.3f %lld %lld %lld %lld\n", m, now_ms() - t0, ms[0] - d0, ms[1] - k0,
                             (long long) (s.hits - s0->hits), (long long) (s.cpu - s0->cpu), (long long) (s.pcie - s0->pcie),
                             (long long) (s.file_tier - s0->file_tier));
                std::fflush(f);
            }
        } pass_log_line { pass_log, m, pl_t0, pl_d0, pl_t1, pv_ms, &pl_s0, &tier };
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
        static const int d3_pf = std::getenv("STRATA_D3_PF") ? std::atoi(std::getenv("STRATA_D3_PF")) : 1;
        // STRATA_D3_PF_EARLY=K (default 0 = off; rounds 117 / 119): K experts of the NEXT layer, ranked as this layer's
        // (noisy-OR when on) from a prediction made one layer early in the same graph (predict_multi2: no extra sync),
        // copied into a second staging right after this layer's tier call has queued its demand copies - they have the
        // rest of this layer and the next attention to land (round 114: a 2nd / 3rd copy started at a layer's own
        // attention lands too late).  The next layer's own STRATA_D3_PF prefetch appends to them.
        static const int d3_pf_early = std::getenv("STRATA_D3_PF_EARLY") ? std::max(0, std::atoi(std::getenv("STRATA_D3_PF_EARLY"))) : 0;
        bool have_next = false;
        constexpr int kD3PfRows = 8;
        static_assert(kD3PfRows <= Ds41MoeTier::kMaxMulti, "the prefetch is consumed by ONE tier call per layer");
        // (STRATA_D3_PF=0 + STRATA_D3_PF_EARLY: the one prefetch copy per layer moved a layer earlier - round 120)
        const bool pf_now = rdev1 && !no_tier_batch && (d3_pf > 0 || d3_pf_early > 0) && m <= kD3PfRows;
        // STRATA_D3_PF_ASYNC=1: the prediction's readback waits on its own event while the layer's attention is already
        // queued behind it (predict_multi_launch, attn_router_launch, predict_multi_wait, attn_router_collect), so the
        // host's ranking and the prefetch DMA overlap the attention instead of preceding it.  The same graphs and
        // predictions; only the DMA starts a little later.
        static const bool pf_async = std::getenv("STRATA_D3_PF_ASYNC") && std::getenv("STRATA_D3_PF_ASYNC")[0] == '1';
        // STRATA_D3_PF_NOISYOR=1 (default off; rounds 111 / 114): rank the union by P(at least one pass row routes it) =
        // 1 - prod over the rows of (1 - p(its rank in that row)), p = how often a row routes its rank-r prediction
        // (round 111, the launcher's own log: 95 89 80 69 57 44 32 23 17 13 10 8 7 5 5 4 %), instead of the summed
        // 16 - rank, which puts rank 5 in four rows (~44% each) ahead of rank 0 in one (95%).  Held out: prefetch
        // precision 61% -> 77% at the same one slot.  STRATA_D3_PF_P=x: only candidates with P >= x go to the prefetch
        // (then STRATA_D3_PF can be raised: P >= 0.8 x 2 slots = 82%).  STRATA_D3_PF_PTAB=F: per-layer p rows, 16 per
        // line, '#' comments (bench/win3090/pred_noisyor.py --write-table) instead of the pooled ones.
        static const bool pf_noisyor = std::getenv("STRATA_D3_PF_NOISYOR") && std::getenv("STRATA_D3_PF_NOISYOR")[0] == '1';
        static const float pf_pmin = std::getenv("STRATA_D3_PF_P") ? (float) std::atof(std::getenv("STRATA_D3_PF_P")) : 0.0f;
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
            if (pf_noisyor) {
                static const float pooled[16] = { 0.950f, 0.886f, 0.798f, 0.690f, 0.564f, 0.433f, 0.313f, 0.228f,
                                                  0.170f, 0.132f, 0.102f, 0.083f, 0.066f, 0.053f, 0.046f, 0.038f };
                const float * p = layer >= 0 && (size_t) layer < pf_ptab.size() ? pf_ptab[(size_t) layer].data() : pooled;
                pv_noq.clear();
                for (int t = 0; t < m; ++t)
                    for (int i = 0; i < 16; ++i) {
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
                for (int i = 0; i < 16; ++i) {
                    const int e = pv_pred[(size_t) t * 16 + i];
                    if (e < 0) continue;
                    auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & p) { return p.first == e; });
                    if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                }
            std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
            pv_rank.clear();
            for (const auto & p : pv_score) pv_rank.push_back(p.first);
        };
        for (int l = 0; l < n_layer; ++l) {
            if (plog2) pl_ahead_make(l, m);   // STRATA_PRED_LOG2: layer l + k's routing, predicted here
            // STRATA_D3_L3WARM=K: the next K predicted experts (not in VRAM, not prefetched to the card) are read into the
            // CPU's cache under the attention, for the tier's pool (Ds41MoeTier::warm_multi)
            static const int d3_l3warm = std::getenv("STRATA_D3_L3WARM") ? std::atoi(std::getenv("STRATA_D3_L3WARM")) : 0;
            if (pf_now && !pf_async) {
                pv_pred.resize((size_t) m * 16);
                const bool two = d3_pf_early > 0 && l + 1 < n_layer;   // STRATA_D3_PF_EARLY: the next layer's too
                if (two) pv_next.resize((size_t) m * 16);
                if (two ? dense.predict_multi2(l, pv_pred.data(), pv_next.data()) : dense.predict_multi(l, pv_pred.data())) {
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
            // D3 speculative NVMe reads (STRATA_D3_SPEC=n, default 0 = off; STRATA_D3_SPEC_AHEAD=k, default 2): the
            // pass's routing for layer l+k predicted from the state entering l (predict_multi, bias included), its
            // union ranked as above, and the NVMe-tier experts among the top n read into the stash by background
            // readers while layers l..l+k-1 run - the verify pass otherwise waits on them (nsys x0.5: pthread_join
            // 7.9 ms/token on the decode thread).  A wrong guess costs SSD bandwidth only.
            static const int d3_spec = std::getenv("STRATA_D3_SPEC") ? std::atoi(std::getenv("STRATA_D3_SPEC")) : 0;
            static const int d3_spec_ahead = std::getenv("STRATA_D3_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_D3_SPEC_AHEAD")) : 2;
            if (rdev1 && !no_tier_batch && d3_spec > 0 && m <= Ds41MoeTier::kMaxMulti && l + d3_spec_ahead < n_layer) {
                pv_pred.resize((size_t) m * 16);
                if (dense.predict_multi(l + d3_spec_ahead, pv_pred.data())) {
                    rank_union(l + d3_spec_ahead);
                    tier.spec_read(l + d3_spec_ahead, pv_rank.data(), std::min(d3_spec, (int) pv_rank.size()), true);
                }
            }
            const float * fn = nullptr;
            bool attn_done = false;
            if (pf_now && pf_async && dense.predict_multi_launch(l)) {
                pv_pred.resize((size_t) m * 16);
                const bool launched = dense.attn_router_launch(l);   // false: no cached pass graph - attn_router below
                if (dense.predict_multi_wait(pv_pred.data())) {
                    rank_union(l);
                    tier.prefetch_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_pf);
                    if (plog) { pl_pred = pv_pred; pl_have = true; }
                }
                if (launched) {
                    if (!dense.attn_router_collect(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
                    attn_done = true;
                }
            }
            if (!attn_done && !dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            if (plog) plog_rec(plog, l, m, pos, pl_have ? pl_pred.data() : nullptr, mids.data());
            if (plog2 && !pl_ahead[(size_t) l].empty()) plog_rec(plog2, l, m, pos, pl_ahead[(size_t) l].data(), mids.data());
            pl_have = false;
            lap(0);
            if (rdev1 && !no_tier_batch) {
                // D3: ONE batched per-class tier call for the pass's m tokens (the union of their experts is served
                // once), not m run_to_dev calls.  Split at Ds41MoeTier::kMaxMulti (16: a depth-4 pass and a 16-row
                // lookup verify / prompt pass are one call each).
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int chunk = std::min(kMM, m - off);
                    if (!tier.run_multi_to_dev(l, chunk, mids.data() + (size_t) off * top_k, mw.data() + (size_t) off * top_k,
                                               fn + (size_t) off * n_embd, rdev1 + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi_to_dev failed"; return false;
                    }
                }
                if (void * ev = tier.take_sum_event()) dense.wait_event(ev);   // STRATA_PASS_NOSYNC: order on the card
                // STRATA_D3_PF_EARLY: the next layer's copies start now - behind this layer's demand copies, ahead of
                // the rest of this layer and the next attention
                if (have_next) tier.prefetch_next(l + 1, pv_rank_next.data(), (int) pv_rank_next.size(), d3_pf_early);
                have_next = false;
            } else {
                for (int t = 0; t < m; ++t) {
                    const bool ok = rdev1 ? tier.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                            fn + (size_t) t * n_embd, rdev1 + (size_t) t * n_embd)
                                          : tier.run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                     fn + (size_t) t * n_embd, routed1.data());
                    if (!ok) { ferr = "tier.run failed"; return false; }
                    if (void * ev = tier.take_sum_event()) dense.wait_event(ev);   // STRATA_SUM_NOSYNC
                    if (!rdev1) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
                }
            }
            lap(1);
            if (!dense.finish(l, rdev1 ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
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
        vlog.resize((size_t) m * nv);
        if (!dense.logits(0, m, vlog.data())) { ferr = dense.last_error(); return false; }
        lap(3);
        return true;
    };
    const int NP = std::max(1, a.prefill_chunk);
    const int SB = std::max(1, S);                 // the dense sub-batch size
    std::vector<int> cids((size_t) NP * top_k);
    std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), clog;
    const bool routed_host = std::getenv("STRATA_ROUTED_HOST") != nullptr;
    std::vector<float> brouted;                    // the layer-major path's routed sums (n_embd * B)
    if (PB > 0) brouted.resize((size_t) n_embd * (size_t) PB);
    // a single run_chunk call streams layer l's experts once for its n tokens; the card's chunk buffers (dominated by
    // c_parts = n * top_k * n_embd * 4) must fit ~1 GiB, so a larger batch is run in pieces (each streams once again)
    // budget 3 GB by default (STRATA_CHUNK_BUDGET_GB): at B=8192 the old 1 GB cap split every layer into 2 pieces and each
    // piece re-uploaded the whole layer over PCIe (1.97M vs 0.98M expert copies); the prompt has ~12 GiB of VRAM free
    // (the VRAM expert cache is only seeded after the prompt, from the memory measured then)
    const double chunk_budget = std::getenv("STRATA_CHUNK_BUDGET_GB") ? std::atof(std::getenv("STRATA_CHUNK_BUDGET_GB")) * 1e9 : 3.0e9;
    const int piece = (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) n_embd * 4 * (top_k + 3)));
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
#endif
    auto open_cache = [&]() -> bool {
        dense.release_pass();
#if defined(DS41_HAVE_DSPARK)
        if (dsp_path && !dsp) {
            Ds41DSparkConfig sc;
            sc.backend = be;
            sc.sidecar = dsp_path;
            sc.main_gguf = a.model;
            sc.max_append = 16;
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
            }
        }
#endif
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
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
        for (int c0 = 0; c0 < n; ) {
            if (stop_prefill()) return false;
            if (PB > 1 && n - c0 >= 2) {
                // ---- layer-major: up to B tokens of every layer, the experts run once per layer over the whole batch
                int B = std::min(pb_cap, n - c0);
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
                    if (B > dense.columns()) {
                        if (dense.set_columns(B)) { if (rdev1) rdev1 = dense.routed_dev(); }
                        else if (B > SB) {
                            std::fprintf(stderr, "ds41_generate: %d state columns did not fit (%s) - retrying at %d\n",
                                         B, dense.last_error().c_str(), std::max(SB, B / 2));
                            B = std::max(SB, B / 2);
                            continue;
                        } else { ferr = dense.last_error(); return false; }
                    }
                    ds41_heapcheck("pre-begin_batch", B);
                    if (dense.begin_batch(ids + c0, B, pos0 + c0)) { ds41_heapcheck("begin_batch", B); break; }
                    if (B > SB) {
                        std::fprintf(stderr, "ds41_generate: batch of %d did not fit (%s) - retrying at %d\n",
                                     B, dense.last_error().c_str(), std::max(SB, B / 2));
                        B = std::max(SB, B / 2);
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
                    const int64_t ta = (int64_t) pos0 + n - ER - gpos;
                    cut = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                    cut = (cut / SB) * SB;
                    if (!enc_reported) { enc_reported = true; std::fprintf(stderr, "enc-skip: first batch B=%d, cut=%d (phase A %d cols, phase B %d)\n", B, cut, cut, B - cut); }
                }
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
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    // GPU: the tier writes the routed sums straight into the dense half's routed tensor (no 2 x n*D*4
                    // host round trip per layer); CPU / STRATA_ROUTED_HOST=1: through the host as before
                    float * rdev = routed_host ? nullptr : dense.routed_dev();
                    if (l < 2) vram_log("pre-tier", l);
                    for (int s = d0; s < B; s += piece) {   // the experts for the dense rows (in pieces if the buffers cap)
                        const int pn = std::min(piece, B - s);
                        const bool ok = rdev ? tier.run_chunk_dev(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                                  dense.batch_wts() + (size_t) s * top_k,
                                                                  dense.batch_fn() + (size_t) s * n_embd,
                                                                  rdev + (size_t) s * n_embd)
                                             : tier.run_chunk(l, pn, dense.batch_ids() + (size_t) s * top_k,
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
                    const bool ok = m == 1 ? tier.run(l, cids.data(), cw.data(), fn, crouted.data())
                                           : tier.run_chunk(l, m, cids.data(), cw.data(), fn, crouted.data());
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
        tier.release_chunk();
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
        // STRATA_SERVE_STEP_MAX: new prompt tokens up to which a request runs on the decode side - passes of
        // STRATA_SERVE_PASS_M tokens with the VRAM cache and the decode biases (~40 tok/s) - instead of the exact
        // layer-major batch (~16.8 s + n / 250 tok/s; measured crossover ~765 tokens); 0 = always the batch (every prompt
        // token exact)
        const int step_max = std::getenv("STRATA_SERVE_STEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_STEP_MAX")) : 768;
        // STRATA_SERVE_PASS_BIAS=0: those passes route exactly (the decode biases only while decoding)
        const bool pass_bias = !(std::getenv("STRATA_SERVE_PASS_BIAS") && std::getenv("STRATA_SERVE_PASS_BIAS")[0] == '0');
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) tier.resident(),
                    (long long) tier.arena_experts());
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
        while (next_line(line)) {
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            dense.clear_vision();
            if (previous_vision) {
                hist.clear();
                snap_len = -1;
                previous_vision = false;
            }
            // GENI = GEN with an image-embedding file: GENI <max_new> [k=v ..] <embeddings file> <ids csv> (the file
            // holds one strata-vision record per image, in prompt order; the engine lands them on the image_token runs).
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, emb_file;
            int max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            float req_rb = cfg_route_bias, req_rbf = cfg_route_bias_file;
            int req_rprot = cfg_rprot, req_rpool = cfg_rpool;
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
                else if (k == "route_bias") req_rb = fin ? std::min(8.0f, std::max(0.0f, fv)) : 0.0f;
                else if (k == "route_bias_file") req_rbf = fin ? std::min(8.0f, std::max(0.0f, fv)) : 0.0f;
                else if (k == "route_protect") req_rprot = std::min(32, std::max(0, std::atoi(v.c_str())));
                else if (k == "route_pool") req_rpool = std::min(32, std::max(0, std::atoi(v.c_str())));
            }
            // a changed bias is applied again just before the decode: a repeated prompt (regenerate) runs no prompt pass,
            // and the passes are what turn the decode bias on - without this it decoded with the previous request's
            bool bias_changed = false;
            if (req_rb != a.route_bias || req_rbf != a.route_bias_file) {
                a.route_bias = req_rb;
                a.route_bias_file = req_rbf;
                bias_changed = true;
                std::fprintf(stderr, "serve: decode route bias %.2f, file penalty %.2f%s\n", req_rb, req_rbf,
                             (req_rb == 0.0f && req_rbf == 0.0f) ? " (exact routing)" : "");
            }
            if ((req_rprot != cur_rprot || req_rpool != cur_rpool) && dense.set_route_select(req_rprot, req_rpool)) {
                cur_rprot = req_rprot;
                cur_rpool = req_rpool;
                std::fprintf(stderr, "serve: smart routing: exact top %d always kept, picks inside the exact top %d%s\n",
                             req_rprot, req_rpool, (req_rprot == 0 && req_rpool == 0) ? " (off: the flat bias)" : "");
            }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
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
                // an image prompt prefills fresh: every image span is the same run of image_token, so a token-only reuse
                // test would "continue" a different image (same ids, different embeddings).  Empty hist / no snapshot / no park.
                if (!previous_vision) save_live();   // a text conversation it overwrites stays cached (STRATA_SERVE_CACHE_SLOTS)
                dense.drop_park();
                park_hist.clear();
                hist.clear();
                snap_len = -1;
                dense.set_vision(vis_rows.data(), vis_pos.data(), (int) vis_pos.size());
                previous_vision = true;
            } else {
                dense.clear_vision();
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
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
                    const bool usable = lp == t.size() || (lp + 16 >= t.size() && lp >= (size_t) park_min);
                    if (usable && lp > lcp && lp > best_lp) { best = (int) i; best_lp = lp; }
                }
                if (best >= 0) {
                    CachedConv c = std::move(convs[(size_t) best]);
                    convs.erase(convs.begin() + best);
                    save_live();   // the conversation it replaces stays cached
                    if (dense.load_conv(*c.st)) {
                        const bool whole = best_lp == c.toks.size();
                        hist = c.toks;
                        dense.set_history(hist.data(), (int64_t) best_lp, 0);   // Engram's n-grams look back over these
                        if (whole) {   // resumes through restore() at the snapshot, as a full unpark
                            snap_len = (int) c.toks.size();
                            snap_logits = c.logits;
                            snap_seed_n = 0;
                        } else {       // a partial resume (<= 16 back, even): the tail past it is re-fed - from the
                                       // saved snapshot's rings, so copy them into the live ones (a whole resume does
                                       // that through restore() below)
                            dense.restore();
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
                const size_t lp_min = (size_t) std::max<int64_t>(park_min, (int64_t) park_hist.size() * 3 / 4);
                if (park_partial && lp < park_hist.size() && lp >= lp_min && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation at lcp %zu (partial: re-feeds from there)\n",
                                 park_hist.size(), lp);
                    hist = park_hist;
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) lp;
                    lcp = lp;
                    unparked = true;
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;        // DSpark's window holds the side request's states
#endif
                } else if (lp == park_hist.size() && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation\n", park_hist.size());
                    hist = park_hist;
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
            // differently (round 92: lcp 1168 of 1169 fed -> 1,005 tokens re-read in 9.4 s; at 32k thinking ~85 s).
            // Safe as after a rejected DSpark draft: every cache is position-indexed, the window ring has 16 spare
            // slots (<= 6 of them holding rejected-draft rows), and an even position starts a fresh ratio-2 compressor
            // group.  K is capped at 8.  The resumed rows are the decode's own (decode routing), as on a plain resume.
            static const size_t rollback_k = (size_t) std::min(8, std::max(0, std::getenv("STRATA_SERVE_ROLLBACK") ?
                                                                              std::atoi(std::getenv("STRATA_SERVE_ROLLBACK")) : 0));
            static const bool rewind1 = std::getenv("STRATA_SERVE_REWIND1") && std::getenv("STRATA_SERVE_REWIND1")[0] == '1';
            if (reuse_override >= 0) reused = reuse_override;   // a partial unpark: positions [0, lp) are the parked one's
            else if (!unparked && !hist.empty() && lcp == hist.size() && lcp < ids.size()) reused = (int) lcp;
            else if (!unparked && rollback_k > 0 && lcp < hist.size() && hist.size() - lcp <= rollback_k && lcp >= 2 &&
                     lcp < ids.size() && (snap_len <= 0 || (lcp & ~(size_t) 1) >= (size_t) snap_len)) {
                reused = (int) (lcp & ~(size_t) 1);
                std::fprintf(stderr, "serve: rolled back %zu of %zu fed tokens (live resume at %d)\n", hist.size() - (size_t) reused,
                             hist.size(), reused);
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
                       lcp < ids.size() && hist.size() - lcp <= (size_t) rewind_gap) {
                // STRATA_SERVE_REWIND_GAP: the live state continues this prompt but overshoots it by a few tokens
                // (an aborted turn's dropped assistant turn).  Keep the state and re-feed from the common prefix;
                // round down to an even position so the ratio-2 compressor's group partner is re-fed too.
                const size_t gap = hist.size() - lcp;
                size_t rs = lcp & ~(size_t) 1;
                if (rs < 1) rs = lcp;
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
            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK), or with
                // STRATA_SERVE_CACHE_SLOTS save it whole into the cache
                if (cache_slots > 0 && !unparked) save_live();
                const int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                if (cache_slots <= 0 && park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len && rows <= park_max
                    && dense.park(rows)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved)\n", snap_len, (long long) rows);
                }
                hist.clear();
                snap_len = -1;
            }
            if (snap_stack_k > 0) {   // entries past the resume point are overwritten from it on; another conversation: none
                if (unparked || reused <= 0) snap_stack.clear();
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
                if (!(reused > 0 && reused == dsp_next) && dsp) dsp->reset();
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
                    const double need = (4.3 * 1073741824.0 + 0.95 * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.vram_margin_gib * 1073741824.0;
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
                        dsp.reset();
                        std::fprintf(stderr, "serve: DSpark weights freed for a %d-token batch\n", dh);
                    }
#endif
                }
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            tier.rebalance_begin_prompt();
            tier.reset_stats();
            if (const char* rl = std::getenv("STRATA_ROUTE_LOG")) {   // a turn marker for the route log (see ds41_moe.cpp)
                if (FILE* f = std::fopen(rl, "a")) { std::fprintf(f, "# turn %d new %d %s\n", n_req, dh, by_step ? "passes" : "batch"); std::fclose(f); }
            }
            const double tp0 = now_ms();
            auto pp = [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            };
            prefill_cancelled = false;
            auto feed_steps = [&]() -> bool {   // the short-turn path: passes of up to pass_max tokens (a lone token: a step)
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
                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - 4.3 * 1073741824.0;
                const double room_g = (double) frb - a.vram_margin_gib * 1073741824.0 - 2.5 * 1073741824.0 - g_need;
                const int fit_line = (int) std::max<int64_t>(2, (int64_t) (room / (0.95 * 1048576.0)));
                const int fit_g = pb_graph ? (int) std::max<int64_t>(2, (int64_t) (room_g / (0.78 * 1048576.0))) : fit_line;
                const int fit = std::min(fit_line, fit_g);
                pb_cap = std::min(PB, std::max(SB, fit));
                if (vram_trace) std::fprintf(stderr, "vram trace: cap free %.2f GiB room %.2f GiB fit %d, graph %.2f GiB fit %d -> pb_cap %d\n",
                                             frb / 1073741824.0, room / 1073741824.0, fit_line, g_need / 1073741824.0, fit_g, pb_cap);
                if (fit_g < fit_line && fit_g < PB)
                    std::fprintf(stderr, "serve: a %d-token context's pass graph needs %.2f GiB -> prompt batch %d (of %d)\n",
                                 total, g_need / 1073741824.0, pb_cap, PB);
                else if (pb_cap < PB) std::fprintf(stderr, "serve: VRAM %.2f GiB free -> prompt batch %d (of %d)\n",
                                                   frb / 1073741824.0, pb_cap, PB);
            }
            const double tph0 = now_ms();
            const double fd0[3] = { fd_dense, fd_tier, fd_logits };   // this request's batch split (stderr)
            bcap_n = bcap; bcap_got = 0;
            if (bcap > 0) bcap_vh.assign((size_t) bcap * 3 * n_embd, 0.0f);
            dense.reset_span_width();   // a sub-batch narrowed by the last prompt's tight card starts wide again
            bool ok = total == reused || dh == 0 || (by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            bcap_n = 0;
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
                if (tier.regrow_cache(kerr) < 0) {
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
                        if (dsp && !dsp->step(bcap_vh.data() + (vh0 + (size_t) (i0 - c0s)) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                              last ? sp_ids : sp_tmp, sp_conf)) {
                            ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                        }
                    }
                    if (dsp && bcap_got < 1) { ferr = "DSpark batch seed captured nothing"; ok = false; }
                }
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
                    if (dsp && !dsp->step(vh.data(), spos.data(), m, last ? sp_t : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                    }
                }
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
                    if (!dsp->step(snap_seed.data() + (size_t) (i0 - s0) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
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
                    tier.release_chunk();
                    tier.rebalance_tick(true);
                    if (cleaned) std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d 0 0 0 0 0 0\n",
                                             total, now_ms() - tp0, reused);
                    else std::printf("ERR cancelled prefill cleanup: %s\n", dense.last_error().c_str());
                } else std::printf("ERR prefill failed: %s\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();
                snap_len = -1;
                continue;
            }
            hist = ids;
            // no snapshot for an image prompt: a text request with the same ids (all image_token) would restore its embeddings
            // STRATA_SERVE_SNAP_STACK: the snapshot this one replaces is kept while its rows are in place (this request
            // resumed at or past it)
            if (snap_stack_k > 0 && total > reused && !is_vision && snap_len > 0 && reused >= snap_len) {
                if (auto st = dense.save_conv(0)) {
                    snap_stack.push_back({ snap_len, st });
                    if ((int) snap_stack.size() > snap_stack_k) snap_stack.erase(snap_stack.begin());
                }
            }
            if (total > reused && !is_vision && dense.snapshot(total)) {
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
            const Ds41MoeStats pst = tier.stats();
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused, %s) in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / "
                                 "%lld pcie / %lld file)\n", total, reused, by_step ? (exact_pass ? "exact passes" : "passes") : "batch", prompt_ms / 1000.0,
                         1000.0 * (total - reused) / std::max(prompt_ms, 1e-9), (long long) pst.hits, (long long) pst.cpu,
                         (long long) pst.pcie, (long long) pst.file_tier);
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
            if (ok) tier.rebalance_plan();
            if (bias_changed) {   // decode: this request's route bias + file penalty (see bias_changed)
                bias_now = true;
                apply_bias();
            }
            tier.reset_stats();
            dense.route_meter(true);
            const double td0 = now_ms();
            int gen = 0, pos = total;
            const char * finish = "length";
            std::vector<float> lg = last_logits;
#if defined(DS41_HAVE_DSPARK)
            // ---- DSpark speculative sampling (as the one-shot loop: P = last fed position, t = the token for P+1,
            // sp_ids = drafts for P+2..; each verify row's distribution is sample_p's, spec_accept keeps it exact)
            long rq_pass = 0, rq_acc = 0;
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
                int lk_cool = 0, lk_back = 1;
                long lk_pass = 0, lk_dr = 0, lk_acc = 0;
                int dr[kVerify] = {0};
                auto emit = [&](int tok) -> bool {   // false = stop here
                    std::printf("T %d\n", tok);
                    std::fflush(stdout);
                    ++gen;
                    if (is_stop(tok)) { finish = "stop"; return false; }
                    if (max_new > 0 && gen >= max_new) return false;
                    return true;
                };
                while (emit(t)) {
                    if (stop_pending()) { finish = "cancel"; break; }
                    int k = sp_depth_serve;
                    if (max_new > 0) k = std::min(k, max_new - gen);
                    k = std::min(k, ctxv - (P + 3));   // the pass writes P+1 .. P+1+k
                    if (k < 0) break;
                    // STRATA_DSPARK_CONF=<tau>, as the one-shot loop: verify only the leading drafts whose confidence-head
                    // probability is >= tau (the emitted tokens are still the trunk's - only the verify width changes)
                    static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
                    if (conf_tau_s >= 0) {
                        int kk = 0;
                        while (kk < k && 1.0f / (1.0f + std::exp(-sp_conf[kk])) >= conf_tau_s) ++kk;
                        k = kk;
                    }
                    // STRATA_DSPARK_CUM=<c> (Maya's MAYA_DS4_DSPARK_CUM): also stop once the PRODUCT of the confidences -
                    // the chance that this draft and every one before it are kept - falls below c (the same tokens)
                    static const float cum_s = std::getenv("STRATA_DSPARK_CUM") ? (float) std::atof(std::getenv("STRATA_DSPARK_CUM")) : 0.0f;
                    if (cum_s > 0) {
                        int kk = 0;
                        float pr = 1.0f;
                        while (kk < k && (pr *= 1.0f / (1.0f + std::exp(-sp_conf[kk]))) >= cum_s) ++kk;
                        k = kk;
                    }
                    const bool refeed = dirty && ((P + 1) & 1);
                    for (int i = 0; i < k; ++i) dr[i] = sp_ids[i];
                    bool is_lk = false;
                    if (lk_max > 0 && lk_cool > 0) --lk_cool;
                    else if (lk_max > 0) {
                        // the sequence is hist + t (t = the token for P+1, not in hist yet); find the latest earlier q with
                        // seq[q-G .. q] == seq[n-G .. n] (G+1 tokens, the last being t), then copy seq[q+1 ..]
                        const int n = (int) hist.size();
                        auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
                        int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                        if (max_new > 0) cap = std::min(cap, max_new - gen);
                        cap = std::min(cap, ctxv - (P + 3));
                        if (cap >= 2 && n > lk_g + 1) {
                            for (int q = n - 1; q >= lk_g; --q) {
                                if (hist[(size_t) q] != t) continue;
                                int m = 1;
                                while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
                                if (m < lk_g + 1) continue;
                                const int L = std::min({cap, m - lk_g + 1, n - q});
                                if (L < 2) break;   // the latest match copies too little: not worth a pass
                                if (lk_agree && sq(q + 1) != sp_ids[0]) break;
                                for (int i = 0; i < L; ++i) dr[i] = sq(q + 1 + i);
                                k = L;
                                is_lk = true;
                                break;
                            }
                        }
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
                    static const int bias_refresh = std::getenv("STRATA_BIAS_REFRESH") ? std::max(0, std::atoi(std::getenv("STRATA_BIAS_REFRESH"))) : 0;
                    if (bias_refresh > 0 && bias_now && bias_set && rq_pass > 0 && rq_pass % bias_refresh == 0) apply_bias(true);
                    if (!pass_v(vt.data(), (int) vt.size(), p0)) { finish = "error"; break; }
                    int acc = 0, tn = -1;
                    for (; acc < k; ++acc) {
                        sample_dist(vlog.data() + (size_t) (r0 + acc) * nv, nv, temp, tk, top_p, min_p, dist);
                        if (!spec_accept(dist, dr[acc], rng, tn)) break;
                    }
                    if (acc == k) tn = sample_p(vlog.data() + (size_t) (r0 + k) * nv, nv, temp, tk, top_p, min_p, rng);
                    ++rq_pass; rq_acc += acc;
                    if (is_lk) {
                        ++lk_pass; lk_dr += k; lk_acc += acc;
                        if (2 * acc < k) { lk_cool = lk_back; lk_back = std::min(64, lk_back * 2); }
                        else lk_back = 1;
                    }
                    int spos[kVerify + 1];
                    for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
                    int drafts[kVerify];
                    for (int i = 0; i < k; ++i) drafts[i] = dr[i];
                    if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, sp_ids, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); finish = "error"; break;
                    }
                    hist.push_back(t);
                    bool more = true;
                    for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); more = emit(drafts[i]); }
                    prev = acc == 0 ? t : drafts[acc - 1];
                    dirty = acc < k;
                    P += 1 + acc;
                    t = tn;
                    tier.rebalance_tick();
                    if (!more) break;
                }
                if (lk_pass > 0)
                    std::fprintf(stderr, "serve: lookup %ld/%ld drafts accepted in %ld passes\n", lk_acc, lk_dr, lk_pass);
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
                if (is_stop(tok)) { finish = "stop"; break; }
                if ((max_new > 0 && gen >= max_new) || pos + 1 >= ctxv) break;
                // STRATA_BIAS_REFRESH without a drafter: every N single steps (the verify loop above counts passes)
                static const int step_refresh = std::getenv("STRATA_BIAS_REFRESH") ? std::max(0, std::atoi(std::getenv("STRATA_BIAS_REFRESH"))) : 0;
                if (step_refresh > 0 && bias_now && bias_set && gen % step_refresh == 0) apply_bias(true);
                if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { finish = "error"; break; }
                hist.push_back(tok);
                ++pos;
                tier.rebalance_tick();
            }
            const double dec_ms = now_ms() - td0;
            const Ds41MoeStats st = tier.stats();
            if (std::strcmp(finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                hist.clear();
                snap_len = -1;
                finish = "length";
            }
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file\n",
                         gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), finish,
                         (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier);
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
            tier.rebalance_tick(true);   // the gap until the next request: retire a large batch for the reader
            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
        }
        tier.close();
        // the stdin reader thread (detached) sits in getline holding cin's lock: a normal exit's stream teardown waits
        // on it until the client closes the pipe - leave without the static destructors
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    }

    // ---- --score-cases FILE --score-out FILE: teacher-forced continuations, one model load for many independent cases
    // (bench/win3090/official_score.py: DeepSeek's own API continuations from antirez/ds4's quality fixtures).  Each line
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
        while (std::getline(cf, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            const std::vector<int> pr = parse_csv(line.substr(0, tab));
            const std::vector<int> tg = parse_csv(line.substr(tab + 1));
            if (pr.empty() || tg.empty()) { ++ci; continue; }
            if ((int64_t) (pr.size() + tg.size()) + 1 >= ctx) { std::fprintf(stderr, "case %d exceeds the context\n", ci); ++ci; continue; }
            int pos = 0;
            for (int t : pr) {   // the prompt, token by token (its last step's logits predict the first continuation token)
                refresh_bias();
                if (!step(t, pos++)) { std::fprintf(stderr, "case %d: %s\n", ci, ferr.c_str()); return 1; }
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
    tier.reset_stats();
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
    const Ds41MoeStats pst = tier.stats();
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
    tier.reset_stats();
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
    // tools/v41/sim/route_sim.py (session 13: in-pass draft pruning and the VRAM replacement policy, replayed offline)
    if (const char * rt = std::getenv("STRATA_ROUTE_TRACE")) tier.trace_open(rt);
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
    const Ds41MoeStats st = tier.stats();
    std::fprintf(stderr, "decode: %d tokens in %.2f s = %.2f tok/s; experts %lld hit / %lld cpu / %lld pcie / %lld file "
                 "(%.1f%% VRAM hits), cpu %.0f ms, file %.0f ms; spec reads %lld issued / %lld used; prefetch %lld issued / "
                 "%lld outside the arena / %lld used\n",
                 gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), (long long) st.hits,
                 (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                 100.0 * st.hits / std::max<int64_t>(1, st.lookups()), st.cpu_ms, st.file_ms,
                 (long long) st.spec_issued, (long long) st.spec_hits, (long long) st.prefetch_issued,
                 (long long) st.prefetch_dummy, (long long) st.prefetched_useful);
    if (spec)
        std::fprintf(stderr, "DSpark: %ld verify passes, %.3f accepted drafts/pass = %.3f tokens/pass, %.1f%% full, %ld re-fed; "
                     "verify %.1f ms/pass, draft %.2f ms/pass (depth %d)\n", sp_pass, (double) sp_acc / std::max(1L, sp_pass),
                     1.0 + (double) sp_acc / std::max(1L, sp_pass), 100.0 * sp_full / std::max(1L, sp_pass), sp_refeed,
                     sp_vms / std::max(1L, sp_pass), sp_dms / std::max(1L, sp_pass), spec_depth);
    if (spec)
        std::fprintf(stderr, "pass_v split (all %ld passes, %.2f tokens each): dense %.1f ms, tier %.1f ms, taps %.1f ms, "
                     "logits %.1f ms per pass\n", pv_n, (double) pv_tok / std::max(1L, pv_n), pv_ms[0] / std::max(1L, pv_n),
                     pv_ms[1] / std::max(1L, pv_n), pv_ms[2] / std::max(1L, pv_n), pv_ms[3] / std::max(1L, pv_n));
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
    tier.close();
    return 0;
}
