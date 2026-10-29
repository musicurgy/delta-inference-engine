// tools/v41/engine/ds41_helper.cpp - the second GPU's process for STRATA_DS41_REMOTE (see ds41_remote.hpp).
// Ported from upstream Xolotl b4c113c (decode / verify requests; upstream's prompt-chunk and MMQ path is not ported).
//
//   ds41_helper -m <gguf> --rank <rank.bin> [--device 1] [--skip-gib 22] [--gib 13.5] [--name ds41_remote]
//
// Holds the experts of the routing profile that come after the first --skip-gib (about what the engine's card keeps
// in its VRAM cache) until --gib of this card's VRAM is used, then serves the engine's requests until it says stop.
// Each request is computed with the engine's own kernels (native_quantize_q8_1, native_expert_grouped per format
// class), so a row is the one the engine's card would have produced from the same blob.
#include "ds41_moe.hpp"
#include "ds41_remote.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#ifdef _WIN32
#include <timeapi.h>   // timeBeginPeriod: 1 ms waits
#endif

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::v41;
namespace sk = strata::kernels;

namespace {

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "ds41_helper: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Args {
    std::string model, rank, name = "ds41_remote";
    int device = 1;
    double skip_gib = 22.0, gib = 13.5;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--rank") a.rank = next();
        else if (k == "--device") a.device = std::atoi(next().c_str());
        else if (k == "--skip-gib") a.skip_gib = std::atof(next().c_str());
        else if (k == "--gib") a.gib = std::atof(next().c_str());
        else if (k == "--name") a.name = next();
        else { std::fprintf(stderr, "ds41_helper: unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.rank.empty();
}

}  // namespace

int main(int argc, char** argv) {
    static const bool dbg = std::getenv("DS41_HELPER_DEBUG") != nullptr;
#define DBG(msg) do { if (dbg) { std::fprintf(stderr, "ds41_helper: [%s]\n", msg); std::fflush(stderr); } } while (0)
    DBG("start");
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: ds41_helper -m <gguf> --rank <rank.bin> [--device 1] [--skip-gib 22] [--gib 13.5] [--name ds41_remote]\n");
        return 2;
    }
    std::string err;
    Ds41RemoteMap map;
    if (!map.open(a.name, true, err)) { std::fprintf(stderr, "ds41_helper: %s\n", err.c_str()); return 1; }
    Ds41RemoteShm& S = *map.shm();
    S.ready.store(0);
    S.stop.store(0);
    S.req.store(0);
    S.done.store(0);
    S.hint_w.store(0);
    auto fail = [&](const std::string& why) {
        std::fprintf(stderr, "ds41_helper: %s\n", why.c_str());
        S.ready.store(-1, std::memory_order_release);
        return 1;
    };

    DBG("map");
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) return fail("model: " + err);
    const Ds41MoeGeom& G = src.geom();
    if (G.n_embd > Ds41RemoteShm::kMaxEmbd || G.n_layers * G.n_experts > Ds41RemoteShm::kMaxPairs)
        return fail("the model is larger than the shared segment's limits");
    S.magic = Ds41RemoteShm::kMagic;
    S.version = Ds41RemoteShm::kVersion;
    S.n_layers = G.n_layers;
    S.n_experts = G.n_experts;
    S.n_embd = G.n_embd;
    std::memset(S.holds, 0, (size_t) (G.n_layers * G.n_experts));

    DBG("model");
    if (!ck(cudaSetDevice(a.device), "cudaSetDevice")) return fail("no CUDA device " + std::to_string(a.device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, a.device);

    DBG("device");
    // the engine's layouts, one per format class (Ds41MoeTier's init does the same)
    const int64_t NCL = G.n_classes();
    std::vector<sk::NativeExpertLayout> gll((size_t) NCL);
    for (int64_t c = 0; c < NCL; ++c) {
        if (!G.per_expert() && !G.routed(c)) continue;
        const int gt = G.per_expert() ? G.cls_gu[(size_t) c] : G.gate_type(c);
        const int ut = G.per_expert() ? gt : G.up_type(c);
        const int dt = G.per_expert() ? G.cls_d[(size_t) c] : G.down_type(c);
        if (!sk::native_expert_supported3(gt, ut, dt, G.n_embd, G.n_ff)) return fail("no kernel for class " + std::to_string(c));
        gll[(size_t) c] = sk::native_expert_layout3(gt, ut, dt, G.n_embd, G.n_ff);
        gll[(size_t) c].swiglu_limit = G.swiglu_limit;
    }

    DBG("layouts");
    // each class's slot stride in the slab: its blob
    std::vector<int64_t> cstride((size_t) NCL, 0);
    for (int64_t l = 0; l < G.n_layers; ++l)
        for (int64_t e = 0; e < G.n_experts; ++e)
            if (G.routed(l) && !cstride[(size_t) G.cls(l, e)]) cstride[(size_t) G.cls(l, e)] = (int64_t) src.expert_bytes(l, e);

    // the set: the profile after the first skip_gib, until gib
    std::vector<std::pair<int32_t, int32_t>> ranked = ds41_read_ranked(a.rank, err);
    if (ranked.empty()) return fail("rank: " + err);
    const double skip_b = a.skip_gib * 1073741824.0, take_b = a.gib * 1073741824.0;
    double seen = 0, held_b = 0;
    std::vector<std::pair<int32_t, int32_t>> set;
    std::vector<char> picked((size_t) (G.n_layers * G.n_experts), 0);
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= G.n_layers || le.second < 0 || le.second >= G.n_experts) continue;
        char& p = picked[(size_t) (le.first * G.n_experts + le.second)];
        if (p) continue;
        p = 1;
        const double b = (double) src.expert_bytes(le.first, le.second);
        seen += b;
        if (seen <= skip_b) continue;
        const double sb = (double) cstride[(size_t) G.cls(le.first, le.second)];   // its slot in the slab
        if (held_b + sb > take_b) break;
        held_b += sb;
        set.push_back(le);
    }

    if (std::getenv("DS41_HELPER_CLASSES")) {   // diagnostic: the set's experts per format class
        std::vector<int> nc((size_t) G.n_classes(), 0);
        for (const auto& le : set) ++nc[(size_t) G.cls(le.first, le.second)];
        for (int64_t c = 0; c < G.n_classes(); ++c)
            if (nc[(size_t) c]) std::fprintf(stderr, "ds41_helper: class %lld (gate/up %d, down %d): %d experts\n", (long long) c,
                                             G.per_expert() ? G.cls_gu[(size_t) c] : -1, G.per_expert() ? G.cls_d[(size_t) c] : -1, nc[(size_t) c]);
    }
    DBG("set");
    // one device slab of uniform slots (the largest blob, 4 KiB aligned; zeroed: a kernel may read a little past a
    // blob) - this fork: any slot can take any expert, so the set can follow the engine's misses (below).  The set
    // starts as the profile's stretch above, as many as fit.
    const int64_t NX = G.n_experts, NLE = G.n_layers * G.n_experts;
    size_t slot_b = 0;
    for (int64_t c = 0; c < NCL; ++c) slot_b = std::max(slot_b, (size_t) cstride[(size_t) c]);
    slot_b = (slot_b + 4095) & ~(size_t) 4095;
    const int64_t nslots = std::max<int64_t>(1, (int64_t) (take_b / (double) slot_b));
    if ((int64_t) set.size() > nslots) set.resize((size_t) nslots);
    if (set.empty()) return fail("no experts in the set (--skip-gib past the profile?)");
    held_b = (double) nslots * (double) slot_b;
    const size_t slab_bytes = (size_t) nslots * slot_b + ((size_t) 1 << 20);
    uint8_t* slab = nullptr;
    if (!ck(cudaMalloc(&slab, slab_bytes), "slab")) return fail("cannot allocate " + std::to_string(held_b / 1073741824.0) + " GiB");
    if (!ck(cudaMemset(slab, 0, slab_bytes), "slab zero")) return fail("slab zero");
    std::vector<unsigned long long> dptr((size_t) NLE, 0);
    std::vector<int32_t> sidx((size_t) NLE, -1);          // (layer, expert) -> its slot
    std::vector<int32_t> slot_le((size_t) nslots, -1);    // slot -> layer * n_experts + expert (-1 = empty)
    std::vector<double> slot_use((size_t) nslots, 0.0);   // slot -> when a request last used it (ms; 0 = never)
    std::vector<uint8_t> host(src.blob_bytes());
    const double t0 = now_ms();
    for (size_t q = 0; q < set.size(); ++q) {
        const int32_t le = set[q].first * (int32_t) NX + set[q].second;
        uint8_t* d = slab + q * slot_b;
        src.read(set[q].first, set[q].second, host.data());
        if (!ck(cudaMemcpy(d, host.data(), src.expert_bytes(set[q].first, set[q].second), cudaMemcpyHostToDevice), "blob upload")) return fail("upload failed");
        dptr[(size_t) le] = (unsigned long long) d;
        sidx[(size_t) le] = (int32_t) q;
        slot_le[q] = le;
        S.holds[(size_t) le] = 1;
    }
    S.held = (int64_t) set.size();
    S.held_bytes = (int64_t) held_b;

    DBG("slab");
    // request buffers
    const int64_t H = G.n_embd;
    const int kMaxEnt = Ds41RemoteShm::kMaxEnt;
    float* d_x = nullptr; void* d_xq = nullptr; float* d_out = nullptr; void* d_scratch = nullptr; uint8_t* d_meta = nullptr;
    const size_t meta_bytes = 64 * 1024;
    bool ok = ck(cudaMalloc(&d_x, (size_t) Ds41RemoteShm::kMaxTok * H * 4), "d_x") &&
              // one column's q8_1 bytes x the tokens (native_q8_1_bytes takes at most 8 columns; the tier sizes it so)
              ck(cudaMalloc(&d_xq, (size_t) Ds41RemoteShm::kMaxTok * sk::native_q8_1_bytes((int) H, 1)), "d_xq") &&
              ck(cudaMalloc(&d_out, (size_t) kMaxEnt * H * 4), "d_out") &&
              ck(cudaMalloc(&d_scratch, sk::native_expert_scratch_bytes(kMaxEnt, G.n_ff)), "d_scratch") &&
              ck(cudaMalloc(&d_meta, meta_bytes), "d_meta");
    // pinned: the metadata copy is a plain DMA (the request ends with a stream sync, so the next one may rewrite it)
    uint8_t* h_meta = nullptr;
    ok = ok && ck(cudaMallocHost((void**) &h_meta, meta_bytes), "h_meta");
    // per-request scratch, kept across requests
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
    // the segment pinned for this process: the copies DMA straight from x / into out
    ok = ok && ck(cudaHostRegister(&S, sizeof(Ds41RemoteShm), cudaHostRegisterDefault), "register the segment");
    cudaStream_t st = nullptr;
    ok = ok && ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    if (!ok) return fail("buffers");

    DBG("buffers");
    // warm-up: one launch per class the set has, so the module loading of each kernel (seconds on the first call)
    // is paid here and not by the engine's first request
    {
        const double tw = now_ms();
        std::vector<char> warm((size_t) NCL, 0);
        int nwarm = 0;
        std::vector<float> zero((size_t) H, 0.0f);
        ok = ck(cudaMemcpy(d_x, zero.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "warm x");
        sk::native_quantize_q8_1(d_x, d_xq, (int) H, 1, st);
        for (const auto& le : set) {
            const int c = G.cls(le.first, le.second);
            if (warm[(size_t) c] || !ok) continue;
            warm[(size_t) c] = 1;
            const unsigned long long p = dptr[(size_t) (le.first * G.n_experts + le.second)];
            const int32_t meta[6] = {0, 1, 1, 0, 0, 0};   // start {0, 1}, n_groups 1, dst 0, tok 0
            ok = ck(cudaMemcpy(d_meta, &p, 8, cudaMemcpyHostToDevice), "warm ptr") &&
                 ck(cudaMemcpy(d_meta + 16, meta, sizeof(meta), cudaMemcpyHostToDevice), "warm meta");
            sk::native_expert_grouped(gll[(size_t) c], (const unsigned long long*) d_meta, (const int32_t*) (d_meta + 16),
                                      (const int32_t*) (d_meta + 24), (const int32_t*) (d_meta + 28),
                                      (const int32_t*) (d_meta + 32), 1, 1, d_xq, d_scratch, d_out, st, 1);
            ++nwarm;
        }
        if (!ok || !ck(cudaStreamSynchronize(st), "warm-up")) return fail("warm-up");
        DBG("warm");
        std::fprintf(stderr, "ds41_helper: warm-up of %d classes in %.1f s\n", nwarm, (now_ms() - tw) / 1000.0);
    }
    std::fprintf(stderr, "ds41_helper: %s (CUDA%d): %zu experts, %.2f GiB (profile from %.1f GiB), loaded in %.1f s - serving '%s'\n",
                 prop.name, a.device, set.size(), held_b / 1073741824.0, a.skip_gib, (now_ms() - t0) / 1000.0, a.name.c_str());
    S.ready.store(1, std::memory_order_release);

    int64_t last = 0, served = 0;
    double busy = 0, busy_max = 0;   // since the last print
    // DS41_HELPER_PROF=1 (diagnostic, slower): a sync after each stage, wall ms per stage summed over the requests
    static const bool sprof = std::getenv("DS41_HELPER_PROF") != nullptr;
    double sp[5] = {};
    double ts = 0;
    auto stage = [&](int i) {
        if (!sprof) return;
        cudaStreamSynchronize(st);
        const double t = now_ms();
        sp[i] += t - ts;
        ts = t;
    };
    // Keep the card awake: between requests (the engine sends one every few layer calls, often 5-20 ms apart) the
    // card drops to an idle clock and a request then took 1-2 ms instead of ~0.14 ms (remote_bench with idle gaps on
    // the RTX 4080: 0 / 1 / 5 / 20 ms gap -> 135 / 152 / 953 / 1931 us median).  While idle, one expert of the set is
    // run every DS41_HELPER_WARM_US (default 100; 0 = off) on a stream and buffers of its own, so a request never waits
    // for it.  Measured (remote_bench, 0 / 5 / 20 ms gaps, median): a one-column quantize every 500 us kept the card at
    // 285 MHz (- / 576 / 1157 us); an expert every 300 us ~1 GHz (- / 584 / 663 us); every 150 us 1.7 GHz, 60 W
    // (165 / 177 / 249 us, p90 802 at 20 ms); every 100 us 2.6 GHz, ~100 W while serving (137 / 142 / 151 us).
    // This fork: OFF by default (DS41_HELPER_WARM_US=100 = upstream) - it needs this thread awake every 100 us, a whole
    // CPU core the engine's pool shares (round 216: one core taken cut another engine run from 15.9 to 10.0 tok/s).
    const double warm_us = std::getenv("DS41_HELPER_WARM_US") ? std::atof(std::getenv("DS41_HELPER_WARM_US")) : 0.0;
    // ... but only until no request came for DS41_HELPER_WARM_IDLE_S (default 10; 0 = always): then the card may idle
    // (the keep-warm holds it at ~83 W, fan on, with no one using the model); the first request after that wakes it
    const double warm_idle_ms = (std::getenv("DS41_HELPER_WARM_IDLE_S") ? std::atof(std::getenv("DS41_HELPER_WARM_IDLE_S")) : 10.0) * 1000.0;
    uint8_t* d_wmeta = nullptr;
    void *d_wxq = nullptr, *d_wscratch = nullptr;
    float* d_wout = nullptr;
    cudaStream_t wst = nullptr;
    const int wc = G.cls(set.front().first, set.front().second);
    if (warm_us > 0) {
        const unsigned long long p = dptr[(size_t) (set.front().first * G.n_experts + set.front().second)];
        const int32_t meta[6] = {0, 1, 1, 0, 0, 0};   // start {0, 1}, n_groups 1, dst 0, tok 0
        if (!ck(cudaMalloc(&d_wmeta, 64), "keep-warm meta") || !ck(cudaMemcpy(d_wmeta, &p, 8, cudaMemcpyHostToDevice), "keep-warm ptr") ||
            !ck(cudaMemcpy(d_wmeta + 16, meta, sizeof(meta), cudaMemcpyHostToDevice), "keep-warm meta") ||
            !ck(cudaMalloc(&d_wxq, sk::native_q8_1_bytes((int) H, 1)), "keep-warm xq") ||
            !ck(cudaMemcpy(d_wxq, d_xq, sk::native_q8_1_bytes((int) H, 1), cudaMemcpyDeviceToDevice), "keep-warm xq copy") ||
            !ck(cudaMalloc(&d_wscratch, sk::native_expert_scratch_bytes(1, G.n_ff)), "keep-warm scratch") ||
            !ck(cudaMalloc(&d_wout, (size_t) H * 4), "keep-warm out") ||
            !ck(cudaStreamCreateWithFlags(&wst, cudaStreamNonBlocking), "keep-warm stream"))
            return fail("keep-warm");
    }
    // ---- dynamic slots (this fork; DS41_HELPER_ADMIT=0 = upstream's fixed set): the engine's card posts its misses
    // as hints; an expert missed DS41_HELPER_ADMIT times (default 2; the counts halve every 65536 hints) is read from
    // the model file by a loader thread into pinned staging, then copied over the least recently used slot that no
    // request used in the last DS41_HELPER_MIN_AGE_MS (default 2000; never-used slots go first).  The victim leaves `holds` first and its
    // slot is rewritten only after two more requests (or 50 ms): a call of the engine that saw it held posts within
    // that window.  One admission at a time, between requests, on a stream of its own.
    const int admit_min = std::getenv("DS41_HELPER_ADMIT") ? std::atoi(std::getenv("DS41_HELPER_ADMIT")) : 2;
    const double min_age = std::getenv("DS41_HELPER_MIN_AGE_MS") ? std::atof(std::getenv("DS41_HELPER_MIN_AGE_MS")) : 2000.0;
    std::vector<uint16_t> hcnt((size_t) NLE, 0);
    std::vector<char> queued((size_t) NLE, 0);
    int64_t hr = 0, nhint = 0, n_admit = 0, n_drop = 0;
    constexpr int kStage = 4;
    uint8_t* stg[kStage] = {};
    for (int i = 0; i < kStage; ++i)
        if (!ck(cudaMallocHost((void**) &stg[i], src.blob_bytes()), "admission staging")) return fail("staging");
    std::mutex mu;
    std::condition_variable cv;
    std::deque<int32_t> want;                    // experts to read (main -> loader)
    std::deque<std::pair<int32_t, int>> ready;   // (expert, staging buffer) read (loader -> main)
    std::vector<int> free_stg = {0, 1, 2, 3};
    bool quit = false;
    std::thread loader([&]() {
        for (;;) {
            int32_t le;
            int b;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return quit || (!want.empty() && !free_stg.empty()); });
                if (quit) return;
                le = want.front(); want.pop_front();
                b = free_stg.back(); free_stg.pop_back();
            }
            src.read(le / NX, le % NX, stg[b]);
            std::lock_guard<std::mutex> lk(mu);
            ready.push_back({le, b});
        }
    });
    cudaStream_t cst = nullptr;
    cudaEvent_t cev = nullptr;
    if (!ck(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking), "admission stream") ||
        !ck(cudaEventCreateWithFlags(&cev, cudaEventDisableTiming), "admission event")) return fail("admission");
    int ast = 0;                 // 0 idle, 1 victim leaving (grace), 2 copy in flight
    int32_t a_le = -1, a_slot = -1, a_buf = -1;
    // the admission copy goes up in DS41_HELPER_CHUNK_KB pieces (default 256), one in flight: on a narrow link a whole
    // blob (~10 MB: ~11 ms on x1) queued ahead of a request's rows held every request up (round 216: tier 86 -> 248 ms
    // a verify pass)
    const size_t a_chunk = (size_t) (std::getenv("DS41_HELPER_CHUNK_KB") ? std::atof(std::getenv("DS41_HELPER_CHUNK_KB")) : 256.0) * 1024;
    size_t a_off = 0, a_bytes = 0;
    int64_t a_mark = 0;
    double a_mark_ms = 0;
    auto admit_step = [&]() {
        if (admit_min <= 0) return;
        // the hints
        const int64_t hw = S.hint_w.load(std::memory_order_acquire);
        if (hw - hr > Ds41RemoteShm::kHint) hr = hw - Ds41RemoteShm::kHint;
        for (; hr < hw; ++hr) {
            const int32_t le = S.hint_le[(size_t) (hr % Ds41RemoteShm::kHint)];
            if (le < 0 || le >= NLE || S.holds[(size_t) le] || queued[(size_t) le]) continue;
            if (++nhint % 65536 == 0) for (auto& c : hcnt) c >>= 1;
            if (hcnt[(size_t) le] < 65535) ++hcnt[(size_t) le];
            if (hcnt[(size_t) le] < admit_min) continue;
            std::lock_guard<std::mutex> lk(mu);
            if (want.size() >= 32) { ++n_drop; continue; }
            queued[(size_t) le] = 1;
            want.push_back(le);
            cv.notify_one();
        }
        if (ast == 0) {
            {
                std::lock_guard<std::mutex> lk(mu);
                if (ready.empty()) return;
                a_le = ready.front().first;
                a_buf = ready.front().second;
                ready.pop_front();
            }
            int32_t v = -1;
            const double tnow = now_ms();
            for (int64_t s = 0; s < nslots; ++s)
                if (slot_use[(size_t) s] + min_age <= tnow && (v < 0 || slot_use[(size_t) s] < slot_use[(size_t) v])) v = (int32_t) s;
            if (v < 0) {   // every slot is in use: drop this one
                std::lock_guard<std::mutex> lk(mu);
                free_stg.push_back(a_buf);
                queued[(size_t) a_le] = 0;
                hcnt[(size_t) a_le] = 0;
                cv.notify_one();
                ++n_drop;
                return;
            }
            a_slot = v;
            if (slot_le[(size_t) v] >= 0) S.holds[(size_t) slot_le[(size_t) v]] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            a_mark = last;
            a_mark_ms = now_ms();
            ast = 1;
        }
        if (ast == 1) {
            if (last < a_mark + 2 && now_ms() - a_mark_ms < 50.0) return;
            const int32_t old = slot_le[(size_t) a_slot];
            if (old >= 0) { dptr[(size_t) old] = 0; sidx[(size_t) old] = -1; }
            slot_le[(size_t) a_slot] = -1;
            a_off = 0;
            a_bytes = src.expert_bytes(a_le / NX, a_le % NX);
            ast = 2;
        }
        if (ast == 2) {
            if (a_off > 0 && cudaEventQuery(cev) == cudaErrorNotReady) return;
            if (a_off < a_bytes) {
                const size_t n = std::min(std::max<size_t>(a_chunk, 4096), a_bytes - a_off);
                ck(cudaMemcpyAsync(slab + (size_t) a_slot * slot_b + a_off, stg[a_buf] + a_off, n, cudaMemcpyHostToDevice, cst),
                   "admission copy");
                ck(cudaEventRecord(cev, cst), "admission event");
                // wake this thread the moment the piece lands (it blocks on the segment's event between requests), so
                // the next piece goes at once instead of after a 1 ms timeout (~40 ms a blob: round 217, 0.6% served)
                ck(cudaLaunchHostFunc(cst, [](void* m) { static_cast<Ds41RemoteMap*>(m)->signal(); }, &map), "admission wake");
                a_off += n;
                return;
            }
            dptr[(size_t) a_le] = (unsigned long long) (slab + (size_t) a_slot * slot_b);
            sidx[(size_t) a_le] = a_slot;
            slot_le[(size_t) a_slot] = a_le;
            slot_use[(size_t) a_slot] = now_ms();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            S.holds[(size_t) a_le] = 1;
            {
                std::lock_guard<std::mutex> lk(mu);
                free_stg.push_back(a_buf);
                queued[(size_t) a_le] = 0;
                hcnt[(size_t) a_le] = 0;
                cv.notify_one();
            }
            ++n_admit;
            ast = 0;
        }
    };
    // Idle without a busy core (this fork): after a request it polls for DS41_HELPER_SPIN_US (default 50; the engine's
    // next layer call usually comes sooner), then blocks on the segment's event, which the engine sets per request -
    // 1 ms timeouts while an admission is under way, 10 ms otherwise (stop / hints / keep-warm are checked then).
    const double spin_us = std::getenv("DS41_HELPER_SPIN_US") ? std::atof(std::getenv("DS41_HELPER_SPIN_US")) : 50.0;
#ifdef _WIN32
    timeBeginPeriod(1);
#endif
    double last_busy = now_ms(), last_req = last_busy;
    bool warm_on = true;
    for (;;) {
        int64_t r;
        while ((r = S.req.load(std::memory_order_acquire)) == last) {
            if (S.stop.load(std::memory_order_relaxed)) goto out;
            admit_step();
            if (warm_us > 0 && warm_idle_ms > 0 && warm_on != (now_ms() - last_req < warm_idle_ms)) {
                warm_on = !warm_on;
                if (std::getenv("DS41_HELPER_PROF"))
                    std::fprintf(stderr, "ds41_helper: keep-warm %s\n", warm_on ? "on" : "off (idle)");
            }
            if (warm_us > 0 && warm_on && (now_ms() - last_busy) * 1000.0 >= warm_us) {
                if (cudaStreamQuery(wst) != cudaErrorNotReady)   // never queue more than one
                    sk::native_expert_grouped(gll[(size_t) wc], (const unsigned long long*) d_wmeta, (const int32_t*) (d_wmeta + 16),
                                              (const int32_t*) (d_wmeta + 24), (const int32_t*) (d_wmeta + 28),
                                              (const int32_t*) (d_wmeta + 32), 1, 1, d_wxq, d_wscratch, d_wout, wst, 1);
                last_busy = now_ms();
            }
            if ((now_ms() - last_req) * 1000.0 < spin_us) { ds41_remote_pause(); continue; }
            bool busy_adm = ast != 0;
            if (!busy_adm) { std::lock_guard<std::mutex> lk(mu); busy_adm = !ready.empty(); }
            map.wait(busy_adm || (warm_us > 0 && warm_on) ? 1 : 10);
        }
        const double tq = now_ms();
        ts = tq;
        const int nt = S.nt, ne = S.n_ent, layer = S.layer;
        int status = 0;
        if (nt <= 0 || nt > Ds41RemoteShm::kMaxTok || ne <= 0 || ne > kMaxEnt || layer < 0 || layer >= G.n_layers) status = 1;
        // every entry's blob must be here (the engine only sends what `holds` says)
        for (int i = 0; i < ne && status == 0; ++i)
            if (!dptr[(size_t) (layer * G.n_experts + S.ent_expert[i])]) status = 2;
        for (int i = 0; i < ne && status == 0; ++i) slot_use[(size_t) sidx[(size_t) (layer * G.n_experts + S.ent_expert[i])]] = tq;
        if (status == 0) {
            ck(cudaMemcpyAsync(d_x, S.x, (size_t) nt * H * 4, cudaMemcpyHostToDevice, st), "x h2d");
            sk::native_quantize_q8_1(d_x, d_xq, (int) H, std::min(nt, 8), st);
            if (nt > 8) {   // as the tier's quantize_rows_q8_1: 8 columns per call
                const size_t col = sk::native_q8_1_bytes((int) H, 1);
                for (int c0 = 8; c0 < nt; c0 += 8)
                    sk::native_quantize_q8_1(d_x + (size_t) c0 * H, (uint8_t*) d_xq + (size_t) c0 * col, (int) H, std::min(8, nt - c0), st);
            }
            stage(0);   // x up + quantize
            // one grouped launch per format class: groups = distinct experts in first-seen order, entries = the
            // requests' entries (output row i = entry i); the metadata of every class goes up in one copy
            struct Launch { int c, ng; size_t ne, o_ptr, o_st, o_ng, o_dst, o_tok; };
            static std::vector<Launch> launches;
            launches.clear();
            auto al16 = [](size_t x) { return (x + 15) & ~(size_t) 15; };
            bool taken[Ds41RemoteShm::kMaxEnt] = {};
            size_t meta_at = 0;
            for (int i0 = 0; i0 < ne && status == 0; ++i0) {
                if (taken[i0]) continue;
                const int c = G.cls(layer, S.ent_expert[i0]);
                ptr.clear(); start.clear(); dst.clear(); tok.clear();
                for (int i = i0; i < ne; ++i) {   // i: the first entry of a not yet grouped expert of class c
                    if (taken[i] || G.cls(layer, S.ent_expert[i]) != c) continue;
                    const int32_t e = S.ent_expert[i];
                    ptr.push_back(dptr[(size_t) (layer * G.n_experts + e)]);
                    start.push_back((int32_t) dst.size());
                    for (int j = i; j < ne; ++j)
                        if (!taken[j] && S.ent_expert[j] == e) { dst.push_back(j); tok.push_back(S.ent_tok[j]); taken[j] = true; }
                }
                Launch L;
                L.c = c;
                L.ng = (int) ptr.size();
                L.ne = dst.size();
                start.push_back((int32_t) L.ne);
                L.o_ptr = meta_at;
                L.o_st = al16(L.o_ptr + 8 * (size_t) L.ng);
                L.o_ng = al16(L.o_st + 4 * (size_t) (L.ng + 1));
                L.o_dst = al16(L.o_ng + 4);
                L.o_tok = al16(L.o_dst + 4 * L.ne);
                meta_at = al16(L.o_tok + 4 * L.ne);
                if (meta_at > meta_bytes) { status = 3; break; }
                std::memcpy(h_meta + L.o_ptr, ptr.data(), 8 * (size_t) L.ng);
                std::memcpy(h_meta + L.o_st, start.data(), 4 * (size_t) (L.ng + 1));
                std::memcpy(h_meta + L.o_ng, &L.ng, 4);
                std::memcpy(h_meta + L.o_dst, dst.data(), 4 * L.ne);
                std::memcpy(h_meta + L.o_tok, tok.data(), 4 * L.ne);
                launches.push_back(L);
            }
            stage(1);   // the groups on the host
            if (status == 0) ck(cudaMemcpyAsync(d_meta, h_meta, meta_at, cudaMemcpyHostToDevice, st), "meta");
            stage(2);   // metadata up
            for (const Launch& L : launches) {
                if (status != 0) break;
                sk::native_expert_grouped(gll[(size_t) L.c], (const unsigned long long*) (d_meta + L.o_ptr),
                                          (const int32_t*) (d_meta + L.o_st), (const int32_t*) (d_meta + L.o_ng),
                                          (const int32_t*) (d_meta + L.o_dst), (const int32_t*) (d_meta + L.o_tok), L.ng,
                                          (int64_t) L.ne, d_xq, d_scratch, d_out, st, L.ng);
            }
            stage(3);   // the grouped launches
            if (status == 0) ck(cudaMemcpyAsync(S.out, d_out, (size_t) ne * H * 4, cudaMemcpyDeviceToHost, st), "out d2h");
            if (!ck(cudaStreamSynchronize(st), "request")) status = 4;
            stage(4);   // rows down
        }
        S.status.store(status, std::memory_order_relaxed);
        S.done.store(r, std::memory_order_release);
        last = r;
        last_busy = last_req = now_ms();
        ++served;
        const double bt = now_ms() - tq;
        busy += bt;
        busy_max = std::max(busy_max, bt);
        if (served % 20000 == 0) {
            std::fprintf(stderr, "ds41_helper: %lld requests: last 20000 %.1f us busy each (max %.1f); %lld admitted, %lld dropped\n",
                         (long long) served, 1000.0 * busy / 20000.0, 1000.0 * busy_max, (long long) n_admit, (long long) n_drop);
            if (sprof) {
                std::fprintf(stderr, "ds41_helper: us/request: x+quant %.1f, groups %.1f, meta %.1f, kernels %.1f, rows %.1f\n",
                             sp[0] / 20.0, sp[1] / 20.0, sp[2] / 20.0, sp[3] / 20.0, sp[4] / 20.0);
                for (double& v : sp) v = 0;
            }
            busy = 0;
            busy_max = 0;
        }
    }
out:
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    loader.join();
    std::fprintf(stderr, "ds41_helper: stop after %lld requests (%lld experts admitted)\n", (long long) served, (long long) n_admit);
    cudaHostUnregister(&S);
    return 0;
}
