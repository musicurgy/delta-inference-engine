// test_indexer_v2.cu - STRATA_INDEXER_V2: lightning_indexer_kernel_v2 vs lightning_indexer_kernel_vec, BITWISE.
//
// Builds random MXFP4 index keys / fp32 (or FP4_POW2_32 fake-quant) queries / weights / visibility masks in the
// engine's layouts (ds41_dense.cpp build_attn: ksrc = view_3d(idxk, 128, 1, cap, row, row), iq [128, IH, nq],
// iw [IH, nq], mask f16 [cap, nq] of 0 / -inf, scores f32 [cap, nq]), runs the shipped vec kernel and the v2 kernel
// on the same inputs, memcmp's the score tensors (first mismatch reported) and times both with cudaEvents.
//
// Build (from the worktree root, Git Bash; same fp flags as ggml-cuda's CMake: -use_fast_math):
//   nvcc -std=c++17 -O3 -use_fast_math -extended-lambda -arch=sm_86 -Xptxas -v \
//     -I third_party/llama.cpp/ggml/src/ggml-cuda -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src \
//     -Xcompiler "/Zc:preprocessor /EHsc" tools/v41/engine/test_indexer_v2.cu -o test_indexer_v2.exe
// Run:  test_indexer_v2.exe            all cases (caps 8K / 64K / 256K x queries 1 / 7 / 128 / 512, H 64, plus H 32,
//                                      n_kv tails, caps 1..100, n_stream 2-3)
//       test_indexer_v2.exe quick      the cases at cap <= 8K+ only (skips 64K / 256K)
//       test_indexer_v2.exe reps N     timing repetitions per kernel (default 5)
// Exit code 0 = every case bit-identical.

#define GGML_CUDA_LIGHTNING_INDEXER_KERNELS_ONLY
#include "lightning-indexer.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

// common.cuh's inline host helpers may reference these; the test never calls them
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort %s:%d %s\n", file, line, fmt);
    std::abort();
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

static constexpr int N_EMBD = 128;
static constexpr int ROW_BYTES = (N_EMBD / QK_MXFP4) * (int) sizeof(block_mxfp4);   // 68

static uint16_t f32_to_f16_bits(float f) {   // only 0 and -inf are needed
    if (f == 0.0f) return 0x0000;
    if (std::isinf(f) && f < 0) return 0xFC00;
    std::abort();
}

enum class mask_kind { causal, partial, holes, all_dead, all_live };
static const char * mask_name(mask_kind k) {
    switch (k) {
        case mask_kind::causal:   return "causal";
        case mask_kind::partial:  return "partial";
        case mask_kind::holes:    return "holes";
        case mask_kind::all_dead: return "all_dead";
        case mask_kind::all_live: return "all_live";
    }
    return "?";
}

struct test_case {
    int64_t cap; int nq; int n_head; mask_kind mk; bool q_fake_quant; bool stress; int skip;
    int ns = 1;     // n_stream (the engine uses 1)
    int nem3 = 1;   // mask streams: 1 = broadcast over n_stream (i_stream % nem3), or ns
};

// e2m1 values (kvalues_mxfp4 / 2 - the kernel's d carries the 0.5)
static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static float fake_quant_pow2_32(float x, float amax) {   // FP4_POW2_32-like: pow2 scale per 32, e2m1 grid
    if (amax == 0.0f) return 0.0f;
    const float s = std::ldexp(1.0f, (int) std::ceil(std::log2(amax / 6.0f)));
    const float a = std::fabs(x) / s;
    int best = 0;
    for (int i = 1; i < 8; ++i) if (std::fabs(e2m1[i] - a) < std::fabs(e2m1[best] - a)) best = i;
    return std::copysign(e2m1[best] * s, x);
}

static bool run_case(const test_case & tc, int reps, std::mt19937_64 & rng, double & t_old_ms, double & t_new_ms) {
    const int64_t cap = tc.cap;
    const int nq = tc.nq, H = tc.n_head, ns = tc.ns, nem3_m = tc.nem3;

    // ---- keys: MXFP4 rows, 4 blocks of {e8m0, 16 x 2 nibbles}
    std::vector<uint8_t> hk((size_t) cap * ns * ROW_BYTES);
    {
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_int_distribution<int> e_norm(118, 128);
        std::uniform_int_distribution<int> e_wide(0, 255);
        std::uniform_int_distribution<int> pct(0, 99);
        for (int64_t r = 0; r < cap * ns; ++r) {
            block_mxfp4 * b = (block_mxfp4 *) &hk[(size_t) r * ROW_BYTES];
            for (int j = 0; j < N_EMBD / QK_MXFP4; ++j) {
                int e = e_norm(rng);
                if (tc.stress && pct(rng) < 5) e = e_wide(rng);   // includes 0, 254 (inf products), 255 (NaN)
                b[j].e = (uint8_t) e;
                for (int i = 0; i < QK_MXFP4 / 2; ++i) b[j].qs[i] = (uint8_t) byte(rng);
            }
        }
    }

    // ---- queries [128, H, nq] f32, weights [H, nq] f32
    std::vector<float> hq((size_t) N_EMBD * H * nq * ns), hw((size_t) H * nq * ns);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_int_distribution<int> pct(0, 999);
        for (auto & v : hq) {
            v = nd(rng);
            if (tc.stress) {
                const int p = pct(rng);
                if      (p < 5)  v = 0.0f;
                else if (p < 10) v = -0.0f;
                else if (p < 15) v *= 1e-39f;      // fp32 subnormal: ftz must agree
                else if (p < 18) v *= 1e30f;       // overflow to inf in the dot products
            }
        }
        if (tc.q_fake_quant) {
            for (size_t i = 0; i < hq.size(); i += 32) {
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(hq[i + j]));
                for (int j = 0; j < 32; ++j) hq[i + j] = fake_quant_pow2_32(hq[i + j], amax);
            }
        }
        const float ws = 1.0f / std::sqrt((float) (N_EMBD * H));
        for (auto & v : hw) v = nd(rng) * ws;
    }

    // ---- visibility mask [cap, nq] f16: 0 visible / -inf hidden
    std::vector<uint16_t> hm((size_t) cap * nq * nem3_m);
    for (int s = 0; s < nem3_m; ++s) {
        std::uniform_int_distribution<int64_t> any_row(0, cap);
        std::uniform_int_distribution<int> pct(0, 99);
        for (int t = 0; t < nq; ++t) {
            int64_t nvis = cap;
            switch (tc.mk) {
                case mask_kind::causal:   nvis = std::max<int64_t>(0, cap - (int64_t) (nq - 1 - t) * 37 - cap / 3 - s * 29); break;
                case mask_kind::partial:  nvis = any_row(rng); break;    // anywhere, mostly mid-block
                case mask_kind::holes:    nvis = any_row(rng); break;
                case mask_kind::all_dead: nvis = 0; break;
                case mask_kind::all_live: nvis = cap; break;
            }
            if (t == 0 && tc.mk == mask_kind::partial) nvis = 64 * 3 + 1;     // one live key in group 3
            if (t == 1 && tc.mk == mask_kind::partial) nvis = 128 + 64;       // exact group boundary
            for (int64_t i = 0; i < cap; ++i) {
                bool vis = i < nvis;
                if (tc.mk == mask_kind::holes) {
                    const int64_t g = i / 64;
                    if (g % 7 == 3) vis = false;                              // whole dead 64-groups inside
                    if (g % 11 == 5 && (i % 64) != 17) vis = false;           // groups with a single live key
                    if (pct(rng) < 3) vis = !vis;                             // scattered
                }
                hm[((size_t) s * nq + t) * cap + i] = f32_to_f16_bits(vis ? 0.0f : -INFINITY);
            }
        }
    }

    // ---- device
    float * dq; char * dk; float * dw; half * dm; float * d_old; float * d_new;
    const size_t n_dst = (size_t) cap * nq * ns;
    CK(cudaMalloc(&dq, hq.size() * sizeof(float)));
    CK(cudaMalloc(&dk, hk.size()));
    CK(cudaMalloc(&dw, hw.size() * sizeof(float)));
    CK(cudaMalloc(&dm, hm.size() * sizeof(uint16_t)));
    CK(cudaMalloc(&d_old, n_dst * sizeof(float)));
    CK(cudaMalloc(&d_new, n_dst * sizeof(float)));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm, hm.data(), hm.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    // poison both outputs with different patterns: an unwritten element shows up as a mismatch
    CK(cudaMemset(d_old, 0x5A, n_dst * sizeof(float)));
    CK(cudaMemset(d_new, 0xA5, n_dst * sizeof(float)));

    // strides as ggml builds them
    const int64_t n_stream = ns, n_batch = nq, n_kv = cap;
    const size_t nb1 = (size_t) cap * sizeof(float), nb2 = nb1 * nq, nb3 = nb2;
    const size_t nbq1 = N_EMBD * sizeof(float), nbq2 = nbq1 * H, nbq3 = nbq2 * nq;
    const size_t nbk1 = ROW_BYTES, nbk2 = ROW_BYTES, nbk3 = (size_t) ROW_BYTES * cap;
    const size_t nbw1 = (size_t) H * sizeof(float), nbw2 = nbw1 * nq, nbw3 = nbw2;
    const size_t nbm1 = (size_t) cap * sizeof(uint16_t), nbm2 = nbm1 * nq, nbm3 = nbm2;
    const int64_t nem3 = nem3_m;
    const int skip = tc.skip;

    auto launch_old = [&]() {
        // ggml_cuda_lightning_indexer's vec path: 8 warps x 8 keys
        constexpr int K_VECS_PER_WARP = 8, WARPS_PER_BLOCK = 8, K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((unsigned) ((n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK), (unsigned) n_batch, (unsigned) n_stream);
        if (H == 64) {
            lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 64, GGML_TYPE_MXFP4><<<grid, block>>>(
                dq, dk, dw, dm, d_old, n_stream, n_batch, n_kv, nb1, nb2, nb3, nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3, skip);
        } else {
            lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 32, GGML_TYPE_MXFP4><<<grid, block>>>(
                dq, dk, dw, dm, d_old, n_stream, n_batch, n_kv, nb1, nb2, nb3, nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3, skip);
        }
    };
    auto launch_new = [&]() {
        if (H == 64) {
            lightning_indexer_launch_v2<64, GGML_TYPE_MXFP4>(dq, dk, dw, dm, d_new, n_stream, n_batch, n_kv,
                nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3, skip, 0);
        } else {
            lightning_indexer_launch_v2<32, GGML_TYPE_MXFP4>(dq, dk, dw, dm, d_new, n_stream, n_batch, n_kv,
                nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3, skip, 0);
        }
    };

    // correctness run
    launch_old(); CK(cudaGetLastError());
    launch_new(); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> ho(n_dst), hn(n_dst);
    CK(cudaMemcpy(ho.data(), d_old, n_dst * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hn.data(), d_new, n_dst * sizeof(float), cudaMemcpyDeviceToHost));
    bool same = std::memcmp(ho.data(), hn.data(), n_dst * sizeof(float)) == 0;
    size_t n_bad = 0, first = 0, n_finite = 0, n_nan = 0;
    for (size_t i = 0; i < n_dst; ++i) {
        float f; std::memcpy(&f, &ho[i], 4);
        n_finite += std::isfinite(f); n_nan += std::isnan(f);
        if (ho[i] != hn[i]) { if (!n_bad) first = i; ++n_bad; }
    }

    // timing: warm, then reps of each, alternated per pair
    cudaEvent_t e0, e1, e2;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
    double so = 0, sn = 0;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0));
        launch_old();
        CK(cudaEventRecord(e1));
        launch_new();
        CK(cudaEventRecord(e2));
        CK(cudaEventSynchronize(e2));
        float a, b;
        CK(cudaEventElapsedTime(&a, e0, e1));
        CK(cudaEventElapsedTime(&b, e1, e2));
        so += a; sn += b;
    }
    CK(cudaGetLastError());
    t_old_ms = so / reps; t_new_ms = sn / reps;
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2));

    const double pairs = (double) cap * nq * ns;
    printf("cap %7lld nq %4d H %2d ns %d/%d mask %-8s q %-4s %s skip %d | %s",
           (long long) cap, nq, H, ns, nem3_m, mask_name(tc.mk), tc.q_fake_quant ? "fq32" : "f32", tc.stress ? "stress" : "normal", skip,
           same ? "BIT-EXACT" : "MISMATCH ");
    printf(" (finite %zu nan %zu) | old %8.3f ms (%.3f ns/pair)  new %8.3f ms (%.3f ns/pair)  x%.2f\n",
           n_finite, n_nan, t_old_ms, t_old_ms * 1e6 / pairs, t_new_ms, t_new_ms * 1e6 / pairs, t_old_ms / t_new_ms);
    if (!same) {
        float fo, fn; std::memcpy(&fo, &ho[first], 4); std::memcpy(&fn, &hn[first], 4);
        const size_t f_kv = first % (size_t) cap, f_q = (first / (size_t) cap) % (size_t) nq, f_s = first / ((size_t) cap * nq);
        const uint16_t f_m = hm[((f_s % (size_t) nem3_m) * nq + f_q) * cap + f_kv];
        printf("    %zu mismatches; first at stream %zu query %zu key %zu: old 0x%08x (%.9g) new 0x%08x (%.9g) mask %s\n",
               n_bad, f_s, f_q, f_kv, ho[first], fo, hn[first], fn, f_m ? "-inf" : "0");
    }

    CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dw)); CK(cudaFree(dm)); CK(cudaFree(d_old)); CK(cudaFree(d_new));
    return same;
}

int main(int argc, char ** argv) {
    bool quick = false;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "quick")) quick = true;
        else if (!std::strcmp(argv[i], "reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    }

    int dev = 0; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    std::vector<test_case> cases;
    const int64_t caps[] = { 8192, 65536, 262144 };
    const int nqs[] = { 1, 7, 128, 512 };
    for (int64_t cap : caps) {
        if (quick && cap > 8192) continue;
        for (int nq : nqs) {
            cases.push_back({ cap, nq, 64, mask_kind::causal,  false, false, 1 });
            cases.push_back({ cap, nq, 64, mask_kind::partial, true,  false, 1 });
            if (cap <= 65536) {
                cases.push_back({ cap, nq, 64, mask_kind::holes,   false, false, 1 });
                cases.push_back({ cap, nq, 64, mask_kind::holes,   false, true,  1 });
            }
        }
    }
    // the rest of the switches, at 8K
    cases.push_back({ 8192, 128, 64, mask_kind::holes,    false, true,  0 });   // GGML_CUDA_INDEXER_SKIP=0
    cases.push_back({ 8192, 128, 64, mask_kind::all_dead, false, false, 1 });
    cases.push_back({ 8192, 128, 64, mask_kind::all_live, true,  false, 1 });
    cases.push_back({ 8192 + 64 * 3 + 5, 7, 64, mask_kind::partial, false, true, 1 });   // n_kv not a block multiple
    // last v2 block whose second 64-group lies wholly past n_kv (n_kv % 128 in (0, 64]), skip on and off
    for (int64_t cap : { (int64_t) 8192 + 37, (int64_t) 8192 + 64 }) {
        for (int sk : { 1, 0 }) {
            cases.push_back({ cap, 7, 64, mask_kind::partial, false, true,  sk });
            cases.push_back({ cap, 7, 64, mask_kind::holes,   true,  false, sk });
        }
    }
    // caps under one v2 block (early-prompt power-of-two caps), both head counts, skip on and off
    for (int64_t cap : { (int64_t) 1, (int64_t) 16, (int64_t) 64, (int64_t) 100 }) {
        for (int H : { 64, 32 }) {
            for (int sk : { 1, 0 }) {
                cases.push_back({ cap, 1, H, mask_kind::causal, false, false, sk });
                cases.push_back({ cap, 7, H, mask_kind::holes,  true,  true,  sk });
            }
        }
    }
    // n_stream > 1 with a broadcast mask (nem3 1) and a per-stream mask (nem3 = n_stream)
    {
        test_case a = { 8192 + 37, 7, 64, mask_kind::holes,  false, true,  1 }; a.ns = 2; a.nem3 = 1; cases.push_back(a);
        test_case b = { 8192,      7, 64, mask_kind::causal, true,  false, 1 }; b.ns = 3; b.nem3 = 3; cases.push_back(b);
        test_case c = { 8192,      7, 32, mask_kind::causal, false, false, 1 }; c.ns = 2; c.nem3 = 2; cases.push_back(c);
    }
    for (int nq : nqs) {
        cases.push_back({ 8192, nq, 32, mask_kind::holes,  false, true,  1 });  // the n_head 32 instance
        cases.push_back({ 8192, nq, 32, mask_kind::causal, true,  false, 1 });
    }

    std::mt19937_64 rng(0x1d3c5e7f9ULL);
    int n_fail = 0;
    double so = 0, sn = 0;
    for (const auto & tc : cases) {
        double a, b;
        if (!run_case(tc, reps, rng, a, b)) ++n_fail;
        so += a; sn += b;
    }
    printf("%zu cases, %d mismatched; total old %.2f ms new %.2f ms (x%.2f)\n", cases.size(), n_fail, so, sn, so / sn);
    return n_fail ? 1 : 0;
}
