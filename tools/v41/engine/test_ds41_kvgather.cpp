// test_ds41_kvgather --cuda - STRATA_KV_GATHER (P3 A6): GGML_OP_DSV41_KV_GATHER (ggml_dsv41_kv_gather keys,
// ggml_dsv41_kv_mask mask) vs the stock chain ds41_dense.cpp build_attn builds without it, on the CUDA backend, byte for
// byte: the flash-attn's F16 key block [window | top-k | +0 pad] and its F16 mask [window | top-k | -inf pad].  Both
// stock variants are checked: f16 (rows_as gathers straight to F16 + zero-row pad gather) and the F32 one (F32 gathers,
// ggml_pad, then the cast to F16 - what a pass wider than the zero index takes).  Cases: ring F32 / F16 / BF16, the
// compressed rows NVFP4 / BF16 / F32 / F16 or none (window-only layers), K = 512 / 64 / 128 (no padding needed), pass
// widths 1 / 4 / 7 / 300, window indices into both the ring and this pass's rows, the selection and both masks as
// strided views (as the engine's IS.sel / i_wmask columns are).  Exit 0 = identical.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(GGML_DSV41_KV_GATHER)
static const float NEG_INF = -INFINITY;

// ds41_dense.cpp rows_as: get_rows whose result is F16 (f16) or the usual F32
static ggml_tensor * rows_as(ggml_context * gc, ggml_tensor * a, ggml_tensor * idx, bool f16) {
    ggml_tensor * t = ggml_get_rows(gc, a, idx);
    if (f16 && t->type == GGML_TYPE_F32) {
        t->type = GGML_TYPE_F16;
        t->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
    return t;
}

static uint16_t bf16_of(float f) {   // round to nearest even (finite inputs)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

// a [DH, rows] tensor's bytes in type t, values spread over many binades (F16 overflow / subnormal edges included)
static std::vector<uint8_t> synth(ggml_type t, int64_t DH, int64_t rows, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> ex(-26, 17), rbyte(0, 255), scale(0x20, 0x50);
    const size_t rb = ggml_row_size(t, DH);
    std::vector<uint8_t> out(rb * (size_t) rows);
    if (t == GGML_TYPE_NVFP4) {
        for (auto & v : out) v = (uint8_t) rbyte(rng);
        const size_t bs = ggml_type_size(t);   // block: 4 UE4M3 scales, then 32 bytes of E2M1 pairs
        for (size_t o = 0; o + bs <= out.size(); o += bs)
            for (int s = 0; s < 4; ++s) out[o + (size_t) s] = (uint8_t) scale(rng);
        return out;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float rs = std::ldexp(1.0f, ex(rng));
        for (int64_t i = 0; i < DH; ++i) {
            float f = nd(rng) * rs;
            if (i % 61 == 7) f = 0.0f;
            if (i % 67 == 9) f = -0.0f;
            if (i % 71 == 3) f = 65519.0f * (i & 1 ? -1.0f : 1.0f);   // rounds to F16 max / inf edge
            if (i % 73 == 5) f = 3.0e-8f;                             // F16 subnormal
            uint8_t * p = out.data() + (size_t) r * rb;
            if (t == GGML_TYPE_F32) {
                std::memcpy(p + (size_t) i * 4, &f, 4);
            } else if (t == GGML_TYPE_BF16) {
                const uint16_t h = bf16_of(f);
                std::memcpy(p + (size_t) i * 2, &h, 2);
            } else {   // F16 from the CPU converter (values in range)
                const ggml_fp16_t h = ggml_fp32_to_fp16(std::fmin(std::fmax(f, -65504.0f), 65504.0f));
                std::memcpy(p + (size_t) i * 2, &h, 2);
            }
        }
    }
    return out;
}

struct Case { ggml_type ring, comp; int64_t K, n; };   // comp == GGML_TYPE_COUNT: window only

static const char * tn(ggml_type t) { return t == GGML_TYPE_COUNT ? "none" : ggml_type_name(t); }

static int run(ggml_backend_t be, const Case & cs) {
    const int64_t DH = 512, SWA = 128, RAW = 128, NCOMP = 1500, n = cs.n;
    const bool has_comp = cs.comp != GGML_TYPE_COUNT;
    const int64_t K = has_comp ? cs.K : 0, KS = K + 5;            // sel / selm rows are strided (ITOP > K)
    const int64_t M0 = SWA + K, MP = (M0 + 255) / 256 * 256, pad = MP - M0;
    const int64_t C0 = 2, NC = n + 3;                             // the masks / indices are columns C0.. of wider tensors

    ggml_init_params ip = { ggml_tensor_overhead() * 1024 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * ring = ggml_new_tensor_2d(c, cs.ring, DH, RAW);
    ggml_tensor * rows = ggml_new_tensor_2d(c, GGML_TYPE_F32, DH, n);
    ggml_tensor * widx_all = ggml_new_tensor_1d(c, GGML_TYPE_I32, SWA * NC);
    ggml_tensor * wmask_all = ggml_new_tensor_2d(c, GGML_TYPE_F32, SWA, NC);
    ggml_tensor * comp = has_comp ? ggml_new_tensor_2d(c, cs.comp, DH, NCOMP) : nullptr;
    ggml_tensor * sel_all = has_comp ? ggml_new_tensor_2d(c, GGML_TYPE_I32, KS, NC) : nullptr;
    ggml_tensor * selm_all = has_comp ? ggml_new_tensor_2d(c, GGML_TYPE_F32, KS, NC) : nullptr;
    ggml_tensor * zpad16 = ggml_new_tensor_2d(c, GGML_TYPE_F16, DH, 1);
    ggml_tensor * zidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::max<int64_t>(1, pad * n));
    for (ggml_tensor * t : { ring, rows, widx_all, wmask_all, comp, sel_all, selm_all, zpad16, zidx }) if (t) ggml_set_input(t);

    ggml_tensor * widx = ggml_view_1d(c, widx_all, SWA * n, (size_t) C0 * SWA * 4);
    ggml_tensor * Mw = ggml_view_2d(c, wmask_all, SWA, n, wmask_all->nb[1], (size_t) C0 * wmask_all->nb[1]);
    ggml_tensor * sel = has_comp ? ggml_view_2d(c, sel_all, K, n, sel_all->nb[1], (size_t) C0 * sel_all->nb[1]) : nullptr;
    ggml_tensor * selm = has_comp ? ggml_view_2d(c, selm_all, K, n, selm_all->nb[1], (size_t) C0 * selm_all->nb[1]) : nullptr;

    // ---- the stock chain (ds41_dense.cpp build_attn without STRATA_KV_GATHER), f16 = true and f16 = false
    ggml_tensor * stock_k[2], * stock_m[2];
    for (int v = 0; v < 2; ++v) {
        const bool f16 = v == 0;
        ggml_tensor * wsrc = ggml_concat(c, ring->type == GGML_TYPE_F32 ? ring : ggml_cast(c, ring, GGML_TYPE_F32), rows, 1);
        ggml_tensor * Kw = ggml_reshape_3d(c, rows_as(c, wsrc, widx, f16), DH, SWA, n);
        ggml_tensor * Kall = Kw, * Mall = Mw;
        if (has_comp) {
            ggml_tensor * Kc = ggml_reshape_3d(c, rows_as(c, comp, ggml_cont(c, ggml_reshape_1d(c, ggml_cont(c, sel), K * n)), f16), DH, K, n);
            Kall = ggml_concat(c, Kw, Kc, 1);
            Mall = ggml_concat(c, Mw, ggml_cont(c, selm), 0);
        }
        if (pad) {
            Kall = f16 ? ggml_concat(c, Kall, ggml_reshape_3d(c, rows_as(c, zpad16, ggml_view_1d(c, zidx, pad * n, 0), true), DH, pad, n), 1)
                       : ggml_pad(c, Kall, 0, (int) pad, 0, 0);
            Mall = ggml_concat(c, Mall, ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        }
        stock_k[v] = Kall->type == GGML_TYPE_F16 ? ggml_cont(c, Kall) : ggml_cast(c, Kall, GGML_TYPE_F16);
        stock_m[v] = ggml_cast(c, Mall, GGML_TYPE_F16);
    }
    // ---- the op
    ggml_tensor * wi = ggml_reshape_2d(c, widx, SWA, n);
    ggml_tensor * new_k = ggml_dsv41_kv_gather(c, ring, rows, wi, comp, sel, MP);
    ggml_tensor * new_m = ggml_dsv41_kv_mask(c, Mw, selm, MP);

    ggml_tensor * outs[6] = { stock_k[0], stock_m[0], stock_k[1], stock_m[1], new_k, new_m };
    for (ggml_tensor * t : outs) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 1024, false);
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 1; }

    std::mt19937_64 rng(20261113 + (uint64_t) n * 131 + (uint64_t) K);
    const std::vector<uint8_t> ring_b = synth(cs.ring, DH, RAW, rng()), rows_b = synth(GGML_TYPE_F32, DH, n, rng());
    ggml_backend_tensor_set(ring, ring_b.data(), 0, ring_b.size());
    ggml_backend_tensor_set(rows, rows_b.data(), 0, rows_b.size());
    std::uniform_int_distribution<int> wpick(0, (int) (RAW + n - 1)), cpick(0, (int) NCOMP - 1), coin(0, 3);
    std::vector<int32_t> wi_h((size_t) (SWA * NC));
    for (auto & v : wi_h) v = coin(rng) == 0 ? (int32_t) (RAW + (wpick(rng) % n)) : wpick(rng);   // many new-row hits
    ggml_backend_tensor_set(widx_all, wi_h.data(), 0, wi_h.size() * 4);
    auto mask_vals = [&](int64_t cnt) {
        std::vector<float> m((size_t) cnt);
        for (auto & v : m) { const int k = coin(rng); v = k == 0 ? NEG_INF : k == 1 ? -0.0f : k == 2 ? 0.0f : -1.5e-3f * (float) (1 + cpick(rng)); }
        return m;
    };
    const std::vector<float> wm = mask_vals(SWA * NC);
    ggml_backend_tensor_set(wmask_all, wm.data(), 0, wm.size() * 4);
    if (has_comp) {
        const std::vector<uint8_t> comp_b = synth(cs.comp, DH, NCOMP, rng());
        ggml_backend_tensor_set(comp, comp_b.data(), 0, comp_b.size());
        std::vector<int32_t> s((size_t) (KS * NC));
        for (auto & v : s) v = cpick(rng);
        ggml_backend_tensor_set(sel_all, s.data(), 0, s.size() * 4);
        const std::vector<float> sm = mask_vals(KS * NC);
        ggml_backend_tensor_set(selm_all, sm.data(), 0, sm.size() * 4);
    }
    if (pad) {                                                    // only then are zpad16 / zidx in gf (allocated)
        const std::vector<uint8_t> z16((size_t) DH * 2, 0);
        ggml_backend_tensor_set(zpad16, z16.data(), 0, z16.size());
        const std::vector<int32_t> zi((size_t) zidx->ne[0], 0);
        ggml_backend_tensor_set(zidx, zi.data(), 0, zi.size() * 4);
    }
    ggml_backend_graph_compute(be, gf);

    auto bytes = [](ggml_tensor * t) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        return b;
    };
    const std::vector<uint8_t> nk = bytes(new_k), nm = bytes(new_m);
    int bad = 0;
    for (int v = 0; v < 2; ++v) {
        const std::vector<uint8_t> sk = bytes(stock_k[v]), sm = bytes(stock_m[v]);
        const bool ok_k = sk.size() == nk.size() && std::memcmp(sk.data(), nk.data(), sk.size()) == 0;
        const bool ok_m = sm.size() == nm.size() && std::memcmp(sm.data(), nm.data(), sm.size()) == 0;
        if (!ok_k || !ok_m) {
            ++bad;
            size_t first = 0;
            if (!ok_k) while (first < std::min(sk.size(), nk.size()) && sk[first] == nk[first]) ++first;
            std::printf("  stock %s: keys %s (first differing byte %zu of %zu), mask %s\n", v ? "f32" : "f16",
                        ok_k ? "equal" : "DIFFER", first, sk.size(), ok_m ? "equal" : "DIFFER");
        }
    }
    std::printf("ring %-4s comp %-5s K %3lld n %3lld (M %lld -> %lld): %s\n", tn(cs.ring), tn(cs.comp), (long long) K,
                (long long) n, (long long) M0, (long long) MP, bad ? "DIFFER" : "identical (both stock variants)");
    ggml_gallocr_free(al);
    ggml_free(c);
    return bad ? 1 : 0;
}
#endif

int main(int argc, char ** argv) {
#if !defined(GGML_DSV41_KV_GATHER)
    (void) argc; (void) argv;
    std::printf("this ggml has no GGML_OP_DSV41_KV_GATHER (tools/xolotl/ggml/xolotl-ggml.patch)\n");
    return 2;
#else
    const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { std::printf("needs --cuda (GGML_OP_DSV41_KV_GATHER is CUDA only)\n"); return 2; }
    int fails = 0;
    for (ggml_type ring : { GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F16 })
        for (ggml_type comp : { GGML_TYPE_NVFP4, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_COUNT })
            for (int64_t K : { (int64_t) 512, (int64_t) 64, (int64_t) 128 })
                for (int64_t n : { (int64_t) 1, (int64_t) 4, (int64_t) 7, (int64_t) 300 }) {
                    if (comp == GGML_TYPE_COUNT && K != 512) continue;   // K is unused without compressed rows
                    fails += run(be, Case { ring, comp, K, n });
                }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "kv_gather + kv_mask == stock chain, byte for byte");
    return fails ? 1 : 0;
#endif
}
