// test_ds41_fuse --cuda - STRATA_FUSE: each fused path vs the stock-op chain ds41_dense.cpp builds
// without it, on the CUDA backend (the engine fuses on CUDA only), bit for bit.  GPU 0, ~1 s, a few MB of VRAM.
//   1. the router: ggml_dsv4_route_vl (vision selection bias inside the fused route) and the plain ggml_dsv4_route vs
//      softplus / sqrt / sub / repeat / mul / add / add / add / argsort_top_k / get_rows / sum_rows / scale_bias / div /
//      scale - ids and weights, NE = 384, k = 6, widths 1 / 3 / 7 / 64, image masks of 0 / 1 per row, rbias zero and
//      not, logits over many binades (softplus' x > 20 branch, ties)
//   2. the shared expert's clamped SwiGLU: ggml_swiglu_clamp vs clamp(up, -lim, lim), clamp(gate, -inf, lim),
//      swiglu_split - (a) elementwise on special values (+-inf, +-lim, +-0, subnormals, huge); (b) the whole shared
//      expert (Q5_K / Q6_K / Q8_0 gate, up and down GEMVs on a [1024, n] activation), widths 1 (ggml-cuda then fuses
//      gate + up + glu into one MMVQ kernel) and 2 / 3 / 7.  NaN inputs are left out: there the fused clamp returns the
//      limit and the stock clamp NaN (a NaN activation is a broken pass either way).
//   3. --cuda --gateup (STRATA_GATEUP_FUSE): sets STRATA_GATEUP_FUSE=1 first, so 1 and 2 also run
//      the multi-column fused kernel (widths 2 / 3 / 7 of 2b), then the shared expert at DS4.1's shape (4096 x 2048,
//      Q4_K / Q5_K / Q6_K / Q8_0, widths 1..8, a zero column) fused vs the same GLU graph with the fusion blocked (the
//      engine's path with the variable off): SwiGLU rows and down products bit for bit.
//   4. --cuda --folds (STRATA_FOLDS): (a) ds41_hc_post_add (the residual add inside the hc_post
//      kernel, src[4]) vs ggml_add + ggml_dsv4_hc_post - D 4096, HC 4, widths 1 / 2 / 5 / 8 / 64, with and without comb,
//      contiguous column views at an offset (finish's cols2) and strided views (the drafter's NU = 1 routed view), values
//      over many binades, subnormals, exact cancellations, +-inf; (b) the DSpark drafter's routed experts: mul_mat_id
//      gate / up (MXFP4 - the sidecar's type - Q4_K, Q8_0) + clamp / clamp / swiglu_split + the routing-weight mul +
//      the down mul_mat_id vs the ggml_swiglu_clamp form, which ggml-cuda runs as the fused MoE MMVQ kernel where the
//      unfused products take MMVQ too (widths 1..8, block 5 = the drafter's); (c) part 2b's shared expert at widths
//      4 / 5 / 8 as well.
//   5. --cuda --qfuse (STRATA_QFUSE; through the CUDA registry's proc addresses): (a) the BYTES -
//      rms_norm_mul_q8_1_f32 (rms_norm * w with the q8_1 image written by the same kernel) vs rms_norm * w then
//      quantize_row_q8_1_cuda with mul_mat_vec_q's arguments: the F32 outputs and the whole q8_1 images (padding
//      included, both sides poisoned differently first), widths 512 (the 256-thread kernel) / 1024 / 2304 (padded to
//      2560) / 4096 / 5120, 1 / 2 / 5 / 8 rows, inputs over many binades with zero blocks, -0, subnormals, one-hot and
//      cancelling blocks; (b) graphs as ds41_dense.cpp builds them - rms_w -> two GEMVs on the same activation, a
//      gate/up/swiglu_clamp (one fused MMVQ at width 1), and a second rms_w on a GEMV's output -> a GEMV, Q8_0 / Q4_K
//      / Q5_K / Q6_K, D 1024 / 4096, widths 1 / 2 / 5 / 8: every output with the switch forced on vs off, and the
//      images made / read counted (the fused path must actually run).
//      With --gateup too (STRATA_GATEUP_FUSE=1 is read once per process, so run both `--cuda --qfuse` and
//      `--cuda --gateup --qfuse`), (b) runs gate + up as the multi-column fused MMVQ at every width as well, and that
//      one launch must read the image (4 reads exactly: q_a, kv, gate+up, q_b), bit for bit against qfuse off.
// Exit 0 = identical.  Build: cmake --build build-ds41-portable --target test_ds41_fuse
#include "ds41_ops.hpp"

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
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const float NEG_INF = -INFINITY;

static bool same(float a, float b) {
    return std::memcmp(&a, &b, 4) == 0 || (std::isnan(a) && std::isnan(b));
}

// one graph on be: inputs set by the caller after allocation, outputs read back
struct G {
    ggml_backend_t be;
    ggml_context * c = nullptr;
    ggml_context * cw = nullptr;   // tensors with their own buffer (weights, the fused route's wts side output)
    ggml_backend_buffer_t bw = nullptr;
    ggml_gallocr_t al = nullptr;
    ggml_cgraph * gf = nullptr;
    explicit G(ggml_backend_t b) : be(b) {
        ggml_init_params ip = { ggml_tensor_overhead() * 1024 + ggml_graph_overhead(), nullptr, true };
        c = ggml_init(ip);
        cw = ggml_init(ip);
    }
    ~G() {
        if (al) ggml_gallocr_free(al);
        if (bw) ggml_backend_buffer_free(bw);
        ggml_free(c);
        ggml_free(cw);
    }
    void build(const std::vector<ggml_tensor *> & outs) {
        bw = ggml_backend_alloc_ctx_tensors(cw, be);
        gf = ggml_new_graph(c);
        for (ggml_tensor * t : outs) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
    }
};

template <class T> static std::vector<T> get(ggml_tensor * t) {
    std::vector<T> v(ggml_nbytes(t) / sizeof(T));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// ---------------------------------------------------------------- 1. the router
static int test_route(ggml_backend_t be, bool vision, int64_t n, bool rb_zero, uint64_t seed) {
    const int64_t NE = 384;
    const int K = 6;
    const float wscale = 2.5f;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 9), ex(-30, 6), bit(0, 1);
    std::vector<float> lg((size_t) (NE * n)), bt((size_t) NE), bv((size_t) NE), rb((size_t) NE), img((size_t) n);
    for (size_t i = 0; i < lg.size(); ++i) {
        switch (kind(rng)) {
            case 0:  lg[i] = 25.0f + 10.0f * std::fabs(nd(rng)); break;    // softplus' x > 20 branch
            case 1:  lg[i] = 20.0f; break;
            case 2:  lg[i] = -30.0f - std::fabs(nd(rng)); break;
            case 3:  lg[i] = std::ldexp(nd(rng), ex(rng)); break;
            case 4:  lg[i] = (float) (i % 5) - 2.0f; break;                // exact ties in the scores
            default: lg[i] = 3.0f * nd(rng); break;
        }
    }
    for (int64_t e = 0; e < NE; ++e) {
        bt[e] = 0.05f * nd(rng);
        bv[e] = (e % 7 == 0) ? bt[e] : 0.05f * nd(rng);
        rb[e] = rb_zero ? 0.0f : ((e % 3 == 0) ? -0.28f : 0.0f);
    }
    for (int64_t t = 0; t < n; ++t) img[t] = (n > 1 && bit(rng)) ? 1.0f : 0.0f;
    if (n == 1 && vision) img[0] = (seed & 1) ? 1.0f : 0.0f;

    G g(be);
    ggml_tensor * rl = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, NE, n);
    ggml_tensor * tbt = ggml_new_tensor_1d(g.c, GGML_TYPE_F32, NE);
    ggml_tensor * tbv = ggml_new_tensor_1d(g.c, GGML_TYPE_F32, NE);
    ggml_tensor * trb = ggml_new_tensor_1d(g.c, GGML_TYPE_F32, NE);
    ggml_tensor * timg = ggml_new_tensor_1d(g.c, GGML_TYPE_F32, n);
    for (ggml_tensor * t : { rl, tbt, tbv, trb, timg }) ggml_set_input(t);

    // the chain, as build_attn builds it
    ggml_tensor * sc = ggml_sqrt(g.c, ggml_softplus(g.c, rl));
    ggml_tensor * bias_sel = tbt;
    if (vision) {
        ggml_tensor * b2 = ggml_reshape_2d(g.c, tbt, NE, 1);
        ggml_tensor * v2 = ggml_reshape_2d(g.c, tbv, NE, 1);
        ggml_tensor * m = ggml_reshape_2d(g.c, timg, 1, n);
        bias_sel = ggml_add(g.c, ggml_mul(g.c, ggml_repeat(g.c, ggml_sub(g.c, v2, b2), sc), m), b2);
    }
    ggml_tensor * s0 = ggml_add(g.c, sc, bias_sel);
    ggml_tensor * sel = ggml_add(g.c, s0, ggml_reshape_2d(g.c, trb, NE, 1));
    ggml_tensor * ids = ggml_cont(g.c, ggml_argsort_top_k(g.c, sel, K));
    ggml_tensor * wts = ggml_reshape_2d(g.c, ggml_get_rows(g.c, ggml_reshape_3d(g.c, sc, 1, NE, n), ids), K, n);
    wts = ggml_div(g.c, wts, ggml_scale_bias(g.c, ggml_sum_rows(g.c, wts), 1.0f, 1e-20f));
    wts = ggml_scale(g.c, wts, wscale);

    // the fused op (writes fw as a side effect)
    ggml_tensor * fw = ggml_new_tensor_2d(g.cw, GGML_TYPE_F32, K, n);
    ggml_tensor * fids = vision ? ggml_dsv4_route_vl(g.c, rl, tbt, tbv, timg, trb, fw, K, true, wscale)
                                : ggml_dsv4_route(g.c, rl, tbt, trb, fw, K, true, wscale);
    g.build({ ids, wts, fids });
    // the plain route (vision off) never reads tbv / timg, so the graph allocator gives them no buffer: set only the
    // inputs the graph has (an earlier version: the unconditional set hit GGML_ASSERT "tensor buffer not set" at route n=1)
    auto set_in = [](ggml_tensor * t, const std::vector<float> & v) {
        if (t->buffer) ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
    };
    set_in(rl, lg);
    set_in(tbt, bt);
    set_in(tbv, bv);
    set_in(trb, rb);
    set_in(timg, img);
    ggml_backend_graph_compute(be, g.gf);

    const std::vector<int32_t> a = get<int32_t>(ids), b = get<int32_t>(fids);
    const std::vector<float> wa = get<float>(wts), wb = get<float>(fw);
    int64_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i] && same(wa[i], wb[i])) continue;
        if (++bad <= 4) std::printf("  row %zu pick %zu: chain %d %.9g fused %d %.9g\n", i / K, i % K, a[i], wa[i], b[i], wb[i]);
    }
    std::printf("route%s n=%lld rbias %s: %lld picks, %lld differ\n", vision ? "_vl" : "   ", (long long) n,
                rb_zero ? "0" : "-0.28/3", (long long) a.size(), (long long) bad);
    return bad > 0;
}

// ---------------------------------------------------------------- 2. clamped SwiGLU
static int test_glu_values(ggml_backend_t be) {
    const float lim = 10.0f;
    const int64_t nc = 4096, n = 3;
    std::mt19937_64 rng(20261116);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 12), ex(-140, 120);
    auto pick = [&](size_t i) -> float {
        switch (kind(rng)) {
            case 0:  return INFINITY;
            case 1:  return -INFINITY;
            case 2:  return lim;
            case 3:  return -lim;
            case 4:  return std::nextafter(lim, 0.0f);
            case 5:  return std::nextafter(lim, 100.0f);
            case 6:  return (i & 1) ? -0.0f : 0.0f;
            case 7:  return nd(rng) * 1e-40f;                              // subnormal
            case 8:  return std::ldexp(nd(rng), ex(rng));
            case 9:  return nd(rng) * 1e30f;
            default: return 6.0f * nd(rng);
        }
    };
    std::vector<float> gv((size_t) (nc * n)), uv((size_t) (nc * n));
    for (size_t i = 0; i < gv.size(); ++i) { gv[i] = pick(i); uv[i] = pick(i + 1); }
    G g(be);
    ggml_tensor * gt = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, nc, n);
    ggml_tensor * up = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, nc, n);
    ggml_set_input(gt);
    ggml_set_input(up);
    ggml_tensor * a = ggml_swiglu_split(g.c, ggml_clamp(g.c, gt, NEG_INF, lim), ggml_clamp(g.c, up, -lim, lim));
    ggml_tensor * b = ggml_swiglu_clamp(g.c, gt, up, lim);
    g.build({ a, b });
    ggml_backend_tensor_set(gt, gv.data(), 0, gv.size() * 4);
    ggml_backend_tensor_set(up, uv.data(), 0, uv.size() * 4);
    ggml_backend_graph_compute(be, g.gf);
    const std::vector<float> va = get<float>(a), vb = get<float>(b);
    int64_t bad = 0;
    for (size_t i = 0; i < va.size(); ++i) {
        if (same(va[i], vb[i])) continue;
        if (++bad <= 4) std::printf("  [%zu] gate %.9g up %.9g: chain %.9g fused %.9g\n", i, gv[i], uv[i], va[i], vb[i]);
    }
    std::printf("swiglu_clamp values: %lld, %lld differ\n", (long long) va.size(), (long long) bad);
    return bad > 0;
}

static std::vector<uint8_t> quant(ggml_type t, int64_t cols, int64_t rows, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> f((size_t) (cols * rows));
    for (auto & v : f) v = nd(rng);
    std::vector<uint8_t> q(ggml_row_size(t, cols) * (size_t) rows);
    ggml_quantize_chunk(t, f.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

static int test_shexp(ggml_backend_t be, ggml_type wt, int64_t n) {
    const int64_t D = 1024, F = 512;
    const float lim = 10.0f;
    G g(be);
    ggml_tensor * wu = ggml_new_tensor_2d(g.cw, wt, D, F);
    ggml_tensor * wg = ggml_new_tensor_2d(g.cw, wt, D, F);
    ggml_tensor * wd = ggml_new_tensor_2d(g.cw, wt, F, D);
    ggml_tensor * fn = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, D, n);
    ggml_set_input(fn);
    // the chain (build_shexp_ops without STRATA_FUSE) and the fused form, on the same activation
    auto shexp = [&](bool fused) {
        ggml_tensor * u = ggml_mul_mat(g.c, wu, fn);
        ggml_tensor * gg = ggml_mul_mat(g.c, wg, fn);
        if (fused) return ggml_mul_mat(g.c, wd, ggml_swiglu_clamp(g.c, gg, u, lim));
        u = ggml_clamp(g.c, u, -lim, lim);
        gg = ggml_clamp(g.c, gg, NEG_INF, lim);
        return ggml_mul_mat(g.c, wd, ggml_swiglu_split(g.c, gg, u));
    };
    ggml_tensor * a = shexp(false);
    ggml_tensor * b = shexp(true);
    g.build({ a, b });
    const std::vector<uint8_t> qu = quant(wt, D, F, 1), qg = quant(wt, D, F, 2), qd = quant(wt, F, D, 3);
    ggml_backend_tensor_set(wu, qu.data(), 0, qu.size());
    ggml_backend_tensor_set(wg, qg.data(), 0, qg.size());
    ggml_backend_tensor_set(wd, qd.data(), 0, qd.size());
    std::mt19937_64 rng(77 + (uint64_t) n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (D * n));
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 97 == 0) ? 400.0f * nd(rng) : 4.0f * nd(rng);   // some clamp hits
    ggml_backend_tensor_set(fn, x.data(), 0, x.size() * 4);
    ggml_backend_graph_compute(be, g.gf);
    const std::vector<float> va = get<float>(a), vb = get<float>(b);
    int64_t bad = 0;
    for (size_t i = 0; i < va.size(); ++i) {
        if (same(va[i], vb[i])) continue;
        if (++bad <= 4) std::printf("  [%zu] chain %.9g fused %.9g\n", i, va[i], vb[i]);
    }
    std::printf("shexp %s n=%lld: %lld outputs, %lld differ\n", ggml_type_name(wt), (long long) n, (long long) va.size(),
                (long long) bad);
    return bad > 0;
}

// ---------------------------------------------------------------- 3. STRATA_GATEUP_FUSE (--gateup)
// The shared expert at DS4.1's shape (D 4096, F 2048) on 1..8 columns, the same GLU-form graph twice: (a) as built, which
// ggml-cuda (with STRATA_GATEUP_FUSE=1) runs as ONE fused gate + up + clamped-SwiGLU MMVQ kernel at every width, and (b)
// with the up / gate products also marked as outputs, which blocks the fusion (ggml_can_fuse_subgraph): quantize + MMVQ
// twice and the GLU kernel - what the engine runs with the variable off.  The SwiGLU rows and the down products must
// match bit for bit.
static int test_gateup(ggml_backend_t be, ggml_type wt, int64_t n) {
    const int64_t D = 4096, F = 2048;
    const float lim = 10.0f;
    G g(be);
    ggml_tensor * wu = ggml_new_tensor_2d(g.cw, wt, D, F);
    ggml_tensor * wg = ggml_new_tensor_2d(g.cw, wt, D, F);
    ggml_tensor * wd = ggml_new_tensor_2d(g.cw, wt, F, D);
    ggml_tensor * fn = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, D, n);
    ggml_set_input(fn);
    ggml_tensor * ua = ggml_mul_mat(g.c, wu, fn);
    ggml_tensor * ga = ggml_mul_mat(g.c, wg, fn);
    ggml_tensor * ha = ggml_swiglu_clamp(g.c, ga, ua, lim);
    ggml_tensor * da = ggml_mul_mat(g.c, wd, ha);
    ggml_tensor * ub = ggml_mul_mat(g.c, wu, fn);
    ggml_tensor * gb = ggml_mul_mat(g.c, wg, fn);
    ggml_tensor * hb = ggml_swiglu_clamp(g.c, gb, ub, lim);
    ggml_tensor * db = ggml_mul_mat(g.c, wd, hb);
    g.build({ ha, da, hb, db, ub, gb });
    const std::vector<uint8_t> qu = quant(wt, D, F, 11), qg = quant(wt, D, F, 12), qd = quant(wt, F, D, 13);
    ggml_backend_tensor_set(wu, qu.data(), 0, qu.size());
    ggml_backend_tensor_set(wg, qg.data(), 0, qg.size());
    ggml_backend_tensor_set(wd, qd.data(), 0, qd.size());
    std::mt19937_64 rng(4242 + (uint64_t) n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (D * n));
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 89 == 0) ? 600.0f * nd(rng) : 4.0f * nd(rng);   // some clamp hits
    if (n > 2)   // a zero column: its up / gate sums are exactly 0
        for (int64_t i = 0; i < D; ++i) x[(size_t) ((n - 1) * D + i)] = 0.0f;
    ggml_backend_tensor_set(fn, x.data(), 0, x.size() * 4);
    ggml_backend_graph_compute(be, g.gf);
    int64_t bad = 0, clamped = 0;
    const std::vector<float> va = get<float>(ha), vb = get<float>(hb), wa = get<float>(da), wb = get<float>(db);
    const std::vector<float> uvb = get<float>(ub), gvb = get<float>(gb);
    for (size_t i = 0; i < va.size(); ++i) {
        clamped += std::fabs(uvb[i]) > lim || gvb[i] > lim;
        if (same(va[i], vb[i])) continue;
        if (++bad <= 4) std::printf("  h[%zu] separate %.9g fused %.9g\n", i, vb[i], va[i]);
    }
    for (size_t i = 0; i < wa.size(); ++i) {
        if (same(wa[i], wb[i])) continue;
        if (++bad <= 8) std::printf("  down[%zu] separate %.9g fused %.9g\n", i, wb[i], wa[i]);
    }
    std::printf("gateup %s n=%lld: %lld + %lld outputs (%lld clamp hits), %lld differ\n", ggml_type_name(wt), (long long) n,
                (long long) va.size(), (long long) wa.size(), (long long) clamped, (long long) bad);
    return bad > 0;
}

// ---------------------------------------------------------------- 4. STRATA_FOLDS (--folds)
// (a) hc_post(a + b, ...) as the chain (ggml_add, then ggml_dsv4_hc_post) and as ONE op (ds41_hc_post_add: b in the
// hc_post node's src[4]).  a and b are [D, n] views at column C0 of wider buffers: contiguous columns (finish's cols2 of
// shexp / routed) or, strided, every other D-row of a [D, 2, NC] buffer (the drafter's routed view of de when NU = 1).
static int test_hc_post_add(ggml_backend_t be, int64_t n, bool with_comb, bool strided, uint64_t seed) {
    const int64_t D = 4096, HC = 4, C0 = 3, NC = n + C0 + 2;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 11), ex(-140, 100);
    auto pick = [&]() -> float {
        switch (kind(rng)) {
            case 0:  return nd(rng) * 1e-40f;                     // subnormal
            case 1:  return std::ldexp(nd(rng), ex(rng));         // any binade
            case 2:  return nd(rng) * 1e30f;
            case 3:  return 0.0f;
            case 4:  return -0.0f;
            default: return 3.0f * nd(rng);
        }
    };
    const int64_t nbuf = strided ? D * 2 * NC : D * NC;
    std::vector<float> av((size_t) nbuf), bv((size_t) nbuf);
    for (size_t i = 0; i < av.size(); ++i) {
        av[i] = pick();
        switch (i % 13) {
            case 0:  bv[i] = -av[i]; break;                       // exact cancellation (a + b = +0)
            case 1:  bv[i] = std::nextafter(-av[i], 0.0f); break; // near-cancellation (a 1-ulp result)
            case 2:  bv[i] = av[i]; break;                        // a doubling (overflow to inf at 1e30 * 2^...)
            case 3:  bv[i] = (i & 1) ? INFINITY : -INFINITY; break;
            default: bv[i] = pick(); break;
        }
    }
    std::vector<float> rv((size_t) (D * HC * n)), pv((size_t) (HC * n)), cv((size_t) (HC * HC * n));
    for (auto & v : rv) v = 2.0f * nd(rng);
    for (auto & v : pv) v = 2.0f / (1.0f + std::exp(-nd(rng)));   // 2 * sigmoid, as hc_mixes' post
    for (auto & v : cv) v = 1.0f / (1.0f + std::exp(-nd(rng)));
    G g(be);
    ggml_tensor * ba = strided ? ggml_new_tensor_3d(g.c, GGML_TYPE_F32, D, 2, NC) : ggml_new_tensor_2d(g.c, GGML_TYPE_F32, D, NC);
    ggml_tensor * bb = strided ? ggml_new_tensor_3d(g.c, GGML_TYPE_F32, D, 2, NC) : ggml_new_tensor_2d(g.c, GGML_TYPE_F32, D, NC);
    ggml_tensor * res = ggml_new_tensor_3d(g.c, GGML_TYPE_F32, D, HC, n);
    ggml_tensor * post = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, HC, n);
    ggml_tensor * comb = ggml_new_tensor_3d(g.c, GGML_TYPE_F32, HC, HC, n);
    for (ggml_tensor * t : { ba, bb, res, post, comb }) ggml_set_input(t);
    auto view = [&](ggml_tensor * t) {
        const size_t rs = strided ? t->nb[2] : t->nb[1];   // the column stride
        return ggml_view_2d(g.c, t, D, n, rs, (size_t) C0 * rs + (strided ? t->nb[1] : 0));
    };
    ggml_tensor * va = view(ba), * vb = view(bb);
    ggml_tensor * cm = with_comb ? comb : nullptr;
    ggml_tensor * chain = ggml_dsv4_hc_post(g.c, ggml_add(g.c, va, vb), res, post, cm);
    ggml_tensor * fold = strata::v41::ds41_hc_post_add(g.c, va, vb, res, post, cm);
    g.build({ chain, fold });
    ggml_backend_tensor_set(ba, av.data(), 0, av.size() * 4);
    ggml_backend_tensor_set(bb, bv.data(), 0, bv.size() * 4);
    ggml_backend_tensor_set(res, rv.data(), 0, rv.size() * 4);
    ggml_backend_tensor_set(post, pv.data(), 0, pv.size() * 4);
    // without comb (with_comb off) neither form reads it, so the graph allocator gives it no buffer: set it only when
    // the graph has it (the route test's crash, same pattern)
    if (comb->buffer) ggml_backend_tensor_set(comb, cv.data(), 0, cv.size() * 4);
    ggml_backend_graph_compute(be, g.gf);
    const std::vector<float> va_ = get<float>(chain), vb_ = get<float>(fold);
    int64_t bad = 0;
    for (size_t i = 0; i < va_.size(); ++i) {
        if (same(va_[i], vb_[i])) continue;
        if (++bad <= 4) std::printf("  [%zu] chain %.9g folded %.9g\n", i, va_[i], vb_[i]);
    }
    std::printf("hc_post_add n=%lld comb %d %s: %lld outputs, %lld differ\n", (long long) n, (int) with_comb,
                strided ? "strided" : "columns", (long long) va_.size(), (long long) bad);
    return bad > 0;
}

// (b) the DSpark drafter's routed experts (ds41_dspark.cpp): gate / up mul_mat_id, clamped SwiGLU, the routing weights,
// the down mul_mat_id - the clamp / clamp / swiglu_split chain vs ggml_swiglu_clamp (the {MUL_MAT_ID, MUL_MAT_ID, GLU}
// group ggml-cuda fuses into one MoE MMVQ kernel at widths up to get_mmvq_mmid_max_batch, where the chain's products
// are MMVQ too).  The GLU rows and the down products bit for bit.
static int test_dspark_routed(ggml_backend_t be, ggml_type wt, int64_t n) {
    const int64_t D = 2048, F = 1024, NE = 8, NU = 4;
    const float lim = 10.0f;
    G g(be);
    ggml_tensor * w1 = ggml_new_tensor_3d(g.cw, wt, D, F, NE);
    ggml_tensor * w3 = ggml_new_tensor_3d(g.cw, wt, D, F, NE);
    ggml_tensor * w2 = ggml_new_tensor_3d(g.cw, wt, F, D, NE);
    ggml_tensor * fn = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, D, n);
    ggml_tensor * ids = ggml_new_tensor_2d(g.c, GGML_TYPE_I32, NU, n);
    ggml_tensor * wts = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, NU, n);
    for (ggml_tensor * t : { fn, ids, wts }) ggml_set_input(t);
    ggml_tensor * x3 = ggml_reshape_3d(g.c, fn, D, 1, n);
    auto routed = [&](bool fold, ggml_tensor ** glu_out) {
        ggml_tensor * ge = ggml_mul_mat_id(g.c, w1, x3, ids);   // [F, NU, n]
        ggml_tensor * ue = ggml_mul_mat_id(g.c, w3, x3, ids);
        ggml_tensor * glu;
        if (fold) glu = ggml_swiglu_clamp(g.c, ge, ue, lim);
        else {
            ue = ggml_clamp(g.c, ue, -lim, lim);
            ge = ggml_clamp(g.c, ge, NEG_INF, lim);
            glu = ggml_swiglu_split(g.c, ge, ue);
        }
        *glu_out = glu;
        ggml_tensor * act = ggml_mul(g.c, glu, ggml_reshape_3d(g.c, wts, 1, NU, n));
        return ggml_mul_mat_id(g.c, w2, act, ids);              // [D, NU, n]
    };
    ggml_tensor * ha, * hb;
    ggml_tensor * da = routed(false, &ha);
    ggml_tensor * db = routed(true, &hb);
    g.build({ ha, da, hb, db });
    const std::vector<uint8_t> q1 = quant(wt, D, F * NE, 21), q3 = quant(wt, D, F * NE, 23), q2 = quant(wt, F, D * NE, 22);
    ggml_backend_tensor_set(w1, q1.data(), 0, q1.size());
    ggml_backend_tensor_set(w3, q3.data(), 0, q3.size());
    ggml_backend_tensor_set(w2, q2.data(), 0, q2.size());
    std::mt19937_64 rng(9000 + (uint64_t) n * 31 + (uint64_t) wt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (D * n)), wv((size_t) (NU * n));
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 89 == 0) ? 600.0f * nd(rng) : 4.0f * nd(rng);   // some clamp hits
    for (auto & v : wv) v = 2.5f / (1.0f + std::exp(-nd(rng)));
    std::vector<int32_t> iv((size_t) (NU * n));
    for (int64_t t = 0; t < n; ++t) {   // NU distinct experts per column
        std::vector<int32_t> perm((size_t) NE);
        for (int32_t e = 0; e < (int32_t) NE; ++e) perm[(size_t) e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t u = 0; u < NU; ++u) iv[(size_t) (t * NU + u)] = perm[(size_t) u];
    }
    ggml_backend_tensor_set(fn, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * 4);
    ggml_backend_tensor_set(wts, wv.data(), 0, wv.size() * 4);
    ggml_backend_graph_compute(be, g.gf);
    const std::vector<float> va = get<float>(ha), vb = get<float>(hb), wa = get<float>(da), wb = get<float>(db);
    int64_t bad = 0;
    for (size_t i = 0; i < va.size(); ++i) {
        if (same(va[i], vb[i])) continue;
        if (++bad <= 4) std::printf("  glu[%zu] chain %.9g folded %.9g\n", i, va[i], vb[i]);
    }
    for (size_t i = 0; i < wa.size(); ++i) {
        if (same(wa[i], wb[i])) continue;
        if (++bad <= 8) std::printf("  down[%zu] chain %.9g folded %.9g\n", i, wa[i], wb[i]);
    }
    std::printf("dspark routed %s n=%lld: %lld + %lld outputs, %lld differ\n", ggml_type_name(wt), (long long) n,
                (long long) va.size(), (long long) wa.size(), (long long) bad);
    return bad > 0;
}

// ---------------------------------------------------------------- 5. STRATA_QFUSE (--qfuse)
#if defined(GGML_USE_CUDA)
typedef int  (*qfuse_compare_fn)(int, const float *, const float *, int, int, float, float *, float *, void *, void *);
typedef void (*qfuse_mode_fn)(int);
typedef void (*qfuse_counters_fn)(int64_t *, int64_t *);

struct QfuseHooks {
    qfuse_compare_fn  compare  = nullptr;
    qfuse_mode_fn     mode     = nullptr;
    qfuse_counters_fn counters = nullptr;
    bool load() {
        ggml_backend_reg_t reg = ggml_backend_cuda_reg();
        compare  = (qfuse_compare_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_cuda_qfuse_compare_rms");
        mode     = (qfuse_mode_fn)     ggml_backend_reg_get_proc_address(reg, "ggml_cuda_qfuse_set_mode");
        counters = (qfuse_counters_fn) ggml_backend_reg_get_proc_address(reg, "ggml_cuda_qfuse_counters");
        return compare && mode && counters;
    }
};

// activations as the decode pass sees them, plus the q8_1 corner cases: whole zero blocks (amax = 0), -0, subnormals,
// a block with one large value (d from it, the rest round to 0 / +-1), a cancelling block (sum ~ 0), huge values
static std::vector<float> qfuse_input(int64_t ncols, int64_t nrows, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 11), ex(-40, 12);
    std::vector<float> x((size_t) (ncols * nrows));
    for (int64_t b = 0; b < ncols * nrows / 32; ++b) {
        float * p = x.data() + b * 32;
        const int k = kind(rng);
        for (int j = 0; j < 32; ++j) {
            switch (k) {
                case 0:  p[j] = 0.0f; break;
                case 1:  p[j] = (j & 1) ? -0.0f : 0.0f; break;
                case 2:  p[j] = nd(rng) * 1e-39f; break;                              // subnormal
                case 3:  p[j] = j == 7 ? 300.0f * nd(rng) : 0.01f * nd(rng); break;    // one hot
                case 4:  p[j] = (j < 16 ? 1.0f : -1.0f) * (1.0f + (float) (j % 16) / 64.0f); break;   // cancelling
                case 5:  p[j] = std::ldexp(nd(rng), ex(rng)); break;
                case 6:  p[j] = nd(rng) * 1e4f; break;
                default: p[j] = nd(rng); break;
            }
        }
    }
    return x;
}

static int test_qfuse_bytes(const QfuseHooks & h, int64_t ncols, int64_t nrows, uint64_t seed) {
    const float eps = 1e-6f;
    std::vector<float> x = qfuse_input(ncols, nrows, seed), w((size_t) ncols);
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);
    std::normal_distribution<float> nd(1.0f, 0.3f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (i % 61 == 0) ? 0.0f : nd(rng);
    const int64_t pad = (ncols + 511) / 512 * 512;
    const size_t nq = (size_t) (nrows * (pad / 32) * 36);
    std::vector<float> fa(x.size()), fb(x.size());
    std::vector<uint8_t> qa(nq), qb(nq);
    if (h.compare(0, x.data(), w.data(), (int) ncols, (int) nrows, eps, fa.data(), fb.data(), qa.data(), qb.data()) != 0) {
        std::printf("qfuse bytes %lld x %lld: CUDA error\n", (long long) ncols, (long long) nrows);
        return 1;
    }
    int64_t badf = 0, badq = 0;
    for (size_t i = 0; i < fa.size(); ++i) badf += std::memcmp(&fa[i], &fb[i], 4) != 0;
    for (size_t i = 0; i < nq; ++i) {
        if (qa[i] == qb[i]) continue;
        if (++badq <= 4) std::printf("  row %zu block %zu byte %zu: fused %02x separate %02x\n", i / (nq / (size_t) nrows),
                                     (i % (nq / (size_t) nrows)) / 36, i % 36, qa[i], qb[i]);
    }
    std::printf("qfuse bytes %lld x %lld: %zu q8_1 bytes, %lld differ; %zu F32 outputs, %lld differ\n", (long long) ncols,
                (long long) nrows, nq, (long long) badq, fa.size(), (long long) badf);
    return badf > 0 || badq > 0;
}

// one ds41_dense-shaped graph, computed with the switch forced to `on`; returns the outputs in order
static std::vector<std::vector<float>> qfuse_graph(ggml_backend_t be, const QfuseHooks & h, bool on, ggml_type wt, int64_t D,
                                                   int64_t n, int64_t * made, int64_t * hits) {
    const int64_t R = 512, F = 256;   // q_a-like rank and a shared-expert width
    const float eps = 1e-6f, lim = 10.0f;
    G g(be);
    ggml_tensor * wn  = ggml_new_tensor_1d(g.cw, GGML_TYPE_F32, D);
    ggml_tensor * wn2 = ggml_new_tensor_1d(g.cw, GGML_TYPE_F32, R);
    ggml_tensor * wa  = ggml_new_tensor_2d(g.cw, wt, D, R);   // attn_q_a-like
    ggml_tensor * wk  = ggml_new_tensor_2d(g.cw, wt, D, F);   // attn_kv-like, same input
    ggml_tensor * wg  = ggml_new_tensor_2d(g.cw, wt, D, F);   // shared expert gate / up
    ggml_tensor * wu  = ggml_new_tensor_2d(g.cw, wt, D, F);
    ggml_tensor * wb  = ggml_new_tensor_2d(g.cw, wt, R, D);   // attn_q_b-like on the second norm
    ggml_tensor * x   = ggml_new_tensor_3d(g.c, GGML_TYPE_F32, D, 2, n);   // a strided [D, n] view, as hc_pre's input
    ggml_set_input(x);
    ggml_tensor * xv = ggml_view_2d(g.c, x, D, n, x->nb[2], 0);
    ggml_tensor * xn = ggml_mul(g.c, ggml_rms_norm(g.c, ggml_cont(g.c, xv), eps), wn);
    ggml_tensor * ya = ggml_mul_mat(g.c, wa, xn);
    ggml_tensor * yk = ggml_mul_mat(g.c, wk, xn);
    ggml_tensor * ys = ggml_swiglu_clamp(g.c, ggml_mul_mat(g.c, wg, xn), ggml_mul_mat(g.c, wu, xn), lim);
    ggml_tensor * qr = ggml_mul(g.c, ggml_rms_norm(g.c, ya, eps), wn2);
    ggml_tensor * yb = ggml_mul_mat(g.c, wb, qr);
    g.build({ xn, ya, yk, ys, qr, yb });
    std::mt19937_64 rng(1234 + (uint64_t) n);
    std::normal_distribution<float> nd(1.0f, 0.2f);
    std::vector<float> wnv((size_t) D), wn2v((size_t) R);
    for (auto & v : wnv) v = nd(rng);
    for (auto & v : wn2v) v = nd(rng);
    ggml_backend_tensor_set(wn, wnv.data(), 0, wnv.size() * 4);
    ggml_backend_tensor_set(wn2, wn2v.data(), 0, wn2v.size() * 4);
    uint64_t s = 11;
    for (ggml_tensor * t : { wa, wk, wg, wu, wb }) {
        const std::vector<uint8_t> q = quant(wt, t->ne[0], t->ne[1], s++);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
    const std::vector<float> xs = qfuse_input(D, 2 * n, 99 + (uint64_t) n);
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * 4);
    int64_t m0, h0, m1, h1;
    h.counters(&m0, &h0);
    h.mode(on ? 1 : 0);
    ggml_backend_graph_compute(be, g.gf);
    ggml_backend_synchronize(be);
    h.mode(-1);
    h.counters(&m1, &h1);
    *made = m1 - m0;
    *hits = h1 - h0;
    std::vector<std::vector<float>> out;
    for (ggml_tensor * t : { xn, ya, yk, ys, qr, yb }) out.push_back(get<float>(t));
    return out;
}

static int test_qfuse_graph(ggml_backend_t be, const QfuseHooks & h, ggml_type wt, int64_t D, int64_t n, bool gateup) {
    int64_t m_off, h_off, m_on, h_on;
    const auto a = qfuse_graph(be, h, false, wt, D, n, &m_off, &h_off);
    const auto b = qfuse_graph(be, h, true, wt, D, n, &m_on, &h_on);
    static const char * names[] = { "xn", "q_a", "kv", "shexp", "qr", "q_b" };
    int64_t bad = 0, total = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        int64_t bk = 0;
        for (size_t i = 0; i < a[k].size(); ++i) bk += !same(a[k][i], b[k][i]);
        if (bk) std::printf("  %s: %lld of %zu differ\n", names[k], (long long) bk, a[k].size());
        bad += bk;
        total += (int64_t) a[k].size();
    }
    // on: 2 images (xn, qr) whenever mul_mat_vec_q takes these GEMVs; xn feeds q_a, kv, gate and up (gate + up is one
    // fused MMVQ call at width 1 - 3 reads - else 4), qr feeds q_b
    // With STRATA_GATEUP_FUSE=1 (--gateup) gate + up is one fused MMVQ call at every width (the four test types are
    // ggml_cuda_mmvq_fuse_multi_type's), and that call must read xn's image too: exactly 4 reads (q_a, kv, gate+up, q_b);
    // both runs then take the fused gate+up kernel, so `off` vs `on` = the fused launch quantizing vs reading the image.
    const bool mmvq_expected = m_on > 0;
    const bool path_ok = m_off == 0 && h_off == 0 && (!mmvq_expected || (gateup ? h_on == 4 : h_on >= 4));
    std::printf("qfuse graph %s D=%lld n=%lld%s: %lld outputs, %lld differ; images made %lld, read %lld (off: %lld / %lld)%s\n",
                ggml_type_name(wt), (long long) D, (long long) n, gateup ? " gateup" : "", (long long) total, (long long) bad, (long long) m_on,
                (long long) h_on, (long long) m_off, (long long) h_off,
                path_ok ? (mmvq_expected ? "" : " - NOT EXERCISED (no MMVQ consumer)") : " - COUNTERS WRONG");
    return bad > 0 || !path_ok || !mmvq_expected;
}
#endif

int main(int argc, char ** argv) {
    bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    bool gateup = false, folds = false, qfuse = false;   // in any order after --cuda
    for (int i = 2; i < argc; ++i) {
        gateup = gateup || std::string(argv[i]) == "--gateup";
        folds = folds || std::string(argv[i]) == "--folds";
        qfuse = qfuse || std::string(argv[i]) == "--qfuse";
    }
    if (gateup) {   // before the backend exists: ggml-cuda reads it once (ggml_cuda_gateup_fuse_on)
#if defined(_WIN32)
        _putenv("STRATA_GATEUP_FUSE=1");
#else
        setenv("STRATA_GATEUP_FUSE", "1", 1);
#endif
    }
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) {
        std::printf("test_ds41_fuse: needs --cuda and a CUDA build (STRATA_FUSE is CUDA only)\n");
        return 2;
    }
    int fails = 0;
    uint64_t seed = 1;
    for (bool vision : { true, false })
        for (int64_t n : { (int64_t) 1, (int64_t) 3, (int64_t) 7, (int64_t) 64 })
            for (bool rbz : { true, false }) fails += test_route(be, vision, n, rbz, seed++);
    fails += test_glu_values(be);
    for (ggml_type t : { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 })
        for (int64_t n : { (int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7 }) fails += test_shexp(be, t, n);
    if (gateup)
        for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 })
            for (int64_t n = 1; n <= 8; ++n) fails += test_gateup(be, t, n);
    if (folds) {
        uint64_t fseed = 100;
        for (int64_t n : { (int64_t) 1, (int64_t) 2, (int64_t) 5, (int64_t) 8, (int64_t) 64 })
            for (bool cm : { true, false })
                for (bool st : { false, true }) fails += test_hc_post_add(be, n, cm, st, fseed++);
        for (ggml_type t : { GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 })
            for (int64_t n = 1; n <= 8; ++n) fails += test_dspark_routed(be, t, n);
        for (ggml_type t : { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 })
            for (int64_t n : { (int64_t) 4, (int64_t) 5, (int64_t) 8 }) fails += test_shexp(be, t, n);
    }
#if defined(GGML_USE_CUDA)
    if (qfuse) {
        QfuseHooks h;
        if (!h.load()) {
            std::printf("qfuse: the CUDA backend has no STRATA_QFUSE hooks (xolotl-ggml.patch without the qfuse hooks?)\n");
            ++fails;
        } else {
            uint64_t qs = 1;
            for (int64_t nc : { (int64_t) 512, (int64_t) 1024, (int64_t) 2304, (int64_t) 4096, (int64_t) 5120 })
                for (int64_t nr : { (int64_t) 1, (int64_t) 2, (int64_t) 5, (int64_t) 8 }) fails += test_qfuse_bytes(h, nc, nr, qs++);
            for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K })
                for (int64_t D : { (int64_t) 1024, (int64_t) 4096 })
                    for (int64_t n : { (int64_t) 1, (int64_t) 2, (int64_t) 5, (int64_t) 8 }) fails += test_qfuse_graph(be, h, t, D, n, gateup);
        }
    }
#endif
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "fused == chain, bit for bit");
    return fails ? 1 : 0;
}
