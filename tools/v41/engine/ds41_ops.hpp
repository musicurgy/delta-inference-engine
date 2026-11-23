// tools/v41/engine/ds41_ops.hpp - DeepSeek-V4.1's activation rounding as plain ggml ops (any backend).
//
// model.py quantize-dequantizes three activations in place (kernel.py act_quant / fp4_act_quant, our torch stand-in
// tools/v41/torch_kernels/kernel.py documents the rounding):
//   the window K           FP8 E4M3, blocks of 32, power-of-two scale 2^ceil(log2(amax/448)), amax >= 1e-4
//   the compressed KV      FP4 E2M1, blocks of 16, E4M3 scale e4m3(amax/6), amax >= 6*2^-9
//   indexer q and k        FP4 E2M1, blocks of 32, power-of-two scale 2^ceil(log2(amax/6)), amax >= 6*2^-126
// The model was trained with these (DwarfStar emulates them too); dropping them moved 3-layer outputs by cos 1.6e-3
// and the indexer's picks (2026-10-10 measurement), so the engine reproduces them.
//
// No custom op: every step is an exact float trick on stock ggml ops, each its own node (so no FMA contraction):
//   round_ne(y)     = (y + 1.5*2^23) - 1.5*2^23          round half to even, |y| < 2^22
//   near_pow2(a)    = Veltkamp split to 1 significant bit (g = a*(2^23+1); hi = g + (a - g))
//   pow2_ceil(a)    = near * (1 + step(a - near));   pow2_floor(a) = ceil * (1 - step(ceil - a)/2)
//   round to a float format with M mantissa bits: p = max(pow2_floor(|v|), min_normal);
//                   q = round_ne(v / p * 2^M) * p / 2^M   (the subnormal step falls out of the min_normal floor)
// RNE matches the hardware conversion torch uses, including E2M1's ties-to-even-code rule.
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdlib>

namespace strata::v41 {

enum class Ds41Quant { FP8_POW2_32, FP4_E4M3_16, FP4_POW2_32 };

inline ggml_tensor * ds41_round_ne(ggml_context * c, ggml_tensor * y) {
    const float C = 12582912.0f;   // 1.5 * 2^23
    return ggml_scale_bias(c, ggml_scale_bias(c, y, 1.0f, C), 1.0f, -C);
}

inline ggml_tensor * ds41_pow2_ceil(ggml_context * c, ggml_tensor * a) {
    ggml_tensor * g = ggml_scale(c, a, 8388609.0f);                       // a * (2^23 + 1)
    ggml_tensor * hi = ggml_add(c, g, ggml_sub(c, a, g));                 // a rounded to 1 significant bit
    return ggml_mul(c, hi, ggml_scale_bias(c, ggml_step(c, ggml_sub(c, a, hi)), 1.0f, 1.0f));
}

inline ggml_tensor * ds41_pow2_floor(ggml_context * c, ggml_tensor * a) {
    ggml_tensor * f = ds41_pow2_ceil(c, a);
    return ggml_mul(c, f, ggml_scale_bias(c, ggml_step(c, ggml_sub(c, f, a)), -0.5f, 1.0f));
}

/// v (already within +-maxv) rounded to the nearest value of a float format with `mbits` mantissa bits and smallest
/// normal `min_normal` (E4M3: 3, 2^-6; E2M1: 1, 1.0).
inline ggml_tensor * ds41_round_fmt(ggml_context * c, ggml_tensor * v, int mbits, float min_normal) {
    ggml_tensor * p = ggml_clamp(c, ds41_pow2_floor(c, ggml_abs(c, v)), min_normal, INFINITY);
    const float k = (float) (1 << mbits);
    ggml_tensor * y = ggml_scale(c, ggml_div(c, v, p), k);
    return ggml_mul(c, ggml_scale(c, ds41_round_ne(c, y), 1.0f / k), p);
}

/// x: F32 with ggml_nelements(x) % block == 0, contiguous; returns the quantize-dequantized values, x's shape.
inline ggml_tensor * ds41_fake_quant(ggml_context * c, ggml_tensor * x, Ds41Quant mode) {
    const int bs = mode == Ds41Quant::FP4_E4M3_16 ? 16 : 32;
    const int64_t nb = ggml_nelements(x) / bs;
    ggml_tensor * x2 = ggml_reshape_2d(c, ggml_is_contiguous(x) ? x : ggml_cont(c, x), bs, nb);
    ggml_tensor * amax = ggml_pool_1d(c, ggml_abs(c, x2), GGML_OP_POOL_MAX, bs, bs, 0);   // [1, nb]
    const bool fp8 = mode == Ds41Quant::FP8_POW2_32;
    const float maxv = fp8 ? 448.0f : 6.0f;
    const float floor_ = fp8 ? 1e-4f : mode == Ds41Quant::FP4_E4M3_16 ? 6.0f / 512.0f : 6.0f * 0x1p-126f;
    ggml_tensor * t = ggml_div(c, ggml_clamp(c, amax, floor_, INFINITY), ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, nb), maxv));
    ggml_tensor * s = mode == Ds41Quant::FP4_E4M3_16 ? ds41_round_fmt(c, t, 3, 0x1p-6f) : ds41_pow2_ceil(c, t);
    ggml_tensor * v = ggml_clamp(c, ggml_div(c, x2, s), -maxv, maxv);
    ggml_tensor * q = fp8 ? ds41_round_fmt(c, v, 3, 0x1p-6f) : ds41_round_fmt(c, v, 1, 1.0f);
    return ggml_reshape(c, ggml_mul(c, q, s), x);
}

/// STRATA_FOLDS=1 (default off): small exact folds of
/// a stock op into the kernel next to it, CUDA only (the callers check the backend):
///   1. the residual add in front of hc_post (finish: shexp + routed; the DSpark drafter: routed + shared) inside the
///      hc_post kernel (ds41_hc_post_add below) - one add kernel fewer per layer-pass;
///   2. the DSpark drafter's clamp / clamp / swiglu_split chains (routed and shared experts) as GGML_GLU_OP_SWIGLU_CLAMP,
///      the STRATA_FUSE shared-expert form - 3 kernels -> 1, and ggml-cuda's MMVQ GLU fusion where it applies.
/// Both are the same F32 steps in the same order (test_ds41_fuse --cuda --folds).
inline bool ds41_folds_on() {
    static const bool on = [] {
        const char * e = std::getenv("STRATA_FOLDS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

/// hc_post(a + b, residual, post, comb) as ONE op: the hc_post node with b in src[4] - xolotl-ggml.patch's
/// dsv4_hc_post kernel (CUDA) and CPU op then read x as a + b, the same single F32 add the separate ggml_add did (IEEE
/// addition is commutative bit for bit).  No ggml API change (src[4] of the existing op), so a ggml tree without the
/// patch would ignore b: the patch stamp in build-ds41.bat keeps tree and patch together.
inline ggml_tensor * ds41_hc_post_add(ggml_context * c, ggml_tensor * a, ggml_tensor * b, ggml_tensor * residual,
                                      ggml_tensor * post, ggml_tensor * comb) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && ggml_are_same_shape(a, b));
    ggml_tensor * t = ggml_dsv4_hc_post(c, a, residual, post, comb);
    GGML_ASSERT(t->src[4] == nullptr);
    t->src[4] = b;
    return t;
}

}  // namespace strata::v41
