// src/kernels/cpu/q8_1_cpu.cpp - STRATA_CPU_Q8_1: the q8_1 activation quantizer and the scalar row kernels
// (q8_1_cpu.hpp).  The integer rules are transcribed from the card's dots: vec_dot_{iq2_xxs,iq2_xs,iq2_s,iq3_xxs}_q8_1
// (iq_kernels.cu), vec_dot_iq1_s_q8_1 (fork/fork_fmt.cuh), vec_dot_q2_K_q8_1 (fork/fork_vecdot.cuh); the quantizer from
// q8_1_store / native_quantize_q8_1_kernel + q8_1_finite.hpp.
#include "strata/kernels/cpu/q8_1_cpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {

bool cpu_q8_1_on() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRATA_CPU_Q8_1"); return v && v[0] == '1'; }();
    return on;
}

bool cpu_q8_1_act(int act_type) noexcept { return act_type == (int) GGML_TYPE_Q8_K && cpu_q8_1_on(); }

bool q8_1_int_type(int type) noexcept {
    return type == 10 || type == 16 || type == 17 || type == 18 || type == 19 || type == 22;
}

size_t q8_1_block_bytes(int type) noexcept {
    switch (type) {
        case 10: return 84;   // Q2_K: scales[16], qs[64], d, dmin
        case 16: return 66;   // IQ2_XXS: d, qs[32] (uint16)
        case 17: return 74;   // IQ2_XS: d, qs[32] (uint16), scales[8]
        case 18: return 98;   // IQ3_XXS: d, qs[96] (64 grid bytes + 8 x uint32 signs/scale)
        case 19: return 50;   // IQ1_S: d, qs[32], qh[8] (uint16)
        case 22: return 82;   // IQ2_S: d, qs[64] (32 grid + 32 sign bytes), qh[8], scales[8]
        default: return 0;
    }
}

namespace {
constexpr float kHalfMax = 65504.0f;
float q8_1_finite(float v) { return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v; }
float h2f(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }
}  // namespace

void q8_1_quant_row(const float* x, void* dst, int64_t n) {
    uint8_t* y = (uint8_t*) dst;
    for (int64_t b = 0; b < n / 32; ++b, x += 32, y += kQ81Bytes) {
        // the warp's XOR butterfly (offsets 16, 8, 4, 2, 1): every lane ends with the same sum; lane 0's order here
        float s[32], amax = 0.0f;
        for (int i = 0; i < 32; ++i) { s[i] = x[i]; amax = std::max(amax, std::fabs(x[i])); }
        for (int o = 16; o > 0; o >>= 1) {
            float t[32];
            for (int i = 0; i < 32; ++i) t[i] = s[i] + s[i ^ o];
            std::memcpy(s, t, sizeof s);
        }
        const float d = q8_1_finite(amax / 127.0f);
        int8_t* q = (int8_t*) (y + 4);
        for (int i = 0; i < 32; ++i) {
            if (amax == 0.0f) { q[i] = 0; continue; }
            const float v = std::round(x[i] / d);   // roundf: half away from zero, as the card's roundf
            q[i] = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
        }
        const uint16_t hd = ggml_fp32_to_fp16(d), hs = ggml_fp32_to_fp16(q8_1_finite(s[0]));
        std::memcpy(y, &hd, 2);
        std::memcpy(y + 2, &hs, 2);
    }
}

namespace {
// One decoded row: int8 weights in element order (signed grid values for the IQ2/IQ3 codes, IQ1_S the card's
// iq1s_grid_gpu nibbles 0..2 = iq1s_grid + 1, Q2_K q * (scale & 15)) and the per-32-block parameters.
struct RowDec {
    std::vector<int8_t> w;
    std::vector<float> a, c;          // a: the block's fp16 d (19: d1q; 10: dm.x) | c: 19 delta, 10 dm.y
    std::vector<int32_t> i0, i1;      // 16 / 18: ls | 17 / 22: ls of values 0-15 / 16-31 | 10: the two mins
};

void put_signed(int8_t* w, uint64_t grid, uint8_t signs) {   // 8 grid bytes, bit j of `signs` negates value j
    for (int j = 0; j < 8; ++j) {
        const int g = (int) (uint8_t) (grid >> (8 * j));
        w[j] = (int8_t) ((signs >> j) & 1 ? -g : g);
    }
}

void decode_row(int type, const uint8_t* row, int n, RowDec& R) {
    const int nb = n / 32;
    R.w.resize((size_t) n);
    R.a.resize((size_t) nb); R.c.resize((size_t) nb); R.i0.resize((size_t) nb); R.i1.resize((size_t) nb);
    const size_t bb = q8_1_block_bytes(type);
    for (int sb = 0; sb < n / 256; ++sb) {
        int8_t* w = R.w.data() + (size_t) sb * 256;
        const uint8_t* B = row + (size_t) sb * bb;
        for (int ib = 0; ib < 8; ++ib) {
            const size_t b = (size_t) (sb * 8 + ib);
            int8_t* wb = w + 32 * ib;
            if (type == 16) {
                uint32_t aux[2];
                std::memcpy(aux, B + 2 + 8 * ib, 8);
                for (int l = 0; l < 4; ++l)
                    put_signed(wb + 8 * l, iq2xxs_grid[(aux[0] >> (8 * l)) & 255], ksigns_iq2xs[(aux[1] >> (7 * l)) & 127]);
                R.a[b] = h2f(B);
                R.i0[b] = (int32_t) (aux[1] >> 27 | 1);
            } else if (type == 17) {
                uint16_t q2[4];
                std::memcpy(q2, B + 2 + 8 * ib, 8);
                for (int l = 0; l < 4; ++l) put_signed(wb + 8 * l, iq2xs_grid[q2[l] & 511], ksigns_iq2xs[q2[l] >> 9]);
                R.a[b] = h2f(B);
                R.i0[b] = B[66 + ib] & 0x0F;
                R.i1[b] = B[66 + ib] >> 4;
            } else if (type == 22) {
                const uint8_t* qs = B + 2 + 4 * ib;
                const uint8_t* sg = B + 2 + 32 + 4 * ib;
                const int qh = B[66 + ib];
                for (int l = 0; l < 4; ++l) put_signed(wb + 8 * l, iq2s_grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300)], sg[l]);
                R.a[b] = h2f(B);
                R.i0[b] = B[74 + ib] & 0x0F;
                R.i1[b] = B[74 + ib] >> 4;
            } else if (type == 18) {
                const uint8_t* q3 = B + 2 + 8 * ib;
                uint32_t aux32;
                std::memcpy(&aux32, B + 2 + 64 + 4 * ib, 4);
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = (uint64_t) iq3xxs_grid[q3[2 * l]] | (uint64_t) iq3xxs_grid[q3[2 * l + 1]] << 32;
                    put_signed(wb + 8 * l, g, ksigns_iq2xs[(aux32 >> (7 * l)) & 127]);
                }
                R.a[b] = h2f(B);
                R.i0[b] = (int32_t) (aux32 >> 28);
            } else if (type == 19) {
                uint16_t qh16;
                std::memcpy(&qh16, B + 34 + 2 * ib, 2);
                const int qh = qh16;
                for (int l = 0; l < 4; ++l) {
                    // iq1s_grid (-1 / 0 / +1 bytes, the CPU's copy of the card's iq1s_grid_gpu nibbles minus one): + 1
                    const uint64_t g = iq1s_grid[B[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                    for (int j = 0; j < 8; ++j) wb[8 * l + j] = (int8_t) ((int8_t) (uint8_t) (g >> (8 * j)) + 1);
                }
                R.a[b] = h2f(B) * (float) (((qh >> 11) & 0x0E) + 1);
                R.c[b] = -1.0f + IQ1S_DELTA - (float) (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
            } else {   // 10: Q2_K; block ib = half h, shift j
                const int h = ib / 4, j = ib % 4;
                const uint8_t* q = B + 16 + 32 * h;
                const uint8_t s0 = B[8 * h + 2 * j], s1 = B[8 * h + 2 * j + 1];
                for (int l = 0; l < 16; ++l) {
                    wb[l] = (int8_t) (((q[l] >> (2 * j)) & 3) * (s0 & 0xF));
                    wb[16 + l] = (int8_t) (((q[l + 16] >> (2 * j)) & 3) * (s1 & 0xF));
                }
                R.a[b] = h2f(B + 80);
                R.c[b] = h2f(B + 82);
                R.i0[b] = s0 >> 4;
                R.i1[b] = s1 >> 4;
            }
        }
    }
}
}  // namespace

void q8_1_rows_ref(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1) {
    const int nb = n / 32;
    if (!q8_1_int_type(type) || n % 256) {   // any other format: ggml's dequantizer, dequantized q8_1
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) type);
        if (tt == nullptr || tt->to_float == nullptr) std::abort();   // native_fmt admitted it: cannot happen for ggml types
        thread_local std::vector<float> wf, yf;
        wf.resize((size_t) n);
        yf.resize((size_t) nt * (size_t) n);
        for (int t = 0; t < nt; ++t)
            for (int b = 0; b < nb; ++b) {
                const uint8_t* yb = (const uint8_t*) act[t] + (size_t) b * kQ81Bytes;
                const float d8 = h2f(yb);
                for (int i = 0; i < 32; ++i) yf[(size_t) t * n + 32 * b + i] = d8 * (float) (int8_t) yb[4 + i];
            }
        for (int r = r0; r < r1; ++r) {
            tt->to_float(w + (size_t) r * row_bytes, wf.data(), n);
            for (int t = 0; t < nt; ++t) {
                const float* y = yf.data() + (size_t) t * n;
                float acc = 0.0f;
                for (int i = 0; i < n; ++i) acc += wf[(size_t) i] * y[i];
                out[t][r] = acc;
            }
        }
        return;
    }
    thread_local RowDec R;
    for (int r = r0; r < r1; ++r) {
        decode_row(type, w + (size_t) r * row_bytes, n, R);
        for (int t = 0; t < nt; ++t) {
            const uint8_t* yb = (const uint8_t*) act[t];
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b, yb += kQ81Bytes) {
                const int8_t* q = (const int8_t*) (yb + 4);
                const int8_t* wb = R.w.data() + 32 * b;
                int lo = 0, hi = 0;
                for (int i = 0; i < 16; ++i) { lo += (int) wb[i] * (int) q[i]; hi += (int) wb[16 + i] * (int) q[16 + i]; }
                const int s = lo + hi;
                const float d8 = h2f(yb);
                const int32_t l0 = R.i0[(size_t) b], l1 = R.i1[(size_t) b];
                switch (type) {
                    case 16: acc += (R.a[(size_t) b] * d8) * (float) (s * l0 / 8); break;
                    case 18: acc += (R.a[(size_t) b] * d8) * (float) ((l0 * s + s / 2) / 2); break;
                    case 17:
                    case 22: acc += (R.a[(size_t) b] * d8) * (float) ((lo * l0 + hi * l1 + s / 2) / 4); break;
                    case 19: acc += R.a[(size_t) b] * (d8 * (float) s + h2f(yb + 2) * R.c[(size_t) b]); break;
                    default: {
                        int u0 = 0, u1 = 0;
                        for (int i = 0; i < 16; ++i) { u0 += q[i]; u1 += q[16 + i]; }
                        acc += d8 * (R.a[(size_t) b] * (float) s - R.c[(size_t) b] * (float) (l0 * u0 + l1 * u1));
                    }
                }
            }
            out[t][r] = acc;
        }
    }
}

void q8_1_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
               int r0, int r1) {
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8_1_AVX2") == nullptr;
    if (avx2 && q8_1_int_type(type) && n % 256 == 0) {
        q8_1_rows_avx2(type, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    // a format without the integer kernels: the float path, slow (a scalar dequantize per row) - said once
    static bool told = false;
    if (!q8_1_int_type(type) && !told) {
        told = true;
        std::fprintf(stderr, "strata: STRATA_CPU_Q8_1: ggml type %d has no q8_1 integer kernel - its rows take the slow float path\n", type);
    }
    q8_1_rows_ref(type, w, row_bytes, n, act, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
