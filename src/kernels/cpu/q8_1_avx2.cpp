// src/kernels/cpu/q8_1_avx2.cpp - STRATA_CPU_Q8_1's AVX2 row kernels (q8_1_cpu.hpp) for Q2_K (10), IQ2_XXS (16),
// IQ2_XS (17), IQ3_XXS (18), IQ1_S (19) and IQ2_S (22): each weight row is decoded ONCE into int8 values in element
// order (plus per-32-block parameters), then dotted with every token's q8_1 activation - per 32-block integer sums
// (maddubs / madd, exact), eight blocks reduced together, then the card's per-block integer rule and float formula
// (q8_1_rows_ref is the scalar twin).
//
// This file is compiled for AVX2 and must not run anything before the caller's cpu_avx2_ok() check: no runtime
// initializer at namespace scope, everything below is reached through q8_1_rows_avx2 only.
#include "strata/kernels/cpu/q8_1_cpu.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

float h2f(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

// +-1 bytes: bit j of the 8-bit sign byte set -> byte j = -1.  k7: ksigns_iq2xs's 7-bit codes (parity bit added);
// k8: a plain sign byte (IQ2_S)
struct Signs { uint64_t k7[128]; uint64_t k8[256]; };
uint64_t pm1(uint8_t s) {
    uint64_t x = 0;
    for (int j = 0; j < 8; ++j) x |= (uint64_t) ((s >> j) & 1 ? 0xFF : 0x01) << (8 * j);
    return x;
}
const Signs& signs() {
    static const Signs s = [] {
        Signs r{};
        for (int i = 0; i < 128; ++i) r.k7[i] = pm1(ksigns_iq2xs[i]);
        for (int i = 0; i < 256; ++i) r.k8[i] = pm1((uint8_t) i);
        return r;
    }();
    return s;
}

struct Dec {
    std::vector<int8_t> w;               // n int8 weights, element order
    std::vector<int16_t> sc;             // 10 / 17 / 22: per block 16 int16 madd multipliers (8 x values 0-15, 8 x 16-31)
    std::vector<float> a, c;             // per block
    std::vector<int32_t> i0, i1;         // per block (16 / 18: ls; 10: mins)
};

inline void put_block(Dec& D, int b, __m256i w) { _mm256_storeu_si256((__m256i*) (D.w.data() + 32 * b), w); }
inline void put_sc(Dec& D, int b, int lo, int hi) {
    _mm256_storeu_si256((__m256i*) (D.sc.data() + 16 * b), _mm256_set_m128i(_mm_set1_epi16((short) hi), _mm_set1_epi16((short) lo)));
}
inline __m256i signed4(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    return _mm256_sign_epi8(_mm256_set_epi64x((long long) g3, (long long) g2, (long long) g1, (long long) g0),
                            _mm256_set_epi64x((long long) s3, (long long) s2, (long long) s1, (long long) s0));
}

void decode(int type, const uint8_t* row, int n, Dec& D) {
    const int nb = n / 32;
    if (D.w.size() < (size_t) n) D.w.resize((size_t) n);
    if (D.a.size() < (size_t) nb) { D.a.resize((size_t) nb); D.c.resize((size_t) nb); D.i0.resize((size_t) nb); D.i1.resize((size_t) nb); }
    if (D.sc.size() < (size_t) nb * 16) D.sc.resize((size_t) nb * 16);
    const Signs& S = signs();
    const size_t bb = q8_1_block_bytes(type);
    const __m256i one8 = _mm256_set1_epi8(1), m3 = _mm256_set1_epi8(3);
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t* B = row + (size_t) sb * bb;
        const float d = type == 10 ? h2f(B + 80) : h2f(B);
        if (type == 10) {   // Q2_K: w = q (0..3), the 16-value scales go in as the madd multipliers
            const float dy = h2f(B + 82);
            for (int h = 0; h < 2; ++h) {
                const __m256i q = _mm256_loadu_si256((const __m256i*) (B + 16 + 32 * h));
                for (int j = 0; j < 4; ++j) {
                    const int b = sb * 8 + 4 * h + j;
                    put_block(D, b, _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3));
                    const uint8_t s0 = B[8 * h + 2 * j], s1 = B[8 * h + 2 * j + 1];
                    put_sc(D, b, s0 & 0xF, s1 & 0xF);
                    D.a[(size_t) b] = d;
                    D.c[(size_t) b] = dy;
                    D.i0[(size_t) b] = s0 >> 4;
                    D.i1[(size_t) b] = s1 >> 4;
                }
            }
            continue;
        }
        for (int ib = 0; ib < 8; ++ib) {
            const int b = sb * 8 + ib;
            D.a[(size_t) b] = d;
            if (type == 16) {
                uint32_t aux[2];
                std::memcpy(aux, B + 2 + 8 * ib, 8);
                const uint8_t* a8 = (const uint8_t*) &aux[0];
                put_block(D, b, signed4(iq2xxs_grid[a8[0]], iq2xxs_grid[a8[1]], iq2xxs_grid[a8[2]], iq2xxs_grid[a8[3]],
                                        S.k7[aux[1] & 127], S.k7[(aux[1] >> 7) & 127], S.k7[(aux[1] >> 14) & 127], S.k7[(aux[1] >> 21) & 127]));
                D.i0[(size_t) b] = (int32_t) (aux[1] >> 27 | 1);
            } else if (type == 17) {
                uint16_t q2[4];
                std::memcpy(q2, B + 2 + 8 * ib, 8);
                put_block(D, b, signed4(iq2xs_grid[q2[0] & 511], iq2xs_grid[q2[1] & 511], iq2xs_grid[q2[2] & 511], iq2xs_grid[q2[3] & 511],
                                        S.k7[q2[0] >> 9], S.k7[q2[1] >> 9], S.k7[q2[2] >> 9], S.k7[q2[3] >> 9]));
                put_sc(D, b, B[66 + ib] & 0x0F, B[66 + ib] >> 4);
            } else if (type == 22) {
                const uint8_t* qs = B + 2 + 4 * ib;
                const uint8_t* sg = B + 2 + 32 + 4 * ib;
                const int qh = B[66 + ib];
                put_block(D, b, signed4(iq2s_grid[qs[0] | ((qh << 8) & 0x300)], iq2s_grid[qs[1] | ((qh << 6) & 0x300)],
                                        iq2s_grid[qs[2] | ((qh << 4) & 0x300)], iq2s_grid[qs[3] | ((qh << 2) & 0x300)],
                                        S.k8[sg[0]], S.k8[sg[1]], S.k8[sg[2]], S.k8[sg[3]]));
                put_sc(D, b, B[74 + ib] & 0x0F, B[74 + ib] >> 4);
            } else if (type == 18) {
                const uint8_t* q3 = B + 2 + 8 * ib;
                uint32_t aux32;
                std::memcpy(&aux32, B + 2 + 64 + 4 * ib, 4);
                auto g2 = [&](int l) { return (uint64_t) iq3xxs_grid[q3[2 * l]] | (uint64_t) iq3xxs_grid[q3[2 * l + 1]] << 32; };
                put_block(D, b, signed4(g2(0), g2(1), g2(2), g2(3),
                                        S.k7[aux32 & 127], S.k7[(aux32 >> 7) & 127], S.k7[(aux32 >> 14) & 127], S.k7[(aux32 >> 21) & 127]));
                D.i0[(size_t) b] = (int32_t) (aux32 >> 28);
            } else {   // 19: IQ1_S - iq1s_grid (-1 / 0 / +1 bytes in element order) + 1 = the card's iq1s_grid_gpu nibbles
                uint16_t qh16;
                std::memcpy(&qh16, B + 34 + 2 * ib, 2);
                const int qh = qh16;
                const uint8_t* qs = B + 2 + 4 * ib;
                const __m256i g = _mm256_set_epi64x((long long) iq1s_grid[qs[3] | (((qh >> 9) & 7) << 8)],
                                                    (long long) iq1s_grid[qs[2] | (((qh >> 6) & 7) << 8)],
                                                    (long long) iq1s_grid[qs[1] | (((qh >> 3) & 7) << 8)],
                                                    (long long) iq1s_grid[qs[0] | ((qh & 7) << 8)]);
                put_block(D, b, _mm256_add_epi8(g, one8));
                D.a[(size_t) b] = d * (float) (((qh >> 11) & 0x0E) + 1);
                D.c[(size_t) b] = -1.0f + IQ1S_DELTA - (float) (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
            }
        }
    }
}

// eight 8-lane int32 vectors -> their eight lane sums, in order
inline __m256i hsum8(const __m256i* s) {
    const __m256i u0 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]), _mm256_hadd_epi32(s[2], s[3]));
    const __m256i u1 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]), _mm256_hadd_epi32(s[6], s[7]));
    return _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31));
}
// C's int division by 2^k (truncation toward zero), lane-wise
inline __m256i tdiv(__m256i v, int k) {
    return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_and_si256(_mm256_srai_epi32(v, 31), _mm256_set1_epi32((1 << k) - 1))), k);
}

inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template<int T>
float dot_row(const Dec& D, const uint8_t* y, const float* d8, const float* s8, const int32_t* u0, const int32_t* u1, int n) {
    constexpr bool kSigned = T == 16 || T == 17 || T == 18 || T == 22;   // signed grids: |w| and sign(q, w)
    constexpr bool kSc = T == 10 || T == 17 || T == 22;                  // per-16 multipliers
    constexpr bool kPlain = T != 10;                                     // the unscaled sum
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        __m256i sp[8], ss[8];
        for (int k = 0; k < 8; ++k) {
            const int b = sb * 8 + k;
            const __m256i w = _mm256_loadu_si256((const __m256i*) (D.w.data() + 32 * b));
            const __m256i q = _mm256_loadu_si256((const __m256i*) (y + (size_t) b * kQ81Bytes + 4));
            const __m256i p = kSigned ? _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q, w))
                                      : _mm256_maddubs_epi16(w, q);   // 19: 0..2, 10: 0..3 - unsigned already
            if (kPlain) sp[k] = _mm256_madd_epi16(p, ones);
            if (kSc) ss[k] = _mm256_madd_epi16(p, _mm256_loadu_si256((const __m256i*) (D.sc.data() + 16 * b)));
        }
        const int o = sb * 8;
        const __m256i Sp = kPlain ? hsum8(sp) : _mm256_setzero_si256();
        const __m256i Ss = kSc ? hsum8(ss) : _mm256_setzero_si256();
        const __m256 dv = _mm256_loadu_ps(d8 + o);
        const __m256 av = _mm256_loadu_ps(D.a.data() + o);
        if (T == 16 || T == 17 || T == 18 || T == 22) {
            __m256i v;
            if (T == 16) v = tdiv(_mm256_mullo_epi32(Sp, _mm256_loadu_si256((const __m256i*) (D.i0.data() + o))), 3);   // s * ls / 8
            else if (T == 18) v = tdiv(_mm256_add_epi32(_mm256_mullo_epi32(Sp, _mm256_loadu_si256((const __m256i*) (D.i0.data() + o))),
                                                        tdiv(Sp, 1)), 1);                                            // (ls * s + s / 2) / 2
            else v = tdiv(_mm256_add_epi32(Ss, tdiv(Sp, 1)), 2);                                                         // (lo*ls0 + hi*ls1 + s/2) / 4
            acc = _mm256_fmadd_ps(_mm256_mul_ps(av, dv), _mm256_cvtepi32_ps(v), acc);   // (d * d8) * sumi
        } else if (T == 19) {   // d1q * (d8 * sumi + s8 * delta)
            const __m256 in = _mm256_fmadd_ps(dv, _mm256_cvtepi32_ps(Sp), _mm256_mul_ps(_mm256_loadu_ps(s8 + o), _mm256_loadu_ps(D.c.data() + o)));
            acc = _mm256_fmadd_ps(av, in, acc);
        } else {                // 10: d8 * (dm.x * sumi - dm.y * (m0 * sum(q[0:16]) + m1 * sum(q[16:32])))
            const __m256i im = _mm256_add_epi32(
                _mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*) (D.i0.data() + o)), _mm256_loadu_si256((const __m256i*) (u0 + o))),
                _mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*) (D.i1.data() + o)), _mm256_loadu_si256((const __m256i*) (u1 + o))));
            const __m256 t = _mm256_fmsub_ps(av, _mm256_cvtepi32_ps(Ss), _mm256_mul_ps(_mm256_loadu_ps(D.c.data() + o), _mm256_cvtepi32_ps(im)));
            acc = _mm256_fmadd_ps(dv, t, acc);
        }
    }
    return hsum_ps(acc);
}

template<int T>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const int nb = n / 32;
    thread_local Dec D;
    thread_local std::vector<float> d8, s8;
    thread_local std::vector<int32_t> u0, u1;
    if (d8.size() < (size_t) nt * nb) { d8.resize((size_t) nt * nb); s8.resize((size_t) nt * nb); }
    if (T == 10 && u0.size() < (size_t) nt * nb) { u0.resize((size_t) nt * nb); u1.resize((size_t) nt * nb); }
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (int t = 0; t < nt; ++t) {   // per token: the blocks' d and sum as float, Q2_K's 16-value sums of q
        const uint8_t* y = (const uint8_t*) act[t];
        for (int b = 0; b < nb; ++b) {
            const uint8_t* yb = y + (size_t) b * kQ81Bytes;
            d8[(size_t) t * nb + b] = h2f(yb);
            s8[(size_t) t * nb + b] = h2f(yb + 2);
            if (T == 10) {
                const __m256i q = _mm256_loadu_si256((const __m256i*) (yb + 4));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16);   // lanes 0-3: q[0:16], 4-7: q[16:32]
                __m128i l2 = _mm_hadd_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
                l2 = _mm_hadd_epi32(l2, l2);
                u0[(size_t) t * nb + b] = _mm_cvtsi128_si32(l2);
                u1[(size_t) t * nb + b] = _mm_extract_epi32(l2, 1);
            }
        }
    }
    for (int r = r0; r < r1; ++r) {
        decode(T, w + (size_t) r * row_bytes, n, D);
        for (int t = 0; t < nt; ++t)
            out[t][r] = dot_row<T>(D, (const uint8_t*) act[t], d8.data() + (size_t) t * nb, s8.data() + (size_t) t * nb,
                                   T == 10 ? u0.data() + (size_t) t * nb : nullptr,
                                   T == 10 ? u1.data() + (size_t) t * nb : nullptr, n);
    }
}

}  // namespace

void q8_1_rows_avx2(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1) {
    switch (type) {
        case 16: rows<16>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 17: rows<17>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 18: rows<18>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 19: rows<19>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 22: rows<22>(w, row_bytes, n, act, nt, out, r0, r1); break;
        default: rows<10>(w, row_bytes, n, act, nt, out, r0, r1); break;
    }
}

}  // namespace strata::kernels::cpu
