// tools/v41/engine/test_ds41_q8_1.cpp - STRATA_CPU_Q8_1: the CPU pool's q8_1 kernels
// against the card's arithmetic, CPU only (no CUDA call).
//
//   test_ds41_q8_1            parity (switch unset): q8_1_quant_row's blocks; q8_1_rows (AVX2) and q8_1_rows_ref
//                             against a transcription of the card's vec_dot_{iq2_xxs,iq1_s,q2_K}_q8_1 per-call
//                             values (summed in double) on synthetic rows at DeepSeek-V4.1-Flash's shapes (4096 / 2048),
//                             1..8 tokens; and, for scale, how far the default Q8_K dot (ggml-cpu) is from that
//   test_ds41_q8_1 --bench    one thread: an expert's gate+up rows (IQ2_XXS, IQ1_S; 2 x 2048 x 4096) and down rows
//                             (Q2_K; 4096 x 2048) at 1 / 2 / 4 / 8 tokens - q8_1 vs the default kernels, ms per expert
//   STRATA_CPU_Q8_1=1 test_ds41_q8_1 --native
//                             the engine's entry points with the switch on: native_quant_act / native_gu_rows /
//                             native_quant_h / native_down_rows on a whole expert (IQ2_XXS / IQ1_S gate/up, Q2_K down)
//                             equal q8_1_rows + the SwiGLU, and the expert output is within tolerance of the card's
//                             formula applied in double
// Exit 0 = pass.
#include "strata/kernels/cpu/q8_1_cpu.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"   // MAXT
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;

namespace {

std::mt19937_64 rng(20261115);
float uni(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }
uint16_t f2h(float v) { return ggml_fp32_to_fp16(v); }
float h2f(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }
void put_h(uint8_t* p, float v) { const uint16_t h = f2h(v); std::memcpy(p, &h, 2); }

size_t block_bytes(int type) { return kc::q8_1_block_bytes(type); }

// random but valid super-blocks (finite fp16 scales of a realistic size; every other bit pattern is a valid code)
void gen_rows(int type, int rows, int n, std::vector<uint8_t>& out) {
    const size_t bb = block_bytes(type), nsb = (size_t) (n / 256);
    out.resize((size_t) rows * nsb * bb);
    for (auto& b : out) b = (uint8_t) rng();
    for (size_t i = 0; i < (size_t) rows * nsb; ++i) {
        uint8_t* B = out.data() + i * bb;
        if (type != 10) put_h(B, uni(0.002f, 0.03f));
        else { put_h(B + 80, uni(0.002f, 0.02f)); put_h(B + 82, uni(0.0005f, 0.01f)); }
    }
}

void gen_x(int n, std::vector<float>& x) {
    x.resize((size_t) n);
    for (int i = 0; i < n; ++i) x[i] = uni(-1.0f, 1.0f) * (rng() % 16 == 0 ? 8.0f : 1.0f);   // some outlier columns
    x[(size_t) (rng() % (uint64_t) n)] = 60.0f;
}

// ---- the card's per-call values (iq_kernels.cu / fork_fmt.cuh / fork_vecdot.cuh), transcribed independently ----
int8_t q8(const uint8_t* y, int b, int i) { return (int8_t) y[(size_t) b * 36 + 4 + i]; }

double card_iq2_xxs(const uint8_t* row, const uint8_t* y, int n) {
    double s = 0;
    for (int kbx = 0; kbx < n / 256; ++kbx) {
        const uint8_t* B = row + (size_t) kbx * 66;
        for (int iqs = 0; iqs < 16; iqs += 2) {
            uint32_t q2, aux32;
            std::memcpy(&q2, B + 2 + 4 * iqs, 4);
            std::memcpy(&aux32, B + 2 + 4 * (iqs + 1), 4);
            const uint8_t* aux8 = (const uint8_t*) &q2;
            const int yb = kbx * 8 + iqs / 2;
            int sumi = 0;
            for (int k0 = 0; k0 < 8; k0 += 2) {
                const uint8_t* g = (const uint8_t*) (iq2xxs_grid + aux8[k0 / 2]);
                const uint8_t v = (uint8_t) (aux32 >> (7 * k0 / 2));
                int pc = 0;
                for (int j = 0; j < 8; ++j) pc += (v >> j) & 1;
                const uint32_t sg = (uint32_t) (v ^ ((pc & 1) << 7));   // unpack_ksigns
                for (int j = 0; j < 8; ++j) {
                    const int w = (sg >> j) & 1 ? -(int) g[j] : (int) g[j];
                    sumi += w * q8(y, yb, 4 * k0 + j);
                }
            }
            const int ls = (int) (aux32 >> 27 | 1);
            sumi = sumi * ls / 8;
            const float d = h2f(B) * h2f(y + (size_t) yb * 36);
            s += (double) (d * (float) sumi);
        }
    }
    return s;
}

double card_iq1_s(const uint8_t* row, const uint8_t* y, int n) {
    double s = 0;
    for (int kbx = 0; kbx < n / 256; ++kbx) {
        const uint8_t* B = row + (size_t) kbx * 50;
        for (int iqs = 0; iqs < 8; ++iqs) {
            const uint8_t* qs = B + 2 + 4 * iqs;
            uint16_t qh16;
            std::memcpy(&qh16, B + 34 + 2 * iqs, 2);
            const int qh = qh16;
            const int yb = kbx * 8 + iqs;
            int sumi = 0;
            for (int l0 = 0; l0 < 8; l0 += 2) {
                // the card reads iq1s_grid_gpu (nibbles = value + 1); ggml-common.h gives the CPU build iq1s_grid
                const uint64_t grid = iq1s_grid[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
                for (int j = 0; j < 8; ++j) sumi += ((int) (int8_t) (uint8_t) (grid >> (8 * j)) + 1) * q8(y, yb, 4 * l0 + j);
            }
            const float d1q = h2f(B) * (((qh >> 11) & 0x0E) + 1);
            const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
            const float dx = h2f(y + (size_t) yb * 36), dy = h2f(y + (size_t) yb * 36 + 2);
            s += (double) (d1q * (dx * sumi + dy * delta));
        }
    }
    return s;
}

double card_q2_K(const uint8_t* row, const uint8_t* y, int n) {
    double s = 0;
    for (int kbx = 0; kbx < n / 256; ++kbx) {
        const uint8_t* B = row + (size_t) kbx * 84;
        for (int iqs = 0; iqs < 16; ++iqs) {   // QI2_K = 16, VDR 1
            const int bq8_offset = 4 * (iqs / 8);
            const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
            const uint8_t* scales = B + scale_offset;
            uint32_t v;
            std::memcpy(&v, B + 16 + 4 * iqs, 4);
            float sumf_d = 0.0f, sumf_m = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const int yb = kbx * 8 + bq8_offset + i;
                const float d8 = h2f(y + (size_t) yb * 36);
                const int sc = scales[2 * i];
                int dpv = 0, dpm = 0;
                for (int j = 0; j < 4; ++j) {
                    const int u = q8(y, yb, 4 * (iqs % 8) + j);
                    dpv += (int) ((v >> (8 * j + 2 * i)) & 3) * u;
                    dpm += (sc >> 4) * u;
                }
                sumf_d += d8 * (float) (dpv * (sc & 0xF));
                sumf_m += d8 * (float) dpm;
            }
            s += (double) (h2f(B + 80) * sumf_d - h2f(B + 82) * sumf_m);
        }
    }
    return s;
}

// vec_dot_iq2_xs_q8_1 / vec_dot_iq2_s_q8_1: two 16-value halves with their own 4-bit scales
double card_iq2_xs_s(int type, const uint8_t* row, const uint8_t* y, int n) {
    const size_t bb = type == 17 ? 74 : 82;
    double s = 0;
    for (int kbx = 0; kbx < n / 256; ++kbx) {
        const uint8_t* B = row + (size_t) kbx * bb;
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t scl = type == 17 ? B[66 + ib] : B[74 + ib];
            const int ls0 = scl & 0x0F, ls1 = scl >> 4;
            int sumi0 = 0, sumi1 = 0;
            for (int l = 0; l < 4; ++l) {
                uint64_t g;
                uint8_t sg;
                if (type == 17) {
                    uint16_t q2;
                    std::memcpy(&q2, B + 2 + 8 * ib + 2 * l, 2);
                    g = iq2xs_grid[q2 & 0x1FF];
                    const uint8_t v = (uint8_t) (q2 >> 9);
                    int pc = 0;
                    for (int j = 0; j < 8; ++j) pc += (v >> j) & 1;
                    sg = (uint8_t) (v ^ ((pc & 1) << 7));   // unpack_ksigns
                } else {
                    const int qh = B[66 + ib];
                    g = iq2s_grid[B[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300)];
                    sg = B[2 + 32 + 4 * ib + l];
                }
                int t = 0;
                for (int j = 0; j < 8; ++j) {
                    const int w = (int) (uint8_t) (g >> (8 * j));
                    t += ((sg >> j) & 1 ? -w : w) * q8(y, kbx * 8 + ib, 8 * l + j);
                }
                if (l < 2) sumi0 += t; else sumi1 += t;
            }
            const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
            const float d = h2f(B) * h2f(y + (size_t) (kbx * 8 + ib) * 36);
            s += (double) (d * (float) sumi);
        }
    }
    return s;
}

double card_iq3_xxs(const uint8_t* row, const uint8_t* y, int n) {
    double s = 0;
    for (int kbx = 0; kbx < n / 256; ++kbx) {
        const uint8_t* B = row + (size_t) kbx * 98;
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q3 = B + 2 + 8 * ib;
            uint32_t aux32;
            std::memcpy(&aux32, B + 2 + 64 + 4 * ib, 4);
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t v = (uint8_t) (aux32 >> (7 * l));
                int pc = 0;
                for (int j = 0; j < 8; ++j) pc += (v >> j) & 1;
                const uint8_t sg = (uint8_t) (v ^ ((pc & 1) << 7));
                const uint64_t g = (uint64_t) iq3xxs_grid[q3[2 * l]] | (uint64_t) iq3xxs_grid[q3[2 * l + 1]] << 32;
                for (int j = 0; j < 8; ++j) {
                    const int w = (int) (uint8_t) (g >> (8 * j));
                    sumi += ((sg >> j) & 1 ? -w : w) * q8(y, kbx * 8 + ib, 8 * l + j);
                }
            }
            const int ls = (int) (aux32 >> 28);
            sumi = (ls * sumi + sumi / 2) / 2;
            const float d = h2f(B) * h2f(y + (size_t) (kbx * 8 + ib) * 36);
            s += (double) (d * (float) sumi);
        }
    }
    return s;
}

double card_dot(int type, const uint8_t* row, const uint8_t* y, int n) {
    if (type == 17 || type == 22) return card_iq2_xs_s(type, row, y, n);
    if (type == 18) return card_iq3_xxs(row, y, n);
    return type == 16 ? card_iq2_xxs(row, y, n) : type == 19 ? card_iq1_s(row, y, n) : card_q2_K(row, y, n);
}

// the row's float values (ggml's dequantizer) . x in double: the exact product the quantized paths approximate
double exact_dot(int type, const uint8_t* row, const float* x, int n) {
    std::vector<float> w((size_t) n);
    ggml_get_type_traits((ggml_type) type)->to_float(row, w.data(), n);
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double) w[(size_t) i] * x[i];
    return s;
}

const char* tname(int t) {
    return t == 16 ? "IQ2_XXS" : t == 17 ? "IQ2_XS" : t == 18 ? "IQ3_XXS" : t == 19 ? "IQ1_S" : t == 22 ? "IQ2_S" : "Q2_K";
}

int parity() {
    int fails = 0;
    // 1. the quantizer: d = fp16(amax / 127), |q| <= 127, q * d within d / 2 of x (the clamp's blocks aside), sum = fp16(sum x)
    {
        std::vector<float> x;
        gen_x(4096, x);
        for (int i = 0; i < 32; ++i) x[(size_t) (64 + i)] = 0.0f;   // an all-zero block
        std::vector<uint8_t> q(kc::q8_1_row_bytes(4096));
        kc::q8_1_quant_row(x.data(), q.data(), 4096);
        double worst = 0;
        for (int b = 0; b < 128; ++b) {
            const uint8_t* yb = q.data() + (size_t) b * 36;
            float amax = 0, sum = 0;
            for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(x[(size_t) (32 * b + i)])); sum += x[(size_t) (32 * b + i)]; }
            const float d = h2f(yb);
            if (d != ggml_fp16_to_fp32(f2h(amax / 127.0f))) { ++fails; std::printf("FAIL quant: block %d d %g want %g\n", b, d, amax / 127.0f); }
            if (std::fabs(h2f(yb + 2) - sum) > 1e-2 * std::max(1.0f, std::fabs(sum))) { ++fails; std::printf("FAIL quant: block %d sum\n", b); }
            for (int i = 0; i < 32; ++i) {
                const double e = std::fabs((double) d * (int8_t) yb[4 + i] - x[(size_t) (32 * b + i)]);
                worst = std::max(worst, amax > 0 ? e / ((double) amax / 127.0) : e);
            }
        }
        std::printf("quant: 128 blocks, worst |q*d - x| = %.3f of a step (<= 0.5 + ~0.06 for the fp16 d)\n", worst);
        // 0.5 + 127 x the fp16 d's relative rounding (2^-11)
        if (worst > 0.57) { ++fails; std::printf("FAIL quant: rounding\n"); }
    }
    // 2. rows vs the card's per-call formula
    struct Case { int type, n, rows; };
    const Case cases[] = { {16, 4096, 64}, {19, 4096, 64}, {22, 4096, 64}, {10, 2048, 64}, {16, 2048, 32}, {17, 2048, 64},
                           {18, 2048, 64}, {22, 2048, 64}, {10, 4096, 32} };
    for (const Case& c : cases) {
        std::vector<uint8_t> W;
        gen_rows(c.type, c.rows, c.n, W);
        const size_t rb = (size_t) (c.n / 256) * block_bytes(c.type);
        for (int nt : {1, 3, 8}) {
            std::vector<std::vector<float>> xs((size_t) nt);
            std::vector<std::vector<uint8_t>> ys((size_t) nt, std::vector<uint8_t>(kc::q8_1_row_bytes(c.n)));
            std::vector<std::vector<uint8_t>> yk((size_t) nt, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, c.n)));
            const void* act[kc::MAXT];
            std::vector<std::vector<float>> o1((size_t) nt, std::vector<float>((size_t) c.rows)), o2 = o1;
            float* p1[kc::MAXT];
            float* p2[kc::MAXT];
            for (int t = 0; t < nt; ++t) {
                gen_x(c.n, xs[(size_t) t]);
                kc::q8_1_quant_row(xs[(size_t) t].data(), ys[(size_t) t].data(), c.n);
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(xs[(size_t) t].data(), yk[(size_t) t].data(), c.n);
                act[t] = ys[(size_t) t].data();
                p1[t] = o1[(size_t) t].data();
                p2[t] = o2[(size_t) t].data();
            }
            kc::q8_1_rows(c.type, W.data(), rb, c.n, act, nt, p1, 0, c.rows);
            kc::q8_1_rows_ref(c.type, W.data(), rb, c.n, act, nt, p2, 0, c.rows);
            double e_fast = 0, e_ref = 0, e_q8k = 0, e_q81_exact = 0, e_card_exact = 0, scale = 0;
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < c.rows; ++r) {
                    const uint8_t* row = W.data() + (size_t) r * rb;
                    const double card = card_dot(c.type, row, ys[(size_t) t].data(), c.n);
                    const double ex = exact_dot(c.type, row, xs[(size_t) t].data(), c.n);
                    float k = 0;
                    ggml_get_type_traits_cpu((ggml_type) c.type)->vec_dot(c.n, &k, 0, row, 0, yk[(size_t) t].data(), 0, 1);
                    scale = std::max(scale, std::fabs(ex));
                    e_fast = std::max(e_fast, std::fabs(o1[(size_t) t][(size_t) r] - card));
                    e_ref = std::max(e_ref, std::fabs(o2[(size_t) t][(size_t) r] - card));
                    e_q8k = std::max(e_q8k, std::fabs((double) k - card));
                    e_q81_exact = std::max(e_q81_exact, std::fabs(o1[(size_t) t][(size_t) r] - ex));
                    e_card_exact = std::max(e_card_exact, std::fabs((double) k - ex));
                }
            const double tol = 2e-5 * scale + 1e-6;
            const bool ok = e_fast <= tol && e_ref <= tol;
            fails += !ok;
            std::printf("%s %-7s n %4d nt %d: max |out| %.4g | vs card: avx2 %.2e ref %.2e (tol %.1e), default Q8_K %.2e | vs exact: q8_1 %.2e Q8_K %.2e\n",
                        ok ? "ok  " : "FAIL", tname(c.type), c.n, nt, scale, e_fast, e_ref, tol, e_q8k, e_q81_exact, e_card_exact);
        }
    }
    return fails;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int bench() {
    const int D = 4096, FF = 2048;
    struct B { int type, rows, n; bool gu; };
    const B bs[] = { {16, FF, D, true}, {19, FF, D, true}, {22, FF, D, true}, {22, D, FF, false}, {18, D, FF, false},
                     {17, D, FF, false}, {16, D, FF, false}, {10, D, FF, false} };
    for (const B& b : bs) {
        std::vector<uint8_t> W, U;
        gen_rows(b.type, b.rows, b.n, W);
        if (b.gu) gen_rows(b.type, b.rows, b.n, U);
        const size_t rb = (size_t) (b.n / 256) * block_bytes(b.type);
        for (int nt : {1, 2, 4, 8}) {
            std::vector<std::vector<float>> xs((size_t) nt);
            std::vector<std::vector<uint8_t>> ys((size_t) nt, std::vector<uint8_t>(kc::q8_1_row_bytes(b.n)));
            std::vector<std::vector<uint8_t>> yk((size_t) nt, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, b.n)));
            const void* a1[kc::MAXT];
            const void* ak[kc::MAXT];
            std::vector<std::vector<float>> o((size_t) nt, std::vector<float>((size_t) b.rows)), o2 = o;
            float* po[kc::MAXT];
            float* po2[kc::MAXT];
            for (int t = 0; t < nt; ++t) {
                gen_x(b.n, xs[(size_t) t]);
                kc::q8_1_quant_row(xs[(size_t) t].data(), ys[(size_t) t].data(), b.n);
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(xs[(size_t) t].data(), yk[(size_t) t].data(), b.n);
                a1[t] = ys[(size_t) t].data();
                ak[t] = yk[(size_t) t].data();
                po[t] = o[(size_t) t].data();
                po2[t] = o2[(size_t) t].data();
            }
            const int reps = 5;
            double best1 = 1e30, bestk = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                kc::q8_1_rows(b.type, W.data(), rb, b.n, a1, nt, po, 0, b.rows);
                if (b.gu) kc::q8_1_rows(b.type, U.data(), rb, b.n, a1, nt, po2, 0, b.rows);
                best1 = std::min(best1, ms_since(t0));
                // the default: the AVX-2 multi-token kernel where the engine takes it (IQ2_XXS from 2 tokens), else
                // ggml-cpu's dot per token (IQ1_S, Q2_K, one token) - native_gu_rows' / native_down_rows' own choice
                t0 = std::chrono::steady_clock::now();
                if (b.gu && b.type != 19 && nt >= 2 && kc::cpu_avx2_ok()) {
                    // iq256_gu_rows reads gate at blob and up at blob + up_off: one buffer, gate then up
                    static std::vector<uint8_t> GU;
                    GU.resize(W.size() + U.size());
                    std::memcpy(GU.data(), W.data(), W.size());
                    std::memcpy(GU.data() + W.size(), U.data(), U.size());
                    t0 = std::chrono::steady_clock::now();
                    kc::iq256_gu_rows(b.type, GU.data(), rb, W.size(), b.n, ak, nt, po, 0, b.rows,
                                      std::numeric_limits<float>::infinity());
                } else {
                    const auto dot = ggml_get_type_traits_cpu((ggml_type) b.type)->vec_dot;
                    for (int rr = 0; rr < b.rows; ++rr)
                        for (int t = 0; t < nt; ++t) {
                            dot(b.n, &po[t][rr], 0, W.data() + (size_t) rr * rb, 0, ak[t], 0, 1);
                            if (b.gu) dot(b.n, &po2[t][rr], 0, U.data() + (size_t) rr * rb, 0, ak[t], 0, 1);
                        }
                }
                bestk = std::min(bestk, ms_since(t0));
            }
            std::printf("bench %-7s %s nt %d: q8_1 %.2f ms, default (Q8_K) %.2f ms per expert, ratio %.2fx\n", tname(b.type),
                        b.gu ? "gate+up" : "down   ", nt, best1, bestk, best1 / bestk);
        }
    }
    return 0;
}

// the engine's entry points with STRATA_CPU_Q8_1=1: a whole expert through native_* vs q8_1_rows + SwiGLU, and vs the
// card's formula in double for the gate/up rows
int native_check() {
    if (!kc::cpu_q8_1_on()) { std::printf("--native needs STRATA_CPU_Q8_1=1\n"); return 2; }
    int fails = 0;
    const int D = 4096, FF = 2048;
    for (int gt : {16, 19}) {
        kc::NativeFmt f;
        std::string err;
        if (!kc::native_fmt(gt, 10, D, FF, f, err)) { std::printf("FAIL native_fmt: %s\n", err.c_str()); return 1; }
        if (!kc::cpu_q8_1_act(f.gu_act) || !kc::cpu_q8_1_act(f.d_act)) { std::printf("FAIL: activations not Q8_K\n"); return 1; }
        std::vector<uint8_t> blob(f.bytes), G, U, Dn;
        gen_rows(gt, FF, D, G);
        gen_rows(gt, FF, D, U);
        gen_rows(10, D, FF, Dn);
        std::memcpy(blob.data(), G.data(), G.size());
        std::memcpy(blob.data() + f.up_off, U.data(), U.size());
        std::memcpy(blob.data() + f.down_off, Dn.data(), Dn.size());
        const int nt = 3;
        std::vector<std::vector<float>> xs((size_t) nt);
        std::vector<std::vector<uint8_t>> act((size_t) nt, std::vector<uint8_t>(kc::kNativeActBytes)),
            hq((size_t) nt, std::vector<uint8_t>(kc::kNativeHBytes));
        std::vector<std::vector<float>> ff((size_t) nt, std::vector<float>((size_t) FF)), out((size_t) nt, std::vector<float>((size_t) D));
        const void* ap[kc::MAXT];
        const void* hp[kc::MAXT];
        float* fp[kc::MAXT];
        float* op[kc::MAXT];
        for (int t = 0; t < nt; ++t) {
            gen_x(D, xs[(size_t) t]);
            for (auto& v : xs[(size_t) t]) v *= 0.05f;
            kc::native_quant_act(f, xs[(size_t) t].data(), act[(size_t) t].data());
            std::vector<uint8_t> want(kc::q8_1_row_bytes(D));
            kc::q8_1_quant_row(xs[(size_t) t].data(), want.data(), D);
            if (std::memcmp(want.data(), act[(size_t) t].data(), want.size())) { ++fails; std::printf("FAIL native_quant_act bytes\n"); }
            ap[t] = act[(size_t) t].data(); fp[t] = ff[(size_t) t].data(); op[t] = out[(size_t) t].data(); hp[t] = hq[(size_t) t].data();
        }
        kc::native_gu_rows(f, blob.data(), ap, nt, fp, 0, FF);
        double e_gu = 0, s_gu = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < FF; ++r) {
                const double g = card_dot(gt, blob.data() + (size_t) r * f.gu_row, act[(size_t) t].data(), D);
                const double u = card_dot(gt, blob.data() + f.up_off + (size_t) r * f.gu_row, act[(size_t) t].data(), D);
                const double h = g / (1.0 + std::exp(-g)) * u;
                e_gu = std::max(e_gu, std::fabs(h - ff[(size_t) t][(size_t) r]));
                s_gu = std::max(s_gu, std::fabs(h));
            }
        for (int t = 0; t < nt; ++t) kc::native_quant_h(f, ff[(size_t) t].data(), hq[(size_t) t].data());
        kc::native_down_rows(f, blob.data(), hp, nt, op, 0, D);
        double e_dn = 0, s_dn = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < D; ++r) {
                const double v = card_q2_K(blob.data() + f.down_off + (size_t) r * f.d_row, hq[(size_t) t].data(), FF);
                e_dn = std::max(e_dn, std::fabs(v - out[(size_t) t][(size_t) r]));
                s_dn = std::max(s_dn, std::fabs(v));
            }
        const bool ok = e_gu <= 1e-4 * s_gu + 1e-7 && e_dn <= 1e-4 * s_dn + 1e-7;
        fails += !ok;
        std::printf("%s native %s/Q2_K expert, 3 tokens: SwiGLU h vs card formula %.2e (max %.3g), down vs card formula %.2e (max %.3g)\n",
                    ok ? "ok  " : "FAIL", tname(gt), e_gu, s_gu, e_dn, s_dn);
    }
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench") return bench();
    if (mode == "--native") {
        const int f = native_check();
        std::printf(f ? "test_ds41_q8_1 --native: %d FAILED\n" : "test_ds41_q8_1 --native: all passed\n", f);
        return f ? 1 : 0;
    }
    std::printf("CPU AVX2: %s (q8_1_rows takes the %s path)\n", kc::cpu_avx2_ok() ? "yes" : "no",
                kc::cpu_avx2_ok() ? "AVX2" : "scalar");
    const int f = parity();
    std::printf(f ? "test_ds41_q8_1: %d FAILED\n" : "test_ds41_q8_1: all passed\n", f);
    return f ? 1 : 0;
}
