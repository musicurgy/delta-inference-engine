// tools/v41/engine/test_ds41_pfused.cpp - the fused prompt-expert kernels (ds41_pfused) against the MMQ
// path gpu_run_chunk runs today (mmq::quantize, gate / up products, swiglu_clamp, quantize, down product) and against an
// FP32 reference (ggml's dequantize + double sums), on random expert blobs at DeepSeek-V4.1-Flash's shape.
//
// For every gate/up + down type pair below (each covered type once on each side, plus pairs of the UD-IQ2_XXS pack):
// 5 experts with 1 / 37 / 64 / 129 / 300 routed rows (531 entries; the partial tiles, a one-row expert and both tile
// heights), rows gathered from 200 random tokens, the outputs scattered through a shuffled row map, the SwiGLU clamp at
// 10 with gate values large enough to hit it.  Printed per pair:
//   fused~mmq   relative L2 distance ||fused - mmq|| / ||mmq|| and the max |fused - mmq| over max |mmq| (all rows)
//   mmq~fp32, fused~fp32   the same against the FP32 reference (every 6th row and the small experts' rows)
//   tile64==128  the fused bits at 64- and 128-row tiles are equal (they must be: the tile only groups rows)
// FAIL when a number is not finite, the tiles differ, or fused~fp32 > 1.5 * mmq~fp32 + 0.002.  Exit code 1 on a FAIL.
//
//   test_ds41_pfused            (needs an sm_80+ GPU; a few seconds of GPU, ~20 s of CPU for the reference)
#include "ds41_pfused.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "test_ds41_pfused: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int64_t H = ds41pf::kEmbd, FF = ds41pf::kFF;
constexpr float kLim = 10.0f;

uint16_t f2h(float f) {   // float -> IEEE half bits (normal range, round to nearest)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7C00u);
    uint32_t h = s | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1FFFu) > 0x1000u || ((m & 0x1FFFu) == 0x1000u && (h & 1))) ++h;
    return (uint16_t) h;
}

// a random matrix of `rows` x `cols` in `type`: random block bytes, every block's fp16 d (the first two bytes of all
// the covered blocks) set so that the weights' RMS is `rms`
std::vector<uint8_t> random_matrix(int type, int64_t rows, int64_t cols, float rms, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, cols), bs = ggml_type_size((ggml_type) type);
    const int64_t nb = (int64_t) (rb / bs) * rows;
    std::vector<uint8_t> m(rb * (size_t) rows);
    for (auto& b : m) b = (uint8_t) (rng() & 0xFF);
    auto set_d = [&](float d) {
        const uint16_t hb = f2h(d);
        for (int64_t i = 0; i < nb; ++i) std::memcpy(m.data() + (size_t) i * bs, &hb, 2);
    };
    set_d(1.0f);
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    std::vector<float> row((size_t) cols);
    double ss = 0;
    const int64_t probe = std::min<int64_t>(rows, 16);
    for (int64_t r = 0; r < probe; ++r) {
        tr->to_float(m.data() + (size_t) r * rb, row.data(), cols);
        for (float v : row) ss += (double) v * v;
    }
    const double cur = std::sqrt(ss / (double) (probe * cols));
    set_d((float) (rms / cur));
    // per-block variety: scale d by a factor in [0.5, 1.5] (keeps d normal, the products see different scales)
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    for (int64_t i = 0; i < nb; ++i) {
        const uint16_t hb = f2h((float) (rms / cur) * u(rng));
        std::memcpy(m.data() + (size_t) i * bs, &hb, 2);
    }
    return m;
}

double rel_l2(const float* a, const float* b, const std::vector<int64_t>& rows) {   // ||a - b|| / ||b|| over rows
    double num = 0, den = 0;
    for (int64_t r : rows)
        for (int64_t j = 0; j < H; ++j) {
            const double d = (double) a[r * H + j] - b[r * H + j];
            num += d * d;
            den += (double) b[r * H + j] * b[r * H + j];
        }
    return std::sqrt(num / std::max(den, 1e-30));
}
double max_rel(const float* a, const float* b, const std::vector<int64_t>& rows) {   // max |a - b| / max |b|
    double md = 0, mb = 0;
    for (int64_t r : rows)
        for (int64_t j = 0; j < H; ++j) {
            md = std::max(md, (double) std::fabs(a[r * H + j] - b[r * H + j]));
            mb = std::max(mb, (double) std::fabs(b[r * H + j]));
        }
    return md / std::max(mb, 1e-30);
}
bool finite_all(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

bool run_pair(int gt, int dt, mmq::Context& ctx) {
    std::mt19937 rng(1234u + 97u * (unsigned) gt + (unsigned) dt);
    const int NX = 5;
    const int cnt[NX] = {1, 37, 64, 129, 300};
    const int T = 200;
    std::vector<int32_t> off(NX + 1, 0);
    for (int q = 0; q < NX; ++q) off[q + 1] = off[q] + cnt[q];
    const int64_t R = off[NX];
    int max_rows = *std::max_element(cnt, cnt + NX);

    // the blobs: [gate rows | up rows | down rows], `stride` apart (whole blocks of both types and 16 bytes, as mmq_plan)
    const size_t gu_row = ggml_row_size((ggml_type) gt, H), d_row = ggml_row_size((ggml_type) dt, FF);
    const size_t up_off = (size_t) FF * gu_row, down_off = 2 * up_off, blob = down_off + (size_t) H * d_row;
    auto gcd = [](size_t a, size_t b) { while (b) { const size_t t = a % b; a = b; b = t; } return a; };
    size_t lcm = 16;
    for (size_t b : {ggml_type_size((ggml_type) gt), ggml_type_size((ggml_type) dt)}) lcm = lcm / gcd(lcm, b) * b;
    const size_t stride = (blob + lcm - 1) / lcm * lcm;
    // gate rms so that gate/up values run past the clamp (|x| ~ 1: a row's dot has rms ~ 64 rms_w)
    std::vector<std::vector<uint8_t>> gm(NX), um(NX), dmx(NX);
    for (int q = 0; q < NX; ++q) {
        gm[q] = random_matrix(gt, FF, H, 0.12f, rng);
        um[q] = random_matrix(gt, FF, H, 0.12f, rng);
        dmx[q] = random_matrix(dt, H, FF, 0.02f, rng);
    }
    std::vector<uint8_t> hb(stride * NX, 0);
    for (int q = 0; q < NX; ++q) {
        std::memcpy(hb.data() + q * stride, gm[q].data(), up_off);
        std::memcpy(hb.data() + q * stride + up_off, um[q].data(), up_off);
        std::memcpy(hb.data() + q * stride + down_off, dmx[q].data(), (size_t) H * d_row);
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = nd(rng);
    std::vector<int32_t> tok((size_t) R), dst((size_t) R);
    for (auto& t : tok) t = (int32_t) (rng() % T);
    std::iota(dst.begin(), dst.end(), 0);
    std::shuffle(dst.begin(), dst.end(), rng);

    // ---- device
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    uint8_t* d_w = nullptr;
    float *d_x = nullptr, *d_gu = nullptr, *d_h = nullptr, *d_out_m = nullptr, *d_out_f = nullptr;
    int32_t *d_tok = nullptr, *d_dst = nullptr, *d_bnd = nullptr, *d_iota = nullptr, *d_tiles = nullptr;
    void *d_xg = nullptr, *d_xd = nullptr, *d_xa = nullptr, *d_ha = nullptr;
    ck(cudaMalloc(&d_w, hb.size() + (1 << 20)), "w");
    ck(cudaMemset(d_w, 0, hb.size() + (1 << 20)), "w0");
    ck(cudaMemcpy(d_w, hb.data(), hb.size(), cudaMemcpyHostToDevice), "w h2d");
    ck(cudaMalloc(&d_x, x.size() * 4), "x");
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x h2d");
    ck(cudaMalloc(&d_tok, 4 * (R + 128)), "tok");
    ck(cudaMemcpy(d_tok, tok.data(), 4 * R, cudaMemcpyHostToDevice), "tok h2d");
    ck(cudaMalloc(&d_dst, 4 * (R + 128)), "dst");
    ck(cudaMemcpy(d_dst, dst.data(), 4 * R, cudaMemcpyHostToDevice), "dst h2d");
    ck(cudaMalloc(&d_bnd, 4 * (NX + 1)), "bnd");
    ck(cudaMemcpy(d_bnd, off.data(), 4 * (NX + 1), cudaMemcpyHostToDevice), "bnd h2d");
    ck(cudaMalloc(&d_iota, 4 * (R + 128)), "iota");
    mmq::iota(d_iota, R + 128, s);
    ck(cudaMalloc(&d_gu, 4 * R * 2 * FF), "gu");
    ck(cudaMalloc(&d_h, 4 * R * FF), "h");
    ck(cudaMalloc(&d_xg, mmq::q8_bytes(R, H)), "xg");
    ck(cudaMalloc(&d_xd, mmq::q8_bytes(R, FF)), "xd");
    ck(cudaMalloc(&d_out_m, 4 * R * H), "out m");
    ck(cudaMalloc(&d_out_f, 4 * R * H), "out f");
    ck(cudaMalloc(&d_xa, ds41pf::act_bytes(R, H)), "xa");
    ck(cudaMalloc(&d_ha, ds41pf::act_bytes(R, FF)), "ha");
    ck(cudaMalloc(&d_tiles, ds41pf::tiles_bytes(R, NX)), "tiles");

    // ---- MMQ (gpu_run_chunk's mmq_launch)
    bool mmq_ok = mmq::fits(gt, FF) && mmq::fits(dt, H);
    std::vector<float> out_m((size_t) R * H, 0.0f);
    if (mmq_ok) {
        mmq::quantize(d_x, d_tok, d_xg, gt, H, H, R, s);
        mmq::Product p;
        p.n = NX; p.bounds = d_bnd; p.total_rows = R; p.max_rows = max_rows;
        p.w = d_w; p.expert_bytes = stride; p.type = gt; p.w_rows = FF; p.w_cols = H; p.xq = d_xg;
        p.ids = d_iota; p.dst = d_gu; p.ld_dst = 2 * FF;
        ctx.run(p, s);
        p.w = d_w + up_off; p.dst = d_gu + FF;
        ctx.run(p, s);
        mmq::swiglu_clamp(d_gu, d_h, R, FF, false, kLim, s);
        mmq::quantize(d_h, nullptr, d_xd, dt, FF, FF, R, s);
        p.w = d_w + down_off; p.type = dt; p.w_rows = H; p.w_cols = FF; p.xq = d_xd; p.ids = d_dst; p.dst = d_out_m; p.ld_dst = H;
        ctx.run(p, s);
        ck(cudaStreamSynchronize(s), "mmq");
        ck(cudaMemcpy(out_m.data(), d_out_m, 4 * R * H, cudaMemcpyDeviceToHost), "out m d2h");
    }

    // ---- fused, at both tile heights
    auto fused = [&](int tr, std::vector<float>& out) {
        ds41pf::Run fr;
        fr.base = d_w; fr.stride = stride; fr.n_exp = NX; fr.gu_type = gt; fr.d_type = dt;
        fr.gu_row = gu_row; fr.d_row = d_row; fr.up_off = up_off; fr.down_off = down_off; fr.swiglu_limit = kLim;
        fr.off = d_bnd; fr.tr = tr;
        std::vector<int32_t> tl(ds41pf::tiles_bytes(R, NX) / 4);
        fr.n_tiles = ds41pf::build_tiles(off.data(), NX, tr, tl.data());
        ck(cudaMemcpy(d_tiles, tl.data(), 8 * (size_t) fr.n_tiles, cudaMemcpyHostToDevice), "tiles h2d");
        fr.tiles = d_tiles;
        ck(cudaMemset(d_out_f, 0xFF, 4 * R * H), "out f nan");   // NaN everywhere: a row the kernels miss shows
        ds41pf::quantize_rows(d_x, d_tok, R, H, d_xa, s);
        ds41pf::experts(fr, d_xa, d_ha, d_dst, d_out_f, s);
        ck(cudaStreamSynchronize(s), "fused");
        out.assign((size_t) R * H, 0.0f);
        ck(cudaMemcpy(out.data(), d_out_f, 4 * R * H, cudaMemcpyDeviceToHost), "out f d2h");
    };
    std::vector<float> f64, f128;
    fused(64, f64);
    fused(128, f128);
    const bool tiles_eq = std::memcmp(f64.data(), f128.data(), 4 * f64.size()) == 0;

    // ---- FP32 reference on a subset of entries (every 6th, and every row of the experts with < 64 rows)
    std::vector<int64_t> sub_e;
    for (int q = 0; q < NX; ++q)
        for (int r = off[q]; r < off[q + 1]; ++r)
            if (cnt[q] < 64 || (r - off[q]) % 6 == 0) sub_e.push_back(r);
    std::vector<float> ref((size_t) R * H, 0.0f);
    {
        const auto* tg = ggml_get_type_traits((ggml_type) gt);
        const auto* td = ggml_get_type_traits((ggml_type) dt);
        for (int q = 0; q < NX; ++q) {
            std::vector<float> Wg((size_t) FF * H), Wu((size_t) FF * H), Wd((size_t) H * FF);
            tg->to_float(gm[q].data(), Wg.data(), FF * H);
            tg->to_float(um[q].data(), Wu.data(), FF * H);
            td->to_float(dmx[q].data(), Wd.data(), H * FF);
            std::vector<int64_t> mine;
            for (int64_t r : sub_e) if (r >= off[q] && r < off[q + 1]) mine.push_back(r);
            const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
            std::vector<std::thread> th;
            for (unsigned t = 0; t < nt; ++t)
                th.emplace_back([&, t] {
                    std::vector<double> hh((size_t) FF);
                    for (size_t k = t; k < mine.size(); k += nt) {
                        const int64_t r = mine[k];
                        const float* xr = x.data() + (size_t) tok[(size_t) r] * H;
                        for (int64_t f = 0; f < FF; ++f) {
                            double g = 0, u = 0;
                            const float* wg = Wg.data() + (size_t) f * H;
                            const float* wu = Wu.data() + (size_t) f * H;
                            for (int64_t j = 0; j < H; ++j) { g += (double) wg[j] * xr[j]; u += (double) wu[j] * xr[j]; }
                            g = std::min(g, (double) kLim);
                            u = std::min(std::max(u, -(double) kLim), (double) kLim);
                            hh[(size_t) f] = g / (1.0 + std::exp(-g)) * u;
                        }
                        float* o = ref.data() + (size_t) dst[(size_t) r] * H;
                        for (int64_t i = 0; i < H; ++i) {
                            double a = 0;
                            const float* wd = Wd.data() + (size_t) i * FF;
                            for (int64_t f = 0; f < FF; ++f) a += (double) wd[f] * hh[(size_t) f];
                            o[i] = (float) a;
                        }
                    }
                });
            for (auto& t : th) t.join();
        }
    }
    std::vector<int64_t> all_rows((size_t) R), sub_rows;
    std::iota(all_rows.begin(), all_rows.end(), 0);
    for (int64_t r : sub_e) sub_rows.push_back(dst[(size_t) r]);   // output rows of the reference subset

    const bool fin = finite_all(f64) && (!mmq_ok || finite_all(out_m));
    const double f_ref = rel_l2(f64.data(), ref.data(), sub_rows), f_ref_max = max_rel(f64.data(), ref.data(), sub_rows);
    double m_ref = NAN, m_ref_max = NAN, fm = NAN, fm_max = NAN;
    if (mmq_ok) {
        m_ref = rel_l2(out_m.data(), ref.data(), sub_rows);
        m_ref_max = max_rel(out_m.data(), ref.data(), sub_rows);
        fm = rel_l2(f64.data(), out_m.data(), all_rows);
        fm_max = max_rel(f64.data(), out_m.data(), all_rows);
    }
    const bool ok = fin && tiles_eq && (!mmq_ok || f_ref <= 1.5 * m_ref + 0.002);
    std::printf("%-8s / %-8s  fused~mmq L2 %.5f max %.5f | mmq~fp32 L2 %.5f max %.5f | fused~fp32 L2 %.5f max %.5f | tile64==128 %s%s  %s\n",
                ggml_type_name((ggml_type) gt), ggml_type_name((ggml_type) dt), fm, fm_max, m_ref, m_ref_max, f_ref, f_ref_max,
                tiles_eq ? "yes" : "NO", mmq_ok ? "" : " (MMQ does not fit: no MMQ comparison)", ok ? "ok" : "FAIL");
    std::fflush(stdout);
    for (void* p : {(void*) d_w, (void*) d_x, (void*) d_gu, (void*) d_h, (void*) d_out_m, (void*) d_out_f, (void*) d_tok,
                    (void*) d_dst, (void*) d_bnd, (void*) d_iota, (void*) d_tiles, d_xg, d_xd, d_xa, d_ha})
        cudaFree(p);
    cudaStreamDestroy(s);
    return ok;
}

}  // namespace

int main() {
    if (!ds41pf::available()) {
        std::fprintf(stderr, "test_ds41_pfused: the fused kernels are not available on this device (sm_80+ needed)\n");
        return 2;
    }
    if (!mmq::built()) std::fprintf(stderr, "test_ds41_pfused: no MMQ in this build: fused vs FP32 only\n");
    mmq::Context ctx;
    const int pairs[][2] = {
        {GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S},     {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS}, {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS},  {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S}, {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS},
        {GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_XXS},   {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S},
    };
    int fails = 0;
    for (const auto& p : pairs) fails += run_pair(p[0], p[1], ctx) ? 0 : 1;
    std::printf("%s (%d of %d pairs failed)\n", fails ? "FAIL" : "PASS", fails, (int) (sizeof(pairs) / sizeof(pairs[0])));
    return fails ? 1 : 0;
}
