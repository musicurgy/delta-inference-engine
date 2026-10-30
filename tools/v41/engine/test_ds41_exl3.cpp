// tools/v41/engine/test_ds41_exl3.cpp - the EXL3 CPU expert kernels (exl3_cpu, ported from Project Maya) against the
// scalar reference (exl3_ref) on synthetic tiles at DeepSeek-V4.1's shapes, and their speed on this CPU.
//
//   test_ds41_exl3                 parity: K 2 / 3 / 4, gate (5120 -> 2304) and down (2304 -> 5120), 4 random x each
//   test_ds41_exl3 --gpu           (test_ds41_exl3_gpu) whole experts, the GPU's grouped kernel (exl3_experts.cu, q8_1
//                                  activations) vs the CPU tier (native_gu_rows / native_down_rows, F32), 1..8 tokens
//   test_ds41_exl3 --bench T [MB [K2 [MODE]]]  (MODE 0 rows, 1 bands)
//                                  one-token experts (gate + up + down) from a buffer of MB megabytes (default 2048,
//                                  far past the L3), T threads each taking whole experts: experts/s and trellis GB/s
//
// The CPU lane keeps the codebook's affine form without the device's FP16 rounding of each weight, so the two differ by
// that rounding (rel ~1e-3); a wrong tile order, shift or transform gives rel ~1.
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"   // MAXT
#if defined(EXL3_TEST_GPU)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace kc = strata::kernels::cpu;
using strata::kernels::f16_from_f32;

namespace {

struct Mat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k = 0, n = 0, k2 = 0;
    kc::Exl3Part part() const { return {tr.data(), suh.data(), svh.data(), k, n, k2}; }
};

Mat make(int k, int n, int k2, std::mt19937& rng) {
    Mat m;
    m.k = k; m.n = n; m.k2 = k2;
    m.tr.resize((size_t) (k / 16) * (n / 16) * 4 * k2);
    for (auto& w : m.tr) w = rng();
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    m.suh.resize((size_t) k);
    m.svh.resize((size_t) n);
    for (auto& s : m.suh) s = f16_from_f32((rng() & 1 ? 1.f : -1.f) * u(rng));
    for (auto& s : m.svh) s = f16_from_f32(u(rng) * 0.01f);
    return m;
}

// y = the whole matrix through the lane's single-token path (prepare, split-K pieces of 16 slices, band outputs)
void lane_mv(const Mat& m, const float* x, float* y) {
    const kc::Exl3Part p = m.part();
    std::vector<float> xh((size_t) m.k), acc((size_t) m.n, 0.f), part((size_t) m.n);
    const float xs = kc::exl3_cpu_prepare(p, x, xh.data());
    for (int ks = 0; ks < m.k / 16; ks += 16) {
        kc::exl3_cpu_rows_acc(p, xh.data(), ks, std::min(ks + 16, m.k / 16), part.data());
        for (int c = 0; c < m.n; ++c) acc[(size_t) c] += part[(size_t) c];
    }
    for (int c0 = 0; c0 < m.n; c0 += 128) kc::exl3_cpu_band_out(p, acc.data() + c0, xs, c0, y + c0);
}

int parity() {
    std::mt19937 rng(1234);
    int bad = 0;
    for (int k2 : {4, 5, 6, 8})
        for (int role = 0; role < 2; ++role) {
            const int k = role == 0 ? 5120 : 2304, n = role == 0 ? 2304 : 5120;
            const Mat m = make(k, n, k2, rng);
            std::vector<float> w((size_t) k * n);
            kc::exl3_inner_k2(m.tr.data(), n / 16, k, n, k2, 2, w.data());
            double worst = 0;
            for (int rep = 0; rep < 4; ++rep) {
                std::vector<float> x((size_t) k), yr((size_t) n), yl((size_t) n), yb((size_t) n);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& v : x) v = nd(rng);
                kc::exl3_mv(w.data(), m.suh.data(), m.svh.data(), k, n, x.data(), yr.data());
                lane_mv(m, x.data(), yl.data());
                // the band path too (exl3_cpu_band: one 128-column band over every input)
                std::vector<float> xh((size_t) k);
                const float xs = kc::exl3_cpu_prepare(m.part(), x.data(), xh.data());
                for (int c0 = 0; c0 < n; c0 += 128) kc::exl3_cpu_band(m.part(), xh.data(), xs, c0, yb.data() + c0);
                double num = 0, den = 0, numb = 0;
                for (int c = 0; c < n; ++c) {
                    num += (double) (yl[c] - yr[c]) * (yl[c] - yr[c]);
                    numb += (double) (yb[c] - yr[c]) * (yb[c] - yr[c]);
                    den += (double) yr[c] * yr[c];
                }
                worst = std::max({worst, std::sqrt(num / den), std::sqrt(numb / den)});
            }
            const bool ok = worst < 5e-3;
            bad += !ok;
            std::printf("K%g %s %5d -> %5d: rel L2 vs reference %.2e  %s\n", k2 / 2.0, role == 0 ? "gate" : "down", k, n,
                        worst, ok ? "ok" : "FAIL");
        }
    std::printf("tier %d (%s)\n", kc::exl3_cpu_tier(), kc::exl3_cpu_tier() == 2 ? "AVX-512" : kc::exl3_cpu_tier() == 1 ? "AVX2" : "scalar");
    return bad ? 1 : 0;
}

// mode 0: split-K rows (exl3_cpu_rows_acc), 1: bands (exl3_cpu_band_acc, the engine's path: exl3_native.cpp)
int bench(int T, size_t mb, int k2, int mode) {
    const int D = 5120, FF = 2304;
    const size_t mat_words = (size_t) (D / 16) * (FF / 16) * 4 * k2;
    const size_t exp_bytes = 3 * mat_words * 4;
    const int NE = std::max<int>(T, (int) (mb * 1000000 / exp_bytes));
    std::vector<uint32_t> pool((size_t) NE * 3 * mat_words);
    std::mt19937 rng(7);
    for (auto& w : pool) w = rng();
    std::vector<uint16_t> suhD(D, f16_from_f32(1.f)), svhF(FF, f16_from_f32(0.01f)), suhF(FF, f16_from_f32(1.f)), svhD(D, f16_from_f32(0.01f));
    std::atomic<int> next{0};
    const int per_thread_runs = 3;
    auto work = [&](int /*t*/) {
        std::vector<float> x(D, 0.5f), xh(D), xh2(D), acc(FF), accu(FF), h(FF), hh(FF), accd(D), y(D), part(D);
        // one matrix's sums (out: m.n floats) by the mode; returns its xsum
        auto sums = [&](const kc::Exl3Part& m, const float* in, float* xhb, float* out) -> float {
            const float xs = kc::exl3_cpu_prepare(m, in, xhb);
            if (mode == 1)
                for (int c0 = 0; c0 < m.n; c0 += 128) kc::exl3_cpu_band_acc(m, xhb, c0, 0, m.k / 16, out + c0);
            else
                kc::exl3_cpu_rows_acc(m, xhb, 0, m.k / 16, out);
            return xs;
        };
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= NE * per_thread_runs) break;
            const uint32_t* base = pool.data() + (size_t) (i % NE) * 3 * mat_words;
            const kc::Exl3Part g{base, suhD.data(), svhF.data(), D, FF, k2}, u{base + mat_words, suhD.data(), svhF.data(), D, FF, k2},
                d{base + 2 * mat_words, suhF.data(), svhD.data(), FF, D, k2};
            const float xs = sums(g, x.data(), xh.data(), acc.data());
            const float xs2 = sums(u, x.data(), xh2.data(), accu.data());
            for (int c0 = 0; c0 < FF; c0 += 128) {
                kc::exl3_cpu_band_out(g, acc.data() + c0, xs, c0, h.data() + c0);
                kc::exl3_cpu_band_out(u, accu.data() + c0, xs2, c0, hh.data() + c0);
            }
            for (int c = 0; c < FF; ++c) h[c] = h[c] / (1.f + std::exp(-h[c])) * hh[c];
            std::vector<float> dh(FF);
            const float xsd = sums(d, h.data(), dh.data(), accd.data());
            for (int c0 = 0; c0 < D; c0 += 128) kc::exl3_cpu_band_out(d, accd.data() + c0, xsd, c0, y.data() + c0);
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(work, t);
    for (auto& t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double n = (double) NE * per_thread_runs;
    std::printf("%s K%g T%2d: %d experts (%.1f MB each) x %d: %.0f experts/s, %.3f ms an expert, trellis %.1f GB/s\n",
                mode == 1 ? "bands" : "rows ", k2 / 2.0,
                T, NE, exp_bytes / 1e6, per_thread_runs, n / s, 1e3 * s / n, n * exp_bytes / s / 1e9);
    return 0;
}


#if defined(EXL3_TEST_GPU)
// one synthetic expert blob [gate | up | down] (exl3_native.cpp's layout) per K, nt tokens: GPU vs CPU tier
int gpu_parity() {
    const int64_t D = 5120, FF = 2304;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool vb = std::getenv("EXL3_TEST_VERBOSE") != nullptr;   // stage markers, to place a crash
    auto stage = [&](const char* s) { if (vb) std::fprintf(stderr, "  [%s]\n", s); };
    std::mt19937 rng(99);
    int bad = 0;
    for (int k2 : {4, 5, 6, 8}) {
        kc::NativeFmt f;
        std::string err;
        const bool mf = k2 == 5;   // K 2.5 in bertholomus' order (mul1 | suh | svh | trellis), the rest in LibertAIDAI's
        const int xt = kc::exl3_type(k2, mf);
        if (!kc::native_fmt3(xt, xt, xt, D, FF, f, err)) {
            std::printf("native_fmt3: %s\n", err.c_str());
            return 1;
        }
        f.swiglu_limit = 10.0f;
        std::vector<uint8_t> blob(f.bytes);
        const size_t parts[3] = {0, f.up_off, f.down_off};
        for (int r = 0; r < 3; ++r) {
            const int64_t k = r == 2 ? FF : D, n = r == 2 ? D : FF;
            const Mat m = make((int) k, (int) n, k2, rng);
            uint8_t* p = blob.data() + parts[r];
            const uint32_t mul1 = 0x83DCD12Du;
            const size_t so = mf ? 4 : 0;
            std::memcpy(p + so, m.suh.data(), 2 * (size_t) k);
            std::memcpy(p + so + 2 * k, m.svh.data(), 2 * (size_t) n);
            std::memcpy(p + (mf ? 0 : 2 * k + 2 * n), &mul1, 4);
            std::memcpy(p + 2 * k + 2 * n + 4, m.tr.data(), m.tr.size() * 4);
        }
        stage("layout");
        strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout3(f.gu_type, f.gu_type, f.d_type, D, FF);
        L.swiglu_limit = f.swiglu_limit;
        if (L.bytes != f.bytes) { std::printf("GPU layout %zu != CPU blob %zu\n", L.bytes, f.bytes); return 1; }
        for (int nt : {1, 3, 8}) {
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> x((size_t) nt * D);
            for (auto& v : x) v = nd(rng);
            // CPU tier
            std::vector<float> cpu_out((size_t) nt * D), ff((size_t) nt * FF);
            std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(kc::kNativeActBytes)), hq(nt, std::vector<uint8_t>(kc::kNativeHBytes));
            const void* ap[kc::MAXT];
            const void* hp[kc::MAXT];
            float* fp[kc::MAXT];
            float* op[kc::MAXT];
            for (int t = 0; t < nt; ++t) {
                kc::native_quant_act(f, x.data() + (size_t) t * D, act[t].data());
                ap[t] = act[t].data();
                fp[t] = ff.data() + (size_t) t * FF;
                op[t] = cpu_out.data() + (size_t) t * D;
            }
            stage("cpu gu");
            kc::native_gu_rows(f, blob.data(), ap, nt, fp, 0, (int) FF);
            for (int t = 0; t < nt; ++t) {
                kc::native_quant_h(f, fp[t], hq[t].data());
                hp[t] = hq[t].data();
            }
            stage("cpu down");
            kc::native_down_rows(f, blob.data(), hp, nt, op, 0, (int) D);
            // GPU: one group (the blob), entries t -> token t, row t
            void *d_blob, *d_x, *d_xq, *d_scr;
            float* d_out;
            int32_t *d_start, *d_ng, *d_dst, *d_tok;
            unsigned long long* d_ptr;
            const size_t xq_row = strata::kernels::native_q8_1_bytes((int) D, 1);
            cudaMalloc(&d_blob, f.bytes);
            cudaMalloc(&d_x, x.size() * 4);
            cudaMalloc(&d_xq, (size_t) nt * xq_row);
            cudaMalloc(&d_scr, strata::kernels::native_expert_scratch_bytes(nt, FF));
            cudaMalloc((void**) &d_out, (size_t) nt * D * 4);
            cudaMalloc((void**) &d_start, 8);
            cudaMalloc((void**) &d_ng, 4);
            cudaMalloc((void**) &d_dst, 4 * nt);
            cudaMalloc((void**) &d_tok, 4 * nt);
            cudaMalloc((void**) &d_ptr, 8);
            cudaMemcpy(d_blob, blob.data(), f.bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            const int32_t start[2] = {0, nt}, ng = 1;
            std::vector<int32_t> ids(nt);
            for (int t = 0; t < nt; ++t) ids[t] = t;
            const unsigned long long ptr = (unsigned long long) d_blob;
            cudaMemcpy(d_start, start, 8, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ng, &ng, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst, ids.data(), 4 * nt, cudaMemcpyHostToDevice);
            cudaMemcpy(d_tok, ids.data(), 4 * nt, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ptr, &ptr, 8, cudaMemcpyHostToDevice);
            stage("gpu");
            cudaStream_t st;
            cudaStreamCreate(&st);
            strata::kernels::native_quantize_q8_1((const float*) d_x, d_xq, (int) D, nt, st);   // the MMVQ entry points require an explicit stream
            if (vb) { cudaError_t e0 = cudaDeviceSynchronize(); std::fprintf(stderr, "  [q8_1: %s]%c", cudaGetErrorString(e0), 10); }
            strata::kernels::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, nt, d_xq, d_scr, d_out, st);
            if (vb) { cudaError_t e1 = cudaDeviceSynchronize(); std::fprintf(stderr, "  [grouped: %s]%c", cudaGetErrorString(e1), 10); }
            std::vector<float> gpu_out((size_t) nt * D);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
            const cudaError_t ce = cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * 4, cudaMemcpyDeviceToHost);
            for (void* p : {d_blob, d_x, d_xq, d_scr, (void*) d_out, (void*) d_start, (void*) d_ng, (void*) d_dst, (void*) d_tok, (void*) d_ptr}) cudaFree(p);
            if (ce != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(ce)); return 1; }
            double worst = 0;
            for (int t = 0; t < nt; ++t) {
                double num = 0, den = 0;
                for (int64_t i = 0; i < D; ++i) {
                    const double a = gpu_out[(size_t) t * D + i], b = cpu_out[(size_t) t * D + i];
                    num += (a - b) * (a - b);
                    den += b * b;
                }
                worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
            }
            const bool ok = worst < 3e-2;   // q8_1 activations and FP16 weights / rotated inputs on the device
            bad += !ok;
            std::printf("GPU K%g, %d token%s: rel L2 vs the CPU tier %.2e  %s\n", k2 / 2.0, nt, nt > 1 ? "s" : " ", worst, ok ? "ok" : "FAIL");
        }
    }
    return bad ? 1 : 0;
}
#endif
// --dump FILE: the band kernels' raw sums (every band, 1..4 activations, K2 4 / 5 / 6 / 8, gate and down shapes) - run
// twice with STRATA_EXL3_VNNI=0 / 1 and compare the files: the VNNI byte sums must give the same bits (C2)
int dump(const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return 1;
    std::mt19937 rng(11);
    for (int k2 : {4, 5, 6, 8})
        for (int role = 0; role < 2; ++role) {
            const int k = role == 0 ? 5120 : 2304, n = role == 0 ? 2304 : 5120;
            const Mat m = make(k, n, k2, rng);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<std::vector<float>> xh(4, std::vector<float>((size_t) k)), acc(4, std::vector<float>((size_t) n));
            for (auto& v : xh)
                for (auto& x : v) x = nd(rng);
            for (int nt = 1; nt <= 4; ++nt) {
                const float* xp[4];
                float* ap[4];
                for (int i = 0; i < nt; ++i) xp[i] = xh[(size_t) i].data();
                for (int c0 = 0; c0 < n; c0 += 128) {
                    for (int i = 0; i < nt; ++i) ap[i] = acc[(size_t) i].data() + c0;
                    if (nt == 1) kc::exl3_cpu_band_acc(m.part(), xp[0], c0, 0, k / 16, ap[0]);
                    else kc::exl3_cpu_band_acc_n(m.part(), xp, nt, c0, 0, k / 16, ap);
                }
                for (int i = 0; i < nt; ++i) std::fwrite(acc[(size_t) i].data(), 4, (size_t) n, f);
            }
        }
    std::fclose(f);
    std::printf("dumped %s (tier %d)\n", path, kc::exl3_cpu_tier());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--dump") == 0) return dump(argv[2]);
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0) {
        const int T = std::atoi(argv[2]);
        const size_t mb = argc >= 4 ? (size_t) std::atoll(argv[3]) : 2048;
        const int k2 = argc >= 5 ? std::atoi(argv[4]) : 6;
        return bench(T, mb, k2, argc >= 6 ? std::atoi(argv[5]) : 0);
    }
#if defined(EXL3_TEST_GPU)
    if (argc >= 2 && std::strcmp(argv[1], "--gpu") == 0) return gpu_parity();
#endif
    return parity();
}
