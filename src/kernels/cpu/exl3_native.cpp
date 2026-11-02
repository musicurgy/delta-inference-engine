// src/kernels/cpu/exl3_native.cpp - EXL3 routed experts (exllamav3's trellis format, `mul1` codebook) as a native
// expert format of the CPU tier: the blob layout, the activations and the gate/up and down rows that native_expert.cpp
// hands to when a layer's type is an EXL3 one (is_exl3_type).  The arithmetic is exl3_cpu's (Project Maya's port of
// exllamav3's CPU expert kernels, MIT): y = H(H(x * suh) W') * svh per 128-column band.
//
// The blob is the checkpoint's own bytes, one part per projection in the order [gate | up | down], each part as
// LibertAIDAI's safetensors store an expert's tensors: suh (FP16, k) | svh (FP16, n) | mul1 (int32) | trellis
// ((k/16) x (n/16) x 4 K2 words) - so a blob is three file spans read verbatim (Ds41FileBlobs, EXL3 mode).
//
// The activations are plain F32 (no quantization): the transform depends on each expert's own suh, so it cannot be
// shared across a layer's experts as a quantized activation is; each row call prepares it (H(x * suh), ~2 us).  Rows are
// taken in whole 128-column bands (NativeFmt::row_align = 128: the pool cuts its row ranges on band edges).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/expert.hpp"   // MAXT
#include "strata/kernels/cpu/expert_layout.hpp"   // cpu_avx2_ok

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {

namespace {

Exl3Part exl3_part(const uint8_t* p, int64_t k, int64_t n, int type) {
    const int k2 = exl3_k2_of(type);
    const size_t so = exl3_suh_off(type);   // suh | svh | mul1 | trellis, or mul1 | suh | svh | trellis
    Exl3Part m;
    m.suh = (const uint16_t*) (p + so);
    m.svh = (const uint16_t*) (p + so + 2 * k);
    m.trellis = (const uint32_t*) (p + 2 * k + 2 * n + 4);
    m.k = (int) k;
    m.n = (int) n;
    m.k2 = k2;
    return m;
}

// per thread: the prepared inputs (H(x * suh)) of up to MAXT tokens and two matrices, and the band sums
struct Scratch {
    std::vector<float> xh, acc;
    void fit(size_t k) {
        if (xh.size() < 2 * MAXT * k) xh.resize(2 * MAXT * k);
        if (acc.size() < 2 * MAXT * 128) acc.resize(2 * MAXT * 128);
    }
};
thread_local Scratch tls;

// rows [r0, r1) (multiples of 128) of y = W x for nt tokens: out[t][r]
void exl3_rows(const Exl3Part& m, const float* const* x, int nt, int r0, int r1, float* const* y, float* xh_base,
               float* acc_base) {
    float* xh[MAXT];
    float* acc[MAXT];
    float xs[MAXT];
    for (int t = 0; t < nt; ++t) {
        xh[t] = xh_base + (size_t) t * m.k;
        acc[t] = acc_base + (size_t) t * 128;
        xs[t] = exl3_cpu_prepare(m, x[t], xh[t]);
    }
    for (int c0 = r0; c0 < r1; c0 += 128) {
        if (nt == 1) {
            exl3_cpu_band(m, xh[0], xs[0], c0, y[0] + c0);
            continue;
        }
        exl3_cpu_band_acc_n(m, xh, nt, c0, 0, m.k / 16, acc);
        for (int t = 0; t < nt; ++t) exl3_cpu_band_out(m, acc[t], xs[t], c0, y[t] + c0);
    }
}

}  // namespace

bool exl3_native_fmt(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f,
                     std::string& err) {
    const int k2 = exl3_k2_of(gate_type);
    if (up_type != gate_type || d_type != gate_type) {
        err = "native experts: an EXL3 expert needs one K for gate, up and down";
        return false;
    }
    // exl3_cpu.cpp is compiled for AVX2 (MSVC has no per-function target; without /arch its intrinsics ran at a
    // fraction of their speed), so nothing in it may run before this check
    if (!cpu_avx2_ok()) {
        err = "native experts: EXL3 experts need an AVX2 CPU";
        return false;
    }
    if (!exl3_cpu_k2_ok(k2) || !exl3_cpu_available()) {
        err = "native experts: no EXL3 CPU kernel for K " + std::to_string(k2 / 2.0) + " on this CPU";
        return false;
    }
    if (n_embd % 128 || n_ff % 128) {
        err = "native experts: EXL3 needs n_embd and n_ff in whole 128-wide bands";
        return false;
    }
    f = NativeFmt();
    f.gu_type = f.d_type = gate_type;
    f.gu_act = f.d_act = kExl3ActType;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    const size_t p = exl3_native_part_bytes(n_embd, n_ff, k2);
    f.gu_row = f.d_row = 0;   // tiled, not row-major: the blob is three parts, not rows
    f.up_off = p;
    f.down_off = 2 * p;
    f.bytes = 3 * p;
    f.act_bytes = (size_t) n_embd * 4;
    f.h_bytes = (size_t) n_ff * 4;
    f.row_align = 128;
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: EXL3 activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void exl3_native_quant(const float* x, int64_t n, void* dst) { std::memcpy(dst, x, (size_t) n * 4); }

void exl3_native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                         int r0, int r1) {
    const Exl3Part g = exl3_part(blob, f.n_embd, f.n_ff, f.gu_type), u = exl3_part(blob + f.up_off, f.n_embd, f.n_ff, f.gu_type);
    tls.fit((size_t) f.n_embd);
    const float* x[MAXT];
    for (int t = 0; t < nt; ++t) x[t] = (const float*) act[t];
    thread_local std::vector<float> gb, ub;
    if (gb.size() < (size_t) MAXT * (size_t) f.n_ff) {
        gb.resize((size_t) MAXT * (size_t) f.n_ff);
        ub.resize((size_t) MAXT * (size_t) f.n_ff);
    }
    float* gp[MAXT];
    float* upp[MAXT];
    for (int t = 0; t < nt; ++t) {
        gp[t] = gb.data() + (size_t) t * f.n_ff;
        upp[t] = ub.data() + (size_t) t * f.n_ff;
    }
    exl3_rows(g, x, nt, r0, r1, gp, tls.xh.data(), tls.acc.data());
    exl3_rows(u, x, nt, r0, r1, upp, tls.xh.data() + (size_t) MAXT * f.n_embd, tls.acc.data() + MAXT * 128);
    const float lim = f.swiglu_limit;
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float gv = std::min(gp[t][r], lim);   // DeepSeek-V4's clamp, as native_gu_rows
            const float uv = std::min(std::max(upp[t][r], -lim), lim);
            ff[t][r] = (gv / (1.f + std::exp(-gv))) * uv;
        }
}

void exl3_native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                           int r0, int r1) {
    const Exl3Part d = exl3_part(blob + f.down_off, f.n_ff, f.n_embd, f.d_type);
    tls.fit((size_t) f.n_embd);
    const float* h[MAXT];
    for (int t = 0; t < nt; ++t) h[t] = (const float*) hq[t];
    exl3_rows(d, h, nt, r0, r1, out, tls.xh.data(), tls.acc.data());
}

}  // namespace strata::kernels::cpu
