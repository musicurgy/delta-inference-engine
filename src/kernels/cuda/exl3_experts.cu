// src/kernels/cuda/exl3_experts.cu - EXL3 routed experts on the GPU (Delta): native_expert_grouped's work for an EXL3
// class (is_exl3_type), on the same arguments - groups of entries, one expert blob each, q8_1 token activations in,
// one output row per entry out.
//
// The device arithmetic is Project Maya's (github.com/mw00/project-maya @ 249f9a8, src/kernels/cuda/exl3.cu, MIT:
// third_party/project-maya/LICENSE), which follows exllamav3 (MIT, turboderp: third_party/exllamav3/LICENSE): the
// mul1 codebook, the trellis windows, the 128-point rotation and the block GEMV that decodes a tile straight into a
// tensor-core fragment (block_gemv, copied with the CUDA paths only and the mul1 codebook only).  The kernels around
// it are ours, shaped after Maya's win_gu / win_down: a block owns 128 outputs of one group's expert and runs that
// group's entries through it 8 at a time, so each weight tile is decoded once for up to 8 tokens.
//
// Blob layout (exl3_native.cpp): [gate | up | down], each part suh (FP16, k) | svh (FP16, n) | mul1 | trellis.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;   // outputs per block: 8 tiles, one rotation block
constexpr int kTok = 8;      // activations a block takes at once (the MMA's n)
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // ggml block_q8_1 (the engine's activation): d, sum, 32 int8
    __half2 ds;
    int8_t qs[32];
};

// ---------------------------------------------------------------- the mul1 codebook (Maya / exllamav3)
__device__ __forceinline__ __half2 decode2h(uint32_t s0, uint32_t s1) {
    const uint32_t a = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t b = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 h = __halves2half2(__ushort_as_half((unsigned short) a), __ushort_as_half((unsigned short) b));
    return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t) ((((uint64_t) hi << 32) | (uint64_t) lo) >> s);
}
__device__ __forceinline__ void decode_states(const uint32_t (&s)[8], uint32_t (&f)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = h2u(decode2h(s[2 * i] & 0xffffu, s[2 * i + 1] & 0xffffu));
}
// K = 4: the lane's steps end in its own word b, the first three reach into a (word lane - 1)
__device__ __forceinline__ void states4(uint32_t a, uint32_t b, uint32_t (&s)[8]) {
    const uint32_t x = fsh(b, a, 20);
    s[0] = x >> 8;
    s[1] = x >> 4;
    s[2] = x;
    s[3] = b >> 16;
    s[4] = b >> 12;
    s[5] = b >> 8;
    s[6] = b >> 4;
    s[7] = b;
}
// the lane's 8 steps (lane * 8 ..) of a tile with an integer K in {2, 3, 4} from its 8 K words: f[i] = steps 2i, 2i + 1
template <int K>
__device__ __forceinline__ void frag8(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    const int p = lane * 8;
    uint32_t s[8];
    if constexpr (K == 4) {
        states4(w[(lane + 31) & 31], w[lane], s);
    } else if constexpr (K == 2) {
        constexpr int NW = 8 * K, per = 32 / (8 * K);   // lanes per word
        const int i1 = lane / per, i0 = (i1 + NW - 1) % NW;
        const uint32_t b = fsh(w[i1], w[i0], (per - 1 - lane % per) * 8 * K);
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = b >> ((7 - j) * K);
    } else {
        static_assert(K == 3, "K 2, 3 or 4");
        constexpr int NW = 24;
        const int b1 = (p + 257) * 3, b0 = b1 - 16, b2 = b1 + 21;
        const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
        const uint32_t a = w[i0 % NW], b = w[i2 % NW];
        const uint32_t hi = fsh(b, a, s2), lo = fsh(b, a, s2 + 12);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[7 - j] = hi >> (3 * j);
            s[3 - j] = lo >> (3 * j);
        }
    }
    decode_states(s, f);
}

// Delta: any K2, for exllamav3 1.6's half-integer K (2.5 bpw experts: K2 5) - each of the lane's 8 steps from its own
// window, by the CPU lane's rule (exl3_cpu.cpp, win_end): step p's 16 bits end at bit floor((p + 1) K2 / 2) of the
// tile's 128 K2-bit ring, MSB first
template <int K2>
__device__ __forceinline__ void frag8_any(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    constexpr int W = 4 * K2;
    uint32_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b1 = ((lane * 8 + j + 1) * K2) / 2 + 128 * K2;
        const int i1 = ((b1 - 1) >> 5) % W, i0 = ((b1 - 16) >> 5) % W;
        s[j] = fsh(w[i1], w[i0], (((b1 - 1) >> 5) + 1) * 32 - b1);
    }
    decode_states(s, f);
}

// ---------------------------------------------------------------- the rotation (Maya)
__device__ __forceinline__ void had128(float* v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}
__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

struct Xin {   // a warp's activation: F32 (f) or q8_1 blocks (q)
    const float* f = nullptr;
    const Q81* q = nullptr;
};
__device__ __forceinline__ void chunk_load(const Xin& x, const uint16_t* __restrict__ suh, int k0, int lane,
                                           float (&v)[4], float (&s)[4]) {
    const int base = k0 + lane * 4;
    if (x.f != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = x.f[base + i];
    } else {
        const Q81* b = x.q + base / 32;
        const float d = __low2float(b->ds);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = h2f(suh[base + i]);
}
__device__ __forceinline__ void chunk_store(float (&v)[4], const float (&s)[4], int lane, __half* dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= s[i];
    had128(v, lane);
    __half2* d = reinterpret_cast<__half2*>(dst + lane * 4);
    d[0] = __floats2half2_rn(v[0], v[1]);
    d[1] = __floats2half2_rn(v[2], v[3]);
}

// ---------------------------------------------------------------- the block GEMV (Maya)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define EXL3_TC 1
#endif

#if !defined(EXL3_TC)
__device__ __forceinline__ float dot4_acc(float a, const float* v, float x0, float x1, float x2, float x3) {
    return __fadd_rn(a, __fmaf_rn(v[3], x3, __fmaf_rn(v[2], x2, __fmaf_rn(v[1], x1, __fmul_rn(v[0], x0)))));
}
#else
__device__ __forceinline__ void mma_f32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

template <int K2>
struct Words {
    static constexpr int W = 4 * K2, R = (W + 31) / 32;
};
template <int K2>
__device__ __forceinline__ void load_tiles(const uint32_t* __restrict__ tile0, size_t sstride, int lane,
                                           uint32_t (&w)[8][Words<K2>::R]) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = lane + 32 * r;
            w[s][r] = (W % 32 == 0 || i < W) ? __ldg(tile0 + (size_t) s * sstride + i) : 0u;
        }
}
template <int K2>
__device__ __forceinline__ void slice_frags(const uint32_t (&w)[Words<K2>::R], uint32_t* st, int lane, uint32_t (&f)[4]) {
    if constexpr (K2 == 8) {
        uint32_t s[8];
        states4(__shfl_sync(0xffffffffu, w[0], (lane + 31) & 31), w[0], s);
        decode_states(s, f);
    } else {
        constexpr int W = Words<K2>::W, R = Words<K2>::R;
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (W % 32 == 0 || lane + 32 * r < W) st[lane + 32 * r] = w[r];
        __syncwarp();
        if constexpr (K2 % 2) frag8_any<K2>(st, lane, f);   // Delta: half-integer K
        else frag8<K2 / 2>(st, lane, f);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
constexpr int kMinBlocks = 4;   // Maya's measured bound for K <= 4 on an RTX 3090
#else
constexpr int kMinBlocks = 2;
#endif

template <int K2>
struct BlockSmem {
    __half xb[2 * kTok * kCols];
    uint32_t stage[kWarps * 4 * K2];
    float out[kTok * kCols];
};

// 128 outputs (tile columns tc0 .. tc0 + 7) over the inputs [k0, k1) for nc <= kTok activations (warp c's); out[c * 128 + i]
template <int K2>
__device__ void block_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int k0, int k1, const Xin& x,
                           const uint16_t* __restrict__ suh, int nc, BlockSmem<K2>& sm) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    const bool rot = warp < nc;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[4], xs4[4];
    load_tiles<K2>(tile, sstride, lane, wc);
    if (rot) chunk_load(x, suh, k0, lane, xv, xs4);
#if defined(EXL3_TC)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#else
    float acc[kTok][2];
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;
#endif
    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (kTok * kCols);
        if (rot) chunk_store(xv, xs4, lane, xq + warp * kCols);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
            if (rot) chunk_load(x, suh, k0 + (q + 1) * kCols, lane, xv, xs4);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2>(wc[s], st, lane, f);
#if defined(EXL3_TC)
            uint32_t b0 = 0u, b1 = 0u;
            if (g < nc) {
                const __half* xr = xq + g * kCols + s * 16 + 2 * t;
                b0 = *reinterpret_cast<const uint32_t*>(xr);
                b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
            }
            mma_f32(c, f[0], f[2], f[1], f[3], b0, b1);
#else
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));
                v[2 * i] = d.x;
                v[2 * i + 1] = d.y;
            }
#pragma unroll
            for (int cc = 0; cc < kTok; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const float x0 = __half2float(xr[0]), x1 = __half2float(xr[1]);
                    const float x2 = __half2float(xr[8]), x3 = __half2float(xr[9]);
                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
    const int o = warp * 16 + g;
#if defined(EXL3_TC)
    if (2 * t < nc) {
        sm.out[(2 * t) * kCols + o] = c[0];
        sm.out[(2 * t) * kCols + o + 8] = c[2];
    }
    if (2 * t + 1 < nc) {
        sm.out[(2 * t + 1) * kCols + o] = c[1];
        sm.out[(2 * t + 1) * kCols + o + 8] = c[3];
    }
#else
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc)
        if (cc < nc) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 1);
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 2);
            }
            if (t == 0) {
                sm.out[cc * kCols + o] = acc[cc][0];
                sm.out[cc * kCols + o + 8] = acc[cc][1];
            }
        }
#endif
    __syncthreads();
}

// one warp: dst[i] = (H(red) * svh)[i] for the block's 128 outputs
__device__ __forceinline__ void block_out(const float* red, const uint16_t* svh, float* dst) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __fmul_rn(v[i], h2f(svh[lane * 4 + i]));
}

// ---------------------------------------------------------------- the grouped experts (ours)
struct KGrp {
    const unsigned long long* grp_ptr;
    const int32_t *grp_start, *n_groups, *ent_dst, *ent_tok;
    const Q81* xq;
    int xq_row;          // q8_1 blocks per token (n_embd / 32)
    float *gate, *up;    // [cap_entries][n_ff] by entry
    float* out;          // rows ent_dst[e], n_embd each
    size_t suh[3], svh[3], tr[3];   // inside a blob: gate, up, down parts
    int n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, groups striding, 2): group's gate (z 0) or up (z 1) for 128 outputs of each of its entries
template <int K2>
__global__ void __launch_bounds__(kThreads, kMinBlocks) exl3_gu_kernel(const KGrp a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    const int g = blockIdx.x, r = blockIdx.z, warp = threadIdx.x >> 5;
    const int ng = *a.n_groups;
    for (int grp = blockIdx.y; grp < ng; grp += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
        const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
        float* dst = r == 0 ? a.gate : a.up;
        for (int mm = m0; mm < m1; mm += kTok) {
            const int nc = min(kTok, m1 - mm);
            Xin x;
            if (warp < nc) x.q = a.xq + (size_t) a.ent_tok[mm + warp] * a.xq_row;
            block_gemv<K2>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                           (const uint16_t*) (blob + a.suh[r]), nc, sm);
            if (warp < nc)
                block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                          dst + (size_t) (mm + warp) * a.n_ff + g * kCols);
            __syncthreads();
        }
    }
}

// the clamped SwiGLU of every live entry, into gate (the native kernels' expression: g / (1 + exp(-g)) * u)
__global__ void exl3_swiglu_kernel(const KGrp a, int64_t cap) {
    const int64_t n = (int64_t) a.grp_start[*a.n_groups] * a.n_ff;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n && i < cap; i += (int64_t) gridDim.x * blockDim.x) {
        const float gv = fminf(a.gate[i], a.limit), uv = fminf(fmaxf(a.up[i], -a.limit), a.limit);
        a.gate[i] = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
    }
}

// grid (n_embd / 128, groups striding): group's down rows for 128 outputs of each entry, into out row ent_dst[e]
template <int K2>
__global__ void __launch_bounds__(kThreads, kMinBlocks) exl3_down_kernel(const KGrp a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    const int g = blockIdx.x, warp = threadIdx.x >> 5;
    const int ng = *a.n_groups;
    for (int grp = blockIdx.y; grp < ng; grp += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
        const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
        for (int mm = m0; mm < m1; mm += kTok) {
            const int nc = min(kTok, m1 - mm);
            Xin x;
            if (warp < nc) x.f = a.gate + (size_t) (mm + warp) * a.n_ff;
            block_gemv<K2>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                           (const uint16_t*) (blob + a.suh[2]), nc, sm);
            if (warp < nc)
                block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                          a.out + (size_t) a.ent_dst[mm + warp] * a.n_embd + g * kCols);
            __syncthreads();
        }
    }
}

template <int K2>
void launch(const KGrp& a, int64_t gy, int64_t cap_entries, cudaStream_t s) {
    exl3_gu_kernel<K2><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) gy, 2), kThreads, 0, s>>>(a);
    const int64_t n = cap_entries * a.n_ff;
    exl3_swiglu_kernel<<<(unsigned) std::min<int64_t>((n + 255) / 256, 4096), 256, 0, s>>>(a, n);
    exl3_down_kernel<K2><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) gy), kThreads, 0, s>>>(a);
}

}  // namespace

bool exl3_expert_supported(int k2, int64_t n_embd, int64_t n_ff) noexcept {
    return (k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8) && n_embd % kCols == 0 && n_ff % kCols == 0;
}

void exl3_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t gy,
                         int64_t cap_entries, const void* x_q8_1, float* gate, float* up, float* out, void* stream) {
    const int k2 = cpu::exl3_k2_of(L.gu_type);
    const int64_t D = L.n_embd, FF = L.n_ff;
    KGrp a;
    a.grp_ptr = grp_ptr;
    a.grp_start = grp_start;
    a.n_groups = n_groups;
    a.ent_dst = ent_dst;
    a.ent_tok = ent_tok;
    a.xq = (const Q81*) x_q8_1;
    a.xq_row = (int) (D / 32);
    a.gate = gate;
    a.up = up;
    a.out = out;
    const size_t parts[3] = {0, L.up_off, L.down_off};
    for (int r = 0; r < 3; ++r) {
        const int64_t k = r == 2 ? FF : D, n = r == 2 ? D : FF;
        a.suh[r] = parts[r] + cpu::exl3_suh_off(L.gu_type);   // Delta: either tensor order (native_expert.hpp)
        a.svh[r] = a.suh[r] + 2 * (size_t) k;
        a.tr[r] = parts[r] + 2 * (size_t) k + 2 * (size_t) n + 4;
    }
    a.n_embd = (int) D;
    a.n_ff = (int) FF;
    a.limit = L.swiglu_limit;   // +inf: fminf / fmaxf leave the values as they are
    cudaStream_t s = (cudaStream_t) stream;
    switch (k2) {
        case 4: launch<4>(a, gy, cap_entries, s); break;
        case 5: launch<5>(a, gy, cap_entries, s); break;   // Delta: K 2.5 (bertholomus' 2.5 bpw experts)
        case 6: launch<6>(a, gy, cap_entries, s); break;
        case 8: launch<8>(a, gy, cap_entries, s); break;
        default: std::fprintf(stderr, "exl3_expert_grouped: no kernel for K2 %d\n", k2); std::exit(1);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "exl3_expert_grouped: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
