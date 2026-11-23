// tools/v41/engine/ds41_pfused.cu - see ds41_pfused.hpp (Strata's fused prompt-expert kernels for
// DeepSeek-V4.1-Flash, opt-in STRATA_PROMPT_FUSED=1).
//
// Ported from Strata src/prefill/moe_fused_iq.cu at baf1636 (native_kernel, its 512-thread instances; load_unit /
// convert for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS; quant_act_nat_kernel; the mma / ldmatrix / cp.async helpers).  The
// arithmetic, as upstream describes it: an activation block of 32 values is x = d_x * a (a = round(x / d_x) in
// -127..127, natural order; per 64 values 64 codes, {d_0, d_1} and the two code sums: 80 bytes).  A weight block of 32
// (or 16) values is w = d_w * q (+ m_w for IQ1_S), q an int8 decoded from the codebook with its signs.  The int32 dot
// starts from C = 0x4B400000 (the bits of 1.5 * 2^23) so as_float(D) - 1.5 * 2^23 is the dot as a float; per output
// and 32 values: d_w * d_x times it, plus m_w * d_x * sum a for IQ1_S.  Per 64 values of K (a stage) a thread decodes
// one 32-value sub-block of one weight row of the work item into shared memory (double-buffered), the activations
// arrive by cp.async four stages deep, the fragments by ldmatrix, the products are mma.sync m16n8k32 (m16n8k16 for
// the formats with a scale per 16 values: IQ2_XS, IQ2_S).
//
// The gate/up product's work item holds gate and up rows of the same features in one m16 tile (gate rows 0-7, up rows
// 8-15), so its epilogue has both in one lane: the clamped SwiGLU, then H rounded to int8 per 32 features (the down
// product's activation layout) - no FP32 gate/up rows, no SwiGLU pass, no quantize pass between the products.
#include "ds41_pfused.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ds41pf {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds41 prompt fused experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, {sum0, sum1}
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert): 256 x 64 or 128 x 128
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww) { return THREADS / 32 / ww * 16; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
constexpr int GU_K = kEmbd, D_K = kFF;            // K of gate/up and of down
constexpr int GU_N = 2 * kFF, D_N = kEmbd;        // weight rows of the gate/up and of the down product
constexpr int H_ROW_BYTES = D_K / 64 * AB;        // a row of H in the int8 activation layout (2,880 bytes)
static_assert(GU_K % 256 == 0 && D_K % 256 == 0, "both K are whole 256-value super-blocks (every covered format)");
static_assert(GU_N % 256 == 0 && D_N % 256 == 0, "both products are whole work items at either tile width");

constexpr int T_IQ1_S = GGML_TYPE_IQ1_S, T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS,
              T_IQ2_S = GGML_TYPE_IQ2_S, T_IQ3_XXS = GGML_TYPE_IQ3_XXS;
static_assert(sizeof(block_iq1_s) == 50 && sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 &&
              sizeof(block_iq2_s) == 82 && sizeof(block_iq3_xxs) == 98, "the block layouts this file decodes");

__host__ __device__ constexpr bool covered_type(int t) {
    return t == T_IQ1_S || t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS;
}
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ1_S ? 50 : t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : 98;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr bool has_min(int t) { return t == T_IQ1_S; }   // w = s0 q + s1 (IQ1_S's delta term)
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ1_S ? 2048 * 4 : t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : 256 * 4;
}
__host__ __device__ constexpr size_t smem_bytes(int t, int ww) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB + (size_t) tile_rows(ww) * 4 +
           grid_bytes(t);
}

// ---- activations in natural order: one warp per 64 values; row r of the output is row tok[r] of x (tok null: r)
__global__ void quant_rows_kernel(const float* __restrict__ x, const int32_t* __restrict__ tok, int64_t nblk, int bpr,
                                  int64_t cols, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const int64_t r = w / bpr, b = w % bpr;
    const float* xr = x + (int64_t) (tok ? tok[r] : r) * cols + 64 * b;
    const float v0 = xr[lane], v1 = xr[32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (IQ1_S's delta term reads them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col)
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
// Raw bytes of sub-block `ib` (0..7) of the 256-value super-block at `bp` (GGUF blocks: 2-byte aligned)
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[4]) {
    if constexpr (T == T_IQ1_S) {   // d, qs[32], qh[8] (u16)
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld16(bp + 34 + 2 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else {   // IQ3_XXS
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    }
}
// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31; IQ1_S: s0 the scale,
// s1 the delta term's coefficient)
template <int T>
__device__ __forceinline__ void convert(const uint32_t (&w)[4], const uint8_t* grid, uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ1_S) {
        // llama.cpp's MMQ (load_tiles_iq1_s): iq1s_grid_gpu's nibbles are g + 1 in 0..2, low nibbles values 0-3 of a
        // group of 8, high nibbles 4-7; w = d1q (g + delta) = d1q (g + 1) + d1q (delta - 1)
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = w[1];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t e = g[((w[0] >> (8 * l)) & 255) | (((qh >> (3 * l)) & 7) << 8)];
            q[2 * l] = e & 0x0F0F0F0Fu;
            q[2 * l + 1] = (e >> 4) & 0x0F0F0F0Fu;
        }
        const float d1q = half_at(w[2]) * (float) (((qh >> 11) & 0x0E) + 1);
        s0 = d1q;
        s1 = d1q * (-1.0f + IQ1S_DELTA - (float) (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000));
    } else if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else {   // IQ3_XXS
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    }
}
template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ1_S) return iq1s_grid_gpu;
    else if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else return iq3xxs_grid;
}
#endif

struct Geo {
    const uint8_t* base;
    size_t stride, gu_row, d_row, up_off, down_off;
    float lim;
};

// One work item = (TR routed rows of expert e: one tile of the run's list, a block of WR weight rows); a persistent
// grid walks the run's items.  GU: gate/up rows of WR / 2 features (format WT) against the run's rows (xa gathered by
// the caller), clamped SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm` row dst[row].  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma
// rows 0-7 and up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1)
fused_kernel(const Geo geo, const int32_t* __restrict__ off, const int2* __restrict__ tiles, int n_tiles,
             const uint8_t* __restrict__ act, uint8_t* __restrict__ out, const int32_t* __restrict__ dst,
             float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    static_assert(2 * WR <= THREADS, "every decode unit is a thread of the block");
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_K : D_K) / 64;          // 64-value stages along K
    constexpr int NFB = (GU ? GU_N : D_N) / WR;         // weight-row blocks per tile
    static_assert(GU_N % WR == 0 && D_N % WR == 0, "whole work items");
    constexpr int ACT_LD = NS * AB;                     // bytes per activation row: 6,400 (a token), 2,880 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [ASTAGES][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int nwork = n_tiles * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tiles[w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        const int nrows = min(TR, off[e + 1] - row0);
        const uint8_t* blob = geo.base + (size_t) e * geo.stride;
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) + (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* { return wrow + (s >> 2) * BS; };   // a 256-value block per four stages
        auto sub = [&](int s) { return 2 * (s & 3) + uj; };
        // the weight bytes into L2 ahead of the register loads: PF super-blocks ahead (the row's two threads take a
        // block's first and last byte)
        constexpr int PF = 2;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < NS / 4) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
#pragma unroll
        for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? row0 + tid : -1;
        __syncthreads();                                          // srow
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[4], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[4] = {0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
            __syncthreads();
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if ((s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]: d_x of the two halves
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
                float2 sx[2][2];                                  // IQ1_S: d_x times the codes' sum, per half
                if constexpr (has_min(WT)) {
#pragma unroll
                    for (int n = 0; n < 2; ++n)
#pragma unroll
                        for (int cc = 0; cc < 2; ++cc) {
                            const float2 sm = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 72);
                            sx[n][cc] = make_float2(dx[n][cc].x * sm.x, dx[n][cc].y * sm.y);
                        }
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                    if constexpr (has_min(WT))   // w = s0 q + s1: + s1 d_x (sum of the codes)
                                        acc[i][n][q] = fmaf(q < 2 ? wa1 : wb1, h ? sx[n][q & 1].y : sx[n][q & 1].x, acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if constexpr (GU) {
            // clamped SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and
            // + 8), tile rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value
            // half-block of the down product's K), with the codes' sum (an IQ1_S down reads it)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = fminf(acc[i][n][c], geo.lim), up = fminf(fmaxf(acc[i][n][2 + c], -geo.lim), geo.lim);
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    int code[4], cs = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { code[i] = __float2int_rn(h[i] * inv); cs += code[i]; }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) cs += __shfl_xor_sync(0xffffffffu, cs, o);
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) code[i];
                        if (g == 0) {
                            *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                            *(float*) (o + 72 + 4 * hh) = (float) cs;
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r >= nrows) continue;
                    float* drow = dm + (size_t) dst[row0 + r] * D_N + rbase + 64 * wf + g;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        drow[16 * i] = acc[i][n][c];
                        drow[16 * i + 8] = acc[i][n][2 + c];
                    }
                }
            }
        }
    }
#else
    (void) geo; (void) off; (void) tiles; (void) n_tiles; (void) act; (void) out; (void) dst; (void) dm;
#endif
}

// ---- per device set-up: the shared memory opt-in of every instance, and the blocks a multiprocessor runs
struct DevInfo {
    bool ok = false;
    int sms = 0, occ = 1 << 30;
};
template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    const size_t sm = smem_bytes(T, WW);
    if (cudaFuncSetAttribute(fused_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, fused_kernel<T, GU, WW>, THREADS, sm) != cudaSuccess || o < 1) return false;
    occ = std::min(occ, o);
    return true;
}
template <int T> bool setup_type(int& occ) {
    return setup_ww<T, true, 2>(occ) && setup_ww<T, true, 4>(occ) && setup_ww<T, false, 2>(occ) && setup_ww<T, false, 4>(occ);
}
constexpr int kMaxDev = 64;
const DevInfo& dev_info() {
    static DevInfo info[kMaxDev];
    static bool done[kMaxDev] = {};
    static std::mutex mu;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDev) {
        (void) cudaGetLastError();
        static const DevInfo none;
        return none;
    }
    std::lock_guard<std::mutex> lk(mu);
    if (!done[dev]) {
        done[dev] = true;
        DevInfo& d = info[dev];
        int major = 0, optin = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaFuncAttributes fa{};
        const bool code = cudaFuncGetAttributes(&fa, fused_kernel<T_IQ2_XXS, true, 4>) == cudaSuccess && fa.ptxVersion >= 80;
        size_t need = 0;
        for (int t : {T_IQ1_S, T_IQ2_XXS, T_IQ2_XS, T_IQ2_S, T_IQ3_XXS})
            for (int ww : {2, 4}) need = std::max(need, smem_bytes(t, ww));
        if (major >= 8 && code && (size_t) optin >= need) {
            int occ = 1 << 30;
            d.ok = setup_type<T_IQ1_S>(occ) && setup_type<T_IQ2_XXS>(occ) && setup_type<T_IQ2_XS>(occ) &&
                   setup_type<T_IQ2_S>(occ) && setup_type<T_IQ3_XXS>(occ);
            d.occ = d.ok ? occ : 1;
        }
        (void) cudaGetLastError();
        if (!d.ok) std::fprintf(stderr, "ds41 prompt fused experts: not available on device %d (sm_%d0+ with %zu KB of shared memory a block needed)\n",
                                dev, 8, need / 1024);
    }
    return info[dev];
}

template <int T, bool GU>
void launch(int ww, unsigned grid, const Geo& geo, const Run& r, const uint8_t* act, uint8_t* out, const int32_t* dst,
            float* dm, cudaStream_t s) {
    const int2* tl = (const int2*) r.tiles;
    if (ww == 4) fused_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(geo, r.off, tl, r.n_tiles, act, out, dst, dm);
    else fused_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(geo, r.off, tl, r.n_tiles, act, out, dst, dm);
}
template <bool GU>
void launch_type(int t, int ww, unsigned grid, const Geo& geo, const Run& r, const uint8_t* act, uint8_t* out,
                 const int32_t* dst, float* dm, cudaStream_t s) {
    switch (t) {
        case T_IQ1_S: launch<T_IQ1_S, GU>(ww, grid, geo, r, act, out, dst, dm, s); break;
        case T_IQ2_XXS: launch<T_IQ2_XXS, GU>(ww, grid, geo, r, act, out, dst, dm, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, GU>(ww, grid, geo, r, act, out, dst, dm, s); break;
        case T_IQ2_S: launch<T_IQ2_S, GU>(ww, grid, geo, r, act, out, dst, dm, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, GU>(ww, grid, geo, r, act, out, dst, dm, s); break;
        default:
            std::fprintf(stderr, "ds41 prompt fused experts: type %d is not covered\n", t);
            std::exit(1);
    }
}

}  // namespace

bool built() { return true; }
bool available() { return dev_info().ok; }
bool enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PROMPT_FUSED"); return v && v[0] == '1'; }();
    return on && available();
}
const char* uncovered(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, size_t up_off,
                      size_t down_off) {
    if (up_type >= 0 && up_type != gate_type) return "up type differs from gate type";
    if (!covered_type(gate_type)) return "gate/up type not IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS";
    if (!covered_type(down_type)) return "down type not IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS";
    if (n_embd != kEmbd) return "n_embd != kEmbd (5120)";
    if (n_ff != kFF) return "n_ff != kFF (2304)";
    if (up_off % 2 != 0 || down_off % 2 != 0) return "odd up/down offset";
    return nullptr;
}
bool covers(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off) {
    return uncovered(gate_type, up_type, down_type, n_embd, n_ff, up_off, down_off) == nullptr;
}
size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 64) * AB; }
size_t tiles_bytes(int64_t rows, int n_exp) { return (size_t) (rows / 64 + n_exp + 1) * 2 * sizeof(int32_t); }

void quantize_rows(const float* x, const int32_t* tok, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int bpr = (int) (cols / 64);
    const int64_t nblk = rows * bpr;
    quant_rows_kernel<<<(unsigned) ((nblk + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, tok, nblk, bpr, cols, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_rows");
}

int pick_tile(int64_t rows, int n_exp) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PROMPT_FUSED_TILE");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 || t == 128 ? t : 0;
    }();
    if (forced) return forced;
    // upstream's band (fused_tile_rule.hpp, the rule for every card outside its sm_120 / GB10 / staged classes): 128
    // routed rows when an expert has about one such tile (56 to 112 rows on average), else 64
    const double avg = (double) rows / (double) std::max(n_exp, 1);
    return avg > 56.0 && avg <= 112.0 ? 128 : 64;
}

int build_tiles(const int32_t* off, int n_exp, int tr, int32_t* out) {
    int k = 0;
    for (int q = 0; q < n_exp; ++q)
        for (int32_t r = off[q]; r < off[q + 1]; r += tr) {
            out[2 * k] = q;
            out[2 * k + 1] = r;
            ++k;
        }
    return k;
}

void experts(const Run& r, const void* xa, void* ha, const int32_t* dst, float* out, void* stream) {
    if (r.n_tiles <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok) {
        std::fprintf(stderr, "ds41 prompt fused experts: called on a device without them\n");
        std::exit(1);
    }
    const int ww = r.tr == 128 ? 2 : 4;
    const Geo geo{r.base, r.stride, r.gu_row, r.d_row, r.up_off, r.down_off, r.swiglu_limit};
    const int64_t cap = (int64_t) d.sms * d.occ;
    const unsigned g_gu = (unsigned) std::min<int64_t>((int64_t) r.n_tiles * (GU_N / weight_rows(ww)), cap);
    const unsigned g_d = (unsigned) std::min<int64_t>((int64_t) r.n_tiles * (D_N / weight_rows(ww)), cap);
    cudaStream_t s = (cudaStream_t) stream;
    launch_type<true>(r.gu_type, ww, g_gu, geo, r, (const uint8_t*) xa, (uint8_t*) ha, nullptr, nullptr, s);
    launch_type<false>(r.d_type, ww, g_d, geo, r, (const uint8_t*) ha, nullptr, dst, out, s);
    ck(cudaGetLastError(), "experts");
}

}  // namespace ds41pf
