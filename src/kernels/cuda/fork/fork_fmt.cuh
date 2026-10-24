// src/kernels/cuda/fork/fork_fmt.cuh - fork formats in the grouped kernels: Fmt<10/11/39> and the STRATA_*_FMTS_FORK lists.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

template<> struct Fmt<10> { static constexpr int qk = 256, ipb = QI2_K / VDR_Q2_K, step = VDR_Q2_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<11> { static constexpr int qk = 256, ipb = QI3_K / VDR_Q3_K, step = VDR_Q3_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q3_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<39> { static constexpr int qk = 32, ipb = QI_MXFP4 / VDR_MXFP4, step = VDR_MXFP4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_mxfp4_q8_1(v, y, kbx, iqs); } };

// PTQ1_0 (143): the whole 128-weight block per call (4 q8_1 sub-blocks), so ipb 1 and iqs always 0.
#ifndef STRATA_PTQ1_QUARTER   // the whole-block dot (llama-tq's) - default until the quarter split is measured
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 1, step = 0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1(v, y, kbx, iqs); } };
#else                         // -DSTRATA_PTQ1_QUARTER: a quarter block per call (fork_vecdot.cuh), 4x the calls per row, 1/4 the chain (UNMEASURED)
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 4, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1_quarter(v, y, kbx, iqs); } };
#endif
// Fork formats, kept out of upstream's lists above so those stay verbatim (fewer merge conflicts): Q2_K (10,
// DeepSeek-V4 down), Q3_K (11, MiMo GSQ-RCO / V4.1 tier), MXFP4 (39, MiMo outliers), PTQ1_0 (143, V4.1 ternary tier).  Each use site expands the _FORK list too.
//
// UNION (xolotl): GLM's branch had this list INLINE in its own iq_kernels.cu, with three entries the fork's
// extraction never picked up - Q4_K (12, GLM-5.3-Flash layers 3-5 down_exps; upstream's list has none) and
// IQ2_S/IQ3_XXS (18/22, the Maya-S-v2 down_exps).  The merge reconciled GLM's inline copy to this header and
// dropped them, so d_qk(18) == 0 and native_expert_grouped rejected the model at tier init:
//   "tier init: native_expert_grouped has no kernel for layer 3's types 16/16/18" (16/16 = IQ2_XXS gate/up,
// 18 = IQ2_S down) - i.e. GLM-5.3 could not load at all, while the DS4/DSV4.1 types were fine.  Measured
// against the working tree: ~/AI/Strata-GLM/build-glm-gpu/glm_generate passes the same point and prints
// "moe lora: routed-expert deltas on 26 layer(s) (n_embd 4096, n_ff 2048)".  Fmt<12>/Fmt<18>/Fmt<22> are all
// defined in this build, so this is the whole fix (see HANDOFF.md §6).
// 2026-10-23: IQ2_XXS / IQ2_XS / IQ1_M (16 / 17 / 29) as DOWN types too - ISTA-DASLab's GSQ-RCO DeepSeek-V4.1 builds
// pick them per layer for ffn_down_exps (FF 2304 = 9 whole 256-blocks), and test_ds41_moe stopped at tier init with
// "native_expert_grouped has no kernel for layer 0's types 29/29/29".  Their Fmt<> dots were already defined (gate/up).
// IQ1_S (19): ggml-cuda's vec_dot_iq1_s_q8_1 (vecdotq.cuh) verbatim - the GSQ-RCO IQ1_M build's main expert type
// (83 of its 120 role tensors).  Same iq1s grid as IQ1_M; no Split / grid staging (kSplit / kStageIqGrid stay false:
// the per-entry kernels call Fmt<19>::dot), so correct first, fast later.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
template<> struct Fmt<19> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_s_q8_1(v, y, kbx, iqs); } };

#define STRATA_GU_FMTS_FORK(X) X(10) X(11) X(19) X(143)
#define STRATA_D_FMTS_FORK(X) X(10) X(11) X(12) X(16) X(17) X(18) X(19) X(22) X(29) X(39) X(143)
#define STRATA_MMVQ_FMTS_FORK(X) X(10) X(19) X(143)
