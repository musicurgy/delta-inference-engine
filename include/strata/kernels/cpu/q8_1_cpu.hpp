// include/strata/kernels/cpu/q8_1_cpu.hpp - STRATA_CPU_Q8_1: the CPU expert pool
// computes with the card's activation quantization.
//
// Default (switch unset): the pool quantizes an expert's activations as Q8_K (one scale per 256 values) and the card
// as q8_1 (one fp16 scale per 32 values), so the same expert gives a different result on each side, and which side
// computes an expert depends on the VRAM cache (measured: p(first token) 0.57-0.99 for one prompt).
// STRATA_CPU_Q8_1=1: every activation whose ggml vec_dot type is Q8_K (the i-quants and K-quants: IQ1_S, IQ2_XXS,
// Q2_K, ...) is quantized as q8_1 instead - the card's quantizer, value for value (q8_1_quant_row: d = amax / 127,
// roundf(x / d), fp16 d and block sum, q8_1_finite's clamp) - and the row kernels compute what the card's
// vec_dot_*_q8_1 computes: per 32-block integer sums with the card's integer rules (IQ2_XXS's sumi * ls / 8
// truncation, IQ1_S's delta times the block's fp16 SUM of the raw activations, Q2_K's mins times the quantized sum),
// scaled per block in float.  Only the float accumulation order differs.  Other formats with Q8_K activations take a
// float path (ggml's dequantizer times the dequantized q8_1 values): the scale blocks match, the card's integer quirks
// are not mirrored.  Activations of other types (Q8_0, F32 for EXL3, Q2_0's ActQ) are unchanged.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// STRATA_CPU_Q8_1=1 (read once).
bool cpu_q8_1_on() noexcept;
/// The switch is on and `act_type` (a NativeFmt gu_act / d_act) is Q8_K: that activation is q8_1 instead.
bool cpu_q8_1_act(int act_type) noexcept;

inline constexpr size_t kQ81Bytes = 36;   ///< one q8_1 block: fp16 d, fp16 sum, 32 int8
inline size_t q8_1_row_bytes(int64_t n) { return (size_t) (n / 32) * kQ81Bytes; }

/// `n` floats (a multiple of 32) as q8_1 blocks - the card's quantize_q8_1 / native_quantize_q8_1 arithmetic with
/// IEEE division (the card's file is fast-math; gpu_run_n_ex uploads these bytes under the switch, so both sides
/// read the same activation there).
void q8_1_quant_row(const float* x, void* dst, int64_t n);

/// The integer-mirroring kernels' formats: Q2_K (10), IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ2_S (22)
/// - every expert type of the UD-IQ2_XXS / UD-IQ1_S packs.
bool q8_1_int_type(int type) noexcept;
/// Bytes of one 256-value super-block of those formats (0 for any other).
size_t q8_1_block_bytes(int type) noexcept;

/// out[t][r] = row r of the `type` matrix at `w` (row stride `row_bytes`, `n` columns) . act[t] (q8_1, n values),
/// for r in [r0, r1) and t < nt.  AVX2 where the CPU has it (the q8_1_int_type formats), else the scalar reference.
void q8_1_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
               int r0, int r1);
/// The scalar reference (any type ggml can dequantize; the integer rules for 10/16/19): the non-AVX2 path and the
/// unit test's comparison.
void q8_1_rows_ref(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1);
/// q8_1_avx2.cpp (compiled for AVX2; call only where cpu_avx2_ok()): the q8_1_int_type formats, n a multiple of 256.
void q8_1_rows_avx2(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
