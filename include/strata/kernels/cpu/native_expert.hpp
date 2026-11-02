// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized gate/up activation any native layer uses.  DeepSeek-V4's n_embd 4096 as Q8_K
/// (IQ2_XXS's vec_dot_type) is 16 x 292 = 4672, the largest; Qwen's 2560 as Q8_K is 2920 and 4096 as Q8_0 is 4352.
inline constexpr size_t kNativeActBytes = 20480;  // Delta: EXL3's F32 activation (DS4 n_embd 5120 x 4); was 5840 (5120 as Q8_K = 20 x 292)
/// Bytes of the largest quantized down activation.  DeepSeek-V4's n_ff 2048 as Q8_K (Q2_K's vec_dot_type) is
/// 8 x 292 = 2336, the largest; Qwen's 640 as Q8_0 is 20 x 34 = 680.
inline constexpr size_t kNativeHBytes = 9216;   // Delta: EXL3's F32 down activation (DS4 n_ff 2304 x 4); was 2628 (2304 as Q8_K = 9 x 292)
/// The widest native expert intermediate (n_ff) any model has.  DeepSeek-V4's 2048 is twice Qwen's 640, so the
/// pool's per-expert row buffers (ExpertPool::SplitBufMulti::ff) must be sized by THIS and not by the canonical
/// `FF` (640) that the Q2_0 pack path uses: `native_gu_rows` writes rows [0, n_ff) of the model's own width, and
/// a buffer sized for Qwen overflows silently into the neighbouring struct (found by tools/ds4/moe_replay).
inline constexpr int kNativeFFMax = 2304;   // xolotl: the MAX over every merged model (DS4 n_ff 2304 > GLM 2048) - undersizing overflows silently

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    /// The up projection's own type when it differs from the gate's (MiMo-V2.6-Flash GSQ-RCO picks a format per
    /// tensor: 16 of its 47 routed layers have gate != up).  -1 = the gate's type (every other model).  Set only by
    /// `native_fmt3`; gate and up must share the activation format (`gu_act`).
    int up_type = -1;
    size_t up_row = 0;                  ///< bytes per up row when `up_type` is set (else gu_row)
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    /// DeepSeek-V4's SwiGLU clamp (`swiglu_clamp_exp`): gate -> min(gate, lim), up -> clamp(up, -lim, lim) before
    /// silu(gate) * up.  +inf (the default, every other model) = no clamp, bit-identical to the unclamped kernels.
    float swiglu_limit = std::numeric_limits<float>::infinity();
    /// NVFP4 (ggml type 40) only: the blob ends with {s_gate, s_up, s_down, 0} at this offset, the per-expert
    /// weight_scale_2 the block format has no room for (tools/iq_pack.py). 0 = no tail. Each scale goes on its own
    /// projection's FP32 output: gate * s_gate, up * s_up, down * s_down. Do not fold s_down into up: the hidden
    /// drops to ~1e-5 and its q8 block scale becomes an FP16 subnormal (expert error 2-12% instead of 1.1%).
    size_t tail_off = 0;
    /// The pool's row ranges start and end on multiples of this (EXL3: 128, its output transform is per 128-wide band;
    /// 1 for every row-major format, whose partition is then exactly as before).
    int row_align = 1;
};

/// EXL3 (exllamav3's trellis format, mul1 codebook) as expert types: kExl3TypeBase + K2 (K2 = 2 x bits per weight),
/// far above ggml's type ids; its activations are plain F32 (kExl3ActType).  src/kernels/cpu/exl3_native.cpp.
/// A part is its four tensors back to back as the checkpoint stores them: suh | svh | mul1 | trellis (LibertAIDAI's
/// files), or with kExl3Mul1First added to the type, mul1 | suh | svh | trellis (bertholomus' 2.5 bpw: the tensors in
/// name order).  The trellis sits at 2 k + 2 n + 4 either way.
constexpr int kExl3TypeBase = 1000;
constexpr int kExl3Mul1First = 100;
constexpr int kExl3ActType = 999;
inline bool is_exl3_type(int t) {
    const int r = (t - kExl3TypeBase) % kExl3Mul1First;
    return t > kExl3TypeBase && t <= kExl3TypeBase + kExl3Mul1First + 16 && r >= 1 && r <= 16;
}
inline int exl3_type(int k2, bool mul1_first = false) { return kExl3TypeBase + (mul1_first ? kExl3Mul1First : 0) + k2; }
inline int exl3_k2_of(int t) { return (t - kExl3TypeBase) % kExl3Mul1First; }
/// The byte offset of suh inside a part (svh follows it, 2 k bytes on).
inline size_t exl3_suh_off(int t) { return t - kExl3TypeBase > kExl3Mul1First ? 4 : 0; }
/// The bytes of one EXL3 projection part (its four tensors) of a k-input, n-output matrix.
inline size_t exl3_native_part_bytes(int64_t k, int64_t n, int k2) {   // (inline: the GPU library uses it too)
    return (size_t) (2 * k + 2 * n + 4) + (size_t) (k / 16) * (size_t) (n / 16) * 16u * (size_t) k2;
}
bool exl3_native_fmt(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);
void exl3_native_quant(const float* x, int64_t n, void* dst);
void exl3_native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0, int r1);
void exl3_native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0, int r1);
constexpr int kNvfp4Type = 40;
constexpr size_t kNvfp4Tail = 16;

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// `native_fmt` with gate and up of different types (MiMo GSQ-RCO).  Equal types give exactly `native_fmt`'s layout.
/// False when the two need different activation formats (one quantized activation serves both).
bool native_fmt3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f,
                 std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
