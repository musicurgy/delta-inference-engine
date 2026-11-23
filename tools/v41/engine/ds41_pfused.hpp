// tools/v41/engine/ds41_pfused.hpp - Strata's fused int8 prompt-expert kernels (src/prefill/moe_fused_iq.cu
// upstream at baf1636, the 512-thread native_kernel) ported to DeepSeek-V4.1-Flash's routed experts, opt-in
// STRATA_PROMPT_FUSED=1.
//
// What the port changes against upstream:
//   - the shape is DeepSeek-V4.1-Flash's (n_embd 5120, n_ff 2304: the GGUF's embedding_length and
//     expert_feed_forward_length), compiled in (kEmbd / kFF below); a model of any other shape is not covered and keeps
//     MMQ.  (The first port had DeepSeek-V4-Flash's 4096 / 2048 here, so no class of the real model engaged.)  The
//     kernels only need both K whole 256-value super-blocks and n_embd, 2 n_ff whole 256-row work items;
//   - the weights are read where gpu_run_chunk already has them: one expert blob [gate rows | up rows | down rows]
//     (the GGUF blocks, NativeExpertLayout's offsets), `stride` bytes apart (MMQ's slot stride), so no gather;
//   - both K (5120 and 2304) are whole 256-value super-blocks, so the down product reads the i-quants the way the
//     gate/up product does (upstream's downs were only 32/64-value block formats: Q2_0, IQ4_NL);
//   - IQ1_S is added (upstream has no IQ1_S): w = d (2 s + 1) (g + delta), g in {-1, 0, 1}; the kernel multiplies
//     the codes g + 1 (0..2, as llama.cpp's MMQ does) and adds d (2 s + 1) (delta - 1) times the activation codes'
//     sum per 32 values (the "minimum" term upstream uses for its K-quant formats);
//   - DeepSeek-V4's SwiGLU clamp in the gate/up epilogue (gate <= lim, |up| <= lim, as mmq::swiglu_clamp);
//   - the rows of a run are the caller's sorted entries: the activations are rounded to int8 for those rows only
//     (gathered through `tok`, as mmq::quantize does), and the down product writes row r into dst[r] of the
//     per-entry output (the rows MMQ writes through its ids);
//   - the GPU grouping (fused::group), the weight-staging, NT = 256, warp-specialized, F16, Turing and HIP variants
//     are not ported: the caller knows its rows on the host and builds the tile list itself.
//
// Numerics: not MMQ's bits.  Both round the activations and H to int8 per 32 values with a max-abs scale, but the
// fused kernels keep the scales in FP32 (MMQ stores some as FP16), sum in another order, and never write the FP32
// gate/up rows.  The difference is of the order of MMQ's own distance from FP32 (test_ds41_pfused prints both).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ds41pf {

constexpr int kEmbd = 5120;   ///< DeepSeek-V4.1-Flash: hidden width (K of gate/up, rows of down)
constexpr int kFF = 2304;     ///< DeepSeek-V4.1-Flash: routed expert width (K of down)

/// This build has the kernels (CUDA).
bool built();
/// The current device can run them (sm_80+: cp.async, mma.sync m16n8k32 s8) and their set-up succeeded there.
bool available();
/// STRATA_PROMPT_FUSED=1 (read once) and available().
bool enabled();
/// The kernels cover this expert: gate == up type, both gate/up and down in IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S /
/// IQ3_XXS (ggml ids), n_embd == kEmbd, n_ff == kFF, even offsets.
bool covers(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off);
/// Why `covers` says no (the first test that fails, for the log), nullptr when it covers the expert.
const char* uncovered(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, size_t up_off,
                      size_t down_off);

/// Bytes of `rows` activation rows of `cols` values (a multiple of 64) in the kernels' int8 form (80 bytes per 64).
size_t act_bytes(int64_t rows, int64_t cols);
/// Bytes of the tile list for a run of `rows` entries over `n_exp` experts (worst case; int2 per tile).
size_t tiles_bytes(int64_t rows, int n_exp);

/// xa row r = int8 of x row tok[r] (tok null: row r), `cols` floats a row, `rows` rows.
void quantize_rows(const float* x, const int32_t* tok, int64_t rows, int64_t cols, void* xa, void* stream);

/// One run: experts q = 0..n_exp-1 at base + q * stride; expert q owns entries [off[q], off[q+1]) of the run
/// (off on the device, n_exp + 1 ints, off[0] = 0).  The caller builds the host tile list with build_tiles (it
/// depends on the tile height `tr` that pick_tile returns) and uploads it to `tiles`.
struct Run {
    const uint8_t* base = nullptr;
    size_t stride = 0;
    int n_exp = 0;
    int gu_type = -1, d_type = -1;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0;
    float swiglu_limit = 0.0f;              ///< +inf = no clamp
    const int32_t* off = nullptr;           ///< device, n_exp + 1
    const int32_t* tiles = nullptr;         ///< device, 2 ints per tile: {expert, first row}
    int n_tiles = 0;
    int tr = 64;                            ///< routed rows per tile (pick_tile)
};
/// The routed rows per work item for a run of `rows` entries over `n_exp` experts: 64 or 128 (upstream's band rule;
/// STRATA_PROMPT_FUSED_TILE=64|128 forces one).  The output bits do not depend on it.
int pick_tile(int64_t rows, int n_exp);
/// Host tile list (2 ints per tile) of a run whose expert q has cnt[q] rows starting at off[q]: returns the tiles.
int build_tiles(const int32_t* off, int n_exp, int tr, int32_t* out);
/// gate/up + clamped SwiGLU + H to int8 (into `ha`, act_bytes(rows, kFF)), then down: entry r's kEmbd floats to
/// out row dst[r] (`out` has kEmbd floats a row).  `xa` from quantize_rows over the run's entries.
void experts(const Run& r, const void* xa, void* ha, const int32_t* dst, float* out, void* stream);

}  // namespace ds41pf
