// tools/v41/v41_sidecar_requant.cpp - requantize ONLY the DSpark routed-expert tensors of the V41-MTP-Vision.gguf
// sidecar (the 9 `mtp.N.ffn.experts.{w1,w2,w3}.weight` tensors, currently MXFP4 ~7.2 GB) to a smaller type, leaving
// every other tensor (attention, shared expert, heads, vision) byte-identical.
//
//   v41_sidecar_requant --in SRC.gguf --out DST.gguf --type Q3_K
//
// Why this is safe and worth doing: the drafter's precision CANNOT change the output quality - every draft is
// verified by the trunk - only the acceptance rate.  So we want the smallest expert format whose acceptance stays
// close to MXFP4's.  MXFP4 -> F32 (ggml's own dequantizer) -> target (ggml_quantize_chunk), so the emitted blocks
// are exactly what ggml-cuda's mul_mat_id takes on the MMVQ path (the 5-token draft pass: ne2 = 5 <= the type's
// get_mmvq_mmid_max_batch on sm_89).  Types that need an imatrix (IQ2_XXS / IQ2_XS) are rejected here.
//
// The source GGUF is loaded through ggml's gguf API (params.ctx != NULL): the whole data blob is read into a ggml
// context and one proper ggml_tensor per tensor is created there, its ->data pointing into the blob.  The output is
// a FRESH context (gguf_init_empty + gguf_set_kv + gguf_add_tensor) because the READING context's own tensor_info
// leaves `t.buffer` uninitialized - gguf_write_to_file then calls ggml_backend_tensor_get() on that garbage and
// segfaults.  Add the reader's ggml_tensors (buffer == NULL, data -> blob) instead.  Changing a tensor's type shifts
// the following tensors' data-section offsets; gguf_set_tensor_type recomputes them.
#include "ggml.h"
#include "gguf.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

static bool is_routed_expert(const char * name) {
    return strstr(name, ".ffn.experts.") != nullptr;
}

static ggml_type parse_type(const std::string & s) {
    for (int t = 1; t < (int) GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (!n || s.size() != std::strlen(n)) continue;
        bool eq = true;
        for (size_t i = 0; i < s.size() && eq; ++i)
            eq = std::tolower((unsigned char) s[i]) == std::tolower((unsigned char) n[i]);
        if (eq) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    std::string in, out, ts;
    // --uniform-imatrix: the i-quants that need importance weights (IQ2_XXS, IQ2_XS, IQ1_*) get all-equal ones.  Only
    // the drafter's acceptance can move (every draft is verified), and the drafter's precision barely moved it so far
    // (iq3_xxs / q2_K / MXFP4 all tied): the smaller type frees VRAM for the trunk's expert cache.
    bool uniform = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--in") in = nx();
        else if (a == "--out") out = nx();
        else if (a == "--type") ts = nx();
        else if (a == "--uniform-imatrix") uniform = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (in.empty() || out.empty() || ts.empty()) {
        std::fprintf(stderr, "usage: v41_sidecar_requant --in SRC.gguf --out DST.gguf --type Q3_K\n");
        return 2;
    }
    const ggml_type target = parse_type(ts);
    if (target == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "unknown type %s\n", ts.c_str());
        return 2;
    }
    const bool needs_im = ggml_quantize_requires_imatrix(target);
    if (needs_im && !uniform) {
        std::fprintf(stderr, "type %s requires an imatrix - not available (--uniform-imatrix: all-equal weights)\n", ts.c_str());
        return 3;
    }
    const ggml_type_traits * tr_dst = ggml_get_type_traits(target);
    ggml_quantize_init(target);   // the i-quant grids, once, before the worker threads

    // Read metadata + tensor infos, then load the whole data blob into a ggml context (params.ctx != NULL) so the
    // source tensor bytes are reachable via the per-tensor ggml_tensor objects it creates.
    struct ggml_context * tctx = nullptr;
    gguf_init_params ip = { /*no_alloc=*/ false, /*ctx=*/ &tctx };
    gguf_context * src = gguf_init_from_file(in.c_str(), ip);
    if (!src) { std::fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }

    const int64_t nt = gguf_get_n_tensors(src);
    gguf_context * g = gguf_init_empty();
    gguf_set_kv(g, src);   // all metadata, verbatim
    // every tensor keeps its original bytes: re-add the reader's tensors (->buffer NULL, ->data into the blob)
    for (int64_t i = 0; i < nt; ++i) {
        const char * nm = gguf_get_tensor_name(src, i);
        struct ggml_tensor * t = ggml_get_tensor(tctx, nm);
        if (!t || !t->data) { std::fprintf(stderr, "no data for %s\n", nm); return 1; }
        gguf_add_tensor(g, t);
    }

    int n_req = 0;
    size_t src_bytes = 0, dst_bytes = 0;
    std::vector<std::vector<uint8_t>> keep;   // the new blocks (outlive the write)
    keep.reserve(16);
    for (int64_t i = 0; i < nt; ++i) {
        const char * nm = gguf_get_tensor_name(src, i);
        if (!is_routed_expert(nm)) continue;
        if (gguf_get_tensor_type(src, i) != GGML_TYPE_MXFP4) { std::fprintf(stderr, "skip %s (type not MXFP4)\n", nm); continue; }
        struct ggml_tensor * t = ggml_get_tensor(tctx, nm);
        const int64_t nprow = t->ne[0];
        const int64_t nrows = ggml_nelements(t) / nprow;
        if (nprow % tr_dst->blck_size != 0) { std::fprintf(stderr, "%s: row %lld not a multiple of %d\n", nm, (long long) nprow, (int) tr_dst->blck_size); return 1; }
        const size_t rs_src = ggml_row_size(GGML_TYPE_MXFP4, nprow);
        const size_t rs_dst = ggml_row_size(target, nprow);

        std::vector<uint8_t> dst(rs_dst * (size_t) nrows);
        const int64_t chunk = std::max<int64_t>(1, (16 * 1024 * 1024) / (nprow * 4));   // ~64 MB of f32 at a time
        const ggml_to_float_t src_to_f32 = ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float;
        const std::vector<float> ones(needs_im ? (size_t) nprow : 0, 1.0f);
        // the i-quant search is slow (~minutes single-threaded for 3.4 G weights): chunks over the CPU threads
        const int64_t nchunks = (nrows + chunk - 1) / chunk;
        std::atomic<int64_t> next { 0 };
        std::atomic<bool> bad { false };
        const int nth = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int w = 0; w < nth; ++w)
            th.emplace_back([&]() {
                std::vector<float> f32((size_t) (nprow * chunk));
                for (int64_t c; (c = next.fetch_add(1)) < nchunks;) {
                    const int64_t r0 = c * chunk, nr = std::min(chunk, nrows - r0);
                    src_to_f32((const char *) t->data + (size_t) r0 * rs_src, f32.data(), nr * nprow);
                    const size_t wrote = ggml_quantize_chunk(target, f32.data(), dst.data() + (size_t) r0 * rs_dst, 0, nr, nprow,
                                                             needs_im ? ones.data() : nullptr);
                    if (wrote != (size_t) nr * rs_dst) bad = true;
                }
            });
        for (auto & x : th) x.join();
        if (bad) { std::fprintf(stderr, "%s: quantize wrote the wrong size\n", nm); return 1; }
        keep.push_back(std::move(dst));
        gguf_set_tensor_type(g, nm, target);
        gguf_set_tensor_data(g, nm, keep.back().data());
        src_bytes += rs_src * (size_t) nrows;
        dst_bytes += rs_dst * (size_t) nrows;
        ++n_req;
        std::fprintf(stderr, "  %-44s %7.1f -> %7.1f MB\n", nm, rs_src * (double) nrows / 1e6, rs_dst * (double) nrows / 1e6);
    }
    if (n_req != 9) std::fprintf(stderr, "WARNING: requantized %d tensors, expected 9\n", n_req);

    if (!gguf_write_to_file(g, out.c_str(), false)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::fprintf(stderr, "requant %s -> %s: %d expert tensors, %.3f -> %.3f GB\n",
                 ggml_type_name(GGML_TYPE_MXFP4), ggml_type_name(target), n_req, src_bytes / 1e9, dst_bytes / 1e9);
    gguf_free(g);
    gguf_free(src);
    if (tctx) ggml_free(tctx);
    return 0;
}
