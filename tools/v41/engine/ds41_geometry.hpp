// tools/v41/engine/ds41_geometry.hpp - DeepSeek-V4.1-Flash (`deepseek41`) geometry, tensor map and expert tiers.
//
// The GGUF is the one tools/v41/v41_gguf.py writes (DwarfStar's dense names, per-expert routed tiers).  Every number
// comes from the file's `deepseek41.*` keys (or the `deepseek41.config` JSON for the Engram shape the flat keys lack);
// a missing key is an error, never a default.  `ds41_check_tensors` resolves the full tensor map: every name in the
// file must be predicted and every predicted name present, with the conditional sets exactly as model.py builds them:
// compressor tensors on kv source layers (gate only where ratio > 1), indexer query/proj on index source layers,
// indexer key on kv sources, Engram on its layers, routed experts either one stacked tensor per role or one per type
// + the I32 type maps.
//
// Header-only; reads the header region only (GgufModel mmaps it), no tensor data except the two I32 type maps.
//
// Day-0 tolerance (R-1 / R-18): the architecture comes from general.architecture against a whitelist (deepseek41 +
// DS41_ACCEPT_ARCH) and the keys are read under its prefix; every guard names the value it saw; a Ds41Issues sink
// (test_ds41_geometry --report) collects every problem in one pass instead of stopping at the first; type ids this
// engine has no layout for (ik_llama's 144+) and mainline ggml's plain F8_E4M3 under id 43 are refused by name.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ds41_sys.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ds41_posix.hpp"

namespace strata::v41 {

namespace ty {
constexpr uint32_t F32 = 0, F16 = 1, Q8_0 = 8, Q3_K = 11, I8 = 24, I32 = 26, Q2_0 = 42, PTQ1_0 = 143;
}

/// One layer's routed experts: the type of each expert's gate+up and of its down, and where each expert's rows are
/// (tensor name + slot inside it).  Uniform roles have a single tensor (slot == expert id).
struct Ds41ExpertMap {
    std::vector<int> gu_type, d_type;              ///< [n_expert] ggml type ids
    std::vector<int> gu_slot, d_slot;              ///< [n_expert] index inside the type's stacked tensor
    std::map<int, std::string> gate_t, up_t, down_t;   ///< type -> tensor name
};

struct Ds41Geometry {
    int64_t n_embd = 0, n_layer = 0, n_head = 0, d_head = 0, d_rope = 0, r_q = 0, o_lora = 0, o_groups = 0;
    int64_t n_swa = 0, n_expert = 0, n_used = 0, n_ff = 0, n_shared = 0;
    int64_t n_used_model = 0;   ///< the model's own experts per token (n_used before STRATA_EXPERTS_USED; 0 = n_used)
    double weights_scale = 0;
    bool norm_topk = false;
    int64_t hc = 0, hc_iters = 0;
    double hc_eps = 0, rms_eps = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0;
    int64_t cand_layer = -1, cand_blocks = 0, cand_bsize = 0;
    double rope_base = 0, compress_rope_base = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0, orig_ctx = 0;
    int64_t vocab = 0, ctx_len = 0;
    double swiglu_limit = 0;
    std::vector<int64_t> ratio;                      ///< [n_layer] 0 / 2 / 1
    std::vector<int64_t> kv_sources, index_sources;
    // Engram
    std::vector<int64_t> engram_layers, engram_rows; ///< layer ids, table rows per layer
    int64_t engram_heads = 0, engram_dim = 0, engram_ngram = 0, engram_pad = 0, engram_cvocab = 0;
    std::vector<uint32_t> engram_token_map;          ///< [vocab] compressed id
    std::vector<uint64_t> engram_primes;             ///< [n_engram_layers * (ngram-1) * heads]
    std::vector<int64_t> engram_mult;                ///< [n_engram_layers * ngram]
    int64_t bos = -1, eos = -1;
    std::vector<Ds41ExpertMap> experts;              ///< [n_layer]

    bool is_kv_source(int64_t l) const { return contains(kv_sources, l); }
    bool is_index_source(int64_t l) const { return contains(index_sources, l); }
    int64_t engram_slot(int64_t l) const {
        for (size_t i = 0; i < engram_layers.size(); ++i) if (engram_layers[i] == l) return (int64_t) i;
        return -1;
    }
    /// The kv source whose compressed KV layer l reads (model.py shared_attn.compress_kv: the latest source <= l), or -1.
    int64_t kv_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : kv_sources) if (k <= l) s = k;
        return s;
    }
    /// The index source whose top-k layer l uses (the latest index source <= l), or -1.
    int64_t idx_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : index_sources) if (k <= l) s = k;
        return s;
    }
    int64_t hc_mix() const { return (2 + hc) * hc; }

private:
    static bool contains(const std::vector<int64_t> & v, int64_t x) {
        for (int64_t y : v) if (y == x) return true;
        return false;
    }
};

// ------------------------------------------------------------------ full metadata arrays
// strata::GgufFile keeps 64 items of an array (a sample); the Engram token map has 129,280.  This walks the KV
// section of the file itself and returns one key's whole array (integers only), or an error.
namespace detail {
struct Rd {
    int fd = -1;
    uint64_t pos = 0;
    std::vector<uint8_t> buf;
    uint64_t buf_at = 0;
    bool fill(uint64_t at, size_t n) {
        if (at >= buf_at && at + n <= buf_at + buf.size()) return true;
        const size_t want = std::max<size_t>(n, 1 << 20);
        buf.resize(want);
        const ssize_t r = ds41_sys::pread(fd, buf.data(), want, (int64_t) at);
        if (r < (ssize_t) n) return false;
        buf.resize((size_t) r);
        buf_at = at;
        return true;
    }
    bool get(void * dst, size_t n) {
        if (!fill(pos, n)) return false;
        std::memcpy(dst, buf.data() + (pos - buf_at), n);
        pos += n;
        return true;
    }
    template <class T> bool get(T & v) { return get(&v, sizeof(T)); }
};
inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;   // U8 I8 BOOL
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}
}  // namespace detail

inline std::string ds41_read_int_array(const std::string & path, const std::string & key, std::vector<int64_t> & out) {
    detail::Rd r;
    r.fd = ds41_sys::open(path.c_str(), O_RDONLY);
    if (r.fd < 0) return "cannot open " + path;
    struct Closer { int fd; ~Closer() { ds41_sys::close(fd); } } closer { r.fd };
    uint32_t magic = 0, version = 0;
    uint64_t n_t = 0, n_kv = 0;
    if (!r.get(magic) || magic != 0x46554747u || !r.get(version) || !r.get(n_t) || !r.get(n_kv)) return "bad GGUF header";
    auto skip_str = [&]() { uint64_t n; if (!r.get(n)) return false; r.pos += n; return true; };
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t kl = 0;
        if (!r.get(kl) || kl > 4096) return "bad key";
        std::string k(kl, '\0');
        if (!r.get(k.data(), kl)) return "bad key";
        uint32_t t = 0;
        if (!r.get(t)) return "bad type";
        if (t == 8) { if (!skip_str()) return "bad string"; continue; }
        if (t != 9) { const size_t s = detail::scalar_size(t); if (!s) return "bad scalar type"; r.pos += s; continue; }
        uint32_t et = 0;
        uint64_t cnt = 0;
        if (!r.get(et) || !r.get(cnt)) return "bad array";
        if (k != key) {
            if (et == 8) { for (uint64_t j = 0; j < cnt; ++j) if (!skip_str()) return "bad string array"; }
            else { const size_t s = detail::scalar_size(et); if (!s) return "bad array type"; r.pos += s * cnt; }
            continue;
        }
        const size_t s = detail::scalar_size(et);
        if (!s || et == 7 || et == 6 || et == 12) return key + ": not an integer array";
        out.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) {
            uint8_t b[8] = {};
            if (!r.get(b, s)) return key + ": truncated";
            int64_t v = 0;
            switch (et) {
                case 0: v = b[0]; break;
                case 1: v = (int8_t) b[0]; break;
                case 2: { uint16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 3: { int16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 4: { uint32_t x; std::memcpy(&x, b, 4); v = x; break; }
                case 5: { int32_t x; std::memcpy(&x, b, 4); v = x; break; }
                default: std::memcpy(&v, b, 8); break;
            }
            out[j] = v;
        }
        return {};
    }
    return "missing " + key;
}

// ------------------------------------------------------------------ architecture, types and the report sink
/// The architectures the deepseek41 code path reads: `deepseek41`, plus any listed in DS41_ACCEPT_ARCH (comma
/// separated, e.g. DS41_ACCEPT_ARCH=deepseek42 to try a same-family checkpoint on day 0).  An accepted architecture
/// only gets past the name check: its keys are read under its own prefix ("deepseek42.") and every key, shape and
/// tensor check below still applies.
inline std::vector<std::string> ds41_accepted_archs() {
    std::vector<std::string> v{ "deepseek41" };
    if (const char * env = std::getenv("DS41_ACCEPT_ARCH")) {
        const std::string s(env);
        size_t a = 0;
        while (a <= s.size()) {
            size_t b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            std::string x = s.substr(a, b - a);
            while (!x.empty() && (x.back() == ' ' || x.back() == '\t')) x.pop_back();
            while (!x.empty() && (x.front() == ' ' || x.front() == '\t')) x.erase(0, 1);
            if (!x.empty() && std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
            a = b + 1;
        }
    }
    return v;
}
inline bool ds41_arch_accepted(const std::string & arch) {
    const std::vector<std::string> v = ds41_accepted_archs();
    return !arch.empty() && std::find(v.begin(), v.end(), arch) != v.end();
}
/// The file's general.architecture ("" when it has none).
inline std::string ds41_arch(const GgufFile & g) {
    const MetaValue * a = g.get("general.architecture");
    return a ? a->s : std::string();
}
/// The prefix of the model's own keys: "<general.architecture>." ("deepseek41." for every V4.1 file; also when the
/// file names none, so a report on such a file still lists the keys it lacks).
inline std::string ds41_prefix(const GgufFile & g) {
    const std::string a = ds41_arch(g);
    return (a.empty() ? std::string("deepseek41") : a) + ".";
}

namespace ty {
/// The vendored ggml's GGML_TYPE_COUNT (Q2_0 = 42 is its last type).  A tensor the dense loader hands to ggml needs a
/// type below it: mainline ggml's F8_E4M3 (43) and our PTQ1_0 (143) are past it.  (ds41_dense.cpp's load_weights also
/// refuses any type >= the GGML_TYPE_COUNT it was built with, so a newer ggml cannot slip past this constant.)
constexpr uint32_t kGgmlTypes = 43;
/// Type 43 is two formats.  Unsloth's DeepSeek-V4.1 Engram tables ("MXFP8", the one block_geometry knows: one E8M0
/// scale + 32 E4M3 bytes per 32 values, 33 bytes) and mainline ggml's GGML_TYPE_F8_E4M3 (ggml 0.29 / llama.cpp
/// v0.9.0, 2026-11: one E4M3 byte a value, block size 1, no scale in the block).  The header does not say which.
constexpr uint32_t FP8_43 = 43;
}

/// Whether this engine knows GGUF type id `t` (has its block layout): ggml's types up to Q2_0 (42) that have one here,
/// type 43 (above) and PTQ1_0 (143).  Not known: Q8_K / F64 / TQ1_0 / TQ2_0 / Q1_0 and every id past 143 - ik_llama's
/// IQK and trellis types sit at 144+ (an Agention file stores its trellis rows as ik's 144-149), so such a file is
/// refused by name instead of read with a wrong block size.
inline bool ds41_type_known(uint32_t t) {
    int be = 0, bb = 0;
    return block_geometry(t, be, bb);
}
/// "Q8_0 (8)", "MXFP8 / F8_E4M3 (43)", "unknown (150)" - for messages
inline std::string ds41_type_label(uint32_t t) {
    if (t == ty::FP8_43) return "MXFP8 / F8_E4M3 (43)";
    const char * n = ggml_type_name(t);
    return std::string(std::strcmp(n, "?") == 0 ? "unknown" : n) + " (" + std::to_string(t) + ")";
}

enum class Ds41Fp8Layout { Mxfp8, F8E4M3, Unknown };
/// Which format a type-43 tensor `t` of shard `s` holds, from the room it has in the file (up to the next tensor's
/// offset, or the end of the data): 33 bytes per 32 values -> Mxfp8; at least one byte a value but less -> F8E4M3;
/// less than one byte a value -> Unknown (truncated).  A tensor so small that the two sizes differ by less than the
/// alignment padding reads as Mxfp8 (what the engine has always read type 43 as).
inline Ds41Fp8Layout ds41_type43_layout(const GgufModel & m, const TensorInfo & t, size_t s) {
    const GgufFile & g = m.shard(s);
    const uint64_t payload = g.file_size() - g.data_start();
    if (t.offset > payload) return Ds41Fp8Layout::Unknown;
    uint64_t end = payload;
    for (const auto & o : g.tensors())
        if (o.offset > t.offset && o.offset < end) end = o.offset;
    const uint64_t room = end - t.offset, n = t.elements();
    if (!t.shape.empty() && t.shape[0] % 32 == 0 && room >= n / 32 * 33) return Ds41Fp8Layout::Mxfp8;
    if (room >= n) return Ds41Fp8Layout::F8E4M3;
    return Ds41Fp8Layout::Unknown;
}

/// What the checks found.  nullptr (the default, every loader): the first problem ends the read and is returned, as
/// always.  A Ds41Issues (test_ds41_geometry --report): every problem is appended to `items` and the read goes on, so
/// one pass lists every missing key, bad value, missing / unexpected tensor and expert type; the functions then return
/// "" and the caller reads `items`.  `seen` collects every metadata key the readers looked up (the report lists the
/// model's keys the engine never reads).
struct Ds41Issues {
    std::vector<std::string> items;
    std::set<std::string> seen;
};

// ------------------------------------------------------------------ key readers (missing key == error)
namespace detail {
/// e == "": nothing.  Otherwise the loader stops (true: return e); a report records e and goes on (false).
inline bool halt(Ds41Issues * iss, const std::string & e) {
    if (e.empty()) return false;
    if (!iss) return true;
    iss->items.push_back(e);
    return false;
}
inline const MetaValue * get(const GgufFile & g, const std::string & k, Ds41Issues * iss) {
    if (iss) iss->seen.insert(k);
    return g.get(k);
}
/// a metadata value for a message: 4, 0.5, "text", [array of 43], (absent)
inline std::string show(const MetaValue * v) {
    if (!v) return "(absent)";
    if (v->type == MetaType::STRING) return "\"" + v->s + "\"";
    if (v->type == MetaType::ARRAY) return "[array of " + std::to_string(v->count) + "]";
    char b[64];
    std::snprintf(b, sizeof b, "%g", v->num());
    return b;
}
inline std::string num(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%g", x);
    return b;
}
inline std::string u64(const GgufFile & g, const std::string & k, int64_t & out, Ds41Issues * iss = nullptr) {
    const MetaValue * v = get(g, k, iss);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = (int64_t) v->num();
    return {};
}
inline std::string f64(const GgufFile & g, const std::string & k, double & out, Ds41Issues * iss = nullptr) {
    const MetaValue * v = get(g, k, iss);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = v->num();
    return {};
}
inline std::string arr(const GgufFile & g, const std::string & k, std::vector<int64_t> & out, Ds41Issues * iss = nullptr) {
    const MetaValue * v = get(g, k, iss);
    if (!v) return "missing " + k;
    if (v->type != MetaType::ARRAY) return k + " is not an array";
    if (v->count > v->items.size()) return k + ": longer than the reader's sample (use ds41_read_int_array)";
    out.clear();
    for (const auto & e : v->items) out.push_back((int64_t) e.num());
    return {};
}
/// `"key":<number>` out of the deepseek41.config JSON (flat search; the keys are unique in DeepSeek's config)
inline std::string json_num(const std::string & js, const std::string & key, double & out) {
    const std::string pat = "\"" + key + "\":";
    const size_t p = js.find(pat);
    if (p == std::string::npos) return "deepseek41.config has no " + key;
    out = std::strtod(js.c_str() + p + pat.size(), nullptr);
    return {};
}
}  // namespace detail

/// A `deepseek41` GGUF in the llama.cpp convention (llama.cpp PR #30817's converter; ISTA-DASLab's GSQ-RCO GGUFs):
/// llama.cpp key names, the Engram hash arrays as I32/I64 tensors, its tables in a separate file.  Ours has
/// `deepseek41.config`.  (Keys under the file's own architecture prefix: ds41_prefix.)
inline bool ds41_is_llamacpp_gguf(const GgufModel & model) {
    const std::string P = ds41_prefix(model.meta());
    return model.meta().get(P + "config") == nullptr && model.meta().get(P + "embedding_length") != nullptr;
}

/// llama.cpp mainline's convention (src/models/deepseek41.cpp; Unsloth's GGUFs) - a llama.cpp-convention file whose
/// Engram is described as the PLE n-gram hash: the hash constants in the metadata (ple.layers, ple.layer_multipliers,
/// ple.head_vocab_sizes = the primes, ple.head_offsets), both Engram tables in ONE IQ4_NL tensor per_layer_token_embd
/// [256, rows(layer 1) + rows(layer 14)], and the two Engram RMSNorm weights stored as their product (engram_gate).
inline bool ds41_is_mainline_gguf(const GgufModel & model) {
    return ds41_is_llamacpp_gguf(model) && model.meta().get(ds41_prefix(model.meta()) + "ple.layers") != nullptr;
}

namespace detail {
/// an I32 / I64 tensor of the file, whole (the llama.cpp convention's Engram arrays)
inline std::string int_tensor(const GgufModel & m, const std::string & n, std::vector<int64_t> & out) {
    size_t si = 0;
    const TensorInfo * t = m.find(n, &si);
    if (!t) return "missing tensor " + n;
    if (t->type != 26 && t->type != 27) return n + ": not I32/I64";
    uint64_t cnt = 1;
    for (uint64_t d : t->shape) cnt *= d;
    if (!m.in_bounds(*t, si)) return n + ": out of bounds";
    const uint8_t * p = m.shard(si).tensor_data(*t);
    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) {
        if (t->type == 26) { int32_t x; std::memcpy(&x, p + 4 * i, 4); out[i] = x; }
        else std::memcpy(&out[i], p + 8 * i, 8);
    }
    return {};
}
}  // namespace detail

/// ds41_read_geometry for the llama.cpp convention: the same Ds41Geometry from its key names (every key still
/// required).  The routing, attention and Engram hash constants are DeepSeek's either way - only where they live
/// differs (checked on ISTA's IQ2_XXS: token map, primes and multipliers byte-equal to ours).
inline std::string ds41_read_geometry_llamacpp(const GgufModel & model, Ds41Geometry & G, Ds41Issues * iss = nullptr) {
    const GgufFile & g = model.meta();
    std::string e;
    const std::string P = ds41_prefix(g);
    const bool ml = ds41_is_mainline_gguf(model);   // mainline's key names where they differ from #30817's
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "embedding_length", &G.n_embd }, { "block_count", &G.n_layer }, { "attention.head_count", &G.n_head },
             { "attention.key_length", &G.d_head }, { "rope.dimension_count", &G.d_rope },
             { "attention.q_lora_rank", &G.r_q }, { "attention.output_lora_rank", &G.o_lora },
             { "attention.output_group_count", &G.o_groups }, { "attention.sliding_window", &G.n_swa },
             { "expert_count", &G.n_expert }, { "expert_used_count", &G.n_used },
             { "expert_feed_forward_length", &G.n_ff }, { "expert_shared_count", &G.n_shared },
             { "hyper_connection.count", &G.hc }, { "hyper_connection.sinkhorn_iterations", &G.hc_iters },
             { "attention.indexer.head_count", &G.idx_heads }, { "attention.indexer.key_length", &G.idx_dim },
             { "attention.indexer.top_k", &G.idx_topk },
             { ml ? "attention.indexer.candidate_source_layer" : "attention.candidate_source_layer_id", &G.cand_layer },
             { ml ? "attention.indexer.candidate_top_k_blocks" : "attention.candidate_topk_blocks", &G.cand_blocks },
             { ml ? "attention.indexer.candidate_block_size" : "attention.candidate_block_size", &G.cand_bsize },
             { "context_length", &G.ctx_len }, { ml ? "ple.heads_per_ngram" : "engram_n_heads", &G.engram_heads },
             { ml ? "embedding_length_per_layer_input" : "engram_head_dim", &G.engram_dim },
             { ml ? "ple.ngram_size" : "engram_max_ngram_size", &G.engram_ngram },
             { ml ? "engram.pad_token_id" : "engram_pad_token_id", &G.engram_pad },
             { ml ? "engram.compressed_vocab_size" : "engram_compressed_vocab_size", &G.engram_cvocab } })
        if (!(ml && u.v == &G.engram_cvocab && !detail::get(g, P + u.k, iss)) && detail::halt(iss, e = detail::u64(g, P + u.k, *u.v, iss))) return e;   // (ISTA's GSQ-RCO files leave that key out: the token map gives it)
    struct F { const char * k; double * v; };
    for (const F & f : std::initializer_list<F>{
             { "expert_weights_scale", &G.weights_scale }, { "hyper_connection.epsilon", &G.hc_eps },
             { "attention.layer_norm_rms_epsilon", &G.rms_eps }, { "rope.freq_base", &G.rope_base },
             { "attention.compress_rope_freq_base", &G.compress_rope_base }, { "rope.scaling.factor", &G.rope_factor },
             { "rope.scaling.yarn_beta_fast", &G.beta_fast }, { "rope.scaling.yarn_beta_slow", &G.beta_slow },
             { "rope.scaling.original_context_length", &G.orig_ctx } })
        if (detail::halt(iss, e = detail::f64(g, P + f.k, *f.v, iss))) return e;
    {
        const MetaValue * v = detail::get(g, P + "expert_weights_norm", iss);
        if (!v) { if (detail::halt(iss, e = "missing " + P + "expert_weights_norm")) return e; }
        else G.norm_topk = v->u != 0;
        if (ml) {   // llama.cpp's enum: LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS = 4
            const MetaValue * gf = detail::get(g, P + "expert_gating_func", iss);
            if ((!gf || !gf->is_num() || (int64_t) gf->num() != 4) &&
                detail::halt(iss, e = P + "expert_gating_func is " + detail::show(gf) + ", not sqrt-softplus (4)")) return e;
        } else {
            const MetaValue * sf = detail::get(g, P + "expert_scoring_func", iss);
            if ((!sf || sf->s != "sqrtsoftplus") &&
                detail::halt(iss, e = P + "expert_scoring_func is " + detail::show(sf) + ", not \"sqrtsoftplus\"")) return e;
        }
        // one SwiGLU clamp for every layer (DeepSeek's swiglu_limit); a per-layer array of equal values here
        const MetaValue * sl = detail::get(g, P + "swiglu_clamp_exp", iss);
        if (!sl || sl->type != MetaType::ARRAY || sl->items.empty()) {
            if (detail::halt(iss, e = "missing " + P + "swiglu_clamp_exp")) return e;
        } else {
            G.swiglu_limit = sl->items[0].num();
            for (size_t i = 0; i < sl->items.size(); ++i)
                if (sl->items[i].num() != G.swiglu_limit) {
                    if (detail::halt(iss, e = P + "swiglu_clamp_exp differs between layers: [" + std::to_string(i) + "] is " +
                                              detail::num(sl->items[i].num()) + ", [0] is " + detail::num(G.swiglu_limit) +
                                              " (one limit supported)")) return e;
                    break;   // (one line in a report)
                }
        }
    }
    if (detail::halt(iss, e = detail::arr(g, P + "attention.compress_ratios", G.ratio, iss))) return e;
    if (G.n_layer >= 0 && (int64_t) G.ratio.size() > G.n_layer) G.ratio.resize((size_t) G.n_layer);   // (+ the 3 MTP blocks' entries)
    if (detail::halt(iss, e = detail::arr(g, P + (ml ? "attention.kv_source_layers" : "kv_source_layer_ids"), G.kv_sources, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + (ml ? "attention.indexer.source_layers" : "index_source_layer_ids"), G.index_sources, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + (ml ? "ple.layers" : "engram.layer_ids"), G.engram_layers, iss))) return e;
    const TensorInfo * te = model.find("token_embd.weight");
    if (!te || te->shape.size() != 2) { if (detail::halt(iss, e = "missing token_embd.weight")) return e; }
    else G.vocab = (int64_t) te->shape[1];
    std::vector<int64_t> tm, pr;
    const std::string & path = g.path();
    auto whole = [&](const std::string & k, std::vector<int64_t> & out) {   // a metadata array, whole
        if (iss) iss->seen.insert(k);
        return ds41_read_int_array(path, k, out);
    };
    if (ml) {
        // the token map, primes and multipliers from the metadata (read whole); a layer's rows = the sum of its primes,
        // and its head offsets must run on from the previous layer's (one table, layer after layer: the Engram reader
        // takes layer e's rows from the sum of the earlier layers' rows)
        std::vector<int64_t> offs;
        if (detail::halt(iss, e = whole(P + "engram.token_map", tm))) return e;
        if (detail::halt(iss, e = whole(P + "ple.head_vocab_sizes", pr))) return e;
        if (detail::halt(iss, e = whole(P + "ple.layer_multipliers", G.engram_mult))) return e;
        if (detail::halt(iss, e = whole(P + "ple.head_offsets", offs))) return e;
        if ((int64_t) tm.size() != G.vocab &&
            detail::halt(iss, e = "engram.token_map has " + std::to_string(tm.size()) + " entries, the vocabulary (token_embd) " +
                                  std::to_string(G.vocab))) return e;
        if (!G.engram_cvocab && !tm.empty()) G.engram_cvocab = *std::max_element(tm.begin(), tm.end()) + 1;
        if (offs.size() != pr.size() &&
            detail::halt(iss, e = "ple.head_offsets (" + std::to_string(offs.size()) + ") / ple.head_vocab_sizes (" +
                                  std::to_string(pr.size()) + ") differ in length")) return e;
        const int64_t nel = (int64_t) G.engram_layers.size(), nc = (G.engram_ngram - 1) * G.engram_heads;
        if ((int64_t) pr.size() != nel * nc &&
            detail::halt(iss, e = "ple.head_vocab_sizes has " + std::to_string(pr.size()) + " entries, its shape needs " +
                                  std::to_string(nel * nc) + " (layers x (ngram - 1) x heads)")) return e;
        int64_t acc = 0;
        G.engram_rows.assign((size_t) (nel > 0 ? nel : 0), 0);
        const bool shaped = nc > 0 && (int64_t) pr.size() == nel * nc && offs.size() == pr.size();
        for (int64_t i = 0; shaped && i < nel * nc; ++i) {
            if (offs[(size_t) i] != acc) {
                if (detail::halt(iss, e = "ple.head_offsets[" + std::to_string(i) + "] is " + std::to_string(offs[(size_t) i]) +
                                          ", not the running sum of the primes " + std::to_string(acc) + " (unsupported layout)")) return e;
                break;
            }
            acc += pr[(size_t) i];
            G.engram_rows[(size_t) (i / nc)] += pr[(size_t) i];
        }
        G.engram_token_map.assign(tm.begin(), tm.end());
        G.engram_primes.assign(pr.begin(), pr.end());
        detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
        detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
        return {};
    }
    if (detail::halt(iss, e = detail::arr(g, P + "engram.num_embeddings", G.engram_rows, iss))) return e;
    if (detail::halt(iss, e = detail::int_tensor(model, "engram_token_map.weight", tm))) return e;
    if (detail::halt(iss, e = detail::int_tensor(model, "engram_primes.weight", pr))) return e;
    if (detail::halt(iss, e = detail::int_tensor(model, "engram_multipliers.weight", G.engram_mult))) return e;
    if ((int64_t) tm.size() != G.vocab &&
        detail::halt(iss, e = "engram_token_map has " + std::to_string(tm.size()) + " entries, the vocabulary (token_embd) " +
                              std::to_string(G.vocab))) return e;
    G.engram_token_map.assign(tm.begin(), tm.end());
    G.engram_primes.assign(pr.begin(), pr.end());
    detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
    detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
    return {};
}

/// the checks both conventions share (after the keys are read).  Each guard names the value it found.
inline std::string ds41_check_geometry(Ds41Geometry & G, Ds41Issues * iss = nullptr) {
    std::string e;
    const bool sized = (int64_t) G.ratio.size() == G.n_layer;
    if (!sized && detail::halt(iss, e = "compress_ratios has " + std::to_string(G.ratio.size()) + " entries, block_count is " +
                                        std::to_string(G.n_layer))) return e;
    for (size_t l = 0; l < G.ratio.size(); ++l) {
        const int64_t r = G.ratio[l];
        if (r != 0 && r != 1 && r != 2 &&
            detail::halt(iss, e = "layer " + std::to_string(l) + ": compress ratio " + std::to_string(r) + " (V4.1 has 0/1/2)")) return e;
    }
    bool sources_ok = true;
    for (int k = 0; k < 2; ++k)
        for (int64_t l : k ? G.index_sources : G.kv_sources) {
            const bool range = l >= 0 && l < G.n_layer && (size_t) l < G.ratio.size();
            if (range && G.ratio[(size_t) l] != 0) continue;
            sources_ok = false;
            if (detail::halt(iss, e = std::string(k ? "index" : "kv") + " source layer " + std::to_string(l) +
                                      (range ? " has compress ratio 0" : " is outside block_count " + std::to_string(G.n_layer)))) return e;
        }
    for (int64_t l = 0; sized && sources_ok && l < G.n_layer; ++l) {
        if (!G.ratio[(size_t) l]) continue;
        const int64_t ks = G.kv_src(l), is = G.idx_src(l);
        if ((ks < 0 || is < 0 || G.ratio[(size_t) ks] != G.ratio[(size_t) l]) &&
            detail::halt(iss, e = "layer " + std::to_string(l) + " has no kv/index source of its ratio (ratio " +
                                  std::to_string(G.ratio[(size_t) l]) + ", kv source " + std::to_string(ks) + ", index source " +
                                  std::to_string(is) + ")")) return e;
    }
    if (G.n_shared != 1 &&
        detail::halt(iss, e = "unsupported shape: " + std::to_string(G.n_shared) + " shared experts (one supported)")) return e;
    if (G.hc < 1 && detail::halt(iss, e = "unsupported shape: hyper-connection count " + std::to_string(G.hc) + " (>= 1 needed)")) return e;
    if (G.d_head <= G.d_rope &&
        detail::halt(iss, e = "unsupported shape: head dim " + std::to_string(G.d_head) + " <= rope dim " + std::to_string(G.d_rope))) return e;
    if ((G.o_groups <= 0 || (G.n_head * G.d_head) % G.o_groups) &&
        detail::halt(iss, e = "unsupported shape: heads x head dim " + std::to_string(G.n_head * G.d_head) +
                              " is not a multiple of the output groups " + std::to_string(G.o_groups))) return e;
    const int64_t nel = (int64_t) G.engram_layers.size(), nb = (G.engram_ngram - 1) * G.engram_heads;
    const bool eshape = (int64_t) G.engram_primes.size() == nel * nb && (int64_t) G.engram_mult.size() == nel * G.engram_ngram &&
                        (int64_t) G.engram_rows.size() == nel;
    if (!eshape &&
        detail::halt(iss, e = "engram arrays do not match its shape: " + std::to_string(nel) + " layers, ngram " +
                              std::to_string(G.engram_ngram) + ", " + std::to_string(G.engram_heads) + " heads; primes " +
                              std::to_string(G.engram_primes.size()) + ", multipliers " + std::to_string(G.engram_mult.size()) +
                              ", rows " + std::to_string(G.engram_rows.size()))) return e;
    for (int64_t i = 0; eshape && i < nel; ++i) {   // the bucket ranges tile the table exactly (v41_gguf_meta checks the same)
        uint64_t s = 0;
        for (int64_t j = 0; j < nb; ++j) s += G.engram_primes[(size_t) (i * nb + j)];
        if ((int64_t) s != G.engram_rows[(size_t) i] &&
            detail::halt(iss, e = "engram primes do not tile the table: Engram layer " + std::to_string(G.engram_layers[(size_t) i]) +
                                  "'s primes sum to " + std::to_string(s) + ", its table has " +
                                  std::to_string(G.engram_rows[(size_t) i]) + " rows")) return e;
    }
    return {};
}

/// STRATA_EXPERTS_USED=k / STRATA_EXPERTS_USED_DECODE=k ("fewer experts per token", upstream Strata's "Faster" mode:
/// 5 of 6): route each token to its k highest-scoring experts (the model's own selection score: sqrt(softplus) + the
/// exp_probs_b bias + the route bias) instead of n_used.  The weights are those of the model's gate with top-k = k:
/// the k kept raw scores divided by THEIR sum (when the GGUF's expert_weights_norm / norm_topk_prob is set - V4.1:
/// true) times expert_weights_scale (routed_scaling_factor, 1.5) - so the routed sum keeps its magnitude next to the
/// shared expert and the residual instead of shrinking by the dropped expert's share.  The shared expert always runs.
/// Unset, empty or out of 1..n_used: n_used (said once on stderr when out of range) - exactly the old behaviour.
inline int64_t ds41_experts_used_env(const char * name, int64_t n_used, bool & said) {
    const char * v = std::getenv(name);
    if (!v || !*v || n_used < 1) return n_used;
    char * end = nullptr;
    const long k = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || k < 1 || k > n_used) {
        if (!said) std::fprintf(stderr, "%s=%s ignored: experts per token must be 1..%lld\n", name, v, (long long) n_used);
        said = true;
        return n_used;
    }
    if (!said && k < n_used)
        std::fprintf(stderr, "experts: %s=%ld - %ld of %lld routed experts per token (weights renormalised over the %ld)\n",
                     name, k, k, (long long) n_used, k);
    said = true;
    return (int64_t) k;
}
/// STRATA_EXPERTS_USED: every pass (prompt, decode, verify).  Applied by ds41_read_geometry to G.n_used, so the router,
/// the routed-expert tier and every ids/weights stride follow it; G.n_used_model keeps the file's value.
inline int64_t ds41_experts_used(int64_t n_used) {
    static bool said = false;
    return ds41_experts_used_env("STRATA_EXPERTS_USED", n_used, said);
}
/// STRATA_EXPERTS_USED_DECODE: the decode / verify passes only (the prompt keeps n_used); ds41_generate.cpp applies it.
inline int64_t ds41_experts_used_decode(int64_t n_used) {
    static bool said = false;
    return ds41_experts_used_env("STRATA_EXPERTS_USED_DECODE", n_used, said);
}

/// The file's geometry as it is (ds41_read_geometry = this + STRATA_EXPERTS_USED).
inline std::string ds41_read_geometry_file(const GgufModel & model, Ds41Geometry & G, Ds41Issues * iss = nullptr) {
    const GgufFile & g = model.meta();
    std::string e;
    const std::string arch = ds41_arch(g);
    if (!ds41_arch_accepted(arch)) {
        std::string acc;
        for (const auto & a : ds41_accepted_archs()) acc += (acc.empty() ? "" : ", ") + a;
        if (detail::halt(iss, e = arch.empty() ? "not a deepseek41 GGUF (no general.architecture)"
                                               : "not a deepseek41 GGUF: general.architecture is \"" + arch + "\" (accepted: " + acc +
                                                     "; DS41_ACCEPT_ARCH=" + arch + " reads it as deepseek41, every check still on)")) return e;
    }
    if (ds41_is_llamacpp_gguf(model)) {
        e = ds41_read_geometry_llamacpp(model, G, iss);
        return e.empty() ? ds41_check_geometry(G, iss) : e;
    }
    const std::string P = ds41_prefix(g);
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "hidden_size", &G.n_embd }, { "num_hidden_layers", &G.n_layer }, { "num_attention_heads", &G.n_head },
             { "head_dim", &G.d_head }, { "qk_rope_head_dim", &G.d_rope }, { "q_lora_rank", &G.r_q },
             { "o_lora_rank", &G.o_lora }, { "o_groups", &G.o_groups }, { "sliding_window", &G.n_swa },
             { "n_routed_experts", &G.n_expert }, { "num_experts_per_tok", &G.n_used },
             { "moe_intermediate_size", &G.n_ff }, { "n_shared_experts", &G.n_shared }, { "hc_mult", &G.hc },
             { "hc_sinkhorn_iters", &G.hc_iters }, { "index_n_heads", &G.idx_heads }, { "index_head_dim", &G.idx_dim },
             { "index_topk", &G.idx_topk }, { "candidate_source_layer_id", &G.cand_layer },
             { "candidate_topk_blocks", &G.cand_blocks }, { "candidate_block_size", &G.cand_bsize },
             { "vocab_size", &G.vocab }, { "max_position_embeddings", &G.ctx_len } })
        if (detail::halt(iss, e = detail::u64(g, P + u.k, *u.v, iss))) return e;
    struct F { const char * k; double * v; };
    for (const F & f : std::initializer_list<F>{
             { "routed_scaling_factor", &G.weights_scale }, { "hc_eps", &G.hc_eps }, { "rms_norm_eps", &G.rms_eps },
             { "rope_theta", &G.rope_base }, { "compress_rope_theta", &G.compress_rope_base },
             { "rope_scaling.factor", &G.rope_factor }, { "rope_scaling.beta_fast", &G.beta_fast },
             { "rope_scaling.beta_slow", &G.beta_slow }, { "rope_scaling.original_max_position_embeddings", &G.orig_ctx },
             { "swiglu_limit", &G.swiglu_limit } })
        if (detail::halt(iss, e = detail::f64(g, P + f.k, *f.v, iss))) return e;
    {
        const MetaValue * v = detail::get(g, P + "norm_topk_prob", iss);
        if (!v) { if (detail::halt(iss, e = "missing " + P + "norm_topk_prob")) return e; }
        else G.norm_topk = v->u != 0;
        const MetaValue * sf = detail::get(g, P + "scoring_func", iss);
        if ((!sf || sf->s != "sqrtsoftplus") &&
            detail::halt(iss, e = P + "scoring_func is " + detail::show(sf) + ", not \"sqrtsoftplus\"")) return e;
    }
    if (detail::halt(iss, e = detail::arr(g, P + "compress_ratios", G.ratio, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + "kv_source_layer_ids", G.kv_sources, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + "index_source_layer_ids", G.index_sources, iss))) return e;
    // Engram: arrays (full, not the reader's sample) + the shape from the config JSON
    const MetaValue * cfg = detail::get(g, P + "config", iss);
    if (!cfg) { if (detail::halt(iss, e = "missing " + P + "config")) return e; }
    else {
        double d = 0;
        if (detail::halt(iss, e = detail::json_num(cfg->s, "engram_n_heads", d))) return e;
        G.engram_heads = (int64_t) d;
        d = 0;
        if (detail::halt(iss, e = detail::json_num(cfg->s, "engram_head_dim", d))) return e;
        G.engram_dim = (int64_t) d;
        d = 0;
        if (detail::halt(iss, e = detail::json_num(cfg->s, "engram_max_ngram_size", d))) return e;
        G.engram_ngram = (int64_t) d;
    }
    if (detail::halt(iss, e = detail::u64(g, P + "engram.pad_id", G.engram_pad, iss))) return e;
    if (detail::halt(iss, e = detail::u64(g, P + "engram.compressed_vocab_size", G.engram_cvocab, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + "engram.layer_ids", G.engram_layers, iss))) return e;
    if (detail::halt(iss, e = detail::arr(g, P + "engram.rows", G.engram_rows, iss))) return e;
    const std::string & path = g.path();
    auto whole = [&](const std::string & k, std::vector<int64_t> & out) {   // a metadata array, whole
        if (iss) iss->seen.insert(k);
        return ds41_read_int_array(path, k, out);
    };
    std::vector<int64_t> tm, pr;
    if (detail::halt(iss, e = whole(P + "engram.token_map", tm))) return e;
    if (detail::halt(iss, e = whole(P + "engram.primes", pr))) return e;
    if (detail::halt(iss, e = whole(P + "engram.multipliers", G.engram_mult))) return e;
    if ((int64_t) tm.size() != G.vocab &&
        detail::halt(iss, e = "engram.token_map has " + std::to_string(tm.size()) + " entries, vocab_size is " +
                              std::to_string(G.vocab))) return e;
    G.engram_token_map.assign(tm.begin(), tm.end());
    G.engram_primes.assign(pr.begin(), pr.end());
    detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
    detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
    return ds41_check_geometry(G, iss);   // (ratios, sources, shape, Engram tiling: the checks both conventions share)
}

inline std::string ds41_read_geometry(const GgufModel & model, Ds41Geometry & G, Ds41Issues * iss = nullptr) {
    std::string e = ds41_read_geometry_file(model, G, iss);
    G.n_used_model = G.n_used;
    G.n_used = ds41_experts_used(G.n_used);   // STRATA_EXPERTS_USED (unset: unchanged)
    return e;
}

/// Whether the engine reads tensor `n` as a routed expert (its own blob reader, never ggml) - ds41_dense.cpp is_routed.
inline bool ds41_is_routed_name(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

/// The full tensor map + per-layer expert tiers.  `layers`: the blocks the file must hold (a partial test file
/// holds a subset - pass them; the real file holds 0..n_layer-1).
inline std::string ds41_check_tensors(const GgufModel & m, Ds41Geometry & G, const std::vector<int64_t> & layers,
                                      Ds41Issues * iss = nullptr) {
    std::string e;
    std::map<std::string, std::pair<std::vector<uint64_t>, uint32_t>> want;   // name -> (shape GGUF order, type; ~0 = any)
    const uint32_t ANY = ~0u;
    // the llama.cpp convention (ISTA-DASLab's GGUFs): the same tensors and shapes, but dense types of the quantizer's
    // choice (Q5_K / Q8_0 / BF16 / F32 - the loader converts what the graph needs), attn_output_a stored 3-D, the
    // Engram projections under llama.cpp's names, no vision router bias, and the Engram hash arrays as tensors
    const bool lc = ds41_is_llamacpp_gguf(m);
    // llama.cpp mainline (Unsloth's GGUFs): engram_gate (= q_norm * k_norm) + engram_kv, the vision router bias kept, the
    // hash constants in the metadata, both Engram tables as one per_layer_token_embd tensor
    const bool ml = ds41_is_mainline_gguf(m);
    auto add = [&](const std::string & n, std::vector<uint64_t> sh, uint32_t t) { want[n] = { std::move(sh), lc ? ANY : t }; };
    const uint64_t D = (uint64_t) G.n_embd, HC = (uint64_t) G.hc, MIX = (uint64_t) G.hc_mix();
    const uint64_t DH = (uint64_t) G.d_head, NH = (uint64_t) G.n_head, RQ = (uint64_t) G.r_q;
    const uint64_t OG = (uint64_t) G.o_groups, OL = (uint64_t) G.o_lora, NE = (uint64_t) G.n_expert, FF = (uint64_t) G.n_ff;
    const uint64_t IH = (uint64_t) G.idx_heads, ID = (uint64_t) G.idx_dim;
    add("token_embd.weight", { D, (uint64_t) G.vocab }, ty::F16);
    add("output_norm.weight", { D }, ty::F32);
    add("output.weight", { D, (uint64_t) G.vocab }, ty::Q8_0);
    G.experts.assign((size_t) (G.n_layer > 0 ? G.n_layer : 0), {});
    for (int64_t l : layers) {
        if (l < 0 || l >= G.n_layer) {
            if (detail::halt(iss, e = "layer " + std::to_string(l) + " is outside block_count " + std::to_string(G.n_layer))) return e;
            continue;
        }
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * site : { "attn", "ffn" }) {
            add(b + "hc_" + site + "_fn.weight", { HC * D, MIX }, ty::F16);
            add(b + "hc_" + site + "_base.weight", { MIX }, ty::F32);
            add(b + "hc_" + site + "_scale.weight", { 3 }, ty::F32);
            add(b + site + "_norm.weight", { D }, ty::F32);
        }
        add(b + "attn_sinks.weight", { NH }, ty::F32);
        add(b + "attn_q_a.weight", { D, RQ }, ty::Q8_0);
        add(b + "attn_q_b.weight", { RQ, NH * DH }, ty::Q8_0);
        add(b + "attn_q_a_norm.weight", { RQ }, ty::F32);
        add(b + "attn_kv.weight", { D, DH }, ty::Q8_0);
        add(b + "attn_kv_a_norm.weight", { DH }, ty::F32);
        if (lc && !ml) add(b + "attn_output_a.weight", { NH * DH / OG, OL, OG }, ANY);   // the same bytes, 3-D (#30817)
        else add(b + "attn_output_a.weight", { NH * DH / OG, OG * OL }, ty::Q8_0);
        add(b + "attn_output_b.weight", { OG * OL, D }, ty::Q8_0);
        if (G.is_kv_source(l)) {
            add(b + "attn_compressor_kv.weight", { D, DH }, ty::F16);
            add(b + "attn_compressor_norm.weight", { DH }, ty::F32);
            add(b + "indexer.attn_k.weight", { DH, ID }, ty::F16);
            add(b + "indexer.k_norm.weight", { ID }, ty::F32);
            if ((size_t) l < G.ratio.size() && G.ratio[(size_t) l] > 1) add(b + "attn_compressor_gate.weight", { D, DH }, ty::F16);
        }
        if (G.is_index_source(l)) {
            add(b + "indexer.attn_q_b.weight", { RQ, IH * ID }, ty::F16);
            add(b + "indexer.proj.weight", { D, IH }, ty::F16);
        }
        add(b + "ffn_gate_inp.weight", { D, NE }, ty::F32);
        add(b + "exp_probs_b.bias", { NE }, ty::F32);
        if (!lc || ml) add(b + "exp_probs_b_vl.bias", { NE }, ty::F32);   // (vision only; absent in #30817's convention)
        add(b + "ffn_gate_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_up_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_down_shexp.weight", { FF, D }, ty::Q8_0);
        if (G.engram_slot(l) >= 0 && ml) {   // mainline: the norm product, ours engram_kv's name
            add(b + "engram_gate.weight", { D, HC }, ty::F32);
            add(b + "engram_kv.weight", { (uint64_t) ((G.engram_ngram - 1) * G.engram_heads * G.engram_dim), (HC + 1) * D }, ty::F16);
        } else if (G.engram_slot(l) >= 0) {   // (llama.cpp names: engram_q / engram_k / engram_wkv - ds41_canonical_name)
            add(b + (lc ? "engram_q.weight" : "engram_q_norm.weight"), { D, HC }, ty::F32);
            add(b + (lc ? "engram_k.weight" : "engram_k_norm.weight"), { D, HC }, ty::F32);
            add(b + (lc ? "engram_wkv.weight" : "engram_kv.weight"),
                { (uint64_t) ((G.engram_ngram - 1) * G.engram_heads * G.engram_dim), (HC + 1) * D }, ty::F16);
        }
        // routed experts: uniform role tensors or per-type tensors + the I32 maps
        Ds41ExpertMap & X = G.experts[(size_t) l];
        X.gu_type.assign(NE, -1); X.d_type.assign(NE, -1); X.gu_slot.assign(NE, -1); X.d_slot.assign(NE, -1);
        for (int k = 0; k < 2; ++k) {   // 0 = gate+up, 1 = down
            const std::string mapn = b + (k ? "ffn_exps_down_type" : "ffn_exps_gu_type");
            std::vector<int> & tv = k ? X.d_type : X.gu_type;
            std::vector<int> & sv = k ? X.d_slot : X.gu_slot;
            const char * roles[2] = { k ? "down" : "gate", k ? nullptr : "up" };
            size_t si = 0;
            const TensorInfo * tm = m.find(mapn, &si);
            if (!tm) {   // uniform: one stacked tensor per role
                const TensorInfo * t0 = m.find(b + "ffn_" + roles[0] + "_exps.weight");
                if (!t0) {
                    if (detail::halt(iss, e = "missing " + b + "ffn_" + roles[0] + "_exps.weight (and no " + mapn + ")")) return e;
                    continue;
                }
                for (uint64_t x = 0; x < NE; ++x) { tv[x] = (int) t0->type; sv[x] = (int) x; }
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string n = b + "ffn_" + r + "_exps.weight";
                    add(n, k ? std::vector<uint64_t>{ FF, D, NE } : std::vector<uint64_t>{ D, FF, NE }, t0->type);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[(int) t0->type] = n;
                }
                continue;
            }
            add(mapn, { NE }, ty::I32);
            if (tm->type != ty::I32 || tm->shape.size() != 1 || tm->shape[0] != NE || !m.in_bounds(*tm, si)) {
                if (detail::halt(iss, e = mapn + ": bad shape/type")) return e;
                continue;
            }
            const int32_t * mv = reinterpret_cast<const int32_t *>(m.shard(si).tensor_data(*tm));
            std::map<int, int> count;
            for (uint64_t x = 0; x < NE; ++x) { tv[x] = mv[x]; sv[x] = count[mv[x]]++; }
            for (const auto & [t, n] : count) {
                if (t != (int) ty::Q2_0 && t != (int) ty::Q3_K && t != (int) ty::PTQ1_0) {
                    if (detail::halt(iss, e = mapn + ": type " + std::to_string(t) + " is not an expert tier")) return e;
                    continue;
                }
                std::string tn = ggml_type_name((uint32_t) t);
                for (char & c : tn) c = (char) std::tolower((unsigned char) c);
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string nm = b + "ffn_" + r + "_exps." + tn + ".weight";
                    add(nm, k ? std::vector<uint64_t>{ FF, D, (uint64_t) n } : std::vector<uint64_t>{ D, FF, (uint64_t) n }, (uint32_t) t);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[t] = nm;
                }
            }
        }
    }
    if (ml) {   // the Engram tables, layer after layer (read by Ds41Engram from --engram)
        uint64_t rows = 0;
        for (int64_t r : G.engram_rows) rows += (uint64_t) r;
        add("per_layer_token_embd.weight", { (uint64_t) G.engram_dim, rows }, ANY);
    } else if (lc) {   // the Engram hash arrays (read by ds41_read_geometry_llamacpp)
        const uint64_t NEL = (uint64_t) G.engram_layers.size(), NG = (uint64_t) G.engram_ngram;
        const uint64_t NB = (NG - 1) * (uint64_t) G.engram_heads;
        add("engram_token_map.weight", { (uint64_t) G.vocab }, ANY);
        add("engram_primes.weight", { NB, NEL }, ANY);
        add("engram_offsets.weight", { NB, NEL }, ANY);
        add("engram_multipliers.weight", { NG, NEL }, ANY);
    }
    // every predicted tensor present with its shape/type; every file tensor predicted
    for (const auto & [n, st] : want) {
        size_t s = 0;
        const TensorInfo * t = m.find(n, &s);
        if (!t) {
            if (detail::halt(iss, e = "missing tensor " + n)) return e;
            continue;
        }
        if (t->shape != st.first) {
            std::string a, w;
            for (auto x : t->shape) a += std::to_string(x) + " ";
            for (auto x : st.first) w += std::to_string(x) + " ";
            if (detail::halt(iss, e = n + ": shape [" + a + "] expected [" + w + "]")) return e;
            continue;
        }
        if (st.second != ANY && t->type != st.second) {
            if (detail::halt(iss, e = n + ": type " + std::string(ggml_type_name(t->type)) + " expected " + ggml_type_name(st.second))) return e;
            continue;
        }
        // a type id this engine has no block layout for (ik_llama's 144+, a newer ggml's): refused by name, never read
        if (!ds41_type_known(t->type)) {
            if (detail::halt(iss, e = n + ": GGUF type " + ds41_type_label(t->type) + " is not a type this engine reads")) return e;
            continue;
        }
        // type 43 here is either Unsloth's MXFP8 Engram table (read by Ds41Engram) or mainline ggml's F8_E4M3; neither
        // reaches this build's ggml, so a dense tensor of it (or of PTQ1_0) is refused here instead of in ggml_new_tensor
        if (t->type == ty::FP8_43 && ds41_type43_layout(m, *t, s) == Ds41Fp8Layout::F8E4M3) {
            if (detail::halt(iss, e = n + ": type 43 holds one byte a value here (ggml's F8_E4M3, llama.cpp v0.9.0+), not the "
                                          "33-byte MXFP8 blocks this engine reads as type 43")) return e;
            continue;
        }
        if (t->type >= ty::kGgmlTypes && !ds41_is_routed_name(n) && n != "per_layer_token_embd.weight") {
            if (detail::halt(iss, e = n + ": type " + ds41_type_label(t->type) + " on a dense tensor (the dense loader reads ggml's types up "
                                      "to Q2_0 = 42; FP8 dense rows are not read yet)")) return e;
            continue;
        }
        if (!m.in_bounds(*t, s) && detail::halt(iss, e = n + ": out of bounds")) return e;
    }
    for (size_t s = 0; s < m.size(); ++s)
        for (const auto & t : m.shard(s).tensors())
            if (!want.count(t.name)) {
                // a DwarfStar (antirez/ds4) single-file GGUF carries the Engram tables inline as blk.L.engram_embd.weight;
                // this engine reads Engram from --engram (the HF shards), so those tensors are simply not used
                if (t.name.rfind("blk.", 0) == 0 && t.name.size() > 19 && t.name.compare(t.name.size() - 19, 19, ".engram_embd.weight") == 0) continue;
                if (detail::halt(iss, e = "unexpected tensor " + t.name + (iss ? " (" + ds41_type_label(t.type) + ")" : std::string()))) return e;
            }
    return {};
}

/// The name the engine uses for a file tensor: ours unchanged; the llama.cpp convention's Engram projections mapped to
/// ours (engram_q -> engram_q_norm, engram_k -> engram_k_norm, engram_wkv -> engram_kv; same shapes, model.py's
/// q_norm / k_norm / wkv).
inline std::string ds41_canonical_name(const std::string & n) {
    static const char * from[3] = { ".engram_q.weight", ".engram_k.weight", ".engram_wkv.weight" };
    static const char * to[3] = { ".engram_q_norm.weight", ".engram_k_norm.weight", ".engram_kv.weight" };
    for (int i = 0; i < 3; ++i) {
        const size_t L = std::strlen(from[i]);
        if (n.size() > L && n.compare(n.size() - L, L, from[i]) == 0) return n.substr(0, n.size() - L) + to[i];
    }
    return n;
}

/// The blocks a file holds: `strata.v41.partial_layers` when the writer made a partial test file, else all.
inline std::vector<int64_t> ds41_file_layers(const GgufModel & m, const Ds41Geometry & G) {
    std::vector<int64_t> v;
    if (detail::arr(m.meta(), "strata.v41.partial_layers", v).empty()) return v;
    for (int64_t l = 0; l < G.n_layer; ++l) v.push_back(l);
    return v;
}

}  // namespace strata::v41
