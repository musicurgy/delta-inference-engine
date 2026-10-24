// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <string.h>
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// STRATA_DENSE_ASYNC=1 (default off): Impl::compute(gf, cached, wait = false) launches a cached graph WITHOUT the host
// wait ggml_backend_graph_compute ends with (every graph launch was a host sync: finish, the shared expert, ...).  The
// backend stream may then still be running when the host goes on, and ggml_backend_tensor_get / _set / _copy /
// buffer_clear run on cudaStreamPerThread, which is NOT ordered after it - so every such call in this file goes through
// these wrappers, which wait for a pending launch first.  A host wait on the stream (dsync_be) clears it.
ggml_backend_t g_async_be = nullptr;
void dsync() {
    if (g_async_be) { ggml_backend_synchronize(g_async_be); g_async_be = nullptr; }
}
void dsync_be(ggml_backend_t be) {
    ggml_backend_synchronize(be);
    if (g_async_be == be) g_async_be = nullptr;
}
void dget(const ggml_tensor * t, void * data, size_t off, size_t n) { dsync(); ggml_backend_tensor_get(t, data, off, n); }
void dset(ggml_tensor * t, const void * data, size_t off, size_t n) { dsync(); ggml_backend_tensor_set(t, data, off, n); }
void dcopy(ggml_tensor * src, ggml_tensor * dst) { dsync(); ggml_backend_tensor_copy(src, dst); }
void dclear(ggml_backend_buffer_t b, uint8_t v) { dsync(); ggml_backend_buffer_clear(b, v); }
bool dense_async() {
    static const bool on = env_on("STRATA_DENSE_ASYNC", false);
    return on;
}

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    /// alloc() will return at once (the graph is allocated at the current epoch): no device memory is touched
    bool ready(const ggml_cgraph * g, bool reuse) const {
        auto it = graph_epoch.find(g);
        return reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch;
    }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        if (!ggml_gallocr_alloc_graph(a, g)) return false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) {
            ++epoch;
            static const bool log = env_on("STRATA_ALLO_LOG", false);
            if (log) std::fprintf(stderr, "ds41_dense: graph buffer %.1f -> %.1f MiB (epoch %llu, %zu graphs re-allocate)\n",
                                  before / 1048576.0, ggml_gallocr_get_buffer_size(a, 0) / 1048576.0,
                                  (unsigned long long) epoch, graph_epoch.size());
        }
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

// STRATA_GRAPH_RESERVE_MIB=N: the decode allocator starts with an N MiB buffer instead of growing to the largest graph
// seen so far.  Every growth frees the buffer all cached graphs live in, so each of them is re-allocated on its next
// use - a host wait, then ggml-cuda runs it kernel by kernel, then re-captures and re-instantiates its CUDA graph
// (round 90: ~590 such re-captures in one 400-token answer as the attention's pass_cap doubled).  Same graphs, same
// kernels: exact.  The N MiB come out of the expert slots (reserved before the tier sizes them).
static void allo_reserve(Allo & al) {
    static const long long mib = std::getenv("STRATA_GRAPH_RESERVE_MIB") ? std::atoll(std::getenv("STRATA_GRAPH_RESERVE_MIB")) : 0;
    if (mib <= 0 || !al.a) return;
    ggml_init_params p = { ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(4, false), nullptr, true };
    ggml_context * c = ggml_init(p);
    ggml_tensor * x = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t) mib << 18);
    ggml_set_input(x);
    ggml_cgraph * g = ggml_new_graph_custom(c, 4, false);
    ggml_build_forward_expand(g, ggml_sum(c, x));
    if (!ggml_gallocr_reserve(al.a, g)) std::fprintf(stderr, "ds41_dense: STRATA_GRAPH_RESERVE_MIB=%lld did not fit\n", mib);
    ggml_free(c);
}

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

// The smart router (STRATA_RB_SMART=1, or STRATA_RB_PROTECT / STRATA_RB_POOL set): the protect / pool ranks come from
// the rknob tensor, so a request can change them (set_route_select) without a graph rebuild, and the router also
// writes the exact top-k for the route meter.  kRK = the deepest rank it can select.
constexpr int kRK = 32;
int env_rank(const char * k) {
    const char * e = std::getenv(k);
    return e ? std::max(0, std::min(kRK, std::atoi(e))) : 0;
}
bool smart_router() {
    static const bool on = env_on("STRATA_RB_SMART", false) || env_rank("STRATA_RB_PROTECT") > 0 || env_rank("STRATA_RB_POOL") > 0;
    return on;
}

}  // namespace

// ================================================================== Impl

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    WStore w;
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * o_xids = nullptr;       // I32 [NU, NC]  route meter (smart router): the exact top-NU
    ggml_tensor * o_xwts = nullptr;       // [NU, NC]      ... and their normalized weights
    ggml_tensor * rknob = nullptr;        // F32 [kRK + 1, 2] smart router: protect / pool rank selectors (set_route_select)
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
    std::vector<Layer> ly;

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)
    ggml_tensor * zpad16 = nullptr;       // F16 [DH, 1] zeros: the flash-attn key padding (STRATA_ATTN_F16_GATHER)
    ggml_tensor * zidx = nullptr;         // I32 [kZidx] zeros: gathers that zero row once per pad slot
    // STRATA_ATTN_F16_GATHER=1 (CUDA): the attention's key rows are gathered straight into F16 and concatenated /
    // padded in F16, instead of gathered F32 -> concat F32 -> pad F32 -> cast F16.  Same values bit for bit (the
    // gather dequantizes in float and rounds with the same __float2half as the cast; concat and pad only move bits),
    // but the 512-query sub-batch's key tensors peak at ~0.74 GB instead of ~1.48 GB (the "1424 MiB" pass-graph
    // buffer of V41_START_HERE run20/21).
    bool f16_gather = false;

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_xids;       // route meter: o_xids / o_xwts readbacks, and the counters since route_meter(true)
    std::vector<float> host_xwts;
    long long rm_slots = 0, rm_swaps = 0, rm_rows = 0;
    double rm_kept = 0;
    const uint8_t * embd = nullptr;       // token_embd F16 rows on the mmap
    // ---- vision (DeepSeek-V4.1 image input): the request's image span rows, by absolute position.  When enabled,
    // set_vision() swaps in exp_probs_b_vl on those positions and
    // marks them DEAD for Engram.  Empty (n_vis = 0) for a text request.
    bool vision = false;                  // opt-in; requires per-layer exp_probs_b_vl.bias
    ggml_tensor * i_img = nullptr;        // F32 [NC]  1.0 at an image position, 0.0 elsewhere (the VL bias / gate mask)
    const float * vis_rows = nullptr;     // n_vis * D floats, already in prompt order
    const int64_t * vis_pos = nullptr;    // n_vis ascending absolute prompt positions
    int n_vis = 0;
    std::vector<std::vector<float>> eng_rows;
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;
    bool eng_up[2] = { false, false };   // STRATA_ENGRAM_SPLIT: this Engram slot's rows already in i_eng

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    struct GKey { int il, kind; int64_t cap, n; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n) < std::tie(o.il, o.kind, o.cap, o.n); } };
    std::map<GKey, ggml_cgraph *> graphs;
    // STRATA_PASS_GRAPHS: cached graphs of multi-row decode passes and of the head, each in its own node context so a
    // superseded pass_cap's graph can be dropped (pass_graph below)
    struct MGraph { ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_tensor * out = nullptr; };
    std::map<GKey, MGraph> mgraphs;
    void free_mgraphs() {
        for (auto & kv : mgraphs) { allo_dec.graph_epoch.erase(kv.second.gf); ggml_free(kv.second.ctx); }
        mgraphs.clear();
    }
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::vector<int32_t> pred_span;            // predict_multi: the readback span (row t's 16 at t * NE)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    bool shexp_pending = false;   ///< STRATA_SHEXP_SPLIT: the last attn_router left the shared expert to shexp_launch
    // STRATA_PF_OVERLAP: finish_predict's readback lands in pinned memory behind an event, so the next layer's attention
    // can be launched before the host waits for the prediction (the prefetch DMAs then overlap that attention)
    ggml_backend_buffer_t pred_host_buf = nullptr;
    int32_t * pred_host = nullptr;
    ggml_backend_event_t pred_ev = nullptr;
    int pred_n = 0;
    // STRATA_D3_PF_ASYNC: predict_multi_launch's readback (pinned, behind pm_ev); pm_rows rows pending, pm_rs ids apart
    ggml_backend_buffer_t pm_host_buf = nullptr;
    int32_t * pm_host = nullptr;
    ggml_backend_event_t pm_ev = nullptr;
    int64_t pm_rows = 0, pm_rs = 0;
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs
    // STRATA_KEEP_DEC_GRAPHS=1 (--serve, elastic columns): the decode-width columns and every graph built on them,
    // parked by set_columns() while a prompt batch runs on its own wider columns, and brought back after it
    struct World {
        bool valid = false;
        ggml_context * nctx = nullptr;
        ggml_backend_buffer_t nbuf = nullptr;
        int64_t NC = 0;
        ggml_tensor * x_state = nullptr, * pre_state = nullptr, * hap = nullptr, * post_f = nullptr, * comb_f = nullptr;
        ggml_tensor * shexp = nullptr, * o_fn = nullptr, * o_ids = nullptr, * o_wts = nullptr, * o_xids = nullptr;
        ggml_tensor * o_xwts = nullptr, * routed = nullptr, * cand = nullptr;
        ggml_tensor * i_emb = nullptr, * i_pos = nullptr, * i_img = nullptr, * i_widx = nullptr, * i_wmask = nullptr;
        ggml_tensor * i_wslot = nullptr, * i_sidx = nullptr, * i_sslot = nullptr, * i_lastblk = nullptr;
        ggml_tensor * i_nvis[3] = {}, * i_cslot[3] = {}, * i_cpos[3] = {}, * i_eng[2] = {};
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> sel;   // per layer: sel, selm
        ggml_context * gctx = nullptr;
        Allo allo_dec, allo_pass;
        std::map<GKey, ggml_cgraph *> graphs;
        std::map<GKey, MGraph> mgraphs;
        std::map<int, ggml_tensor *> logits_out, predict_out, fpredict_out;
        std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;
        std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead;
    } parked_dec;
    void free_world(World & o) {   // the graphs / columns a World holds (its allocators go with it)
        for (auto & kv : o.mgraphs) ggml_free(kv.second.ctx);
        o.mgraphs.clear();
        o.graphs.clear();
        o.logits_out.clear();
        o.predict_out.clear();
        o.fpredict_out.clear();
        o.predict_out_n.clear();
        o.fpredict_ahead.clear();
        for (Allo * al : { &o.allo_dec, &o.allo_pass }) {
            if (al->a) ggml_gallocr_free(al->a);
            al->a = nullptr;
            al->graph_epoch.clear();
            al->compute_bufs.clear();
        }
        if (o.gctx) ggml_free(o.gctx);
        o.gctx = nullptr;
        if (o.nbuf) ggml_backend_buffer_free(o.nbuf);
        o.nbuf = nullptr;
        if (o.nctx) ggml_free(o.nctx);
        o.nctx = nullptr;
        o.valid = false;
    }
    void swap_world(World & o) {   // this Impl's columns + graphs <-> o's
        std::swap(nctx, o.nctx);
        std::swap(nbuf, o.nbuf);
        std::swap(NC, o.NC);
        std::swap(x_state, o.x_state);
        std::swap(pre_state, o.pre_state);
        std::swap(hap, o.hap);
        std::swap(post_f, o.post_f);
        std::swap(comb_f, o.comb_f);
        std::swap(shexp, o.shexp);
        std::swap(o_fn, o.o_fn);
        std::swap(o_ids, o.o_ids);
        std::swap(o_wts, o.o_wts);
        std::swap(o_xids, o.o_xids);
        std::swap(o_xwts, o.o_xwts);
        std::swap(routed, o.routed);
        std::swap(cand, o.cand);
        std::swap(i_emb, o.i_emb);
        std::swap(i_pos, o.i_pos);
        std::swap(i_img, o.i_img);
        std::swap(i_widx, o.i_widx);
        std::swap(i_wmask, o.i_wmask);
        std::swap(i_wslot, o.i_wslot);
        std::swap(i_sidx, o.i_sidx);
        std::swap(i_sslot, o.i_sslot);
        std::swap(i_lastblk, o.i_lastblk);
        for (int r = 0; r < 3; ++r) {
            std::swap(i_nvis[r], o.i_nvis[r]);
            std::swap(i_cslot[r], o.i_cslot[r]);
            std::swap(i_cpos[r], o.i_cpos[r]);
        }
        for (int e = 0; e < 2; ++e) std::swap(i_eng[e], o.i_eng[e]);
        o.sel.resize(ly.size());
        for (size_t i = 0; i < ly.size(); ++i) {
            std::swap(ly[i].sel, o.sel[i].first);
            std::swap(ly[i].selm, o.sel[i].second);
        }
        std::swap(gctx, o.gctx);
        for (auto pr : { std::make_pair(&allo_dec, &o.allo_dec), std::make_pair(&allo_pass, &o.allo_pass) }) {
            std::swap(pr.first->a, pr.second->a);
            std::swap(pr.first->epoch, pr.second->epoch);
            pr.first->graph_epoch.swap(pr.second->graph_epoch);
            pr.first->compute_bufs.swap(pr.second->compute_bufs);
        }
        graphs.swap(o.graphs);
        mgraphs.swap(o.mgraphs);
        logits_out.swap(o.logits_out);
        predict_out.swap(o.predict_out);
        fpredict_out.swap(o.fpredict_out);
        predict_out_n.swap(o.predict_out_n);
        fpredict_ahead.swap(o.fpredict_ahead);
        span_memo_pos0 = -1;   // the index tensors now in place hold another pass's spans
        span_memo_col0 = -1;
        span_memo_n = -1;
    }

    ~Impl() {
        dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the buffers freed here
        if (parked_dec.valid) free_world(parked_dec);
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        for (auto & kv : mgraphs) ggml_free(kv.second.ctx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    /// wait = false: under STRATA_DENSE_ASYNC a cached, already allocated graph is launched without the host wait
    bool compute(ggml_cgraph * gf, bool cached, bool wait = true);
    bool upload_inputs(const int * tids, int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), (int) ti.shape.size(), ne);
        ggml_set_name(t, ti.name.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry
    // engram_embd: a DwarfStar single-file GGUF's inline Engram tables (read from --engram, never loaded here)
    auto skip = [&](const std::string & n) { return is_routed(n) || n == "token_embd.weight" || n.find(".engram_embd.") != std::string::npos; };
    if (im.cpu) {
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        // STRATA_DENSE_REQUANT_ONLY=a,b,... : only the matrices whose names contain one of these (e.g. attn_q_b,shexp) -
        // a per-class quality sweep, and a mix that keeps the sensitive classes at Q8_0
        std::vector<std::string> rq_only;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT_ONLY"); e && *e) {
            std::stringstream ss(e);
            for (std::string part; std::getline(ss, part, ',');) if (!part.empty()) rq_only.push_back(part);
        }
        auto requant = [&](const TensorInfo & ti) {
            bool listed = rq_only.empty();
            for (const auto & p : rq_only) listed = listed || ti.name.find(p) != std::string::npos;
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight") && listed;
        };
        // STRATA_ENGRAM_KV_TYPE=<ggml type> (q8_0, ...; OFF by default, a quality experiment): the two Engram layers' F16
        // engram_kv projections ([6144, 25600], 300 MiB each, read by ggml_mul_mat only) re-quantized at load, like
        // STRATA_DENSE_REQUANT - ~0.3 GiB of VRAM back to the expert cache at q8_0.
        ggml_type ekv = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_ENGRAM_KV_TYPE"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) ekv = (ggml_type) t;
        auto target_of = [&](const TensorInfo & ti) -> int {
            if (requant(ti)) return (int) rq;
            if (ekv != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_F16 && ti.shape.size() == 2 &&
                ti.name.find("engram_kv") != std::string::npos && (int64_t) ti.shape[0] % ggml_blck_size(ekv) == 0)
                return (int) ekv;
            return -1;
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[ti.name] = make(ti, target_of(ti));
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        // rows of the file's tensor `ti` (bytes `src`) converted to t->type and uploaded; returns the bytes uploaded
        auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                        else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            dset(t, dst.data(), 0, dst.size());
            return dst.size();
        };
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[ti.name];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    dset(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                rq_to += convert_upload(t, ti, src);
                ++rq_n;
                rq_from += (size_t) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
            }
        if (rq_n) std::fprintf(stderr, "dense requant: %zu matrices (%s%s%s), %.2f -> %.2f GB\n", rq_n,
                               rq != GGML_TYPE_COUNT ? "Q8_0 -> " : "", rq != GGML_TYPE_COUNT ? ggml_type_name(rq) : "",
                               ekv != GGML_TYPE_COUNT ? (std::string("; engram_kv F16 -> ") + ggml_type_name(ekv)).c_str() : "",
                               rq_from / 1e9, rq_to / 1e9);
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
    }
    size_t si = 0;
    const TensorInfo * te = model.find("token_embd.weight", &si);
    im.embd = model.shard(si).tensor_data(*te);
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                  // [24, n]
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached, bool wait) {
    Allo & a = cached ? allo_dec : allo_pass;
    static const bool reuse_on = env_on("DS41_GRAPH_REUSE", true);
    const bool reuse = cached && reuse_on;
    // STRATA_DENSE_ASYNC: no launch without a host wait while the allocator may grow (free) the buffer an earlier
    // launch is still reading; the cached graphs are allocated once, so this is the warm-up only
    const bool ready = a.ready(gf, reuse);
    if (!ready) dsync();
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    bool ok_alloc = a.alloc(gf, reuse);
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
    if (!wait && ready && !cpu && dense_async()) {
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
        g_async_be = backend;
    } else {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
        if (g_async_be == backend) g_async_be = nullptr;   // the wait above covered the pending launch too
    }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// A graph result to the host.  STRATA_DENSE_ASYNC: the copy queued on the backend stream behind the (unwaited) graph,
// then ONE host wait - instead of the graph's wait plus ggml_backend_tensor_get's own copy-and-wait on another stream.
static void dread(Ds41Dense::Impl & im, const ggml_tensor * t, void * data, size_t n) {
    if (dense_async() && !im.cpu) {
        ggml_backend_tensor_get_async(im.backend, t, data, 0, n);
        dsync_be(im.backend);
    } else {
        dget(t, data, 0, n);
    }
}

// STRATA_PASS_GRAPHS=1: the graphs of a multi-row decode pass (a DSpark / lookup verify of n <= STRATA_PASS_GRAPH_MAX_N
// rows, default 8: begin, every layer's attention, finish) and the head are cached per (layer, width, pass_cap) like
// the n = 1 graphs, instead of being rebuilt and re-allocated every pass.  A rebuilt graph never matches ggml-cuda's
// captured CUDA graph, so each of its ~70 kernels per layer went through WDDM one launch at a time; a cached graph
// carries a uid and replays as one launch.  Same graph structure, same kernels, same inputs: exact.
static bool pass_graphs_on(int64_t n) {
    static const bool on = env_on("STRATA_PASS_GRAPHS", false);
    static const int64_t mx = std::getenv("STRATA_PASS_GRAPH_MAX_N") ? std::atoll(std::getenv("STRATA_PASS_GRAPH_MAX_N")) : 8;
    return on && n >= 1 && n <= mx;
}

// STRATA_FUSE_ROUTER=1 (default off): the router's tail - sqrt(softplus), the two bias adds, top-k, the weight gather,
// sum, div and scale, ~11 kernels per layer-pass - and the prefetch prediction's top-16 (~5) as ONE ggml_dsv4_route
// each, the router's weights written straight into o_wts.  The same float steps (ggml-cuda/dsv4-hc.cu): bit-identical
// ids and weights.  Kernel listing 2026-10-24: each kernel removed also saves ~1.7 us of WDDM graph launch.  CUDA only;
// the plain router only (no smart router / margin / vision bias).
static bool fuse_router(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_FUSE_ROUTER", false);
    return on && !im.cpu;
}

template <class F>
static Ds41Dense::Impl::MGraph & pass_graph(Ds41Dense::Impl & im, const Ds41Dense::Impl::GKey & key, F build) {
    auto it = im.mgraphs.find(key);
    if (it != im.mgraphs.end()) return it->second;
    // pass_cap only grows (a power of two): the same (layer, kind, width) at an older cap is never used again
    for (auto j = im.mgraphs.begin(); j != im.mgraphs.end();) {
        if (j->first.il == key.il && j->first.kind == key.kind && j->first.n == key.n) {
            im.allo_dec.graph_epoch.erase(j->second.gf);
            ggml_free(j->second.ctx);
            j = im.mgraphs.erase(j);
        } else {
            ++j;
        }
    }
    ggml_init_params p = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false), nullptr, true };
    Ds41Dense::Impl::MGraph g;
    g.ctx = ggml_init(p);
    g.gf = build(g.ctx, &g.out);
    static size_t peak = 0;   // node memory headroom, reported once per new high (a 4096-tensor budget per graph)
    if (ggml_used_mem(g.ctx) > peak) {
        peak = ggml_used_mem(g.ctx);
        std::fprintf(stderr, "ds41_dense: pass graph (kind %d, %lld rows) uses %zu of %zu KB node memory\n", key.kind,
                     (long long) key.n, peak / 1024, ggml_get_mem_size(g.ctx) / 1024);
    }
    return im.mgraphs.emplace(key, g).first->second;
}

// STRATA_SHEXP_SPLIT=1: the single-token decode graph leaves the shared expert out; Ds41Dense::shexp_launch runs it as
// its own cached graph from o_fn (the same bytes as fn) right before the routed tier, so the card computes it while
// the tier waits on the CPU pool / PCIe instead of on the attention's critical path.  Same ops, same inputs: exact.
static bool shexp_split() {
    static const bool on = std::getenv("STRATA_SHEXP_SPLIT") && std::getenv("STRATA_SHEXP_SPLIT")[0] == '1';
    return on;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
// get_rows whose result is F16 (f16) or the usual F32.  ggml_get_rows always declares F32; the CUDA kernel writes any
// of F32 / F16 / BF16, dequantizing in float and rounding with the same conversion a later cast would use, so retyping
// the (not yet allocated) result is the gather + cast in one step without the F32 intermediate.  CUDA only.
static ggml_tensor * rows_as(ggml_context * gc, ggml_tensor * a, ggml_tensor * idx, bool f16) {
    ggml_tensor * t = ggml_get_rows(gc, a, idx);
    if (f16 && t->type == GGML_TYPE_F32) {
        t->type = GGML_TYPE_F16;
        t->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
    return t;
}

static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

// the shared expert's ops on an [D, n] activation (model.py Expert, clamped SwiGLU)
static ggml_tensor * build_shexp_ops(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor * fn) {
    const float lim = (float) im.g.swiglu_limit;
    ggml_tensor * up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    return ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));
}

// STRATA_SHEXP_SPLIT: the shared expert alone, reading o_fn (written by this layer's attention graph), into shexp
static ggml_cgraph * build_shexp(Ds41Dense::Impl & im, ggml_context * gc, int il) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    ggml_tensor * fn = b.cols2(im.o_fn, im.D, 1, 0);
    ggml_tensor * se = build_shexp_ops(im, gc, il, fn);
    ggml_build_forward_expand(gf, ggml_cpy(gc, se, b.cols2(im.shexp, im.D, 1, 0)));
    return gf;
}

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool no_shexp = false) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));   // gate * (1 - image)
        }
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
               &a_pre, &a_post, &a_comb);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));             // [D, n]
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = b.rope(q, pos, r);

    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    ggml_tensor * wsrc = ggml_concat(gc, L.raw->type == GGML_TYPE_F32 ? L.raw : ggml_cast(gc, L.raw, GGML_TYPE_F32), kv2, 1);     // [DH, RAW + n]
    // F16 key path for this graph only when the zero index covers its padding (255 pad rows x n queries at most)
    const bool f16 = im.f16_gather && 255 * n <= im.zidx->ne[0];
    ggml_tensor * Kw = ggml_reshape_3d(gc, rows_as(gc, wsrc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4), f16), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, col0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * g_sel = nullptr, * g_comp = nullptr;      // STRATA_KGATHER: the selection and its source, for one gather
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this pass's writes (later reads depend on them)
    if (r) {
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;                                                  // [DH, n]
            if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);                     // [DH, 2 + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, n]
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                const int64_t k = std::min<int64_t>(n, 2);
                late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
                late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
            // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp, lr, cslot);
            outs.push_back(comp_w);
        }
        if (cap > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            const int64_t K = std::min<int64_t>(cap, im.ITOP);
            ggml_tensor * comp = comp_w ? comp_w : KS.comp;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
            ggml_tensor * sel = nullptr, * selm = nullptr;
            if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, ggml_cont(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], n, (size_t) col0 * 4), 1, n);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, n)), nvs)));
                ggml_tensor * sc = ggml_lightning_indexer(gc, iq, ksrc, iw, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, n]
                sc = ggml_reshape_2d(gc, sc, cap, n);
                if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, n), NEG_INF);
                    ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, n), 0.0f);
                    m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                    m = ggml_view_2d(gc, m, cap, n, m->nb[2], 0);
                    sc = ggml_add(gc, sc, m);
                }
                if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    GGML_ASSERT(cap % im.CBS == 0);
                    ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, n]
                    // pin: + 1e30 at block (nvis - 1) / CBS of each query
                    ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                    ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, n, (size_t) col0 * 4), 1, n);
                    ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                    bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                    ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, n]
                    outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                }
                sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, n]
                // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, n),
                                                        ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                     GGML_SORT_ORDER_ASC)), K, n);
                selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, n), sel), K, n);   // vis at sel
                outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1])));
                outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1])));
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            g_sel = sel;
            g_comp = comp;
            Kc = ggml_reshape_3d(gc, rows_as(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n)), f16), DH, K, n);
            Mc = ggml_cont(gc, selm);
            // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
        }
    }

    // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, n]
    int64_t M = Kall->ne[1];
    // STRATA_KGATHER=1 (default off): the key sets in ONE gather from [window source | compressed rows 0..cap | a zero
    // row] instead of two gathers, a concat and the padding concat - every query's [DH, M] keys were written three
    // times (round 100: concat_cont 3.25 s of a 16K prompt).  The same F16 values: the window rows round F32 -> F16
    // as rows_as does, the compressed rows dequantize the same way, the padding is the same zeros.  Only where the
    // whole-cap dequant is cheaper than the key sets it saves copying (cap <= n * M / 2: prompt passes, not decode).
    static const bool kgather = std::getenv("STRATA_KGATHER") && std::getenv("STRATA_KGATHER")[0] == '1';
    if (kgather && f16 && !im.cpu && g_sel && Kc) {
        const int64_t K = g_sel->ne[0], M0 = SWA + K, MP = (M0 + 255) / 256 * 256, nsrc = wsrc->ne[1];
        if (cap * 2 <= n * MP) {
            ggml_tensor * iota = ggml_cast(gc, ggml_scale(gc, ggml_view_1d(gc, im.neg_ramp, cap, 0), -1.0f), GGML_TYPE_I32);
            ggml_tensor * c16 = rows_as(gc, g_comp, iota, true);                                         // [DH, cap]
            ggml_tensor * src = ggml_concat(gc, ggml_concat(gc, ggml_cast(gc, wsrc, GGML_TYPE_F16), c16, 1), im.zpad16, 1);
            ggml_tensor * wi = ggml_reshape_2d(gc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4), SWA, n);
            ggml_tensor * si = ggml_cast(gc, ggml_scale_bias(gc, ggml_cast(gc, ggml_cont(gc, g_sel), GGML_TYPE_F32), 1.0f, (float) nsrc),
                                         GGML_TYPE_I32);                                                // [K, n]
            ggml_tensor * idx = ggml_concat(gc, wi, si, 0);                                              // [M0, n]
            if (MP > M0) {
                ggml_tensor * zi = ggml_cast(gc, ggml_scale_bias(gc, ggml_cast(gc, ggml_view_1d(gc, im.zidx, (MP - M0) * n, 0),
                                                                                GGML_TYPE_F32), 1.0f, (float) (nsrc + cap)), GGML_TYPE_I32);
                idx = ggml_concat(gc, idx, ggml_reshape_2d(gc, zi, MP - M0, n), 0);                      // [MP, n]
                Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, MP - M0, n), NEG_INF), 0);
            }
            Kall = ggml_reshape_3d(gc, rows_as(gc, src, ggml_reshape_1d(gc, ggml_cont(gc, idx), MP * n), true), DH, MP, n);
            M = MP;
        }
    }
    if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
        const int64_t pad = 256 - M % 256;
        // F16 has no CUDA pad: concat zeros gathered from the one zero row (F16 -> F16, exact) - the same zeros
        Kall = f16 ? ggml_concat(gc, Kall, ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * n, 0), true),
                                                           DH, pad, n), 1)
             : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, n)
                                                   : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);              // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);                       // [D, HC, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
               &f_pre, &f_post, &f_comb);
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));   // [D, n]

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    // vision: image positions select with exp_probs_b_vl.bias instead (model.py Gate: torch.where(image_mask, bias_vl, bias))
    ggml_tensor * bias_sel = im.BL(il, "exp_probs_b.bias");
    if (im.vision && im.i_img) {
        ggml_tensor * bt = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b.bias"), im.NE, 1);
        ggml_tensor * bv = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b_vl.bias"), im.NE, 1);
        ggml_tensor * m = ggml_reshape_2d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, n);
        bias_sel = ggml_add(gc, ggml_mul(gc, ggml_repeat(gc, ggml_sub(gc, bv, bt), sc), m), bt);
    }
    ggml_tensor * s0 = ggml_add(gc, sc, bias_sel);                                             // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    //   STRATA_RB_POOL=M     the bias only moves selections inside the exact top-M; every pick stays a top-M expert
    //                        (colibri's CACHE_ROUTE max-rank window, arXiv:2412.00099: PROTECT=2 POOL=12 there)
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    // PROTECT / POOL (and STRATA_RB_SMART=1) build them from rknob: P / M then change per request (set_route_select),
    // 0 = off, and P = M = 0 selects exactly as the flat bias.
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    ggml_tensor * xids = nullptr;   // the exact top-NU (route meter)
    if (smart_router() || rb_margin >= 0.0f) {
        const int kt = smart_router() ? kRK : (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr, * pool = nullptr;
        if (smart_router()) {
            xids = ggml_cont(gc, ggml_view_2d(gc, ti, im.NU, n, ti->nb[1], 0));
            // rknob column c = a one-hot over the kRK exact ranks + an offset: the P-th / M-th exact score, or +-1e30 for
            // rank 0 (protect nothing / pool everything).  mul + sum_rows, not mul_mat: one nonzero term, exact in F32.
            auto thr = [&](int c) {
                ggml_tensor * oh = ggml_view_2d(gc, im.rknob, kRK, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1]);
                ggml_tensor * off = ggml_view_2d(gc, im.rknob, 1, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1] + (size_t) kRK * 4);
                return ggml_repeat(gc, ggml_add(gc, ggml_sum_rows(gc, ggml_mul(gc, tv, oh)), off), like);
            };
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(0)), 1.0f, 1e-6f));   // s0 >= the P-th exact score
            pool = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(1)), 1.0f, 1e-6f));      // s0 >= the M-th
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        if (pool) unprot = ggml_mul(gc, unprot, pool);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
        if (pool) sel = ggml_add(gc, sel, ggml_scale_bias(gc, pool, 1e4f, -1e4f));   // outside the pool: -1e4 (pool = 1: exactly 0)
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = nullptr, * wts = nullptr;   // wts stays null when the fused op writes o_wts itself
    if (fuse_router(im) && !smart_router() && rb_margin < 0.0f && bias_sel == im.BL(il, "exp_probs_b.bias")) {
        ids = ggml_dsv4_route(gc, rl, bias_sel, im.ly[(size_t) il].rbias, b.cols2(im.o_wts, im.NU, n, col0), (int) im.NU,
                              G.norm_topk, (float) G.weights_scale);
    } else {
        ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
        wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
        if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
        wts = ggml_scale(gc, wts, (float) G.weights_scale);
    }

    // ---- shared expert (model.py Expert, clamped SwiGLU) - in its own graph under STRATA_SHEXP_SPLIT (n == 1)
    ggml_tensor * se = no_shexp ? nullptr : build_shexp_ops(im, gc, il, fn);   // no_shexp: shexp_launch computes it

    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    if (se) outs.push_back(ggml_cpy(gc, se, b.cols2(im.shexp, D, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    if (wts) outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    if (xids) {   // route meter: the exact top-NU and their normalized weights
        ggml_tensor * xw = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), xids), im.NU, n);
        xw = ggml_div(gc, xw, ggml_scale_bias(gc, ggml_sum_rows(gc, xw), 1.0f, 1e-20f));
        outs.push_back(ggml_cpy(gc, xids, b.cols2(im.o_xids, im.NU, n, col0)));
        outs.push_back(ggml_cpy(gc, xw, b.cols2(im.o_xwts, im.NU, n, col0)));
    }
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));
        }
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = fuse_router(im) ? ggml_dsv4_route(gc, rl, im.BL(il, "exp_probs_b.bias"), im.ly[(size_t) il].rbias, nullptr, 16, false, 1.0f)
                           : ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// predict for layers il and il + 1 on the same state (hc_pre once): predict_multi2, STRATA_D3_PF_EARLY
static ggml_cgraph * build_predict2(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, ggml_tensor ** out2, int64_t n) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    for (int j = 0; j < 2; ++j) {
        const int l = il + j;
        ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(l), b.rms_w(x, im.BL(l, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
        ggml_tensor * o = nullptr;
        if (fuse_router(im)) {
            o = ggml_dsv4_route(gc, rl, im.BL(l, "exp_probs_b.bias"), im.ly[(size_t) l].rbias, nullptr, 16, false, 1.0f);
        } else {
            ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(l, "exp_probs_b.bias")),
                                        ggml_reshape_2d(gc, im.ly[(size_t) l].rbias, im.NE, 1));
            o = ggml_argsort_top_k(gc, sc, 16);
        }
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
        *(j == 0 ? out : out2) = o;
    }
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    // the layers after next_il on the same state (speculative reads): their scores are stacked under next_il's so ONE
    // argsort gives all the rows and ONE readback fetches them ([16, 1 + ahead] I32)
    bool stacked = false;   // rows of other layers under next_il's: their biases differ, so no fused route
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        if (il >= (int) im.ly.size()) break;
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, ra)), im.BL(il, "exp_probs_b.bias")),
                                    ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
        stacked = true;
    }
    *pout = fuse_router(im) && !stacked
        ? ggml_dsv4_route(gc, rl, im.BL(next_il, "exp_probs_b.bias"), im.ly[(size_t) next_il].rbias, nullptr, 16, false, 1.0f)
        : ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }
bool Ds41Dense::vision() const { return p_->vision; }

void Ds41Dense::set_vision(const float * rows, const int64_t * pos, int n_rows) {
    Impl & im = *p_;
    im.vis_rows = rows;
    im.vis_pos = pos;
    im.n_vis = n_rows;
}

void Ds41Dense::clear_vision() {
    Impl & im = *p_;
    im.vis_rows = nullptr;
    im.vis_pos = nullptr;
    im.n_vis = 0;
}

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the buffer freed here
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    if (smart_router()) {
        im.o_xids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
        im.o_xwts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    }
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (auto & L : im.ly)
        if (L.idx_src) { L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC); L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC); }
    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_img = im.vision ? ggml_new_tensor_1d(c, GGML_TYPE_F32, NC) : nullptr;   // vision: the image-position mask
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, 2));
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    for (int e = 0; e < 2; ++e) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    dclear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    dsync_be(im.backend);
    // STRATA_KEEP_DEC_GRAPHS=1: a prompt batch parks the decode-width world (its columns, allocators and every cached
    // graph - ggml-cuda keeps their captured CUDA graphs by node address) and gets an empty one of its own; on the way
    // back down the parked world returns as it was, so the next answer does not rebuild ~1,000 graphs, run each kernel
    // by kernel once and re-capture it (round 90).  The same graphs on the same tensors: exact.
    static const bool keep = env_on("STRATA_KEEP_DEC_GRAPHS", false);
    if (keep && im.parked_dec.valid && want == im.parked_dec.NC) {
        dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the batch world freed here
        im.swap_world(im.parked_dec);
        im.free_world(im.parked_dec);
        return true;
    }
    if (keep && !im.parked_dec.valid && im.NC == im.S && want > im.S) {
        dsync();
        im.swap_world(im.parked_dec);   // this Impl is left with an empty world: nothing below frees the parked one
        im.parked_dec.valid = true;
    }
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.graphs.clear();
    dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the buffers freed below
    im.free_mgraphs();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
    if (want == im.S) allo_reserve(im.allo_dec);   // the decode width (a prompt batch's graphs need no head room)
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    return alloc_cols(im, want, im.err);
}

int Ds41Dense::columns() const { return (int) p_->NC; }

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (!load_weights(im, err)) return false;
    im.engram.set_hot_path(cfg.engram_hot);
    im.vision = cfg.vision;
    if (im.vision) {
        for (int64_t l = 0; l < im.nl; ++l) {
            if (!im.w.has("blk." + std::to_string(l) + ".exp_probs_b_vl.bias")) {
                err = "--vision requires exp_probs_b_vl.bias in every layer";
                return false;
            }
        }
    }
    im.engram.set_hot_path(cfg.engram_hot);
    // STRATA_ENGRAM_THREADS=N: the Engram readers (the disk's queue depth; default cfg.engram_threads = 32)
    const int engram_threads = std::getenv("STRATA_ENGRAM_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_ENGRAM_THREADS")))
                                                                     : cfg.engram_threads;
    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, engram_threads, err)) return false;

    // layers
    im.ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state
    if (cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    ggml_init_params sp = { ggml_tensor_overhead() * 512, nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.raw = T(kvt, { im.DH, im.RAW });
        L.raw_snap = T(kvt, { im.DH, im.RAW });
        L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
        if (L.kv_src) {
            // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
            static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
            const bool fc = cfg.kv_fp4 && !(only && std::string(only) == "idx");
            const bool fi = cfg.kv_fp4 && !(only && std::string(only) == "comp");
            L.comp = T(fc ? GGML_TYPE_NVFP4 : kvt, { im.DH, im.cmax[L.ratio] + 1 });
            L.idxk = T(fi ? GGML_TYPE_MXFP4 : kvt, { im.ID, im.cmax[L.ratio] + 1 });
            if (L.ratio == 2) {
                L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
            }
        }
    }
    if (smart_router()) im.rknob = T(GGML_TYPE_F32, { kRK + 1, 2 });
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    dclear(im.sbuf, 0);
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); dset(L.st_sc, v.data(), 0, v.size() * 4); }
    if (im.rknob) set_route_select(env_rank("STRATA_RB_PROTECT"), env_rank("STRATA_RB_POOL"));   // all-zero knobs would protect everything
    // inputs
    ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
    im.ictx = ggml_init(ipp);
    auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
    im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
    im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
    // default ON (CUDA) since 2026-10-18: 1017 8K prompt 239.8 -> 255.2 tok/s, +1.06 GiB min free VRAM, logits + tokens
    // identical (verify/s7/vram-f16.sh GATE A).  STRATA_ATTN_F16_GATHER=0 restores the F32 gather.
    im.f16_gather = !im.cpu && !(std::getenv("STRATA_ATTN_F16_GATHER") && std::getenv("STRATA_ATTN_F16_GATHER")[0] == '0');
    im.zidx = I(GGML_TYPE_I32, im.f16_gather ? 255 * 2048 : 1);   // ibuf is cleared to 0 below: all zeros
    if (im.f16_gather) im.zpad16 = I(GGML_TYPE_F16, im.DH, 1);
    im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
    if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
    dclear(im.ibuf, 0);
    {
        std::vector<float> v((size_t) im.neg_ramp->ne[0]);
        for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
        dset(im.neg_ramp, v.data(), 0, v.size() * 4);
        v.assign((size_t) im.blk_ramp->ne[0], 0);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
        dset(im.blk_ramp, v.data(), 0, v.size() * 4);
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    allo_reserve(im.allo_dec);
    if (!alloc_cols(im, im.NC, err)) return false;
    // the host mirrors are sized for the largest batch once (pinned registrations must not move)
    im.host_fn.resize((size_t) (D * NCM));
#ifdef DS41_HAVE_CUDA
    // pinned: the fn rows go device -> host -> the tier's device input every layer (pageable copies ran at ~2 GB/s)
    if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
    im.host_ids.resize((size_t) (im.NU * NCM));
    im.host_wts.resize((size_t) (im.NU * NCM));
    if (smart_router()) {
        im.host_xids.resize((size_t) (im.NU * NCM));
        im.host_xwts.resize((size_t) (im.NU * NCM));
    }
#ifdef DS41_HAVE_CUDA
    // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
    // pinned memory)
    if (!im.cpu) {
        ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
        ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
        if (!im.host_xids.empty()) {
            ggml_backend_cuda_register_host_buffer(im.host_xids.data(), im.host_xids.size() * 4);
            ggml_backend_cuda_register_host_buffer(im.host_xwts.data(), im.host_xwts.size() * 4);
        }
    }
#endif
    return true;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    dset(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

bool Ds41Dense::set_route_select(int protect, int pool) {
    Impl & im = *p_;
    if (!im.rknob) return false;
    protect = std::max(0, std::min(protect, (int) im.NU));
    pool = pool <= 0 ? 0 : std::max((int) im.NU, std::min(pool, kRK));   // a pool smaller than top-k would force outsiders in
    std::vector<float> k((size_t) (kRK + 1) * 2, 0.0f);
    if (protect > 0) k[(size_t) protect - 1] = 1.0f; else k[kRK] = 1e30f;                          // column 0: protect
    if (pool > 0) k[(size_t) (kRK + 1 + pool - 1)] = 1.0f; else k[(size_t) (2 * kRK + 1)] = -1e30f;  // column 1: pool
    dset(im.rknob, k.data(), 0, k.size() * 4);
    return true;
}

Ds41Dense::RouteMeter Ds41Dense::route_meter(bool reset) {
    Impl & im = *p_;
    RouteMeter m;
    m.on = !im.host_xids.empty();
    m.slots = im.rm_slots;
    m.swaps = im.rm_swaps;
    m.weight_kept = im.rm_rows > 0 ? im.rm_kept / (double) im.rm_rows : 1.0;
    if (reset) { im.rm_slots = im.rm_swaps = im.rm_rows = 0; im.rm_kept = 0; }
    return m;
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the buffer freed here
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    ++im.allo_pass.epoch;
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
// STRATA_PASS_CAP_MIN=N (default 0): never below N.  The attention / indexer graphs are cached per pass_cap, so each
// doubling of the context builds, runs directly once and captures ~360 new graphs (round 93: ~0.5 ms/token in the
// steady state); with a floor a chat stays on one set until the context passes N.  The extra rows are masked (as the
// rows between nvis and pass_cap always are), so only the flash-attn work split changes: equal to rounding.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    static const int64_t cap_min = std::getenv("STRATA_PASS_CAP_MIN") ? std::atoll(std::getenv("STRATA_PASS_CAP_MIN")) : 0;
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, std::max<int64_t>(im.CBS, cap_min))), im.cmax[r]);
    }
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
// STRATA_ENGRAM_SPLIT=1 (default off): a layer waits for ITS slot's rows only (Ds41Engram::wait_table) - layer 1 no
// longer waits for layer 14's half of a long prompt's reads, which go on under layers 1-13 (round 104's profile: a
// 2.4 s idle gap before layer 1 of a 16K prompt).  The same rows, uploaded per slot: exact.
static bool engram_ready(Ds41Dense::Impl & im, int slot = -1) {
    if (!im.eng_batch) return true;
    static const bool split = std::getenv("STRATA_ENGRAM_SPLIT") && std::getenv("STRATA_ENGRAM_SPLIT")[0] == '1';
    if (split && slot >= 0 && slot < 2 && (size_t) slot < im.eng_rows.size()) {
        if (!im.eng_up[slot]) {
            if (!im.engram.wait_table(im.eng_batch, slot, im.err)) { im.eng_batch.reset(); return false; }
            dset(im.i_eng[slot], im.eng_rows[(size_t) slot].data(), 0, im.eng_rows[(size_t) slot].size() * 4);
            im.eng_up[slot] = true;
        }
        bool all = true;
        for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e) all &= im.eng_up[e];
        if (!all) return true;   // the other slot's reads are still going
    }
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
        if (!im.eng_up[e]) dset(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    im.eng_up[0] = im.eng_up[1] = false;
    return true;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // embeddings (F16 rows on the mmap)
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
    // vision: overwrite the image span positions with the encoder's rows (model.py merge_image_embeddings) and mark
    // them, so the router picks exp_probs_b_vl there and Engram skips them.  vis_pos is ascending, the pass contiguous.
    std::vector<float> img;
    if (im.vision) {
        img.assign((size_t) n, 0.0f);
        for (int v = 0; v < im.n_vis; ++v) {
            const int64_t p = im.vis_pos[v];
            if (p < pos0 || p >= (int64_t) pos0 + n) continue;
            const int t = (int) (p - pos0);
            std::memcpy(im.host_emb.data() + (size_t) t * D, im.vis_rows + (size_t) v * D, (size_t) D * 4);
            img[(size_t) t] = 1.0f;
        }
        dset(im.i_img, img.data(), 0, (size_t) n * 4);
    }
    dset(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    dset(im.i_pos, pos.data(), 0, (size_t) n * 4);
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    dset(im.i_widx, widx.data(), 0, widx.size() * 4);
    dset(im.i_wmask, wm.data(), 0, wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        dset(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        dset(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        dset(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        dset(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            dset(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        dset(im.i_sidx, si.data(), 0, si.size() * 4);
        dset(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) {
            if (!img.empty() && img[(size_t) t] > 0.0f) im.engram.set_dead(pos0 + t);   // an image span: n-grams never span it
            else im.engram.set_token(pos0 + t, tids[t]);
        }
        // STRATA_ENGRAM_ASYNC=1: decode / verify passes take the async path below too, so their reads overlap the
        // embeddings upload, the init graph and layer 0 (attention + its expert tier) instead of stalling the pass
        // here (~2.7 ms per verify pass, round 52).  Every builder that reads i_eng waits first (engram_ready).
        static const bool eng_async = std::getenv("STRATA_ENGRAM_ASYNC") && std::getenv("STRATA_ENGRAM_ASYNC")[0] == '1';
        if (n <= 64 && !eng_async) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                dset(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            im.eng_up[0] = im.eng_up[1] = false;
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 0, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 0, 0, 1 }, build_init(im, im.gctx, 1)).first;
        gf = it->second;
    } else if (pass_graphs_on(n)) {
        gf = pass_graph(im, { -1, 0, 0, n }, [&](ggml_context * c, ggml_tensor **) { return build_init(im, c, n); }).gf;
        cached = true;
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_init(im, gc, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    return ok;
}

// ================================================================== the layer-major prompt batch

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_init(im, gc, sn, c0);
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const size_t si = (size_t) (col0 / im.S);
    if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
    im.pass_cap[0] = im.span_cap[si][0];
    im.pass_cap[1] = im.span_cap[si][1];
    im.pass_cap[2] = im.span_cap[si][2];
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        dset(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        dset(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        dset(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        dset(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;
}

// route meter: columns col0..col0+n-1 of the host ids against the exact top-k read beside them
static void meter_add(Ds41Dense::Impl & im, int64_t col0, int64_t n) {
    const int64_t K = im.NU;
    for (int64_t t = col0; t < col0 + n; ++t) {
        const int32_t * c = im.host_ids.data() + t * K, * x = im.host_xids.data() + t * K;
        const float * w = im.host_xwts.data() + t * K;
        for (int64_t j = 0; j < K; ++j) {
            bool kept = false;
            for (int64_t i = 0; i < K && !kept; ++i) kept = c[i] == x[j];
            if (kept) im.rm_kept += w[j]; else ++im.rm_swaps;
        }
        im.rm_slots += K;
        ++im.rm_rows;
    }
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;   // wait the pass's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_attn(im, gc, il, n, col0);
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    if (im.o_xids) {
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    }
    dsync_be(im.backend);
    if (im.o_xids) meter_add(im, col0, n);
    lap(4);
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    dset(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

bool Ds41Dense::wait_event(void * cuda_event) {
    Impl & im = *p_;
    if (!cuda_event || im.cpu) return true;
    // a ggml_backend_event around a foreign cudaEvent_t: the CUDA backend's event_wait is cudaStreamWaitEvent on its
    // own stream with event->context as the event
    ggml_backend_event ev { ggml_backend_get_device(im.backend), cuda_event };
    ggml_backend_event_wait(im.backend, &ev);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }

bool Ds41Dense::shexp_launch(int il) {
    Impl & im = *p_;
    if (!im.shexp_pending) return true;   // the last attention graph computed the shared expert itself
    im.shexp_pending = false;
    const Impl::GKey key { il, 7, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) it = im.graphs.emplace(key, build_shexp(im, im.gctx, il)).first;
    return im.compute(it->second, true, false);   // no host wait (STRATA_DENSE_ASYNC): finish (same stream) reads shexp after it
}

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev,
                            bool defer_shexp) {
    Impl & im = *p_;
    const int64_t n = im.n;
    const bool split = defer_shexp && shexp_split() && n == 1;   // the shared expert then comes from shexp_launch
    im.shexp_pending = false;
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    bool cached = n == 1;
    if (cached) {
        const Impl::GKey key { il, split ? 8 : 1, cap, 1 };   // the split graph is its own cache entry
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1, 0, split)).first;
        gf = it->second;
        im.shexp_pending = split;
    } else if (pass_graphs_on(n)) {
        gf = pass_graph(im, { il, 1, cap, n }, [&](ggml_context * c, ggml_tensor **) { return build_attn(im, c, il, n); }).gf;
        cached = true;
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n);
    }
    const bool ok = im.compute(gf, cached, false);   // STRATA_DENSE_ASYNC: the readbacks' wait below is the only one
    if (gc) ggml_free(gc);
    if (!ok) return false;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    dsync_be(im.backend);
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true, false)) return false;
    int32_t buf[16];
    dread(im, im.predict_out[il], buf, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true, false)) return false;
    // argsort_top_k is a VIEW of the full argsort: row t starts at t * nb[1] (NE ids apart).  One read of the span that
    // covers every row's first 16, then each row's 16.  (Until 2026-10-21 this read 16 * n contiguous ids: rows 1.. were
    // row 0's ranks 16.., which the top-k kernel never writes - stale ids, so the verify pass prefetched on row 0 alone.)
    const size_t rs = o->nb[1] / sizeof(int32_t);
    im.pred_span.resize(rs * (size_t) (n - 1) + 16);
    dread(im, o, im.pred_span.data(), im.pred_span.size() * sizeof(int32_t));
    for (int64_t t = 0; t < n; ++t)
        std::copy(im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t), im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t + 16),
                  top_ids + 16 * t);
    return true;
}

bool Ds41Dense::predict_multi2(int il, int * top_ids, int * next_ids) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1 || il + 1 >= (int) im.ly.size()) return false;
    // the two outputs live in predict_out_n under keys no layer uses (-(il + 1) and -(il + 1) - 1000), so they follow
    // that map through the world swaps and clears like predict_multi's own
    const Impl::GKey key { il, 11, 0, (int) n };
    const std::pair<int, int> k1 { -(il + 1), (int) n }, k2 { -(il + 1) - 1000, (int) n };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o1 = nullptr, * o2 = nullptr;
        ggml_cgraph * gf = build_predict2(im, im.gctx, il, &o1, &o2, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[k1] = o1;
        im.predict_out_n[k2] = o2;
    }
    if (!im.compute(it->second, true, false)) return false;
    int * dst[2] = { top_ids, next_ids };
    for (int j = 0; j < 2; ++j) {
        ggml_tensor * o = im.predict_out_n[j == 0 ? k1 : k2];
        const size_t rs = o->nb[1] / sizeof(int32_t);   // a VIEW of the full argsort (or contiguous [16, n]): see predict_multi
        im.pred_span.resize(rs * (size_t) (n - 1) + 16);
        dread(im, o, im.pred_span.data(), im.pred_span.size() * sizeof(int32_t));
        for (int64_t t = 0; t < n; ++t)
            std::copy(im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t), im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t + 16),
                      dst[j] + 16 * t);
    }
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    (void) il;
    const int64_t n = im.n;
    if (routed_sum) dset(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 2, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 2, 0, 1 }, build_finish(im, im.gctx, 1)).first;
        gf = it->second;
    } else if (pass_graphs_on(n)) {
        gf = pass_graph(im, { -1, 2, 0, n }, [&](ggml_context * c, ggml_tensor **) { return build_finish(im, c, n); }).gf;
        cached = true;
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_finish(im, gc, n);
    }
    const bool ok = im.compute(gf, cached, false);   // STRATA_DENSE_ASYNC: the next graph / read waits for it
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    (void) il;
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) dset(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true, false)) return false;
    int32_t buf[16];
    dread(im, im.fpredict_out[next_il], buf, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_launch(int il, const float * routed_sum, int next_il) {
    Impl & im = *p_;
    (void) il;
    if (im.n != 1 || im.cpu || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (!im.pred_host) {
#ifdef DS41_HAVE_CUDA
        im.pred_host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 4096);
#else
        im.err = "finish_predict_launch needs the CUDA build";
        return false;
#endif
        if (!im.pred_host_buf) { im.err = "pinned prediction buffer"; return false; }
        im.pred_host = (int32_t *) ggml_backend_buffer_get_base(im.pred_host_buf);
        im.pred_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        if (!im.pred_ev) { im.err = "prediction event"; return false; }
    }
    if (routed_sum) dset(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };   // the same cached graph as finish_predict
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true, false)) return false;
    ggml_backend_tensor_get_async(im.backend, im.fpredict_out[next_il], im.pred_host, 0, 16 * sizeof(int32_t));
    ggml_backend_event_record(im.pred_ev, im.backend);
    im.pred_n = 16;
    return true;
}

void * Ds41Dense::predict_event() {
    Impl & im = *p_;
    return im.pred_ev && im.pred_n > 0 ? im.pred_ev->context : nullptr;   // the CUDA backend's event: a cudaEvent_t
}

bool Ds41Dense::predict_take(int * top_ids, int n_top) {   // after the caller has seen predict_event() complete
    Impl & im = *p_;
    if (im.pred_n < n_top) { im.err = "no prediction pending"; return false; }
    for (int i = 0; i < n_top; ++i) top_ids[i] = im.pred_host[i];
    im.pred_n = 0;
    return true;
}

bool Ds41Dense::predict_wait(int * top_ids, int n_top) {
    Impl & im = *p_;
    if (!im.pred_ev || im.pred_n < n_top) { im.err = "no prediction pending"; return false; }
    ggml_backend_event_synchronize(im.pred_ev);
    for (int i = 0; i < n_top; ++i) top_ids[i] = im.pred_host[i];
    im.pred_n = 0;
    return true;
}

bool Ds41Dense::attn_router_launch(int il) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n != 1 && !pass_graphs_on(n)) { im.err = "attn_router_launch: a multi-row pass needs STRATA_PASS_GRAPHS"; return false; }
    im.shexp_pending = false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    if (n > 1)   // the same cached pass graph attn_router uses
        return im.compute(pass_graph(im, { il, 1, cap, n }, [&](ggml_context * c, ggml_tensor **) { return build_attn(im, c, il, n); }).gf,
                          true, false);
    const Impl::GKey key { il, 1, cap, 1 };   // the same cached graph attn_router uses
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1)).first;
    return im.compute(it->second, true, false);
}

bool Ds41Dense::predict_multi_launch(int il) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1 || im.cpu) return false;
    if (!im.pm_host) {
#ifdef DS41_HAVE_CUDA
        im.pm_host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 65536);
#else
        im.err = "predict_multi_launch needs the CUDA build";
        return false;
#endif
        if (!im.pm_host_buf) { im.err = "pinned prediction buffer"; return false; }
        im.pm_host = (int32_t *) ggml_backend_buffer_get_base(im.pm_host_buf);
        im.pm_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        if (!im.pm_ev) { im.err = "prediction event"; return false; }
    }
    // the graph predict / predict_multi use for this width
    const Impl::GKey key { il, 3, 0, (int) n };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        if (n == 1) im.predict_out[il] = o; else im.predict_out_n[{ il, (int) n }] = o;
    }
    ggml_tensor * o = n == 1 ? im.predict_out[il] : im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true, false)) return false;
    const int64_t rs = (int64_t) (o->nb[1] / sizeof(int32_t));   // a VIEW of the full argsort (see predict_multi)
    const size_t span = (size_t) (rs * (n - 1) + 16);
    if (span * sizeof(int32_t) > 65536) { im.err = "predict_multi_launch: pass too wide"; return false; }
    ggml_backend_tensor_get_async(im.backend, o, im.pm_host, 0, span * sizeof(int32_t));
    ggml_backend_event_record(im.pm_ev, im.backend);
    im.pm_rows = n;
    im.pm_rs = rs;
    return true;
}

bool Ds41Dense::predict_multi_wait(int * top_ids) {
    Impl & im = *p_;
    if (!im.pm_ev || im.pm_rows < 1) { im.err = "no prediction pending"; return false; }
    ggml_backend_event_synchronize(im.pm_ev);
    for (int64_t t = 0; t < im.pm_rows; ++t)
        std::copy(im.pm_host + im.pm_rs * t, im.pm_host + im.pm_rs * t + 16, top_ids + 16 * t);
    im.pm_rows = 0;
    return true;
}

bool Ds41Dense::attn_router_collect(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    (void) il;
    const int64_t n = im.n;
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {   // as attn_router: the smart router's exact picks, for the route meter
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    dsync_be(im.backend);
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) dset(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao);
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true, false)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    dread(im, ao[0], buf.data(), buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    if (pass_graphs_on(nrows) && row0 < 16) {   // the head of a decode step / verify pass (kind 9, il = row0)
        Impl::MGraph & g = pass_graph(im, { row0, 9, 0, nrows }, [&](ggml_context * c, ggml_tensor ** lg) {
            return build_head(im, c, row0, nrows, lg); });
        if (!im.compute(g.gf, true)) return false;
        dget(g.out, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
        return true;
    }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    const bool ok = im.compute(gf, false);
    if (ok) dget(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    dget(im.x_state, x.data(), 0, x.size() * 4);
    dget(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    dget(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

std::string Ds41Dense::vram_ledger() const {
    const Impl & im = *p_;
    auto gib = [](size_t b) { return (double) b / 1073741824.0; };
    auto dev = [](ggml_backend_buffer_t b) { return b && !ggml_backend_buffer_is_host(b) ? ggml_backend_buffer_get_size(b) : (size_t) 0; };
    size_t w = 0;
    for (auto b : im.w.bufs) w += dev(b);
    const size_t gd = im.allo_dec.a ? ggml_gallocr_get_buffer_size(im.allo_dec.a, 0) : 0;
    const size_t gp = im.allo_pass.a ? ggml_gallocr_get_buffer_size(im.allo_pass.a, 0) : 0;
    char line[320];
    std::snprintf(line, sizeof line, "dense weights %.2f, KV state %.2f, columns %.2f (NC %lld), inputs %.2f, graphs dec %.2f + "
                  "pass %.2f = %.2f GiB", gib(w), gib(dev(im.sbuf)), gib(dev(im.nbuf)), (long long) im.NC, gib(dev(im.ibuf)),
                  gib(gd), gib(gp), gib(w + dev(im.sbuf) + dev(im.nbuf) + dev(im.ibuf) + gd + gp));
    return line;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    for (auto & L : im.ly) {
        dcopy(L.raw, L.raw_snap);
        if (L.st_kv) { dcopy(L.st_kv, L.st_kv_snap); dcopy(L.st_sc, L.st_sc_snap); }
    }
    dsync_be(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        dcopy(L.raw_snap, L.raw);
        if (L.st_kv) { dcopy(L.st_kv_snap, L.st_kv); dcopy(L.st_sc_snap, L.st_sc); }
    }
    dsync_be(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

// The snapshot rings + rows [0, positions] of the position-indexed caches (and their spare rows), copied to host
// memory: what park() and save_conv() keep; positions 0 = the rings alone (a snapshot of the SAME conversation, whose
// rows are still in place - STRATA_SERVE_SNAP_STACK).  The caller syncs the backend first.
static void collect_conv_rows(Ds41Dense::Impl & im, int64_t positions, std::vector<Ds41Dense::Impl::ParkedRows> & out) {
    out.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Ds41Dense::Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        dget(t, r.b.data(), off, n);
        out.push_back(std::move(r));
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t || positions <= 0) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
}

bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    dsync_be(im.backend);
    collect_conv_rows(im, positions, im.parked);
    im.parked_pos = im.snap_pos;
    return true;
}

struct Ds41Dense::SavedConv {
    std::vector<Impl::ParkedRows> rows;
    int snap_pos = -1;
    size_t bytes = 0;
};

std::shared_ptr<Ds41Dense::SavedConv> Ds41Dense::save_conv(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 0) return nullptr;   // 0: the snapshot rings alone
    dsync_be(im.backend);
    auto s = std::make_shared<SavedConv>();
    collect_conv_rows(im, positions, s->rows);
    s->snap_pos = im.snap_pos;
    for (const auto & r : s->rows) s->bytes += r.b.size();
    return s;
}

bool Ds41Dense::load_conv(const SavedConv & s) {
    Impl & im = *p_;
    if (s.snap_pos < 0) return false;
    dsync_be(im.backend);
    for (const auto & r : s.rows) dset(r.t, r.b.data(), r.off, r.b.size());
    dsync_be(im.backend);
    im.snap_pos = s.snap_pos;
    return true;
}

size_t Ds41Dense::saved_bytes(const SavedConv & s) { return s.bytes; }

void Ds41Dense::set_history(const int * ids, int64_t n, int64_t pos0) {
    Impl & im = *p_;
    if (!im.engram.enabled() || n <= 0) return;
    if (!engram_ready(im)) return;   // a still-pending batch first (none between requests)
    for (int64_t i = 0; i < n; ++i) im.engram.set_token(pos0 + i, ids[i]);
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    dsync_be(im.backend);
    for (auto & r : im.parked) dset(r.t, r.b.data(), r.off, r.b.size());
    dsync_be(im.backend);
    im.snap_pos = im.parked_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!L.idx_src || !L.ratio) return true;
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    dget(L.sel, s.data(), 0, s.size() * 4);
    dget(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}
