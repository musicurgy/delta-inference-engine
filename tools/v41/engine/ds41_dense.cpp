// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"
#include "ds41_embd.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"
#include "ds41_env.hpp"
#include "ds41_fatal.hpp"
#include "ds41_unmap.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)
#ifdef DS41_HAVE_X3B_GATE
#include "ds41_x3b_gate.hpp"   // STRATA_X3B: the device-side gate
#endif
namespace strata { namespace v41 { struct X3bGate; } }

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
#include <tuple>

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

bool env_on(const char * k, const char * old_k, bool dflt) {   // k, else its older name old_k (ds41_env.hpp)
    const char * e = ds41_env(k, old_k);
    return e ? std::atoi(e) != 0 : dflt;
}

// STRATA_DENSE_ASYNC=1 (default off): Impl::compute(gf, cached, wait = false) launches a cached graph WITHOUT the host
// wait ggml_backend_graph_compute ends with (every graph launch was a host sync: finish, the shared expert, ...).  The
// backend stream may then still be running when the host goes on, and ggml_backend_tensor_get / _set / _copy /
// buffer_clear run on cudaStreamPerThread, which is NOT ordered after it - so every such call in this file goes through
// these wrappers, which wait for a pending launch first.  A host wait on the stream (dsync_be) clears it.
ggml_backend_t g_async_be = nullptr;
void x3_flush_pending();   // STRATA_X3: launch a deferred begin / finish graph now (defined after Ds41Dense::Impl)
void x3b_sync_hit();       // STRATA_X3B: a host stream sync while the gate is armed (defined after Ds41Dense::Impl)
bool g_x3b_gated = false;  // STRATA_X3B: the dense stream waits behind an armed gate the host has not released yet
void dsync() {
    if (g_x3b_gated) x3b_sync_hit();   // (would wait forever: the gate opens only after the host's next tier call)
    x3_flush_pending();
    if (g_async_be) { ggml_backend_synchronize(g_async_be); g_async_be = nullptr; }
}
void dsync_be(ggml_backend_t be) {
    if (g_x3b_gated) x3b_sync_hit();
    x3_flush_pending();
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
    bool failed = false;   // the last alloc failed: the gallocr holds a plan without a buffer (see alloc)
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
        // A failed allocation leaves the gallocr PLANNED for that graph with NO buffer (ggml_gallocr_reserve_n frees
        // the old buffer, then the larger one does not fit).  The next alloc_graph of the same graph then finds the plan
        // current, skips the reserve and places the tensors in the null buffer - a segfault (2026-10-20: the oom_hook
        // retry on a 768K prompt).  So after a failure re-plan explicitly; the freed buffer also makes every graph
        // allocated before it stale (the epoch).  (upstream 828353c)
        if (failed && !ggml_gallocr_reserve(a, g)) { ++epoch; return false; }
        if (!ggml_gallocr_alloc_graph(a, g)) { failed = true; ++epoch; return false; }
        failed = false;
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
// (measured: ~590 such re-captures in one 400-token answer as the attention's pass_cap doubled).  Same graphs, same
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
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); ds41_fatal_exit(); }
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
    bool hap_alias = false;               // hap is a view of x_state (set_columns' hap_in_state: wide prompt batches)
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
        ggml_tensor * tring = nullptr;    // [DH, TR]        STRATA_SNAP_TAIL: the window rows again, slot = pos % TR (never read by a graph)
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    // ---- STRATA_SNAP_TAIL (cfg.snap_tail; 0 = off and none of this exists): a longer window tail per snapshot.  A
    // resume at an EVEN position rs (a fresh ratio-2 compressor group) needs, besides the position-indexed caches, only
    // the window rows of [rs - SWA + 1, rs) in every layer; the ring keeps the last RAW.  tring (TR = RAW + tail rows)
    // gets the same rows (one set_rows a layer), snapshot() copies it to host memory (SnapTail) and snap_tail_resume()
    // writes the rows of an earlier rs back into the ring.  ttag mirrors which position each tring slot holds (upload
    // order = write order).  The layers past last_kv hold exact rows only where a read ran them (enc-skip); the serve
    // loop picks a resume point that re-runs those (tail_point in ds41_generate.cpp).
    int64_t TR = 0;
    std::vector<int64_t> ttag;
    std::shared_ptr<Ds41Dense::SnapTail> snap_tail, parked_tail;
    // ---- STRATA_SHARED_PREFIX (armed by the serve loop only, else none of this runs): every layer's window
    // ring and ratio-2 state ring as they stand right after that layer's layer-major span ENDING at position pc_pos (a
    // sub-batch boundary of the read), copied to host memory - the rings a snapshot taken at pc_pos would hold.  The read
    // goes on past pc_pos; take_prefix_capture() adds the position-indexed rows afterwards (they stay in place).
    struct PrefixRings { std::vector<uint8_t> raw, st_kv, st_sc; bool got = false; };
    int64_t pc_pos = -1;
    std::vector<PrefixRings> pc;
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
    ggml_tensor * i_wmask_hi = nullptr;   // STRATA_ENC_BOUND only: the mask of the layers past the last kv source
    int64_t win_floor = -1;               // ... whose windows stop at this position (set_window_floor)
    // The floor belongs to the state it describes (0.94 fix): the snapshot keeps the floor it was taken with, and so do
    // the park and every SavedConv; restore() / unpark() / load_conv() bring it back with the rows.
    int64_t snap_floor = -1, parked_floor = -1;
    int last_kv = -1;                     // the last kv source layer
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_tslot = nullptr;      // I32 [min(NC, TR)]   STRATA_SNAP_TAIL: tring slots of the pass's last rows
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
    // A sub-batch whose graph does not fit is run as narrower spans (attn_router_span): span_max is the width that
    // fitted, kept for the rest of the prompt (release_pass resets it); 0 = the full sub-batch.  alloc_failed: the last
    // compute() failed in the graph allocation (not in the compute) - the only failure a narrower span can cure.
    int64_t span_max = 0;
    bool alloc_failed = false;
    // The indexer's selection over a long context (build_attn): its [visible compressed rows x queries] score / mask
    // tensors are built for at most idx_tile elements at a time (query chunks); 0 = one piece.  Default 2^28 = a 512K
    // context's 512-query sub-batch, so every context up to 512K builds the very same graph as before.
    int64_t idx_tile = 0;
    std::map<std::array<int64_t, 3>, size_t> graph_bytes_memo;   // span_graph_bytes: (cap 1, cap 2, n) -> bytes
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    // STRATA_PINNED_STAGE (default on; =0 off): a pinned host buffer for a verify pass's logits (kLogitsPinRows rows,
    // ~8 MB), allocated once at init.  The caller reads the pass's logits into it (logits_host) instead of a pageable
    // vector, so the ~3 MB a pass is one DMA, not the driver's chunked pageable copy.  Same bytes.
    ggml_backend_buffer_t lpin = nullptr;
    float * lpin_p = nullptr;
    size_t lpin_n = 0;
    // the attention-sink watch (Ds41Dense::sink_watch): [2][kSinkRows][HC] floats (slot 0 = the tap layer, slot 1 =
    // after the last layer), pinned; sink_mid_n = rows the tap layer captured in this pass (begin / begin_batch clear it)
    std::vector<float> sink_host;
    int sink_mid_n = 0, sink_n = 0, sink_row0 = 0;
    long sink_seq = 0;
    bool fn_host = true;                  // attn_router_span copies fn to host_fn (set_batch_fn_host; wide batches: off)
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_xids;       // route meter: o_xids / o_xwts readbacks, and the counters since route_meter(true)
    std::vector<float> host_xwts;
    long long rm_slots = 0, rm_swaps = 0, rm_rows = 0;
    double rm_kept = 0;
    const uint8_t * embd = nullptr;       // token_embd rows on the mmap (embd_type: F16 ours, Q5_K ISTA's)
    std::shared_ptr<const std::vector<uint8_t>> embd_copy;   // STRATA_FILE_TIER_UNMAP: the rows copied out (embd points here)
    int embd_type = GGML_TYPE_F16;
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
    // STRATA_ENGRAM_AHEAD (default on, =0 off; upstream 1510ca5a, Basalt's idea): the NEXT prompt chunk's reads (engram_next), started
    // into eng_rows once this pass's rows are on the card, so they run beside this chunk's layers 1.. and not only layer 0
    std::shared_ptr<Ds41Engram::Batch> eng_ahead;
    int64_t eng_ahead_pos0 = -1, eng_next_pos0 = -1;
    int eng_ahead_n = 0;
    std::vector<int> eng_next_tids;
    std::vector<int> eng_ahead_tids;      // the tokens the pending read-ahead hashed (adopted only by a pass with these)
    // STRATA_ENGRAM_READAHEAD (default off; Abyss 77d4541's row store): two row buffers - the pass's batch reads into *eng_cur,
    // the next chunk's read-ahead (started by engram_next, queued behind this pass's reads) into the other one; a taken
    // read-ahead makes its buffer the current one.  eng_n: the pass's columns (a read-ahead taken by a shorter pass at the
    // same place covers more); eng_upc[slot]: columns [0, eng_upc) of i_eng[slot] already uploaded (spans wait per block).
    std::vector<std::vector<float>> eng_rows2;
    std::vector<std::vector<float>> * eng_cur = &eng_rows;
    std::vector<std::vector<float>> * eng_ahead_buf = &eng_rows;
    int eng_n = 0;
    int eng_upc[2] = { 0, 0 };

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
    /// STRATA_X3: a begin (1) or finish (2) graph not launched yet - the next dense graph launch carries it in front of
    /// its own nodes as ONE graph (x3_take), or it is launched alone first (x3_flush).  x3_n: the pass width it is for.
    int x3_pend = 0;
    int64_t x3_n = 0;
    bool x3_fail = false;   ///< a pending graph launched from dsync() failed: the next compute() returns false
    // STRATA_PF_OVERLAP: finish_predict's readback lands in pinned memory behind an event, so the next layer's attention
    // can be launched before the host waits for the prediction (the prefetch DMAs then overlap that attention)
    ggml_backend_buffer_t pred_host_buf = nullptr;
    int32_t * pred_host = nullptr;
    ggml_backend_event_t pred_ev = nullptr;
    int pred_n = 0;
    // STRATA_ZC_PRED: the prediction graphs' ids land in pinned host memory written by a kernel (zc_pred_tensor):
    // F32 [16, n] tensors at fixed addresses, one per (layer, width, which), kept across graph rebuilds
    ggml_context * zc_ctx = nullptr;
    ggml_backend_buffer_t zc_buf = nullptr;
    size_t zc_off = 0;
    std::map<std::tuple<int, int, int>, ggml_tensor *> zc_map;
    // STRATA_D3_PF_ASYNC: predict_multi_launch's readback (pinned, behind pm_ev); pm_rows rows pending, pm_rs ids apart
    ggml_backend_buffer_t pm_host_buf = nullptr;
    int32_t * pm_host = nullptr;
    ggml_backend_event_t pm_ev = nullptr;
    int64_t pm_rows = 0, pm_rs = 0;
    const float * pm_zc = nullptr;   ///< STRATA_ZC_PRED: the pending prediction is this zero-copy host tensor's floats
    // STRATA_X3B + STRATA_D3_SPEC: the spec prediction x3b_prelaunch queued behind the gate (x3b_spec_wait collects it;
    // xs_zc as pm_zc: STRATA_ZC_PRED's host floats, read after xs_ev)
    ggml_backend_buffer_t xs_host_buf = nullptr;
    int32_t * xs_host = nullptr;
    ggml_backend_event_t xs_ev = nullptr;
    int64_t xs_rows = 0, xs_rs = 0;
    const float * xs_zc = nullptr;
    // STRATA_X3B: the gate (created on first use), whether it is armed and not yet released, a failure seen (a spin
    // gate's timeout, a host sync while armed), collect_issue's readback event, and the "next start" timing mark
    strata::v41::X3bGate * x3b_gate = nullptr;
    bool x3b_gated = false, x3b_fail = false, x3b_dead = false;
    long x3b_timeouts = 0;
    ggml_backend_event_t rb_ev = nullptr;
    int x3b_go_pending = -1;
    int x3b_arm_layer = -1;   ///< the layer whose gate is armed (x3b_release checks it: one arm, one release)
    /// launches per graph uid (a uid is stamped at each allocation) and the last one's time - x3b_prelaunch launches only
    /// graphs ggml-cuda REPLAYS (warmup done, captured, instantiated, not evicted): see x3b_replays
    struct X3bUse { int n = 0; double t = 0; };
    std::map<uint64_t, X3bUse> x3b_use;
    long x3b_declined_cold = 0;
    /// layers declined only because the X3-merged prediction graph (finish + prediction) was not built or not replaying
    /// - every other graph ready: the graph form the normal path does not launch.  Before 11-17 that was every layer
    /// under X3 + file-tier D3_SPEC (the spec prediction took the pending finish, so {l + 1, kX3Pred + 1} was never
    /// built); x3b_prelaunch now queues the spec prediction itself, so this stays at the warm-up's few
    long x3b_declined_merged = 0;
    long x3b_declined_engram = 0;   ///< layers not pre-launched because they are Engram layers (see x3b_prelaunch)
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs
    // STRATA_KEEP_DEC_GRAPHS=1 (--serve, elastic columns): the decode-width columns and every graph built on them,
    // parked by set_columns() while a prompt batch runs on its own wider columns, and brought back after it
    struct World {
        bool valid = false;
        ggml_context * nctx = nullptr;
        ggml_backend_buffer_t nbuf = nullptr;
        int64_t NC = 0;
        ggml_tensor * x_state = nullptr, * pre_state = nullptr, * hap = nullptr, * post_f = nullptr, * comb_f = nullptr;
        bool hap_alias = false;
        ggml_tensor * shexp = nullptr, * o_fn = nullptr, * o_ids = nullptr, * o_wts = nullptr, * o_xids = nullptr;
        ggml_tensor * o_xwts = nullptr, * routed = nullptr, * cand = nullptr;
        ggml_tensor * i_emb = nullptr, * i_pos = nullptr, * i_img = nullptr, * i_widx = nullptr, * i_wmask = nullptr, * i_wmask_hi = nullptr;
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
        std::swap(hap_alias, o.hap_alias);
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
        std::swap(i_wmask_hi, o.i_wmask_hi);
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
        // STRATA_ENGRAM_READAHEAD: no read may still write into the row buffers freed with this object
        if (std::getenv("STRATA_ENGRAM_READAHEAD") && std::atoi(std::getenv("STRATA_ENGRAM_READAHEAD")) > 0) {
            std::string e;
            if (eng_ahead) { Ds41Engram::cancel(eng_ahead); engram.wait(eng_ahead, e); }
            if (eng_batch) { Ds41Engram::cancel(eng_batch); engram.wait(eng_batch, e); }
        }
#ifdef DS41_HAVE_X3B_GATE
        if (x3b_gate) { x3b_gate_destroy(x3b_gate); x3b_gate = nullptr; }   // (opens an armed gate first)
        g_x3b_gated = false;
#endif
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
        if (zc_buf) ggml_backend_buffer_free(zc_buf);
        if (zc_ctx) ggml_free(zc_ctx);
        if (lpin) ggml_backend_buffer_free(lpin);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // Engram's q_norm * k_norm: model.py's two RMSNorm weights scale the same dot product, so the graph multiplies them
    // once.  llama.cpp mainline's GGUFs (Unsloth's) store that product already, as engram_gate.
    ggml_tensor * engram_wqk(ggml_context * gc, int64_t il) const {
        const std::string g = "blk." + std::to_string(il) + ".engram_gate.weight";
        if (w.has(g)) return w.get(g);
        return ggml_mul(gc, BL(il, "engram_q_norm.weight"), BL(il, "engram_k_norm.weight"));
    }
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
    // the llama.cpp convention (ISTA-DASLab's GGUFs): its Engram names map to ours (ds41_canonical_name), its 3-D
    // attn_output_a [OGD, OL, OG] is our 2-D [OGD, OG*OL] (the same bytes), and its BF16 tensors are converted at load
    // to the type our file has there (fp32 router / Engram norms, F16 compressor / indexer) - BF16 is not an operand
    // type of every op the graph runs on them.  Quantized matrices (Q5_K ...) stay as they are: ggml multiplies them.
    const bool lc = ds41_is_llamacpp_gguf(model);
    auto canon = [&](const TensorInfo & ti) { return lc ? ds41_canonical_name(ti.name) : ti.name; };
    auto lc_type = [&](const TensorInfo & ti) -> int {   // -1 = keep the file's
        if (!lc || (ggml_type) ti.type != GGML_TYPE_BF16) return -1;
        const std::string n = canon(ti);
        auto ends = [&](const char * x) { const size_t L = std::strlen(x); return n.size() >= L && n.compare(n.size() - L, L, x) == 0; };
        return ends("ffn_gate_inp.weight") || ends("engram_q_norm.weight") || ends("engram_k_norm.weight") ||
               ends("engram_gate.weight") ? GGML_TYPE_F32 : GGML_TYPE_F16;
    };
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        int nd = (int) ti.shape.size();
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        const std::string cn = canon(ti);
        static const std::string oa = "attn_output_a.weight";
        if (lc && nd == 3 && cn.size() >= oa.size() && cn.compare(cn.size() - oa.size(), oa.size(), oa) == 0) {
            ne[1] *= ne[2];
            ne[2] = 1;
            nd = 2;
        }
        if (type < 0) type = lc_type(ti);
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), nd, ne);
        ggml_set_name(t, cn.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry
    // engram_embd: a DwarfStar single-file GGUF's inline Engram tables (read from --engram, never loaded here);
    // per_layer_token_embd: llama.cpp mainline's (Unsloth's) - both Engram tables in one IQ4_NL tensor, the same
    auto skip = [&](const std::string & n) {
        return is_routed(n) || n == "token_embd.weight" || n.find(".engram_embd.") != std::string::npos || n == "per_layer_token_embd.weight";
    };
    // a file type this build's ggml does not have (mainline ggml's F8_E4M3 = 43, ik_llama's 144+): refused by name here
    // instead of an abort inside ggml_new_tensor (ds41_check_tensors names it first on the usual path)
    auto foreign = [&](const TensorInfo & ti) {
        if ((int) ti.type < (int) GGML_TYPE_COUNT) return false;
        err = ti.name + ": GGUF type " + ds41_type_label(ti.type) + " is not a ggml type of this build";
        return true;
    };
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
                if (foreign(ti)) return false;
                ggml_tensor * t = make(ti, (int) ti.type);   // (bound in place: the file's own type)
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[canon(ti)] = t;
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
                if (foreign(ti)) return false;
                w.t[canon(ti)] = make(ti, target_of(ti));
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
                ggml_tensor * t = w.t[canon(ti)];
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
    if (!strata::v41::ds41_embd_type_ok((int) te->type)) { err = "token_embd type has no row reader"; return false; }
    im.embd = model.shard(si).tensor_data(*te);
    im.embd_type = (int) te->type;
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

// STRATA_X3_STATS counters (see x3_stats_pass)
struct X3Stats {
    long passes = 0, rows = 0, launches = 0, merged = 0, alone = 0;
    double launch_ms = 0, wait_ms = 0;
};
static X3Stats g_x3s;
static bool x3_stats_on() {
    static const bool on = env_on("STRATA_X3_STATS", false);
    return on;
}

// STRATA_X3B: launches per graph allocation (uid), for x3b_replays (host bookkeeping only; nothing when X3B is off)
static bool x3b_env_on();
static double x3b_now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void x3b_count(Ds41Dense::Impl & im, const ggml_cgraph * gf) {
    if (!x3b_env_on()) return;
    const double t = x3b_now_ms();
    auto & u = im.x3b_use[gf->uid];
    ++u.n;
    u.t = t;
    if (im.x3b_use.size() > 8192)   // (uids of re-allocated graphs never come back: drop the long-unused ones)
        for (auto it = im.x3b_use.begin(); it != im.x3b_use.end();) it = t - it->second.t > 60000.0 ? im.x3b_use.erase(it) : std::next(it);
}

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached, bool wait) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph goes first (a merging launch has taken it already)
    if (x3_fail) { x3_fail = false; return false; }   // ... or failed when dsync() launched it (err says why)
    Allo & a = cached ? allo_dec : allo_pass;
    static const bool reuse_on = env_on("DS41_GRAPH_REUSE", true);
    const bool reuse = cached && reuse_on;
    // STRATA_DENSE_ASYNC: no launch without a host wait while the allocator may grow (free) the buffer an earlier
    // launch is still reading; the cached graphs are allocated once, so this is the warm-up only
    const bool ready = a.ready(gf, reuse);
    if (!ready && x3b_gated) {   // STRATA_X3B: x3b_prelaunch checks readiness first, so this is a bug - fail, never hang
        err = "STRATA_X3B: a graph not allocated yet behind an armed gate";
        x3b_fail = true;
        return false;
    }
    if (!ready) dsync();
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    bool ok_alloc = a.alloc(gf, reuse);
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    alloc_failed = !ok_alloc;
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
#ifdef DS41_HAVE_X3B_GATE
    if (x3b_go_pending >= 0) {   // STRATA_X3B_STATS: the dense stream's next start after a routed sum (timing only)
        if (void * e = x3b_gate ? strata::v41::x3b_gate_tev(x3b_gate, 2 * x3b_go_pending + 1) : nullptr) {
            ggml_backend_event ev { ggml_backend_get_device(backend), e };
            ggml_backend_event_record(&ev, backend);
        }
        x3b_go_pending = -1;
    }
#endif
    if (x3_stats_on()) {   // STRATA_X3_STATS: the same calls (ggml_backend_graph_compute = compute_async + synchronize), timed
        const bool async = !wait && ready && !cpu && dense_async();
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
        const auto t1 = std::chrono::steady_clock::now();
        ++g_x3s.launches;
        g_x3s.launch_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (async) {
            g_async_be = backend;
        } else {
            ggml_backend_synchronize(backend);
            g_x3s.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            if (g_async_be == backend) g_async_be = nullptr;
        }
    } else if (!wait && ready && !cpu && dense_async()) {
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
        g_async_be = backend;
    } else {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
        if (g_async_be == backend) g_async_be = nullptr;   // the wait above covered the pending launch too
    }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    if (reuse && gf->uid != 0) x3b_count(*this, gf);   // STRATA_X3B: how often this allocation of the graph launched
    return true;
}

// A graph result to the host.  STRATA_DENSE_ASYNC: the copy queued on the backend stream behind the (unwaited) graph,
// then ONE host wait - instead of the graph's wait plus ggml_backend_tensor_get's own copy-and-wait on another stream.
static void dread(Ds41Dense::Impl & im, const ggml_tensor * t, void * data, size_t n) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
// the plain router only (no smart router / margin); the vision bias through ggml_dsv4_route_vl.
static bool fuse_router(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_FUSE_ROUTER", false) || env_on("STRATA_FUSE", false);
    return on && !im.cpu;
}

// STRATA_FUSE=1 (default off): the decode pass's launch-bound kernel chains as fewer
// kernels, each bit-identical to the chain it replaces (the same F32 steps in the same order; CUDA only).  It turns on:
//   1. the fused router tail above (STRATA_FUSE_ROUTER) - now also with the vision selection bias (the picture reader
//      on: sub / repeat / mul / add of exp_probs_b_vl, which kept the main router on the ~15-kernel chain), through
//      ggml_dsv4_route_vl; and the prefetch predictions' top-16 (~5 kernels each -> 1).
//   2. the shared expert's clamped SwiGLU (clamp, clamp, swiglu: 3 kernels) as ggml's GGML_GLU_OP_SWIGLU_CLAMP (1);
//      on a single-row pass ggml-cuda then also fuses the gate and up GEMVs with it (one quantize, one MMVQ kernel).
// An nsys capture (4K, verify passes of ~4,000 kernels): ~700 fewer kernels a pass from 1, ~100 from 2.
static bool fuse_on(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_FUSE", false);
    return on && !im.cpu;
}

// STRATA_GATEUP_FUSE=1 (default off): the shared expert's gate and
// up GEMVs + clamped SwiGLU as ONE MMVQ kernel (one q8_1 quantization) on the multi-row passes too - a DSpark / lookup
// verify of 2..8 rows - where ggml-cuda fused only single-row passes: per layer quantize + MMVQ twice + the GLU kernel
// (5 launches) -> 2.  The xolotl-ggml.patch side reads the same variable (ggml-cuda.cu ggml_cuda_gateup_fuse_on) and
// fuses only where the unfused products would run MMVQ themselves, for Q8_0 / Q6_K / Q5_K / Q4_K; the fused kernel is
// the unfused one's template with the gate sums riding along: the same per-row dot order and float sums, so the
// output is bit-identical (test_ds41_fuse --cuda --gateup).  It needs the GLU form of item 2 of STRATA_FUSE, so it
// turns that on for the shared expert (exact: STRATA_FUSE's test).  CUDA only.  Routed experts are already one
// gate/up launch (native_expert_grouped's gu kernels cover 2 * n_ff rows: gate and up in the same grid).
static bool gateup_fuse_on(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_GATEUP_FUSE", false);
    return on && !im.cpu;
}

// STRATA_ZC_PRED=1 (default off): the prefetch predictions come back without the copy engine.  Measured: a
// prediction's small device-to-host readback queued behind a 10 MB prefetch DMA still crossing the bus (161 us per layer
// under WDDM).  Here the graph ends in an I32 -> F32 ggml_cpy into pinned host memory - a different type, so ggml-cuda
// runs it as a KERNEL writing over the bus (zero-copy), not as a copy-engine memcpy - and the host reads the floats after
// a stream sync.  The host tensors (F32 [16, n], ids < 2^24 are exact) sit at fixed addresses per (layer, width, which):
// cached CUDA graphs keep their pointers across rebuilds and world swaps.
static bool zc_pred_on(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_ZC_PRED", false);
    return on && !im.cpu;
}
static ggml_tensor * zc_pred_tensor(Ds41Dense::Impl & im, int il, int64_t n, int which) {
#ifdef DS41_HAVE_CUDA
    const auto key = std::make_tuple(il, (int) n, which);
    auto it = im.zc_map.find(key);
    if (it != im.zc_map.end()) return it->second;
    constexpr size_t kBuf = (size_t) 8 << 20;
    if (!im.zc_buf) {
        ggml_init_params p = { ggml_tensor_overhead() * 8192, nullptr, true };
        im.zc_ctx = ggml_init(p);
        im.zc_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), kBuf);
        if (!im.zc_buf) return nullptr;
    }
    // full (8 MiB, or the context's 8192 tensor slots): the caller keeps the device output.  Checked before
    // ggml_new_tensor_2d, which aborts on a full context (each failed call used to take a slot)
    const size_t sz = ((size_t) 16 * sizeof(float) * (size_t) n + 255) & ~(size_t) 255;
    if (im.zc_off + sz > kBuf || im.zc_map.size() >= 8000) return nullptr;
    ggml_tensor * t = ggml_new_tensor_2d(im.zc_ctx, GGML_TYPE_F32, 16, n);
    if (ggml_backend_tensor_alloc(im.zc_buf, t, (char *) ggml_backend_buffer_get_base(im.zc_buf) + im.zc_off) != GGML_STATUS_SUCCESS)
        return nullptr;
    im.zc_off += sz;
    im.zc_map[key] = t;
    return t;
#else
    (void) im; (void) il; (void) n; (void) which;
    return nullptr;
#endif
}
// the graph's ids (an I32 [16, n] view) into its zero-copy host tensor; the original when zero-copy is off / failed
static ggml_tensor * zc_pred_out(Ds41Dense::Impl & im, ggml_context * gc, ggml_tensor * ids, int il, int64_t n, int which) {
    if (!zc_pred_on(im)) return ids;
    ggml_tensor * h = zc_pred_tensor(im, il, n, which);
    return h ? ggml_cpy(gc, ids, h) : ids;
}
// rows of 16 ids from a prediction output: the zero-copy F32 host tensor, or the device I32 view (rows nb[1] apart)
static void read_pred_rows(Ds41Dense::Impl & im, ggml_tensor * o, int64_t n, int * dst) {
    if (o->type == GGML_TYPE_F32) {   // zero-copy: on the host already; wait for the kernel that writes it
        dsync_be(im.backend);
        const float * f = (const float *) o->data;
        for (int64_t i = 0; i < 16 * n; ++i) dst[i] = (int) f[i];
        return;
    }
    const size_t rs = o->nb[1] / sizeof(int32_t);
    im.pred_span.resize(rs * (size_t) (n - 1) + 16);
    dread(im, o, im.pred_span.data(), im.pred_span.size() * sizeof(int32_t));
    for (int64_t t = 0; t < n; ++t)
        std::copy(im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t), im.pred_span.begin() + (ptrdiff_t) (rs * (size_t) t + 16),
                  dst + 16 * t);
}

// ---- the attention-sink watch (see Ds41Dense::sink_watch in the header).  spark-ds41f (patch 0221, vLLM
// models/deepseek_v4_1/nvidia/model.py) keeps hidden_states[:, :, 2455] after the last layer's mhc.post, before the hc
// collapse and the final norm, and flags a sampled row when max over the 4 streams of |value| > threshold.  Here that
// state is x_state [D, HC, NC] when the head runs.  Off unless STRATA_SINK_TRACE or STRATA_SINK_REJECT is set.
struct SinkWatch { bool on = false, mid = false; int64_t col = 2455; int layer = 20; };
static const SinkWatch & sink_cfg() {
    static const SinkWatch w = [] {
        SinkWatch c;
        const char * tr = std::getenv("STRATA_SINK_TRACE");
        const char * rj = std::getenv("STRATA_SINK_REJECT");
        c.mid = tr && tr[0] && tr[0] != '0';
        c.on = c.mid || (rj && std::atof(rj) > 0.0);
        if (const char * e = std::getenv("STRATA_SINK_COL")) c.col = std::atoll(e);
        if (const char * e = std::getenv("STRATA_SINK_LAYER")) c.layer = std::atoi(e);
        return c;
    }();
    return w;
}
// queue the column of rows [row0, row0 + nr) of x_state into sink_host slot `slot` (async on the dense stream: ordered
// after the graphs already launched there; the caller's next stream sync makes it visible)
static void sink_enqueue(Ds41Dense::Impl & im, int slot, int64_t row0, int64_t nr) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    const SinkWatch & w = sink_cfg();
    if (im.sink_host.empty() || !im.x_state || w.col < 0 || w.col >= im.D) return;
    nr = std::min<int64_t>(nr, Ds41Dense::kSinkRows);
    if (row0 < 0 || row0 + nr > im.x_state->ne[2]) return;
    float * dst = im.sink_host.data() + (size_t) slot * Ds41Dense::kSinkRows * (size_t) im.HC;
    for (int64_t r = 0; r < nr; ++r)
        for (int64_t s = 0; s < im.HC; ++s) {
            const size_t off = (size_t) (row0 + r) * im.x_state->nb[2] + (size_t) s * im.x_state->nb[1] + (size_t) w.col * sizeof(float);
            ggml_backend_tensor_get_async(im.backend, im.x_state, dst + r * im.HC + s, off, sizeof(float));
        }
}
// the tap layer's capture at the entry of layer il (the state the layer before it wrote): single-pass rows 0..n-1
static void sink_mid_tap(Ds41Dense::Impl & im, int il) {
    const SinkWatch & w = sink_cfg();
    if (!w.mid || il != w.layer + 1 || im.n > Ds41Dense::kSinkRows) return;
    sink_enqueue(im, 0, 0, im.n);
    im.sink_mid_n = (int) im.n;
}

template <class F>
static Ds41Dense::Impl::MGraph & pass_graph(Ds41Dense::Impl & im, const Ds41Dense::Impl::GKey & key, F build,
                                            size_t n_tensors = 4096, size_t n_nodes = 8192) {
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
    ggml_init_params p = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
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
    if (lim > 0 && (fuse_on(im) || gateup_fuse_on(im))) {   // (STRATA_GATEUP_FUSE: the GLU form ggml-cuda fuses)
        // STRATA_FUSE: one kernel, silu(fminf(gt, lim)) * fmaxf(fminf(up, lim), -lim) (unary.cuh's
        // ggml_cuda_op_swiglu_clamp_single) - for every non-NaN input the same values as clamp.cu's
        // fminf(fmaxf(x, lo), hi) with lo = -inf / -lim followed by the gated op_silu(gt) * up
        return ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_clamp(gc, gt, up, lim));
    }
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    return ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));
}

// STRATA_SHEXP_SPLIT: the shared expert alone, reading o_fn (written by this layer's attention graph), into shexp
static ggml_cgraph * build_shexp(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    ggml_tensor * fn = b.cols2(im.o_fn, im.D, n, 0);
    ggml_tensor * se = build_shexp_ops(im, gc, il, fn);
    ggml_build_forward_expand(gf, ggml_cpy(gc, se, b.cols2(im.shexp, im.D, n, 0)));
    return gf;
}

// STRATA_SHEXP_LATE (default on, =0 off; upstream b7762927, Basalt's idea): the same split on multi-row passes (DSpark's verify
// widths, prompt chunks).  Their attention graph stops at the router and the state copies, and the shared expert is queued right
// after the router's readback, so the host starts the routed experts' work while the card computes it; the finish graph reads
// shexp behind it on the same stream.  Same ops on the same values (o_fn is fn's copy): exact.
static bool shexp_late_on() {
    static const bool on = [] { const char * e = std::getenv("STRATA_SHEXP_LATE"); return !e || e[0] != '0'; }();
    return on;
}

static bool shexp_late_launch(Ds41Dense::Impl & im, int il, int64_t n) {
    return im.compute(pass_graph(im, { il, 21, 0, n }, [&](ggml_context * c, ggml_tensor **) { return build_shexp(im, c, il, n); }).gf, true, false);
}

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool no_shexp = false, ggml_cgraph * into = nullptr) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = into ? into : ggml_new_graph_custom(gc, 8192, false);   // into: STRATA_X3, appended after a prefix
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
        ggml_tensor * wqk = im.engram_wqk(gc, il);   // [D, HC]
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
    ggml_tensor * Mw = b.cols2(im.i_wmask_hi && il > im.last_kv ? im.i_wmask_hi : im.i_wmask, SWA, n, col0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
        if (L.tring) {   // STRATA_SNAP_TAIL: the same rows into the tail ring (nothing reads it in a graph)
            const int64_t kt = std::min<int64_t>(n, im.TR);
            ggml_tensor * lt = ggml_view_2d(gc, kv2, DH, kt, kv2->nb[1], (size_t) (n - kt) * kv2->nb[1]);
            late.push_back(ggml_set_rows(gc, L.tring, lt, b.v1(im.i_tslot, kt, 0)));
        }
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * g_sel = nullptr, * g_comp = nullptr;      // STRATA_KGATHER: the selection and its source, for one gather
    ggml_tensor * g_selm = nullptr;                         // STRATA_KV_GATHER: the selection's mask, read in place
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
                // The selection of queries [q0, q0 + nq) of this span.  Its score / mask tensors are [cap, nq]: with
                // cap = the visible compressed rows (a power of two) they grow with the context - a 1M context's
                // ratio-1 layers hold ~18 bytes x 2^20 rows x 512 queries = a 9.3 GB graph buffer (2026-10-20: the
                // 768K / 1M prompts' last batch, the first to run layers 21-39 past 2^19 rows, OOM'd on it).  Every
                // op here is per query (the indexer kernel's grid is [kv blocks, queries], top-k / argsort / get_rows
                // are per column), so the span's queries are taken in chunks of at most idx_tile score elements, each
                // chunk's tensors dying before the next is built.  Up to idx_tile (default: a 512K context) the span
                // is ONE chunk and the graph is the very one it always was.
                auto select = [&](int64_t q0, int64_t nq, ggml_tensor ** sel_out, ggml_tensor ** selm_out) {
                    const bool whole = q0 == 0 && nq == n;
                    const size_t c0q = (size_t) (col0 + q0);
                    ggml_tensor * iq_c = whole ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                    ggml_tensor * iw_c = whole ? iw : ggml_view_2d(gc, iw, im.IH, nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                    // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                    ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, c0q * 4), 1, nq);
                    ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                    ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, nq)), nvs)));
                    ggml_tensor * sc = ggml_lightning_indexer(gc, iq_c, ksrc, iw_c, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                    sc = ggml_reshape_2d(gc, sc, cap, nq);
                    if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                        ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                        m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                        m = ggml_view_2d(gc, m, cap, nq, m->nb[2], 0);
                        sc = ggml_add(gc, sc, m);
                    }
                    if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                        const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                        GGML_ASSERT(cap % im.CBS == 0);
                        ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, nq]
                        // pin: + 1e30 at block (nvis - 1) / CBS of each query
                        ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                        ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, c0q * 4), 1, nq);
                        ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                        bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                        ggml_tensor * cbk = ggml_top_k(gc, bm, (int) ncb);
                        cbk->op_params[0] = GGML_TOP_K_XOLOTL_RADIX_ASC;   // (see s1 below; the set feeds a set_rows)
                        ggml_tensor * cb = ggml_cont(gc, cbk);                                                // I32 [ncb, nq]
                        outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                    }
                    // GGML_TOP_K_XOLOTL_RADIX_ASC: on CUDA a row wider than 1024 columns is
                    // selected by one radix-select block instead of a full CUB sort of the row (measured: 3.40 ms a
                    // pass at 40K) - the same set (equal scores to the lower row, as the stable sort did), handed back
                    // in ascending row order.  GGML_CUDA_RADIX_TOPK=0 restores the sort; the CPU ignores the flag.
                    ggml_tensor * tk = ggml_top_k(gc, sc, (int) K);
                    tk->op_params[0] = GGML_TOP_K_XOLOTL_RADIX_ASC;
                    ggml_tensor * s1 = ggml_cont(gc, tk);                                                   // I32 [K, nq]
                    // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                    // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                    // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                    // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                    // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                    // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                    s1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s1, 1, K, nq),
                                                           ggml_argsort(gc, ggml_cast(gc, s1, GGML_TYPE_F32),
                                                                        GGML_SORT_ORDER_ASC)), K, nq);
                    ggml_tensor * m1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, nq), s1), K, nq);   // vis at sel
                    // the chunk's writes go to outs now, so the graph runs (and frees) this chunk before the next
                    outs.push_back(ggml_cpy(gc, s1, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], c0q * L.sel->nb[1])));
                    outs.push_back(ggml_cpy(gc, m1, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], c0q * L.selm->nb[1])));
                    *sel_out = s1;
                    *selm_out = m1;
                };
                const int64_t qc = im.idx_tile > 0 && cap * n > im.idx_tile ? std::max<int64_t>(1, im.idx_tile / cap) : n;
                for (int64_t q0 = 0; q0 < n; q0 += qc) {
                    ggml_tensor * s1 = nullptr, * m1 = nullptr;
                    select(q0, std::min(qc, n - q0), &s1, &m1);
                    sel = sel ? ggml_concat(gc, sel, s1, 1) : s1;      // [K, n] for the gather below
                    selm = selm ? ggml_concat(gc, selm, m1, 1) : m1;
                }
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            g_sel = sel;
            g_comp = comp;
            g_selm = selm;
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
    // times (measured: concat_cont 3.25 s of a 16K prompt).  The same F16 values: the window rows round F32 -> F16
    // as rows_as does, the compressed rows dequantize the same way, the padding is the same zeros.  Only where the
    // whole-cap dequant is cheaper than the key sets it saves copying (cap <= n * M / 2: prompt passes, not decode).
    static const bool kgather = std::getenv("STRATA_KGATHER") && std::getenv("STRATA_KGATHER")[0] == '1';
    bool kgathered = false;
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
            kgathered = true;
        }
    }
    // STRATA_KV_GATHER=1 (default off; P3 A6): every pass's keys and mask in ONE op each (GGML_OP_DSV41_KV_GATHER,
    // xolotl-ggml.patch), reading the ring, this pass's new rows, the compressed rows, the indices and the masks in
    // place.  Replaces the ring cast + [ring | new rows] concat, the two gathers, the conts of sel / selm, the key and
    // mask concats, the zero-row pad gather + concat, the -inf fill + concat and the mask cast (~14 small launches per
    // layer-pass; NSIGHT_387: concat_cont 1.50 ms a decode pass).  The same bytes into the flash-attn: F16 keys rounded
    // from the same F32 values with the same conversion (the F32 / f16 == false path rounds them at its final cast the
    // same way), the same order (window rows, then the top-k in sel's order, then +0 rows to a multiple of 256), the
    // same mask values with -inf padding.  The prompt-only STRATA_KGATHER path keeps its passes when it runs.
#if defined(GGML_DSV41_KV_GATHER)
    static const bool p3_kvgather = [] {
        const char * e = ds41_env("STRATA_KV_GATHER", "STRATA_P3_KVGATHER");
        const bool on = e && e[0] == '1';
        if (on) std::fprintf(stderr, "ds41: STRATA_KV_GATHER=1 - attention keys + mask in one op each\n");
        return on;
    }();
    if (p3_kvgather && !kgathered && !im.cpu && kv2->type == GGML_TYPE_F32 && kv2->nb[0] == sizeof(float) &&
        (L.raw->type == GGML_TYPE_F32 || L.raw->type == GGML_TYPE_F16 || L.raw->type == GGML_TYPE_BF16) &&
        (!Kc || (g_sel && g_comp && g_selm && g_selm->type == GGML_TYPE_F32 &&
                 (g_comp->type == GGML_TYPE_F32 || g_comp->type == GGML_TYPE_F16 || g_comp->type == GGML_TYPE_BF16 ||
                  (g_comp->type == GGML_TYPE_NVFP4 && DH % 256 == 0))))) {
        const int64_t K = Kc ? g_sel->ne[0] : 0, MP = (SWA + K + 255) / 256 * 256;
        ggml_tensor * wi = ggml_reshape_2d(gc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4), SWA, n);
        Kall = ggml_dsv41_kv_gather(gc, L.raw, kv2, wi, Kc ? g_comp : nullptr, Kc ? g_sel : nullptr, MP);   // F16 [DH, MP, n]
        Mall = ggml_dsv41_kv_mask(gc, Mw, Kc ? g_selm : nullptr, MP);                                      // F16 [MP, n]
        M = MP;
    }
#else
    (void) kgathered;
    if (ds41_env("STRATA_KV_GATHER", "STRATA_P3_KVGATHER") && ds41_env("STRATA_KV_GATHER", "STRATA_P3_KVGATHER")[0] == '1') {
        static bool warned = false;
        if (!warned) std::fprintf(stderr, "ds41: STRATA_KV_GATHER=1 ignored - this ggml has no GGML_OP_DSV41_KV_GATHER "
                                          "(rebuild against tools/xolotl/ggml/xolotl-ggml.patch)\n");
        warned = true;
    }
#endif
    if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
        const int64_t pad = 256 - M % 256;
        // F16 has no CUDA pad: concat zeros gathered from the one zero row (F16 -> F16, exact) - the same zeros.
        // The F32 pad kernel launches one grid row per key row (ggml-cuda/pad.cu: gridDim.y = ne1), and CUDA caps
        // gridDim.y at 65,535, so past that build the pad rows explicitly too - the same zeros (upstream 5b06a59).
        Kall = (f16 || M + pad > 65535)
             ? ggml_concat(gc, Kall, f16
                   ? ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * n, 0), true), DH, pad, n)
                   : ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, DH, pad, n), 0.0f), 1)
             : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, n)
                                                   : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    ggml_tensor * m4 = Mall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Mall, M, 1, 1, n)   // STRATA_KV_GATHER: F16 already
                                                   : ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
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
    ggml_tensor * vl_img = nullptr;   // vision: the image mask the fused route reads (ggml_dsv4_route_vl)
    if (im.vision && im.i_img) {
        ggml_tensor * bt = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b.bias"), im.NE, 1);
        ggml_tensor * bv = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b_vl.bias"), im.NE, 1);
        vl_img = b.v1(im.i_img, n, (size_t) col0 * 4);
        ggml_tensor * m = ggml_reshape_2d(gc, vl_img, 1, n);
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
    } else if (fuse_router(im) && !smart_router() && rb_margin < 0.0f && vl_img) {
        // the vision selection bias ((bias_vl - bias) * img + bias, the chain above) inside the fused route
        ids = ggml_dsv4_route_vl(gc, rl, im.BL(il, "exp_probs_b.bias"), im.BL(il, "exp_probs_b_vl.bias"), vl_img,
                                 im.ly[(size_t) il].rbias, b.cols2(im.o_wts, im.NU, n, col0), (int) im.NU,
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
        ggml_tensor * wqk = im.engram_wqk(gc, il);
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
        if (L.tring) {   // STRATA_SNAP_TAIL (as build_attn)
            const int64_t kt = std::min<int64_t>(n, im.TR);
            ggml_tensor * lt = ggml_view_2d(gc, kv2, DH, kt, kv2->nb[1], (size_t) (n - kt) * kv2->nb[1]);
            late.push_back(ggml_set_rows(gc, L.tring, lt, b.v1(im.i_tslot, kt, 0)));
        }
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
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1,
                                   ggml_cgraph * into = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = into ? into : ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = fuse_router(im) ? ggml_dsv4_route(gc, rl, im.BL(il, "exp_probs_b.bias"), im.ly[(size_t) il].rbias, nullptr, 16, false, 1.0f)
                           : ggml_argsort_top_k(gc, sc, 16);
    *out = zc_pred_out(im, gc, *out, il, n, 0);   // STRATA_ZC_PRED: straight into pinned host memory
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
        o = zc_pred_out(im, gc, o, il, n, 1 + j);   // STRATA_ZC_PRED
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
        *(j == 0 ? out : out2) = o;
    }
    return gf;
}

// STRATA_FOLDS=1 (default off; ds41_ops.hpp ds41_folds_on): finish's shexp + routed add inside the hc_post kernel
// (ds41_hc_post_add: the same F32 add, then hc_post's same steps) - one kernel fewer per finish.  CUDA only.
static bool folds_on(const Ds41Dense::Impl & im) {
    return ds41_folds_on() && !im.cpu;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0,
                                  ggml_cgraph * into = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = into ? into : ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x;
    if (folds_on(im)) {   // STRATA_FOLDS: the add inside hc_post
        ggml_tensor * se = b.cols2(im.shexp, D, n, col0);
        ggml_tensor * ro = b.cols2(im.routed, D, n, col0);
        x = ds41_hc_post_add(gc, se, ro, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                             b.cols3(im.comb_f, HC, HC, n, col0));
    } else {
        ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
        x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                              b.cols3(im.comb_f, HC, HC, n, col0));
    }
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x;
    if (folds_on(im)) {   // STRATA_FOLDS: the add inside hc_post (build_finish)
        ggml_tensor * se = b.cols2(im.shexp, D, 1, 0);
        ggml_tensor * ro = b.cols2(im.routed, D, 1, 0);
        x = ds41_hc_post_add(gc, se, ro, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                             b.cols3(im.comb_f, HC, HC, 1, 0));
    } else {
        ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
        x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                              b.cols3(im.comb_f, HC, HC, 1, 0));
    }
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
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg,
                                ggml_cgraph * into = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = into ? into : ggml_new_graph_custom(gc, 64, false);
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
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0,
                                ggml_cgraph * into = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = into ? into : ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== STRATA_X3: fewer, larger decode graphs
//
// STRATA_X3=1 (default off).  A decode pass launched one
// ggml graph per piece: begin, then per layer [prediction] + attention/router + finish, then the head - ~3 dense graph
// launches a layer (+ the tier's own per-class graphs), each ~15-18 us of fixed host cost in cudaGraphLaunch under WDDM
// on top of the ~1.6 us per kernel (kernel listing 2026-10-24: 15 us + 1.66 us/kernel).  Of those boundaries only two
// per layer are real host round trips: the router readback (the host picks the tiers) and - with STRATA_D3_PF_ASYNC -
// the prediction readback that sits between the prediction and the attention on the stream (its DMA must start before
// the attention ends).  The finish -> next prediction / attention and begin -> layer 0 / last finish -> head boundaries
// exist only because each piece was its own graph: nothing on the host reads or decides anything between them.
//
// So begin() and finish() no longer launch: they leave their graph PENDING, and the next dense launch - predict /
// predict_multi / predict_multi_launch, attn_router / attn_router_launch, or the head (logits) - builds ONE cached graph
// = the pending graph's nodes followed by its own (x3_take).  Any other dense call (a readback, an upload, an event, a
// graph of another kind) first launches the pending graph alone, exactly as before (x3_flush) - so nothing is ever
// reordered, only launched later on the host, and at most until the next dense call.  Engram layers launch it before
// their disk wait (no GPU time lost behind the wait).
//
// Exact: the merged graph is the two graphs' node lists back to back - the same ops on the same tensors in the same order
// on the same stream (ggml-cuda executes a graph's nodes in order; its fusions need an edge between the fused nodes, and
// the two halves share none: the second half reads the persistent x_state / pre_state through its own views).  Only the
// gallocr placement of intermediates differs, which no kernel's arithmetic depends on.  Same kernels, same inputs, same
// bits.  A verify pass of L layers (the launcher: STRATA_PASS_GRAPHS, D3_PF_ASYNC, SHEXP_LATE) launched 4L + 2 dense
// graphs (begin; per layer prediction, attention, late shared expert, finish; head); with X3 3L + 1 (begin rides on
// layer 0's prediction, each finish on the next layer's prediction, the last finish on the head), and begin's own host
// wait is gone.  A single step (n = 1, finish_predict per layer) saves begin and the last finish only.
static bool x3_on(const Ds41Dense::Impl & im) {
    static const bool on = env_on("STRATA_X3", false);
    return on && !im.cpu;
}

// STRATA_X3_STATS=1 (measurement only; X3 on or off): one "x3:" line every STRATA_X3_STATS_EVERY passes (default 64) -
// dense graph launches per pass (and how many were X3 merges / pending graphs launched alone), and the host time per pass
// inside the launch call itself (ggml-cuda's checks + cudaGraphLaunch; a graph run kernel by kernel while ggml-cuda warms
// up its CUDA graph counts too) and in the host waits that end a waiting compute (begin, the head).  With it on, a
// waiting compute is split into compute_async + synchronize - exactly what ggml_backend_graph_compute is.
static void x3_stats_pass(int64_t n) {   // begin(): a new pass
    if (!x3_stats_on()) return;
    static const long every = [] { const char * e = std::getenv("STRATA_X3_STATS_EVERY"); return e && *e ? std::max(1L, std::atol(e)) : 64L; }();
    X3Stats & s = g_x3s;
    if (s.passes >= every) {
        const double P = (double) s.passes;
        std::fprintf(stderr, "x3: %ld passes, %.2f rows/pass | dense graph launches/pass %.1f (x3 merged %.1f, pending alone %.1f) | "
                             "host ms/pass: graph launch %.3f, graph waits %.3f | STRATA_X3 %s\n",
                     s.passes, s.rows / P, s.launches / P, s.merged / P, s.alone / P, s.launch_ms / P, s.wait_ms / P,
                     env_on("STRATA_X3", false) ? "on" : "off");
        s = X3Stats();
    }
    ++s.passes;
    s.rows += (long) n;
}

static Ds41Dense::Impl * g_x3_im = nullptr;   // the dense half holding a pending graph (dsync() reaches it through this)

// begin() / finish(): leave the graph pending (true) instead of launching it
static bool x3_defer(Ds41Dense::Impl & im, int kind) {
    if (!x3_on(im) || !(im.n == 1 || pass_graphs_on(im.n))) return false;   // only the cached decode graphs
    x3_flush_pending();   // (an older one: cannot happen - every dense call before this one launched it)
    if (im.x3_fail) return false;
    im.x3_pend = kind;
    im.x3_n = im.n;
    g_x3_im = &im;
    return true;
}

// launch the pending graph alone - the very graph and call begin() / finish() made before X3
static bool x3_flush(Ds41Dense::Impl & im) {
    if (!im.x3_pend) return true;
    const int kind = im.x3_pend;
    const int64_t n = im.x3_n;
    im.x3_pend = 0;
    g_x3_im = nullptr;
    if (x3_stats_on()) ++g_x3s.alone;
    ggml_cgraph * gf = nullptr;
    if (kind == 1) {
        if (n == 1) {
            auto it = im.graphs.find({ -1, 0, 0, 1 });
            if (it == im.graphs.end()) it = im.graphs.emplace(Ds41Dense::Impl::GKey { -1, 0, 0, 1 }, build_init(im, im.gctx, 1)).first;
            gf = it->second;
        } else {
            gf = pass_graph(im, { -1, 0, 0, n }, [&](ggml_context * c, ggml_tensor **) { return build_init(im, c, n); }).gf;
        }
        return im.compute(gf, true);
    }
    if (n == 1) {
        auto it = im.graphs.find({ -1, 2, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Ds41Dense::Impl::GKey { -1, 2, 0, 1 }, build_finish(im, im.gctx, 1)).first;
        gf = it->second;
    } else {
        gf = pass_graph(im, { -1, 2, 0, n }, [&](ggml_context * c, ggml_tensor **) { return build_finish(im, c, n); }).gf;
    }
    return im.compute(gf, true, false);
}

namespace {   // (declared in the anonymous namespace above, for dsync())
void x3_flush_pending() {
    if (!g_x3_im) return;
    Ds41Dense::Impl & im = *g_x3_im;
    if (!x3_flush(im)) {   // reported by the next compute() (callers of dsync() return void)
        im.x3_fail = true;
        std::fprintf(stderr, "ds41_dense: STRATA_X3 pending graph failed: %s\n", im.err.c_str());
    }
}
}  // namespace

// a merged graph's launch: when it fails, the pending graph it carried did not run either - every later compute()
// fails too (x3_fail), so no caller can go on with that layer's state missing
static bool x3_run(Ds41Dense::Impl & im, Ds41Dense::Impl::MGraph & g, bool wait) {
    if (im.compute(g.gf, true, wait)) return true;
    im.x3_fail = true;
    return false;
}

// the merged graphs' kinds (pass_graph keys; the plain kinds stay below 40)
constexpr int kX3Pred = 40;   // + 0 begin / 1 finish: prefix + build_predict (il, 0, n)
constexpr int kX3Attn = 42;   // + 0 / 1, + 2 when the attention leaves the shared expert out: prefix + build_attn (il, cap, n)
constexpr int kX3Head = 46;   // + 0 / 1: prefix + build_head (row0, 0, n * 4096 + nrows)

// a merging launch: the pending graph's nodes, then build_main's, as ONE cached graph under key { il, kind_base + the
// pending kind - 1, cap, nkey }; nullptr = nothing pending (the caller launches its own graph as before)
template <class F>
static Ds41Dense::Impl::MGraph * x3_take(Ds41Dense::Impl & im, int kind_base, int il, int64_t cap, int64_t nkey, F build_main) {
    if (!im.x3_pend) return nullptr;
    if (im.x3_n != im.n) { x3_flush_pending(); return nullptr; }   // (a pending graph is always this pass's)
    const int pend = im.x3_pend;
    const int64_t n = im.x3_n;
    im.x3_pend = 0;
    g_x3_im = nullptr;
    if (x3_stats_on()) ++g_x3s.merged;
    // node memory: an attention graph's budget, or a small one for a prefix + prediction / head (~40 tensors) - one
    // such graph per (layer, width), so the full 4096-tensor budget would cost ~1.6 MB of host RAM each
    const bool big = kind_base == kX3Attn;
    const size_t nodes = big ? 8192 : 256;
    return &pass_graph(im, { il, kind_base + pend - 1, cap, nkey }, [&](ggml_context * c, ggml_tensor ** out) {
        ggml_cgraph * gf = ggml_new_graph_custom(c, nodes, false);
        if (pend == 1) build_init(im, c, n, 0, gf);
        else build_finish(im, c, n, 0, gf);
        build_main(c, out, gf);
        return gf;
    }, big ? 4096 : 256, nodes);
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
int64_t Ds41Dense::ring_spare() const { return p_->RAW > 0 ? p_->RAW - p_->SWA : 16; }   // 16 before init()
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }
bool Ds41Dense::vision() const { return p_->vision; }

void Ds41Dense::set_vision(const float * rows, const int64_t * pos, int n_rows) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    im.vis_rows = rows;
    im.vis_pos = pos;
    im.n_vis = n_rows;
    im.eng_next_tids.clear();   // no Engram read-ahead for a request with pictures (engram_ready / engram_next)
}

void Ds41Dense::clear_vision() {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    im.vis_rows = nullptr;
    im.vis_pos = nullptr;
    im.n_vis = 0;
}

// The bytes alloc_cols below allocates for NC columns with this hap layout (the sum of its tensors' ggml_nbytes; the
// backend buffer adds at most its alignment per tensor).  STRATA_PROMPT_SWEEP_FIT sizes the wide prompt batch from it
// (ds41_generate.cpp) instead of fitted constants.  KEEP IN STEP WITH alloc_cols: every tensor there has its line
// here, and alloc_cols compares the two once per process (a drift prints "state_bytes drift").  All tensors are
// 4-byte types (F32 / I32).
static size_t cols_bytes(const Ds41Dense::Impl & im, int64_t NC, bool hap_alias) {
    const int64_t D = im.D, HC = im.HC;
    int64_t e = 0;
    e += D * HC * NC;                              // x_state
    e += HC * NC;                                  // pre_state
    if (!hap_alias) e += D * HC * NC;              // hap (wide: a view of x_state)
    e += HC * NC;                                  // post_f
    e += HC * HC * NC;                             // comb_f
    e += D * NC;                                   // shexp
    e += D * NC;                                   // o_fn
    e += 2 * im.NU * NC;                           // o_ids, o_wts
    if (smart_router()) e += 2 * im.NU * NC;       // o_xids, o_xwts
    e += D * NC;                                   // routed
    e += im.CB * NC;                               // cand
    for (const auto & L : im.ly)
        if (L.idx_src) e += 2 * im.ITOP * NC;      // sel, selm
    e += D * NC;                                   // i_emb
    e += NC;                                       // i_pos
    if (im.vision) e += NC;                        // i_img
    e += im.SWA * NC;                              // i_widx
    e += im.SWA * NC;                              // i_wmask
    if (im.cfg.win_floor) e += im.SWA * NC;        // i_wmask_hi
    e += std::min<int64_t>(NC, im.RAW);            // i_wslot
    if (im.TR > 0) e += std::min<int64_t>(NC, im.TR);   // i_tslot
    e += 2 * 3 * NC;                               // i_nvis, i_cslot, i_cpos (r = 1, 2)
    e += 2 * NC;                                   // i_sidx
    e += std::min<int64_t>(NC, 2);                 // i_sslot
    e += NC;                                       // i_lastblk
    e += 2 * im.EC * im.ED * NC;                   // i_eng[0], i_eng[1]
    return (size_t) e * 4;
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
    // im.hap_alias (set_columns(nc, true): STRATA_WIDE_PROMPT's wide prompt batches, ported from Keisawaakira/Xolotl's
    // STRATA_WIDE_PROMPT): fewer bytes per state column.  The FFN's hc state (hap) lives in the x_state columns it
    // would shadow: x_state is dead from the attention's hc_post until finish writes the new state back.  In
    // build_attn every read of x_state (the Engram add, hc_mixes, hc_pre, hc_post) is an ancestor of h, so the h -> hap
    // copy runs after all of them; finish reads hap into hc_post's own output tensor and only then copies it over
    // x_state.  The same values flow through the same ops: exact.  (-64 KiB a column at D = 4096, HC = 4.)  Every other
    // allocation (decode widths, classic batches; init unless fixed columns with the mode set) gives hap its own tensor,
    // as with the mode unset.
    im.hap = im.hap_alias ? ggml_view_3d(c, im.x_state, D, HC, NC, im.x_state->nb[1], im.x_state->nb[2], 0)
                          : ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
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
    im.i_wmask_hi = im.cfg.win_floor ? ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC) : nullptr;
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    im.i_tslot = im.TR > 0 ? ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.TR)) : nullptr;
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
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t))
            if (!t->view_src) tot += (double) ggml_nbytes(t);   // (wide: hap is a view of x_state)
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    {   // cols_bytes (the wide prompt budget, STRATA_PROMPT_SWEEP_FIT) must count what was just declared: say so once if not
        static bool drift_said = false;
        if (!drift_said) {
            size_t real = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t))
                if (!t->view_src) real += ggml_nbytes(t);
            const size_t want = cols_bytes(im, NC, im.hap_alias);
            if (real != want) {
                drift_said = true;
                std::fprintf(stderr, "ds41_dense: state_bytes drift: %lld columns allocate %zu bytes, cols_bytes says %zu - "
                                     "update cols_bytes beside alloc_cols\n", (long long) NC, real, want);
            }
        }
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    dclear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc, bool hap_in_state) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC && hap_in_state == im.hap_alias) return true;
    dsync_be(im.backend);
    // STRATA_KEEP_DEC_GRAPHS=1: a prompt batch parks the decode-width world (its columns, allocators and every cached
    // graph - ggml-cuda keeps their captured CUDA graphs by node address) and gets an empty one of its own; on the way
    // back down the parked world returns as it was, so the next answer does not rebuild ~1,000 graphs, run each kernel
    // by kernel once and re-capture it (measured).  The same graphs on the same tensors: exact.
    static const bool keep = env_on("STRATA_KEEP_DEC_GRAPHS", false);
    if (keep && im.parked_dec.valid && want == im.parked_dec.NC && hap_in_state == im.parked_dec.hap_alias) {
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
        al->failed = false;
        ++al->epoch;
    }
    if (want == im.S) allo_reserve(im.allo_dec);   // the decode width (a prompt batch's graphs need no head room)
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.hap_alias = hap_in_state;
    // a failed allocation leaves no columns (NC 0), so the caller's retry at a narrower width (or the same width with
    // the other hap layout) reallocates instead of taking the unbacked tensors as current
    if (!alloc_cols(im, want, im.err)) { im.NC = 0; return false; }
    return true;
}

int Ds41Dense::columns() const { return (int) p_->NC; }
size_t Ds41Dense::state_bytes(int nc, bool hap_in_state) const {
    const Impl & im = *p_;
    return cols_bytes(im, std::max<int64_t>(im.S, nc), hap_in_state);   // (set_columns also caps nc at the prompt batch)
}
bool Ds41Dense::hap_in_state() const { return p_->hap_alias; }

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
    // STRATA_INDEXER_TILE=<elements>: the indexer selection's per-graph piece (see build_attn); 0 = never split
    im.idx_tile = std::getenv("STRATA_INDEXER_TILE") ? std::max<long long>(0, std::atoll(std::getenv("STRATA_INDEXER_TILE")))
                                                     : (int64_t) 1 << 28;
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // STRATA_WIDE_PROMPT with fixed columns (one-shot, STRATA_ELASTIC_COLS=0): the NCmax columns start with hap in
    // x_state, as the mode always allocated them, so a wide prompt never peaks at the classic size; a classic feed
    // (below STRATA_WIDE_PROMPT_MIN) reallocates them with hap its own tensor before its first batch (set_columns).
    im.hap_alias = !cfg.elastic_cols && env_on("STRATA_WIDE_PROMPT", "STRATA_D11_WIDE_PROMPT", false);
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    // STRATA_SNAP_TAIL: tail rows past the ring (capped by the context); 0 = off - no tensor, no graph op, no copy
    im.TR = cfg.snap_tail > 0 ? std::min<int64_t>(im.RAW + (int64_t) cfg.snap_tail, (int64_t) cfg.ctx_max) : 0;
    if (im.TR > 0 && im.TR <= im.RAW) im.TR = 0;
    if (im.TR > 0) im.ttag.assign((size_t) im.TR, -1);
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
    // STRATA_FILE_TIER_UNMAP (ds41_unmap.hpp, default off): the weights are on the card, so the mapping is needed only for the
    // token_embd rows - copy them out (the same bytes) and release every shard's mapping, so the file tier's unbuffered
    // reads of these shards no longer queue on NTFS's coherency check.  The CPU backend binds its weights in the
    // mapping (load_weights) and keeps it.
    if (ds41_p3_unmap() && !im.cpu) {
        im.embd_copy = ds41_embd_copy(*im.model, err);
        if (!im.embd_copy) return false;
        im.embd = im.embd_copy->data();
        for (size_t s = 0; s < im.model->size(); ++s) im.model->release_mapping(s);
        std::fprintf(stderr, "unmap: dense mapping of %zu shard(s) released\n", im.model->size());
    }
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
        if (L.kv_src) im.last_kv = std::max(im.last_kv, (int) l);
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
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (512 + (im.TR > 0 ? im.nl : 0)), nullptr, true };
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
        if (im.TR > 0) L.tring = T(kvt, { im.DH, im.TR });
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
    // the pinned logits rows (see Impl::lpin): at init, never mid-request; no host buffer type or no memory = the
    // caller's pageable buffer, as before
    if (!im.cpu && !(ds41_env("STRATA_PINNED_STAGE", "STRATA_P3_HOST_STAGE") && ds41_env("STRATA_PINNED_STAGE", "STRATA_P3_HOST_STAGE")[0] == '0')) {
        constexpr size_t kLogitsPinRows = 16;   // Ds41MoeTier::kMaxMulti: the widest verify pass
        ggml_backend_dev_t dev = ggml_backend_get_device(im.backend);
        ggml_backend_buffer_type_t hbt = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        const size_t n = kLogitsPinRows * (size_t) im.g.vocab;
        if (hbt && n > 0) im.lpin = ggml_backend_buft_alloc_buffer(hbt, n * 4);
        if (im.lpin) {
            im.lpin_p = (float *) ggml_backend_buffer_get_base(im.lpin);
            im.lpin_n = n;
        }
    }
    if (sink_cfg().on) {   // the attention-sink watch: never allocated (nor read) when off
        im.sink_host.assign((size_t) 2 * kSinkRows * (size_t) im.HC, 0.0f);
#ifdef DS41_HAVE_CUDA
        if (!im.cpu) ggml_backend_cuda_register_host_buffer(im.sink_host.data(), im.sink_host.size() * sizeof(float));
#endif
        std::fprintf(stderr, "ds41_dense: attention-sink watch on: column %lld of the %lld hc streams after the last layer; "
                             "trace tap after layer %d: %s\n", (long long) sink_cfg().col, (long long) im.HC, sink_cfg().layer,
                     sink_cfg().mid ? "on" : "off");
    }
    return true;
}

bool Ds41Dense::sink_watch() const { return sink_cfg().on; }
int Ds41Dense::sink_layer() const { return sink_cfg().layer; }
int Ds41Dense::sink_rows() const { return p_->sink_n; }
long Ds41Dense::sink_seq() const { return p_->sink_seq; }
float Ds41Dense::sink_value(int r, bool mid) const {
    const Impl & im = *p_;
    if (r < 0 || r >= im.sink_n) return -1.0f;
    const int64_t row = mid ? im.sink_row0 + r : r;
    if (mid && row >= im.sink_mid_n) return -1.0f;
    const float * v = im.sink_host.data() + ((size_t) (mid ? 0 : 1) * kSinkRows + (size_t) row) * (size_t) im.HC;
    float m = 0.0f;
    for (int64_t s = 0; s < im.HC; ++s) {
        const float a = std::fabs(v[s]);
        if (a > m) m = a;   // a NaN never wins (nan_to_num 0)
    }
    return m;
}

float * Ds41Dense::logits_host(size_t n) {
    Impl & im = *p_;
    return n <= im.lpin_n ? im.lpin_p : nullptr;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    dset(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

bool Ds41Dense::set_route_select(int protect, int pool) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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

void Ds41Dense::reset_span_width() { p_->span_max = 0; }

void Ds41Dense::release_pass() {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    dsync();   // STRATA_DENSE_ASYNC: nothing may still run in the buffer freed here
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    im.allo_pass.failed = false;
    ++im.allo_pass.epoch;
    im.span_max = 0;   // the next prompt starts at full sub-batch width again (see attn_router_span)
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
// STRATA_PASS_CAP_MIN=N (default 0): never below N.  The attention / indexer graphs are cached per pass_cap, so each
// doubling of the context builds, runs directly once and captures ~360 new graphs (measured: ~0.5 ms/token in the
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

size_t Ds41Dense::span_graph_bytes(int pos_end, int n) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    n = (int) std::max<int64_t>(1, std::min<int64_t>(n, im.S));
    pos_end = (int) std::max<int64_t>(n, std::min<int64_t>(pos_end, im.cfg.ctx_max));
    int64_t caps[3];
    span_pass_caps(im, pos_end - n, n, caps);
    const std::array<int64_t, 3> key { caps[1], caps[2], n };
    auto it = im.graph_bytes_memo.find(key);
    if (it != im.graph_bytes_memo.end()) return it->second;
    // build every layer KIND's span graph (they differ in ratio, kv / index / candidate source, Engram) and let a
    // throwaway allocator plan it: ggml_gallocr_reserve_n_size sizes the buffer without allocating anything
    int64_t saved[3] = { im.pass_cap[0], im.pass_cap[1], im.pass_cap[2] };
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = caps[r];
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    std::set<std::array<int64_t, 6>> kinds;
    size_t mx = 0;
    for (int64_t l = 0; l < im.nl; ++l) {
        const Impl::Layer & L = im.ly[(size_t) l];
        if (!kinds.insert({ L.ratio, L.kv_src, L.idx_src, L.cand_src, L.uses_cand, L.eslot >= 0 }).second) continue;
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_attn(im, gc, (int) l, n, 0);
        size_t sz = 0;
        ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, &sz);
        mx = std::max(mx, sz);
        ggml_free(gc);
    }
    ggml_gallocr_free(ga);
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = saved[r];
    im.graph_bytes_memo[key] = mx;
    return mx;
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
// STRATA_ENGRAM_SPLIT=1 (default off): a layer waits for ITS slot's rows only (Ds41Engram::wait_table) - layer 1 no
// longer waits for layer 14's half of a long prompt's reads, which go on under layers 1-13 (a measured profile: a
// 2.4 s idle gap before layer 1 of a 16K prompt).  The same rows, uploaded per slot: exact.
static bool engram_ahead_on() {
    static const bool on = [] { const char * e = std::getenv("STRATA_ENGRAM_AHEAD"); return !e || e[0] != '0'; }();
    return on;
}

// STRATA_ENGRAM_READAHEAD (default off; Abyss 77d4541): 1 = on with 256-token blocks, N >= 16 = N-token blocks.  On:
//  - a pass's Engram reads go block by block (table 0's blocks in token order, then table 1's; Ds41Engram::start blk), and a
//    layer-major span waits only for ITS columns' blocks (engram_ready c0 / cn) and uploads them, so layer 1's first spans
//    run while the rest of the chunk is still being read (before: layer 1 waited for every row of the chunk);
//  - the next chunk's reads (engram_next) start at once into the second buffer, queued behind this chunk's (before: only
//    once this chunk's rows were on the card), and a shorter pass at the same place takes them too (a halved or re-cut
//    batch used to wait for them and read its rows again);
//  - a read-ahead nobody takes is cancelled before it is waited.
// The same rows in the same columns: exact.  Off: the old path, unchanged.
static int engram_readahead_blk() {
    static const int v = [] {
        const char * e = std::getenv("STRATA_ENGRAM_READAHEAD");
        const int x = e ? std::atoi(e) : 0;
        return x <= 0 ? 0 : x < 16 ? 256 : x;
    }();
    return v;
}

static bool engram_ready(Ds41Dense::Impl & im, int slot = -1, int c0 = 0, int cn = -1) {
    if (!im.eng_batch) return true;
    if (engram_readahead_blk() > 0) {   // STRATA_ENGRAM_READAHEAD: per-table, per-column waits; the pass's eng_n columns only
        auto & rows = *im.eng_cur;
        const size_t per = (size_t) (im.EC * im.ED);
        const int N = im.eng_n;
        const int ne = (int) std::min<size_t>(rows.size(), 2);
        if (slot >= 0 && slot < ne) {
            const int want = cn >= 0 ? std::min(N, c0 + cn) : N;
            int & up = im.eng_upc[slot];
            if (up < want) {
                if (!im.engram.wait_cols(im.eng_batch, slot, up, want, im.err)) {
                    // review 11-19: cancel and join the batch's reads before dropping it - workers hold a raw pointer
                    Ds41Engram::cancel(im.eng_batch); std::string e2; im.engram.wait(im.eng_batch, e2);
                    im.eng_batch.reset(); return false;
                }
                dset(im.i_eng[slot], rows[(size_t) slot].data() + (size_t) up * per, (size_t) up * per * 4, (size_t) (want - up) * per * 4);
                up = want;
            }
            for (int e = 0; e < ne; ++e) if (im.eng_upc[e] < N) return true;   // the rest is still being read
        }
        if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
        im.eng_batch.reset();
        for (int e = 0; e < ne; ++e)
            if (im.eng_upc[e] < N)
                dset(im.i_eng[e], rows[(size_t) e].data() + (size_t) im.eng_upc[e] * per, (size_t) im.eng_upc[e] * per * 4,
                     (size_t) (N - im.eng_upc[e]) * per * 4);
        im.eng_upc[0] = im.eng_upc[1] = 0;
        im.eng_next_tids.clear();   // (engram_next started the read-ahead itself)
        return true;
    }
    static const bool split = std::getenv("STRATA_ENGRAM_SPLIT") && std::getenv("STRATA_ENGRAM_SPLIT")[0] == '1';
    if (split && slot >= 0 && slot < 2 && (size_t) slot < im.eng_rows.size()) {
        if (!im.eng_up[slot]) {
            if (!im.engram.wait_table(im.eng_batch, slot, im.err)) {
                Ds41Engram::cancel(im.eng_batch); std::string e2; im.engram.wait(im.eng_batch, e2);   // as above
                im.eng_batch.reset(); return false;
            }
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
    // STRATA_ENGRAM_AHEAD: eng_rows is free again (dset is synchronous): the next chunk's reads go into it now.  Its tokens are
    // recorded first (positions past this pass: start() took this pass's hashes already, and the next pass sets them again).
    // Text only: a request with pictures (n_vis > 0) needs its image positions dead, which only upload_inputs knows; a
    // text request on a --vision engine is plain text (the 0.94 fix: this tested the engine's --vision flag).
    if (engram_ahead_on() && !im.eng_next_tids.empty() && im.n_vis == 0 && !im.eng_ahead) {
        const int nn = (int) im.eng_next_tids.size();
        for (int t = 0; t < nn; ++t) im.engram.set_token(im.eng_next_pos0 + t, im.eng_next_tids[(size_t) t]);
        im.eng_ahead = im.engram.start(im.eng_next_pos0, nn, im.eng_rows, im.err);
        if (!im.eng_ahead) return false;
        im.eng_ahead_pos0 = im.eng_next_pos0;
        im.eng_ahead_n = nn;
        im.eng_ahead_tids = im.eng_next_tids;
    }
    im.eng_next_tids.clear();
    return true;
}

// Before anything else writes eng_rows: a pending read-ahead becomes this pass's batch when it is exactly this pass, else its
// reads are finished and dropped.  `adopted`: the pass need not start reads.
static bool engram_ahead_take(Ds41Dense::Impl & im, const int * tids, int64_t pos0, int n, bool & adopted) {
    adopted = false;
    if (!im.eng_ahead) return true;
    std::shared_ptr<Ds41Engram::Batch> b = std::move(im.eng_ahead);
    im.eng_ahead.reset();
    // the same pass: its place, its tokens (a read-ahead left by a cancelled read must not serve another request's pass at
    // the same positions) and no pictures (a pass with pictures reads its own)
    if (engram_readahead_blk() > 0) {   // STRATA_ENGRAM_READAHEAD: a pass at the same place with a prefix of its tokens takes it
        // (n > 64: a smaller pass may read its rows synchronously into the current buffer - upload_inputs)
        if (pos0 == im.eng_ahead_pos0 && n > 64 && n <= im.eng_ahead_n && im.n_vis == 0 &&
            (size_t) im.eng_ahead_n == im.eng_ahead_tids.size() && std::equal(tids, tids + n, im.eng_ahead_tids.begin())) {
            im.eng_batch = std::move(b);
            im.eng_cur = im.eng_ahead_buf;
            im.eng_n = n;
            im.eng_upc[0] = im.eng_upc[1] = 0;
            adopted = true;
            return true;
        }
        Ds41Engram::cancel(b);   // nobody takes it: skip the reads not started yet
        return im.engram.wait(b, im.err);
    }
    if (pos0 == im.eng_ahead_pos0 && n == im.eng_ahead_n && im.n_vis == 0 && (size_t) n == im.eng_ahead_tids.size() &&
        std::equal(tids, tids + n, im.eng_ahead_tids.begin())) {
        im.eng_batch = std::move(b);
        im.eng_up[0] = im.eng_up[1] = false;
        adopted = true;
        return true;
    }
    return im.engram.wait(b, im.err);
}

// STRATA_ENC_BOUND: the window mask of the layers past the last kv source.  A row at or past the floor sees nothing
// before it: those rows never ran these layers, so their ring slots hold older positions.  A row before the floor has
// the plain window (0.94 fix: it used to see only itself, so a floor left over from another conversation cut layers
// 21-39 off from every earlier token of a later short request; the scorer's rows before its floor are read by nothing).
static void upload_wmask_hi(Ds41Dense::Impl & im, int n, int pos0) {
    const int64_t SWA = im.SWA, F = im.win_floor;
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = (int64_t) pos0 + t, lo = (F >= 0 && p >= F) ? F : 0;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            wm[(size_t) (t * SWA + j)] = (q < 0 || q < lo) ? NEG_INF : 0.0f;
        }
    }
    dset(im.i_wmask_hi, wm.data(), 0, wm.size() * 4);
}

void Ds41Dense::set_window_floor(int64_t floor, int n, int pos0) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    im.win_floor = floor;
    if (im.i_wmask_hi && n > 0) upload_wmask_hi(im, n, pos0);
}

// STRATA_SNAP_TAIL: the tail ring slots of a pass's (or span's) last min(n, TR) rows - the rows its graphs write, as
// i_wslot for the window ring - and their position tags.  Spans of a layer-major batch are contiguous and every one but
// the last is full, so the union of their writes covers the batch's last TR positions (the tags set here).
static void tail_slots(Ds41Dense::Impl & im, int64_t pbase, int64_t n) {
    const int64_t k = std::min<int64_t>(n, im.TR);
    std::vector<int32_t> ts((size_t) k);
    for (int64_t j = 0; j < k; ++j) {
        const int64_t q = pbase + n - k + j;
        ts[(size_t) j] = (int32_t) (q % im.TR);
        im.ttag[(size_t) (q % im.TR)] = q;
    }
    dset(im.i_tslot, ts.data(), 0, ts.size() * 4);
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
        strata::v41::ds41_embd_row(im.embd, im.embd_type, D, tids[t], im.host_emb.data() + (size_t) (t * D));
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
    if (im.i_wmask_hi) upload_wmask_hi(im, n, pos0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        dset(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    if (im.TR > 0) tail_slots(im, (int64_t) pos0, n);
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
        bool ahead_hit = false;                 // STRATA_ENGRAM_AHEAD: this pass's reads already running
        if (!engram_ahead_take(im, tids, pos0, n, ahead_hit)) return false;
        for (int t = 0; t < n; ++t) {
            if (!img.empty() && img[(size_t) t] > 0.0f) im.engram.set_dead(pos0 + t);   // an image span: n-grams never span it
            else im.engram.set_token(pos0 + t, tids[t]);
        }
        // STRATA_ENGRAM_ASYNC=1: decode / verify passes take the async path below too, so their reads overlap the
        // embeddings upload, the init graph and layer 0 (attention + its expert tier) instead of stalling the pass
        // here (~2.7 ms per verify pass, measured).  Every builder that reads i_eng waits first (engram_ready).
        static const bool eng_async = std::getenv("STRATA_ENGRAM_ASYNC") && std::getenv("STRATA_ENGRAM_ASYNC")[0] == '1';
        if (n <= 64 && !eng_async) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, *im.eng_cur, im.err)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_cur->size() && e < 2; ++e)
                dset(im.i_eng[e], (*im.eng_cur)[e].data(), 0, (*im.eng_cur)[e].size() * 4);
        } else if (!ahead_hit) {
            im.eng_batch = im.engram.start(pos0, n, *im.eng_cur, im.err, engram_readahead_blk());
            im.eng_up[0] = im.eng_up[1] = false;
            im.eng_n = n;
            im.eng_upc[0] = im.eng_upc[1] = 0;
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

void Ds41Dense::engram_next(const int * tids, int n, int pos0) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    im.eng_next_tids.clear();
    // only a prompt-sized chunk (> 64 rows: a smaller pass may read its rows synchronously into eng_rows) of a request
    // without pictures (n_vis: this request's image rows, set_vision / clear_vision)
    if (!engram_ahead_on() || !im.engram.enabled() || im.n_vis > 0 || !tids || n <= 64 || pos0 < 0 || (int64_t) pos0 + n > im.cfg.ctx_max) return;
    im.eng_next_tids.assign(tids, tids + n);
    im.eng_next_pos0 = pos0;
    // STRATA_ENGRAM_READAHEAD: the next chunk's reads start now, into the buffer this pass does not use, queued behind this
    // pass's reads (the pool's queue is FIFO).  Its tokens are recorded first (positions past this pass: this pass's start()
    // hashed its own already, and the next pass sets them again).
    if (engram_readahead_blk() > 0 && !im.eng_ahead) {
        for (int t = 0; t < n; ++t) im.engram.set_token((int64_t) pos0 + t, tids[t]);
        std::vector<std::vector<float>> * buf = im.eng_cur == &im.eng_rows ? &im.eng_rows2 : &im.eng_rows;
        std::string e;
        im.eng_ahead = im.engram.start(pos0, n, *buf, e, engram_readahead_blk());
        if (im.eng_ahead) {
            im.eng_ahead_buf = buf;
            im.eng_ahead_pos0 = pos0;
            im.eng_ahead_n = n;
            im.eng_ahead_tids = im.eng_next_tids;
        }
        im.eng_next_tids.clear();
    }
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.x3_fail = false;   // STRATA_X3: a failure of an earlier pass was reported by that pass
    x3_flush_pending();   // (the previous pass's pending graph, if it ended without a head)
    if (im.x3_fail) { im.x3_fail = false; return false; }
    x3_stats_pass(n);
    im.pos0 = pos0;
    im.n = n;
    im.sink_mid_n = 0;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    span_pass_caps(im, pos0, n, im.pass_cap);
    if (x3_defer(im, 1)) return true;   // STRATA_X3: rides on layer 0's first graph
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
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    im.sink_mid_n = 0;
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
// A span is a whole sub-batch (col0 % S == 0) or, when a sub-batch's graph did not fit, a narrower piece of one
// (attn_router_span): a piece at an unaligned column gets the pass_cap of its own end position.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    if (col0 % im.S == 0) {
        const size_t si = (size_t) (col0 / im.S);
        if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
        im.pass_cap[0] = im.span_cap[si][0];
        im.pass_cap[1] = im.span_cap[si][1];
        im.pass_cap[2] = im.span_cap[si][2];
    } else {
        span_pass_caps(im, im.batch_pos0 + col0, n, im.pass_cap);
    }
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
    if (im.TR > 0) tail_slots(im, pbase, n);
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        dset(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

// attn_router_span / attn_router_span_kv run their span as pieces of at most span_max columns (the whole span until a
// graph does not fit).  A piece whose GRAPH ALLOCATION fails - nothing has run, the state is untouched - is retried as
// halves down to kMinSpan columns, and span_max keeps the width that fitted for the rest of the prompt, so a tight
// card (a long context's graph, another process taking VRAM mid-prompt) makes the prompt a little slower instead of
// failing it.  The pieces walk the columns in order, as the sub-batches do, so every position-indexed state (window
// ring, compressor state, compressed rows, candidate blocks) advances exactly as before; the per-query arithmetic is
// the same, but the dense matmuls see fewer columns, so a split prompt is not bit-identical to an unsplit one (it only
// happens where the unsplit one would have failed).
static constexpr int kMinSpan = 16;
// STRATA_SHARED_PREFIX: layer il's rings to host memory when the span (or piece) that just ran ends at the armed
// position.  Spans walk a layer's columns in order and write the ring in the same graph (build_attn / build_attn_kv_only's
// late set_rows), so the ring now holds exactly the window rows below pc_pos, at their slots (q % RAW).
static void prefix_capture_at(Ds41Dense::Impl & im, int il, int col0, int n) {
    if (im.pc_pos < 0 || (int64_t) im.batch_pos0 + col0 + n != im.pc_pos || (size_t) il >= im.pc.size()) return;
    const auto & L = im.ly[(size_t) il];
    auto & c = im.pc[(size_t) il];
    dsync_be(im.backend);
    c.raw.resize(ggml_nbytes(L.raw));
    dget(L.raw, c.raw.data(), 0, c.raw.size());
    if (L.st_kv) {
        c.st_kv.resize(ggml_nbytes(L.st_kv));
        dget(L.st_kv, c.st_kv.data(), 0, c.st_kv.size());
        c.st_sc.resize(ggml_nbytes(L.st_sc));
        dget(L.st_sc, c.st_sc.data(), 0, c.st_sc.size());
    }
    c.got = true;
}
static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool run_spans(Ds41Dense::Impl & im, int il, int col0, int n, const std::function<bool(int, int)> & one) {
    int64_t w = im.span_max > 0 ? std::min<int64_t>(im.span_max, n) : n;
    for (int c = col0; c < col0 + n; ) {
        const int m = (int) std::min<int64_t>(w, (int64_t) col0 + n - c);
        im.alloc_failed = false;
        if (one(c, m)) { c += m; continue; }
        if (!im.alloc_failed || m <= kMinSpan) return false;
        w = std::max(kMinSpan, m / 2);
        im.span_max = w;
        std::fprintf(stderr, "ds41_dense: layer %d: the graph of a %d-column span did not fit - %lld-column spans for the rest "
                     "of this prompt\n", il, m, (long long) w);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_kv_one(im, il, c, m); });
}

static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot, col0, n)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    if (okk && im.pc_pos >= 0) prefix_capture_at(im, il, col0, n);   // STRATA_SHARED_PREFIX
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
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_attn_one(im, il, c, m); });
}

static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot, col0, n)) return false;   // wait the pass's Engram reads here
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
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D); a wide
    // batch whose tier call reads fn on the card (batch_fn_dev) skips the fn copy (set_batch_fn_host(false))
    if (im.fn_host)
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
    if (im.pc_pos >= 0) prefix_capture_at(im, il, col0, n);   // STRATA_SHARED_PREFIX
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    dset(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

bool Ds41Dense::wait_event(void * cuda_event) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (!cuda_event || im.cpu) return true;
    // a ggml_backend_event around a foreign cudaEvent_t: the CUDA backend's event_wait is cudaStreamWaitEvent on its
    // own stream with event->context as the event
    ggml_backend_event ev { ggml_backend_get_device(im.backend), cuda_event };
    ggml_backend_event_wait(im.backend, &ev);
    return true;
}

bool Ds41Dense::record_event(void * cuda_event) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (!cuda_event || im.cpu) return false;
    // as wait_event: the CUDA backend's event_record is cudaEventRecord on its own stream with event->context
    ggml_backend_event ev { ggml_backend_get_device(im.backend), cuda_event };
    ggml_backend_event_record(&ev, im.backend);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
const float * Ds41Dense::batch_fn_dev() const { return p_->cpu || !p_->o_fn ? nullptr : (const float *) p_->o_fn->data; }
void Ds41Dense::set_batch_fn_host(bool on) { p_->fn_host = on; }

// STRATA_BM_PREFILL_READAHEAD (Big-MoE-Small-Box's warm_prefill.py read-ahead): which experts layer next_il will route,
// guessed from layer il's routed-expert input.  o_fn holds rms_norm(hc_pre(h)) * ffn_norm(il); the guess swaps the
// norm weight for next_il's (the ratio is folded into an F32 copy of next_il's router, [D, NE], not the [D, n] rows)
// and ranks by the same selection score build_attn uses (sqrt(softplus) + exp_probs_b + rbias).  Only which experts the
// tier reads EARLY depends on it; build_attn(next_il) routes from its own input as always.  A one-off graph through
// allo_pass (the span graphs' allocator, which already holds the wider attention graph) - no cached state.
bool Ds41Dense::guess_route(int il, int next_il, int col0, int n, int k, int32_t * ids) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (im.cpu || !im.o_fn || !ids || n < 1 || k < 1 || (int64_t) k > im.NE || col0 < 0 || (int64_t) col0 + n > im.batch ||
        il < 0 || next_il < 0 || il >= (int) im.ly.size() || next_il >= (int) im.ly.size() || !im.ly[(size_t) next_il].rbias)
        return false;
    auto nm = [](int l, const char * s) { return "blk." + std::to_string(l) + "." + s; };
    if (!im.w.has(nm(next_il, "ffn_gate_inp.weight")) || !im.w.has(nm(next_il, "exp_probs_b.bias")) ||
        !im.w.has(nm(next_il, "ffn_norm.weight")) || !im.w.has(nm(il, "ffn_norm.weight")))
        return false;
    // the elementwise ops below want F32 operands (the file's router and norms are F32; a llama.cpp GGUF's BF16 router
    // is converted to F32 at load) - anything else: no guess rather than a ggml assert
    if (im.BL(next_il, "ffn_gate_inp.weight")->type != GGML_TYPE_F32 || im.BL(next_il, "ffn_norm.weight")->type != GGML_TYPE_F32 ||
        im.BL(il, "ffn_norm.weight")->type != GGML_TYPE_F32)
        return false;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    // ffn_norm(next) / ffn_norm(il), clamped: a zero weight in il's norm would otherwise put inf into the guess
    ggml_tensor * ratio = ggml_clamp(gc, ggml_div(gc, im.BL(next_il, "ffn_norm.weight"), im.BL(il, "ffn_norm.weight")), -64.0f, 64.0f);
    ggml_tensor * wr = ggml_mul(gc, im.BL(next_il, "ffn_gate_inp.weight"), ratio);             // [D, NE]
    ggml_tensor * rl = ggml_mul_mat(gc, wr, b.cols2(im.o_fn, im.D, n, col0));                  // [NE, n]
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    ggml_tensor * o = ggml_argsort_top_k(gc, sc, k);   // a VIEW of the full argsort: rows nb[1] apart (read_pred_rows)
    ggml_set_output(o);
    ggml_build_forward_expand(gf, o);
    bool ok = im.compute(gf, false);
    if (ok) {
        const size_t rs = o->nb[1] / sizeof(int32_t);
        std::vector<int32_t> span(rs * (size_t) (n - 1) + (size_t) k);
        dread(im, o, span.data(), span.size() * sizeof(int32_t));
        for (int t = 0; t < n; ++t)
            std::copy(span.begin() + (ptrdiff_t) (rs * (size_t) t), span.begin() + (ptrdiff_t) (rs * (size_t) t + (size_t) k),
                      ids + (size_t) t * (size_t) k);
    }
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::shexp_launch(int il) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
    const bool late = n > 1 && shexp_late_on() && !im.cpu;       // ... or, on a multi-row pass, from shexp_late_launch
    im.shexp_pending = false;
    if (!im.sink_host.empty()) sink_mid_tap(im, il);   // STRATA_SINK_TRACE: the state layer il - 1 left
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0) x3_flush_pending();   // STRATA_X3: the card runs the pending graph meanwhile
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    bool cached = n == 1;
    const bool x3_noshexp = n == 1 ? split : late;
    Impl::MGraph * x3g = (n == 1 || pass_graphs_on(n))   // STRATA_X3: the pending begin / finish + this attention
        ? x3_take(im, kX3Attn + (x3_noshexp ? 2 : 0), il, cap, n, [&](ggml_context * c, ggml_tensor **, ggml_cgraph * g) {
              build_attn(im, c, il, n, 0, x3_noshexp, g); })
        : nullptr;
    if (x3g) {
        gf = x3g->gf;
        cached = true;
        if (n == 1) im.shexp_pending = split;
    } else if (cached) {
        const Impl::GKey key { il, split ? 8 : 1, cap, 1 };   // the split graph is its own cache entry
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1, 0, split)).first;
        gf = it->second;
        im.shexp_pending = split;
    } else if (pass_graphs_on(n)) {
        gf = pass_graph(im, { il, 1, cap, n }, [&](ggml_context * c, ggml_tensor **) { return build_attn(im, c, il, n, 0, late); }).gf;
        cached = true;
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n, 0, late);
    }
    const bool ok = x3g ? x3_run(im, *x3g, false) : im.compute(gf, cached, false);   // STRATA_DENSE_ASYNC: the readbacks' wait below is the only one
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
    if (late && !shexp_late_launch(im, il, n)) return false;   // behind the readback; the caller's tier work runs beside it
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
    int buf[16];
    if (Impl::MGraph * g = x3_take(im, kX3Pred, il, 0, 1, [&](ggml_context * c, ggml_tensor ** o, ggml_cgraph * gf) {
            build_predict(im, c, il, o, 1, gf); })) {   // STRATA_X3: the pending begin / finish + this prediction
        if (!x3_run(im, *g, false)) return false;
        read_pred_rows(im, g->out, 1, buf);
        for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
        return true;
    }
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true, false)) return false;
    read_pred_rows(im, im.predict_out[il], 1, buf);   // the device view, or STRATA_ZC_PRED's host floats
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    if (Impl::MGraph * g = x3_take(im, kX3Pred, il, 0, n, [&](ggml_context * c, ggml_tensor ** o, ggml_cgraph * gf) {
            build_predict(im, c, il, o, n, gf); })) {   // STRATA_X3: the pending begin / finish + this prediction
        if (!x3_run(im, *g, false)) return false;
        read_pred_rows(im, g->out, n, top_ids);
        return true;
    }
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
    // argsort_top_k is a VIEW of the full argsort: row t starts at t * nb[1] (NE ids apart) - read_pred_rows reads the
    // span that covers every row's first 16.  (Until 2026-10-21 this read 16 * n contiguous ids: rows 1.. were row 0's
    // ranks 16.., which the top-k kernel never writes - stale ids, so the verify pass prefetched on row 0 alone.)
    read_pred_rows(im, o, n, top_ids);
    return true;
}

bool Ds41Dense::predict_multi2(int il, int * top_ids, int * next_ids) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
    read_pred_rows(im, im.predict_out_n[k1], n, top_ids);
    read_pred_rows(im, im.predict_out_n[k2], n, next_ids);
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    (void) il;
    const int64_t n = im.n;
    if (routed_sum) dset(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    if (x3_defer(im, 2)) return true;   // STRATA_X3: rides on the next layer's first graph (or the head)
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
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
    if (!im.sink_host.empty()) sink_mid_tap(im, il);   // STRATA_SINK_TRACE: the state layer il - 1 left
    if (im.ly[(size_t) il].eslot >= 0) x3_flush_pending();   // STRATA_X3: the card runs the pending graph meanwhile
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im, (int) im.ly[(size_t) il].eslot)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    {   // STRATA_X3: the pending begin / finish + this attention (attn_router's merged graph for the same width)
        const bool no_shexp = n > 1 && shexp_late_on() && !im.cpu;
        if (Impl::MGraph * g = x3_take(im, kX3Attn + (no_shexp ? 2 : 0), il, cap, n, [&](ggml_context * c, ggml_tensor **, ggml_cgraph * gf) {
                build_attn(im, c, il, n, 0, no_shexp, gf); }))
            return x3_run(im, *g, false);
    }
    if (n > 1) {   // the same cached pass graph attn_router uses
        const bool late = shexp_late_on() && !im.cpu;
        return im.compute(pass_graph(im, { il, 1, cap, n }, [&](ggml_context * c, ggml_tensor **) { return build_attn(im, c, il, n, 0, late); }).gf,
                          true, false);
    }
    const Impl::GKey key { il, 1, cap, 1 };   // the same cached graph attn_router uses
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1)).first;
    return im.compute(it->second, true, false);
}

// a pinned id buffer + its event (predict_multi_launch's, and X3B's spec prediction's)
static bool pinned_ids(Ds41Dense::Impl & im, ggml_backend_buffer_t & buf, int32_t *& host, ggml_backend_event_t & ev) {
    if (host) return true;
#ifdef DS41_HAVE_CUDA
    buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 65536);
#else
    (void) buf; (void) ev;
    im.err = "predict_multi_launch needs the CUDA build";
    return false;
#endif
    if (!buf) { im.err = "pinned prediction buffer"; return false; }
    host = (int32_t *) ggml_backend_buffer_get_base(buf);
    ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
    if (!ev) { im.err = "prediction event"; return false; }
    return true;
}

// layer il's prediction graph for the pass width, launched without a host wait (*po: its output) - the graph predict /
// predict_multi launch for il: the X3-merged one when a begin / finish is pending, else the plain cached one
static bool pred_launch(Ds41Dense::Impl & im, int il, ggml_tensor ** po) {
    const int64_t n = im.n;
    ggml_tensor * o = nullptr;
    if (Ds41Dense::Impl::MGraph * g = x3_take(im, kX3Pred, il, 0, n, [&](ggml_context * c, ggml_tensor ** pp, ggml_cgraph * gf) {
            build_predict(im, c, il, pp, n, gf); })) {   // STRATA_X3: the pending begin / finish + this prediction
        o = g->out;
        if (!x3_run(im, *g, false)) return false;
    } else {
        // the graph predict / predict_multi use for this width
        const Ds41Dense::Impl::GKey key { il, 3, 0, (int) n };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
            it = im.graphs.emplace(key, gf).first;
            if (n == 1) im.predict_out[il] = o; else im.predict_out_n[{ il, (int) n }] = o;
        }
        o = n == 1 ? im.predict_out[il] : im.predict_out_n[{ il, (int) n }];
        if (!im.compute(it->second, true, false)) return false;
    }
    *po = o;
    return true;
}

// queue a prediction's readback (the span covering every row's first 16) into pinned memory behind an event.
// STRATA_ZC_PRED (o is the F32 [16, n] zero-copy host tensor): the graph's last kernel already writes the ids (as floats)
// into pinned host memory - there is nothing to copy, and o is not in a CUDA device buffer, so ggml_backend_tensor_get_async
// would abort ("unsupported buffer type").  The event after the graph marks the kernel's writes done; *zc
// points the waiter (predict_multi_wait / x3b_spec_wait) at the floats, which it reads only after the event
static bool pred_readback(Ds41Dense::Impl & im, ggml_tensor * o, int32_t * host, ggml_backend_event_t ev, int64_t & rows,
                          int64_t & rs_out, const float *& zc) {
    const int64_t n = im.n;
    if (o->type == GGML_TYPE_F32) {
        zc = (const float *) o->data;
        ggml_backend_event_record(ev, im.backend);
        rows = n;
        rs_out = 16;
        return true;
    }
    zc = nullptr;
    const int64_t rs = (int64_t) (o->nb[1] / sizeof(int32_t));   // a VIEW of the full argsort (see predict_multi)
    const size_t span = (size_t) (rs * (n - 1) + 16);
    if (span * sizeof(int32_t) > 65536) { im.err = "predict_multi_launch: pass too wide"; return false; }
    ggml_backend_tensor_get_async(im.backend, o, host, 0, span * sizeof(int32_t));
    ggml_backend_event_record(ev, im.backend);
    rows = n;
    rs_out = rs;
    return true;
}

bool Ds41Dense::predict_multi_launch(int il) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1 || im.cpu) return false;
    if (!pinned_ids(im, im.pm_host_buf, im.pm_host, im.pm_ev)) return false;
    ggml_tensor * o = nullptr;
    if (!pred_launch(im, il, &o)) return false;
    return pred_readback(im, o, im.pm_host, im.pm_ev, im.pm_rows, im.pm_rs, im.pm_zc);
}

bool Ds41Dense::predict_multi_wait(int * top_ids) {
    Impl & im = *p_;
    if (!im.pm_ev || im.pm_rows < 1) { im.err = "no prediction pending"; return false; }
    ggml_backend_event_synchronize(im.pm_ev);
    if (im.pm_zc) {   // STRATA_ZC_PRED: ids < 2^24 written as floats - exact both ways (read_pred_rows does the same)
        for (int64_t i = 0; i < 16 * im.pm_rows; ++i) top_ids[i] = (int) im.pm_zc[i];
        im.pm_rows = 0;
        return true;
    }
    for (int64_t t = 0; t < im.pm_rows; ++t)
        std::copy(im.pm_host + im.pm_rs * t, im.pm_host + im.pm_rs * t + 16, top_ids + 16 * t);
    im.pm_rows = 0;
    return true;
}

bool Ds41Dense::attn_router_collect(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    const int64_t n = im.n;
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {   // as attn_router: the smart router's exact picks, for the route meter
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    dsync_be(im.backend);
    if (n > 1 && shexp_late_on() && !im.cpu && !shexp_late_launch(im, il, n)) return false;   // (attn_router's, on this path)
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

// ================================================================== STRATA_X3B
// Per decode layer the host is blocked on the card twice - the prediction readback (predict_multi_wait) and the router
// readback (attn_router_collect) - and after the second it still has the tier call to make.  Everything the dense
// stream runs after the routed sum - finish(l), prediction(l + 1) + its readback, attention(l + 1) - was launched only
// after that tier call returned, so its host launch cost (~1.6 us per kernel node on WDDM) sat between the sum landing
// and the next dense kernel.  X3B launches those three while the host still waits for the router readback, behind a
// device-side gate (ds41_x3b_gate.hpp): the dense stream waits on an event recorded behind the gate, and the gate opens
// when the tier's routed sum has landed (x3b_release, behind the tier's sum event).
//
// Exact: the dense stream's queue is the same - attention(l), the readbacks, the late shared expert, a wait for the
// routed sum, finish(l) (+ prediction(l + 1) when X3 merges them), the prediction readback + its event, attention(l + 1) -
// with the event wait on the sum replaced by an event wait on the gate that opens behind the sum.  With the file-tier
// D3 spec prediction (STRATA_D3_SPEC; 11-17) layer l + 1's iteration starts with predict_multi(l + 1 + ahead): it is
// queued in the same place - finish(l) (+ that prediction when X3 merges them: the same {l + 1 + ahead, kX3Pred + 1}
// graph the normal path launches), its readback, then prediction(l + 1) (plain), its readback, attention(l + 1) - and
// only its host wait moves (a stream-ordered readback + event, collected by x3b_spec_wait where predict_multi waited).  The same cached graphs
// (x3b_prelaunch only runs when every graph it will launch is allocated at the current epoch: nothing is built,
// allocated or synced behind the gate), the same buffers, the same kernels in the same order.  The host-side order of
// readbacks is unchanged too: layer l's router readback is consumed before the tier call that opens the gate, and the
// pre-launched layer l + 1 work writes only device buffers until its own readbacks are issued (collect_issue(l + 1),
// after the tier call).  The prefetch of layer l + 1 still waits for prediction(l + 1)'s readback (behind the gate).
// Never a hang: a host stream sync while armed opens the gate early and fails the pass (x3b_sync_hit), and the default
// spin gate gives up after STRATA_X3B_TIMEOUT_MS (default 2000) and fails the pass (x3b_check).
static bool x3b_env_on() {
    static const bool on = env_on("STRATA_X3B", false);
    return on;
}
static bool x3b_stats_env() {
    static const bool on = env_on("STRATA_X3B_STATS", false);
    return on;
}

#ifdef DS41_HAVE_X3B_GATE
static Ds41Dense::Impl * g_x3b_im = nullptr;   // the dense half that armed the gate (x3b_sync_hit reaches it through this)

// the gate, created on first use (STRATA_X3B_GATE=spin (default) | wait; STRATA_X3B_TIMEOUT_MS: the spin gate's limit)
static bool x3b_gate_ensure(Ds41Dense::Impl & im) {
    if (im.x3b_gate) return true;
    if (im.x3b_dead || im.cpu) return false;
    const char * dn = ggml_backend_dev_name(ggml_backend_get_device(im.backend));   // "CUDA<n>"
    int dev = 0;
    if (dn) for (const char * p = dn; *p; ++p) if (*p >= '0' && *p <= '9') { dev = std::atoi(p); break; }
    const char * gm = std::getenv("STRATA_X3B_GATE");
    const int mode = gm && std::strcmp(gm, "wait") == 0 ? 1 : 0;
    const char * to = std::getenv("STRATA_X3B_TIMEOUT_MS");
    const double timeout_ms = to && *to ? std::atof(to) : 1000.0;
    char e[256] = {};
    im.x3b_gate = strata::v41::x3b_gate_create(dev, mode, timeout_ms, e, (int) sizeof(e));
    if (!im.x3b_gate) {
        im.x3b_dead = true;
        std::fprintf(stderr, "ds41_dense: STRATA_X3B: no %s gate on device %d (%s) - X3B off\n", mode ? "wait-value" : "spin", dev, e);
        return false;
    }
    if (mode) std::fprintf(stderr, "ds41_dense: STRATA_X3B%s: wait gate on device %d\n", x3b_env_on() ? "" : "_STATS (timing only)", dev);
    else std::fprintf(stderr, "ds41_dense: STRATA_X3B%s: spin gate on device %d (timeout %.0f ms)\n", x3b_env_on() ? "" : "_STATS (timing only)",
                      dev, timeout_ms);
    return true;
}
#endif

namespace {
void x3b_sync_hit() {
#ifdef DS41_HAVE_X3B_GATE
    std::fprintf(stderr, "ds41_dense: STRATA_X3B: a host stream sync while the gate is armed - opened early; this pass fails\n");
    if (Ds41Dense::Impl * im = g_x3b_im) {
        if (im->x3b_gate) strata::v41::x3b_gate_abort(im->x3b_gate);
        im->x3b_gated = false;
        im->x3b_fail = true;
    }
#endif
    g_x3b_gated = false;
}
}  // namespace

bool Ds41Dense::x3b_on() const {
#ifdef DS41_HAVE_X3B_GATE
    return x3b_env_on() && !p_->cpu && !p_->x3b_dead;
#else
    return false;
#endif
}

bool Ds41Dense::x3b_stats_on() const {
#ifdef DS41_HAVE_X3B_GATE
    return x3b_stats_env() && !p_->cpu && !p_->x3b_dead;
#else
    return false;
#endif
}

bool Ds41Dense::attn_router_collect_issue(int il) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    const int64_t n = im.n;
    if (im.cpu) { im.err = "attn_router_collect_issue needs the CUDA build"; return false; }
    if (!im.rb_ev) {
        im.rb_ev = ggml_backend_event_new(ggml_backend_get_device(im.backend));
        if (!im.rb_ev) { im.err = "readback event"; return false; }
    }
    // attn_router_collect's readbacks, in its order, then an event instead of the stream sync
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    ggml_backend_event_record(im.rb_ev, im.backend);
    // the late shared expert: behind the readbacks on the stream, as in attn_router_collect
    if (n > 1 && shexp_late_on() && !shexp_late_launch(im, il, n)) return false;
    return true;
}

bool Ds41Dense::attn_router_collect_wait(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    (void) il;
    Impl & im = *p_;
    const int64_t n = im.n;
    if (!im.rb_ev) { im.err = "attn_router_collect_wait: nothing issued"; return false; }
    ggml_backend_event_synchronize(im.rb_ev);   // the readbacks only - not what was queued behind them
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

int Ds41Dense::x3b_prelaunch(int il, int spec_il) {
#ifndef DS41_HAVE_X3B_GATE
    (void) il; (void) spec_il;
    return 0;
#else
    Impl & im = *p_;
    if (!x3b_on() || im.x3b_gated) return 0;
    const int64_t n = im.n;
    const int nx = il + 1;
    // only what launches without a host wait: async dense launches, cached graphs, nothing pending, the pass's
    // prediction already collected, no Engram wait in front of the next attention
    if (n < 1 || nx >= (int) im.ly.size() || !dense_async() || !(n == 1 || pass_graphs_on(n)) || im.x3_pend ||
        !im.pm_host || im.pm_rows > 0 || im.ly[(size_t) nx].eslot >= 0)
        return 0;
    // STRATA_X3B_SKIP_ENGRAM=1 (default off since 11-18): not at an Engram layer itself.  Measured: with layers 1 / 14
    // skipped the stall moved to layer 2 - it was the pass's first gate after a batch prompt, not the Engram layer (the
    // cause: lazy CUDA module loading, see ds41_generate's main).  Kept as a switch, counted on the x3b: line.
    static const bool skip_engram = env_on("STRATA_X3B_SKIP_ENGRAM", false);
    if (im.ly[(size_t) il].eslot >= 0 && skip_engram) { ++im.x3b_declined_engram; return 0; }
    // spec_il >= 0: layer nx's iteration starts with the D3 spec prediction predict_multi(spec_il) (STRATA_D3_SPEC with
    // a file tier) - queued here too, in its place.  STRATA_ZC_PRED too: predict_multi reads its zero-copy floats after a
    // stream sync (read_pred_rows' dsync_be - never behind a gate), but here nothing syncs the stream: pred_readback
    // records xs_ev after the graph whose last kernel writes them, and x3b_spec_wait reads them only after that event
    // (as predict_multi_launch / predict_multi_wait do with pm_zc / pm_ev)
    const bool spec = spec_il >= 0;
    if (spec && spec_il >= (int) im.ly.size()) return 0;
    static const bool reuse_on = env_on("DS41_GRAPH_REUSE", true);
    if (!reuse_on) return 0;
    // every graph the three calls below will launch must exist and be allocated at the current epoch (compute() then
    // neither builds, allocates nor syncs) - else decline: the normal path runs this layer and warms them
    auto mg = [&](const Impl::GKey & k) -> ggml_cgraph * { auto it = im.mgraphs.find(k); return it == im.mgraphs.end() ? nullptr : it->second.gf; };
    auto pg = [&](const Impl::GKey & k) -> ggml_cgraph * { auto it = im.graphs.find(k); return it == im.graphs.end() ? nullptr : it->second; };
    const int64_t r = im.ly[(size_t) nx].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    // The graphs layer nx's iteration launches, in its order.  X3 on: finish defers (x3_defer) and the FIRST prediction
    // after it carries it (x3_take: kX3Pred + finish) - the spec prediction when there is one (11-17: before, X3B
    // asked for {nx, kX3Pred + 1}, which the normal path never launched with a spec prediction in front, so it never
    // replayed and X3B never engaged), else predict_multi_launch(nx); the other prediction is the plain graph.  X3 off:
    // finish alone, then the plain ones.
    ggml_cgraph * need[4] = {};
    bool merged[4] = {};
    int nneed = 0;
    const int first = spec ? spec_il : nx;
    if (x3_on(im)) {
        merged[nneed] = true;
        need[nneed++] = mg({ first, kX3Pred + 1, 0, n });
    } else {
        need[nneed++] = n == 1 ? pg({ -1, 2, 0, 1 }) : mg({ -1, 2, 0, n });
        need[nneed++] = pg({ first, 3, 0, (int) n });
    }
    if (spec) need[nneed++] = pg({ nx, 3, 0, (int) n });
    need[nneed++] = n == 1 ? pg({ nx, 1, cap, 1 }) : mg({ nx, 1, cap, n });
    // ... and ggml-cuda must REPLAY each of them (a plain cudaGraphLaunch).  Measured: with readiness alone, a graph's
    // 2nd launch at an allocation - ggml-cuda's warmup end: stream capture, cudaGraphInstantiate - or the first launch
    // after its 10 s eviction ran inside the armed window; a spin gate timed out about once per request (a driver call
    // there waits for the device, which waits for the gate the host has not released).  ggml-cuda executes a graph's
    // first launch directly, captures + instantiates at the second (same uid: no property change), replays from the
    // third, and evicts a graph unused for 10 s; so: >= 3 launches of this allocation, the last one < 4 s ago.
    const double tnow = x3b_now_ms();
    int bad = 0, bad_merged = 0;
    bool missing = false;
    for (int i = 0; i < nneed; ++i) {
        if (!need[i] || !im.allo_dec.ready(need[i], true)) { missing = true; ++bad; bad_merged += merged[i]; continue; }
        auto u = im.x3b_use.find(need[i]->uid);
        if (u == im.x3b_use.end() || u->second.n < 3 || tnow - u->second.t > 4000.0) { ++bad; bad_merged += merged[i]; }
    }
    if (bad) {
        if (!missing) ++im.x3b_declined_cold;
        if (bad == bad_merged) ++im.x3b_declined_merged;   // every other graph ready and replaying
        return 0;
    }
    if (!x3b_gate_ensure(im)) return 0;
    if (spec && !pinned_ids(im, im.xs_host_buf, im.xs_host, im.xs_ev)) return -1;   // (before the arm: no host alloc behind it)
    strata::v41::x3b_gate_note(im.x3b_gate, il, strata::v41::X3B_PH_LAUNCH);   // (the arm records the layer)
    void * open_ev = strata::v41::x3b_gate_arm(im.x3b_gate);
    if (!open_ev) { im.err = "STRATA_X3B: gate arm failed"; return -1; }
    {   // the dense stream waits for the gate (as wait_event: the CUDA backend's event_wait is cudaStreamWaitEvent)
        ggml_backend_event ev { ggml_backend_get_device(im.backend), open_ev };
        ggml_backend_event_wait(im.backend, &ev);
    }
    im.x3b_gated = true;
    im.x3b_arm_layer = il;
    g_x3b_gated = true;
    g_x3b_im = &im;
    if (x3b_stats_env()) im.x3b_go_pending = il;   // recorded right before the first launch below: when the gate opens
    // the very calls the caller would make after the tier call - finish(il), then layer il + 1's launches: the spec
    // prediction (predict_multi(spec_il)'s graph, its readback queued at the same point - into pinned memory behind an
    // event instead of a host wait: x3b_spec_wait), the prediction, the attention
    if (!finish(il, nullptr)) return -1;   // (the caller aborts)
    if (spec) {
        ggml_tensor * o = nullptr;
        if (!pred_launch(im, spec_il, &o) || !pred_readback(im, o, im.xs_host, im.xs_ev, im.xs_rows, im.xs_rs, im.xs_zc)) return -1;
    }
    if (!predict_multi_launch(nx) || !attn_router_launch(nx)) return -1;
    return spec ? 2 : 1;
#endif
}

bool Ds41Dense::x3b_spec_wait(int * top_ids) {
    Impl & im = *p_;
    if (!im.xs_ev || im.xs_rows < 1) { im.err = "STRATA_X3B: no spec prediction pending"; return false; }
    ggml_backend_event_synchronize(im.xs_ev);   // the readback only (as predict_multi_wait)
    if (im.xs_zc) {   // STRATA_ZC_PRED: the graph's kernel wrote the ids as floats; the event marks it done
        for (int64_t i = 0; i < 16 * im.xs_rows; ++i) top_ids[i] = (int) im.xs_zc[i];
        im.xs_rows = 0;
        return true;
    }
    for (int64_t t = 0; t < im.xs_rows; ++t)
        std::copy(im.xs_host + im.xs_rs * t, im.xs_host + im.xs_rs * t + 16, top_ids + 16 * t);
    im.xs_rows = 0;
    return true;
}

bool Ds41Dense::x3b_release(void * sum_event, int il) {
    Impl & im = *p_;
#ifdef DS41_HAVE_X3B_GATE
    if (!im.x3b_gated || !im.x3b_gate) {
        im.err = im.x3b_fail ? "STRATA_X3B: the gate opened early (a host sync while armed)" : "STRATA_X3B: release without an armed gate";
        return false;
    }
    if (il != im.x3b_arm_layer) {   // invariant: the release is the armed layer's
        im.err = "STRATA_X3B: release for layer " + std::to_string(il) + " but layer " + std::to_string(im.x3b_arm_layer) + " is armed";
        x3b_abort();
        im.x3b_fail = true;
        return false;
    }
    const bool ok = strata::v41::x3b_gate_release(im.x3b_gate, sum_event);
    im.x3b_gated = false;
    im.x3b_arm_layer = -1;
    g_x3b_gated = false;
    if (!ok) { im.err = "STRATA_X3B: gate release failed (or the watchdog had opened it)"; im.x3b_fail = true; }
    return ok;
#else
    (void) sum_event; (void) il;
    im.err = "STRATA_X3B: not built";
    return false;
#endif
}

void Ds41Dense::x3b_abort() {
#ifdef DS41_HAVE_X3B_GATE
    Impl & im = *p_;
    if (!im.x3b_gated) return;
    if (im.x3b_gate) strata::v41::x3b_gate_abort(im.x3b_gate);
    im.x3b_gated = false;
    im.x3b_arm_layer = -1;
    g_x3b_gated = false;
#endif
}

void Ds41Dense::x3b_note(int il, int phase) {
#ifdef DS41_HAVE_X3B_GATE
    if (p_->x3b_gate) strata::v41::x3b_gate_note(p_->x3b_gate, il, phase);
#else
    (void) il; (void) phase;
#endif
}

bool Ds41Dense::x3b_armed() const {
#ifdef DS41_HAVE_X3B_GATE
    return p_->x3b_gated || (p_->x3b_gate && strata::v41::x3b_gate_armed(p_->x3b_gate));
#else
    return false;
#endif
}

bool Ds41Dense::x3b_check() {
#ifdef DS41_HAVE_X3B_GATE
    Impl & im = *p_;
    if (im.x3b_fail) { im.x3b_fail = false; im.err = "STRATA_X3B: the gate opened early (a host sync while armed / a failed release)"; return false; }
    const long t = im.x3b_gate ? strata::v41::x3b_gate_timeouts(im.x3b_gate) : 0;
    if (t > im.x3b_timeouts) {
        im.x3b_timeouts = t;
        im.err = "STRATA_X3B: a gate timed out (STRATA_X3B_TIMEOUT_MS; watchdog or spin) - a layer ran before its routed sum; the pass is invalid";
        char d[512];
        strata::v41::x3b_gate_describe(im.x3b_gate, d, (int) sizeof(d));   // which gate, the host's times, the spin's own account
        im.err += d;
        std::fprintf(stderr, "x3b: gate timeout%s\n", d);
        return false;
    }
#endif
    return true;
}

void Ds41Dense::x3b_mark_sum(int il, void * sum_event) {
#ifdef DS41_HAVE_X3B_GATE
    Impl & im = *p_;
    if (!x3b_stats_on() || !x3b_gate_ensure(im)) return;
    strata::v41::x3b_gate_mark(im.x3b_gate, sum_event, 2 * il);
#else
    (void) il; (void) sum_event;
#endif
}

void Ds41Dense::x3b_mark_go(int il) {
    if (!x3b_stats_on()) return;
    p_->x3b_go_pending = il;   // compute() records it right before the dense stream's next launch
}

void Ds41Dense::x3b_stats_pass(int rows, int layers_pre, double host_pre_ms, double host_blocked_ms) {
#ifdef DS41_HAVE_X3B_GATE
    Impl & im = *p_;
    if (!x3b_stats_on() || !im.x3b_gate) return;
    struct S { long passes = 0, rows = 0, pre = 0, gaps = 0; double pre_ms = 0, blk_ms = 0, gap_ms = 0; };
    static S s;
    static const long every = [] { const char * e = std::getenv("STRATA_X3_STATS_EVERY"); return e && *e ? std::max(1L, std::atol(e)) : 64L; }();
    for (int l = 0; l < (int) im.ly.size(); ++l) {   // the head's launch was the last "next start": all recorded by now
        const double g = strata::v41::x3b_gate_elapsed(im.x3b_gate, 2 * l, 2 * l + 1);
        if (g >= 0) { s.gap_ms += g; ++s.gaps; }
    }
    ++s.passes;
    s.rows += rows;
    s.pre += layers_pre;
    s.pre_ms += host_pre_ms;
    s.blk_ms += host_blocked_ms;
    if (s.passes >= every) {
        const double P = (double) s.passes;
        std::fprintf(stderr, "x3b: %ld passes, %.2f rows/pass | pre-launched layers/pass %.1f | host ms/pass launching: pre-launched "
                             "%.3f, while the card waits %.3f | GPU gap routed sum -> next dense start %.3f ms/pass (%.1f us/layer, "
                             "%.1f layers/pass) | %s gate, timeouts %ld, declined (graph not replaying yet) %ld, X3-merged graph only "
                             "(not built / not replaying) %ld, Engram layers skipped %ld | STRATA_X3B %s\n",
                     s.passes, s.rows / P, s.pre / P, s.pre_ms / P, s.blk_ms / P, s.gap_ms / P,
                     s.gaps > 0 ? 1000.0 * s.gap_ms / (double) s.gaps : 0.0, s.gaps / P, strata::v41::x3b_gate_mode(im.x3b_gate),
                     strata::v41::x3b_gate_timeouts(im.x3b_gate), im.x3b_declined_cold, im.x3b_declined_merged, im.x3b_declined_engram,
                     x3b_env_on() ? "on" : "off");
        s = S();
    }
#else
    (void) rows; (void) layers_pre; (void) host_pre_ms; (void) host_blocked_ms;
#endif
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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

// the head's rows: their column after the last layer, read behind the logits (one stream sync for the small copies)
static void sink_capture(Ds41Dense::Impl & im, int row0, int nrows) {
    sink_enqueue(im, 1, row0, nrows);
    dsync_be(im.backend);
    im.sink_n = std::min(nrows, Ds41Dense::kSinkRows);
    im.sink_row0 = row0;
    ++im.sink_seq;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    if (pass_graphs_on(nrows) && row0 < 16) {   // the head of a decode step / verify pass (kind 9, il = row0)
        // STRATA_X3: the pending finish (the last layer's) + the head as one graph
        Impl::MGraph * x3g = x3_take(im, kX3Head, row0, 0, im.n * 4096 + nrows, [&](ggml_context * c, ggml_tensor ** lg, ggml_cgraph * gf) {
            build_head(im, c, row0, nrows, lg, gf); });
        Impl::MGraph & g = x3g ? *x3g : pass_graph(im, { row0, 9, 0, nrows }, [&](ggml_context * c, ggml_tensor ** lg) {
            return build_head(im, c, row0, nrows, lg); });
        if (!(x3g ? x3_run(im, g, true) : im.compute(g.gf, true))) return false;
        dget(g.out, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
        if (!im.sink_host.empty()) sink_capture(im, row0, nrows);
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
    if (ok && !im.sink_host.empty()) sink_capture(im, row0, nrows);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    dget(im.x_state, x.data(), 0, x.size() * 4);
    dget(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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

// STRATA_SNAP_TAIL: a snapshot's window tail in host memory - every layer's tail ring as it was at the snapshot (slot
// = pos % TR), valid for positions [lo, pos).  Shared (not copied) by the park and every SavedConv taken from it.
struct Ds41Dense::SnapTail {
    int64_t pos = -1, lo = -1;
    std::vector<std::vector<uint8_t>> rows;   // [layer][TR * row bytes]; empty = the tail adds nothing to the ring
    size_t bytes = 0;
};

// The lowest EVEN resume position a tail makes exact (its first re-fed query attends [rs - SWA + 1, rs]), or -1.
static int64_t tail_from(const Ds41Dense::Impl & im, const Ds41Dense::SnapTail * st) {
    if (!st || st->rows.empty() || st->lo < 0 || st->lo >= st->pos) return -1;
    const int64_t f = st->lo == 0 ? 2 : st->lo + im.SWA - 1;
    const int64_t fe = (f + 1) & ~(int64_t) 1;
    return fe <= st->pos ? fe : -1;
}

// The live tail ring made to hold the snapshot's rows [lo, upto) again (only slots whose tag says otherwise are written:
// a continuation's answer decode overwrote some).  reset: the ring belongs to another conversation (load_conv / unpark),
// so no tag is trusted.
static void tail_rearm(Ds41Dense::Impl & im, int64_t upto, bool reset) {
    if (im.TR <= 0) return;
    if (reset) std::fill(im.ttag.begin(), im.ttag.end(), (int64_t) -1);
    const Ds41Dense::SnapTail * st = im.snap_tail.get();
    if (!st || st->pos != im.snap_pos || st->rows.empty()) return;
    upto = std::min(upto, st->pos);
    const size_t rb = im.ly[0].tring->nb[1];
    for (int64_t x = std::max<int64_t>(st->lo, 0); x < upto; ) {
        if (im.ttag[(size_t) (x % im.TR)] == x) { ++x; continue; }
        int64_t e = x + 1;   // a run of mismatched positions in consecutive slots
        while (e < upto && (e % im.TR) != 0 && im.ttag[(size_t) (e % im.TR)] != e) ++e;
        const size_t s0 = (size_t) (x % im.TR), cnt = (size_t) (e - x);
        for (size_t l = 0; l < im.ly.size(); ++l)
            dset(im.ly[l].tring, st->rows[l].data() + s0 * rb, s0 * rb, cnt * rb);
        for (int64_t q = x; q < e; ++q) im.ttag[(size_t) (q % im.TR)] = q;
        x = e;
    }
    dsync_be(im.backend);
}

bool Ds41Dense::snapshot(int pos) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    for (auto & L : im.ly) {
        dcopy(L.raw, L.raw_snap);
        if (L.st_kv) { dcopy(L.st_kv, L.st_kv_snap); dcopy(L.st_sc, L.st_sc_snap); }
    }
    dsync_be(im.backend);
    im.snap_pos = pos;
    im.snap_floor = im.win_floor;
    if (im.TR > 0) {   // STRATA_SNAP_TAIL: the tail ring's exact rows below pos, to host memory
        // a buffer nobody else holds is reused; one a park / SavedConv shares stays theirs
        std::shared_ptr<SnapTail> st = im.snap_tail && im.snap_tail.use_count() == 1 ? im.snap_tail : std::make_shared<SnapTail>();
        st->pos = pos;
        // valid: consecutive positions below pos whose slot still holds them (a later write - a rejected draft, another
        // branch - evicts)
        int64_t lo = pos;
        const int64_t lim = std::max<int64_t>(0, (int64_t) pos - im.TR);
        while (lo > lim && im.ttag[(size_t) ((lo - 1) % im.TR)] == lo - 1) --lo;
        st->lo = lo;
        st->bytes = 0;
        if (lo < (int64_t) pos - im.RAW) {   // anything past what the ring itself keeps
            st->rows.resize(im.ly.size());
            for (size_t l = 0; l < im.ly.size(); ++l) {
                const size_t nb = ggml_nbytes(im.ly[l].tring);
                st->rows[l].resize(nb);
                dget(im.ly[l].tring, st->rows[l].data(), 0, nb);
                st->bytes += nb;
            }
            // STRATA_SNAP_TAIL_VERIFY=1 (debug, off by default): the tail's rows of the positions the window ring still
            // holds must be the ring's bit for bit (same source rows, same type, same set_rows) - checked per layer
            static const bool verify = std::getenv("STRATA_SNAP_TAIL_VERIFY") && std::getenv("STRATA_SNAP_TAIL_VERIFY")[0] == '1';
            if (verify) {
                const size_t rb = im.ly[0].raw->nb[1];
                std::vector<uint8_t> ring(ggml_nbytes(im.ly[0].raw));
                long long rows = 0, bad = 0;
                for (size_t l = 0; l < im.ly.size(); ++l) {
                    dget(im.ly[l].raw_snap, ring.data(), 0, ring.size());
                    for (int64_t q = std::max<int64_t>(lo, (int64_t) pos - im.RAW); q < pos; ++q, ++rows)
                        if (std::memcmp(ring.data() + (size_t) (q % im.RAW) * rb, st->rows[l].data() + (size_t) (q % im.TR) * rb, rb)) ++bad;
                }
                std::fprintf(stderr, "snap-tail verify: snapshot %d, kept %lld, %lld ring rows compared (all layers), %lld differ\n",
                             pos, (long long) (pos - lo), rows, bad);
            }
        } else {
            st->rows.clear();
            st->rows.shrink_to_fit();
            st->lo = pos;
        }
        im.snap_tail = std::move(st);
    }
    return true;
}

bool Ds41Dense::restore() {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        dcopy(L.raw_snap, L.raw);
        if (L.st_kv) { dcopy(L.st_kv_snap, L.st_kv); dcopy(L.st_sc_snap, L.st_sc); }
    }
    dsync_be(im.backend);
    im.win_floor = im.snap_floor;   // the next upload_inputs builds the mask from it
    tail_rearm(im, im.snap_pos, false);   // STRATA_SNAP_TAIL (a no-op when off)
    return true;
}

int64_t Ds41Dense::tail_rows() const { return p_->TR; }

int64_t Ds41Dense::snap_tail_from() const {
    const Impl & im = *p_;
    if (im.TR <= 0 || !im.snap_tail || im.snap_tail->pos != im.snap_pos) return -1;
    return tail_from(im, im.snap_tail.get());
}

int64_t Ds41Dense::parked_tail_from() const {
    const Impl & im = *p_;
    if (im.TR <= 0 || im.parked_pos < 0 || !im.parked_tail || im.parked_tail->pos != im.parked_pos) return -1;
    return tail_from(im, im.parked_tail.get());
}

int64_t Ds41Dense::snap_tail_kept() const {
    const Impl & im = *p_;
    return im.TR > 0 && im.snap_tail && !im.snap_tail->rows.empty() ? im.snap_tail->pos - im.snap_tail->lo : 0;
}

bool Ds41Dense::snap_tail_resume(int64_t rs) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    const int64_t f = snap_tail_from();
    if (f < 0) return false;
    const SnapTail * st = im.snap_tail.get();
    if (rs < f || rs > st->pos || (rs & 1)) return false;
    // the window ring: the rows of [rs - SWA + 1, rs) at their slots (q % RAW); restore() made it the snapshot's,
    // whose rows below pos - RAW are gone.  The ratio-2 state ring is not read (rs even: a fresh group), the compressed
    // KV / indexer keys / Engram history are position-indexed (rows below rs untouched), and rows from rs on are re-fed.
    const size_t rb = im.ly[0].raw->nb[1];
    for (int64_t q = std::max<int64_t>(0, rs - im.SWA + 1); q < rs; ) {
        int64_t e = q + 1;   // consecutive in both rings
        while (e < rs && (e % im.RAW) != 0 && (e % im.TR) != 0) ++e;
        const size_t sr = (size_t) (q % im.RAW), s0 = (size_t) (q % im.TR), cnt = (size_t) (e - q);
        for (size_t l = 0; l < im.ly.size(); ++l)
            dset(im.ly[l].raw, st->rows[l].data() + s0 * rb, sr * rb, cnt * rb);
        q = e;
    }
    dsync_be(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

// The snapshot rings + rows [0, positions] of the position-indexed caches (and their spare rows), copied to host
// memory: what park() and save_conv() keep; positions 0 = the rings alone (a snapshot of the SAME conversation, whose
// rows are still in place - STRATA_SERVE_SNAP_STACK).  The caller syncs the backend first.
// rings = false (STRATA_SHARED_PREFIX's take_prefix_capture): the position-indexed rows only, appended to `out`.
static void collect_conv_rows(Ds41Dense::Impl & im, int64_t positions, std::vector<Ds41Dense::Impl::ParkedRows> & out,
                              bool rings = true) {
    if (rings) out.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Ds41Dense::Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        dget(t, r.b.data(), off, n);
        out.push_back(std::move(r));
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t && rings) put(t, 0, ggml_nbytes(t));
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
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    dsync_be(im.backend);
    collect_conv_rows(im, positions, im.parked);
    im.parked_pos = im.snap_pos;
    im.parked_floor = im.snap_floor;
    im.parked_tail = im.snap_tail;   // STRATA_SNAP_TAIL: shared, host memory already
    return true;
}

struct Ds41Dense::SavedConv {
    std::vector<Impl::ParkedRows> rows;
    int snap_pos = -1;
    int64_t snap_floor = -1;   // STRATA_ENC_BOUND: the window floor the snapshot was taken with
    size_t bytes = 0;
    std::shared_ptr<SnapTail> tail;   // STRATA_SNAP_TAIL: the snapshot's window tail (shared; counted in bytes)
};

std::shared_ptr<Ds41Dense::SavedConv> Ds41Dense::save_conv(int64_t positions) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 0) return nullptr;   // 0: the snapshot rings alone
    dsync_be(im.backend);
    auto s = std::make_shared<SavedConv>();
    collect_conv_rows(im, positions, s->rows);
    s->snap_pos = im.snap_pos;
    s->snap_floor = im.snap_floor;
    for (const auto & r : s->rows) s->bytes += r.b.size();
    if (im.TR > 0 && im.snap_tail && im.snap_tail->pos == im.snap_pos) {
        s->tail = im.snap_tail;
        s->bytes += s->tail->bytes;
    }
    return s;
}

bool Ds41Dense::load_conv(const SavedConv & s) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (s.snap_pos < 0) return false;
    dsync_be(im.backend);
    for (const auto & r : s.rows) dset(r.t, r.b.data(), r.off, r.b.size());
    dsync_be(im.backend);
    im.snap_pos = s.snap_pos;
    // its rows are the live ones now: the floor comes with them (restore() sets the same again)
    im.snap_floor = s.snap_floor;
    im.win_floor = s.snap_floor;
    if (im.TR > 0) {   // STRATA_SNAP_TAIL: its tail is the snapshot's; the live tail ring holds another conversation
        im.snap_tail = s.tail;
        tail_rearm(im, 0, true);
    }
    return true;
}

size_t Ds41Dense::saved_bytes(const SavedConv & s) { return s.bytes; }

int64_t Ds41Dense::saved_tail_from(const SavedConv & s) const {
    const Impl & im = *p_;
    if (im.TR <= 0 || !s.tail || s.tail->pos != s.snap_pos) return -1;
    return tail_from(im, s.tail.get());
}

void Ds41Dense::arm_prefix_capture(int64_t pos) {
    Impl & im = *p_;
    im.pc_pos = pos > 0 ? pos : -1;
    if (im.pc_pos < 0) { im.pc.clear(); im.pc.shrink_to_fit(); return; }
    im.pc.assign(im.ly.size(), Impl::PrefixRings{});
}

std::shared_ptr<Ds41Dense::SavedConv> Ds41Dense::take_prefix_capture(int64_t pos, bool need_all) {
    Impl & im = *p_;
    std::shared_ptr<SavedConv> s;
    bool ok = pos > 0 && (pos & 1) == 0 && im.pc_pos == pos && im.pc.size() == im.ly.size();
    for (size_t l = 0; ok && l < im.ly.size(); ++l)
        if (!im.pc[l].got && ((int) l <= im.last_kv || need_all)) ok = false;
    if (ok) {
        dsync_be(im.backend);
        s = std::make_shared<SavedConv>();
        for (size_t l = 0; l < im.ly.size(); ++l) {
            auto & L = im.ly[l];
            auto & c = im.pc[l];
            if (c.got) {   // the rings at pos, into the snapshot rings (load_conv writes them there; restore() makes them live)
                s->rows.push_back({ L.raw_snap, 0, std::move(c.raw) });
                if (L.st_kv) {
                    s->rows.push_back({ L.st_kv_snap, 0, std::move(c.st_kv) });
                    s->rows.push_back({ L.st_sc_snap, 0, std::move(c.st_sc) });
                }
                continue;
            }
            // a layer past the last kv source that the read did not run at pos (enc-skip phase A): a full read does not run
            // it there either, and its rows below the replay start are never read (masked by the STRATA_ENC_BOUND floor,
            // or outside the plain replay's reach) - the live rings stand in, as the stale ring does in a full read
            const std::pair<ggml_tensor *, ggml_tensor *> rr[3] = { { L.raw, L.raw_snap }, { L.st_kv, L.st_kv_snap }, { L.st_sc, L.st_sc_snap } };
            for (const auto & pr : rr) {
                if (!pr.first || !pr.second) continue;
                Impl::ParkedRows r{ pr.second, 0, std::vector<uint8_t>(ggml_nbytes(pr.first)) };
                dget(pr.first, r.b.data(), 0, r.b.size());
                s->rows.push_back(std::move(r));
            }
        }
        collect_conv_rows(im, pos, s->rows, false);   // rows [0, pos] of the position-indexed caches + their spare rows
        s->snap_pos = (int) pos;
        s->snap_floor = -1;   // the resuming read sets its own (ds41_generate.cpp: tail_point)
        for (const auto & r : s->rows) s->bytes += r.b.size();
    }
    arm_prefix_capture(-1);
    return s;
}

void Ds41Dense::set_history(const int * ids, int64_t n, int64_t pos0) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (!im.engram.enabled() || n <= 0) return;
    if (!engram_ready(im)) return;   // a still-pending batch first (none between requests)
    for (int64_t i = 0; i < n; ++i) im.engram.set_token(pos0 + i, ids[i]);
}

bool Ds41Dense::unpark() {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    dsync_be(im.backend);
    for (auto & r : im.parked) dset(r.t, r.b.data(), r.off, r.b.size());
    dsync_be(im.backend);
    im.snap_pos = im.parked_pos;
    im.snap_floor = im.parked_floor;   // as load_conv: the parked rows are live again, their floor with them
    im.win_floor = im.parked_floor;
    if (im.TR > 0) {   // STRATA_SNAP_TAIL: as load_conv
        im.snap_tail = im.parked_tail;
        tail_rearm(im, 0, true);
    }
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
    p_->parked_floor = -1;
    p_->parked_tail.reset();
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    x3_flush_pending();   // STRATA_X3: a pending begin / finish graph launches first (order kept)
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
