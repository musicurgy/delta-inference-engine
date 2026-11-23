// test_ds41_geometry <file.gguf> - P1 gate: geometry + full tensor map + expert tiers of a v41_gguf.py file, and
// mutation checks (a missing layer / a layer the file does not hold must fail the map check).
//
// test_ds41_geometry <file.gguf> --report - day 0 of a new checkpoint (R-1): one pass that lists EVERY problem instead
// of stopping at the first - missing / bad keys and the geometry guards with the values they saw, missing /
// unexpected / mis-shaped / mis-typed tensors - then the model's keys the engine never reads, the tensor type census
// (unknown ids and both meanings of type 43 named) and each (role, type) of the routed experts, flagging the ones the
// grouped expert kernels have no format for.  Run it on today's file and on the new one and diff the two outputs.
// Exit 0 when nothing is wrong.  DS41_ACCEPT_ARCH=<arch> (comma separated) lets the loaders take another architecture.
//
// test_ds41_geometry --selftest - no file: the type table, the architecture whitelist and the report-vs-stop
// behaviour of the geometry guards on synthetic shapes.
#include "ds41_geometry.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace strata;
using namespace strata::v41;

static void print_geometry(const Ds41Geometry & G) {
    std::printf("D %lld layers %lld heads %lld d_head %lld rope %lld q_lora %lld o %lldx%lld swa %lld experts %lld/%lld ff %lld "
                "hc %lld idx %lldx%lld top %lld cand L%lld %lldx%lld eps %g vocab %lld ctx %lld clamp %g\n",
                (long long) G.n_embd, (long long) G.n_layer, (long long) G.n_head, (long long) G.d_head, (long long) G.d_rope,
                (long long) G.r_q, (long long) G.o_groups, (long long) G.o_lora, (long long) G.n_swa, (long long) G.n_expert,
                (long long) G.n_used, (long long) G.n_ff, (long long) G.hc, (long long) G.idx_heads, (long long) G.idx_dim,
                (long long) G.idx_topk, (long long) G.cand_layer, (long long) G.cand_blocks, (long long) G.cand_bsize,
                G.rms_eps, (long long) G.vocab, (long long) G.ctx_len, G.swiglu_limit);
    std::printf("engram layers %zu heads %lld dim %lld ngram %lld pad %lld cvocab %lld token_map %zu primes %zu mult %zu\n",
                G.engram_layers.size(), (long long) G.engram_heads, (long long) G.engram_dim, (long long) G.engram_ngram,
                (long long) G.engram_pad, (long long) G.engram_cvocab, G.engram_token_map.size(), G.engram_primes.size(),
                G.engram_mult.size());
}

// The routed-expert formats the grouped kernels have: src/kernels/cuda/iq_kernels.cu STRATA_GU_FMTS / STRATA_D_FMTS +
// src/kernels/cuda/fork/fork_fmt.cuh STRATA_GU_FMTS_FORK / STRATA_D_FMTS_FORK (Q6_K = 14 only in a
// -DSTRATA_Q6K_EXPERTS build, so not listed).  Kept by hand: update it with those lists.
static const std::set<int> kGuFmts = { 2, 3, 6, 8, 10, 11, 12, 13, 16, 17, 18, 19, 21, 22, 23, 29, 39, 40, 42, 143 };
static const std::set<int> kDFmts = { 2, 3, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 42, 143 };

static int report(const char * path) {
    std::unique_ptr<GgufModel> mp;
    try {
        mp = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & ex) {
        std::printf("PROBLEM [file] cannot open %s: %s\nreport: 1 problem(s)\n", path, ex.what());
        return 1;
    }
    const GgufModel & m = *mp;
    const GgufFile & g = m.meta();
    const std::string arch = ds41_arch(g), P = ds41_prefix(g);
    std::string acc;
    for (const auto & a : ds41_accepted_archs()) acc += (acc.empty() ? "" : ", ") + a;
    std::printf("file %s (%zu shard(s))\n", path, m.size());
    std::printf("arch %s (%s; accepted: %s)  keys %s*  convention %s\n", arch.empty() ? "(none)" : arch.c_str(),
                ds41_arch_accepted(arch) ? "accepted" : "NOT accepted", acc.c_str(), P.c_str(),
                ds41_is_mainline_gguf(m) ? "llama.cpp mainline (Unsloth)"
                : ds41_is_llamacpp_gguf(m) ? "llama.cpp #30817 (ISTA)" : "Xolotl (v41_gguf.py)");
    Ds41Issues iss;
    Ds41Geometry G;
    ds41_read_geometry(m, G, &iss);
    print_geometry(G);
    const size_t n_geo = iss.items.size();
    // the tensor pass predicts one compress ratio a layer: a wrong-length list was reported above; pad / cut it so the
    // pass can still name the tensors
    if (G.n_layer > 0 && (int64_t) G.ratio.size() != G.n_layer) G.ratio.resize((size_t) G.n_layer, 0);
    const std::vector<int64_t> layers = ds41_file_layers(m, G);
    ds41_check_tensors(m, G, layers, &iss);
    size_t problems = iss.items.size();
    std::printf("\n== problems (keys / geometry: %zu, tensors: %zu)\n", n_geo, iss.items.size() - n_geo);
    for (size_t i = 0; i < iss.items.size(); ++i)
        std::printf("PROBLEM [%s] %s\n", i < n_geo ? "geometry" : "tensors", iss.items[i].c_str());

    // keys of the model's own namespace that no reader looked up: new keys of a new checkpoint show here (info)
    std::printf("\n== %s* keys the engine does not read (info)\n", P.c_str());
    size_t unread = 0;
    for (const auto & [k, v] : g.metadata())
        if (k.rfind(P, 0) == 0 && !iss.seen.count(k)) {
            std::printf("unread %s = %s\n", k.c_str(), detail::show(&v).c_str());
            ++unread;
        }
    std::printf("%zu unread\n", unread);

    // every tensor's type over every shard; unknown ids and type 43's two meanings named
    std::printf("\n== tensor types (all shards)\n");
    std::map<uint32_t, size_t> census;
    for (size_t s = 0; s < m.size(); ++s)
        for (const auto & t : m.shard(s).tensors()) {
            ++census[t.type];
            if (!ds41_type_known(t.type)) {
                std::printf("PROBLEM [types] %s: GGUF type %s is not a type this engine reads\n", t.name.c_str(),
                            ds41_type_label(t.type).c_str());
                ++problems;
            } else if (t.type == ty::FP8_43) {
                const Ds41Fp8Layout lay = ds41_type43_layout(m, t, s);
                std::printf("type43 %s: %s\n", t.name.c_str(),
                            lay == Ds41Fp8Layout::Mxfp8 ? "MXFP8 blocks (33 bytes / 32 values; Unsloth's Engram tables)"
                            : lay == Ds41Fp8Layout::F8E4M3 ? "one byte a value (ggml's F8_E4M3) - not read by this engine"
                                                           : "truncated");
            }
        }
    for (const auto & [t, n] : census) std::printf("  %-24s x%zu\n", ds41_type_label(t).c_str(), n);

    // the routed experts: each (role, type) and how many layers / experts use it
    std::printf("\n== routed experts (role, type)\n");
    std::map<std::pair<int, int>, std::pair<std::set<int64_t>, size_t>> roles;   // (0 gate+up / 1 down, type) -> (layers, experts)
    for (int64_t l : layers) {
        if (l < 0 || (size_t) l >= G.experts.size()) continue;
        const auto & X = G.experts[(size_t) l];
        for (int k = 0; k < 2; ++k)
            for (int t : k ? X.d_type : X.gu_type) {
                if (t < 0) continue;   // (its tensors were reported missing)
                auto & r = roles[{ k, t }];
                r.first.insert(l);
                ++r.second;
            }
    }
    for (const auto & [kt, r] : roles) {
        const bool ok = (kt.first ? kDFmts : kGuFmts).count(kt.second) != 0;
        std::printf("%s [experts] %-7s %-22s %zu layer(s), %zu expert(s)%s\n", ok ? "ok     " : "PROBLEM", kt.first ? "down" : "gate+up",
                    ds41_type_label((uint32_t) kt.second).c_str(), r.first.size(), r.second,
                    ok ? "" : " - no grouped-kernel format for this role (tier init would refuse it)");
        problems += ok ? 0 : 1;
    }
    std::printf("\nreport: %zu problem(s)\n", problems);
    return problems ? 1 : 0;
}

static int selftest() {
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & detail = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  ", detail.c_str());
        fails += !ok;
    };
    bool known = true, unknown = true;
    for (uint32_t t : { 0u, 1u, 2u, 8u, 12u, 13u, 14u, 16u, 20u, 24u, 26u, 27u, 29u, 30u, 39u, 40u, 42u, 43u, 143u })
        known = known && ds41_type_known(t);
    for (uint32_t t : { 4u, 15u, 28u, 34u, 35u, 41u, 44u, 100u, 142u, 144u, 145u, 149u, 168u, 1000u })
        unknown = unknown && !ds41_type_known(t);
    check("type table: ggml types up to 42, 43 and PTQ1_0 143 known", known);
    check("type table: 44..142, ik_llama's 144+ and unlaid ggml ids unknown", unknown);
    check("type label: 43", ds41_type_label(43) == "MXFP8 / F8_E4M3 (43)", ds41_type_label(43));
    check("type label: 150", ds41_type_label(150) == "unknown (150)", ds41_type_label(150));
    check("type label: 8", ds41_type_label(8) == "Q8_0 (8)", ds41_type_label(8));
    check("arch: deepseek41 accepted", ds41_arch_accepted("deepseek41"));
    check("arch: empty refused", !ds41_arch_accepted(""));
    if (!std::getenv("DS41_ACCEPT_ARCH")) check("arch: deepseek42 refused without DS41_ACCEPT_ARCH", !ds41_arch_accepted("deepseek42"));
    else std::printf("skip arch: DS41_ACCEPT_ARCH is set (%s); deepseek42 accepted: %d\n", std::getenv("DS41_ACCEPT_ARCH"),
                     (int) ds41_arch_accepted("deepseek42"));
    {   // five broken guards: the loader stops at the first, the report lists all five
        Ds41Geometry G;
        G.n_layer = 4; G.ratio = { 0, 2, 3 }; G.kv_sources = { 1, 9 }; G.index_sources = { 1 };
        G.n_shared = 2; G.hc = 4; G.d_head = 512; G.d_rope = 64; G.n_head = 64; G.o_groups = 0;
        const std::string first = ds41_check_geometry(G);
        Ds41Issues iss;
        const std::string rep = ds41_check_geometry(G, &iss);
        check("guards: loader stops at the first (ratio count)", first == "compress_ratios has 3 entries, block_count is 4", first);
        check("guards: report returns \"\" and lists 5", rep.empty() && iss.items.size() == 5, std::to_string(iss.items.size()));
        check("guards: report's first == the loader's", !iss.items.empty() && iss.items[0] == first);
        for (const auto & s : iss.items) std::printf("      %s\n", s.c_str());
    }
    {   // a layer without a source of its ratio + Engram primes that do not tile their table
        Ds41Geometry G;
        G.n_layer = 3; G.ratio = { 0, 2, 1 }; G.kv_sources = { 1 }; G.index_sources = { 1 };
        G.n_shared = 1; G.hc = 4; G.d_head = 512; G.d_rope = 64; G.n_head = 64; G.o_groups = 8;
        G.engram_layers = { 1 }; G.engram_ngram = 3; G.engram_heads = 2;
        G.engram_primes = { 1, 2, 3, 4 }; G.engram_mult = { 1, 2, 3 }; G.engram_rows = { 11 };
        const std::string first = ds41_check_geometry(G);
        Ds41Issues iss;
        ds41_check_geometry(G, &iss);
        check("guards: source of layer 2 named", first.rfind("layer 2 has no kv/index source of its ratio", 0) == 0, first);
        check("guards: report adds the Engram tiling", iss.items.size() == 2 && iss.items[1].find("primes sum to 10") != std::string::npos,
              iss.items.size() == 2 ? iss.items[1] : std::to_string(iss.items.size()));
        G.engram_rows = { 10 };
        G.ratio = { 0, 2, 2 };
        const std::string good = ds41_check_geometry(G);
        check("guards: a good shape passes", good.empty(), good);
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) { std::fprintf(stderr, "usage: test_ds41_geometry <file.gguf> [--report] | --selftest\n"); return 2; }
    if (argc >= 3 && std::strcmp(argv[2], "--report") == 0) return report(argv[1]);
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & detail = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  ", detail.c_str());
        fails += !ok;
    };
    GgufModel m = GgufModel::open(argv[1]);
    Ds41Geometry G;
    std::string e = ds41_read_geometry(m, G);
    check("geometry", e.empty(), e);
    if (!e.empty()) return 1;
    print_geometry(G);
    check("V4.1 shape", G.n_embd == 5120 && G.n_layer == 40 && G.n_expert == 384 && G.n_used == 6 && G.n_ff == 2304 &&
                        G.r_q == 1280 && G.d_head == 512 && G.idx_heads == 32 && G.idx_dim == 128 && G.hc == 4);
    uint32_t mx = 0;
    for (uint32_t v : G.engram_token_map) mx = std::max(mx, v);
    check("engram token map within compressed vocab", (int64_t) mx < G.engram_cvocab && mx + 1 == (uint32_t) G.engram_cvocab);
    std::printf("sources: kv_src/idx_src per layer:");
    for (int64_t l = 0; l < G.n_layer; ++l) std::printf(" %lld:%lld/%lld", (long long) l, (long long) G.kv_src(l), (long long) G.idx_src(l));
    std::printf("\n");
    const std::vector<int64_t> layers = ds41_file_layers(m, G);
    e = ds41_check_tensors(m, G, layers);
    check("full tensor map", e.empty(), e);
    for (int64_t l : layers) {
        std::map<std::pair<int, int>, int> pairs;
        const auto & X = G.experts[(size_t) l];
        for (int64_t x = 0; x < G.n_expert; ++x) pairs[{ X.gu_type[(size_t) x], X.d_type[(size_t) x] }]++;
        std::printf("L%lld tiers:", (long long) l);
        for (const auto & [p, n] : pairs) std::printf(" %s/%s x%d", ggml_type_name((uint32_t) p.first), ggml_type_name((uint32_t) p.second), n);
        std::printf("\n");
    }
    // mutations: a layer missing from the expectation -> its tensors are unexpected; one the file lacks -> missing
    if (layers.size() > 1) {
        std::vector<int64_t> fewer(layers.begin() + 1, layers.end());
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, fewer);
        check("mutation: unpredicted layer rejected", e.rfind("unexpected tensor", 0) == 0, e);
    }
    if ((int64_t) layers.size() < G.n_layer) {
        std::vector<int64_t> more = layers;
        for (int64_t l = 0; l < G.n_layer; ++l) if (std::find(layers.begin(), layers.end(), l) == layers.end()) { more.push_back(l); break; }
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, more);
        check("mutation: absent layer rejected", e.find("missing") != std::string::npos, e);
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
