// tools/v41/engine/test_ds41_engram_src.cpp - the Engram rows from two sources must be identical: DeepSeek's safetensors
// shards (--engram DIR) vs a GGUF that carries the tables inline (--engram FILE.gguf, packed rows).  CPU + disk only.
//
//   test_ds41_engram_src --gguf MODEL.gguf --a SRC --b SRC --seq tokens.i32 [--rows 20000] [--threads 16] [--approx TOL]
//
// --approx TOL: the sources hold different formats (DeepSeek's FP8 vs a re-quantized IQ4_NL table): no raw check; the
// batch rows of --b against --a's as relative L2 error per row (median / p99 / max printed, max must stay under TOL),
// and --b's own paths (batch reader, row cache, reference reader) still byte for byte against each other.
//
// SRC: a shard folder or a .gguf.  Checks, all byte for byte (memcmp of the raw rows / the dequantized floats):
//   1. read_raw on --rows random rows per slot (plus the first and last row);
//   2. the batch reader (start + wait) over the token file, n = 1, 7, 512, 4096 at several positions, no row cache;
//   3. the same through the row cache (every batch twice: the second pass is served from the cached raw rows);
//   4. the reference reader (rows_reference) on --b against the batch reader on --a.
// Exit 0 and "IDENTICAL" when every check matches.
#include "ds41_engram.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

std::vector<std::string> split_src(const std::string & s) {
#ifdef _WIN32
    const char sep = ';';
#else
    const char sep = ':';
#endif
    std::vector<std::string> v;
    for (size_t a = 0, b; a <= s.size(); a = b + 1) {
        b = s.find(sep, a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(s.substr(a, b - a));
    }
    return v;
}

bool same(const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y) {
    if (x.size() != y.size()) return false;
    for (size_t e = 0; e < x.size(); ++e)
        if (x[e].size() != y[e].size() || std::memcmp(x[e].data(), y[e].data(), x[e].size() * sizeof(float)) != 0) return false;
    return true;
}

// per row of `dim` floats: |x - y| / |x| (rows where x is all zero are skipped)
void row_errs(const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y, int64_t dim,
              std::vector<double> & errs) {
    for (size_t e = 0; e < x.size() && e < y.size(); ++e)
        for (size_t r = 0; r + (size_t) dim <= x[e].size() && r + (size_t) dim <= y[e].size(); r += (size_t) dim) {
            double nd = 0, nx = 0;
            for (int64_t i = 0; i < dim; ++i) {
                const double a = x[e][r + (size_t) i], b = y[e][r + (size_t) i];
                nd += (a - b) * (a - b);
                nx += a * a;
            }
            if (nx > 0) errs.push_back(std::sqrt(nd / nx));
        }
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, a_src, b_src, seq;
    int rows_n = 20000, threads = 16;
    double tol = -1;   // --approx
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nxt = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (k == "--gguf") gguf = nxt();
        else if (k == "--a") a_src = nxt();
        else if (k == "--b") b_src = nxt();
        else if (k == "--seq") seq = nxt();
        else if (k == "--rows") rows_n = std::atoi(nxt().c_str());
        else if (k == "--threads") threads = std::atoi(nxt().c_str());
        else if (k == "--approx") tol = std::atof(nxt().c_str());
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (gguf.empty() || a_src.empty() || b_src.empty() || seq.empty()) {
        std::fprintf(stderr, "usage: test_ds41_engram_src --gguf MODEL.gguf --a SRC --b SRC --seq tokens.i32\n");
        return 2;
    }
    GgufModel m = GgufModel::open(gguf);
    Ds41Geometry G;
    const std::string ge = ds41_read_geometry(m, G);
    if (!ge.empty()) { std::fprintf(stderr, "geometry: %s\n", ge.c_str()); return 1; }
    std::vector<int32_t> toks;
    {
        std::ifstream f(seq, std::ios::binary);
        int32_t x;
        while (f.read(reinterpret_cast<char *>(&x), 4)) if (x >= 0) toks.push_back(x);
    }
    if (toks.size() < 9000) { std::fprintf(stderr, "%s: need 9000+ tokens, got %zu\n", seq.c_str(), toks.size()); return 2; }
    toks.resize(std::min<size_t>(toks.size(), 20000));

    ds41_sys::setenv("STRATA_ENGRAM_HOT", "0", 1);
    ds41_sys::setenv("STRATA_ENGRAM_CACHE_ROWS", "0", 1);
    Ds41Engram A, B, C;   // C: --b with the row cache on
    std::string err;
    if (!A.open(G, split_src(a_src), threads, err)) { std::fprintf(stderr, "open --a: %s\n", err.c_str()); return 1; }
    if (!B.open(G, split_src(b_src), threads, err)) { std::fprintf(stderr, "open --b: %s\n", err.c_str()); return 1; }
    ds41_sys::setenv("STRATA_ENGRAM_CACHE_ROWS", "2000000", 1);
    if (!C.open(G, split_src(b_src), threads, err)) { std::fprintf(stderr, "open --b (cache): %s\n", err.c_str()); return 1; }
    int fails = 0;

    // 1. raw rows
    const int64_t rsz = A.row_dim() + A.row_dim() / 32;
    std::vector<uint8_t> ra((size_t) rsz), rb((size_t) rsz);
    std::mt19937_64 rng(1234);
    int64_t raw_checked = 0;
    for (size_t e = 0; tol < 0 && e < G.engram_layers.size(); ++e) {
        const int64_t rows = G.engram_rows[e];
        for (int k = 0; k < rows_n + 2; ++k) {
            const uint32_t r = k == 0 ? 0u : k == 1 ? (uint32_t) (rows - 1) : (uint32_t) (rng() % (uint64_t) rows);
            if (!A.read_raw((int) e, r, ra.data()) || !B.read_raw((int) e, r, rb.data())) {
                std::fprintf(stderr, "read_raw failed: slot %zu row %u\n", e, r);
                return 1;
            }
            if (std::memcmp(ra.data(), rb.data(), (size_t) rsz) != 0) {
                if (++fails <= 5) std::fprintf(stderr, "RAW DIFFERS: slot %zu row %u\n", e, r);
            }
            ++raw_checked;
        }
    }
    if (tol < 0) std::printf("1. read_raw: %" PRId64 " rows compared, %d differ\n", raw_checked, fails);
    else std::printf("1. read_raw: skipped (--approx: the sources hold different formats)\n");
    std::vector<double> errs;

    // 2-4. the batch reader over real tokens
    for (int64_t p = 0; p < (int64_t) toks.size(); ++p) { A.set_token(p, toks[(size_t) p]); B.set_token(p, toks[(size_t) p]); C.set_token(p, toks[(size_t) p]); }
    struct Case { int64_t pos0; int n; };
    const Case cases[] = { {0, 1}, {1, 1}, {100, 7}, {0, 512}, {777, 512}, {0, 4096}, {4096, 4096}, {8192, 4096},
                           {(int64_t) toks.size() - 1, 1}, {(int64_t) toks.size() - 512, 512} };
    int batches = 0, bfail = 0;
    std::vector<std::vector<float>> oa, ob, oc, orf;
    for (const Case & c : cases) {
        auto ba = A.start(c.pos0, c.n, oa, err);
        if (!ba || !A.wait(ba, err)) { std::fprintf(stderr, "A batch: %s\n", err.c_str()); return 1; }
        auto bb = B.start(c.pos0, c.n, ob, err);
        if (!bb || !B.wait(bb, err)) { std::fprintf(stderr, "B batch: %s\n", err.c_str()); return 1; }
        // exact mode: every --b path against --a; --approx: --b's batch rows near --a's, its other paths equal to it
        const std::vector<std::vector<float>> & ref = tol < 0 ? oa : ob;
        bool s2 = true;
        if (tol < 0) s2 = same(oa, ob);
        else {
            const size_t e0 = errs.size();
            row_errs(oa, ob, A.row_dim(), errs);
            for (size_t i = e0; i < errs.size(); ++i) s2 = s2 && errs[i] < tol;
        }
        bool s3 = true;
        for (int rep = 0; rep < 2; ++rep) {   // the second pass comes from the row cache
            auto bc = C.start(c.pos0, c.n, oc, err);
            if (!bc || !C.wait(bc, err)) { std::fprintf(stderr, "C batch: %s\n", err.c_str()); return 1; }
            s3 = s3 && same(ref, oc);
        }
        if (!B.rows_reference(c.pos0, c.n, orf, err)) { std::fprintf(stderr, "B reference: %s\n", err.c_str()); return 1; }
        const bool s4 = same(ref, orf);
        ++batches;
        if (!s2 || !s3 || !s4) {
            ++bfail;
            std::fprintf(stderr, "BATCH DIFFERS at pos %" PRId64 " n %d: batch %s, cached %s, reference %s\n", c.pos0, c.n,
                         s2 ? "same" : "DIFF", s3 ? "same" : "DIFF", s4 ? "same" : "DIFF");
        }
    }
    std::printf("2-4. batches: %d cases (batch reader, row cache x2, reference reader), %d differ; --b row cache: %" PRIu64
                " hits, %" PRIu64 " misses\n", batches, bfail, C.cache_hits(), C.cache_misses());
    if (tol >= 0 && !errs.empty()) {
        std::sort(errs.begin(), errs.end());
        std::printf("   --b vs --a, relative L2 error per row over %zu rows: median %.4f, p99 %.4f, max %.4f (tolerance %.3f)\n",
                    errs.size(), errs[errs.size() / 2], errs[errs.size() * 99 / 100], errs.back(), tol);
    }
    if (fails || bfail) { std::printf("DIFFERENT\n"); return 1; }
    std::printf(tol < 0 ? "IDENTICAL\n" : "CLOSE (within tolerance; --b's own paths identical)\n");
    return 0;
}
