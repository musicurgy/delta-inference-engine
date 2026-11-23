// tools/v41/engine/test_ds41_lookup.cpp - STRATA_LOOKUP_LONG (ds41_lookup.hpp, Basalt 97bf1f168): CPU only, no model.
// The two lookup call sites of ds41_generate.cpp (the latest-match path and STRATA_LOOKUP_YIELD's longest-of-256 scan)
// are mirrored here with the long scan off and on:
//   1. an old source behind a common n-gram's newer repeats is found (off: the short newer copy);
//   2. random sequences: on proposes the same copy or a longer one, never a shorter one; every copy is a real match
//      (its last g + 1 tokens equal the context's) and its drafts are the tokens that followed it.
// Exit 0 and "LOOKUP OK".
#include "ds41_lookup.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace strata::v41;

namespace {

int fails = 0;
void check(bool ok, const char * what) {
    if (!ok) { ++fails; std::fprintf(stderr, "FAIL: %s\n", what); }
}

struct Copy { int q = -1, L = 0; };

// ds41_generate.cpp, STRATA_SERVE_LOOKUP without YIELD: the latest match (+ STRATA_LOOKUP_LONG's 4 newest older 8-token
// matches); agreement checked on the chosen copy
Copy latest_path(const std::vector<int> & hist, int t, int g, int cap, bool lk_long, int agree) {
    const int n = (int) hist.size();
    auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
    Copy c;
    if (cap < 2 || n <= g + 1) return c;
    for (int q = n - 1; q >= g; --q) {
        if (hist[(size_t) q] != t) continue;
        int m = 1;
        while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
        if (m < g + 1) continue;
        int L = std::min({ cap, m - g + 1, n - q });
        if (lk_long && L < cap) {
            int q2 = -1;
            const int L2 = lookup_long_scan(hist.data(), n, t, q - 1, g, cap, 4, agree, L, q2);
            if (L2 > L) { L = L2; q = q2; }
        }
        if (L < 2) break;
        if (agree >= 0 && sq(q + 1) != agree) break;
        c.q = q;
        c.L = L;
        break;
    }
    return c;
}

// ds41_generate.cpp, STRATA_LOOKUP_YIELD's scan: the longest copy among the 256 newest matches (+ STRATA_LOOKUP_LONG)
Copy yield_path(const std::vector<int> & hist, int t, int g, int cap, bool lk_long, int agree) {
    const int n = (int) hist.size();
    auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
    Copy c;
    if (cap < 2 || n <= g + 1) return c;
    int bq = -1, bL = 0, seen = 0;
    int q = n - 1;
    for (; q >= g && seen < 256; --q) {
        if (hist[(size_t) q] != t) continue;
        int m = 1;
        while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
        if (m < g + 1) continue;
        ++seen;
        if (agree >= 0 && sq(q + 1) != agree) continue;
        const int L = std::min({ cap, m - g + 1, n - q });
        if (L > bL) { bL = L; bq = q; if (L >= cap) break; }
    }
    if (lk_long && seen >= 256 && bL < cap) {
        int q2 = -1;
        const int L2 = lookup_long_scan(hist.data(), n, t, q, g, cap, 256, agree, bL, q2);
        if (L2 > bL) { bL = L2; bq = q2; }
    }
    if (bL >= 2) { c.q = bq; c.L = bL; }
    return c;
}

// a copy is a real match of >= g + 1 tokens, inside the history
bool valid(const std::vector<int> & hist, int t, int g, const Copy & c) {
    if (c.L == 0) return true;
    const int n = (int) hist.size();
    auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
    if (c.q < 0 || c.q >= n || c.q + c.L > n) return false;
    for (int j = 0; j <= g; ++j)
        if (c.q - j < 0 || sq(c.q - j) != sq(n - j)) return false;
    return true;
}

}  // namespace

int main() {
    const int g = 3, cap = 14;
    // 1. Basalt's case: a 20-token source, then R later repeats of the 4-gram the quote ends on, then the quote again
    for (int R : { 8, 300 }) {
        std::vector<int> doc;
        for (int i = 0; i < 20; ++i) doc.push_back(1000 + i);
        for (int r = 0; r < R; ++r) {
            for (int i = 0; i < 4; ++i) doc.push_back(2000 + r * 10 + i);
            for (int i = 8; i < 12; ++i) doc.push_back(1000 + i);   // ... 1008 1009 1010 1011: a 4-token match only
            doc.push_back(3000 + r);
        }
        for (int i = 0; i < 12; ++i) doc.push_back(1000 + i);       // quoting the source: 1000 .. 1011
        const int t = doc.back();
        doc.pop_back();                                             // hist = everything before t
        const Copy a0 = latest_path(doc, t, g, cap, false, -1), a1 = latest_path(doc, t, g, cap, true, -1);
        check(a0.L == 2, "latest path, off: the newest (short) copy");
        check(a1.L == 10 && a1.q == 11 && doc[(size_t) a1.q + 1] == 1012, "latest path, on: the old source, 10 tokens");
        const Copy y0 = yield_path(doc, t, g, cap, false, -1), y1 = yield_path(doc, t, g, cap, true, -1);
        if (R > 256) {
            check(y0.L == 2, "yield path, 300 repeats, off: only the newest 256 looked at");
            check(y1.L == 10 && y1.q == 11, "yield path, 300 repeats, on: the old source");
        } else {
            check(y0.L == 10 && y1.L == 10 && y0.q == y1.q, "yield path, 8 repeats: found either way, the same copy");
        }
        // agreement: a draft that disagrees with the copy's first token takes nothing from the long scan either
        const Copy y2 = yield_path(doc, t, g, cap, true, 4242);
        check(y2.L == 0, "yield path, on: the agreement rule holds for long matches");
        std::printf("case R=%d: latest off %d on %d, yield off %d on %d\n", R, a0.L, a1.L, y0.L, y1.L);
    }
    // 2. random sequences over a small vocabulary with copied stretches: on >= off, valid copies, same when off is full
    std::mt19937 rng(1234);
    long more = 0, same = 0;
    for (int it = 0; it < 3000; ++it) {
        const int V = 3 + (int) (rng() % 6), n = 50 + (int) (rng() % 3000);
        std::vector<int> s;
        while ((int) s.size() < n + 1) {
            if (s.size() > 40 && rng() % 4 == 0) {   // copy a stretch from earlier
                const int a = (int) (rng() % (s.size() - 20)), len = 5 + (int) (rng() % 30);
                for (int j = 0; j < len && (int) s.size() < n + 1; ++j) s.push_back(s[(size_t) (a + j)]);
            } else s.push_back((int) (rng() % V));
        }
        const int t = s.back();
        s.pop_back();
        const int agree = rng() % 2 ? -1 : (int) (rng() % V);
        for (int path = 0; path < 2; ++path) {
            const Copy c0 = path ? yield_path(s, t, g, cap, false, agree) : latest_path(s, t, g, cap, false, agree);
            const Copy c1 = path ? yield_path(s, t, g, cap, true, agree) : latest_path(s, t, g, cap, true, agree);
            check(valid(s, t, g, c0) && valid(s, t, g, c1), "a copy is a real match");
            check(c1.L >= c0.L, "on proposes the same copy length or more");
            if (c0.L >= cap) check(c1.q == c0.q && c1.L == c0.L, "a full copy is left alone");
            if (c1.L > c0.L) ++more; else if (c1.L == c0.L && c1.q == c0.q) ++same;
        }
    }
    std::printf("random: %ld longer copies, %ld the same\n", more, same);
    if (fails) { std::printf("LOOKUP FAILED (%d)\n", fails); return 1; }
    std::printf("LOOKUP OK\n");
    return 0;
}
