// tools/v41/engine/ds41_lookup.hpp - STRATA_LOOKUP_LONG (default off; Basalt 97bf1f168's second 8-gram index): the
// lookup drafter (STRATA_SERVE_LOOKUP, ds41_generate.cpp) also looks past the matches it already checked for the latest
// earlier places whose last kLookupLong (8) tokens equal the context's, so an old source of a quoted stretch (a file read
// 50K tokens back, now being edited) is still found when a common short n-gram has many newer repeats.  The longest copy
// wins; a tie keeps the newer place.  Drafts only: the verify pass decides every token, so the output is unchanged.
//
// The sequence is hist[0 .. n) followed by t (the token for the next position, not in hist yet): seq(i) = i == n ? t :
// hist[i].  A place q (hist[q] == t) matches m tokens when seq(q - j) == seq(n - j) for j < m (m <= 64, as the drafter's
// own scan); its copy is seq(q + 1 ..) of length min(cap, m - g + 1, n - q) (g = STRATA_LOOKUP_NGRAM).
#pragma once

#include <algorithm>

namespace strata::v41 {

constexpr int kLookupLong = 8;     // Basalt's LONG_ORDER
constexpr int kLookupMaxM = 64;    // the drafter's match cap

// Scan q = q_from, q_from - 1, .. >= g for places matching >= kLookupLong tokens; at most `max_hits` such places are
// looked at (newest first).  Among those whose copy starts with `agree` (agree < 0: any), the one with the longest copy
// > `floor_len` wins (the newest on a tie); the scan stops once a copy reaches `cap`.  Returns its copy length (0: none)
// and sets q_best.
inline int lookup_long_scan(const int * hist, int n, int t, int q_from, int g, int cap, int max_hits, int agree,
                            int floor_len, int & q_best) {
    auto sq = [&](int i) { return i == n ? t : hist[i]; };
    int best = 0, hits = 0;
    for (int q = std::min(q_from, n - 1); q >= g && hits < max_hits; --q) {
        if (hist[q] != t) continue;
        int m = 1;
        while (m < kLookupMaxM && q - m >= 0 && hist[q - m] == sq(n - m)) ++m;
        if (m < kLookupLong) continue;
        ++hits;
        if (agree >= 0 && sq(q + 1) != agree) continue;
        const int L = std::min({ cap, m - g + 1, n - q });
        if (L > floor_len && L > best) {
            best = L;
            q_best = q;
            if (L >= cap) break;
        }
    }
    return best;
}

}  // namespace strata::v41
