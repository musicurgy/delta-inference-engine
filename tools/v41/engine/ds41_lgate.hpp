// ds41_lgate.hpp - STRATA_DSPARK_LGATE (0.95): a learned acceptance predictor plus coin-first verify sizing for the serve
// loop's DSpark passes (header only, so test_ds41_lgate checks the coin / accept logic on the CPU without the engine).
//
// What it replaces: the shipped gate (STRATA_DSPARK_GATE) verifies draft j while the calibrated survival
// prod_{i<=j} s_i >= lambda x deltaC, s_i a binned calibration of the confidence head alone.  Offline (CV by request)
// an oracle width is +9-11% over that gate, but the predictors recover little of it.  A 0.94 trace (0.93's drafter):
// logistic + coin +1.5 / +2.2 / +2.8% at cost models A / B / C.  A 0.95 trace (imx drafter, KEEP 112/80/48; speed
// cards, agent, decode_bench; the constants below): logistic alone +0.6%, logistic + coin +0.9 / +1.2 / +1.4%, GBDT +
// coin +1.2 / +1.6 / +1.9% - under the +2% (cost B) bar for a GPU A/B, so this stays opt-in and unmeasured.
//
// 1. Predictor: ahat_j = sigmoid(b + w . x_j), x_j = [the head's logit c_j, slot j == 1..4, the drafter's top-1
//    probability q1_j, top-1 minus top-2, entropy H_j, sum_{i<j} log sigmoid(c_i)].  Every feature is known before
//    draft j is drawn (c_j reads draft j-1; q's row at slot j is the drafter's distribution, not its pick), never
//    q(d_j) or "d_j is the argmax": a width that depends on the draw it verifies is not exact.
// 2. Conditional table: per (slot, ahat decile) a histogram of the label y = min(1, p(d)/q(d)) of verified drafts
//    (20 bins of 0.05 and one for y >= 0.999), seeded from a 0.95 drafter trace (at most 64 pseudo-counts a cell) and updated
//    online, halved past 512 so it follows a drifting drafter.  P(a > u | slot, ahat) is read from it.
// 3. Coin-first width: the acceptance coins u_j and residual coins r_j are drawn BEFORE the pass; walking j = 0..k-1
//    the survival sv *= P(a_j > u_j); the pass is cut at the first j where sv < lambda x deltaC (the GateCost price).
//    Rows t, d_0..d_(K-1) are verified with u_j / r_j, and when they all survive, position K is resolved from row K
//    (it exists anyway: it is today's bonus row) with ITS OWN coins u_K / r_K against its own draft d_K - not a fresh
//    sample_p.  A fresh sample_p stays only for a pass the coins did not cut (drafts ran out, the max_new / context caps).
//    Exactness: position j is resolved in this pass iff every earlier position was accepted and the cut is not before
//    j - a function of u_0..u_(j-1) only - and either way (a verified row or the cut row) its token is the same function
//    of (p_j, d_j, u_j, r_j) as spec_accept_q's, so u_j / r_j stay fresh uniforms for the position they decide; coins
//    of positions past the resolved ones are thrown away (the next pass draws its own).  test_ds41_lgate checks the
//    joint distribution of the emitted tokens by simulation, and that the naive form (a fresh coin at the cut) fails.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace ds41_lgate {

constexpr int kPos = 5;    // DSpark slots
constexpr int kDec = 10;   // ahat deciles
constexpr int kLab = 21;   // label bins: 20 of 0.05, then y >= 0.999
constexpr int kFeat = 9;   // logit_c, slot1..4, q1, margin, H, log_cprod_prefix

// q(x) of an inverse-CDF row (as spec_accept_q / draft_q: the last token takes the rest up to 1)
inline double row_q(const float * cdf, int64_t V, int x) {
    if (x < 0 || x >= V) return 0.0;
    const double hi = x == V - 1 ? 1.0 : std::min(1.0, (double) cdf[x]);
    const double lo = x == 0 ? 0.0 : std::min(1.0, (double) cdf[x - 1]);
    return std::max(0.0, hi - lo);
}

struct SlotFeat { double q1 = 0.0, margin = 0.0, H = 0.0; };

// the drafter's distribution at one slot from its CDF row: top-1, top-1 minus top-2, entropy (nats).  The entropy
// skips q <= 1e-6 (one log per token over 129K tokens x 5 slots would cost milliseconds; the skipped tail carries
// ~no mass at the drafter's sharpened temperature).  One pass over the row: ~0.1 ms a slot on the host.
inline SlotFeat slot_features(const float * cdf, int64_t V) {
    SlotFeat f;
    double q1 = -1.0, q2 = -1.0, prv = 0.0, H = 0.0;
    for (int64_t x = 0; x < V; ++x) {
        const double c = x == V - 1 ? 1.0 : std::min(1.0, (double) cdf[x]);
        const double q = std::max(0.0, c - prv);
        prv = c;
        if (q > 1e-6) H -= q * std::log(q);
        if (q > q1) { q2 = q1; q1 = q; } else if (q > q2) q2 = q;
    }
    f.q1 = q1;
    f.margin = q1 - std::max(0.0, q2);
    f.H = H;
    return f;
}

// the logistic predictor; constants fitted on a 0.95 drafter trace (all rows), raw units.
// STRATA_DSPARK_LGATE_W="b,w0,...,w8" replaces them (a refit without a rebuild; a malformed list keeps these).
struct Predictor {
    double b = 4.422514;
    double w[kFeat] = { 0.786758, 0.089814, 0.083646, 0.093535, 0.074710, -5.886797, 1.988958, -1.984387, 0.005853 };
    bool from_env = false;
    void load_env() {
        const char * e = std::getenv("STRATA_DSPARK_LGATE_W");
        if (!e || !*e) return;
        double v[kFeat + 1];
        int n = 0;
        const char * s = e;
        while (n < kFeat + 1) {
            char * end = nullptr;
            const double x = std::strtod(s, &end);
            if (end == s) break;
            v[n++] = x;
            s = end;
            while (*s == ',' || *s == ' ') ++s;
            if (!*s) break;
        }
        if (n != kFeat + 1 || *s) {
            std::fprintf(stderr, "lgate: STRATA_DSPARK_LGATE_W needs %d numbers (b, then %d weights) - built-in weights kept\n",
                         kFeat + 1, kFeat);
            return;
        }
        b = v[0];
        for (int i = 0; i < kFeat; ++i) w[i] = v[i + 1];
        from_env = true;
    }
    // j: slot; conf_logit: the head's raw logit for slot j; f: slot j's drafter features; lcp: sum_{i<j} log sigmoid(c_i)
    double predict(int j, float conf_logit, const SlotFeat & f, double lcp) const {
        const double lc = std::max(-9.21, std::min(9.21, (double) conf_logit));   // the trace's c clamp [1e-4, 1 - 1e-4]
        double z = b + w[0] * lc + w[5] * f.q1 + w[6] * f.margin + w[7] * f.H + w[8] * std::max(lcp, std::log(1e-6));
        if (j >= 1 && j <= 4) z += w[j];
        const double a = 1.0 / (1.0 + std::exp(-z));
        return std::max(0.001, std::min(0.999, a));
    }
};

// P(a > u | slot, ahat decile): histograms of verified-draft labels
struct CondTable {
    double h[kPos][kDec][kLab];
    long obs = 0;
    static int dec(double a) { return std::max(0, std::min(kDec - 1, (int) (a * kDec))); }
    static int lab(double y) { return y >= 0.999 ? kLab - 1 : std::max(0, std::min(19, (int) (y * 20.0))); }
    CondTable() {
        static const float prior[kPos][kDec][kLab] = {
#include "ds41_lgate_prior.inc"
        };
        for (int j = 0; j < kPos; ++j)
            for (int d = 0; d < kDec; ++d) {
                for (int l = 0; l < kLab; ++l) h[j][d][l] = prior[j][d][l];
                h[j][d][lab((d + 0.5) / kDec)] += 1.0;   // one pseudo-count at the decile's centre: an empty cell is a step there
            }
    }
    void add(int j, double ahat, double y) {
        if (j < 0 || j >= kPos || !(y >= 0.0)) return;
        double * r = h[j][dec(ahat)];
        r[lab(y)] += 1.0;
        double t = 0.0;
        for (int l = 0; l < kLab; ++l) t += r[l];
        if (t > 512.0) for (int l = 0; l < kLab; ++l) r[l] *= 0.5;
        ++obs;
    }
    double p_gt(int j, double ahat, double u) const {
        const double * r = h[std::max(0, std::min(kPos - 1, j))][dec(ahat)];
        double tot = 0.0;
        for (int l = 0; l < kLab; ++l) tot += r[l];
        double above = 0.0;
        if (u < 0.999) {
            const int bu = std::max(0, std::min(19, (int) (u * 20.0)));
            for (int l = bu + 1; l < kLab; ++l) above += r[l];
            above += r[bu] * ((bu + 1) / 20.0 - u) * 20.0;   // the part of u's own bin above u (uniform inside the bin)
        }
        return (above + 1.0) / (tot + 2.0);
    }
};

// the coin-first cut: K = the first slot j < k where prod_{i<=j} P(a_i > u_i) < thr, else k
inline int coin_cut(const double * ahat, const double * u, int k, double thr, const CondTable & t) {
    double sv = 1.0;
    for (int j = 0; j < k; ++j) {
        sv *= t.p_gt(j, ahat[j], u[j]);
        if (sv < thr) return j;
    }
    return k;
}

// spec_accept_q with its two uniforms given (u: the acceptance coin, r: the residual draw) - the same arithmetic:
// accept d with min(1, p(d) / q(d)); on rejection sample from norm(max(0, p - q)).  true = d accepted.
template <class Dist>
inline bool accept_q_coins(const Dist & dist, int d, const float * cdf, int64_t V, double u, double r, int & out) {
    double pd = 0.0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    if (pd > 0.0 && u * row_q(cdf, V, d) < pd) return true;
    double z = 0.0;
    for (const auto & c : dist) z += std::max(0.0, c.second - row_q(cdf, V, c.first));
    double rr = r * (z > 0.0 ? z : 1.0);
    int last = d;
    for (const auto & c : dist) {
        const double wgt = z > 0.0 ? std::max(0.0, c.second - row_q(cdf, V, c.first)) : c.second;
        if (wgt <= 0.0) continue;
        last = c.first;
        if ((rr -= wgt) <= 0.0) break;
    }
    out = last;
    return false;
}

}  // namespace ds41_lgate
