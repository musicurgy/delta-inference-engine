// test_ds41_lgate [--n N] - CPU unit test of the coin-first verify sizing (ds41_lgate.hpp, STRATA_DSPARK_LGATE).
//
// A toy language: 4 tokens, a first-order target chain p(x | prev) and a different drafter chain q(x | prev).  Each
// verify pass drafts D = 3 tokens from q, predicts their acceptance from q's row (any function of the prefix will do),
// draws the coins first, cuts the pass with coin_cut against a fixed price, verifies the kept rows with
// accept_q_coins and resolves the cut position with its own coins - the serve loop's order.  The first L = 4 emitted
// tokens of N sequences are compared with the exact chain probabilities (chi-square over the 256 outcomes):
//   exact      the shipped design                                   must PASS
//   gate-only  no coins: a fixed width from the predicted survival   must PASS (control: plain speculative sampling)
//   naive      the cut position resolved with a FRESH coin           must FAIL (the test can see the bias it guards)
// Exit 0 when all three behave.  No GPU, no model.
#include "ds41_lgate.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int V = 4, D = 3, L = 4;
// target p(x | prev) and drafter q(x | prev): rows sum to 1; q is deliberately off p so drafts get rejected
const double P[V][V] = { { 0.55, 0.25, 0.15, 0.05 }, { 0.10, 0.60, 0.20, 0.10 }, { 0.30, 0.10, 0.30, 0.30 }, { 0.05, 0.05, 0.10, 0.80 } };
const double Q[V][V] = { { 0.20, 0.60, 0.10, 0.10 }, { 0.40, 0.30, 0.20, 0.10 }, { 0.70, 0.10, 0.10, 0.10 }, { 0.25, 0.25, 0.25, 0.25 } };
float CDF[V][V];

using Dist = std::vector<std::pair<int, double>>;
Dist dist_of(int prev) {
    Dist d;
    for (int x = 0; x < V; ++x) if (P[prev][x] > 0) d.push_back({ x, P[prev][x] });
    return d;
}
int draw(const double * row, double u) {
    for (int x = 0; x < V; ++x) { if ((u -= row[x]) <= 0) return x; }
    return V - 1;
}
enum Mode { kExact, kGateOnly, kNaive };

// one sequence: the first L tokens after the start token 0
void sequence(Mode mode, std::mt19937_64 & rng, const ds41_lgate::CondTable & tab, double thr, int * out) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int n = 0;
    int t = draw(P[0], U(rng));   // the first token, sampled plainly (the prompt's last row)
    out[n++] = t;
    while (n < L) {
        int d[D], prev[D + 1];
        double ahat[D];
        prev[0] = t;
        for (int j = 0; j < D; ++j) {
            ahat[j] = std::min(0.999, 0.15 + 0.8 * *std::max_element(Q[prev[j]], Q[prev[j]] + V));   // depends on the prefix only
            d[j] = draw(Q[prev[j]], U(rng));
            prev[j + 1] = d[j];
        }
        double u[D + 1], r[D + 1];
        for (int j = 0; j <= D; ++j) { u[j] = U(rng); r[j] = U(rng); }
        int k = D;
        bool cut = false;
        if (mode == kGateOnly) {   // the width from the predicted survival alone (no coin looked at)
            double sv = 1.0;
            int kk = 0;
            while (kk < D && (sv *= ahat[kk]) >= thr) ++kk;
            k = kk;
        } else {
            k = ds41_lgate::coin_cut(ahat, u, D, thr, tab);
            cut = k < D;
        }
        int acc = 0, tn = -1;
        for (; acc < k; ++acc)
            if (!ds41_lgate::accept_q_coins(dist_of(prev[acc]), d[acc], CDF[prev[acc]], V, u[acc], r[acc], tn)) break;
        if (acc == k) {
            if (cut) {
                const double uk = mode == kNaive ? U(rng) : u[k], rk = mode == kNaive ? U(rng) : r[k];
                if (ds41_lgate::accept_q_coins(dist_of(prev[k]), d[k], CDF[prev[k]], V, uk, rk, tn)) tn = d[k];
            } else {
                tn = draw(P[prev[k]], U(rng));   // drafts ran out / no cut: a fresh draw, as sample_p
            }
        }
        for (int i = 0; i < acc && n < L; ++i) out[n++] = d[i];
        if (n < L) out[n++] = tn;
        t = tn;
    }
}

double chi2(Mode mode, long N, uint64_t seed, double thr, double * maxdev) {
    ds41_lgate::CondTable tab;
    std::mt19937_64 rng(seed);
    std::vector<long> cnt(V * V * V * V, 0);
    int s[L];
    for (long i = 0; i < N; ++i) {
        sequence(mode, rng, tab, thr, s);
        ++cnt[((s[0] * V + s[1]) * V + s[2]) * V + s[3]];
    }
    double x2 = 0.0, md = 0.0;
    for (int a = 0; a < V; ++a) for (int b = 0; b < V; ++b) for (int c = 0; c < V; ++c) for (int e = 0; e < V; ++e) {
        const double p = P[0][a] * P[a][b] * P[b][c] * P[c][e];
        const double o = (double) cnt[((a * V + b) * V + c) * V + e];
        x2 += (o - N * p) * (o - N * p) / (N * p);
        md = std::max(md, std::fabs(o / N - p));
    }
    *maxdev = md;
    return x2;
}

}  // namespace

int main(int argc, char ** argv) {
    long N = 2000000;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--n") && i + 1 < argc) N = std::atol(argv[++i]);
    for (int p = 0; p < V; ++p) {
        double c = 0.0;
        for (int x = 0; x < V; ++x) { c += Q[p][x]; CDF[p][x] = (float) c; }
    }
    // the price: high enough that coin_cut cuts most passes somewhere (the cut path is what the test is about)
    const double thr = 0.35;
    const int dof = V * V * V * V - 1;
    const double lim = dof + 5.0 * std::sqrt(2.0 * dof);   // ~5 sigma above the mean of a chi-square on 255 dof
    bool ok = true;
    struct { Mode m; const char * name; bool want_pass; } arms[] = {
        { kExact, "exact (coin-first, own coins at the cut)", true },
        { kGateOnly, "gate-only (no coins)", true },
        { kNaive, "naive (fresh coin at the cut)", false },
    };
    for (auto & a : arms) {
        double md = 0.0;
        const double x2 = chi2(a.m, N, 0x5eed + (uint64_t) a.m, thr, &md);
        const bool pass = x2 < lim;
        const bool good = pass == a.want_pass;
        ok = ok && good;
        std::printf("%-44s chi2 %9.1f on %d dof (limit %.0f), max |freq - p| %.5f -> %s%s\n", a.name, x2, dof, lim, md,
                    pass ? "matches the chain" : "DIFFERS from the chain", good ? "" : "   <-- UNEXPECTED");
    }
    // the coin cut must actually cut: count how often the width is below D
    {
        ds41_lgate::CondTable tab;
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        long cuts = 0, n = 100000;
        for (long i = 0; i < n; ++i) {
            double ah[D], u[D];
            for (int j = 0; j < D; ++j) { ah[j] = 0.15 + 0.8 * U(rng); u[j] = U(rng); }
            cuts += ds41_lgate::coin_cut(ah, u, D, thr, tab) < D;
        }
        std::printf("coin cut below the full width in %.1f%% of random passes\n", 100.0 * cuts / n);
        if (cuts == 0 || cuts == n) { ok = false; std::printf("  <-- the cut never / always fires: the test does not exercise it\n"); }
    }
    std::printf("%s\n", ok ? "test_ds41_lgate: PASS" : "test_ds41_lgate: FAIL");
    return ok ? 0 : 1;
}
