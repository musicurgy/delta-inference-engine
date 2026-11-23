// tools/v41/engine/test_ds41_dspark.cpp - the engine's DSpark draft stack on the offline harness's records.
//
//   test_ds41_dspark --sidecar V41-MTP-Vision.gguf -m V41-GSQ.gguf --main main.bin --gen gen.txt [--max N] [--dump F] [--park 1|2]
//
// main.bin / gen.txt = what tools/v41/dspark_accept.py reads (engine STRATA_DUMP_MAIN records {int32 token, int32 pos,
// 3 x n_embd f32}: the first 128 are the prompt tail that seeds the window, the rest decode steps).  The same loop as
// the harness: step j appends record j's main_hidden at its position and drafts with bonus = the next token; drafts are
// scored against the tokens that followed (accepted length = the matching prefix).  --dump writes one line per scored
// step: "j pos d0..d4 c0..c4" (compare with dspark_accept.py --dump-drafts).
// --park 1|2 (STRATA_DSPARK_PARK, A7): after the first 64 seed records, park the stack (1 = pageable host copy, 2 =
// pinned), unpark it and start again from record 0 - the --dump must be byte-identical to a run without --park (the
// same weights, a cleared state and an empty window, as a freshly loaded stack).
#include "ds41_dspark.hpp"

#include "ggml-backend.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string side, model, mainf, genf, dump;
    int max = 0, park = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--sidecar") side = nx();
        else if (a == "-m") model = nx();
        else if (a == "--main") mainf = nx();
        else if (a == "--gen") genf = nx();
        else if (a == "--max") max = std::atoi(nx().c_str());
        else if (a == "--dump") dump = nx();
        else if (a == "--park") park = std::atoi(nx().c_str());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (side.empty() || model.empty() || mainf.empty() || genf.empty()) {
        std::fprintf(stderr, "usage: test_ds41_dspark --sidecar F -m F --main main.bin --gen gen.txt [--max N] [--dump F] [--park 1|2]\n");
        return 2;
    }
#ifdef DS41_HAVE_CUDA
    ggml_backend_t be = ggml_backend_cuda_init(0);
#else
    ggml_backend_t be = nullptr;
#endif
    if (!be) { std::fprintf(stderr, "no CUDA backend\n"); return 1; }
    Ds41DSparkConfig cfg;
    cfg.backend = be;
    cfg.sidecar = side;
    cfg.main_gguf = model;
    Ds41DSpark ds;
    std::string err;
    if (!ds.init(cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int64_t MD = ds.main_dim();
    std::fprintf(stderr, "DSpark stack: %.2f GB of weights on the device\n", ds.weight_bytes() / 1e9);

    // records
    std::ifstream f(mainf, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t rs = 8 + (size_t) MD * 4;
    if (raw.empty() || raw.size() % rs) { std::fprintf(stderr, "%s: not a whole number of records\n", mainf.c_str()); return 1; }
    const int n = (int) (raw.size() / rs);
    std::vector<int> tok((size_t) n), pos((size_t) n);
    for (int j = 0; j < n; ++j) {
        std::memcpy(&tok[(size_t) j], raw.data() + (size_t) j * rs, 4);
        std::memcpy(&pos[(size_t) j], raw.data() + (size_t) j * rs + 4, 4);
    }
    std::vector<int> gen;
    {
        std::ifstream g(genf);
        std::string w;
        while (g >> w) {
            bool num = !w.empty();
            for (size_t i = (w[0] == '-'); i < w.size(); ++i) num = num && std::isdigit((unsigned char) w[i]);
            if (num) gen.push_back(std::atoi(w.c_str()));
        }
    }
    std::vector<int> seq(tok);
    seq.push_back(gen.back());
    const int BS = cfg.block, n_seed = std::min(128, n);
    FILE * df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "w");
    std::vector<int> acc;
    int first_ok = 0;
    double ms_sum = 0;
    int ms_n = 0;
    for (int j = 0; j < n; ++j) {
        const float * mh = (const float *) (raw.data() + (size_t) j * rs + 8);
        const bool score = j >= n_seed && j + 1 + BS < (int) seq.size();
        int ids[8];
        float conf[8];
        if (!ds.step(mh, &pos[(size_t) j], 1, score ? seq[(size_t) j + 1] : -1, ids, conf)) {
            std::fprintf(stderr, "step %d: %s\n", j, ds.last_error().c_str());
            return 1;
        }
        if (park > 0 && j == 63) {   // --park: still seeding (no step scored yet), so restarting at 0 changes nothing
            const auto t0 = std::chrono::steady_clock::now();
            if (!ds.park(park >= 2, 0, err)) { std::fprintf(stderr, "park: %s\n", err.c_str()); return 1; }
            const auto t1 = std::chrono::steady_clock::now();
            if (!ds.unpark(err)) { std::fprintf(stderr, "unpark: %s\n", err.c_str()); return 1; }
            const auto t2 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "park (%s, %.2f GB host copy) %.1f ms, unpark %.1f ms\n", ds.host_pinned() ? "pinned" : "pageable",
                         ds.host_bytes() / 1e9, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                         std::chrono::duration<double, std::milli>(t2 - t1).count());
            park = 0;
            j = -1;
            continue;
        }
        if (!score) continue;
        ms_sum += ds.last_ms();
        ++ms_n;
        int k = 0;
        while (k < BS && ids[k] == seq[(size_t) j + 2 + k]) ++k;
        acc.push_back(k);
        first_ok += ids[0] == seq[(size_t) j + 2];
        if (df) {
            std::fprintf(df, "%d %d", j, pos[(size_t) j]);
            for (int i = 0; i < BS; ++i) std::fprintf(df, " %d", ids[i]);
            for (int i = 0; i < BS; ++i) std::fprintf(df, " %.4f", conf[i]);
            std::fprintf(df, "\n");
        }
        if (max && (int) acc.size() >= max) break;
    }
    if (df) std::fclose(df);
    double mean = 0;
    std::vector<int> hist((size_t) BS + 1, 0);
    for (int a : acc) { mean += a; ++hist[(size_t) a]; }
    mean /= std::max<size_t>(1, acc.size());
    std::printf("engine DSpark over %zu steps: mean accepted %.3f -> tokens per verify pass %.3f; first draft %.1f%%; "
                "draft %.2f ms/step\n", acc.size(), mean, 1 + mean, 100.0 * first_ok / std::max<size_t>(1, acc.size()),
                ms_sum / std::max(1, ms_n));
    std::printf("  distribution:");
    for (int k = 0; k <= BS; ++k) std::printf(" %d:%.1f%%", k, 100.0 * hist[(size_t) k] / std::max<size_t>(1, acc.size()));
    std::printf("\n");
    for (int d = 1; d <= BS; ++d) {
        double t = 0;
        for (int a : acc) t += std::min(a, d);
        std::printf("  verify depth %d: tokens per pass %.3f\n", d, 1 + t / std::max<size_t>(1, acc.size()));
    }
    ggml_backend_free(be);
    return 0;
}
