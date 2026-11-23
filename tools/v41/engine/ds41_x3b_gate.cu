// tools/v41/engine/ds41_x3b_gate.cu - see ds41_x3b_gate.hpp (STRATA_X3B).
#include "ds41_x3b_gate.hpp"

#include <cuda.h>           // CUresult / CUstream / CUdeviceptr only: the driver entry points come from the runtime
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace strata {
namespace v41 {

namespace {

__device__ __forceinline__ unsigned long long x3b_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// the spin gate: one thread polls the flag (host-mapped; written by x3b_set on the release stream, or by the host
// watchdog) until it reaches seq; after timeout_ns of waiting it gives up and raises *timed_out so the host fails the
// pass.  The wait is counted step by step on %globaltimer: a step backwards counts 0 and a step forwards at most 1 ms
// (one poll is ~1 us), so an adjustment of the clock is not taken for a wait.  Until 11-17 it compared now - t0 as
// unsigned: a clock step backwards wrapped to ~1.8e19 ns and gave the gate up at once (measured: three "a gate timed
// out" with no slow window and no watchdog line).  The give-up records its account in diag[0..7] (host-mapped words):
// seq, the counted wait (us), the raw span now - t0 (us, signed), steps backwards, the largest step forwards (us).
__global__ void x3b_spin(const volatile uint32_t * flag, uint32_t seq, unsigned long long timeout_ns, volatile int * timed_out,
                         volatile int32_t * diag) {
    const unsigned long long t0 = x3b_now_ns();
    diag[8] = (int32_t) seq;                       // [8] this gate, [10..11] its start on the card's clock
    ((volatile unsigned long long *) diag)[5] = t0;
    unsigned long long prev = t0, waited = 0, maxstep = 0;
    int32_t back = 0;
    while ((int32_t) (*flag - seq) < 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(256);
#endif
        const unsigned long long now = x3b_now_ns();
        const long long d = (long long) (now - prev);
        prev = now;
        if (d < 0) { ++back; continue; }
        if ((unsigned long long) d > maxstep) maxstep = (unsigned long long) d;
        waited += (unsigned long long) d < 1000000ull ? (unsigned long long) d : 1000000ull;
        if (waited > timeout_ns) {
            diag[0] = (int32_t) seq;
            diag[1] = (int32_t) (waited / 1000);
            diag[2] = (int32_t) ((long long) (now - t0) / 1000);
            diag[3] = back;
            diag[4] = (int32_t) (maxstep / 1000 < 0x7fffffffull ? maxstep / 1000 : 0x7fffffffull);
            __threadfence_system();
            *timed_out = *timed_out + 1;
            __threadfence_system();
            return;
        }
    }
    __threadfence();   // acquire side of the flag: the work behind ev_open sees what came before the release
}

__global__ void x3b_set(volatile uint32_t * flag, uint32_t seq, volatile int32_t * diag) {
    __threadfence();   // (the tier's sum is complete: R waited on its event before this kernel)
    *flag = seq;
    __threadfence_system();
    diag[12] = (int32_t) seq;                      // [12] the gate released last, [14..15] when (card clock)
    ((volatile unsigned long long *) diag)[7] = x3b_now_ns();
}

// STRATA_X3B_STATS: on the probe stream, behind the tier's sum event - "the sum landed" for gate seq ([13], [16..17] when)
__global__ void x3b_probe(volatile int32_t * diag, uint32_t seq) {
    diag[13] = (int32_t) seq;
    ((volatile unsigned long long *) diag)[8] = x3b_now_ns();
    __threadfence_system();
}

typedef CUresult (CUDAAPI * PfnWaitValue32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult (CUDAAPI * PfnWriteValue32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

struct DevGuard {
    int prev = -1;
    explicit DevGuard(int d) {
        if (cudaGetDevice(&prev) != cudaSuccess) prev = -1;
        if (prev != d) cudaSetDevice(d);
    }
    ~DevGuard() {
        int cur = -1;
        if (prev >= 0 && cudaGetDevice(&cur) == cudaSuccess && cur != prev) cudaSetDevice(prev);
    }
};

}  // namespace

static const char * x3b_phase_name(int p) {
    static const char * const n[X3B_PH_N] = { "idle", "launching the gated work", "waiting for the router readback",
                                              "in the tier call", "in the release call", "past the release" };
    return p >= 0 && p < X3B_PH_N ? n[p] : "?";
}

// The flag lives in mapped pinned HOST memory (word 0; the spin's timeout count in word 16): the release stream writes
// it from the card (x3b_set / cuStreamWriteValue32 through its device address), and the host watchdog can write it with
// a plain store - no CUDA call, so it still opens the gate when the decode thread is stuck inside the driver (a
// device-wide sync with gated work queued is a deadlock no CUDA call can break).
struct X3bGate {
    // host-side record of the recent gates (decode thread), for x3b_gate_describe
    struct Rec {
        uint32_t seq = 0;
        int layer = -1;
        double arm_ms = 0, rel0_ms = -1, rel1_ms = -1;
        void * sum_ev = nullptr;   // the tier's sum event the release waited on
        int sum_done = -1;         // ... already complete when the release was issued (1), not yet (0), none (-1)
    };
    Rec rec[8];
    long described_spin = 0, described_wd = 0;   // spin / watchdog timeouts already described
    std::atomic<int> note_layer { -1 }, note_phase { 0 };
    std::atomic<long long> note_t_us { 0 };
    int device = 0;
    int mode = 0;
    double timeout_ms = 0;
    unsigned long long spin_timeout_ns = 0;
    cudaStream_t g = nullptr, r = nullptr;      // the gate's stream, the release stream (both non-blocking)
    cudaStream_t q = nullptr;                   // STRATA_X3B_STATS: the probe stream (sum landed?), non-blocking
    cudaEvent_t ev_open = nullptr;
    uint32_t * host = nullptr;                  // mapped pinned host words: [0] flag, [16] spin timeouts
    uint32_t * dev = nullptr;                   // their device address
    uint32_t seq = 0;                           // the last armed gate (decode thread)
    PfnWaitValue32 wait32 = nullptr;
    PfnWriteValue32 write32 = nullptr;
    std::vector<cudaEvent_t> tev;               // STRATA_X3B_STATS timing events
    std::vector<bool> tev_rec;
    // the watchdog (host thread; touches only `host` and these)
    std::mutex mu;
    std::condition_variable cv;
    // armed: the host has not issued the release yet.  watch: the watchdog guards the gate until it is OPEN ON THE
    // CARD - not only until the release is issued (11-17: a release issued but never landing left
    // the wait gate shut with no watchdog; the request hung until the server's 300 s engine-silence limit -> 503)
    bool armed = false, watch = false, quit = false;
    uint32_t armed_seq = 0;
    std::chrono::steady_clock::time_point armed_at;
    std::atomic<long> wd_fired { 0 };
    std::thread wd;

    volatile uint32_t * flag_host() const { return (volatile uint32_t *) host; }
    volatile int * spin_timeouts_host() const { return (volatile int *) (host + 16); }
    volatile int32_t * spin_diag_host() const { return (volatile int32_t *) (host + 24); }   // [24, 40): see x3b_spin / x3b_set
    int32_t * spin_diag_dev() const { return (int32_t *) (dev + 24); }
    // the watchdog's account of its last firing (written before wd_fired is raised)
    uint32_t wd_seq = 0;
    int wd_layer = -1, wd_phase = 0;
    int wd_rel_ran = -1, wd_sum_landed = -1;   // at the firing, from the mapped words (no CUDA call): 1 yes, 0 no, -1 unknown
    double wd_since_arm_ms = 0, wd_since_note_ms = 0;
    void watchdog() {
        std::unique_lock<std::mutex> lk(mu);
        while (!quit) {
            if (!watch) { cv.wait(lk); continue; }
            const uint32_t s = armed_seq;
            const auto deadline = armed_at + std::chrono::microseconds((long long) (timeout_ms * 1000.0));
            if (cv.wait_until(lk, deadline, [&] { return quit || !watch || armed_seq != s; })) continue;
            // still shut on the card past the deadline (released or not): open it from the host (the pass is then
            // invalid - counted, x3b_check)
            if ((int32_t) (*flag_host() - s) < 0) {
                const auto now = std::chrono::steady_clock::now();
                const long long now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                wd_seq = s;
                wd_layer = note_layer.load();
                wd_phase = note_phase.load();
                wd_since_arm_ms = std::chrono::duration<double, std::milli>(now - armed_at).count();
                wd_since_note_ms = (double) (now_us - note_t_us.load()) / 1000.0;
                {   // the card's own marks, read before the store above lets the gated work run (host memory only)
                    volatile int32_t * dg = spin_diag_host();
                    wd_rel_ran = (int32_t) ((uint32_t) dg[12] - s) >= 0 ? 1 : 0;
                    wd_sum_landed = q ? ((int32_t) ((uint32_t) dg[13] - s) >= 0 ? 1 : 0) : -1;
                }
                *flag_host() = s;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                wd_fired.fetch_add(1);
                std::fprintf(stderr, "ds41_dense: STRATA_X3B: watchdog opened gate %u (layer %d) %.0f ms after its arm (%s); "
                                     "the decode thread: %s for %.0f ms; on the card: the tier's sum %s, the release kernel %s\n",
                             s, wd_layer, wd_since_arm_ms, armed ? "no release" : "released, but not open on the card",
                             x3b_phase_name(wd_phase), wd_since_note_ms,
                             wd_sum_landed == 1 ? "had landed" : wd_sum_landed == 0 ? "had NOT landed" : "(no probe)",
                             wd_rel_ran == 1 ? "had run" : "had NOT run");
            }
            armed = false;
            watch = false;
        }
    }
};

static double host_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool fail(char * err, int err_n, const char * what, cudaError_t e) {
    if (err && err_n > 0) std::snprintf(err, (size_t) err_n, "%s: %s", what, cudaGetErrorString(e));
    return false;
}

X3bGate * x3b_gate_create(int device, int mode, double timeout_ms, char * err, int err_n) {
    DevGuard dg(device);
    X3bGate * g = new X3bGate();
    g->device = device;
    g->mode = mode == 1 ? 1 : 0;
    g->timeout_ms = timeout_ms > 0 ? timeout_ms : 1000.0;
    // the spin's own limit a little past the watchdog's: the watchdog normally opens it first (both are counted)
    g->spin_timeout_ns = (unsigned long long) ((g->timeout_ms + 250.0) * 1e6);
    cudaError_t e;
    bool ok = true;
    if (ok && (e = cudaStreamCreateWithFlags(&g->g, cudaStreamNonBlocking)) != cudaSuccess) ok = fail(err, err_n, "gate stream", e);
    if (ok && (e = cudaStreamCreateWithFlags(&g->r, cudaStreamNonBlocking)) != cudaSuccess) ok = fail(err, err_n, "release stream", e);
    if (ok && (e = cudaEventCreateWithFlags(&g->ev_open, cudaEventDisableTiming)) != cudaSuccess) ok = fail(err, err_n, "gate event", e);
    // load the gate's own kernels now, while nothing is armed: under lazy module loading the release kernel's first
    // launch came inside the run's first armed window and its load waited for the spinning gate (458c-458e, gate 1)
    if (ok) {
        cudaFuncAttributes fa;
        if (cudaFuncGetAttributes(&fa, x3b_spin) != cudaSuccess || cudaFuncGetAttributes(&fa, x3b_set) != cudaSuccess ||
            cudaFuncGetAttributes(&fa, x3b_probe) != cudaSuccess) { e = cudaGetLastError(); ok = fail(err, err_n, "gate kernels", e); }
    }
    if (ok) {   // lazy loading still on (CUDA_MODULE_LOADING set by the caller, or CUDA started before main): say so
        typedef CUresult (CUDAAPI * PfnLoadingMode)(CUmoduleLoadingMode *);
        void * f = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        CUmoduleLoadingMode lm = CU_MODULE_EAGER_LOADING;
        if (cudaGetDriverEntryPointByVersion("cuModuleGetLoadingMode", &f, 12000, cudaEnableDefault, &q) == cudaSuccess &&
            q == cudaDriverEntryPointSuccess && f && ((PfnLoadingMode) f)(&lm) == CUDA_SUCCESS && lm == CU_MODULE_LAZY_LOADING)
            std::fprintf(stderr, "ds41_dense: STRATA_X3B: WARNING - lazy CUDA module loading: a kernel launched for the first time "
                                 "inside an armed gate can stall until the watchdog (set CUDA_MODULE_LOADING=EAGER)\n");
        (void) cudaGetLastError();
    }
    if (ok && std::getenv("STRATA_X3B_STATS") && std::getenv("STRATA_X3B_STATS")[0] == '1' &&
        cudaStreamCreateWithFlags(&g->q, cudaStreamNonBlocking) != cudaSuccess) { (void) cudaGetLastError(); g->q = nullptr; }
    if (ok && (e = cudaHostAlloc((void **) &g->host, 256, cudaHostAllocMapped)) != cudaSuccess) ok = fail(err, err_n, "gate words", e);
    if (ok) {
        for (int i = 0; i < 64; ++i) ((volatile uint32_t *) g->host)[i] = 0;
        if ((e = cudaHostGetDevicePointer((void **) &g->dev, g->host, 0)) != cudaSuccess) ok = fail(err, err_n, "gate words (device view)", e);
    }
    if (ok && g->mode == 1) {
        cudaDriverEntryPointQueryResult q1 = cudaDriverEntryPointSymbolNotFound, q2 = cudaDriverEntryPointSymbolNotFound;
        void * f1 = nullptr, * f2 = nullptr;
        if (cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", &f1, 12000, cudaEnableDefault, &q1) != cudaSuccess ||
            q1 != cudaDriverEntryPointSuccess || !f1 ||
            cudaGetDriverEntryPointByVersion("cuStreamWriteValue32", &f2, 12000, cudaEnableDefault, &q2) != cudaSuccess ||
            q2 != cudaDriverEntryPointSuccess || !f2) {
            (void) cudaGetLastError();
            if (err && err_n > 0) std::snprintf(err, (size_t) err_n, "cuStreamWaitValue32 / cuStreamWriteValue32 not found in the driver");
            ok = false;
        } else {
            g->wait32 = (PfnWaitValue32) f1;
            g->write32 = (PfnWriteValue32) f2;
            // probe once, on the host-mapped word: a wait that is already satisfied (0 >= 0) and a write of 0
            if (g->wait32((CUstream) g->g, (CUdeviceptr) g->dev, 0, CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS ||
                g->write32((CUstream) g->r, (CUdeviceptr) g->dev, 0, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS ||
                cudaStreamSynchronize(g->g) != cudaSuccess || cudaStreamSynchronize(g->r) != cudaSuccess) {
                (void) cudaGetLastError();
                if (err && err_n > 0) std::snprintf(err, (size_t) err_n, "stream wait / write value operations on mapped host memory are not supported on this device / driver");
                ok = false;
            }
        }
    }
    if (!ok) { x3b_gate_destroy(g); return nullptr; }
    g->wd = std::thread([g] { g->watchdog(); });
    return g;
}

void x3b_gate_destroy(X3bGate * g) {
    if (!g) return;
    DevGuard dg(g->device);
    x3b_gate_abort(g);
    if (g->host) { *g->flag_host() = g->seq; std::atomic_thread_fence(std::memory_order_seq_cst); }   // (a released gate not yet open on the card too)
    if (g->wd.joinable()) {
        { std::lock_guard<std::mutex> lk(g->mu); g->quit = true; }
        g->cv.notify_all();
        g->wd.join();
    }
    if (g->g) cudaStreamSynchronize(g->g);
    if (g->r) cudaStreamSynchronize(g->r);
    if (g->q) { cudaStreamSynchronize(g->q); cudaStreamDestroy(g->q); }
    for (cudaEvent_t e : g->tev) if (e) cudaEventDestroy(e);
    if (g->ev_open) cudaEventDestroy(g->ev_open);
    if (g->host) cudaFreeHost(g->host);
    if (g->g) cudaStreamDestroy(g->g);
    if (g->r) cudaStreamDestroy(g->r);
    delete g;
}

const char * x3b_gate_mode(const X3bGate * g) { return g && g->mode == 1 ? "wait" : "spin"; }

bool x3b_gate_armed(X3bGate * g) {
    if (!g) return false;
    std::lock_guard<std::mutex> lk(g->mu);
    return g->armed && (int32_t) (*g->flag_host() - g->armed_seq) < 0;
}

void * x3b_gate_arm(X3bGate * g) {
    if (!g) return nullptr;
    {   // invariant: every armed gate was released (or aborted) before the next arm
        std::lock_guard<std::mutex> lk(g->mu);
        if (g->armed && (int32_t) (*g->flag_host() - g->armed_seq) < 0) {
            std::fprintf(stderr, "ds41_dense: STRATA_X3B: arm while gate %u is still closed - refused\n", g->armed_seq);
            return nullptr;
        }
    }
    DevGuard dg(g->device);
    const uint32_t s = g->seq + 1;
    if (g->mode == 1) {
        if (g->wait32((CUstream) g->g, (CUdeviceptr) g->dev, s, CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) return nullptr;
    } else {
        x3b_spin<<<1, 1, 0, g->g>>>(g->dev, s, g->spin_timeout_ns, (volatile int *) (g->dev + 16), g->spin_diag_dev());
        if (cudaGetLastError() != cudaSuccess) return nullptr;
    }
    if (cudaEventRecord(g->ev_open, g->g) != cudaSuccess) return nullptr;
    (void) cudaStreamQuery(g->g);   // WDDM: submit the gate now (the spin must be running before the release lands)
    g->seq = s;
    {
        X3bGate::Rec & r = g->rec[s % 8];
        r = X3bGate::Rec();
        r.seq = s;
        r.layer = g->note_layer.load();
        r.arm_ms = host_ms();
    }
    {
        std::lock_guard<std::mutex> lk(g->mu);
        g->armed = true;
        g->watch = true;
        g->armed_seq = s;
        g->armed_at = std::chrono::steady_clock::now();
    }
    g->cv.notify_all();
    return (void *) g->ev_open;
}

static bool x3b_write(X3bGate * g) {
    if (g->mode == 1) return g->write32((CUstream) g->r, (CUdeviceptr) g->dev, g->seq, CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
    x3b_set<<<1, 1, 0, g->r>>>(g->dev, g->seq, g->spin_diag_dev());
    return cudaGetLastError() == cudaSuccess;
}

// released (keep_watch: the watchdog still guards it until it opens on the card) or opened by the host (abort)
static void x3b_disarm(X3bGate * g, bool keep_watch) {
    {
        std::lock_guard<std::mutex> lk(g->mu);
        g->armed = false;
        if (!keep_watch) g->watch = false;
    }
    g->cv.notify_all();
}

bool x3b_gate_release(X3bGate * g, void * after_event) {
    if (!g) return false;
    {
        std::lock_guard<std::mutex> lk(g->mu);
        if (!g->armed || g->armed_seq != g->seq) return false;   // (not armed, or the watchdog opened it: the pass fails)
    }
    DevGuard dg(g->device);
    X3bGate::Rec & rr = g->rec[g->seq % 8];
    rr.rel0_ms = host_ms();
    rr.sum_ev = after_event;
    if (after_event) {   // (a query also hands the work up to it to the card under WDDM)
        const cudaError_t q = cudaEventQuery((cudaEvent_t) after_event);
        rr.sum_done = q == cudaSuccess ? 1 : 0;
        if (q != cudaSuccess && q != cudaErrorNotReady) (void) cudaGetLastError();
    }
    bool ok = true;
    if (g->q && after_event && cudaStreamWaitEvent(g->q, (cudaEvent_t) after_event, 0) == cudaSuccess) {   // the probe, first
        x3b_probe<<<1, 1, 0, g->q>>>(g->spin_diag_dev(), g->seq);
        (void) cudaGetLastError();
        (void) cudaStreamQuery(g->q);
    }
    if (after_event && cudaStreamWaitEvent(g->r, (cudaEvent_t) after_event, 0) != cudaSuccess) ok = false;
    if (ok) ok = x3b_write(g);
    (void) cudaStreamQuery(g->r);   // WDDM: submit the release now, not at the host's next wait
    rr.rel1_ms = host_ms();
    if (!ok) { *g->flag_host() = g->seq; std::atomic_thread_fence(std::memory_order_seq_cst); }   // never leave it closed
    x3b_disarm(g, ok);
    return ok;
}

void x3b_gate_abort(X3bGate * g) {
    if (!g) return;
    {
        std::lock_guard<std::mutex> lk(g->mu);
        if (!g->armed) return;
    }
    *g->flag_host() = g->seq;   // a host store: opens it at once, with no CUDA call
    std::atomic_thread_fence(std::memory_order_seq_cst);
    x3b_disarm(g, false);
}

long x3b_gate_timeouts(const X3bGate * g) {
    if (!g || !g->host) return 0;
    return (long) *g->spin_timeouts_host() + g->wd_fired.load();
}

void x3b_gate_note(X3bGate * g, int layer, int phase) {
    if (!g) return;
    g->note_layer.store(layer);
    g->note_phase.store(phase);
    g->note_t_us.store(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// a gate's host record and what the card shows of it now (the tier's sum event, the release stream, the release kernel)
static int x3b_describe_gate(X3bGate * g, uint32_t s, char * buf, int n) {
    int off = 0;
    volatile int32_t * d = g->spin_diag_host();
    const X3bGate::Rec & r = g->rec[s % 8];
    if (r.seq != s) return std::snprintf(buf, (size_t) n, "; no host record of gate %u", s);
    off += std::snprintf(buf + off, (size_t) (n - off), "; layer %d, host: release issued %.1f ms after the arm (the call took %.2f ms), "
                         "the sum event %s then",
                         r.layer, r.rel0_ms >= 0 ? r.rel0_ms - r.arm_ms : -1.0, r.rel1_ms >= 0 && r.rel0_ms >= 0 ? r.rel1_ms - r.rel0_ms : -1.0,
                         r.sum_done == 1 ? "complete" : r.sum_done == 0 ? "pending" : "absent (opened at once)");
    if (off < n && r.rel0_ms >= 0) {
        DevGuard dg(g->device);
        const cudaError_t qs = r.sum_ev ? cudaEventQuery((cudaEvent_t) r.sum_ev) : cudaSuccess;
        const cudaError_t qr = cudaStreamQuery(g->r);
        (void) cudaGetLastError();
        off += std::snprintf(buf + off, (size_t) (n - off), "; now: sum %s, release stream %s",
                             !r.sum_ev ? "-" : qs == cudaSuccess ? "complete" : "pending", qr == cudaSuccess ? "idle" : "busy");
        const unsigned long long t0 = ((volatile unsigned long long *) d)[5];   // the spin's start (spin gate)
        const unsigned long long tr = ((volatile unsigned long long *) d)[7];   // the release kernel ran
        const unsigned long long tp = ((volatile unsigned long long *) d)[8];   // the probe: the sum landed
        const bool spun = (uint32_t) d[8] == s, ran = (uint32_t) d[12] == s, landed = g->q && (uint32_t) d[13] == s;
        if (off < n && spun && ran)
            off += std::snprintf(buf + off, (size_t) (n - off), "; the release ran on the card %.1f ms after the spin started", (double) (long long) (tr - t0) / 1e6);
        else if (off < n && !ran)
            off += std::snprintf(buf + off, (size_t) (n - off), "; the release has not run on the card (last released gate %u)", (uint32_t) d[12]);
        if (off < n && landed && spun)
            off += std::snprintf(buf + off, (size_t) (n - off), "; the sum landed %.1f ms after the spin started", (double) (long long) (tp - t0) / 1e6);
        else if (off < n && landed && ran)
            off += std::snprintf(buf + off, (size_t) (n - off), "; the sum landed %.3f ms before the release ran", (double) (long long) (tr - tp) / 1e6);
    }
    return off;
}

void x3b_gate_describe(X3bGate * g, char * buf, int n) {
    if (!buf || n < 1) return;
    buf[0] = 0;
    if (!g || !g->host) return;
    int off = 0;
    const long spins = (long) *g->spin_timeouts_host(), wds = g->wd_fired.load();
    if (spins > g->described_spin) {   // a spin gave up since the last description
        g->described_spin = spins;
        volatile int32_t * d = g->spin_diag_host();
        const uint32_t s = (uint32_t) d[0];
        off += std::snprintf(buf + off, (size_t) (n - off),
                             " [spin gave up gate %u: waited %.1f ms by its count (raw clock span %.1f ms, %d steps backwards, largest step %.3f ms)",
                             s, d[1] / 1000.0, d[2] / 1000.0, (int) d[3], d[4] / 1000.0);
        if (off < n) off += x3b_describe_gate(g, s, buf + off, n - off);
        if (off < n) off += std::snprintf(buf + off, (size_t) (n - off), "]");
    }
    if (wds > g->described_wd && off < n) {   // the watchdog fired since the last description
        g->described_wd = wds;
        off += std::snprintf(buf + off, (size_t) (n - off),
                             " [watchdog: gate %u, %.0f ms after its arm; decode thread %s for %.0f ms; at the firing the tier's sum %s, the release kernel %s",
                             g->wd_seq, g->wd_since_arm_ms, x3b_phase_name(g->wd_phase), g->wd_since_note_ms,
                             g->wd_sum_landed == 1 ? "had landed" : g->wd_sum_landed == 0 ? "had NOT landed" : "(no probe)",
                             g->wd_rel_ran == 1 ? "had run" : "had NOT run");
        if (off < n) off += x3b_describe_gate(g, g->wd_seq, buf + off, n - off);
        if (off < n) off += std::snprintf(buf + off, (size_t) (n - off), "]");
    }
}

void * x3b_gate_tev(X3bGate * g, int idx) {
    if (!g || idx < 0) return nullptr;
    DevGuard dg(g->device);
    while ((int) g->tev.size() <= idx) {
        cudaEvent_t e = nullptr;
        if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
        g->tev.push_back(e);
        g->tev_rec.push_back(false);
    }
    g->tev_rec[(size_t) idx] = true;   // the caller records it now (on the dense stream)
    return (void *) g->tev[(size_t) idx];
}

bool x3b_gate_mark(X3bGate * g, void * after_event, int idx) {
    cudaEvent_t e = (cudaEvent_t) x3b_gate_tev(g, idx);
    if (!e) return false;
    DevGuard dg(g->device);
    if (after_event && cudaStreamWaitEvent(g->r, (cudaEvent_t) after_event, 0) != cudaSuccess) return false;
    return cudaEventRecord(e, g->r) == cudaSuccess;
}

double x3b_gate_elapsed(X3bGate * g, int a, int b) {
    if (!g || a < 0 || b < 0 || a >= (int) g->tev.size() || b >= (int) g->tev.size() || !g->tev_rec[(size_t) a] || !g->tev_rec[(size_t) b])
        return -1.0;
    DevGuard dg(g->device);
    float ms = 0.0f;
    if (cudaEventSynchronize(g->tev[(size_t) a]) != cudaSuccess || cudaEventSynchronize(g->tev[(size_t) b]) != cudaSuccess ||
        cudaEventElapsedTime(&ms, g->tev[(size_t) a], g->tev[(size_t) b]) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1.0;
    }
    g->tev_rec[(size_t) a] = g->tev_rec[(size_t) b] = false;
    return (double) ms;
}

}  // namespace v41
}  // namespace strata
