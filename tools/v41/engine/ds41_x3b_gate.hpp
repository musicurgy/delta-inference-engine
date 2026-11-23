// tools/v41/engine/ds41_x3b_gate.hpp - STRATA_X3B's device-side gate.
//
// The dense stream cannot wait (cudaStreamWaitEvent) for the routed sum of a tier call the host has not issued yet: an
// event wait binds to the event's LAST record at the time of the call.  The gate gets around that with a flag:
//   arm()     - on the gate's own stream G: wait until flag >= seq (a one-warp spin kernel, or cuStreamWaitValue32),
//               then record ev_open; the caller makes the dense stream wait for ev_open (an ordinary event wait - the
//               record is already queued), then launches the work that must run after the sum.
//   release() - on the gate's release stream R: wait for the tier's sum event, then write flag = seq.
// So the dense stream's work starts the moment the sum lands, and the host launched it while it was still waiting for
// the router readback.  Plain stream operations on our own streams: nothing is captured into a CUDA graph.
//
// Plain C++ interface (no CUDA types): cudaEvent_t / cudaStream_t travel as void *.
#pragma once

namespace strata {
namespace v41 {

struct X3bGate;

/// mode 0: spin kernel (default); mode 1: cuStreamWaitValue32 / cuStreamWriteValue32 (no SM held).  Both wait on a
/// flag in mapped pinned host memory, and a host watchdog thread opens a gate still shut on the card timeout_ms after its arm
/// (released or not)
/// with a plain store (no CUDA call: it works while the decode thread is stuck in the driver) and counts it - so
/// neither mode can hang the process.  The spin kernel also gives up by itself after timeout_ms + 250 of its own
/// waiting, counted on %globaltimer in steps (a step backwards counts 0, a step forwards at most 1 ms: a clock
/// adjustment is not a wait - 11-17).  nullptr + err on failure.
X3bGate * x3b_gate_create(int device, int mode, double timeout_ms, char * err, int err_n);
void x3b_gate_destroy(X3bGate * g);
const char * x3b_gate_mode(const X3bGate * g);   ///< "spin" / "wait"
/// queue the next gate on G; returns ev_open (a cudaEvent_t), recorded on G behind the gate - the dense stream waits on it
void * x3b_gate_arm(X3bGate * g);
/// open the armed gate on R once after_event (a cudaEvent_t, may be null = at once) has completed on the card
bool x3b_gate_release(X3bGate * g, void * after_event);
/// open the armed gate at once (error paths: never leave a gated dense stream behind)
void x3b_gate_abort(X3bGate * g);
/// gates opened by the watchdog or given up by the spin kernel (their dense work ran early: the pass is invalid)
long x3b_gate_timeouts(const X3bGate * g);
/// the decode thread's host phase (X3B_PH_*) and layer, for the diagnostics: the watchdog prints them when it fires
enum { X3B_PH_IDLE = 0, X3B_PH_LAUNCH, X3B_PH_READBACK, X3B_PH_TIER, X3B_PH_RELEASE, X3B_PH_RELEASED, X3B_PH_N };
void x3b_gate_note(X3bGate * g, int layer, int phase);
/// a line on the last timed-out gate(s): which gate and layer, the host's arm -> release times, the phase the decode
/// thread was in, and the spin kernel's own account (its waited time, the raw clock span, clock steps backwards / the
/// largest step) - so a timeout says whether the card waited for the release or the clock jumped; "" if none
void x3b_gate_describe(X3bGate * g, char * buf, int n);
/// armed and not yet open (the invariant check: every arm reaches a release before the next)
bool x3b_gate_armed(X3bGate * g);

// ---- STRATA_X3B_STATS: timing events (created on first use, with timing)
void * x3b_gate_tev(X3bGate * g, int idx);                 ///< timing event #idx (a cudaEvent_t)
/// on R: wait for after_event (may be null), then record timing event #idx
bool x3b_gate_mark(X3bGate * g, void * after_event, int idx);
/// ms from timing event a to b (waits for both); < 0 if either was never recorded
double x3b_gate_elapsed(X3bGate * g, int a, int b);

}  // namespace v41
}  // namespace strata
