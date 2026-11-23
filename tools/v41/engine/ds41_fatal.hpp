// tools/v41/engine/ds41_fatal.hpp - the engine's fatal stop.
//
// abort() on Windows runs Windows Error Reporting before the process ends; with 100+ GB of pinned expert arena that
// pass (the fault report / dump) takes minutes, and the server cannot start a fresh engine until it is over
// (ds41_end_if_gpu_fault already avoids it).  ds41_fatal_exit() flushes the log (stderr) and ends with exit code 3 - the code
// abort() gives on Windows, so the server sees the same end - without that pass.  Elsewhere it is abort() as before.
// STRATA_FATAL_ABORT=1 keeps abort() on Windows too (a WER dump for a debugger).
#pragma once

#include "ds41_env.hpp"

#include <cstdio>
#include <cstdlib>

[[noreturn]] inline void ds41_fatal_exit() {
    std::fflush(stderr);   // the log; stdout (the protocol) is left as abort() left it
#if defined(_WIN32)
    const char * e = ds41_env("STRATA_FATAL_ABORT", "STRATA_P3_HOST_ABORT");
    if (!(e && e[0] == '1')) std::_Exit(3);
#endif
    std::abort();
}
