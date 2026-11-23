// tools/v41/engine/ds41_env.hpp - settings read under a descriptive name with an older name as an alias.
//
// Some switches first shipped in the 0.95 release candidates under internal plan labels (STRATA_P3_*, STRATA_D11_*).
// They now have descriptive names (docs/DETAILS.md lists both); the old name is still read when the new one is not set,
// so scripts and launch files written for the candidates keep working.  The new name wins when both are set.
#pragma once

#include <cstdlib>

/// getenv(name), else getenv(old_name): nullptr when neither is set.
inline const char * ds41_env(const char * name, const char * old_name) {
    const char * v = std::getenv(name);
    return v ? v : std::getenv(old_name);
}
