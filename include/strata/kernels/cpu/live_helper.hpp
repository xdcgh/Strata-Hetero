#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <cstdlib>
#include <cstring>
namespace strata::kernels::cpu {
inline bool live_helper_forced() noexcept {
    const char* mode=std::getenv("STRATA_EXPERT_HELPER");
    return mode && !std::strcmp(mode,"forced");
}
// Experimental forced-only boundary. False leaves every output untouched.
bool live_helper(const NativeFmt&, const ExpertJobMulti&, const float* const* raw,
                 int layer, int expert, const char* phase) noexcept;
}
