// Owned pool threads only; no model weights or GPU runtime.
#include "strata/kernels/cpu/pool.hpp"
#include <chrono>
#include <cstdio>
#include <thread>

namespace cpu = strata::kernels::cpu;

static bool inspect(bool pin) {
    cpu::ExpertPool pool(2, pin, false, cpu::PoolAffinity::All);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::vector<cpu::WorkerAffinityReport> reports;
    do {
        reports = pool.worker_affinity();
        if (reports.size() == 2 && reports[0].ready && reports[1].ready) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    if (reports.size() != 2) return false;
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto& r = reports[i];
        if (!r.ready || r.worker != static_cast<int>(i) || r.pin_requested != pin) return false;
        if (!pin && (r.requested_core != -1 || r.pin_applied || r.mask_matches_request)) return false;
#if defined(_WIN32)
        if (!r.mask_observed || r.startup_processor < 0) return false;
        if (pin && (!r.pin_applied || !r.mask_matches_request || r.requested_core < 0)) return false;
#else
        if (r.mask_observed || r.startup_processor != -1) return false; // Readback unavailable on this platform.
#endif
    }
    const auto again = pool.worker_affinity();
    for (size_t i = 0; i < reports.size(); ++i) {
        if (again[i].requested_core != reports[i].requested_core ||
            again[i].pin_applied != reports[i].pin_applied ||
            again[i].observed_mask != reports[i].observed_mask || !again[i].ready) return false;
    }
    return true;
}

int main() {
    if (!inspect(false) || !inspect(true)) {
        std::fprintf(stderr, "Worker startup affinity observation failed\n");
        return 1;
    }
    std::puts("Worker startup affinity observations verified");
    return 0;
}
