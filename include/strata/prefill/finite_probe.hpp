// CUDA-only, opt-in diagnostics. No model arithmetic or activation writes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill::finite_probe {

inline bool requested(const char* value) {
    return value && value[0] == '1' && value[1] == '\0';
}

enum class Stage : int {
    input_residual = 1,
    ple_residual,
    hc_mixed,
    gdn_output,
    qsa_output,
    moe_output,
    residual_write,
};

inline const char* stage_name(Stage stage) {
    switch (stage) {
    case Stage::input_residual: return "input_residual";
    case Stage::ple_residual: return "ple_residual";
    case Stage::hc_mixed: return "hc_mixed";
    case Stage::gdn_output: return "gdn_output";
    case Stage::qsa_output: return "qsa_output";
    case Stage::moe_output: return "moe_output";
    case Stage::residual_write: return "residual_write";
    }
    return "unknown";
}

struct Checkpoint {
    int64_t chunk = 0, pos0 = 0, layer = -1, rows = 0, width = 0;
    int half = -1;
    Stage stage = Stage::input_residual;
};

// The earliest scan containing a non-finite value wins. Within that scan the
// lowest element index and complete NaN/+Inf/-Inf counts are deterministic.
// The record belongs to one stream and one run_impl call, not all GPU stages.
struct Record {
    unsigned ordinal = 0;
    Checkpoint point{};
    unsigned long long index = ~0ull;
    unsigned long long nan = 0, positive_inf = 0, negative_inf = 0;
};

class Probe {
public:
    Probe() = default;
    ~Probe();
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    // Disabled: no CUDA call, allocation, launch, transfer, or synchronization.
    bool init(bool enabled, void* stream, int device, int64_t pos0, int64_t tokens, std::string& err);
    bool scan(const float* values, const Checkpoint& point, std::string& err);
    bool enqueue_read(std::string& err);
    // Call only after the caller's existing stream synchronization completes.
    void report_after_sync(int64_t chunk, int64_t pos0, int64_t rows);
    bool allocated() const { return device_record_ != nullptr; }
    const Record* completed_record() const { return host_record_; }

private:
    bool check(int status, const char* operation, std::string& err);
    void* stream_ = nullptr;
    Record* device_record_ = nullptr;
    Record* host_record_ = nullptr;
    unsigned ordinal_ = 0;
    unsigned long long run_ = 0;
    int device_ = 0;
    bool read_pending_ = false;
};

}  // namespace strata::prefill::finite_probe
