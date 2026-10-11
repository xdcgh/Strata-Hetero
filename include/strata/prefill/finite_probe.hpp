// CUDA-only, opt-in diagnostics. No model arithmetic or activation writes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill::finite_probe {

inline bool requested(const char* value) {
    return value && value[0] == '1' && value[1] == '\0';
}

// Absent: no detail scans. Strict decimal, within this model's layer range.
inline bool parse_detail_layer(const char* value, int64_t layers, int& layer, std::string& err) {
    layer = -1;
    if (!value) return true;
    if (!*value || layers <= 0) { err = "prefill finite probe: invalid detail layer"; return false; }
    int64_t number = 0;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || number > (layers - 1) / 10) {
            err = "prefill finite probe: invalid detail layer"; return false;
        }
        number = number * 10 + (*p - '0');
        if (number >= layers) { err = "prefill finite probe: detail layer out of range"; return false; }
    }
    layer = (int) number;
    return true;
}

enum class Format : int { f32 = 0, f16 = 1, bf16 = 2 };
inline const char* format_name(Format format) {
    switch (format) {
    case Format::f32: return "f32";
    case Format::f16: return "f16";
    case Format::bf16: return "bf16";
    }
    return "unknown";
}

enum class Stage : int {
    input_residual = 1,
    ple_residual,
    hc_mixed,
    gdn_output,
    qsa_output,
    moe_output,
    residual_write,
    hc_mixed_f16,
    hc_mixed_bf16,
    hc_mixed_bf16_low,
    route_logits,
    route_weights,
    shared_gate,
    shared_up,
    shared_activation_f16,
    shared_output,
    shared_scalar_gate,
    routed_gate_up,
    routed_input_f16,
    dequant_gate_up_f16,
    dequant_down_f16,
    routed_activation_f16,
    routed_activation_f32,
    routed_output,
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
    case Stage::hc_mixed_f16: return "hc_mixed_f16";
    case Stage::hc_mixed_bf16: return "hc_mixed_bf16";
    case Stage::hc_mixed_bf16_low: return "hc_mixed_bf16_low";
    case Stage::route_logits: return "route_logits";
    case Stage::route_weights: return "route_weights";
    case Stage::shared_gate: return "shared_gate";
    case Stage::shared_up: return "shared_up";
    case Stage::shared_activation_f16: return "shared_activation_f16";
    case Stage::shared_output: return "shared_output";
    case Stage::shared_scalar_gate: return "shared_scalar_gate";
    case Stage::routed_gate_up: return "routed_gate_up";
    case Stage::routed_input_f16: return "routed_input_f16";
    case Stage::dequant_gate_up_f16: return "dequant_gate_up_f16";
    case Stage::dequant_down_f16: return "dequant_down_f16";
    case Stage::routed_activation_f16: return "routed_activation_f16";
    case Stage::routed_activation_f32: return "routed_activation_f32";
    case Stage::routed_output: return "routed_output";
    }
    return "unknown";
}

struct Checkpoint {
    int64_t chunk = 0, pos0 = 0, layer = -1, rows = 0, width = 0;
    int half = -1;
    Stage stage = Stage::input_residual;
    // Routed group rows are in expert order: row_offset and expert preserve
    // that coordinate. They are not automatically the original token index.
    int64_t row_offset = 0;
    int expert = -1;
    Format format = Format::f32;
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
    bool scan16(const uint16_t* values, const Checkpoint& point, Format format, std::string& err);
    void report_moe_path(const Checkpoint& point, int experts, bool native, bool mmq, bool fused,
                         int gu_type, int down_type, bool stream_all, bool peer,
                         bool peer_sums, int64_t cpu_rows);
    bool enqueue_read(std::string& err);
    // Call only after the caller's existing stream synchronization completes.
    void report_after_sync(int64_t chunk, int64_t pos0, int64_t rows);
    bool allocated() const { return device_record_ != nullptr; }
    const Record* completed_record() const { return host_record_; }

private:
    bool check(int status, const char* operation, std::string& err);
    bool scan_impl(const void* values, Checkpoint point, Format format, std::string& err);
    void* stream_ = nullptr;
    Record* device_record_ = nullptr;
    Record* host_record_ = nullptr;
    unsigned ordinal_ = 0;
    unsigned long long run_ = 0;
    int device_ = 0;
    bool read_pending_ = false;
};

}  // namespace strata::prefill::finite_probe
