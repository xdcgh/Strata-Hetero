#pragma once
#include <cstdint>
#include <string>

namespace strata::core::verify_probe {
enum class Format : int { f32, f16, bf16 };
enum class Stage : int { input_residual, hc_attention, attention_output, hc_ffn,
    route_logits, route_weights, shared_input_bf16, shared_scalar_gate, shared_output, routed_parts, moe_output,
    head_mixed, head_logits };
struct Point {
    int64_t layer = -1, rows = 0, width = 0, row_offset = 0;
    int half = -1;
    Stage stage = Stage::input_residual;
    Format format = Format::f32;
    int64_t rows_per_token = 1; // routed_parts: K slots per token, not expert-order rows
};
struct Record {
    unsigned ordinal = 0, checkpoints = 0;
    Point point{};
    unsigned long long index = ~0ull, nan = 0, positive_inf = 0, negative_inf = 0;
};
constexpr int kRouteMaxWidth = 4096;
constexpr int kRouteMaxQ8Bytes = kRouteMaxWidth / 32 * 36;
// Group-local pointers, valid only at the post/scan boundary. Packed q8_1
// bytes are never inspected as FP32. counts[1] is the number of GPU entries.
struct RouteView {
    const int32_t *ids = nullptr, *counts = nullptr, *dst = nullptr, *start2 = nullptr;
    const float *weights = nullptr, *mixed = nullptr, *parts = nullptr, *hit = nullptr, *cpu = nullptr;
    const uint8_t* native_q8 = nullptr;
    const uint32_t* skip = nullptr;
    int rows = 0, width = 0, k = 0, experts = 0;
    bool all_resident = false, device_plan = false, remote = false, native_q8_valid = false;
    uint32_t ring = 0;
    int gu_type = -1, down_type = -1;
};
struct RouteSnapshot {
    int valid = 0, error = 0, ordinal = 0, width = 0, k = 0;
    int local_token = -1, slot = -1, expert = -1, gpu_entry = -1, gpu_entries = 0;
    int counts[3]{}, gpu_valid = 0, cpu_valid = 0, native_q8_valid = 0, gpu_pcie = 0;
    int all_resident = 0, device_plan = 0, remote = 0, skipped = 0, gu_type = -1, down_type = -1;
    float weight = 0;
    uint32_t ring = 0;
    float mixed[kRouteMaxWidth], parts[kRouteMaxWidth], hit[kRouteMaxWidth], cpu[kRouteMaxWidth];
    uint8_t native_q8[kRouteMaxQ8Bytes];
};
// One owner per Verifier/device. Allocate before graph capture; reset/scan/read
// enqueue only capture-safe operations. A captured reset runs on EVERY replay.
class Probe {
public:
    Probe() = default;
    ~Probe();
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;
    bool init(bool enabled, void* stream, int device, std::string& err, const char* route_prefix = nullptr);
    void release(); // caller drains stream and destroys graphs first
    bool begin_graph(std::string& err);
    bool scan(const void* values, Point point, std::string& err);
    bool end_graph(std::string& err);
    bool capture_route(RouteView view, std::string& err);
    bool report_route(std::string& err); // after report() and existing completion
    const RouteSnapshot* completed_route() const { return host_route_; }
    // Caller has observed its EXISTING graph completion sync/event. Positions
    // are runtime host metadata, never constants baked into a captured graph.
    // Returns the reported runtime position, or -1 for finite/no-token results.
    int64_t report(int rows, int64_t pos0, const int64_t* positions = nullptr);
    bool allocated() const { return device_record_ != nullptr; }
    const Record* completed_record() const { return host_record_; }
private:
    bool check(int status, const char* operation, std::string& err);
    Record* device_record_ = nullptr;
    Record* host_record_ = nullptr;
    RouteSnapshot *device_route_ = nullptr, *host_route_ = nullptr;
    std::string route_prefix_;
    void* stream_ = nullptr;
    int device_ = -1;
    unsigned ordinal_ = 0;
    uint64_t window_ = 0;
    uint64_t owner_ = 0;
    int64_t last_position_ = -1;
};
inline bool requested(const char* value) { return value && value[0]=='1' && value[1]=='\0'; }
}
