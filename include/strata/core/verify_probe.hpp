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
// One owner per Verifier/device. Allocate before graph capture; reset/scan/read
// enqueue only capture-safe operations. A captured reset runs on EVERY replay.
class Probe {
public:
    Probe() = default;
    ~Probe();
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;
    bool init(bool enabled, void* stream, int device, std::string& err);
    void release(); // caller drains stream and destroys graphs first
    bool begin_graph(std::string& err);
    bool scan(const void* values, Point point, std::string& err);
    bool end_graph(std::string& err);
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
    void* stream_ = nullptr;
    int device_ = -1;
    unsigned ordinal_ = 0;
    uint64_t window_ = 0;
    uint64_t owner_ = 0;
};
inline bool requested(const char* value) { return value && value[0]=='1' && value[1]=='\0'; }
}
