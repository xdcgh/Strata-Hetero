// Synthetic device diagnostics only: no model, engine, GEMM, or timing claim.
#include "strata/prefill/finite_probe.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#define strata_dup _dup
#define strata_dup2 _dup2
#define strata_close _close
#define strata_fileno _fileno
#else
#include <unistd.h>
#define strata_dup dup
#define strata_dup2 dup2
#define strata_close close
#define strata_fileno fileno
#endif

namespace fp = strata::prefill::finite_probe;
namespace {
void require(bool okay, const char* what) { if (!okay) throw std::runtime_error(what); }
void require(bool okay, const std::string& what) { if (!okay) throw std::runtime_error(what); }
void ck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
struct Device {
    float* values = nullptr;
    explicit Device(size_t count) { ck(cudaMalloc(&values, count * sizeof(float)), "allocate inputs"); }
    ~Device() { cudaFree(values); }
};
struct Device16 {
    uint16_t* values = nullptr;
    explicit Device16(size_t count) { ck(cudaMalloc(&values, count * sizeof(uint16_t)), "allocate 16-bit inputs"); }
    ~Device16() { cudaFree(values); }
};

struct StderrCapture {
    FILE* file = nullptr;
    int saved = -1;
    StderrCapture() {
        file = std::tmpfile();
        if (!file) throw std::runtime_error("create stderr capture");
        std::fflush(stderr);
        saved = strata_dup(strata_fileno(stderr));
        if (saved < 0 || strata_dup2(strata_fileno(file), strata_fileno(stderr)) < 0) {
            if (saved >= 0) strata_close(saved);
            std::fclose(file);
            throw std::runtime_error("redirect stderr capture");
        }
    }
    ~StderrCapture() {
        if (saved >= 0) {
            std::fflush(stderr);
            strata_dup2(saved, strata_fileno(stderr));
            strata_close(saved);
        }
        if (file) std::fclose(file);
    }
    std::string read() {
        std::fflush(stderr);
        std::rewind(file);
        std::string data;
        char buffer[512];
        size_t n = 0;
        while ((n = std::fread(buffer, 1, sizeof(buffer), file)) != 0) data.append(buffer, n);
        return data;
    }
};

void disabled_case() {
    require(!fp::requested(nullptr) && !fp::requested("") && !fp::requested("0") &&
            !fp::requested("true") && !fp::requested("10") && fp::requested("1"), "exact opt-in");
    fp::Probe probe;
    std::string err;
    require(probe.init(false, nullptr, 0, 0, 0, err), "disabled init");
    require(!probe.allocated() && probe.completed_record() == nullptr, "disabled owns no buffers");
    require(probe.scan(nullptr, {}, err) && probe.scan16(nullptr, {}, fp::Format::f16, err) &&
            probe.enqueue_read(err), "disabled accepts no-device calls");
    probe.report_moe_path({}, 45, true, false, false, 1, 2, false, false, false, 0);
    probe.report_after_sync(0, 0, 0);
    std::puts("PASS disabled: no probe CUDA calls (before any context creation)");
}

void detail_layer_cases() {
    int layer = 99;
    std::string err;
    require(fp::parse_detail_layer(nullptr, 48, layer, err) && layer == -1 && err.empty(),
            "absent detail layer disables detail scans");
    require(fp::parse_detail_layer("0", 48, layer, err) && layer == 0, "detail layer zero");
    require(fp::parse_detail_layer("45", 48, layer, err) && layer == 45, "detail layer 45");
    require(fp::parse_detail_layer("47", 48, layer, err) && layer == 47, "last valid detail layer");
    for (const char* value : {"", "-1", "+1", " 1", "1 ", "48", "4x", "999999999999999999999999999999999"}) {
        err.clear();
        require(!fp::parse_detail_layer(value, 48, layer, err) && layer == -1 && !err.empty(),
                std::string("invalid detail layer rejected: ") + (value[0] ? value : "<empty>"));
    }
    require(std::strcmp(fp::format_name(fp::Format::f32), "f32") == 0 &&
            std::strcmp(fp::format_name(fp::Format::f16), "f16") == 0 &&
            std::strcmp(fp::format_name(fp::Format::bf16), "bf16") == 0, "format names");
    require(std::strcmp(fp::stage_name(fp::Stage::route_logits), "route_logits") == 0 &&
            std::strcmp(fp::stage_name(fp::Stage::routed_activation_f16), "routed_activation_f16") == 0 &&
            std::strcmp(fp::stage_name(fp::Stage::routed_output), "routed_output") == 0,
            "new MoE stage names");
    std::puts("PASS host: detail layer parser boundaries and MoE stage/format names");
}

void one_case(size_t count, bool bad, bool overwrite, int repeat) {
    Stream stream;
    std::vector<float> values(count, 0.125f);
    if (bad) {
        values[0] = std::numeric_limits<float>::infinity();
        if (count > 1) values[count - 1] = -std::numeric_limits<float>::infinity();
        if (count > 2) values[count / 2] = std::numeric_limits<float>::quiet_NaN();
    }
    Device device(count);
    ck(cudaMemcpy(device.values, values.data(), count * sizeof(float), cudaMemcpyHostToDevice), "upload");
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 23, (int64_t) count, err), err);
    const fp::Checkpoint first{2, 23, 7, 1, (int64_t) count, 0, fp::Stage::gdn_output};
    require(probe.scan(device.values, first, err), err);
    if (overwrite) ck(cudaMemsetAsync(device.values, 0, count * sizeof(float), stream.value), "overwrite after scan");
    const fp::Checkpoint later{3, 24, 8, 1, (int64_t) count, 1, fp::Stage::moe_output};
    require(probe.scan(device.values, later, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "existing boundary");
    probe.report_after_sync(3, 24, 1);
    const fp::Record& r = *probe.completed_record();
    if (!bad) require(r.ordinal == 0 && r.nan == 0 && r.positive_inf == 0 && r.negative_inf == 0, "finite record");
    else {
        require(r.ordinal == 1 && r.index == 0 && r.point.stage == fp::Stage::gdn_output &&
                r.point.chunk == 2 && r.point.pos0 == 23 && r.point.layer == 7 && r.point.half == 0 &&
                r.point.rows == 1 && r.point.width == (int64_t) count, "first stage and minimum index");
        require(r.positive_inf == 1 && r.negative_inf == (count > 1 ? 1ull : 0ull) &&
                r.nan == (count > 2 ? 1ull : 0ull), "complete first-stage nonfinite counts");
    }
    std::vector<float> after(count);
    ck(cudaMemcpy(after.data(), device.values, count * sizeof(float), cudaMemcpyDeviceToHost), "verify inputs");
    if (overwrite) for (float value : after) require(value == 0, "producer overwrite order");
    else require(std::memcmp(after.data(), values.data(), count * sizeof(float)) == 0, "probe preserves input bytes");
    // A second boundary must preserve the earliest record, including counts.
    const fp::Record saved = r;
    require(probe.scan(device.values, later, err) && probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "second boundary");
    probe.report_after_sync(4, 25, 1);
    const fp::Record& again = *probe.completed_record();
    require(again.ordinal == saved.ordinal && again.index == saved.index && again.nan == saved.nan &&
            again.positive_inf == saved.positive_inf && again.negative_inf == saved.negative_inf,
            "later scans do not replace or recount the first stage");
    std::printf("PASS device: elements=%zu bad=%d overwrite=%d repeat=%d\n", count, (int) bad, (int) overwrite, repeat);
}

void finite_then_nan() {
    Stream stream;
    Device device(1025);
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 50, 1025, err), err);
    ck(cudaMemsetAsync(device.values, 0, 1025 * sizeof(float), stream.value), "clear producer");
    require(probe.scan(device.values, {0, 50, 0, 1, 1025, -1, fp::Stage::input_residual}, err), err);
    // 0xffffffff is NaN. The write and next scan are ordered on the same stream.
    ck(cudaMemsetAsync(device.values + 513, 0xff, sizeof(float), stream.value), "later NaN producer");
    require(probe.scan(device.values, {1, 51, 3, 1, 1025, 1, fp::Stage::moe_output}, err), err);
    require(probe.scan(device.values, {2, 52, 4, 1, 1025, 0, fp::Stage::qsa_output}, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "boundary");
    probe.report_after_sync(2, 52, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 2 && r.index == 513 && r.nan == 1 && r.positive_inf == 0 && r.negative_inf == 0 &&
            r.point.stage == fp::Stage::moe_output && r.point.layer == 3, "finite then first NaN checkpoint");
    std::puts("PASS device: finite stage before later NaN, nonzero minimum index");
}

void dense_nan() {
    Stream stream;
    constexpr int64_t count = 65537;
    Device device((size_t) count);
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 0, count, err), err);
    ck(cudaMemsetAsync(device.values, 0xff, count * sizeof(float), stream.value), "all-NaN producer");
    require(probe.scan(device.values, {0, 0, 11, 1, count, 1, fp::Stage::moe_output}, err), err);
    require(probe.enqueue_read(err), err);
    require(!probe.enqueue_read(err), "duplicate pending read rejected");
    err.clear();
    ck(cudaStreamSynchronize(stream.value), "dense boundary");
    probe.report_after_sync(0, 0, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 1 && r.index == 0 && r.nan == (unsigned long long) count &&
            r.positive_inf == 0 && r.negative_inf == 0, "all-NaN count across blocks");
    require(!probe.scan(device.values, {0, 0, 12, 0, count, 1, fp::Stage::residual_write}, err),
            "zero row count rejected before launch");
    std::puts("PASS device: all-NaN count across blocks and invalid API guards");
}

void finite_f16_edges() {
    Stream stream;
    const std::vector<uint16_t> values = {0x0000u, 0x8000u, 0x0001u, 0x7bffu, 0xfbffu};
    Device16 device(values.size());
    ck(cudaMemcpy(device.values, values.data(), values.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload f16 finite edges");
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 70, (int64_t) values.size(), err), err);
    fp::Checkpoint point{0, 70, 45, 1, (int64_t) values.size(), 0, fp::Stage::routed_activation_f16};
    point.row_offset = 5;
    point.expert = 3;
    require(!probe.scan16(device.values, point, fp::Format::f32, err), "scan16 rejects f32 format");
    err.clear();
    fp::Checkpoint invalid_point = point;
    invalid_point.row_offset = -1;
    require(!probe.scan16(device.values, invalid_point, fp::Format::f16, err), "negative row offset rejected");
    err.clear();
    require(probe.scan16(device.values, point, fp::Format::f16, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "f16 finite boundary");
    probe.report_after_sync(0, 70, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 0 && r.nan == 0 && r.positive_inf == 0 && r.negative_inf == 0,
            "f16 signed zero, subnormal, and max finite stay finite");
    std::vector<uint16_t> after(values.size());
    ck(cudaMemcpy(after.data(), device.values, after.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read f16 finite edges");
    require(std::memcmp(after.data(), values.data(), values.size() * sizeof(uint16_t)) == 0,
            "f16 finite scan preserves input bytes");
    std::puts("PASS device: f16 signed zeros, subnormal, and maximum finite");
}

void f16_cross_block_first_record() {
    Stream stream;
    constexpr size_t count = 1025;
    std::vector<uint16_t> values(count, 0x3c00u);
    values[255] = 0x7c00u;  // +Inf, lowest index
    values[256] = 0xfc00u;  // -Inf, next block
    values[513] = 0x7e55u;  // +NaN
    values[514] = 0xfc01u;  // negative signaling NaN
    values[1024] = 0x7c00u; // +Inf, final block
    Device16 device(count);
    ck(cudaMemcpy(device.values, values.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload f16 classifications");
    std::vector<float> later_values(16, 0.0f);
    later_values[0] = std::numeric_limits<float>::quiet_NaN();
    Device later_device(later_values.size());
    ck(cudaMemcpy(later_device.values, later_values.data(), later_values.size() * sizeof(float), cudaMemcpyHostToDevice), "upload later f32 NaN");

    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 71, (int64_t) count, err), err);
    fp::Checkpoint half_point{2, 71, 45, 1, (int64_t) count, 0, fp::Stage::routed_activation_f16};
    half_point.row_offset = 1000;
    half_point.expert = 7;
    require(probe.scan16(device.values, half_point, fp::Format::f16, err), err);
    require(probe.scan(later_device.values, {3, 72, 45, 1, (int64_t) later_values.size(), 1,
                                            fp::Stage::routed_activation_f32}, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "f16/f32 first-record boundary");
    probe.report_after_sync(3, 72, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 1 && r.index == 255 && r.nan == 2 && r.positive_inf == 2 && r.negative_inf == 1,
            "f16 lowest cross-block index and complete signed counts");
    require(r.point.stage == fp::Stage::routed_activation_f16 && r.point.format == fp::Format::f16 &&
            r.point.row_offset == 1000 && r.point.expert == 7 && r.point.layer == 45,
            "first f16 checkpoint retains stage, format, row offset, expert, and layer");
    std::vector<uint16_t> after(count);
    ck(cudaMemcpy(after.data(), device.values, count * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read f16 classifications");
    require(std::memcmp(after.data(), values.data(), count * sizeof(uint16_t)) == 0, "f16 scan preserves every input bit");
    std::puts("PASS device: f16 signed NaN/Inf counts, cross-block min, and first-format metadata");
}

void finite_bf16_edges_including_f16_infinity_pattern() {
    Stream stream;
    // 0x7c00 is +Inf as IEEE binary16 but finite as bfloat16 (exponent 248/255).
    const std::vector<uint16_t> finite = {0x0000u, 0x8000u, 0x0001u, 0x7f7fu, 0x7c00u};
    Device16 finite_device(finite.size());
    ck(cudaMemcpy(finite_device.values, finite.data(), finite.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload bf16 finite edges");
    fp::Probe finite_probe;
    std::string err;
    require(finite_probe.init(true, stream.value, 0, 80, (int64_t) finite.size(), err), err);
    require(finite_probe.scan16(finite_device.values, {0, 80, 45, 1, (int64_t) finite.size(), 0,
                                                       fp::Stage::hc_mixed_bf16}, fp::Format::bf16, err), err);
    require(finite_probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "bf16 finite boundary");
    finite_probe.report_after_sync(0, 80, 1);
    const fp::Record& finite_record = *finite_probe.completed_record();
    require(finite_record.ordinal == 0 && finite_record.nan == 0 && finite_record.positive_inf == 0 &&
            finite_record.negative_inf == 0, "bf16 0x7c00 and finite edge values");

    Stream bad_stream;
    constexpr size_t count = 257;
    std::vector<uint16_t> values(count, 0x3f80u);
    values[256] = 0x7f80u; // +Inf
    values[0] = 0xff80u;   // -Inf, minimum
    values[1] = 0x7fc1u;   // +NaN
    values[128] = 0xff81u; // -NaN
    Device16 device(count);
    ck(cudaMemcpy(device.values, values.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload bf16 classifications");
    fp::Probe probe;
    require(probe.init(true, bad_stream.value, 0, 81, (int64_t) count, err), err);
    fp::Checkpoint point{1, 81, 45, 1, (int64_t) count, 0, fp::Stage::hc_mixed_bf16_low};
    point.row_offset = 17;
    point.expert = 9;
    require(probe.scan16(device.values, point, fp::Format::bf16, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(bad_stream.value), "bf16 nonfinite boundary");
    probe.report_after_sync(1, 81, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 1 && r.index == 0 && r.positive_inf == 1 && r.negative_inf == 1 && r.nan == 2 &&
            r.point.format == fp::Format::bf16 && r.point.stage == fp::Stage::hc_mixed_bf16_low &&
            r.point.row_offset == 17 && r.point.expert == 9, "bf16 nonfinite signs, counts, and metadata");
    std::vector<uint16_t> after(count);
    ck(cudaMemcpy(after.data(), device.values, count * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read bf16 classifications");
    require(std::memcmp(after.data(), values.data(), count * sizeof(uint16_t)) == 0, "bf16 scan preserves every input bit");
    std::puts("PASS device: bf16 finite 0x7c00 distinction and signed NaN/Inf counts");
}

void fp32_finite_can_overflow_fp16() {
    const float source = 65520.0f;
    require(std::isfinite(source), "65520 is finite in FP32");
    const __half converted = __float2half_rn(source);
    const uint16_t half_bits = __half_as_ushort(converted);
    require(half_bits == 0x7c00u, "CUDA round-to-nearest FP32-to-FP16 conversion produces +Inf");

    Stream stream;
    Device float_device(1);
    ck(cudaMemcpy(float_device.values, &source, sizeof(source), cudaMemcpyHostToDevice), "upload FP32 source");
    Device16 half_device(1);
    ck(cudaMemcpy(half_device.values, &half_bits, sizeof(half_bits), cudaMemcpyHostToDevice), "upload converted FP16");
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 90, 1, err), err);
    require(probe.scan(float_device.values, {0, 90, 45, 1, 1, -1, fp::Stage::routed_activation_f32}, err), err);
    require(probe.scan16(half_device.values, {1, 90, 45, 1, 1, 0, fp::Stage::routed_activation_f16},
                        fp::Format::f16, err), err);
    require(probe.enqueue_read(err), err);
    ck(cudaStreamSynchronize(stream.value), "FP32/FP16 conversion boundary");
    probe.report_after_sync(1, 90, 1);
    const fp::Record& r = *probe.completed_record();
    require(r.ordinal == 2 && r.index == 0 && r.positive_inf == 1 && r.nan == 0 &&
            r.point.format == fp::Format::f16 && r.point.stage == fp::Stage::routed_activation_f16,
            "finite FP32 conversion overflow is detected only on the FP16 scan");
    std::puts("PASS device: finite FP32 65520 rounds to FP16 +Inf");
}

void moe_path_is_one_host_json_event() {
    Stream stream;
    fp::Probe probe;
    std::string err;
    require(probe.init(true, stream.value, 0, 100, 1, err), err);
    fp::Checkpoint point{4, 100, 45, 2, 8, 1, fp::Stage::routed_output};
    point.row_offset = 32;
    point.expert = 12;
    std::string captured;
    {
        StderrCapture capture;
        probe.report_moe_path(point, 512, true, false, false, 12, 7, false, false, false, 0);
        captured = capture.read();
    }
    require(captured.rfind("strata pf_nonfinite: {\"event\":\"moe_path\",\"schema\":1,", 0) == 0 &&
            captured.find("\n") == captured.size() - 1 && captured.find("\"experts\":512") != std::string::npos &&
            captured.find("\"cpu_rows\":0") != std::string::npos &&
            captured.find("\"event\":\"checkpoint\"") == std::string::npos,
            "MoE path report is one host JSON metadata event");
    std::puts("PASS host: MoE path emits one metadata-only event");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        disabled_case();
        detail_layer_cases();
        if (argc == 2 && std::strcmp(argv[1], "--disabled-only") == 0) return 0;
        require(argc == 1, "usage: prefill_finite_probe_test [--disabled-only]");
        ck(cudaSetDevice(0), "select device");
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (size_t count : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1025), size_t(1048577)}) {
                one_case(count, false, false, repeat);
                one_case(count, true, false, repeat);
                one_case(count, true, true, repeat);
            }
            finite_then_nan();
            dense_nan();
        }
        finite_f16_edges();
        f16_cross_block_first_record();
        finite_bf16_edges_including_f16_infinity_pattern();
        fp32_finite_can_overflow_fp16();
        moe_path_is_one_host_json_event();
        std::puts("PASS all synthetic CUDA finite probe cases; no model or performance claim");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
