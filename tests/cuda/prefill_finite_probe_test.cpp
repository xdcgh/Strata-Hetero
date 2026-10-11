// Synthetic device diagnostics only: no model, engine, GEMM, or timing claim.
#include "strata/prefill/finite_probe.hpp"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

void disabled_case() {
    require(!fp::requested(nullptr) && !fp::requested("") && !fp::requested("0") &&
            !fp::requested("true") && !fp::requested("10") && fp::requested("1"), "exact opt-in");
    fp::Probe probe;
    std::string err;
    require(probe.init(false, nullptr, 0, 0, 0, err), "disabled init");
    require(!probe.allocated() && probe.completed_record() == nullptr, "disabled owns no buffers");
    require(probe.scan(nullptr, {}, err) && probe.enqueue_read(err), "disabled accepts no-device calls");
    probe.report_after_sync(0, 0, 0);
    std::puts("PASS disabled: no probe CUDA calls (before any context creation)");
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
}  // namespace

int main(int argc, char** argv) {
    try {
        disabled_case();
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
        std::puts("PASS all synthetic CUDA finite probe cases; no model or performance claim");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
