#include "strata/prefill/finite_probe.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>

namespace strata::prefill::finite_probe {
namespace {
std::atomic<unsigned long long> next_run{0};

__global__ void scan_kernel(const float* values, unsigned long long count, Record* record,
                            Checkpoint point, unsigned ordinal) {
    // All probes and their producers are queued on the same stream. A later
    // checkpoint cannot claim the record until this kernel has completed.
    const unsigned first = atomicAdd(&record->ordinal, 0u);
    if (first != 0 && first != ordinal) return;
    for (unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long) gridDim.x * blockDim.x) {
        // Inspect IEEE-754 bits, independent of CUDA fast-math assumptions.
        const unsigned bits = __float_as_uint(values[i]);
        const unsigned magnitude = bits & 0x7fffffffu;
        if (magnitude < 0x7f800000u) continue;
        const unsigned previous = atomicCAS(&record->ordinal, 0u, ordinal);
        if (previous == 0) record->point = point;
        if (previous == 0 || previous == ordinal) {
            atomicMin(&record->index, i);
            if (magnitude > 0x7f800000u) atomicAdd(&record->nan, 1ull);
            else if (bits >> 31) atomicAdd(&record->negative_inf, 1ull);
            else atomicAdd(&record->positive_inf, 1ull);
        }
    }
}
}  // namespace

bool Probe::check(int status, const char* operation, std::string& err) {
    if (status == (int) cudaSuccess) return true;
    err = std::string("prefill finite probe: ") + operation + ": " + cudaGetErrorString((cudaError_t) status);
    return false;
}

Probe::~Probe() {
    if (device_record_ || host_record_) {
        // On an early return a queued scan/copy may still own these buffers.
        // On the normal path the caller already drained this stream.
        cudaStreamSynchronize((cudaStream_t) stream_);
        if (device_record_) cudaFree(device_record_);
        if (host_record_) cudaFreeHost(host_record_);
    }
}

bool Probe::init(bool enabled, void* stream, int device, int64_t pos0, int64_t tokens, std::string& err) {
    if (!enabled) return true;
    if (device_record_ || host_record_) { err = "prefill finite probe: already initialized"; return false; }
    stream_ = stream;
    device_ = device;
    run_ = ++next_run;
    if (!check((int) cudaMalloc(&device_record_, sizeof(Record)), "device allocation", err) ||
        !check((int) cudaMallocHost(&host_record_, sizeof(Record)), "pinned record allocation", err) ||
        !check((int) cudaMemsetAsync(device_record_, 0, sizeof(Record), (cudaStream_t) stream_), "clear", err) ||
        !check((int) cudaMemsetAsync(&device_record_->index, 0xff, sizeof(device_record_->index),
                                    (cudaStream_t) stream_), "clear index", err)) return false;
    std::fprintf(stderr, "strata pf_nonfinite: {\"event\":\"begin\",\"schema\":1,\"run\":%llu,\"device\":%d,"
                         "\"pos0\":%lld,\"tokens\":%lld,\"diagnostic_only\":true}\n",
                 run_, device_, (long long) pos0, (long long) tokens);
    return true;
}

bool Probe::scan(const float* values, const Checkpoint& point, std::string& err) {
    if (!device_record_) return true;
    if (!values || point.rows <= 0 || point.width <= 0 ||
        (unsigned long long) point.rows > std::numeric_limits<unsigned long long>::max() /
                                          (unsigned long long) point.width ||
        ordinal_ == std::numeric_limits<unsigned>::max()) {
        err = "prefill finite probe: invalid checkpoint or ordinal overflow";
        return false;
    }
    const unsigned long long count = (unsigned long long) point.rows * (unsigned long long) point.width;
    const int blocks = (int) std::min<unsigned long long>(4096, 1 + (count - 1) / 256);
    scan_kernel<<<blocks, 256, 0, (cudaStream_t) stream_>>>(values, count, device_record_, point, ++ordinal_);
    return check((int) cudaGetLastError(), "scan launch (may also surface a producer error)", err);
}

bool Probe::enqueue_read(std::string& err) {
    if (!device_record_) return true;
    if (read_pending_) { err = "prefill finite probe: unread checkpoint"; return false; }
    if (!check((int) cudaMemcpyAsync(host_record_, device_record_, sizeof(Record), cudaMemcpyDeviceToHost,
                                    (cudaStream_t) stream_), "queue record read", err)) return false;
    read_pending_ = true;
    return true;
}

void Probe::report_after_sync(int64_t chunk, int64_t pos0, int64_t rows) {
    if (!read_pending_) return;
    read_pending_ = false;
    const Record& r = *host_record_;
    std::fprintf(stderr, "strata pf_nonfinite: {\"event\":\"checkpoint\",\"schema\":1,\"run\":%llu,\"device\":%d,"
                         "\"completed_chunk\":%lld,\"completed_pos0\":%lld,\"completed_rows\":%lld,\"scans\":%u,"
                         "\"first_bad\":", run_, device_, (long long) chunk, (long long) pos0, (long long) rows, ordinal_);
    if (r.ordinal == 0) std::fprintf(stderr, "null");
    else std::fprintf(stderr, "{\"ordinal\":%u,\"stage\":\"%s\",\"chunk\":%lld,\"pos0\":%lld,\"layer\":%lld,"
                             "\"half\":%d,\"rows\":%lld,\"width\":%lld,\"index\":%llu,\"row\":%llu,\"column\":%llu,"
                             "\"nan\":%llu,\"positive_inf\":%llu,\"negative_inf\":%llu}",
                      r.ordinal, stage_name(r.point.stage), (long long) r.point.chunk, (long long) r.point.pos0,
                      (long long) r.point.layer, r.point.half, (long long) r.point.rows, (long long) r.point.width,
                      r.index, r.index / (unsigned long long) r.point.width, r.index % (unsigned long long) r.point.width,
                      r.nan, r.positive_inf, r.negative_inf);
    std::fprintf(stderr, "}\n");
    std::fflush(stderr);
}

}  // namespace strata::prefill::finite_probe
