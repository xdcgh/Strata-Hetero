#include "strata/prefill/finite_probe.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>

namespace strata::prefill::finite_probe {
namespace {
std::atomic<unsigned long long> next_run{0};

template<Format format>
__global__ void scan_kernel(const void* values, unsigned long long count, Record* record,
                            Checkpoint point, unsigned ordinal) {
    // All probes and their producers are queued on the same stream. A later
    // checkpoint cannot claim the record until this kernel has completed.
    const unsigned first = atomicAdd(&record->ordinal, 0u);
    if (first != 0 && first != ordinal) return;
    for (unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long) gridDim.x * blockDim.x) {
        // Inspect IEEE-754 bits, independent of CUDA fast-math assumptions.
        unsigned magnitude, infinity, negative;
        if constexpr (format == Format::f32) {
            const unsigned bits = __float_as_uint(((const float*) values)[i]);
            magnitude = bits & 0x7fffffffu; infinity = 0x7f800000u; negative = bits >> 31;
        } else {
            const unsigned bits = ((const uint16_t*) values)[i];
            magnitude = bits & 0x7fffu; negative = bits >> 15;
            infinity = format == Format::f16 ? 0x7c00u : 0x7f80u;
        }
        if (magnitude < infinity) continue;
        const unsigned previous = atomicCAS(&record->ordinal, 0u, ordinal);
        if (previous == 0) record->point = point;
        if (previous == 0 || previous == ordinal) {
            atomicMin(&record->index, i);
            if (magnitude > infinity) atomicAdd(&record->nan, 1ull);
            else if (negative) atomicAdd(&record->negative_inf, 1ull);
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
    return scan_impl(values, point, Format::f32, err);
}

bool Probe::scan16(const uint16_t* values, const Checkpoint& point, Format format, std::string& err) {
    if (!device_record_) return true;
    if (format != Format::f16 && format != Format::bf16) {
        err = "prefill finite probe: 16-bit scan requires f16 or bf16";
        return false;
    }
    return scan_impl(values, point, format, err);
}

bool Probe::scan_impl(const void* values, Checkpoint point, Format format, std::string& err) {
    if (!device_record_) return true;
    if (!values || point.rows <= 0 || point.width <= 0 ||
        (unsigned long long) point.rows > std::numeric_limits<unsigned long long>::max() /
                                          (unsigned long long) point.width ||
        point.row_offset < 0 || ordinal_ == std::numeric_limits<unsigned>::max()) {
        err = "prefill finite probe: invalid checkpoint or ordinal overflow";
        return false;
    }
    const unsigned long long count = (unsigned long long) point.rows * (unsigned long long) point.width;
    const int blocks = (int) std::min<unsigned long long>(4096, 1 + (count - 1) / 256);
    point.format = format;
    const unsigned ordinal = ++ordinal_;
    if (format == Format::f32)
        scan_kernel<Format::f32><<<blocks, 256, 0, (cudaStream_t) stream_>>>(values, count, device_record_, point, ordinal);
    else if (format == Format::f16)
        scan_kernel<Format::f16><<<blocks, 256, 0, (cudaStream_t) stream_>>>(values, count, device_record_, point, ordinal);
    else
        scan_kernel<Format::bf16><<<blocks, 256, 0, (cudaStream_t) stream_>>>(values, count, device_record_, point, ordinal);
    return check((int) cudaGetLastError(), "scan launch (may also surface a producer error)", err);
}

void Probe::report_moe_path(const Checkpoint& point, int experts, bool native, bool mmq, bool fused,
                            int gu_type, int down_type, bool stream_all, bool peer,
                            bool peer_sums, int64_t cpu_rows) {
    if (!device_record_) return;
    std::fprintf(stderr, "strata pf_nonfinite: {\"event\":\"moe_path\",\"schema\":1,\"run\":%llu,\"device\":%d,"
                         "\"chunk\":%lld,\"pos0\":%lld,\"layer\":%lld,\"rows\":%lld,\"experts\":%d,"
                         "\"native\":%d,\"mmq\":%d,\"fused\":%d,\"gu_type\":%d,\"down_type\":%d,"
                         "\"stream_all\":%d,\"peer\":%d,\"peer_sums\":%d,\"cpu_rows\":%lld}\n",
                 run_, device_, (long long) point.chunk, (long long) point.pos0, (long long) point.layer,
                 (long long) point.rows, experts, (int) native, (int) mmq, (int) fused, gu_type, down_type,
                 (int) stream_all, (int) peer, (int) peer_sums, (long long) cpu_rows);
    std::fflush(stderr);
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
    char bad[1024] = "null";
    if (r.ordinal != 0) std::snprintf(bad, sizeof bad, "{\"ordinal\":%u,\"stage\":\"%s\",\"chunk\":%lld,\"pos0\":%lld,\"layer\":%lld,"
                             "\"half\":%d,\"rows\":%lld,\"width\":%lld,\"index\":%llu,\"row\":%llu,\"column\":%llu,"
                             "\"nan\":%llu,\"positive_inf\":%llu,\"negative_inf\":%llu,\"format\":\"%s\","
                             "\"row_offset\":%lld,\"source_row\":%llu,\"expert\":%d}",
                      r.ordinal, stage_name(r.point.stage), (long long) r.point.chunk, (long long) r.point.pos0,
                      (long long) r.point.layer, r.point.half, (long long) r.point.rows, (long long) r.point.width,
                      r.index, r.index / (unsigned long long) r.point.width, r.index % (unsigned long long) r.point.width,
                      r.nan, r.positive_inf, r.negative_inf, format_name(r.point.format),
                      (long long) r.point.row_offset, (unsigned long long) r.point.row_offset + r.index / (unsigned long long) r.point.width,
                      r.point.expert);
    // One stdio call per JSON line avoids interleaving a partial report with
    // another engine thread's progress messages. Readers still frame by LF.
    std::fprintf(stderr, "strata pf_nonfinite: {\"event\":\"checkpoint\",\"schema\":1,\"run\":%llu,\"device\":%d,"
                         "\"completed_chunk\":%lld,\"completed_pos0\":%lld,\"completed_rows\":%lld,\"scans\":%u,"
                         "\"first_bad\":%s}\n", run_, device_, (long long) chunk, (long long) pos0, (long long) rows, ordinal_, bad);
    std::fflush(stderr);
}

}  // namespace strata::prefill::finite_probe
