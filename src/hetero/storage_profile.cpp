// CPU-only O_DIRECT / FILE_FLAG_NO_BUFFERING storage profiler. It never writes to the input file.
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#endif

namespace strata::hetero::storage {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kAlignment = strata::platform::DirectFile::alignment();
constexpr int kWaitSliceMs = 100;
constexpr uint64_t kDrainTimeoutMs = 30000;
constexpr size_t kLatencySampleCap = 65536;

struct Options {
    std::filesystem::path file;
    std::filesystem::path output;
    uint32_t block_bytes = 65536;
    uint32_t queue_depth = 8;
    uint64_t duration_ms = 1000;
    uint64_t seed = 1;
    std::string pattern = "random";
    uint64_t warmup_ms = 250;
    int repeats = 3;
    bool validate_only = false;
};

struct Preflight {
    uint64_t file_bytes = 0;
    uint64_t readable_bytes = 0;
    bool can_run = false;
};

struct Reader {
    virtual ~Reader() = default;
    virtual bool submit(uint64_t offset, void* buffer, uint32_t bytes, uint64_t tag, std::string& error) = 0;
    virtual int wait(strata::platform::Completion* out, int max, int timeout_ms) = 0;
};

struct DirectReader final : Reader {
    strata::platform::DirectFile file;
    bool open(const std::filesystem::path& path, std::string& error) {
        return file.open(path.string(), error);
    }
    uint64_t size() const { return file.size(); }
    bool submit(uint64_t offset, void* buffer, uint32_t bytes, uint64_t tag, std::string& error) override {
        return file.submit(offset, buffer, bytes, tag, error);
    }
    int wait(strata::platform::Completion* out, int max, int timeout_ms) override {
        return file.wait(out, max, timeout_ms);
    }
};

struct ClockSource {
    virtual ~ClockSource() = default;
    virtual double wall_us() = 0;
    virtual double cpu_us() = 0;
};

struct RealClock final : ClockSource {
    double wall_us() override {
        return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count() / 1000.0;
    }
    double cpu_us() override {
#if defined(_WIN32)
        FILETIME created{}, exited{}, kernel{}, user{};
        if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return 0.0;
        ULARGE_INTEGER k{}, u{};
        k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
        u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
        return (double) (k.QuadPart + u.QuadPart) / 10.0; // 100 ns to microseconds
#else
        timespec ts{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0.0;
        return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1000.0;
#endif
    }
};

struct PhaseStats {
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t successful = 0;
    uint64_t failed = 0;
    uint64_t bytes = 0;
    uint64_t uncompleted = 0;
    uint64_t latency_count_total = 0;
    double latency_sum_ms = 0;
    uint32_t max_qd = 0;
    double wall_ms = 0;
    double submit_cpu_ms = 0;
    double wait_cpu_ms = 0;
    double wait_wall_ms = 0;
    bool drain_timeout = false;
    std::string error;
    std::vector<double> latency_ms;
    std::vector<void*> retained_buffers; // non-empty only when pending I/O forbids safe cleanup
};

struct RepeatStats {
    PhaseStats warmup;
    PhaseStats measured;
};

struct RunStats {
    bool passed = true;
    std::vector<RepeatStats> repeats;
    std::string error;
    bool unsafe_pending_io = false;
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const size_t at = (size_t) std::ceil(p * (double) values.size()) - 1;
    return values[std::min(at, values.size() - 1)];
}

uint64_t bounded_random(std::mt19937_64& rng, uint64_t bound) {
    const uint64_t threshold = (uint64_t)(-bound) % bound;
    for (;;) {
        const uint64_t value = rng();
        if (value >= threshold) return value % bound;
    }
}

bool allowed_block(uint32_t bytes) {
    return bytes == 4096 || bytes == 16384 || bytes == 65536 || bytes == 1048576;
}

bool allowed_qd(uint32_t qd) {
    return qd == 1 || qd == 4 || qd == 8 || qd == 16 || qd == 32 || qd == 64;
}

bool preflight(const Options& o, Preflight& result, std::string& error) {
    error.clear();
    if (!allowed_block(o.block_bytes) || o.block_bytes % kAlignment != 0) {
        error = "block size must be one of 4096, 16384, 65536, 1048576 bytes"; return false;
    }
    if (!allowed_qd(o.queue_depth)) { error = "queue depth must be one of 1, 4, 8, 16, 32, 64"; return false; }
    if (o.duration_ms == 0 || o.warmup_ms > 3600000 || o.duration_ms > 3600000) {
        error = "duration must be 1..3600000 ms and warmup 0..3600000 ms"; return false;
    }
    if (o.repeats < 3 || o.repeats > 100) { error = "repeats must be 3..100"; return false; }
    if (o.pattern != "random" && o.pattern != "sequential") { error = "pattern must be random or sequential"; return false; }
    if (o.file.empty()) { result.can_run = false; return true; }
    std::error_code ec;
    if (!std::filesystem::exists(o.file, ec) || ec || !std::filesystem::is_regular_file(o.file, ec) || ec) {
        error = "input must be an existing regular file"; return false;
    }
    const uintmax_t file_bytes = std::filesystem::file_size(o.file, ec);
    if (ec || file_bytes < o.block_bytes || file_bytes > std::numeric_limits<uint64_t>::max()) {
        error = "input file must contain at least one full requested block"; return false;
    }
    result.file_bytes = (uint64_t) file_bytes;
    result.readable_bytes = result.file_bytes / o.block_bytes * o.block_bytes;
    result.can_run = true;
    if (!o.output.empty()) {
        if (std::filesystem::exists(o.output, ec)) { error = "output path already exists"; return false; }
        const std::filesystem::path output_dir = o.output.parent_path().empty()
            ? std::filesystem::path(".") : o.output.parent_path();
        if (ec || !std::filesystem::exists(output_dir, ec) || ec) {
            error = "output directory must already exist"; return false;
        }
    }
    return true;
}

uint64_t next_offset(const Options& o, uint64_t readable_bytes, uint64_t& sequential_index,
                     std::mt19937_64& rng) {
    const uint64_t blocks = readable_bytes / o.block_bytes;
    if (blocks == 0) return 0;
    uint64_t index;
    if (o.pattern == "sequential") index = sequential_index++ % blocks;
    else index = bounded_random(rng, blocks);
    return index * (uint64_t) o.block_bytes;
}

PhaseStats run_phase(Reader& reader, ClockSource& clock, const Options& o, uint64_t file_bytes,
                     uint64_t seed, uint64_t duration_ms, bool measure) {
    PhaseStats stats;
    struct Slot { void* buffer = nullptr; bool in_flight = false; uint64_t tag = 0; uint64_t offset = 0; double submitted_us = 0; };
    std::vector<Slot> slots(o.queue_depth);
    for (Slot& slot : slots) {
        slot.buffer = strata::platform::DirectFile::alloc_aligned(o.block_bytes);
        if (slot.buffer == nullptr) {
            stats.failed = 1; stats.error = "aligned buffer allocation failed";
            for (Slot& allocated : slots) strata::platform::DirectFile::free_aligned(allocated.buffer);
            return stats;
        }
    }
    std::unordered_map<uint64_t, size_t> by_tag;
    std::mt19937_64 rng(seed);
    std::mt19937_64 sample_rng(seed ^ 0xD1B54A32D192ED03ull);
    uint64_t sequential_index = 0, next_tag = 1;
    size_t next_slot = 0;
    uint64_t outstanding = 0;
    const double phase_start = clock.wall_us();
    const double submit_until = phase_start + (double) duration_ms * 1000.0;
    double drain_until = 0;
    bool drain_timer_started = false;
    double end_wall = phase_start;
    bool stop_submit = false;
    std::vector<strata::platform::Completion> completions(o.queue_depth);

    while (!stop_submit || outstanding != 0) {
        const double now = clock.wall_us();
        if (now >= submit_until && !stop_submit) {
            stop_submit = true;
            drain_until = now + (double) kDrainTimeoutMs * 1000.0;
            drain_timer_started = true;
        }
        while (!stop_submit && outstanding < o.queue_depth) {
            size_t tries = 0;
            while (tries < slots.size() && slots[next_slot].in_flight) { next_slot = (next_slot + 1) % slots.size(); ++tries; }
            if (tries == slots.size()) break;
            Slot& slot = slots[next_slot];
            const uint64_t offset = next_offset(o, file_bytes / o.block_bytes * o.block_bytes, sequential_index, rng);
            const double submitted_us = clock.wall_us();
            std::string submit_error;
            const double cpu_before = clock.cpu_us();
            const bool queued = reader.submit(offset, slot.buffer, o.block_bytes, next_tag, submit_error);
            stats.submit_cpu_ms += std::max(0.0, clock.cpu_us() - cpu_before) / 1000.0;
            if (!queued) {
                stats.failed++;
                stats.error = submit_error.empty() ? "read submission failed" : submit_error;
                stop_submit = true;
                if (!drain_timer_started) { drain_until = clock.wall_us() + (double) kDrainTimeoutMs * 1000.0; drain_timer_started = true; }
                break;
            }
            slot.in_flight = true; slot.tag = next_tag; slot.offset = offset; slot.submitted_us = submitted_us;
            by_tag[next_tag] = next_slot;
            ++stats.submitted; ++outstanding; ++next_tag;
            stats.max_qd = std::max(stats.max_qd, (uint32_t) outstanding);
            next_slot = (next_slot + 1) % slots.size();
            if (clock.wall_us() >= submit_until) stop_submit = true;
        }
        if (outstanding == 0) {
            end_wall = clock.wall_us();
            if (stop_submit) break;
            continue;
        }
        const double wait_wall_before = clock.wall_us();
        const double wait_cpu_before = clock.cpu_us();
        const int n = reader.wait(completions.data(), (int) completions.size(), kWaitSliceMs);
        const double completion_observed_us = clock.wall_us();
        stats.wait_cpu_ms += std::max(0.0, clock.cpu_us() - wait_cpu_before) / 1000.0;
        stats.wait_wall_ms += std::max(0.0, clock.wall_us() - wait_wall_before) / 1000.0;
        if (n < 0 || n > (int) completions.size()) {
            if (stats.error.empty()) { stats.error = "completion wait returned an invalid count"; ++stats.failed; }
            stop_submit = true;
        }
        for (int i = 0; i < std::max(0, std::min(n, (int) completions.size())); ++i) {
            const auto& c = completions[(size_t) i];
            const auto it = by_tag.find(c.tag);
            if (it == by_tag.end()) {
                if (stats.error.empty()) { stats.error = "completion tag was unknown"; ++stats.failed; }
                stop_submit = true; continue;
            }
            Slot& slot = slots[it->second];
            slot.in_flight = false;
            by_tag.erase(it);
            --outstanding; ++stats.completed;
            const double latency = std::max(0.0, completion_observed_us - slot.submitted_us) / 1000.0;
            const bool exact = c.ok && c.bytes == o.block_bytes;
            if (exact) {
                ++stats.successful;
                stats.bytes += c.bytes;
                if (measure) {
                    ++stats.latency_count_total;
                    stats.latency_sum_ms += latency;
                    if (stats.latency_ms.size() < kLatencySampleCap) stats.latency_ms.push_back(latency);
                    else {
                        const uint64_t sample_slot = bounded_random(sample_rng, stats.latency_count_total);
                        if (sample_slot < kLatencySampleCap) stats.latency_ms[(size_t)sample_slot] = latency;
                    }
                }
            } else {
                ++stats.failed;
                if (stats.error.empty()) stats.error = c.ok ? "short or zero-byte read; only full blocks are valid" : "direct read failed";
                stop_submit = true;
            }
        }
        if (stop_submit && !drain_timer_started) {
            drain_until = clock.wall_us() + (double) kDrainTimeoutMs * 1000.0;
            drain_timer_started = true;
        }
        end_wall = clock.wall_us();
        if (outstanding != 0 && drain_timer_started && end_wall >= drain_until) {
            stats.drain_timeout = true;
            if (stats.error.empty()) stats.error = "bounded drain timeout with I/O still pending";
            stats.wall_ms = std::max(0.0, end_wall - phase_start) / 1000.0;
            stats.uncompleted = outstanding;
            stats.failed += outstanding;
            for (const Slot& slot : slots) {
                if (slot.in_flight) stats.retained_buffers.push_back(slot.buffer);
                else strata::platform::DirectFile::free_aligned(slot.buffer);
            }
            return stats; // Caller must write the failed receipt then terminate without destroying the reader.
        }
    }
    stats.wall_ms = std::max(0.0, end_wall - phase_start) / 1000.0;
    for (Slot& slot : slots) strata::platform::DirectFile::free_aligned(slot.buffer);
    return stats;
}

RunStats run_profile(Reader& reader, ClockSource& clock, const Options& o, uint64_t file_bytes) {
    RunStats all;
    all.repeats.reserve((size_t) o.repeats);
    for (int repeat = 0; repeat < o.repeats; ++repeat) {
        RepeatStats one;
        const uint64_t seed = o.seed + (uint64_t) repeat;
        if (o.warmup_ms > 0) one.warmup = run_phase(reader, clock, o, file_bytes, seed, o.warmup_ms, false);
        if (!one.warmup.retained_buffers.empty()) {
            one.measured = one.warmup;
            all.unsafe_pending_io = true; all.passed = false;
            all.error = one.warmup.error;
            all.repeats.push_back(std::move(one));
            break;
        }
        if (one.warmup.failed != 0 || !one.warmup.error.empty()) {
            one.measured.error = "warmup failed: " + one.warmup.error;
            one.measured.failed = 1;
            all.passed = false;
            if (all.error.empty()) all.error = one.measured.error;
            all.repeats.push_back(std::move(one));
            continue;
        }
        one.measured = run_phase(reader, clock, o, file_bytes, seed, o.duration_ms, true);
        if (!one.measured.retained_buffers.empty()) {
            all.unsafe_pending_io = true; all.passed = false; all.error = one.measured.error;
        }
        if (one.measured.failed != 0 || one.measured.drain_timeout || !one.measured.error.empty()) {
            all.passed = false;
            if (all.error.empty()) all.error = one.measured.error;
        }
        all.repeats.push_back(std::move(one));
        if (all.unsafe_pending_io) break;
    }
    return all;
}

std::string quote_json(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
        case '\"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: if (c < 0x20) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; } else out += (char)c;
        }
    }
    return out + "\"";
}

std::string number_json(double value) {
    if (!std::isfinite(value)) return "null";
    char b[64]; std::snprintf(b, sizeof b, "%.9g", value); return b;
}

std::string phase_json(const PhaseStats& s, uint64_t requested_duration_ms, uint32_t requested_qd) {
    const bool eligible = s.failed == 0 && !s.drain_timeout && s.error.empty() && s.successful != 0;
    const bool metrics = eligible && !s.latency_ms.empty();
    const double seconds = s.wall_ms / 1000.0;
    const double iops = eligible && seconds > 0 ? (double) s.successful / seconds : std::numeric_limits<double>::quiet_NaN();
    const double mib_s = eligible && seconds > 0 ? (double) s.bytes / (1024.0 * 1024.0) / seconds : std::numeric_limits<double>::quiet_NaN();
    std::string out = "{\"status\":" + quote_json(s.failed == 0 && !s.drain_timeout && s.error.empty() ? "ok" : "failed") +
        ",\"requested_duration_ms\":" + std::to_string(requested_duration_ms) +
        ",\"requested_qd\":" + std::to_string(requested_qd) +
        ",\"actual_max_qd\":" + std::to_string(s.max_qd) +
        ",\"submitted\":" + std::to_string(s.submitted) +
        ",\"completed\":" + std::to_string(s.completed) +
        ",\"successful\":" + std::to_string(s.successful) +
        ",\"failed\":" + std::to_string(s.failed) +
        ",\"uncompleted\":" + std::to_string(s.uncompleted) +
        ",\"bytes\":" + std::to_string(s.bytes) +
        ",\"wall_ms\":" + number_json(s.wall_ms) +
        ",\"iops\":" + number_json(iops) +
        ",\"bandwidth_mib_s\":" + number_json(mib_s) +
        ",\"submit_to_completion_observed_ms\":{\"mean_all\":" +
        (eligible && s.latency_count_total ? number_json(s.latency_sum_ms / (double)s.latency_count_total) : "null") +
        ",\"p50_sampled\":" + (metrics ? number_json(percentile(s.latency_ms, .50)) : "null") +
        ",\"p95_sampled\":" + (metrics ? number_json(percentile(s.latency_ms, .95)) : "null") +
        ",\"p99_sampled\":" + (metrics ? number_json(percentile(s.latency_ms, .99)) : "null") +
        ",\"total_count\":" + std::to_string(s.latency_count_total) +
        ",\"sample_count\":" + std::to_string(s.latency_ms.size()) +
        ",\"sample_cap\":" + std::to_string(kLatencySampleCap) +
        ",\"meaning\":\"submit-call start to host observation after wait return; includes queueing and completion reaping, not device-only service time\"}" +
        ",\"calling_thread_cpu_ms\":{\"submit\":" + number_json(s.submit_cpu_ms) +
        ",\"wait\":" + number_json(s.wait_cpu_ms) +
        ",\"scope\":\"calling thread only; DirectFile worker-pool CPU is excluded\"}" +
        ",\"wait_wall_ms\":" + number_json(s.wait_wall_ms) +
        ",\"drain_timeout\":" + std::string(s.drain_timeout ? "true" : "false") +
        ",\"error\":" + (s.error.empty() ? "null" : quote_json(s.error)) + "}";
    return out;
}

bool write_exclusive(const std::filesystem::path& path, const std::string& bytes, std::string& error) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = GetLastError() == ERROR_FILE_EXISTS ? "output path already exists" : "cannot create output file"; return false; }
    size_t at = 0;
    bool ok = true;
    while (at < bytes.size()) {
        DWORD wrote = 0;
        const DWORD ask = (DWORD) std::min<size_t>(bytes.size() - at, 1u << 20);
        if (!WriteFile(h, bytes.data() + at, ask, &wrote, nullptr) || wrote == 0) { ok = false; break; }
        at += wrote;
    }
    if (!FlushFileBuffers(h)) ok = false;
    CloseHandle(h);
    if (!ok) error = "failed while writing output file";
    return ok;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { error = errno == EEXIST ? "output path already exists" : "cannot create output file"; return false; }
    size_t at = 0;
    bool ok = true;
    while (at < bytes.size()) {
        const ssize_t wrote = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (wrote <= 0) { ok = false; break; }
        at += (size_t) wrote;
    }
    if (::fsync(fd) != 0) ok = false;
    ::close(fd);
    if (!ok) error = "failed while writing output file";
    return ok;
#endif
}

std::string manifest_json(const Options& o, const Preflight& p, const RunStats& run) {
    std::string out = "{\n  \"schema_version\":1,\n  \"tool\":\"strata-storage-profile\",\n  \"cache_policy\":\"unbuffered\",\n  \"read_only\":true,\n  \"observer_only\":false,\n  \"input_file\":" + quote_json(o.file.string()) +
        ",\n  \"file_bytes\":" + std::to_string(p.file_bytes) +
        ",\n  \"readable_bytes\":" + std::to_string(p.readable_bytes) +
        ",\n  \"ignored_tail_bytes\":" + std::to_string(p.file_bytes - p.readable_bytes) +
        ",\n  \"block_bytes\":" + std::to_string(o.block_bytes) +
        ",\n  \"queue_depth\":" + std::to_string(o.queue_depth) +
        ",\n  \"duration_ms\":" + std::to_string(o.duration_ms) +
        ",\n  \"warmup_ms\":" + std::to_string(o.warmup_ms) +
        ",\n  \"repeats_requested\":" + std::to_string(o.repeats) +
        ",\n  \"seed\":" + std::to_string(o.seed) +
        ",\n  \"pattern\":" + quote_json(o.pattern) +
        ",\n  \"status\":" + quote_json(run.passed ? "ok" : "failed") +
        ",\n  \"model_test\":false,\n  \"error\":" + (run.error.empty() ? "null" : quote_json(run.error)) +
        ",\n  \"repeats\":[";
    for (size_t i = 0; i < run.repeats.size(); ++i) {
        const RepeatStats& r = run.repeats[i];
        if (i) out += ",";
        out += "{\"repeat\":" + std::to_string(i + 1) + ",\"warmup\":" +
            phase_json(r.warmup, o.warmup_ms, o.queue_depth) + ",\"measured\":" +
            phase_json(r.measured, o.duration_ms, o.queue_depth) + "}";
    }
    return out + "]\n}\n";
}

void usage() {
    std::fprintf(stderr, "Usage: strata-storage-profile [--file PATH] [--output NEW.json] [--validate-only] "
                         "[--block-bytes 4096|16384|65536|1048576] [--queue-depth 1|4|8|16|32|64] "
                         "[--duration-ms N] [--warmup-ms N] [--repeats N>=3] [--seed N] [--pattern random|sequential]\n");
}

bool parse_u64(const char* text, uint64_t& out) {
    if (text == nullptr || *text == '\0' || *text == '-') return false;
    const char* end = text + std::strlen(text);
    uint64_t parsed = 0;
    const auto result = std::from_chars(text, end, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != end) return false;
    out = parsed;
    return true;
}

bool parse_args(int argc, char** argv, Options& o, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        uint64_t n = 0;
        if (arg == "--file") { const char* v = value(); if (!v) { error = "--file requires a value"; return false; } o.file = v; }
        else if (arg == "--output") { const char* v = value(); if (!v) { error = "--output requires a value"; return false; } o.output = v; }
        else if (arg == "--block-bytes") { const char* v = value(); if (!parse_u64(v, n) || n > UINT32_MAX) { error = "invalid --block-bytes"; return false; } o.block_bytes = (uint32_t)n; }
        else if (arg == "--queue-depth") { const char* v = value(); if (!parse_u64(v, n) || n > UINT32_MAX) { error = "invalid --queue-depth"; return false; } o.queue_depth = (uint32_t)n; }
        else if (arg == "--duration-ms") { const char* v = value(); if (!parse_u64(v, o.duration_ms)) { error = "invalid --duration-ms"; return false; } }
        else if (arg == "--warmup-ms") { const char* v = value(); if (!parse_u64(v, o.warmup_ms)) { error = "invalid --warmup-ms"; return false; } }
        else if (arg == "--seed") { const char* v = value(); if (!parse_u64(v, o.seed)) { error = "invalid --seed"; return false; } }
        else if (arg == "--repeats") { const char* v = value(); if (!parse_u64(v, n) || n > INT32_MAX) { error = "invalid --repeats"; return false; } o.repeats = (int)n; }
        else if (arg == "--pattern") { const char* v = value(); if (!v) { error = "--pattern requires a value"; return false; } o.pattern = v; }
        else if (arg == "--validate-only") o.validate_only = true;
        else { error = "unknown argument: " + arg; return false; }
    }
    return true;
}

int run_cli(int argc, char** argv, const std::function<DirectReader*()>& reader_factory = [] { return new DirectReader(); }) {
    Options o;
    std::string error;
    if (!parse_args(argc, argv, o, error)) { usage(); std::fprintf(stderr, "storage profile: %s\n", error.c_str()); return 2; }
    Preflight p;
    if (!preflight(o, p, error)) { std::fprintf(stderr, "storage profile: %s\n", error.c_str()); return 2; }
    // Absence of --file and --validate-only are both deliberate no-I/O preflight modes.
    if (o.validate_only || o.file.empty()) {
        std::printf("{\"status\":\"preflight_only\",\"read_started\":false,\"output_written\":false,\"file_bytes\":%llu}\n",
                    (unsigned long long) p.file_bytes);
        return 0;
    }
    if (o.output.empty()) { std::fprintf(stderr, "storage profile: --output is required for a measurement\n"); return 2; }

    DirectReader* reader = reader_factory(); // kept alive if a kernel read outlives bounded drain
    if (reader == nullptr) { std::fprintf(stderr, "storage profile: reader allocation failed\n"); return 1; }
    if (!reader->open(o.file, error)) {
        RunStats failed; failed.passed = false; failed.error = error;
        const std::string json = manifest_json(o, p, failed);
        std::string write_error;
        if (!write_exclusive(o.output, json, write_error)) { std::fprintf(stderr, "storage profile: %s\n", write_error.c_str()); delete reader; return 1; }
        std::fprintf(stderr, "storage profile: failed: %s\n", error.c_str()); delete reader; return 1;
    }
    if (reader->size() != p.file_bytes) {
        RunStats failed; failed.passed = false; failed.error = "input file size changed after preflight";
        const std::string json = manifest_json(o, p, failed);
        std::string write_error;
        if (!write_exclusive(o.output, json, write_error)) { std::fprintf(stderr, "storage profile: %s\n", write_error.c_str()); delete reader; return 1; }
        delete reader; return 1;
    }
    RealClock clock;
    RunStats stats = run_profile(*reader, clock, o, p.file_bytes);
    const std::string json = manifest_json(o, p, stats);
    std::string write_error;
    if (!write_exclusive(o.output, json, write_error)) {
        std::fprintf(stderr, "storage profile: %s\n", write_error.c_str());
        if (stats.unsafe_pending_io) std::_Exit(4);
        delete reader; return 1;
    }
    std::printf("{\"status\":\"%s\",\"output_written\":true,\"repeats\":%zu}\n",
                stats.passed ? "ok" : "failed", stats.repeats.size());
    if (stats.unsafe_pending_io) {
        // DirectFile exposes no cancel-and-drain operation. Keep its object and any pending buffers alive until OS
        // process teardown; destructors here could invalidate outstanding Windows OVERLAPPED buffers.
        std::fflush(nullptr);
        std::_Exit(4); // dedicated exit: outstanding OS I/O still owns read buffers
    }
    delete reader;
    return stats.passed ? 0 : 1;
}

} // namespace
} // namespace strata::hetero::storage

#ifndef STRATA_STORAGE_PROFILE_TEST
int main(int argc, char** argv) { return strata::hetero::storage::run_cli(argc, argv); }
#endif
