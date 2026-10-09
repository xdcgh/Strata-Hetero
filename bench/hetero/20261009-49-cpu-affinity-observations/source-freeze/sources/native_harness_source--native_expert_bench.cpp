// Explicit, single-process CPU reference for one native GGML expert.
// The default path validates metadata only; kernels run only after an explicit --run.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <cstdlib>
#define NOMINMAX
#include <windows.h>
#else
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef HETERO_STRATA_SOURCE_HEAD
#define HETERO_STRATA_SOURCE_HEAD "unknown"
#endif
#ifndef HETERO_GGML_PIN
#define HETERO_GGML_PIN "3cf03257f219afbe7334045ff7c6a06ac68c627d"
#endif
#ifndef HETERO_NATIVE_SOURCE_SHA256
#define HETERO_NATIVE_SOURCE_SHA256 "unknown"
#endif
#ifndef HETERO_NATIVE_SHA_HEADER_SHA256
#define HETERO_NATIVE_SHA_HEADER_SHA256 "unknown"
#endif
#ifndef HETERO_NATIVE_CMAKE_SHA256
#define HETERO_NATIVE_CMAKE_SHA256 "unknown"
#endif

namespace fs = std::filesystem;
namespace cpu = strata::kernels::cpu;
using strata::kernels::cpu::NativeFmt;

namespace {

constexpr uint64_t kMaxRows = 256;
constexpr uint64_t kMaxWidth = 65536;
constexpr uint64_t kMaxIoBytes = 256ull * 1024 * 1024;
constexpr uint64_t kMaxIterations = 1000;

struct Args {
    bool run = false;
    bool help = false;
    bool validate_only = false;
    bool canonical_q8k = false;
    int rows = 1;                 // tokens / activation rows
    int hidden = 2560;            // n_embd
    int intermediate = 640;       // n_ff
    int gu_type = 12;             // GGML_TYPE_Q4_K
    int down_type = 7;            // GGML_TYPE_Q5_1
    uint64_t warmup = 1;
    uint64_t repeat = 3;
    int pool_workers = 0;           // 0 keeps the original single-thread path
    std::string pool_affinity = "none";
    bool pool_workers_set = false, pool_affinity_set = false;
    bool has_rows = false, has_hidden = false, has_intermediate = false;
    bool has_gu_type = false, has_down_type = false;
    fs::path blob, input, output, receipt;
};

template <typename T>
class AlignedArray {
public:
    explicit AlignedArray(size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("invalid aligned buffer size");
        }
        data_ = static_cast<T*>(::operator new[](count_ * sizeof(T), std::align_val_t{64}));
        std::fill_n(data_, count_, T{});
    }
    ~AlignedArray() { if (data_) ::operator delete[](data_, std::align_val_t{64}); }
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;
    AlignedArray(AlignedArray&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr; other.count_ = 0;
    }
    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            if (data_) ::operator delete[](data_, std::align_val_t{64});
            data_ = other.data_; count_ = other.count_;
            other.data_ = nullptr; other.count_ = 0;
        }
        return *this;
    }
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }
private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

uint64_t parse_u64(std::string_view text, const char* option) {
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error(std::string("invalid integer for ") + option);
    }
    return value;
}

int parse_int(std::string_view text, const char* option) {
    const uint64_t value = parse_u64(text, option);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("integer out of range for ") + option);
    }
    return static_cast<int>(value);
}

Args parse_args(int argc, char** argv) {
    Args args;
    std::set<std::string> seen;
    auto value_after = [&](int& i, const char* option) -> std::string_view {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + option);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        const std::string key = (opt == "--receipt-json") ? "--receiptJSON" : opt;
        if (!seen.insert(key).second) throw std::runtime_error("option supplied more than once: " + opt);
        if (opt == "--run") args.run = true;
        else if (opt == "--validate-only") args.validate_only = true;
        else if (opt == "--canonical-q8k") args.canonical_q8k = true;
        else if (opt == "--help" || opt == "-h") args.help = true;
        else if (opt == "--rows") { args.rows = parse_int(value_after(i, "--rows"), "--rows"); args.has_rows = true; }
        else if (opt == "--hidden") { args.hidden = parse_int(value_after(i, "--hidden"), "--hidden"); args.has_hidden = true; }
        else if (opt == "--intermediate") { args.intermediate = parse_int(value_after(i, "--intermediate"), "--intermediate"); args.has_intermediate = true; }
        else if (opt == "--gu-type") { args.gu_type = parse_int(value_after(i, "--gu-type"), "--gu-type"); args.has_gu_type = true; }
        else if (opt == "--down-type") { args.down_type = parse_int(value_after(i, "--down-type"), "--down-type"); args.has_down_type = true; }
        else if (opt == "--warmup") args.warmup = parse_u64(value_after(i, "--warmup"), "--warmup");
        else if (opt == "--repeat") args.repeat = parse_u64(value_after(i, "--repeat"), "--repeat");
        else if (opt == "--pool-workers") { args.pool_workers = parse_int(value_after(i, "--pool-workers"), "--pool-workers"); args.pool_workers_set = true; }
        else if (opt == "--pool-affinity") { args.pool_affinity = std::string(value_after(i, "--pool-affinity")); args.pool_affinity_set = true; }
        else if (opt == "--blob") args.blob = fs::u8path(value_after(i, "--blob"));
        else if (opt == "--input") args.input = fs::u8path(value_after(i, "--input"));
        else if (opt == "--output") args.output = fs::u8path(value_after(i, "--output"));
        else if (opt == "--receiptJSON" || opt == "--receipt-json") args.receipt = fs::u8path(value_after(i, opt.c_str()));
        else throw std::runtime_error("unknown option: " + opt);
    }
    if ((args.pool_workers_set && (args.pool_workers < 1 || args.pool_workers > 64)) ||
        (args.pool_affinity_set && !args.pool_workers_set) ||
        (args.pool_affinity != "none" && args.pool_affinity != "all" && args.pool_affinity != "auto" && args.pool_affinity != "p-cores"))
        throw std::runtime_error("pool mode requires --pool-workers 1..64 and affinity none|all|auto|p-cores");
    return args;
}

void print_help() {
    std::cout <<
        "hetero_native_expert [metadata options]\n"
        "  Default: validate-only metadata; calls native_fmt only, reads/writes no files, runs no kernel.\n"
        "  --validate-only explicitly selects the same metadata-only mode.\n"
        "  --rows N --hidden H --intermediate I --gu-type ID --down-type ID\n"
        "  Explicit run:\n"
        "    --run --blob RAW_QUANTIZED_BLOB --input F32LE_INPUT --rows N --hidden H\n"
        "    --intermediate I --gu-type ID --down-type ID --output F32LE_OUTPUT\n"
        "    --receiptJSON RECEIPT.json [--warmup N] [--repeat N] [--canonical-q8k]\n"
        "    [--pool-workers 1..64] [--pool-affinity none|all|auto|p-cores]\n"
        "  Default --run keeps the model Q8_K activation quantizer path; --canonical-q8k selects ggml's Q8_K from_float path.\n"
        "  Input is token-major raw little-endian float32 [rows, hidden]. Output has the same shape.\n"
        "  Native blob layout follows native_fmt: [gate rows][up rows][down rows].\n"
        "  Rows: 1..256; hidden/intermediate: 1..65536; warmup >=1; repeat >=3.\n";
}

uint64_t checked_mul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string("size overflow computing ") + what);
    }
    return a * b;
}

void validate_dimensions(const Args& args) {
    if (args.rows < 1 || static_cast<uint64_t>(args.rows) > kMaxRows) throw std::runtime_error("--rows must be in 1..256");
    if (args.hidden < 1 || static_cast<uint64_t>(args.hidden) > kMaxWidth) throw std::runtime_error("--hidden must be in 1..65536");
    if (args.intermediate < 1 || static_cast<uint64_t>(args.intermediate) > kMaxWidth) throw std::runtime_error("--intermediate must be in 1..65536");
    if (args.warmup < 1 || args.warmup > kMaxIterations) throw std::runtime_error("--warmup must be in 1..1000");
    if (args.repeat < 3 || args.repeat > kMaxIterations) throw std::runtime_error("--repeat must be in 3..1000");
}

void set_process_reference_environment(bool canonical_q8k) {
    // Keep the KQ row kernels off, while matching the model's Q8_K activation fast-path by default.
#if defined(_WIN32)
    if (_putenv_s("STRATA_KQ256", "0") != 0 ||
        _putenv_s("STRATA_NO_Q8K_AVX2", canonical_q8k ? "1" : "") != 0) {
        throw std::runtime_error("could not set process-local reference environment");
    }
#else
    if (setenv("STRATA_KQ256", "0", 1) != 0 ||
        (canonical_q8k ? setenv("STRATA_NO_Q8K_AVX2", "1", 1) : unsetenv("STRATA_NO_Q8K_AVX2")) != 0) {
        throw std::runtime_error("could not set process-local reference environment");
    }
#endif
}

std::string json_quote(std::string_view s) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) { out += "\\u00"; out.push_back(hex[c >> 4]); out.push_back(hex[c & 0xf]); }
            else out.push_back(static_cast<char>(c));
        }
    }
    out.push_back('"');
    return out;
}

std::string path_string(const fs::path& p) {
    const auto u8 = p.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    if (gmtime_s(&tm, &t) != 0) throw std::runtime_error("UTC timestamp conversion failed");
#else
    if (gmtime_r(&t, &tm) == nullptr) throw std::runtime_error("UTC timestamp conversion failed");
#endif
    char buf[32]{};
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) throw std::runtime_error("UTC timestamp formatting failed");
    return buf;
}

std::string normalized_path_key(const fs::path& path) {
    std::error_code ec;
    fs::path resolved = fs::weakly_canonical(path, ec);
    if (ec) resolved = fs::absolute(path).lexically_normal();
    std::string key = path_string(resolved);
#if defined(_WIN32)
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return key;
}

uint64_t exact_file_size(const fs::path& path, uint64_t expected, const char* label) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) throw std::runtime_error(std::string(label) + " is not a regular file: " + path_string(path));
    const uint64_t actual = fs::file_size(path, ec);
    if (ec) throw std::runtime_error(std::string("cannot stat ") + label + ": " + path_string(path));
    if (actual != expected) {
        throw std::runtime_error(std::string(label) + " size mismatch: expected " + std::to_string(expected) +
                                 " bytes, got " + std::to_string(actual));
    }
    return actual;
}

template <typename T>
void read_exact_file(const fs::path& path, AlignedArray<T>& buffer, const char* label) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + label + ": " + path_string(path));
    const uint64_t byte_count = checked_mul(buffer.size(), sizeof(T), label);
    if (byte_count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) throw std::runtime_error("file exceeds stream read limit");
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(byte_count));
    if (in.gcount() != static_cast<std::streamsize>(byte_count)) throw std::runtime_error(std::string("short read for ") + label);
    char extra = 0;
    if (in.read(&extra, 1)) throw std::runtime_error(std::string(label) + " grew after size validation");
    if (!in.eof() && in.fail()) throw std::runtime_error(std::string("read error for ") + label);
}

std::string sha256(const void* data, size_t size) {
    return hetero_native_cpu::sha256_hex(data, size);
}

std::string utc_path(const fs::path& p) {
    return json_quote(path_string(fs::absolute(p).lexically_normal()));
}

NativeFmt make_format(const Args& args) {
    validate_dimensions(args);
    if (args.gu_type < 0 || args.gu_type >= GGML_TYPE_COUNT || args.down_type < 0 || args.down_type >= GGML_TYPE_COUNT) {
        throw std::runtime_error("GGML type ID is outside the pinned GGML_TYPE_COUNT range");
    }
    NativeFmt fmt{};
    std::string error;
    if (!strata::kernels::cpu::native_fmt(args.gu_type, args.down_type, args.hidden, args.intermediate, fmt, error)) {
        throw std::runtime_error("unsupported native format: " + error);
    }
    if (args.pool_workers > 0 && (fmt.gu_type != 12 || fmt.d_type != 7))
        throw std::runtime_error("pool mode is restricted to the prepared Q4_K/Q5_1 native expert (types 12/7)");
    if (fmt.bytes == 0 || fmt.bytes > kMaxIoBytes || fmt.act_bytes == 0 || fmt.h_bytes == 0) {
        throw std::runtime_error("native format exceeds bounded harness buffers");
    }
    return fmt;
}

std::string metadata_json(const Args& args, const NativeFmt& fmt) {
    std::ostringstream s;
    s << "{\n"
      << "  \"schema\": \"strata-hetero-native-cpu-metadata-v1\",\n"
      << "  \"mode\": \"validate_only\",\n"
      << "  \"native_fmt_called\": true,\n"
      << "  \"native_kernel_called\": false,\n"
      << "  \"output_written\": false,\n"
      << "  \"strata_head\": " << json_quote(HETERO_STRATA_SOURCE_HEAD) << ",\n"
      << "  \"ggml_pin\": " << json_quote(HETERO_GGML_PIN) << ",\n"
      << "  \"native_harness_source_sha256\": " << json_quote(HETERO_NATIVE_SOURCE_SHA256) << ",\n"
      << "  \"sha256_header_sha256\": " << json_quote(HETERO_NATIVE_SHA_HEADER_SHA256) << ",\n"
      << "  \"harness_cmake_sha256\": " << json_quote(HETERO_NATIVE_CMAKE_SHA256) << ",\n"
      << "  \"rows\": " << args.rows << ", \"hidden\": " << args.hidden << ", \"intermediate\": " << args.intermediate << ",\n"
      << "  \"gu_type\": " << fmt.gu_type << ", \"down_type\": " << fmt.d_type << ",\n"
      << "  \"gu_activation_type\": " << fmt.gu_act << ", \"down_activation_type\": " << fmt.d_act << ",\n"
      << "  \"gu_row_bytes\": " << fmt.gu_row << ", \"down_row_bytes\": " << fmt.d_row << ",\n"
      << "  \"up_offset_bytes\": " << fmt.up_off << ", \"down_offset_bytes\": " << fmt.down_off << ",\n"
      << "  \"blob_bytes\": " << fmt.bytes << ", \"act_bytes_per_row\": " << fmt.act_bytes
      << ", \"hidden_bytes_per_row\": " << fmt.h_bytes;
    if (args.pool_workers > 0)
        s << ",\n  \"pool\": {\"requested_workers\": " << args.pool_workers
          << ", \"affinity\": " << json_quote(args.pool_affinity)
          << ", \"kernel_started\": false, \"topology_queried\": false}";
    s << "\n}\n";
    return s.str();
}

struct RunContext {
    bool receipt_path_safe = false;
    bool output_write_started = false;
    bool output_write_complete = false;
    bool receipt_write_complete = false;
    std::string phase = "argument_validation";
};

void validate_no_path_collisions(const Args& args, RunContext& context) {
    const std::string blob = normalized_path_key(args.blob);
    const std::string input = normalized_path_key(args.input);
    const std::string output = normalized_path_key(args.output);
    const std::string receipt = normalized_path_key(args.receipt);
    if (blob == input || blob == output || blob == receipt || input == output || input == receipt || output == receipt) {
        throw std::runtime_error("blob, input, output, and receipt paths must all be distinct");
    }
    if (fs::exists(args.receipt)) throw std::runtime_error("receipt already exists; refusing to overwrite");
    const fs::path out_parent = fs::absolute(args.output).parent_path();
    const fs::path receipt_parent = fs::absolute(args.receipt).parent_path();
    if (!fs::is_directory(out_parent) || !fs::is_directory(receipt_parent)) {
        throw std::runtime_error("output and receipt parent directories must already exist");
    }
    context.receipt_path_safe = true;
    if (fs::exists(args.output)) throw std::runtime_error("output already exists; refusing to overwrite");
}

bool all_finite(const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(data[i])) return false;
    return true;
}

struct WorkBuffers {
    std::vector<AlignedArray<float>> input;
    std::vector<AlignedArray<uint8_t>> act, hidden_q;
    std::vector<AlignedArray<float>> ff, output;
    std::vector<const void*> act_ptrs, hidden_q_ptrs;
    std::vector<float*> ff_ptrs, output_ptrs;
};

struct PoolRunInfo {
    cpu::PoolAffinity affinity = cpu::PoolAffinity::All;
    cpu::CpuTopology topology;
    int workers = 0, actual_workers = 0, unpinned_overflow = 0;
    bool pin = false, host_pinned = false;
    std::vector<cpu::WorkerAffinityReport> worker_affinity;
    std::vector<uint64_t> activation_ns, pool_ns, gu_ns, hidden_q_ns, down_ns;
    std::vector<uint64_t> wait_park_ns, drain_ns, repark_ns;
};

struct ScopedHostPin {
    cpu::ThreadAffinity previous;
    explicit ScopedHostPin(int core) : previous(cpu::pin_current_thread(core)) {
        if (!previous.valid) throw std::runtime_error("pool host affinity could not be saved/applied");
    }
    ~ScopedHostPin() { cpu::restore_thread_affinity(previous); }
};

cpu::PoolAffinity pool_affinity(const Args& args) {
    if (args.pool_affinity == "auto") return cpu::PoolAffinity::Auto;
    if (args.pool_affinity == "p-cores") return cpu::PoolAffinity::PCores;
    return cpu::PoolAffinity::All;
}

const char* planned_affinity(const Args& args) {
    return args.pool_affinity == "none" ? "all" : args.pool_affinity.c_str();
}

std::vector<cpu::ExpertJobMulti> pool_jobs(const uint8_t* blob, WorkBuffers& b) {
    const int groups = (int) ((b.input.size() + cpu::MAXT - 1) / cpu::MAXT);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) groups);
    for (int j = 0; j < groups; ++j) {
        auto& job = jobs[(size_t) j]; job.blob = blob;
        job.nt = (int) (std::min)((size_t) cpu::MAXT, b.input.size() - (size_t) j * cpu::MAXT);
        for (int t = 0; t < job.nt; ++t) {
            const size_t row = (size_t) j * cpu::MAXT + (size_t) t;
            // Q4_K/Q5_1 pool kernels consume nact; job.act is only used by the Q2 fallback.
            job.nact[t] = b.act[row].data(); job.out[t] = b.output[row].data();
        }
    }
    return jobs;
}

uint64_t ms_ns(double ms) { return (uint64_t) (ms * 1.0e6 + 0.5); }

uint64_t pool_iteration(const NativeFmt& fmt, WorkBuffers& b, cpu::ExpertPool& pool,
                        std::vector<cpu::ExpertJobMulti>& jobs, PoolRunInfo& info, bool record) {
    double w0, d0, r0, w1, d1, r1;
    pool.phase_ms(w0, d0, r0);
    const double gu0 = pool.ms_multi_gu, q0 = pool.ms_multi_q, down0 = pool.ms_multi_down;
    const auto a = std::chrono::steady_clock::now();
    for (size_t t = 0; t < b.input.size(); ++t) cpu::native_quant_act(fmt, b.input[t].data(), b.act[t].data());
    const auto bq = std::chrono::steady_clock::now();
    pool.run_split_multi_native(fmt, jobs.data(), (int) jobs.size());
    const auto c = std::chrono::steady_clock::now();
    pool.phase_ms(w1, d1, r1);
    if (record) {
        info.activation_ns.push_back((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(bq-a).count());
        info.pool_ns.push_back((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(c-bq).count());
        info.gu_ns.push_back(ms_ns(pool.ms_multi_gu-gu0)); info.hidden_q_ns.push_back(ms_ns(pool.ms_multi_q-q0));
        info.down_ns.push_back(ms_ns(pool.ms_multi_down-down0)); info.wait_park_ns.push_back(ms_ns(w1-w0));
        info.drain_ns.push_back(ms_ns(d1-d0)); info.repark_ns.push_back(ms_ns(r1-r0));
    }
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(c-a).count();
}

WorkBuffers allocate_buffers(const Args& args, const NativeFmt& fmt, const uint8_t* input_bytes) {
    WorkBuffers b;
    b.input.reserve(args.rows); b.act.reserve(args.rows); b.hidden_q.reserve(args.rows);
    b.ff.reserve(args.rows); b.output.reserve(args.rows);
    b.act_ptrs.reserve(args.rows); b.hidden_q_ptrs.reserve(args.rows);
    b.ff_ptrs.reserve(args.rows); b.output_ptrs.reserve(args.rows);
    for (int t = 0; t < args.rows; ++t) {
        b.input.emplace_back(static_cast<size_t>(args.hidden));
        float* row = b.input.back().data();
        for (int c = 0; c < args.hidden; ++c) {
            const size_t off = (static_cast<size_t>(t) * args.hidden + c) * 4;
            const uint32_t bits = static_cast<uint32_t>(input_bytes[off]) |
                                  (static_cast<uint32_t>(input_bytes[off+1]) << 8) |
                                  (static_cast<uint32_t>(input_bytes[off+2]) << 16) |
                                  (static_cast<uint32_t>(input_bytes[off+3]) << 24);
            std::memcpy(&row[c], &bits, sizeof(bits));
        }
        if (!all_finite(row, args.hidden)) throw std::runtime_error("input contains NaN or infinity; run is ineligible");
        b.act.emplace_back(fmt.act_bytes); b.act_ptrs.push_back(b.act.back().data());
        b.ff.emplace_back(static_cast<size_t>(args.intermediate)); b.ff_ptrs.push_back(b.ff.back().data());
        b.hidden_q.emplace_back(fmt.h_bytes); b.hidden_q_ptrs.push_back(b.hidden_q.back().data());
        b.output.emplace_back(static_cast<size_t>(args.hidden)); b.output_ptrs.push_back(b.output.back().data());
    }
    return b;
}

void one_iteration(const Args& args, const NativeFmt& fmt, const uint8_t* blob, WorkBuffers& b) {
    for (int t = 0; t < args.rows; ++t) {
        strata::kernels::cpu::native_quant_act(fmt, b.input[t].data(), b.act[t].data());
    }
    strata::kernels::cpu::native_gu_rows(fmt, blob, b.act_ptrs.data(), args.rows, b.ff_ptrs.data(), 0, args.intermediate);
    for (int t = 0; t < args.rows; ++t) strata::kernels::cpu::native_quant_h(fmt, b.ff[t].data(), b.hidden_q[t].data());
    strata::kernels::cpu::native_down_rows(fmt, blob, b.hidden_q_ptrs.data(), args.rows, b.output_ptrs.data(), 0, args.hidden);
}

void validate_finite_outputs(const Args& args, const WorkBuffers& b) {
    for (int t = 0; t < args.rows; ++t) {
        if (!all_finite(b.ff[t].data(), args.intermediate)) throw std::runtime_error("native gate/up produced non-finite intermediate values");
        if (!all_finite(b.output[t].data(), args.hidden)) throw std::runtime_error("native down produced non-finite output values");
    }
}

std::vector<uint8_t> encode_output_le(const Args& args, const WorkBuffers& b) {
    const uint64_t count = checked_mul(args.rows, args.hidden, "output float count");
    const uint64_t bytes = checked_mul(count, sizeof(float), "output bytes");
    if (bytes > kMaxIoBytes || bytes > std::numeric_limits<size_t>::max()) throw std::runtime_error("output exceeds harness byte limit");
    std::vector<uint8_t> encoded(static_cast<size_t>(bytes));
    for (int t = 0; t < args.rows; ++t) {
        for (int c = 0; c < args.hidden; ++c) {
            uint32_t bits = 0;
            std::memcpy(&bits, &b.output[t].data()[c], sizeof(bits));
            const size_t off = (static_cast<size_t>(t) * args.hidden + c) * 4;
            encoded[off] = static_cast<uint8_t>(bits);
            encoded[off+1] = static_cast<uint8_t>(bits >> 8);
            encoded[off+2] = static_cast<uint8_t>(bits >> 16);
            encoded[off+3] = static_cast<uint8_t>(bits >> 24);
        }
    }
    return encoded;
}

std::string run_json(const Args& args, const NativeFmt& fmt, uint64_t blob_bytes, const std::string& blob_sha,
                     uint64_t input_bytes, const std::string& input_sha, uint64_t output_bytes,
                     const std::string& output_sha, const std::vector<uint64_t>& formal_ns, bool avx2_available,
                     const PoolRunInfo* pool_info) {
    std::vector<uint64_t> sorted = formal_ns;
    std::sort(sorted.begin(), sorted.end());
    const uint64_t min_ns = sorted.front();
    const uint64_t median_ns = sorted[sorted.size() / 2];
    std::ostringstream s;
    s << "{\n"
      << "  \"schema\": \"strata-hetero-native-cpu-run-v1\",\n"
      << "  \"created_utc\": " << json_quote(utc_now()) << ",\n"
      << "  \"status\": \"success\",\n"
      << "  \"source\": {\"strata_head\": " << json_quote(HETERO_STRATA_SOURCE_HEAD)
      << ", \"ggml_pin\": " << json_quote(HETERO_GGML_PIN)
      << ", \"native_harness_source_sha256\": " << json_quote(HETERO_NATIVE_SOURCE_SHA256)
      << ", \"sha256_header_sha256\": " << json_quote(HETERO_NATIVE_SHA_HEADER_SHA256)
      << ", \"harness_cmake_sha256\": " << json_quote(HETERO_NATIVE_CMAKE_SHA256) << "},\n"
      << "  \"execution\": {\"mode\": " << json_quote(pool_info ? "actual_native_quantized_cpu_expert_pool" : "actual_native_quantized_cpu_expert")
      << ", \"native_kernel_called\": true, \"engine_pool_called\": " << (pool_info ? "true" : "false")
      << ", \"threads\": " << (pool_info ? pool_info->actual_workers + 1 : 1)
      << ", \"gpu_executed\": false, \"model_cli_executed\": false},\n"
      << "  \"parameters\": {\"rows\": " << args.rows << ", \"hidden\": " << args.hidden
      << ", \"intermediate\": " << args.intermediate << ", \"gu_type\": " << fmt.gu_type
      << ", \"down_type\": " << fmt.d_type << ", \"warmup\": " << args.warmup << ", \"repeat\": " << args.repeat << "},\n"
      << "  \"reference_controls\": {\"STRATA_KQ256\": \"0 (process-local)\", \"STRATA_NO_Q8K_AVX2\": "
      << json_quote(args.canonical_q8k ? "1 (explicit --canonical-q8k, process-local)" : "unset (model-default Q8_K activation path, process-local)")
      << ", \"q8k_avx2_available\": " << (avx2_available ? "true" : "false")
      << ", \"q8k_avx2_selected\": " << ((!args.canonical_q8k && avx2_available) ? "true" : "false")
      << ", \"activation_quantizer\": " << json_quote((!args.canonical_q8k && avx2_available) ? "q8k_quant_avx2" : "ggml Q8_K from_float")
      << ", \"down_activation_quantizer\": \"ggml Q8_1 from_float\", \"STRATA_IQ_MT_MIN\": \"not used by Q4_K/Q5_1 path\"},\n"
      ;
    if (pool_info) {
        s << "  \"pool\": {\"requested_background_workers\": " << pool_info->workers
          << ", \"actual_background_workers\": " << pool_info->actual_workers << ", \"host_works\": true"
          << ", \"effective_compute_participants\": " << pool_info->actual_workers + 1 << ", \"pin\": " << (pool_info->pin ? "true" : "false")
          << ", \"affinity\": " << json_quote(args.pool_affinity)
          << ", \"topology_scope\": \"planned_only\", \"planned_topology_affinity\": " << json_quote(planned_affinity(args))
          << ", \"planned_host_cpu_id\": " << pool_info->topology.host_core
          << ", \"host_pin_applied\": " << (pool_info->host_pinned ? "true" : "false") << ", \"planned_worker_cpu_ids\": [";
        for (size_t i = 0; i < pool_info->topology.worker_cores.size(); ++i) { if (i) s << ", "; s << pool_info->topology.worker_cores[i]; }
        s << "], \"unpinned_worker_overflow\": " << pool_info->unpinned_overflow
          << ", \"planned_pinned_worker_ids\": [";
        if (pool_info->pin) {
            for (int i = 0; i < pool_info->workers && i < (int) pool_info->topology.worker_cores.size(); ++i) {
                if (i) s << ", ";
                s << pool_info->topology.worker_cores[(size_t) i];
            }
        }
        s << "], \"worker_pin_success_available\": true, \"worker_affinity_observations\": [";
        for (size_t i = 0; i < pool_info->worker_affinity.size(); ++i) {
            if (i) s << ", ";
            const auto& r = pool_info->worker_affinity[i];
            s << "{\"worker\":" << r.worker << ",\"requested_core\":" << r.requested_core
              << ",\"ready\":" << (r.ready ? "true" : "false")
              << ",\"pin_requested\":" << (r.pin_requested ? "true" : "false")
              << ",\"pin_applied\":" << (r.pin_applied ? "true" : "false")
              << ",\"mask_observed\":" << (r.mask_observed ? "true" : "false")
              << ",\"mask_matches_request\":" << (r.mask_matches_request ? "true" : "false")
              << ",\"observed_group\":" << r.observed_group << ",\"observed_mask\":" << r.observed_mask
              << ",\"startup_processor\":" << r.startup_processor << '}';
        }
        s << ']'
          << ", \"topology_basis\": \"upstream OS class heuristic; P/E labels are not independently verified\"},\n"
          << "  \"pool_timing_ns\": {";
        const std::vector<uint64_t>* arrays[] = {&pool_info->activation_ns, &pool_info->pool_ns, &pool_info->gu_ns,
            &pool_info->hidden_q_ns, &pool_info->down_ns, &pool_info->wait_park_ns, &pool_info->drain_ns, &pool_info->repark_ns};
        const char* names[] = {"activation_quant", "pool_call", "gate_up", "hidden_quant", "down", "wait_park", "drain", "repark"};
        for (size_t a = 0; a < 8; ++a) {
            if (a) s << ", ";
            s << "\"" << names[a] << "\": [";
            for (size_t i = 0; i < arrays[a]->size(); ++i) { if (i) s << ", "; s << (*arrays[a])[i]; }
            s << "]";
        }
        s << "},\n  \"pool_timing_scope\": \"gate_up/hidden_quant/down are included in pool_call; wait_park/drain/repark are pool host phase counters\",\n";
    }
    s << "  \"layout\": {\"gu_row_bytes\": " << fmt.gu_row << ", \"down_row_bytes\": " << fmt.d_row
      << ", \"up_offset_bytes\": " << fmt.up_off << ", \"down_offset_bytes\": " << fmt.down_off
      << ", \"blob_bytes\": " << fmt.bytes << ", \"activation_bytes\": " << fmt.act_bytes
      << ", \"hidden_quantized_bytes\": " << fmt.h_bytes << "},\n"
      << "  \"blob\": {\"path\": " << utc_path(args.blob) << ", \"bytes\": " << blob_bytes << ", \"sha256\": " << json_quote(blob_sha) << "},\n"
      << "  \"input\": {\"path\": " << utc_path(args.input) << ", \"bytes\": " << input_bytes << ", \"format\": \"token-major raw little-endian float32\", \"sha256\": " << json_quote(input_sha) << "},\n"
      << "  \"timing_scope\": " << json_quote(pool_info ? "formal total includes native input quantization plus ExpertPool gate/up, hidden quantization and down; pool_call is separately timed; file I/O, allocation, SHA-256, and output serialization are excluded"
                                                        : "each formal iteration includes x-to-Q8 activation quantization, full gate/up rows with SiLU-times-up, hidden-to-Q8 requantization, and full down rows; file I/O, allocation, SHA-256, and output serialization are excluded") << ",\n"
      << "  \"timings_ns\": {\"warmup_iterations\": " << args.warmup << ", \"formal_iterations\": [";
    for (size_t i = 0; i < formal_ns.size(); ++i) { if (i) s << ", "; s << formal_ns[i]; }
    s << "], \"min\": " << min_ns << ", \"median\": " << median_ns << "},\n"
      << "  \"output\": {\"path\": " << utc_path(args.output) << ", \"bytes\": " << output_bytes
      << ", \"format\": \"token-major raw little-endian float32\", \"sha256\": " << json_quote(output_sha) << "},\n"
      << "  \"claim_limit\": " << json_quote(pool_info ? "standalone native expert CPU-pool microbenchmark; not model throughput or GPU performance"
                                                       : "single-thread native expert reference only; not engine scheduling, model throughput, or GPU performance") << "\n}\n";
    return s.str();
}

void write_new_file(const fs::path& path, const void* data, uint64_t bytes, const char* label) {
    if (fs::exists(path)) throw std::runtime_error(std::string(label) + " already exists; refusing to overwrite");
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t offset = 0;
#if defined(_WIN32)
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("exclusive create failed for ") + label + ": " + path_string(path));
    try {
        while (offset < bytes) {
            const DWORD chunk = static_cast<DWORD>((bytes - offset) > (1u << 20) ? (1u << 20) : (bytes - offset));
            DWORD written = 0;
            if (!WriteFile(file, p + offset, chunk, &written, nullptr) || written == 0) {
                throw std::runtime_error(std::string("write failed for ") + label + " (partial file preserved): " + path_string(path));
            }
            offset += written;
        }
        if (!FlushFileBuffers(file)) throw std::runtime_error(std::string("flush failed for ") + label + " (file preserved): " + path_string(path));
        HANDLE closing = file;
        file = INVALID_HANDLE_VALUE;
        if (!CloseHandle(closing)) throw std::runtime_error(std::string("close failed for ") + label + " (file preserved): " + path_string(path));
    } catch (...) {
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        throw;
    }
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) throw std::runtime_error(std::string("exclusive create failed for ") + label + ": " + path_string(path));
    int open_fd = fd;
    try {
        while (offset < bytes) {
            const size_t chunk = static_cast<size_t>((bytes - offset) > (1u << 20) ? (1u << 20) : (bytes - offset));
            const ssize_t written = ::write(open_fd, p + offset, chunk);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error(std::string("write failed for ") + label + " (partial file preserved): " + path_string(path));
            offset += static_cast<uint64_t>(written);
        }
        if (::fsync(open_fd) != 0) throw std::runtime_error(std::string("flush failed for ") + label + " (file preserved): " + path_string(path));
        const int closing = open_fd;
        open_fd = -1;
        if (::close(closing) != 0) throw std::runtime_error(std::string("close failed for ") + label + " (file preserved): " + path_string(path));
    } catch (...) {
        if (open_fd >= 0) ::close(open_fd);
        throw;
    }
#endif
}

std::string failure_receipt_json(const Args& args, const RunContext& context, std::string_view message) {
    std::ostringstream s;
    s << "{\n"
      << "  \"schema\": \"strata-hetero-native-cpu-run-v1\",\n"
      << "  \"created_utc\": " << json_quote(utc_now()) << ",\n"
      << "  \"status\": \"failed\",\n"
      << "  \"failure\": {\"phase\": " << json_quote(context.phase)
      << ", \"message\": " << json_quote(message) << "},\n"
      << "  \"source\": {\"strata_head\": " << json_quote(HETERO_STRATA_SOURCE_HEAD)
      << ", \"ggml_pin\": " << json_quote(HETERO_GGML_PIN)
      << ", \"native_harness_source_sha256\": " << json_quote(HETERO_NATIVE_SOURCE_SHA256)
      << ", \"sha256_header_sha256\": " << json_quote(HETERO_NATIVE_SHA_HEADER_SHA256)
      << ", \"harness_cmake_sha256\": " << json_quote(HETERO_NATIVE_CMAKE_SHA256) << "},\n"
      << "  \"parameters\": {\"rows\": " << args.rows << ", \"hidden\": " << args.hidden
      << ", \"intermediate\": " << args.intermediate << ", \"gu_type\": " << args.gu_type
      << ", \"down_type\": " << args.down_type << ", \"warmup\": " << args.warmup
      << ", \"repeat\": " << args.repeat << "},\n";
    if (args.pool_workers_set)
        s << "  \"pool_request\": {\"workers\": " << args.pool_workers << ", \"affinity\": "
          << json_quote(args.pool_affinity) << "},\n";
    s << "  \"paths\": {\"blob\": " << json_quote(path_string(fs::absolute(args.blob).lexically_normal()))
      << ", \"input\": " << json_quote(path_string(fs::absolute(args.input).lexically_normal()))
      << ", \"output\": " << json_quote(path_string(fs::absolute(args.output).lexically_normal()))
      << ", \"receipt\": " << json_quote(path_string(fs::absolute(args.receipt).lexically_normal())) << "},\n"
      << "  \"output_write_started\": " << (context.output_write_started ? "true" : "false") << ",\n"
      << "  \"output_write_complete\": " << (context.output_write_complete ? "true" : "false") << ",\n"
      << "  \"partial_output_preserved_if_created\": true,\n"
      << "  \"gpu_executed\": false, \"model_cli_executed\": false\n}\n";
    return s.str();
}

void preserve_run_failure(const Args& args, const RunContext& context, std::string_view message) {
    if (!context.receipt_path_safe || context.receipt_write_complete || args.receipt.empty()) return;
    try {
        const std::string json = failure_receipt_json(args, context, message);
        write_new_file(args.receipt, json.data(), json.size(), "failure JSON receipt");
    } catch (const std::exception& e) {
        std::cerr << "hetero_native_expert: could not preserve failure receipt without overwriting: " << e.what() << '\n';
    }
}

int run(const Args& args, RunContext& context) {
    if (!args.has_rows || !args.has_hidden || !args.has_intermediate || !args.has_gu_type || !args.has_down_type ||
        args.blob.empty() || args.input.empty() || args.output.empty() || args.receipt.empty()) {
        throw std::runtime_error("--run requires --blob, --input, --rows, --hidden, --intermediate, --gu-type, --down-type, --output, and --receiptJSON");
    }
    context.phase = "path_preflight";
    validate_no_path_collisions(args, context);
    context.phase = "dimension_validation";
    validate_dimensions(args);
    const uint64_t input_float_count = checked_mul(args.rows, args.hidden, "input float count");
    const uint64_t expected_input_bytes = checked_mul(input_float_count, sizeof(float), "input bytes");
    if (expected_input_bytes > kMaxIoBytes) throw std::runtime_error("input exceeds harness byte limit");
    context.phase = "reference_environment";
    set_process_reference_environment(args.canonical_q8k);
    const bool avx2_available = strata::kernels::cpu::cpu_avx2_ok();
    context.phase = "native_format_validation";
    const NativeFmt fmt = make_format(args);
    context.phase = "blob_and_input_size_validation";
    const uint64_t blob_size = exact_file_size(args.blob, fmt.bytes, "quantized native blob");
    const uint64_t input_size = exact_file_size(args.input, expected_input_bytes, "F32LE input");
    AlignedArray<uint8_t> blob(static_cast<size_t>(blob_size));
    AlignedArray<uint8_t> input_raw(static_cast<size_t>(input_size));
    context.phase = "read_blob_and_input";
    read_exact_file(args.blob, blob, "quantized native blob");
    read_exact_file(args.input, input_raw, "F32LE input");
    const std::string blob_sha = sha256(blob.data(), static_cast<size_t>(blob_size));
    const std::string input_sha = sha256(input_raw.data(), static_cast<size_t>(input_size));
    WorkBuffers buffers = allocate_buffers(args, fmt, input_raw.data());
    PoolRunInfo pool_info;
    std::unique_ptr<ScopedHostPin> host_pin; // declared before the pool: pool workers join before the caller is restored
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::ExpertJobMulti> jobs;
    if (args.pool_workers_set) {
        context.phase = "pool_initialization";
        pool_info.workers = args.pool_workers; pool_info.pin = args.pool_affinity != "none";
        pool_info.affinity = pool_affinity(args); pool_info.topology = cpu::detect_cpu_topology(true, pool_info.affinity);
        if (pool_info.affinity == cpu::PoolAffinity::PCores && pool_info.workers > (int) pool_info.topology.worker_cores.size())
            throw std::runtime_error("p-cores pool worker count exceeds eligible worker IDs; refusing unpinned overflow");
        pool_info.unpinned_overflow = pool_info.pin
            ? (std::max)(0, pool_info.workers - (int) pool_info.topology.worker_cores.size()) : 0;
        if (pool_info.pin) {
            host_pin = std::make_unique<ScopedHostPin>(pool_info.topology.host_core);
            pool_info.host_pinned = true;
        }
        pool = std::make_unique<cpu::ExpertPool>(pool_info.workers, pool_info.pin, true, pool_info.affinity);
        pool_info.actual_workers = pool->workers();
        jobs = pool_jobs(blob.data(), buffers);
    }

    if (pool) {
        pool_info.activation_ns.reserve((size_t) args.repeat); pool_info.pool_ns.reserve((size_t) args.repeat);
        pool_info.gu_ns.reserve((size_t) args.repeat); pool_info.hidden_q_ns.reserve((size_t) args.repeat);
        pool_info.down_ns.reserve((size_t) args.repeat); pool_info.wait_park_ns.reserve((size_t) args.repeat);
        pool_info.drain_ns.reserve((size_t) args.repeat); pool_info.repark_ns.reserve((size_t) args.repeat);
    }
    context.phase = "warmup_and_native_kernel_run";
    for (uint64_t i = 0; i < args.warmup; ++i) {
        if (pool) (void) pool_iteration(fmt, buffers, *pool, jobs, pool_info, false);
        else one_iteration(args, fmt, blob.data(), buffers);
        validate_finite_outputs(args, buffers);
    }
    std::vector<uint64_t> formal_ns;
    formal_ns.reserve(static_cast<size_t>(args.repeat));
    for (uint64_t i = 0; i < args.repeat; ++i) {
        if (pool) formal_ns.push_back(pool_iteration(fmt, buffers, *pool, jobs, pool_info, true));
        else {
            const auto start = std::chrono::steady_clock::now();
            one_iteration(args, fmt, blob.data(), buffers);
            const auto stop = std::chrono::steady_clock::now();
            formal_ns.push_back(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
        }
        validate_finite_outputs(args, buffers);
    }
    context.phase = "output_serialization";
    if (pool) pool_info.worker_affinity = pool->worker_affinity();
    std::vector<uint8_t> encoded = encode_output_le(args, buffers);
    const std::string output_sha = sha256(encoded.data(), encoded.size());
    const std::string receipt_json = run_json(args, fmt, blob_size, blob_sha, input_size, input_sha,
                                              encoded.size(), output_sha, formal_ns, avx2_available,
                                              pool ? &pool_info : nullptr);
    context.phase = "write_output";
    context.output_write_started = true;
    write_new_file(args.output, encoded.data(), encoded.size(), "binary output");
    context.output_write_complete = true;
    context.phase = "write_receipt";
    write_new_file(args.receipt, receipt_json.data(), receipt_json.size(), "JSON receipt");
    context.receipt_write_complete = true;
    std::cout << "{\"status\":\"success\",\"receipt\":" << json_quote(path_string(fs::absolute(args.receipt)))
              << ",\"output\":" << json_quote(path_string(fs::absolute(args.output))) << "}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    RunContext context;
    bool parsed = false;
    try {
        args = parse_args(argc, argv);
        parsed = true;
        if (args.help) { print_help(); return 0; }
        if (args.validate_only && args.run) throw std::runtime_error("--validate-only cannot be combined with --run");
        if (!args.run) {
            if (args.canonical_q8k) throw std::runtime_error("--canonical-q8k requires explicit --run");
            if (!args.blob.empty() || !args.input.empty() || !args.output.empty() || !args.receipt.empty()) {
                throw std::runtime_error("file options require explicit --run");
            }
            const NativeFmt fmt = make_format(args);
            std::cout << metadata_json(args, fmt);
            return 0;
        }
        return run(args, context);
    } catch (const std::exception& e) {
        if (parsed && args.run) preserve_run_failure(args, context, e.what());
        std::cerr << "hetero_native_expert: " << e.what() << '\n';
        return 2;
    } catch (...) {
        if (parsed && args.run) preserve_run_failure(args, context, "unknown non-standard exception");
        std::cerr << "hetero_native_expert: unknown non-standard exception\n";
        return 2;
    }
}
