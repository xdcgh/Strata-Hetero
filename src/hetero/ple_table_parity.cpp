// Bounded, CUDA-free PLE-table row/gather parity harness.
// Default invocation is metadata-only. Reading a GGUF requires --run and explicit input/output paths.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using strata::kernels::PLE_HEAD_DIM;
using strata::kernels::PLE_N_HEADS;
using strata::kernels::PleIo;
using strata::kernels::PleIoOptions;
using strata::kernels::PleTable;

namespace {

constexpr size_t kTokens = 512;
constexpr size_t kRowsPerToken = PLE_N_HEADS;
constexpr size_t kTotalRows = kTokens * kRowsPerToken;
constexpr size_t kWarmups = 1;
constexpr size_t kRepeats = 3;
constexpr std::array<size_t, 5> kBatchSizes = {1, 2, 6, 96, 512};
constexpr uint64_t kSeed = 42;
constexpr uint64_t kRamReserve = 12ull << 30;
constexpr uint32_t kIq4NlRowBytes = 90;

struct Args {
    bool run = false;
    bool help = false;
    bool run_seen = false;
    bool file_seen = false;
    bool output_seen = false;
    fs::path input;
    fs::path output;
};

struct BatchResult {
    size_t batch_size = 0;
    size_t formal_repeats = 0;
    bool warmup_ok = false;
    std::string warmup_sha256;
    std::string error;
    std::vector<std::string> hashes;
    std::vector<size_t> differing_floats;
    std::vector<double> max_abs_diff;
    std::vector<size_t> finite_values;
    std::vector<std::vector<float>> outputs;
    bool read_rows_match_gather = true;
};

struct ArmResult {
    std::string name;
    std::string mode;
    bool lock_requested = false;
    bool locked = false;
    uint64_t locked_bytes = 0;
    uint64_t expected_locked_bytes = 0;
    std::string open_started_utc;
    std::string input_identity_before;
    std::string input_identity_after;
    bool input_identity_stable = false;
    double open_ms = 0;
    bool open_ok = false;
    bool close_ok = false;
    bool selected_read_row_matches_gather = true;
    uint64_t read_rows_checked = 0;
    std::string error;
    std::vector<BatchResult> batches;
    std::map<uint32_t, std::string> selected_row_hashes;
};

std::string input_identity(const fs::path& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot inspect input file identity");
    BY_HANDLE_FILE_INFORMATION info{};
    const BOOL ok = GetFileInformationByHandle(h, &info);
    FILE_BASIC_INFO basic{};
    const BOOL basic_ok = GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic));
    CloseHandle(h);
    if (!ok || !basic_ok) throw std::runtime_error("cannot read input file identity/change time");
    const uint64_t size = ((uint64_t)info.nFileSizeHigh << 32) | info.nFileSizeLow;
    const uint64_t index = ((uint64_t)info.nFileIndexHigh << 32) | info.nFileIndexLow;
    const uint64_t mtime = ((uint64_t)info.ftLastWriteTime.dwHighDateTime << 32) | info.ftLastWriteTime.dwLowDateTime;
    std::ostringstream out;
    out << "volume=" << info.dwVolumeSerialNumber << ";file_index=" << index << ";size=" << size
        << ";mtime=" << mtime << ";ctime=" << basic.ChangeTime.QuadPart;
    return out.str();
#else
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot inspect input file identity");
    std::ostringstream out;
    out << "device=" << (uint64_t)st.st_dev << ";inode=" << (uint64_t)st.st_ino
        << ";size=" << (uint64_t)st.st_size << ";mtime_ns="
        << ((int64_t)st.st_mtim.tv_sec * 1000000000ll + st.st_mtim.tv_nsec) << ";ctime_ns="
        << ((int64_t)st.st_ctim.tv_sec * 1000000000ll + st.st_ctim.tv_nsec);
    return out.str();
#endif
}

std::string json_escape(std::string_view s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned)c << std::dec;
            else out << (char)c;
        }
    }
    return out.str();
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") a.help = true;
        else if (arg == "--run") {
            if (a.run_seen) throw std::runtime_error("duplicate --run");
            a.run_seen = true;
            a.run = true;
        }
        else if (arg == "--file" || arg == "--output") {
            if (++i >= argc) throw std::runtime_error("missing path after " + std::string(arg));
            if (arg == "--file") {
                if (a.file_seen) throw std::runtime_error("duplicate --file");
                a.file_seen = true;
                a.input = argv[i];
            } else {
                if (a.output_seen) throw std::runtime_error("duplicate --output");
                a.output_seen = true;
                a.output = argv[i];
            }
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (a.help && argc != 2) throw std::runtime_error("--help cannot be combined with other arguments");
    if (a.run) {
        if (a.input.empty() || a.output.empty()) throw std::runtime_error("--run requires --file and --output");
        if (!a.input.is_absolute() || !a.output.is_absolute())
            throw std::runtime_error("--file and --output must be absolute paths");
    } else if (!a.input.empty() || !a.output.empty()) {
        throw std::runtime_error("--file/--output require explicit --run");
    }
    return a;
}

void print_help() {
    std::puts("strata-ple-table-parity (CUDA-free)\n"
              "  no arguments       print metadata only; no files are opened\n"
              "  --run --file ABS_GGUF --output NEW_ABS_JSON\n"
              "                     run bounded Direct/mmap/page-lock row parity");
}

class ExclusiveOutput {
public:
    explicit ExclusiveOutput(const fs::path& path) {
#if defined(_WIN32)
        handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot exclusively create output receipt");
#else
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd_ < 0) throw std::runtime_error("cannot exclusively create output receipt");
#endif
    }
    ~ExclusiveOutput() { close(); }
    ExclusiveOutput(const ExclusiveOutput&) = delete;
    ExclusiveOutput& operator=(const ExclusiveOutput&) = delete;
    void write_all(const std::string& s) {
        write_started_ = true;
        size_t at = 0;
        while (at < s.size()) {
#if defined(_WIN32)
            const DWORD n = (DWORD)std::min<size_t>(s.size() - at, 1u << 20);
            DWORD wrote = 0;
            if (!WriteFile(handle_, s.data() + at, n, &wrote, nullptr) || wrote == 0)
                throw std::runtime_error("failed writing receipt");
            at += wrote;
#else
            const ssize_t n = ::write(fd_, s.data() + at, s.size() - at);
            if (n <= 0) throw std::runtime_error("failed writing receipt");
            at += (size_t)n;
#endif
        }
    }
    bool write_started() const { return write_started_; }
    void finish() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE && !FlushFileBuffers(handle_)) {
            close();
            throw std::runtime_error("FlushFileBuffers failed for receipt");
        }
        if (handle_ != INVALID_HANDLE_VALUE) {
            HANDLE h = handle_;
            handle_ = INVALID_HANDLE_VALUE;
            if (!CloseHandle(h)) throw std::runtime_error("CloseHandle failed for receipt");
        }
#else
        if (fd_ >= 0 && ::fsync(fd_) != 0) {
            close();
            throw std::runtime_error("fsync failed for receipt");
        }
        if (fd_ >= 0) {
            const int fd = fd_;
            fd_ = -1;
            if (::close(fd) != 0) throw std::runtime_error("close failed for receipt");
        }
#endif
    }
    void close() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); handle_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
    }
private:
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool write_started_ = false;
};

void validate_paths(const fs::path& input, const fs::path& output) {
    if (!fs::is_regular_file(input)) throw std::runtime_error("input is not an existing regular file");
    if (fs::exists(output)) throw std::runtime_error("output already exists; refusing overwrite");
    if (!fs::is_directory(output.parent_path())) throw std::runtime_error("output parent directory must already exist");
    const fs::path canonical_input = fs::canonical(input);
    const fs::path canonical_output = fs::weakly_canonical(output.parent_path()) / output.filename();
    if (canonical_input == canonical_output) throw std::runtime_error("output path aliases the input file");
}

std::vector<uint32_t> select_rows(uint64_t nrows, uint64_t table_offset, std::vector<uint32_t>& selected) {
    if (nrows < 2 || nrows > UINT32_MAX) throw std::runtime_error("PLE table row count is outside supported range");
    std::mt19937_64 rng(kSeed);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)nrows - 1);
    std::vector<uint32_t> rows(kTotalRows);
    for (auto& row : rows) row = dist(rng);

    std::vector<uint32_t> special = {0, (uint32_t)nrows - 1};
    for (uint64_t r = 0; r < std::min<uint64_t>(nrows, 64); ++r) {
        const uint64_t first = table_offset + r * kIq4NlRowBytes;
        const uint64_t last = first + kIq4NlRowBytes - 1;
        if (first / 4096 != last / 4096) { special.push_back((uint32_t)r); break; }
    }
    for (size_t i = 0; i < special.size(); ++i) rows[i] = special[i];
    selected = special;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return rows;
}

std::string hash_floats(const float* p, size_t n) {
    return hetero_native_cpu::sha256_hex(p, n * sizeof(float));
}

bool all_finite(const std::vector<float>& data, size_t& finite_count) {
    finite_count = 0;
    for (float x : data) if (std::isfinite(x)) ++finite_count;
    return finite_count == data.size();
}

BatchResult gather_sweep(PleTable& table, size_t batch, const std::vector<uint32_t>& rows,
                         const std::vector<float>* reference) {
    BatchResult result;
    result.batch_size = batch;
    std::vector<float> output(kTokens * kRowsPerToken * PLE_HEAD_DIM);
    std::string err;
    for (size_t at = 0; at < kTokens; at += batch) {
        const size_t count = std::min(batch, kTokens - at);
        if (!table.gather_batch(rows.data() + at * kRowsPerToken, count,
                                output.data() + at * kRowsPerToken * PLE_HEAD_DIM, err)) {
            result.error = err.empty() ? "PleTable::gather_batch failed without diagnostic" : err;
            return result;
        }
    }
    size_t finite = 0;
    if (!all_finite(output, finite)) result.error = "non-finite value in gathered output";
    result.warmup_ok = true;
    result.formal_repeats = 1;
    result.hashes.push_back(hash_floats(output.data(), output.size()));
    result.finite_values.push_back(finite);
    if (reference) {
        const size_t n = output.size();
        size_t diffs = 0;
        double max_abs = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&output[i], &(*reference)[i], sizeof(float)) != 0) ++diffs;
            const double delta = std::fabs((double)output[i] - (double)(*reference)[i]);
            if (std::isfinite(delta)) max_abs = std::max(max_abs, delta);
        }
        result.differing_floats.push_back(diffs);
        result.max_abs_diff.push_back(max_abs);
    }
    result.outputs.push_back(std::move(output));
    return result;
}

std::string arm_json(const ArmResult& a) {
    std::ostringstream o;
    o << "{\"name\":\"" << json_escape(a.name) << "\",\"mode\":\"" << a.mode
      << "\",\"lock_requested\":" << (a.lock_requested ? "true" : "false")
      << ",\"locked\":" << (a.locked ? "true" : "false") << ",\"locked_bytes\":" << a.locked_bytes
      << ",\"expected_locked_bytes\":" << a.expected_locked_bytes << ",\"open_started_utc\":\""
      << json_escape(a.open_started_utc) << "\",\"input_identity_before\":\""
      << json_escape(a.input_identity_before) << "\",\"input_identity_after\":\""
      << json_escape(a.input_identity_after) << "\",\"input_identity_stable\":"
      << (a.input_identity_stable ? "true" : "false") << ",\"open_ms\":" << a.open_ms << ",\"open_ok\":"
      << (a.open_ok ? "true" : "false") << ",\"close_ok\":" << (a.close_ok ? "true" : "false")
      << ",\"selected_read_row_matches_gather\":" << (a.selected_read_row_matches_gather ? "true" : "false")
      << ",\"read_rows_checked\":" << a.read_rows_checked
      << ",\"error\":\"" << json_escape(a.error) << "\",\"batch_results\":[";
    for (size_t i = 0; i < a.batches.size(); ++i) {
        if (i) o << ',';
        const auto& b = a.batches[i];
        o << "{\"batch_size\":" << b.batch_size << ",\"warmups\":" << kWarmups
          << ",\"formal_repeats\":" << b.formal_repeats << ",\"warmup_ok\":" << (b.warmup_ok ? "true" : "false")
          << ",\"warmup_sha256\":\"" << b.warmup_sha256 << "\",\"error\":\""
          << json_escape(b.error) << "\",\"output_sha256\":[";
        for (size_t j = 0; j < b.hashes.size(); ++j) { if (j) o << ','; o << '"' << b.hashes[j] << '"'; }
        o << "],\"finite_value_counts\":[";
        for (size_t j = 0; j < b.finite_values.size(); ++j) { if (j) o << ','; o << b.finite_values[j]; }
        o << "],\"bit_different_float_counts_vs_direct\":[";
        for (size_t j = 0; j < b.differing_floats.size(); ++j) { if (j) o << ','; o << b.differing_floats[j]; }
        o << "],\"max_abs_diff_vs_direct\":[";
        for (size_t j = 0; j < b.max_abs_diff.size(); ++j) { if (j) o << ','; o << b.max_abs_diff[j]; }
        o << "]}";
    }
    o << "],\"selected_row_sha256\":{";
    size_t n = 0;
    for (const auto& [row, hash] : a.selected_row_hashes) { if (n++) o << ','; o << '"' << row << "\":\"" << hash << '"'; }
    o << "}}";
    return o.str();
}

std::string failure_json(const Args& args, const std::string& phase, const std::string& error,
                         const std::vector<ArmResult>& arms, const std::string& run_identity) {
    std::ostringstream o;
    o << "{\"schema_version\":1,\"status\":\"failed\",\"phase\":\"" << json_escape(phase)
      << "\",\"error\":\"" << json_escape(error) << "\",\"input\":\"" << json_escape(args.input.string())
      << "\",\"output\":\"" << json_escape(args.output.string()) << "\",\"run_input_identity\":\""
      << json_escape(run_identity) << "\",\"gpu_executed\":false,\"arms\":[";
    for (size_t i = 0; i < arms.size(); ++i) { if (i) o << ','; o << arm_json(arms[i]); }
    o << "]}\n";
    return o.str();
}

ArmResult run_arm(const std::string& name, PleIo mode, bool lock, const fs::path& input,
                  const std::vector<uint32_t>& rows, const std::vector<uint32_t>& selected,
                  const std::vector<std::vector<float>>* direct_reference, uint64_t expected_lock_bytes,
                  const std::string& run_identity) {
    ArmResult arm;
    arm.name = name;
    arm.mode = mode == PleIo::Direct ? "direct" : "mmap";
    arm.lock_requested = lock;
    arm.expected_locked_bytes = expected_lock_bytes;
    arm.open_started_utc = utc_now();
    arm.input_identity_before = input_identity(input);
    if (arm.input_identity_before != run_identity) {
        arm.input_identity_after = arm.input_identity_before;
        arm.input_identity_stable = false;
        arm.error = "input identity differs from this run's initial identity; refusing this arm";
        return arm;
    }
    const auto t0 = std::chrono::steady_clock::now();
    PleTable table;
    PleIoOptions io;
    io.mode = mode;
    io.lock = lock;
    io.ram_reserve_bytes = kRamReserve;
    io.cache_rows = 0;
    std::string err;
    arm.open_ok = table.open(input.string(), err, io);
    arm.open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!arm.open_ok) {
        arm.error = err;
        try {
            arm.input_identity_after = input_identity(input);
            arm.input_identity_stable = arm.input_identity_before == arm.input_identity_after &&
                                        arm.input_identity_before == run_identity;
        } catch (...) {}
        return arm;
    }
    arm.locked = table.locked();
    arm.locked_bytes = table.locked_bytes();
    if (table.format() == nullptr || std::string_view(table.format()) != "IQ4_NL") {
        arm.error = "input PLE format is not IQ4_NL";
    } else if (lock && (!arm.locked || arm.locked_bytes != expected_lock_bytes)) {
        arm.error = "locked arm did not lock the exact runtime table byte count";
    }

    for (size_t bi = 0; arm.error.empty() && bi < kBatchSizes.size(); ++bi) {
        const size_t batch = kBatchSizes[bi];
        BatchResult warm = gather_sweep(table, batch, rows, nullptr);
        if (!warm.error.empty()) { arm.error = warm.error; break; }
        BatchResult combined;
        combined.batch_size = batch;
        combined.warmup_ok = true;
        combined.warmup_sha256 = warm.hashes.front();
        const std::vector<float>* ref = direct_reference ? &(*direct_reference)[bi] : nullptr;
        for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
            BatchResult formal = gather_sweep(table, batch, rows, ref);
            if (!formal.error.empty()) { arm.error = formal.error; combined.error = formal.error; break; }
            if (repeat == 0 && formal.hashes.front() != combined.warmup_sha256) {
                arm.error = "formal output differs from its warmup output";
                combined.error = arm.error;
                break;
            }
            ++combined.formal_repeats;
            combined.hashes.push_back(formal.hashes.front());
            combined.finite_values.push_back(formal.finite_values.front());
            if (ref) {
                combined.differing_floats.push_back(formal.differing_floats.front());
                combined.max_abs_diff.push_back(formal.max_abs_diff.front());
            } else {
                size_t diffs = 0;
                double max_abs = 0;
                if (combined.outputs.empty()) {
                    combined.differing_floats.push_back(0);
                    combined.max_abs_diff.push_back(0);
                } else {
                    const auto& baseline = combined.outputs.front();
                    const auto& current = formal.outputs.front();
                    for (size_t i = 0; i < baseline.size(); ++i) {
                        if (std::memcmp(&baseline[i], &current[i], sizeof(float)) != 0) ++diffs;
                        const double delta = std::fabs((double)baseline[i] - (double)current[i]);
                        if (std::isfinite(delta)) max_abs = std::max(max_abs, delta);
                    }
                    combined.differing_floats.push_back(diffs);
                    combined.max_abs_diff.push_back(max_abs);
                    if (diffs != 0 || combined.hashes.front() != formal.hashes.front()) {
                        combined.error = "Direct formal repeats are not bit-exact";
                        arm.error = combined.error;
                    }
                }
            }
            if (repeat == 0) combined.outputs.push_back(std::move(formal.outputs.front()));
        }
        if (combined.error.empty() && batch == 512 && !combined.outputs.empty()) {
            const auto& output = combined.outputs.front();
            for (uint32_t row : selected) {
                auto it = std::find(rows.begin(), rows.end(), row);
                if (it == rows.end()) continue;
                const size_t index = (size_t)(it - rows.begin()) * PLE_HEAD_DIM;
                arm.selected_row_hashes[row] = hash_floats(output.data() + index, PLE_HEAD_DIM);
            }
        }
        arm.batches.push_back(std::move(combined));
        if (!arm.error.empty()) break;
    }

    arm.close_ok = true;
    if (arm.error.empty() && !arm.batches.empty() && arm.batches.back().batch_size == 512 &&
        !arm.batches.back().outputs.empty()) {
        const auto& gathered = arm.batches.back().outputs.front();
        for (size_t i = 0; i < rows.size(); ++i) {
            const uint32_t row = rows[i];
            float row_values[PLE_HEAD_DIM]{};
            table.read_row(row, row_values);
            const size_t offset = i * PLE_HEAD_DIM;
            ++arm.read_rows_checked;
            if (std::memcmp(row_values, gathered.data() + offset, sizeof(row_values)) != 0)
                arm.selected_read_row_matches_gather = false;
            if (std::find(selected.begin(), selected.end(), row) != selected.end())
                arm.selected_row_hashes[row] = hash_floats(row_values, PLE_HEAD_DIM);
        }
        if (!arm.selected_read_row_matches_gather)
            arm.error = "one or more PleTable::read_row values differ from corresponding gather_batch rows";
        if (arm.read_rows_checked != kTotalRows) arm.error = "not all generated rows were checked with read_row";
    }
    table.close();
    if (table.is_open()) { arm.close_ok = false; if (arm.error.empty()) arm.error = "PleTable remained open after close"; }
    try {
        arm.input_identity_after = input_identity(input);
        arm.input_identity_stable = arm.input_identity_before == arm.input_identity_after &&
                                    arm.input_identity_before == run_identity;
        if (!arm.input_identity_stable && arm.error.empty()) arm.error = "input file identity changed during arm";
    } catch (const std::exception& e) {
        arm.error = std::string("cannot verify input identity after arm: ") + e.what();
    }
    return arm;
}

void write_success(ExclusiveOutput& out, const Args& args, const std::vector<ArmResult>& arms,
                   uint64_t rows, uint64_t table_offset, const std::vector<uint32_t>& row_ids,
                   const std::vector<uint32_t>& selected, const std::string& run_identity,
                   const std::string& direct_first_hash, bool parity) {
    std::ostringstream o;
    o << "{\"schema_version\":1,\"status\":\"complete\",\"created_utc\":\"" << utc_now()
      << "\",\"gpu_executed\":false,\"performance_claim\":false,\"input\":\"" << json_escape(args.input.string())
      << "\",\"format\":\"IQ4_NL\",\"rows\":" << rows << ",\"row_bytes\":" << kIq4NlRowBytes
      << ",\"row_count_source\":\"PleTable::rows checked against GGUF tensor shape\""
      << ",\"table_file_offset\":" << table_offset << ",\"run_input_identity\":\""
      << json_escape(run_identity) << "\",\"selected_seed\":" << kSeed
      << ",\"row_id_buffer_sha256\":\""
      << hetero_native_cpu::sha256_hex(row_ids.data(), row_ids.size() * sizeof(uint32_t))
      << "\",\"row_id_encoding\":\"native-endian uint32_t bytes; std::endian::native="
      << (std::endian::native == std::endian::little ? "little" : "big")
      << "\",\"row_id_generator\":\"std::mt19937_64 seed 42 + std::uniform_int_distribution<uint32_t>; distribution mapping may vary by standard library\""
      << ",\"direct_first_formal_sha256\":\"" << direct_first_hash << "\""
      << ",\"tokens\":" << kTokens << ",\"heads_per_token\":" << kRowsPerToken
      << ",\"warmups_per_batch\":" << kWarmups << ",\"formal_repeats_per_batch\":" << kRepeats
      << ",\"batch_sizes\":[1,2,6,96,512],\"selected_rows\":[";
    for (size_t i = 0; i < selected.size(); ++i) { if (i) o << ','; o << selected[i]; }
    o << "],\"lock_is_single_startup_observation\":true,\"parity\":" << (parity ? "true" : "false")
      << ",\"arms\":[";
    for (size_t i = 0; i < arms.size(); ++i) { if (i) o << ','; o << arm_json(arms[i]); }
    o << "]}\n";
    out.write_all(o.str());
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try { args = parse_args(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    if (args.help) { print_help(); return 0; }
    if (!args.run) {
        std::puts("{\"test\":\"strata-ple-table-parity\",\"mode\":\"metadata_only\",\"gpu_executed\":false,\"file_opened\":false}");
        return 0;
    }

    std::string phase = "validate_paths";
    std::string run_identity;
    std::unique_ptr<ExclusiveOutput> receipt;
    std::vector<ArmResult> arms;
    try {
        validate_paths(args.input, args.output);
        receipt = std::make_unique<ExclusiveOutput>(args.output);
        run_identity = input_identity(args.input);
        phase = "inspect_table_metadata";
        uint64_t table_offset = 0, nrows = 0;
        {
            strata::GgufFile gguf(args.input.string());
            const auto* tensor = gguf.find("per_layer_token_embd.weight");
            if (!tensor || tensor->shape.size() != 2 || tensor->shape[0] != PLE_HEAD_DIM)
                throw std::runtime_error("input has no expected PLE table tensor");
            table_offset = gguf.data_start() + tensor->offset;
            nrows = tensor->shape[1];
            if (strata::ggml_type_name(tensor->type) == nullptr ||
                std::string_view(strata::ggml_type_name(tensor->type)) != "IQ4_NL")
                throw std::runtime_error("input tensor is not IQ4_NL");
        }
        phase = "probe_ple_table_rows";
        uint64_t ple_table_rows = 0;
        {
            PleTable probe;
            PleIoOptions io;
            io.mode = PleIo::Direct;
            io.cache_rows = 0;
            std::string err;
            if (!probe.open(args.input.string(), err, io))
                throw std::runtime_error("PleTable metadata probe failed: " + err);
            ple_table_rows = probe.rows();
            probe.close();
            if (probe.is_open()) throw std::runtime_error("PleTable metadata probe did not close");
        }
        if (ple_table_rows != nrows) throw std::runtime_error("PleTable::rows disagrees with parsed GGUF tensor shape");
        nrows = ple_table_rows;
        if (input_identity(args.input) != run_identity)
            throw std::runtime_error("input identity changed during metadata inspection");
        if (nrows > UINT32_MAX || nrows < 2) throw std::runtime_error("unsupported PLE table row count");
        std::vector<uint32_t> selected;
        const std::vector<uint32_t> rows = select_rows(nrows, table_offset, selected);
        const uint64_t expected_lock_bytes = nrows * (uint64_t)kIq4NlRowBytes;

        phase = "direct_arm";
        arms.push_back(run_arm("direct", PleIo::Direct, false, args.input, rows, selected, nullptr,
                               expected_lock_bytes, run_identity));
        if (!arms.back().error.empty() || !arms.back().open_ok || arms.back().batches.size() != kBatchSizes.size())
            throw std::runtime_error("Direct arm failed: " + arms.back().error);
        std::vector<std::vector<float>> direct_reference;
        for (const auto& batch : arms.front().batches) {
            if (batch.outputs.empty()) throw std::runtime_error("Direct arm did not retain reference output");
            direct_reference.push_back(batch.outputs.front());
        }

        phase = "mmap_pageable_arm";
        arms.push_back(run_arm("mmap_pageable", PleIo::Mmap, false, args.input, rows, selected,
                               &direct_reference, expected_lock_bytes, run_identity));
        if (!arms.back().error.empty() || !arms.back().open_ok || arms.back().batches.size() != kBatchSizes.size())
            throw std::runtime_error("pageable mmap arm failed: " + arms.back().error);

        phase = "mmap_locked_arm";
        arms.push_back(run_arm("mmap_locked", PleIo::Mmap, true, args.input, rows, selected,
                               &direct_reference, expected_lock_bytes, run_identity));
        if (!arms.back().error.empty() || !arms.back().open_ok || arms.back().batches.size() != kBatchSizes.size())
            throw std::runtime_error("locked mmap arm failed: " + arms.back().error);

        const std::string direct_first_hash = arms.front().batches.front().hashes.front();
        bool parity = true;
        for (size_t ai = 0; ai < arms.size(); ++ai) {
            for (const auto& batch : arms[ai].batches) {
                if (!batch.error.empty() || batch.hashes.size() != kRepeats) parity = false;
                if (batch.warmup_sha256 != direct_first_hash) parity = false;
                for (const std::string& hash : batch.hashes) if (hash != direct_first_hash) parity = false;
                for (size_t diffs : batch.differing_floats) if (diffs != 0) parity = false;
                for (size_t finite : batch.finite_values)
                    if (finite != kTokens * kRowsPerToken * PLE_HEAD_DIM) parity = false;
            }
        }
        if (!parity) throw std::runtime_error("row/gather parity or finite gate failed");
        phase = "write_receipt";
        write_success(*receipt, args, arms, nrows, table_offset, rows, selected, run_identity,
                      direct_first_hash, parity);
        receipt->finish();
        return 0;
    } catch (const std::exception& e) {
        if (receipt && !receipt->write_started()) {
            try {
                receipt->write_all(failure_json(args, phase, e.what(), arms, run_identity));
                receipt->finish();
            }
            catch (...) { std::fprintf(stderr, "failure; unable to write failure receipt: %s\n", e.what()); }
        }
        std::fprintf(stderr, "%s: %s\n", phase.c_str(), e.what());
        return 1;
    }
}
