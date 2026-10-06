#define STRATA_STORAGE_PROFILE_TEST
#include "storage_profile.cpp"

#include <array>
#include <iostream>

namespace {
using namespace strata::hetero::storage;
int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } } while (0)

class FakeClock final : public ClockSource {
public:
    double wall = 0;
    double cpu = 0;
    double wall_us() override { return wall; }
    double cpu_us() override { cpu += 4; return cpu; }
    void advance(double us) { wall += us; }
};

class FakeReader final : public Reader {
public:
    enum class Mode { success, fail, stall } mode = Mode::success;
    struct Pending { uint64_t tag; uint32_t bytes; };
    std::vector<Pending> pending;
    std::vector<uint64_t> offsets;
    int queued = 0, max_queued = 0;
    double submit_delay_us = 0;
    double wait_delay_us = 1000;
    bool submit(uint64_t offset, void*, uint32_t bytes, uint64_t tag, std::string&) override {
        offsets.push_back(offset);
        pending.push_back({tag, bytes});
        ++queued; max_queued = std::max(max_queued, queued);
        if (fake_clock) fake_clock->advance(submit_delay_us);
        return true;
    }
    int wait(strata::platform::Completion* out, int max, int) override {
        if (mode == Mode::stall || pending.empty()) { fake_clock->advance(10000); return 0; }
        const int n = std::min<int>(max, (int) pending.size());
        for (int i = 0; i < n; ++i) {
            const Pending p = pending[(size_t)i];
            out[i] = {p.tag, mode == Mode::success ? p.bytes : 0, mode == Mode::success};
        }
        pending.erase(pending.begin(), pending.begin() + n);
        queued -= n;
        fake_clock->advance(wait_delay_us);
        return n;
    }
    FakeClock* fake_clock = nullptr;
};

bool make_fixture(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    std::array<uint8_t, 65536> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t)((i * 37 + i / 251) & 0xff);
    out.write((const char*)bytes.data(), (std::streamsize)bytes.size());
    return (bool)out;
}

bool wait_one(DirectReader& reader, strata::platform::Completion& completion) {
    for (int tries = 0; tries < 50; ++tries) {
        const int n = reader.wait(&completion, 1, 100);
        if (n == 1) return true;
        if (n < 0) return false;
    }
    return false;
}

bool directfile_fixture_checks(const std::filesystem::path& file) {
    DirectReader reader;
    std::string error;
    if (!reader.open(file, error)) { CHECK(false, "DirectFile open fixture"); return false; }
    auto* buffer = (uint8_t*)strata::platform::DirectFile::alloc_aligned(16384);
    if (!buffer) { CHECK(false, "aligned fixture buffer"); return false; }
    const uint64_t file_bytes = reader.size();
    CHECK(file_bytes == 65536, "fixture file is exactly 64 KiB");
    CHECK(!reader.submit(1, buffer, 4096, 1, error), "misaligned offset is rejected before queueing");

    CHECK(reader.submit(49152, buffer, 16384, 2, error), "last full 16 KiB block submits");
    strata::platform::Completion c{};
    if (!wait_one(reader, c)) { std::cerr << "DirectFile fixture completion timed out; exiting before freeing a possibly in-flight buffer\n"; std::fflush(nullptr); std::_Exit(1); }
    CHECK(c.tag == 2 && c.ok && c.bytes == 16384, "last full block has exact completion counts");
    bool bytes_match = true;
    for (size_t i = 0; i < c.bytes; ++i)
        bytes_match = bytes_match && buffer[i] == (uint8_t)(((49152 + i) * 37 + (49152 + i) / 251) & 0xff);
    CHECK(bytes_match, "synthetic last-block bytes match expected file content");

    CHECK(reader.submit(file_bytes, buffer, 4096, 3, error), "aligned read at EOF is accepted by DirectFile");
    if (!wait_one(reader, c)) { std::cerr << "DirectFile EOF completion timed out; exiting before freeing a possibly in-flight buffer\n"; std::fflush(nullptr); std::_Exit(1); }
    CHECK(c.tag == 3 && c.ok && c.bytes == 0, "read at EOF reports a successful zero-byte completion");
    strata::platform::DirectFile::free_aligned(buffer);
    return true;
}

bool scheduler_checks(uint64_t size) {
    Options o; o.block_bytes = 4096; o.queue_depth = 4; o.duration_ms = 3; o.warmup_ms = 0; o.repeats = 3;
    FakeClock clock;
    FakeReader reader; reader.fake_clock = &clock;
    RunStats result = run_profile(reader, clock, o, size);
    CHECK(result.passed && result.repeats.size() == 3, "fixture scheduler completes three measured repeats");
    CHECK(reader.max_queued <= (int)o.queue_depth, "queue depth never exceeds requested cap");
    for (const auto& r : result.repeats) {
        CHECK(r.measured.max_qd <= o.queue_depth, "recorded actual QD stays within requested QD");
        CHECK(r.measured.successful > 0 && !r.measured.latency_ms.empty(), "successful reads have measured latencies");
    }
    std::mt19937_64 a(88), b(88); uint64_t ai = 0, bi = 0;
    for (int i = 0; i < 10; ++i) CHECK(next_offset(o, size, ai, a) == next_offset(o, size, bi, b), "seeded random offsets replay exactly");

    Options latency_o; latency_o.block_bytes = 4096; latency_o.queue_depth = 1;
    latency_o.duration_ms = 1; latency_o.warmup_ms = 0; latency_o.pattern = "sequential";
    FakeClock latency_clock; FakeReader latency_reader; latency_reader.fake_clock = &latency_clock;
    latency_reader.submit_delay_us = 150; latency_reader.wait_delay_us = 50;
    PhaseStats latency = run_phase(latency_reader, latency_clock, latency_o, size, 7, 1, true);
    CHECK(latency.latency_count_total > 0 && !latency.latency_ms.empty(), "latency fixture produced samples");
    CHECK(latency.latency_ms.front() >= 0.20, "submit-call delay is included in submit-to-completion-observed latency");

    Options sampled_o; sampled_o.block_bytes = 4096; sampled_o.queue_depth = 64;
    sampled_o.duration_ms = 1100; sampled_o.warmup_ms = 0; sampled_o.pattern = "random"; sampled_o.seed = 9123;
    FakeClock sampled_clock_a; FakeReader sampled_reader_a; sampled_reader_a.fake_clock = &sampled_clock_a;
    PhaseStats sampled_a = run_phase(sampled_reader_a, sampled_clock_a, sampled_o, size, sampled_o.seed, sampled_o.duration_ms, true);
    FakeClock sampled_clock_b; FakeReader sampled_reader_b; sampled_reader_b.fake_clock = &sampled_clock_b;
    PhaseStats sampled_b = run_phase(sampled_reader_b, sampled_clock_b, sampled_o, size, sampled_o.seed, sampled_o.duration_ms, false);
    CHECK(sampled_a.latency_count_total > kLatencySampleCap, "long synthetic run exceeds latency reservoir cap");
    CHECK(sampled_a.latency_ms.size() == kLatencySampleCap, "latency reservoir remains bounded");
    CHECK(sampled_a.successful == sampled_b.successful && sampled_b.latency_count_total == 0 &&
          sampled_reader_a.offsets == sampled_reader_b.offsets,
          "latency sampling RNG does not alter the seeded read offset trace");

    FakeClock failed_clock; FakeReader failed; failed.fake_clock = &failed_clock; failed.mode = FakeReader::Mode::fail;
    PhaseStats bad = run_phase(failed, failed_clock, o, size, 2, 1, true);
    CHECK(bad.failed > 0 && bad.successful == 0 && bad.latency_ms.empty(), "failed I/O is not reported as success or latency");
    const std::string failure_json = phase_json(bad, 1, o.queue_depth);
    CHECK(failure_json.find("\"iops\":null") != std::string::npos &&
          failure_json.find("\"bandwidth_mib_s\":null") != std::string::npos,
          "failed reads have unavailable performance, not a fake zero-speed disk profile");

    FakeClock stuck_clock; FakeReader stuck; stuck.fake_clock = &stuck_clock; stuck.mode = FakeReader::Mode::stall;
    PhaseStats timeout = run_phase(stuck, stuck_clock, o, size, 2, 1, true);
    CHECK(timeout.drain_timeout && timeout.uncompleted > 0 && !timeout.retained_buffers.empty(), "bounded drain timeout preserves pending buffers and reports uncompleted count");
    for (void* p : timeout.retained_buffers) strata::platform::DirectFile::free_aligned(p); // fake reader has no real I/O
    return true;
}

bool cli_and_output_checks(const std::filesystem::path& file, const std::filesystem::path& dir) {
    const auto output = dir / "validate_should_not_exist.json";
    std::string file_s = file.string(), output_s = output.string();
    std::array<char*, 6> argv = {const_cast<char*>("storage-profile-test"), const_cast<char*>("--file"),
        file_s.data(), const_cast<char*>("--output"), output_s.data(), const_cast<char*>("--validate-only")};
    int factory_calls = 0;
    const int rc = run_cli((int)argv.size(), argv.data(), [&]() -> DirectReader* { ++factory_calls; return new DirectReader(); });
    CHECK(rc == 0 && factory_calls == 0 && !std::filesystem::exists(output), "validate-only performs no reader open/read and writes no output");

    const auto protected_path = dir / "overwrite_guard.json";
    { std::ofstream f(protected_path); f << "keep"; }
    std::string write_error;
    CHECK(!write_exclusive(protected_path, "replacement", write_error), "existing output cannot be overwritten");
    std::ifstream f(protected_path); std::string contents; f >> contents;
    CHECK(contents == "keep", "overwrite guard preserves prior file bytes");

    Options tail_options; tail_options.file = dir / "tail_file.bin"; tail_options.block_bytes = 16384;
    { std::ofstream tail(tail_options.file, std::ios::binary); tail.seekp(65536 + 127 - 1); tail.put('\0'); }
    Preflight tail_preflight; std::string error;
    CHECK(preflight(tail_options, tail_preflight, error), "file with an incomplete tail passes preflight");
    CHECK(tail_preflight.readable_bytes == 65536 && tail_preflight.file_bytes - tail_preflight.readable_bytes == 127,
          "readable region excludes a partial block tail");
    tail_options.pattern = "sequential";
    std::mt19937_64 rng(1); uint64_t sequential = 0;
    for (int i = 0; i < 8; ++i) {
        const uint64_t offset = next_offset(tail_options, tail_preflight.readable_bytes, sequential, rng);
        CHECK(offset + tail_options.block_bytes <= tail_preflight.readable_bytes, "planner never requests the incomplete tail");
    }

    const auto small_path = dir / "short_file.bin";
    { std::ofstream small(small_path, std::ios::binary); small.seekp(4094); small.put('\0'); }
    Options small_options; small_options.file = small_path; small_options.block_bytes = 4096;
    Preflight small_preflight;
    CHECK(!preflight(small_options, small_preflight, error), "file shorter than one full block is rejected before open/read");

    std::array<char*, 3> overflow_argv = {const_cast<char*>("storage-profile-test"), const_cast<char*>("--seed"),
                                          const_cast<char*>("18446744073709551616")};
    Options parsed; std::string parse_error;
    CHECK(!parse_args((int)overflow_argv.size(), overflow_argv.data(), parsed, parse_error),
          "unsigned numeric overflow is rejected by full-range parsing");
    return true;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--dir") {
        std::cerr << "usage: storage_profile_test --dir NEW_TEST_DIRECTORY\n"; return 2;
    }
    const std::filesystem::path dir(argv[2]);
    std::error_code ec;
    if (!std::filesystem::create_directory(dir, ec) || ec) {
        std::cerr << "fixture directory must be new; refusing reuse\n"; return 2;
    }
    const auto owned_root = std::filesystem::canonical(dir, ec);
    if (ec) { std::cerr << "cannot canonicalize newly created fixture directory\n"; return 2; }
    const auto file = dir / "synthetic_64KiB.bin";
    if (!make_fixture(file)) { std::cerr << "cannot create synthetic fixture\n"; return 2; }
    directfile_fixture_checks(file);
    scheduler_checks(65536);
    cli_and_output_checks(file, dir);
    const auto now_root = std::filesystem::canonical(dir, ec);
    if (ec || now_root != owned_root) {
        std::cerr << "fixture directory identity changed; preserving contents\n"; ++failures;
    } else {
        for (const char* name : {"synthetic_64KiB.bin", "validate_should_not_exist.json", "overwrite_guard.json",
                                 "tail_file.bin", "short_file.bin"}) {
            const auto child = dir / name;
            if (!std::filesystem::exists(child)) continue;
            if (std::filesystem::canonical(child.parent_path(), ec) != owned_root || ec) {
                std::cerr << "fixture child escaped owned directory; preserving contents\n"; ++failures; break;
            }
            if (!std::filesystem::remove(child, ec) || ec) { std::cerr << "fixture file cleanup failed\n"; ++failures; }
        }
        if (failures == 0 && !std::filesystem::remove(dir, ec)) { std::cerr << "fixture directory not empty; preserving it\n"; ++failures; }
        if (ec) { std::cerr << "fixture directory cleanup failed\n"; ++failures; }
    }
    if (failures) { std::cerr << failures << " storage fixture check(s) failed\n"; return 1; }
    std::cout << "storage profile fixture checks passed\n";
    return 0;
}
