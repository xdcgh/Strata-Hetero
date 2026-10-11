// Persistent stdio worker for the native CPU expert. This TU reuses the benchmark's
// pinned native format, buffer, quantization, kernel, pool, and output helpers.
#define HETERO_NATIVE_NO_MAIN
#include "native_expert_bench.cpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <utility>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#ifndef HETERO_CPU_SERVICE_SOURCE_SHA256
#define HETERO_CPU_SERVICE_SOURCE_SHA256 "unknown"
#endif
#ifndef HETERO_CPU_SERVICE_CMAKE_SHA256
#define HETERO_CPU_SERVICE_CMAKE_SHA256 "unknown"
#endif
#ifndef HETERO_CPU_POOL_CPP_SHA256
#define HETERO_CPU_POOL_CPP_SHA256 "unknown"
#endif
#ifndef HETERO_CPU_POOL_HPP_SHA256
#define HETERO_CPU_POOL_HPP_SHA256 "unknown"
#endif

namespace {
constexpr uint16_t kInitKind = 1;
constexpr uint16_t kInferKind = 2;
constexpr uint16_t kResponseBit = 0x8000;
constexpr uint32_t kMaxFramePayload = 8u * 1024u * 1024u;
constexpr uint64_t kBlobBytes = 3072000;
constexpr size_t kIdentityMaxBytes = 1024u * 1024u;
constexpr size_t kInitMaxBytes = 64u * 1024u;
constexpr char kSchema[] = "strata-xpu-init-v1";

struct ServiceArgs {
    bool serve = false, help = false, validate_only = false;
    bool workers_set = false, affinity_set = false;
    fs::path blob, identity;
    std::string blob_sha, identity_sha, affinity = "none";
    int workers = 0;
};

bool lower_sha256(std::string_view s) {
    if (s.size() != 64) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

ServiceArgs parse_service_args(int argc, char** argv) {
    ServiceArgs a;
    std::set<std::string> seen;
    auto value = [&](int& i, const char* name) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + name);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        if (opt == "--help" || opt == "-h") { a.help = true; continue; }
        if (!seen.insert(opt).second) throw std::runtime_error("option supplied more than once: " + opt);
        if (opt == "--serve") a.serve = true;
        else if (opt == "--validate-only") a.validate_only = true;
        else if (opt == "--blob") a.blob = fs::u8path(value(i, "--blob"));
        else if (opt == "--blob-sha256") a.blob_sha = value(i, "--blob-sha256");
        else if (opt == "--identity") a.identity = fs::u8path(value(i, "--identity"));
        else if (opt == "--identity-sha256") a.identity_sha = value(i, "--identity-sha256");
        else if (opt == "--pool-workers") { a.workers = parse_int(value(i, "--pool-workers"), "--pool-workers"); a.workers_set = true; }
        else if (opt == "--pool-affinity") { a.affinity = value(i, "--pool-affinity"); a.affinity_set = true; }
        else throw std::runtime_error("unknown option: " + opt);
    }
    if (a.serve && a.validate_only) throw std::runtime_error("--serve and --validate-only are mutually exclusive");
    if (a.workers_set && (a.workers < 1 || a.workers > 64))
        throw std::runtime_error("--pool-workers must be in 1..64");
    if (a.affinity != "none" && a.affinity != "all" && a.affinity != "auto" && a.affinity != "p-cores")
        throw std::runtime_error("--pool-affinity must be none|all|auto|p-cores");
    if (a.affinity_set && !a.workers_set) throw std::runtime_error("--pool-affinity requires --pool-workers");
    if (a.serve) {
        if (a.blob.empty() || a.identity.empty() || !a.blob.is_absolute() || !a.identity.is_absolute())
            throw std::runtime_error("--serve requires absolute --blob and --identity paths");
        if (!lower_sha256(a.blob_sha) || !lower_sha256(a.identity_sha))
            throw std::runtime_error("--serve requires lowercase 64-character --blob-sha256 and --identity-sha256");
    } else if (!a.blob.empty() || !a.identity.empty() || !a.blob_sha.empty() || !a.identity_sha.empty()) {
        throw std::runtime_error("asset options require explicit --serve");
    }
    return a;
}

void service_help() {
    std::cout << "hetero_native_cpu_service [--validate-only | --serve] [options]\n"
                 "  Default/--validate-only: metadata only; no files opened and no pool created.\n"
                 "  --serve --blob ABS --blob-sha256 LOWER64 --identity ABS --identity-sha256 LOWER64\n"
                 "         [--pool-workers 1..64] [--pool-affinity none|all|auto|p-cores]\n";
}

void skip_ws(std::string_view s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) ++p;
}
std::string parse_json_string(std::string_view s, size_t& p) {
    if (p >= s.size() || s[p++] != '"') throw std::runtime_error("expected JSON string");
    std::string out;
    while (p < s.size() && s[p] != '"') {
        const unsigned char c = static_cast<unsigned char>(s[p++]);
        if (c < 0x20 || c >= 0x80 || c == '\\') throw std::runtime_error("INIT strings must use narrow ASCII without escapes");
        out.push_back(static_cast<char>(c));
    }
    if (p >= s.size()) throw std::runtime_error("unterminated JSON string");
    ++p;
    return out;
}

struct InitRequest { std::string nonce, identity_sha; int rows = 0; };
InitRequest parse_init_json(std::string_view s) {
    if (s.size() > kInitMaxBytes) throw std::runtime_error("INIT JSON exceeds 65536 bytes");
    std::map<std::string, std::string> strings;
    std::optional<int> rows;
    size_t p = 0; skip_ws(s, p);
    if (p >= s.size() || s[p++] != '{') throw std::runtime_error("INIT must be a JSON object");
    skip_ws(s, p);
    if (p < s.size() && s[p] == '}') { ++p; }
    else for (;;) {
        skip_ws(s, p); std::string key = parse_json_string(s, p); skip_ws(s, p);
        if (p >= s.size() || s[p++] != ':') throw std::runtime_error("expected ':' in INIT object");
        skip_ws(s, p);
        if (key == "rows") {
            if (rows) throw std::runtime_error("duplicate INIT key: rows");
            size_t begin = p;
            while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
            if (begin == p || (p - begin > 1 && s[begin] == '0')) throw std::runtime_error("rows must be a JSON integer");
            int parsed = parse_int(s.substr(begin, p - begin), "rows");
            if (parsed < 1 || parsed > 256) throw std::runtime_error("rows must be an integer in 1..256");
            rows = parsed;
        } else if (key == "schema" || key == "nonce" || key == "identity_sha256") {
            if (strings.contains(key)) throw std::runtime_error("duplicate INIT key: " + key);
            strings.emplace(key, parse_json_string(s, p));
        } else throw std::runtime_error("unknown INIT key: " + key);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') { ++p; continue; }
        if (p < s.size() && s[p] == '}') { ++p; break; }
        throw std::runtime_error("expected ',' or '}' in INIT object");
    }
    skip_ws(s, p);
    if (p != s.size()) throw std::runtime_error("trailing bytes after INIT JSON");
    if (strings.size() != 3 || !rows) throw std::runtime_error("INIT keys must be exactly schema, nonce, rows, identity_sha256");
    if (strings.at("schema") != kSchema) throw std::runtime_error("unsupported INIT schema");
    const std::string& nonce = strings.at("nonce");
    if (nonce.size() != 32 || !std::all_of(nonce.begin(), nonce.end(), [](unsigned char c) { return std::isxdigit(c) != 0; }))
        throw std::runtime_error("INIT nonce must be exactly 32 hexadecimal characters");
    const std::string& identity = strings.at("identity_sha256");
    if (!lower_sha256(identity)) throw std::runtime_error("INIT identity_sha256 must be lowercase 64-character SHA-256");
    return {nonce, identity, *rows};
}

bool read_exact(std::istream& in, uint8_t* out, size_t n, bool allow_clean_eof = false) {
    size_t done = 0;
    while (done < n) {
        in.read(reinterpret_cast<char*>(out + done), static_cast<std::streamsize>(n - done));
        const auto got = static_cast<size_t>(in.gcount());
        done += got;
        if (done == 0 && allow_clean_eof && in.eof()) return false;
        if (done < n && (in.eof() || in.fail())) throw std::runtime_error("truncated protocol stream");
        if (in.bad()) throw std::runtime_error("protocol stream read failure");
    }
    return true;
}
uint16_t get16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8)); }
uint32_t get32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
uint64_t get64(const uint8_t* p) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | p[i]; return v; }
void put16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }

struct RequestFrame { uint16_t kind; uint64_t id; std::vector<uint8_t> payload; };
std::optional<RequestFrame> read_frame(std::istream& in) {
    uint8_t h[20];
    if (!read_exact(in, h, sizeof(h), true)) return std::nullopt;
    if (std::memcmp(h, "SXPU", 4) != 0) throw std::runtime_error("bad frame magic");
    if (get16(h + 4) != 1) throw std::runtime_error("unsupported frame version");
    const uint16_t kind = get16(h + 6);
    if (kind & kResponseBit) throw std::runtime_error("request kind has response bit set");
    const uint32_t size = get32(h + 16);
    if (size > kMaxFramePayload) throw std::runtime_error("frame payload exceeds 8388608 byte limit");
    RequestFrame f{kind, get64(h + 8), std::vector<uint8_t>(size)};
    if (size) read_exact(in, f.payload.data(), size);
    return f;
}
void write_response(std::ostream& out, uint16_t kind, uint64_t id, uint32_t status, uint64_t wall_ns,
                    const std::vector<uint8_t>& body) {
    if (body.size() + 12 > kMaxFramePayload) throw std::runtime_error("response exceeds frame payload limit");
    uint8_t h[20]{}; std::memcpy(h, "SXPU", 4); put16(h + 4, 1); put16(h + 6, uint16_t(kind | kResponseBit));
    put64(h + 8, id); put32(h + 16, static_cast<uint32_t>(body.size() + 12));
    uint8_t prefix[12]; put32(prefix, status); put64(prefix + 4, wall_ns);
    out.write(reinterpret_cast<char*>(h), sizeof(h)); out.write(reinterpret_cast<char*>(prefix), sizeof(prefix));
    if (!body.empty()) out.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(body.size()));
    out.flush(); if (!out) throw std::runtime_error("protocol response write failed");
}
std::vector<uint8_t> bytes_of(std::string_view s) { return std::vector<uint8_t>(s.begin(), s.end()); }
std::vector<uint8_t> error_body(std::string_view message) { return bytes_of("{\"message\":" + json_quote(message.substr(0, 256)) + "}"); }

void validate_infer_payload(const std::vector<uint8_t>& payload, int rows) {
    const size_t expected = static_cast<size_t>(rows) * 2560u * 4u;
    if (payload.size() != expected) throw std::runtime_error("INFER payload size mismatch");
    for (size_t off = 0; off < payload.size(); off += 4) {
        uint32_t bits = get32(payload.data() + off); float value; std::memcpy(&value, &bits, 4);
        if (!std::isfinite(value)) throw std::runtime_error("INFER input contains NaN or infinity");
    }
}

void verify_fixed_file(const fs::path& path, uint64_t expected, const std::string& expected_sha, const char* label,
                       AlignedArray<uint8_t>* retained) {
    if (!path.is_absolute()) throw std::runtime_error(std::string(label) + " path must be absolute");
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) throw std::runtime_error(std::string(label) + " must be a regular file");
    if (fs::file_size(path, ec) != expected || ec) throw std::runtime_error(std::string(label) + " size mismatch");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + label);
    AlignedArray<uint8_t> data(static_cast<size_t>(expected));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (in.gcount() != static_cast<std::streamsize>(data.size())) throw std::runtime_error(std::string("short read for ") + label);
    char extra = 0; if (in.read(&extra, 1)) throw std::runtime_error(std::string(label) + " grew during read");
    if (!in.eof() && in.fail()) throw std::runtime_error(std::string("read error for ") + label);
    if (sha256(data.data(), data.size()) != expected_sha) throw std::runtime_error(std::string(label) + " SHA-256 mismatch");
    if (retained) *retained = std::move(data);
}

struct CpuService {
    ServiceArgs args;
    NativeFmt fmt{};
    // Declaration order is intentional: pool joins before host affinity is restored, then
    // jobs/buffers/blob are released.
    AlignedArray<uint8_t> blob{static_cast<size_t>(kBlobBytes)};
    WorkBuffers buffers;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::unique_ptr<ScopedHostPin> host_pin;
    std::unique_ptr<cpu::ExpertPool> pool;
    PoolRunInfo pool_info;
    bool initialized = false;
    bool reference_environment_set = false;
    int rows = 0;
    uint64_t identity_bytes = 0;
    std::string nonce;
    std::string identity_sha;

    explicit CpuService(ServiceArgs a) : args(std::move(a)) {
        set_process_reference_environment(false);
        reference_environment_set = true;
        Args native; native.rows = 1; native.hidden = 2560; native.intermediate = 640;
        native.gu_type = GGML_TYPE_Q4_K; native.down_type = GGML_TYPE_Q5_1;
        fmt = make_format(native);
        verify_fixed_file(args.blob, kBlobBytes, args.blob_sha, "native blob", &blob);
        if (!args.identity.is_absolute()) throw std::runtime_error("identity path must be absolute");
        std::error_code ec;
        if (!fs::is_regular_file(args.identity, ec) || ec) throw std::runtime_error("identity must be a regular file");
        const auto n = fs::file_size(args.identity, ec);
        if (ec || n > kIdentityMaxBytes) throw std::runtime_error("identity must be at most 1048576 bytes");
        identity_bytes = n;
        std::ifstream in(args.identity, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open identity file");
        std::vector<uint8_t> id(static_cast<size_t>(n));
        if (n) in.read(reinterpret_cast<char*>(id.data()), static_cast<std::streamsize>(n));
        if (in.gcount() != static_cast<std::streamsize>(n)) throw std::runtime_error("short identity read");
        char extra = 0; if (in.read(&extra, 1)) throw std::runtime_error("identity grew during read");
        if (!in.eof() && in.fail()) throw std::runtime_error("identity read error");
        identity_sha = sha256(id.data(), id.size());
        if (identity_sha != args.identity_sha) throw std::runtime_error("identity SHA-256 mismatch");
        if (args.workers) setup_pool();
    }
    void setup_pool() {
        pool_info.workers = args.workers; pool_info.pin = args.affinity != "none";
        pool_info.affinity = pool_affinity_for_service();
        pool_info.topology = cpu::detect_cpu_topology(true, pool_info.affinity);
        if (pool_info.affinity == cpu::PoolAffinity::PCores && args.workers > static_cast<int>(pool_info.topology.worker_cores.size()))
            throw std::runtime_error("p-cores pool worker count exceeds eligible worker IDs");
        pool_info.unpinned_overflow = pool_info.pin ? (std::max)(0, args.workers - static_cast<int>(pool_info.topology.worker_cores.size())) : 0;
        if (pool_info.pin) { host_pin = std::make_unique<ScopedHostPin>(pool_info.topology.host_core); pool_info.host_pinned = true; }
        pool = std::make_unique<cpu::ExpertPool>(args.workers, pool_info.pin, true, pool_info.affinity);
        pool_info.actual_workers = pool->workers();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (;;) {
            const auto reports = pool->worker_affinity();
            if (std::all_of(reports.begin(), reports.end(), [](const auto& r) { return r.ready; })) break;
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("CPU pool worker startup readback timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    cpu::PoolAffinity pool_affinity_for_service() const {
        if (args.affinity == "auto") return cpu::PoolAffinity::Auto;
        if (args.affinity == "p-cores") return cpu::PoolAffinity::PCores;
        return cpu::PoolAffinity::All;
    }
    void clear_rows() {
        jobs.clear(); buffers = WorkBuffers{}; rows = 0; initialized = false;
    }
    void configure_rows(int new_rows) {
        if (initialized && rows == new_rows) return;
        clear_rows();
        Args a; a.rows = new_rows; a.hidden = 2560; a.intermediate = 640;
        a.gu_type = GGML_TYPE_Q4_K; a.down_type = GGML_TYPE_Q5_1;
        const size_t input_size = static_cast<size_t>(new_rows) * 2560u * 4u;
        std::vector<uint8_t> zero(input_size, 0);
        buffers = allocate_buffers(a, fmt, zero.data());
        if (pool) jobs = pool_jobs(blob.data(), buffers);
        rows = new_rows; initialized = true;
    }
    std::string ready_json(const InitRequest& req, bool cached_rows_reused) const {
        std::ostringstream s;
        s << "{\"schema\":\"strata-xpu-ready-v1\",\"status\":\"ready\",\"rows\":" << rows
          << ",\"nonce\":" << json_quote(req.nonce) << ",\"identity_sha256\":" << json_quote(identity_sha)
          << ",\"cached_rows_reused\":" << (cached_rows_reused ? "true" : "false")
          << ",\"hidden\":2560,\"intermediate\":640,\"gu_type\":\"GGML_Q4_K\",\"down_type\":\"GGML_Q5_1\""
          << ",\"operator\":\"native-q4k-q5_1\",\"operator_signature\":\"Q4_K_Q8_K_SILU_Q5_1_Q8_1\""
          << ",\"native_blob_loaded_and_hash_verified\":true"
          << ",\"blob_bytes\":" << blob.size() << ",\"blob_sha256\":" << json_quote(args.blob_sha)
          << ",\"identity_bytes\":" << identity_bytes
          << ",\"identity_bytes_sha256\":" << json_quote(identity_sha)
          << ",\"native_source_sha256\":" << json_quote(HETERO_NATIVE_SOURCE_SHA256)
          << ",\"service_source_sha256\":" << json_quote(HETERO_CPU_SERVICE_SOURCE_SHA256)
          << ",\"service_cmake_sha256\":" << json_quote(HETERO_CPU_SERVICE_CMAKE_SHA256)
          << ",\"pool_cpp_sha256\":" << json_quote(HETERO_CPU_POOL_CPP_SHA256)
          << ",\"pool_hpp_sha256\":" << json_quote(HETERO_CPU_POOL_HPP_SHA256)
          << ",\"pool\":{";
        s << "\"requested_workers\":" << args.workers << ",\"actual_workers\":" << (pool ? pool->workers() : 0)
          << ",\"affinity\":" << json_quote(args.affinity) << ",\"host_pinned\":" << (pool_info.host_pinned ? "true" : "false")
          << ",\"host_core\":" << pool_info.topology.host_core << ",\"eligible_workers\":" << pool_info.topology.worker_cores.size()
          << ",\"worker_affinity_readback\":[";
        if (pool) {
            auto reports = pool->worker_affinity();
            for (size_t i = 0; i < reports.size(); ++i) {
                if (i) s << ',';
                const auto& r = reports[i];
                s << "{\"worker\":" << r.worker << ",\"requested_core\":" << r.requested_core
                  << ",\"ready\":" << (r.ready ? "true" : "false")
                  << ",\"pin_requested\":" << (r.pin_requested ? "true" : "false")
                  << ",\"pin_applied\":" << (r.pin_applied ? "true" : "false")
                  << ",\"mask_observed\":" << (r.mask_observed ? "true" : "false")
                  << ",\"mask_matches_request\":" << (r.mask_matches_request ? "true" : "false")
                  << ",\"observed_group\":" << r.observed_group << ",\"observed_mask\":" << r.observed_mask
                  << ",\"startup_processor\":" << r.startup_processor << '}';
            }
        }
        s << "]}}";
        return s.str();
    }
    std::vector<uint8_t> infer(const std::vector<uint8_t>& payload, uint64_t& wall) {
        if (!initialized) throw std::runtime_error("INFER received before a successful INIT");
        // Arc rejects non-finite input before starting worker timing.
        validate_infer_payload(payload, rows);
        Args a; a.rows = rows; a.hidden = 2560; a.intermediate = 640;
        a.gu_type = GGML_TYPE_Q4_K; a.down_type = GGML_TYPE_Q5_1;
        if (!reference_environment_set) { set_process_reference_environment(false); reference_environment_set = true; }
        const auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < rows; ++r) for (int c = 0; c < 2560; ++c) {
            const size_t off = (static_cast<size_t>(r) * 2560u + c) * 4u;
            uint32_t bits = get32(payload.data() + off); std::memcpy(&buffers.input[r].data()[c], &bits, 4);
        }
        if (pool) (void)pool_iteration(fmt, buffers, *pool, jobs, pool_info, false);
        else one_iteration(a, fmt, blob.data(), buffers);
        std::vector<uint8_t> output = encode_output_le(a, buffers);
        wall = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        validate_finite_outputs(a, buffers); // matches Arc: output validation follows the timed worker section
        return output;
    }
};

int service_loop(CpuService& service, std::istream& in, std::ostream& out) {
    for (;;) {
        std::optional<RequestFrame> frame;
        try { frame = read_frame(in); }
        catch (const std::exception& e) { std::cerr << "hetero_native_cpu_service: protocol error: " << e.what() << '\n'; return 2; }
        if (!frame) return 0;
        const auto began = std::chrono::steady_clock::now();
        uint32_t status = 0; uint64_t worker_ns = 0; std::vector<uint8_t> body;
        try {
            if (frame->kind == kInitKind) {
                service.nonce.clear();
                InitRequest req = parse_init_json(std::string_view(reinterpret_cast<const char*>(frame->payload.data()), frame->payload.size()));
                if (req.identity_sha != service.identity_sha) throw std::runtime_error("INIT identity_sha256 does not match fixed identity file");
                const bool reused = service.initialized && service.rows == req.rows;
                service.configure_rows(req.rows); service.nonce = req.nonce;
                body = bytes_of(service.ready_json(req, reused));
                worker_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
            } else if (frame->kind == kInferKind) {
                uint64_t infer_ns = 0; body = service.infer(frame->payload, infer_ns); worker_ns = infer_ns;
            } else throw std::runtime_error("unknown operation kind");
        } catch (const std::exception& e) {
            status = 1; body = error_body(e.what());
            if (frame->kind == kInitKind) { service.clear_rows(); service.nonce.clear(); }
            worker_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
        }
        try { write_response(out, frame->kind, frame->id, status, worker_ns, body); }
        catch (const std::exception& e) { std::cerr << "hetero_native_cpu_service: response write failed: " << e.what() << '\n'; return 2; }
    }
}

std::string metadata_only_json() {
    Args a; a.rows = 1; a.hidden = 2560; a.intermediate = 640; a.gu_type = GGML_TYPE_Q4_K; a.down_type = GGML_TYPE_Q5_1;
    NativeFmt f = make_format(a);
    std::ostringstream s;
    s << "{\"schema\":\"strata-hetero-native-cpu-service-metadata-v1\",\"mode\":\"validate_only\",\"rows_min\":1,\"rows_max\":256"
      << ",\"hidden\":2560,\"intermediate\":640,\"gu_type\":" << f.gu_type << ",\"down_type\":" << f.d_type
      << ",\"blob_bytes\":" << f.bytes << ",\"files_opened\":false,\"pool_created\":false,\"kernel_called\":false}\n";
    return s.str();
}
} // namespace

#ifndef HETERO_NATIVE_CPU_SERVICE_NO_MAIN
int main(int argc, char** argv) {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        ServiceArgs args = parse_service_args(argc, argv);
        if (args.help) { service_help(); return 0; }
        if (!args.serve) { std::cout << metadata_only_json(); return 0; }
        CpuService service(std::move(args));
        return service_loop(service, std::cin, std::cout);
    } catch (const std::exception& e) {
        std::cerr << "hetero_native_cpu_service: " << e.what() << '\n'; return 2;
    } catch (...) {
        std::cerr << "hetero_native_cpu_service: unknown non-standard exception\n"; return 2;
    }
}
#endif
