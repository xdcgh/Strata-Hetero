#define HETERO_NATIVE_CPU_SERVICE_NO_MAIN
#include "native_cpu_service.cpp"

#include <sstream>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void synthetic_identity_handshake() {
    const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir = fs::current_path() / ("cpu-identity-fixture-" + std::to_string(serial));
    require(fs::create_directory(dir), "fixture directory already exists");
    const fs::path blob = dir / "synthetic-zero-blob.bin";
    const fs::path identity = dir / "synthetic-identity.json";
    auto cleanup = [&] { fs::remove(blob); fs::remove(identity); fs::remove(dir); };
    try {
        const std::vector<uint8_t> zero_blob(static_cast<size_t>(kBlobBytes), 0);
        const std::string id = "{\"fixture_only\":true,\"model_loaded\":false}\n";
        write_new_file(blob, zero_blob.data(), zero_blob.size(), "synthetic blob");
        write_new_file(identity, id.data(), id.size(), "synthetic identity");
        ServiceArgs args;
        args.serve = true; args.blob = blob; args.identity = identity;
        args.blob_sha = sha256(zero_blob.data(), zero_blob.size());
        args.identity_sha = sha256(id.data(), id.size());
        CpuService service(args); // no pool, inference, real weights, or device
        require(service.identity_sha == args.identity_sha, "verified identity was not retained for INIT");
        auto request = [&](const std::string& digest) {
            const std::string payload = "{\"schema\":\"strata-xpu-init-v1\",\"nonce\":\"0123456789abcdef0123456789abcdef\",\"rows\":1,\"identity_sha256\":\"" + digest + "\"}";
            std::string frame(20 + payload.size(), '\0');
            auto* bytes = reinterpret_cast<uint8_t*>(frame.data());
            std::memcpy(bytes, "SXPU", 4); put16(bytes+4, 1); put16(bytes+6, 1);
            put64(bytes+8, 7); put32(bytes+16, static_cast<uint32_t>(payload.size()));
            std::memcpy(bytes+20, payload.data(), payload.size());
            std::istringstream input(frame, std::ios::in | std::ios::binary);
            std::ostringstream output(std::ios::out | std::ios::binary);
            require(service_loop(service, input, output) == 0, "INIT fixture failed framing/EOF");
            const std::string result = output.str();
            require(result.size() >= 32 && result.substr(0, 4) == "SXPU", "INIT response missing");
            return get32(reinterpret_cast<const uint8_t*>(result.data()) + 20);
        };
        require(request(args.identity_sha) == 0 && service.initialized, "verified identity INIT rejected");
        require(request(std::string(64, '0')) == 1 && !service.initialized, "wrong identity did not clear ready state");
        require(request(args.identity_sha) == 0 && service.initialized, "valid INIT after rejection failed");
        cleanup();
    } catch (...) { cleanup(); throw; }
}

int main() {
    const std::string init = R"({ "identity_sha256" : "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "rows": 2, "nonce":"0123456789abcdef0123456789abcdef", "schema":"strata-xpu-init-v1" })";
    const InitRequest parsed = parse_init_json(init);
    require(parsed.rows == 2 && parsed.nonce == "0123456789abcdef0123456789abcdef", "INIT parse mismatch");
    require(parsed.identity_sha == std::string(64, 'a'), "identity SHA parse mismatch");

    bool duplicate_rejected = false;
    try {
        (void)parse_init_json(R"({"schema":"strata-xpu-init-v1","schema":"strata-xpu-init-v1","nonce":"0123456789abcdef0123456789abcdef","rows":1,"identity_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})");
    } catch (const std::exception&) { duplicate_rejected = true; }
    require(duplicate_rejected, "duplicate INIT key accepted");

    std::string encoded("SXPU", 4);
    encoded.push_back(1); encoded.push_back(0); // version
    encoded.push_back(2); encoded.push_back(0); // kind
    encoded.append("\x08\x07\x06\x05\x04\x03\x02\x01", 8); // request id LE
    encoded.append("\x04\0\0\0", 4); encoded.append("data", 4);
    std::istringstream framed(encoded, std::ios::in | std::ios::binary);
    auto frame = read_frame(framed);
    require(frame && frame->kind == 2 && frame->id == 0x0102030405060708ull, "SXPU header endian mismatch");
    require(std::string(frame->payload.begin(), frame->payload.end()) == "data", "SXPU payload mismatch");
    std::istringstream clean_eof(std::string{}, std::ios::in | std::ios::binary);
    require(!read_frame(clean_eof), "clean EOF was not accepted");

    std::ostringstream response(std::ios::out | std::ios::binary);
    write_response(response, 1, 0x0102030405060708ull, 0x01020304u, 0x0807060504030201ull, bytes_of("x"));
    const std::string response_bytes = response.str();
    require(response_bytes.size() == 33 && get16(reinterpret_cast<const uint8_t*>(response_bytes.data()) + 6) == 0x8001,
            "response frame header mismatch");
    const auto* result_prefix = reinterpret_cast<const uint8_t*>(response_bytes.data()) + 20;
    require(get32(result_prefix) == 0x01020304u && get64(result_prefix + 4) == 0x0807060504030201ull,
            "status/worker wall prefix is not little-endian <IQ>");

    std::string oversized("SXPU", 4);
    oversized.push_back(1); oversized.push_back(0); oversized.push_back(2); oversized.push_back(0);
    oversized.append(8, '\0'); oversized.push_back(1); oversized.push_back(0); oversized.push_back(128); oversized.push_back(0);
    std::istringstream oversized_stream(oversized, std::ios::in | std::ios::binary);
    bool size_rejected = false;
    try { (void)read_frame(oversized_stream); } catch (const std::exception&) { size_rejected = true; }
    require(size_rejected, "oversized frame accepted");

    std::vector<uint8_t> valid(2560u * 4u, 0);
    validate_infer_payload(valid, 1);
    valid[3] = 0x7f; valid[2] = 0x80; // +infinity in little-endian F32
    bool nonfinite_rejected = false;
    try { validate_infer_payload(valid, 1); } catch (const std::exception&) { nonfinite_rejected = true; }
    require(nonfinite_rejected, "non-finite INFER payload accepted");
    synthetic_identity_handshake();
    return 0;
}
