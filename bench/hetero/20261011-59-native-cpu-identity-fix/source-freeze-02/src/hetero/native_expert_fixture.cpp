// Tiny synthetic metadata fixture: validates the real ggml/native layout API without reading weights or running dots.
#include "strata/kernels/cpu/native_expert.hpp"
#include "sha256.hpp"

#include <iostream>
#include <string>

int main() {
    using strata::kernels::cpu::NativeFmt;
    using strata::kernels::cpu::native_fmt;

    if (hetero_native_cpu::sha256_hex("abc", 3) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        std::cerr << "fixture: SHA-256 known-vector mismatch\n";
        return 1;
    }
    if (hetero_native_cpu::sha256_hex("", 0) !=
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        std::cerr << "fixture: empty SHA-256 known-vector mismatch\n";
        return 6;
    }
    constexpr char kLongMessage[] =
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    hetero_native_cpu::Sha256 chunked;
    chunked.update(kLongMessage, 17);
    chunked.update(kLongMessage + 17, sizeof(kLongMessage) - 1 - 17);
    if (hetero_native_cpu::sha256_hex(kLongMessage, sizeof(kLongMessage) - 1) !=
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" ||
        chunked.finish() != "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1") {
        std::cerr << "fixture: multi-block/chunked SHA-256 known-vector mismatch\n";
        return 7;
    }

    NativeFmt fmt{};
    std::string error;
    if (!native_fmt(12, 7, 2560, 640, fmt, error)) {
        std::cerr << "fixture: expected Q4_K/Q5_1 geometry rejected: " << error << '\n';
        return 2;
    }
    if (fmt.gu_act != 15 || fmt.d_act != 9 || fmt.gu_row != 1440 || fmt.d_row != 480 ||
        fmt.up_off != 921600 || fmt.down_off != 1843200 || fmt.bytes != 3072000 ||
        fmt.act_bytes != 2920 || fmt.h_bytes != 720) {
        std::cerr << "fixture: unexpected Q4_K/Q5_1 native layout\n";
        return 3;
    }

    NativeFmt invalid{};
    error.clear();
    if (native_fmt(12, 7, 2570, 640, invalid, error)) {
        std::cerr << "fixture: non-block-aligned hidden width was accepted\n";
        return 4;
    }
    std::cout << "{\"status\":\"fixture_pass\",\"native_kernel_called\":false,\"blob_bytes\":3072000}\n";
    return 0;
}
