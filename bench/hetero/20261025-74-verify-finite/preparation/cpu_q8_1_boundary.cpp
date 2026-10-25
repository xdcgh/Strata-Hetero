// Candidate boundary only: real pinned ggml CPU quantizer/dot, no model input.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <array>
#include <cmath>
#include <cstdio>
int main() {
    ggml_cpu_init();
    const auto* q = ggml_get_type_traits_cpu(GGML_TYPE_Q8_1);
    const auto* d = ggml_get_type_traits_cpu(GGML_TYPE_Q5_1);
    if (!q || !q->from_float || !d || !d->vec_dot) return 2;
    std::array<float,32> zeros{}, h{};
    block_q5_1 w{}; quantize_row_q5_1_ref(zeros.data(), &w, 32);
    int failures=0;
    for (float value : {1.0f, 2047.0f, 2048.0f}) {
        h.fill(value); block_q8_1 b{}; q->from_float(h.data(), &b, 32);
        float out=0; d->vec_dot(32, &out, 0, &w, 0, &b, 0, 1);
        const float scale=ggml_fp16_to_fp32(b.d), sum=ggml_fp16_to_fp32(b.s);
        std::printf("{\"value\":%.9g,\"d\":%.9g,\"sum_finite\":%s,\"out_finite\":%s}\n",
                    value,scale,std::isfinite(sum)?"true":"false",std::isfinite(out)?"true":"false");
        if (value < 2048 && (!std::isfinite(sum) || !std::isfinite(out))) ++failures;
        if (value == 2048 && (std::isfinite(sum) || std::isfinite(out))) ++failures;
    }
    h.fill(1024.0f); block_q8_1 scaled{}; q->from_float(h.data(), &scaled, 32);
    float out=0; d->vec_dot(32,&out,0,&w,0,&scaled,0,1); out*=2.0f;
    if (!std::isfinite(out) || out!=0.0f) ++failures;
    std::printf("{\"candidate_boundary_fixture\":\"%s\",\"model_cause_established\":false,\"scaled_zero_weight_dot\":%.9g}\n", failures?"failed":"pass",out);
    return failures?1:0;
}
