// Host-only fallback and eligible mock-child contracts. No model/device math.
#include "strata/kernels/cpu/live_helper.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
using namespace strata::kernels::cpu;
static void mode(const char* value) {
#if defined(_WIN32)
    _putenv_s("STRATA_EXPERT_HELPER", value);
#else
    setenv("STRATA_EXPERT_HELPER", value, 1);
#endif
}
int main(int argc,char** argv) {
    if(argc==2) {
#if !defined(_WIN32)
        return 77; // eligible transport is Windows-only
#else
        const bool success=!std::strcmp(argv[1],"eligible-success");
        if(!success && std::strcmp(argv[1],"eligible-failure"))return 5;
        mode("forced");
        NativeFmt f;f.gu_type=12;f.d_type=7;f.n_embd=2560;f.n_ff=640;
        f.up_off=921600;f.down_off=1843200;f.bytes=3072000;
        std::vector<uint8_t> blob(f.bytes,0);
        std::vector<float> input0(H,.25f),input1(H,.5f),out0(H,-123.f),out1(H,-123.f);
        ExpertJobMulti job;job.nt=2;job.blob=blob.data();job.out[0]=out0.data();job.out[1]=out1.data();
        const float* raw[MAXT]={input0.data(),input1.data()};
        const bool completed=live_helper(f,job,raw,28,288,"eligible_host_fixture");
        if(completed!=success)return 6;
        const float expected0=success?.5f:-123.f,expected1=success?1.f:-123.f;
        if(!std::all_of(out0.begin(),out0.end(),[&](float x){return x==expected0;}) ||
           !std::all_of(out1.begin(),out1.end(),[&](float x){return x==expected1;}))return 7;
        if(!success) {
            // Poisoned transport must not restart or change output on the next
            // eligible call. Mock PID receipt is exclusive to detect restarts.
            if(live_helper(f,job,raw,28,288,"eligible_host_fixture_retry"))return 8;
            if(!std::all_of(out0.begin(),out0.end(),[](float x){return x==-123.f;}) ||
               !std::all_of(out1.begin(),out1.end(),[](float x){return x==-123.f;}))return 9;
        }
        std::fprintf(stdout,"eligible_fixture success_expected=%d completed=%d outputs_verified=1\n",(int)success,(int)completed);
        return 0;
#endif
    }
    if(argc!=1)return 10;
    NativeFmt unsupported{};
    std::vector<float> input(H, .25f), output(H, -123.f);
    ExpertJobMulti job; job.nt=1;job.out[0]=output.data();
    const float* raw[MAXT]={input.data()};
    for(const char* selected : {"off", "auto", "forced"}) {
        mode(selected);
        if(live_helper(unsupported,job,raw,28,288,"host_fixture"))return 1;
        if(!std::all_of(output.begin(),output.end(),[](float x){return x==-123.f;}))return 2;
    }
    mode("forced");
    if(live_helper(unsupported,job,raw,27,288,"host_fixture"))return 3;
    if(live_helper(unsupported,job,raw,28,287,"host_fixture"))return 4;
    mode("off");
    return 0;
}
