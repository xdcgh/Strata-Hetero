// Synthetic CUDA Graph replay only. No model/framework inputs.
#include "strata/core/verify_probe.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
using namespace strata::core::verify_probe;
namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
void cuda_ok(cudaError_t status) { if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); }
void probe_ok(bool ok,const std::string& error) { if(!ok)throw std::runtime_error(error); }
void disabled() {
    Probe p;std::string error;
    probe_ok(p.init(false,nullptr,-1,error),error);
    probe_ok(p.begin_graph(error),error);
    // Invalid pointer/shape are intentional: disabled code must not inspect
    // them, query CUDA state, allocate, launch, transfer, or synchronize.
    probe_ok(p.scan(nullptr,{},error),error);
    probe_ok(p.end_graph(error),error);p.report(0,-1);p.release();
    require(!p.allocated() && !p.completed_record(),"disabled owner allocated");
    require(!requested(nullptr) && !requested("0") && !requested("true") && requested("1"),"opt-in parser");
}
}
int main(int argc,char** argv) {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    uint32_t* d32=nullptr;uint16_t *d16=nullptr,*dbf=nullptr;
    try {
        disabled();
        if(argc==2 && !std::strcmp(argv[1],"--disabled-only")) {
            std::puts("PASS disabled_no_cuda_api_path");return 0;
        }
        cuda_ok(cudaStreamCreate(&stream));
        cuda_ok(cudaMalloc(&d32,8*sizeof(uint32_t)));
        cuda_ok(cudaMalloc(&d16,8*sizeof(uint16_t)));
        cuda_ok(cudaMalloc(&dbf,8*sizeof(uint16_t)));
        {
            Probe p;std::string error;
            probe_ok(p.init(true,stream,0,error),error);
            cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            probe_ok(p.begin_graph(error),error);
            probe_ok(p.scan(d32,{3,2,4,0,0,Stage::hc_attention,Format::f32},error),error);
            probe_ok(p.scan(d16,{4,2,4,0,1,Stage::hc_ffn,Format::f16},error),error);
            probe_ok(p.scan(dbf,{5,2,4,0,1,Stage::shared_input_bf16,Format::bf16},error),error);
            probe_ok(p.end_graph(error),error);
            cuda_ok(cudaStreamEndCapture(stream,&graph));
            cuda_ok(cudaGraphInstantiate(&exec,graph,0));
            std::array<uint32_t,8> a{};
            std::array<uint16_t,8> b{},c{};
            auto replay=[&](int64_t position,unsigned ordinal,Format format,unsigned long long index,
                            unsigned long long nan,unsigned long long positive,unsigned long long negative) {
                cuda_ok(cudaMemcpyAsync(d32,a.data(),sizeof(a),cudaMemcpyHostToDevice,stream));
                cuda_ok(cudaMemcpyAsync(d16,b.data(),sizeof(b),cudaMemcpyHostToDevice,stream));
                cuda_ok(cudaMemcpyAsync(dbf,c.data(),sizeof(c),cudaMemcpyHostToDevice,stream));
                cuda_ok(cudaGraphLaunch(exec,stream));cuda_ok(cudaStreamSynchronize(stream));
                const Record& r=*p.completed_record();
                require(r.checkpoints==3,"captured checkpoint count");
                require(r.ordinal==ordinal,"first checkpoint or replay reset leaked");
                if(ordinal) {
                    require(r.point.format==format && r.index==index,"typed format / first index");
                    require(r.nan==nan && r.positive_inf==positive && r.negative_inf==negative,"typed counts");
                } else require(r.index==~0ull && !r.nan && !r.positive_inf && !r.negative_inf,"finite replay leaked counts");
                const int64_t reported=p.report(2,position);
                require(reported==(ordinal?position+(int64_t)(index/4):-1),"runtime position baked into graph or wrong row");
            };
            // All three formats contain nonfinites. Only the earliest scan wins;
            // its entire NaN/+Inf/-Inf census is retained.
            a[5]=0x7fc00001u;a[6]=0x7f800000u;a[7]=0xff800000u;
            b[0]=0x7e01u;c[0]=0x7fc1u;
            replay(100,1,Format::f32,5,1,1,1);
            a.fill(0);b.fill(0);c.fill(0);
            b[1]=0x7e01u;b[2]=0x7c00u;b[3]=0xfc00u;c[0]=0x7fc1u;
            replay(900,2,Format::f16,1,1,1,1);
            b.fill(0);c.fill(0);
            c[4]=0x7fc1u;c[5]=0x7f80u;c[6]=0xff80u;
            replay(1900,3,Format::bf16,4,1,1,1);
            const int64_t positions[2]={17,90001};
            require(p.report(2,-1,positions)==90001,"batch positions treated as consecutive");
            // Finite maxima/subnormals must not be misread as infinities.
            a.fill(0x7f7fffffu);b.fill(0x7bffu);c.fill(0x7f7fu);
            a[0]=1;b[0]=1;c[0]=1;
            replay(2900,0,Format::f32,~0ull,0,0,0);
            cuda_ok(cudaGraphExecDestroy(exec));exec=nullptr;
            cuda_ok(cudaGraphDestroy(graph));graph=nullptr;
            p.release();
        }
        cuda_ok(cudaFree(d32));d32=nullptr;cuda_ok(cudaFree(d16));d16=nullptr;cuda_ok(cudaFree(dbf));dbf=nullptr;
        cuda_ok(cudaStreamDestroy(stream));stream=nullptr;
        std::puts("PASS typed_graph_replay_reset_runtime_positions");return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL %s\n",e.what());
        // Only fixture-owned resources. Normal owner destruction drains scans.
        if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
        if(d32)cudaFree(d32);if(d16)cudaFree(d16);if(dbf)cudaFree(dbf);
        if(stream)cudaStreamDestroy(stream);return 1;
    }
}
