#include "strata/core/verify_probe.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>

namespace strata::core::verify_probe {
namespace {
std::atomic<uint64_t> next_owner{0};
__global__ void reset(Record* record) { if (!threadIdx.x && !blockIdx.x) *record = Record{}; }
template<Format format>
__global__ void inspect(const void* values, unsigned long long count, Record* record, Point point, unsigned ordinal) {
    if (!threadIdx.x && !blockIdx.x) record->checkpoints = ordinal;
    const unsigned first = atomicAdd(&record->ordinal, 0u);
    if (first && first != ordinal) return;
    for (unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<count; i+=(unsigned long long)gridDim.x*blockDim.x) {
        unsigned bits, magnitude, infinity, negative;
        if constexpr(format==Format::f32) {
            bits=__float_as_uint(static_cast<const float*>(values)[i]);
            magnitude=bits&0x7fffffffu; infinity=0x7f800000u; negative=bits>>31;
        } else {
            bits=static_cast<const uint16_t*>(values)[i];
            magnitude=bits&0x7fffu; infinity=format==Format::f16?0x7c00u:0x7f80u; negative=bits>>15;
        }
        if(magnitude<infinity) continue;
        const unsigned previous=atomicCAS(&record->ordinal,0u,ordinal);
        if(!previous) record->point=point;
        if(!previous || previous==ordinal) {
            atomicMin(&record->index,i);
            if(magnitude>infinity) atomicAdd(&record->nan,1ull);
            else if(negative) atomicAdd(&record->negative_inf,1ull);
            else atomicAdd(&record->positive_inf,1ull);
        }
    }
}
const char* stage_name(Stage s) {
    switch(s) {
    case Stage::input_residual:return "input_residual";
    case Stage::hc_attention:return "hc_attention";
    case Stage::attention_output:return "attention_output";
    case Stage::hc_ffn:return "hc_ffn";
    case Stage::route_logits:return "route_logits";
    case Stage::route_weights:return "route_weights";
    case Stage::shared_input_bf16:return "shared_input_bf16";
    case Stage::shared_scalar_gate:return "shared_scalar_gate";
    case Stage::shared_output:return "shared_output";
    case Stage::routed_parts:return "routed_parts";
    case Stage::moe_output:return "moe_output";
    case Stage::head_mixed:return "head_mixed";
    case Stage::head_logits:return "head_logits";
    }
    return "unknown";
}
}
bool Probe::check(int status,const char* operation,std::string& err) {
    if(status==(int)cudaSuccess)return true;
    err=std::string("verify finite probe: ")+operation+": "+cudaGetErrorString((cudaError_t)status);return false;
}
Probe::~Probe() { release(); }
void Probe::release() {
    // Normal Verifier destruction explicitly releases after graphs are gone and
    // before stream destruction. This fallback handles partial initialization.
    if(device_record_ || host_record_) {
        int previous_device=-1;
        cudaGetDevice(&previous_device);
        if(device_>=0 && previous_device!=device_)cudaSetDevice(device_);
        cudaStreamSynchronize((cudaStream_t)stream_);
        if(device_record_)cudaFree(device_record_);
        if(host_record_)cudaFreeHost(host_record_);
        device_record_=nullptr;host_record_=nullptr;
        if(previous_device>=0 && previous_device!=device_)cudaSetDevice(previous_device);
    }
}
bool Probe::init(bool enabled,void* stream,int device,std::string& err) {
    if(!enabled)return true; // no CUDA call, including device/error queries
    if(device_record_ || host_record_) {err="verify finite probe: already initialized";return false;}
    stream_=stream;device_=device;owner_=++next_owner;
    return check((int)cudaMalloc(&device_record_,sizeof(Record)),"allocate device",err) &&
           check((int)cudaMallocHost(&host_record_,sizeof(Record)),"allocate receipt",err);
}
bool Probe::begin_graph(std::string& err) {
    if(!device_record_)return true;
    ordinal_=0;reset<<<1,1,0,(cudaStream_t)stream_>>>(device_record_);
    return check((int)cudaGetLastError(),"reset launch",err);
}
bool Probe::scan(const void* values,Point point,std::string& err) {
    if(!device_record_)return true;
    if(!values || point.rows<=0 || point.width<=0 || point.row_offset<0 || point.rows_per_token<=0 ||
       (uint64_t)point.rows>std::numeric_limits<uint64_t>::max()/(uint64_t)point.width ||
       ordinal_==std::numeric_limits<unsigned>::max()) {err="verify finite probe: invalid point";return false;}
    const auto count=(unsigned long long)point.rows*(unsigned long long)point.width;
    const int blocks=(int)std::min<unsigned long long>(4096,1+(count-1)/256);
    const unsigned ordinal=++ordinal_;
    switch(point.format) {
    case Format::f32:inspect<Format::f32><<<blocks,256,0,(cudaStream_t)stream_>>>(values,count,device_record_,point,ordinal);break;
    case Format::f16:inspect<Format::f16><<<blocks,256,0,(cudaStream_t)stream_>>>(values,count,device_record_,point,ordinal);break;
    case Format::bf16:inspect<Format::bf16><<<blocks,256,0,(cudaStream_t)stream_>>>(values,count,device_record_,point,ordinal);break;
    default:err="verify finite probe: invalid format";return false;
    }
    return check((int)cudaGetLastError(),"scan launch (may surface producer error)",err);
}
bool Probe::end_graph(std::string& err) {
    if(!device_record_)return true;
    return check((int)cudaMemcpyAsync(host_record_,device_record_,sizeof(Record),cudaMemcpyDeviceToHost,
                                     (cudaStream_t)stream_),"queue receipt",err);
}
int64_t Probe::report(int rows,int64_t pos0,const int64_t* positions) {
    if(!host_record_)return -1;
    const Record& r=*host_record_;++window_;
    const int64_t local=r.ordinal?(int64_t)(r.index/(uint64_t)r.point.width):-1;
    const int64_t row=r.ordinal?r.point.row_offset+local:-1;
    const int64_t token=r.ordinal?row/r.point.rows_per_token:-1;
    const bool token_row=r.ordinal && token>=0 && token<rows;
    const int64_t position=token_row?(positions?positions[token]:pos0+token):-1;
    std::fprintf(stderr,"strata verify_nonfinite: {\"schema\":1,\"owner\":%llu,\"window\":%llu,\"device\":%d,\"pos0\":%lld,\"rows\":%d,"
        "\"position_kind\":\"%s\",\"diagnostic_only\":true,\"timing_may_perturb\":true,\"bad\":%s,\"checkpoints\":%u,"
        "\"ordinal\":%u,\"stage\":\"%s\",\"format\":\"%s\",\"layer\":%lld,\"half\":%d,\"scan_rows\":%lld,\"width\":%lld,"
        "\"row_offset\":%lld,\"local_row\":%lld,\"row\":%lld,\"token_row\":%lld,\"rows_per_token\":%lld,\"position\":%lld,\"column\":%lld,"
        "\"nan\":%llu,\"posinf\":%llu,\"neginf\":%llu}\n",
        (unsigned long long)owner_,(unsigned long long)window_,device_,(long long)pos0,rows,positions?"per_row":"consecutive",r.ordinal?"true":"false",r.checkpoints,
        r.ordinal,r.ordinal?stage_name(r.point.stage):"none",r.point.format==Format::f32?"f32":r.point.format==Format::f16?"f16":"bf16",
        (long long)r.point.layer,r.point.half,(long long)r.point.rows,(long long)r.point.width,(long long)r.point.row_offset,
        (long long)local,(long long)row,(long long)token,(long long)r.point.rows_per_token,(long long)position,
        r.ordinal?(long long)(r.index%(uint64_t)r.point.width):-1ll,
        r.nan,r.positive_inf,r.negative_inf);
    std::fflush(stderr);
    return position;
}
}
