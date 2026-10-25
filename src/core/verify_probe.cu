#include "strata/core/verify_probe.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core::verify_probe {
namespace {
std::atomic<uint64_t> next_owner{0};
__global__ void reset(Record* record) { if (!threadIdx.x && !blockIdx.x) *record = Record{}; }
__global__ void reset_route(RouteSnapshot* s) { if (!threadIdx.x) { s->valid=0; s->error=0; } }
__global__ void capture_route_kernel(const Record* r, RouteSnapshot* s, RouteView v, unsigned ordinal) {
    if(r->ordinal!=ordinal || r->point.stage!=Stage::routed_parts) return;
    const auto local_row=r->index/(unsigned long long)v.width;
    const int token=(int)(local_row/v.k), slot=(int)(local_row%v.k), cap=v.rows*v.k;
    if(!threadIdx.x) {
        s->valid=1;s->error=0;s->ordinal=(int)ordinal;s->width=v.width;s->k=v.k;
        s->local_token=token;s->slot=slot;s->ring=v.ring;
        s->all_resident=v.all_resident;s->device_plan=v.device_plan;s->remote=v.remote;
        s->gu_type=v.gu_type;s->down_type=v.down_type;
        s->skipped=v.skip && *v.skip==v.ring;
        for(int i=0;i<3;++i)s->counts[i]=v.counts[i];
        s->gpu_entries=v.counts[1];s->gpu_entry=-1;
        if(local_row>=(unsigned long long)cap || token<0 || token>=v.rows || v.counts[0]<0 || v.counts[0]>cap ||
           v.counts[2]<0 || v.counts[2]>cap || v.counts[1]<0 || v.counts[1]>cap ||
           r->point.width!=v.width || r->point.rows_per_token!=v.k ||
           r->point.rows!=cap || r->point.row_offset%v.k || r->point.format!=Format::f32) s->error=1;
        if(!s->error) {
            for(int i=0;i<v.counts[1];++i) {
                const int dst=v.dst[i];
                if(dst<0 || dst>=cap) {s->error=2;break;}
                if(dst==(int)local_row) {if(s->gpu_entry>=0){s->error=3;break;}s->gpu_entry=i;}
            }
        }
        s->expert=s->error?-1:v.ids[token*v.k+slot];
        s->weight=s->error?0.0f:v.weights[token*v.k+slot];
        if(!s->error && (s->expert<0 || s->expert>=v.experts))s->error=5;
        s->gpu_pcie=0;
        if(!s->error && v.counts[2]>0) {
            const int boundary=v.start2[0];
            if(boundary<0 || boundary>v.counts[1])s->error=6;
            else s->gpu_pcie=s->gpu_entry>=boundary;
        }
        s->gpu_valid=!s->error && s->gpu_entry>=0 && !v.all_resident;
        s->cpu_valid=!s->error && s->gpu_entry<0 && !v.all_resident && !s->skipped && !v.remote;
        s->native_q8_valid=!s->error && v.native_q8_valid;
        if(!s->error && (v.all_resident || s->skipped) && s->gpu_entry<0)s->error=4;
    }
    __syncthreads();
    if(s->error)return;
    for(int i=threadIdx.x;i<v.width;i+=blockDim.x) {
        s->mixed[i]=v.mixed[(unsigned long long)token*v.width+i];
        s->parts[i]=v.parts[local_row*v.width+i];
        if(s->gpu_valid)s->hit[i]=v.hit[local_row*v.width+i];
        if(s->cpu_valid) {
            s->cpu[i]=v.cpu[local_row*v.width+i];
            s->cpu_input[i]=v.cpu_input[(unsigned long long)token*v.width+i];
        }
    }
    if(s->native_q8_valid) {
        const int bytes=v.width/32*36;
        for(int i=threadIdx.x;i<bytes;i+=blockDim.x)s->native_q8[i]=v.native_q8[(unsigned long long)token*bytes+i];
    }
    if(s->cpu_valid)
        for(int i=threadIdx.x;i<v.k;i+=blockDim.x)s->cpu_ids[i]=v.cpu_ids[token*v.k+i];
}
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
namespace {
bool exclusive_bytes(const std::string& path,const void* data,size_t bytes,std::string& err) {
#if defined(_WIN32)
    const int fd=_open(path.c_str(),_O_CREAT|_O_EXCL|_O_WRONLY|_O_BINARY,_S_IREAD|_S_IWRITE);
#else
    const int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);
#endif
    if(fd<0){err="verify route snapshot: exclusive create "+path+": "+std::strerror(errno);return false;}
    const char* p=(const char*)data;size_t done=0;bool ok=true;
    while(done<bytes) {
#if defined(_WIN32)
        const int n=_write(fd,p+done,(unsigned)std::min<size_t>(bytes-done,1u<<20));
#else
        const auto n=write(fd,p+done,bytes-done);
#endif
        if(n<=0){ok=false;err="verify route snapshot: write "+path+": "+std::strerror(errno);break;}
        done+=(size_t)n;
    }
#if defined(_WIN32)
    if(_close(fd)!=0 && ok){ok=false;err="verify route snapshot: close "+path;}
#else
    if(close(fd)!=0 && ok){ok=false;err="verify route snapshot: close "+path;}
#endif
    return ok; // failed/partial evidence is never removed or overwritten
}
std::string json_string(const std::string& s) {
    std::string o="\"";const char* hex="0123456789abcdef";
    for(unsigned char c:s) {
        if(c=='"' || c=='\\'){o+='\\';o+=(char)c;}
        else if(c<32){o+="\\u00";o+=hex[c>>4];o+=hex[c&15];}
        else o+=(char)c;
    }
    return o+'"';
}
std::string float_number(float f) {char b[48];std::snprintf(b,sizeof(b),"%.9g",(double)f);return b;}
}
bool Probe::report_route(std::string& err) {
    if(!host_route_ || !host_route_->valid)return true;
    const auto& s=*host_route_;const auto& r=*host_record_;
    if(s.error || r.point.stage!=Stage::routed_parts || s.ordinal!=(int)r.ordinal ||
       s.width<=0 || s.width>kRouteMaxWidth || s.width%32 || s.k<=0 || s.k>kRouteMaxK || !window_ || !std::isfinite(s.weight)) {
        err="verify route snapshot: invalid captured plan/context (error="+std::to_string(s.error)+")";return false;
    }
    const auto base=route_prefix_+".owner-"+std::to_string(owner_)+".window-"+std::to_string(window_);
    if(!exclusive_bytes(base+".mixed.f32",s.mixed,s.width*sizeof(float),err) ||
       !exclusive_bytes(base+".parts.f32",s.parts,s.width*sizeof(float),err))return false;
    if(s.native_q8_valid && !exclusive_bytes(base+".native.q8_1",s.native_q8,s.width/32*36,err))return false;
    if(s.gpu_valid && !exclusive_bytes(base+".gpu-hit.f32",s.hit,s.width*sizeof(float),err))return false;
    if(s.cpu_valid && !exclusive_bytes(base+".mapped-non-gpu.f32",s.cpu,s.width*sizeof(float),err))return false;
    if(s.cpu_valid && !exclusive_bytes(base+".mapped-input.f32",s.cpu_input,s.width*sizeof(float),err))return false;
    if(s.cpu_valid && !exclusive_bytes(base+".mapped-ids.i32",s.cpu_ids,s.k*sizeof(int32_t),err))return false;
    double maxabs=0;int nonfinite=0,ds_nan=0,ds_posinf=0,ds_neginf=0;
    for(int i=0;i<s.width;++i)if(std::isfinite(s.mixed[i]))maxabs=std::max(maxabs,(double)std::fabs(s.mixed[i]));else ++nonfinite;
    if(s.native_q8_valid)for(int b=0;b<s.width/32;++b)for(int j=0;j<2;++j) {
        uint16_t bits;std::memcpy(&bits,s.native_q8+b*36+j*2,2);const unsigned m=bits&0x7fff;
        if(m>0x7c00)++ds_nan;else if(m==0x7c00){if(bits&0x8000)++ds_neginf;else ++ds_posinf;}
    }
    std::string json="{\"schema\":1,\"diagnostic_only\":true,\"timing_may_perturb\":true,\"owner\":"+std::to_string(owner_)+
        ",\"window\":"+std::to_string(window_)+",\"layer\":"+std::to_string(r.point.layer)+
        ",\"position\":"+std::to_string(last_position_)+",\"row_offset\":"+std::to_string(r.point.row_offset)+
        ",\"local_token\":"+std::to_string(s.local_token)+",\"slot\":"+std::to_string(s.slot)+
        ",\"expert\":"+std::to_string(s.expert)+",\"width\":"+std::to_string(s.width)+",\"k\":"+std::to_string(s.k)+
        ",\"gpu_entry\":"+std::to_string(s.gpu_entry)+",\"gpu_entries\":"+std::to_string(s.gpu_entries)+
        ",\"gpu_pcie\":"+(s.gpu_pcie?"true":"false")+
        ",\"gpu_hit_valid\":"+(s.gpu_valid?"true":"false")+",\"mapped_non_gpu_valid\":"+(s.cpu_valid?"true":"false")+
        ",\"cpu_producer_proven\":false,\"native_q8_valid\":"+(s.native_q8_valid?"true":"false")+",\"all_resident\":"+(s.all_resident?"true":"false")+
        ",\"device_plan\":"+(s.device_plan?"true":"false")+",\"skipped\":"+(s.skipped?"true":"false")+
        ",\"remote\":"+(s.remote?"true":"false")+",\"ring\":"+std::to_string(s.ring)+
        ",\"gu_type\":"+std::to_string(s.gu_type)+",\"down_type\":"+std::to_string(s.down_type)+
        ",\"counts\":["+std::to_string(s.counts[0])+","+std::to_string(s.counts[1])+","+std::to_string(s.counts[2])+
        "],\"route_weight\":"+float_number(s.weight)+",\"input_maxabs_finite\":"+float_number((float)maxabs)+
        ",\"input_nonfinite\":"+std::to_string(nonfinite)+",\"q8_half_ds_nan\":"+std::to_string(ds_nan)+
        ",\"q8_half_ds_posinf\":"+std::to_string(ds_posinf)+",\"q8_half_ds_neginf\":"+std::to_string(ds_neginf)+
        ",\"path_base\":"+json_string(base)+"}\n";
    if(!exclusive_bytes(base+".json",json.data(),json.size(),err))return false;
    std::fprintf(stderr,"strata verify_route_snapshot: %s",json.c_str());std::fflush(stderr);return true;
}
bool Probe::check(int status,const char* operation,std::string& err) {
    if(status==(int)cudaSuccess)return true;
    err=std::string("verify finite probe: ")+operation+": "+cudaGetErrorString((cudaError_t)status);return false;
}
Probe::~Probe() { release(); }
void Probe::release() {
    // Normal Verifier destruction explicitly releases after graphs are gone and
    // before stream destruction. This fallback handles partial initialization.
    if(device_record_ || host_record_ || device_route_ || host_route_) {
        int previous_device=-1;
        cudaGetDevice(&previous_device);
        if(device_>=0 && previous_device!=device_)cudaSetDevice(device_);
        cudaStreamSynchronize((cudaStream_t)stream_);
        if(device_record_)cudaFree(device_record_);
        if(host_record_)cudaFreeHost(host_record_);
        if(device_route_)cudaFree(device_route_);
        if(host_route_)cudaFreeHost(host_route_);
        device_record_=nullptr;host_record_=nullptr;
        device_route_=nullptr;host_route_=nullptr;
        if(previous_device>=0 && previous_device!=device_)cudaSetDevice(previous_device);
    }
}
bool Probe::init(bool enabled,void* stream,int device,std::string& err,const char* route_prefix) {
    if(!enabled)return true; // no CUDA call, including device/error queries
    if(device_record_ || host_record_) {err="verify finite probe: already initialized";return false;}
    stream_=stream;device_=device;owner_=++next_owner;
    if(!check((int)cudaMalloc(&device_record_,sizeof(Record)),"allocate device",err) ||
       !check((int)cudaMallocHost(&host_record_,sizeof(Record)),"allocate receipt",err))return false;
    if(route_prefix && *route_prefix) {
        route_prefix_=route_prefix;
        if(!check((int)cudaMalloc(&device_route_,sizeof(RouteSnapshot)),"allocate route snapshot",err) ||
           !check((int)cudaMallocHost(&host_route_,sizeof(RouteSnapshot)),"allocate route receipt",err))return false;
        // Outside graph capture. Invalid producer arrays are never read, but
        // zeroing once also avoids transferring uninitialized allocation bytes.
        if(!check((int)cudaMemset(device_route_,0,sizeof(RouteSnapshot)),"initialize route allocation",err))return false;
    }
    return true;
}
bool Probe::begin_graph(std::string& err) {
    if(!device_record_)return true;
    ordinal_=0;reset<<<1,1,0,(cudaStream_t)stream_>>>(device_record_);
    if(device_route_)reset_route<<<1,1,0,(cudaStream_t)stream_>>>(device_route_);
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
    if(device_route_ && !check((int)cudaMemcpyAsync(host_route_,device_route_,sizeof(RouteSnapshot),cudaMemcpyDeviceToHost,
                              (cudaStream_t)stream_),"queue route receipt",err))return false;
    return check((int)cudaMemcpyAsync(host_record_,device_record_,sizeof(Record),cudaMemcpyDeviceToHost,
                                     (cudaStream_t)stream_),"queue receipt",err);
}
bool Probe::capture_route(RouteView v,std::string& err) {
    if(!device_route_)return true;
    if(!v.ids || !v.cpu_ids || !v.weights || !v.mixed || !v.parts || !v.counts || !v.dst || !v.start2 || !v.hit || !v.cpu || !v.cpu_input || v.experts<=0 ||
       v.rows<=0 || v.k<=0 || v.k>kRouteMaxK || v.rows>std::numeric_limits<int>::max()/v.k || v.width<=0 ||
       v.width>kRouteMaxWidth || v.width%32 || (v.native_q8_valid && !v.native_q8)) {
        err="verify route snapshot: invalid/bounded view";return false;
    }
    capture_route_kernel<<<1,256,0,(cudaStream_t)stream_>>>(device_record_,device_route_,v,ordinal_);
    return check((int)cudaGetLastError(),"capture route",err);
}
int64_t Probe::report(int rows,int64_t pos0,const int64_t* positions) {
    if(!host_record_)return -1;
    const Record& r=*host_record_;++window_;
    const int64_t local=r.ordinal?(int64_t)(r.index/(uint64_t)r.point.width):-1;
    const int64_t row=r.ordinal?r.point.row_offset+local:-1;
    const int64_t token=r.ordinal?row/r.point.rows_per_token:-1;
    const bool token_row=r.ordinal && token>=0 && token<rows;
    const int64_t position=token_row?(positions?positions[token]:pos0+token):-1;
    last_position_=position;
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
