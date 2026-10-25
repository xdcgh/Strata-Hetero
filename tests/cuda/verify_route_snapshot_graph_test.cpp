// Synthetic graph only. Output prefix must point into the caller's fresh run.
#include "strata/core/verify_probe.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
using namespace strata::core::verify_probe;
namespace {
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void cu(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
struct Graphs {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
 ~Graphs(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}};
void disabled(){Probe p;std::string e;need(p.init(false,nullptr,-1,e,"invalid-prefix"),e.c_str());
 need(p.begin_graph(e)&&p.capture_route({},e)&&p.end_graph(e)&&p.report_route(e),e.c_str());
 need(!p.allocated()&&!p.completed_route(),"disabled allocated");}
}
int main(int argc,char** argv){
 try {
  disabled();
  if(argc==2&&!std::strcmp(argv[1],"--disabled-only")){std::puts("PASS route_disabled_no_cuda");return 0;}
  need(argc==3&&!std::strcmp(argv[1],"--prefix"),"usage: --prefix <fresh-output-prefix>");
  cudaStream_t stream;cu(cudaStreamCreate(&stream));
  float *mixed,*parts,*hit,*cpu,*weights;int32_t *ids,*counts,*dst,*start2;uint8_t* q8;
  cu(cudaMalloc(&mixed,64*sizeof(float)));cu(cudaMalloc(&parts,128*sizeof(float)));
  cu(cudaMalloc(&hit,128*sizeof(float)));cu(cudaMalloc(&cpu,128*sizeof(float)));
  cu(cudaMalloc(&weights,4*sizeof(float)));cu(cudaMalloc(&ids,4*sizeof(int32_t)));
  cu(cudaMalloc(&counts,3*sizeof(int32_t)));cu(cudaMalloc(&dst,4*sizeof(int32_t)));
  cu(cudaMalloc(&start2,sizeof(int32_t)));cu(cudaMalloc(&q8,72));
  {
   Probe p;std::string e;need(p.init(true,stream,0,e,argv[2]),e.c_str());
   RouteView v;v.ids=ids;v.counts=counts;v.dst=dst;v.start2=start2;v.weights=weights;v.mixed=mixed;
   v.parts=parts;v.hit=hit;v.cpu=cpu;v.native_q8=q8;v.native_q8_valid=true;
   v.rows=2;v.width=32;v.k=2;v.experts=512;v.ring=83;v.gu_type=12;v.down_type=7;
   Graphs graphs;auto& graph=graphs.graph;auto& exec=graphs.exec;
   cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
   need(p.begin_graph(e),e.c_str());
   need(p.scan(parts,{41,4,32,4,1,Stage::routed_parts,Format::f32,2},e),e.c_str());
   need(p.capture_route(v,e),e.c_str());
   // Deliberately overwrite the producers AFTER capture, BEFORE graph tail.
   cu(cudaMemsetAsync(mixed,0,64*sizeof(float),stream));cu(cudaMemsetAsync(ids,0,4*sizeof(int32_t),stream));
   need(p.end_graph(e),e.c_str());cu(cudaStreamEndCapture(stream,&graph));cu(cudaGraphInstantiate(&exec,graph,0));
   std::array<float,64> x;for(int i=0;i<64;++i)x[i]=float(1000+i);
   std::array<float,128> part{},gpu{},mapped{};part[101]=std::numeric_limits<float>::quiet_NaN();
   for(int i=0;i<128;++i){gpu[i]=float(2000+i);mapped[i]=float(3000+i);}
   std::array<float,4> w{.1f,.2f,.3f,.4f};std::array<int32_t,4> id{11,22,33,409},d{0,1,2,3};
   std::array<int32_t,3> count{1,1,0};int32_t boundary=1;std::array<uint8_t,72> packed{};
   // The selected token's q8_1 d/s header includes +Inf to prove raw preservation.
   packed[38]=0;packed[39]=0x7c;
   auto replay=[&] {
    cu(cudaMemcpyAsync(mixed,x.data(),sizeof(x),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(parts,part.data(),sizeof(part),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(hit,gpu.data(),sizeof(gpu),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(cpu,mapped.data(),sizeof(mapped),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(weights,w.data(),sizeof(w),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(ids,id.data(),sizeof(id),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(counts,count.data(),sizeof(count),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(dst,d.data(),sizeof(d),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(start2,&boundary,sizeof(boundary),cudaMemcpyHostToDevice,stream));
    cu(cudaMemcpyAsync(q8,packed.data(),sizeof(packed),cudaMemcpyHostToDevice,stream));
    cu(cudaGraphLaunch(exec,stream));cu(cudaStreamSynchronize(stream));
   };
   replay();auto s=p.completed_route();
   need(s&&s->valid&&!s->error&&s->expert==409&&s->local_token==1&&s->slot==1,"route identity");
   need(s->gpu_entry==-1&&!s->gpu_valid&&s->cpu_valid,"mapped membership");
   need(s->mixed[0]==1032&&s->cpu[0]==3096&&s->native_q8[3]==0x7c,"producer lifetime/raw snapshot");
   need(p.report(4,900)==903,"runtime offset");need(p.report_route(e),e.c_str());
   need(!p.report_route(e),"exclusive files must reject overwrite");e.clear();
   d[0]=3;replay();s=p.completed_route();
   need(s->gpu_entry==0&&s->gpu_valid&&!s->cpu_valid&&s->hit[0]==2096,"GPU membership reset");
   const int64_t positions[4]={10,20,30,90001};need(p.report(4,-1,positions)==90001,"runtime per-row positions");
   need(p.report_route(e),e.c_str());
   count[1]=5;replay();need(p.completed_route()->error==1,"invalid count bounds fail closed");
   p.report(4,1900);need(!p.report_route(e),"invalid plan must fail report");e.clear();
   count[1]=1;part.fill(0);replay();need(!p.completed_route()->valid,"finite replay leaked snapshot");
   need(p.report(4,2900)==-1&&p.report_route(e),"finite report/reset");
   cu(cudaGraphExecDestroy(exec));exec=nullptr;cu(cudaGraphDestroy(graph));graph=nullptr;p.release();
  }
  cu(cudaFree(mixed));cu(cudaFree(parts));cu(cudaFree(hit));cu(cudaFree(cpu));cu(cudaFree(weights));
  cu(cudaFree(ids));cu(cudaFree(counts));cu(cudaFree(dst));cu(cudaFree(start2));cu(cudaFree(q8));cu(cudaStreamDestroy(stream));
  std::puts("PASS route_graph_snapshot_membership_lifetime_reset_positions_exclusive");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
