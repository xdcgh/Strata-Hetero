// Synthetic metadata/files only. No GGUF, model operators, backend or device.
// Compile/run only after root review; this file supplies the one copy API as a fake.
#include "../../tools/vision/vision_stage_trace.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

static unsigned copies = 0;
extern "C" void ggml_backend_tensor_get(const ggml_tensor*, void* data, size_t, size_t size) {
    ++copies;
    std::memset(data,0,size);  // Explicit small synthetic bytes, no tensor pointer dereference.
}

static void check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
static ggml_tensor tensor(const char* name, int64_t features=1152, int64_t rows=36) {
    ggml_tensor t{};
    t.type=GGML_TYPE_F32; t.op=GGML_OP_ADD;
    t.ne[0]=features; t.ne[1]=rows; t.ne[2]=t.ne[3]=1;
    size_t stride=4;
    for (int i=0;i<4;++i) { t.nb[i]=stride; stride*=size_t(t.ne[i]); }
    t.data=reinterpret_cast<void*>(1);
    t.buffer=reinterpret_cast<ggml_backend_buffer_t>(1);
    std::strncpy(t.name,name,sizeof(t.name)-1);
    return t;
}
static void emit(VisionStageTrace& trace, ggml_tensor& t) {
    check(VisionStageTrace::callback(&t,true,&trace),"selector not requested");
    check(VisionStageTrace::callback(&t,false,&trace),"synthetic capture failed");
}
static void full_capture(VisionStageTrace& trace) {
    for (const char* name : {"patch_bias","inp_pos_emb","ln1-0","ln1-1","ln1-26"}) {
        auto t=tensor(name); emit(trace,t);
    }
    auto mm=tensor("SYNTHETIC_MM"); mm.op=GGML_OP_MUL_MAT;
    auto parent=tensor("SYNTHETIC_QKV",3456); parent.src[0]=&mm;
    auto q=tensor("Qcur-0",72,16); q.op=GGML_OP_VIEW;
    q.ne[2]=36; q.nb[1]=288; q.nb[2]=13824; q.nb[3]=36*13824;
    q.view_src=q.src[0]=&parent; emit(trace,q);
    for (const char* name : {"layer_out-0","layer_out-1","layer_out-26"}) {
        auto t=tensor(name); emit(trace,t);
    }
    auto post=tensor("norm_b-27"); emit(trace,post);
    auto merged=tensor("norm_b-27 (reshaped)",4608,9);
    merged.op=GGML_OP_RESHAPE; merged.view_src=merged.src[0]=&post; emit(trace,merged);
}

int main(int argc, char** argv) {
    try {
        check(argc==2,"provide a new owned fixture directory");
        const auto root=std::filesystem::absolute(argv[1]).lexically_normal();
        check(std::filesystem::create_directory(root),"fixture directory already exists");
        const auto image=(root/"SYNTHETIC.png").string();
        auto patch=tensor("patch_bias");
        VisionStageTrace trace;
        check(!VisionStageTrace::callback(&patch,true,&trace),"normal ENC mode requested capture");
        check(VisionStageTrace::callback(&patch,false,&trace) && copies==0,"normal ENC copied a tensor");
        for (int i=1;i<=3;++i) {
            trace.begin(image,(root/("SYNTHETIC-out-"+std::to_string(i)+".sve")).string(),(root/("formal-"+std::to_string(i))).string());
            full_capture(trace);
            check(trace.finish().empty(),"complete whitelist failed");
            const auto prefix=root/("formal-"+std::to_string(i));
            check(std::filesystem::file_size(prefix/"qkv.0.f32")==3456*36*4,"QKV was packed from strided view");
            check(std::filesystem::file_size(prefix/"merged.f32")==4608*9*4,"merger byte shape incorrect");
            std::ifstream manifest(prefix/"manifest.json");
            const std::string body((std::istreambuf_iterator<char>(manifest)),std::istreambuf_iterator<char>{});
            check(body.find("\"capture_bytes\":2156544")!=std::string::npos,"capture total differs from 13NE");
        }
        check(copies==33,"unexpected whitelist copy count");
        bool rejected=false;
        try { trace.begin(image,(root/"fourth.sve").string(),(root/"formal-4").string()); }
        catch (...) { rejected=true; }
        check(rejected && !std::filesystem::exists(root/"formal-4"),"fourth formal capture accepted");

        VisionStageTrace duplicate;
        duplicate.begin(image,(root/"dup.sve").string(),(root/"dup").string()); emit(duplicate,patch);
        check(!VisionStageTrace::callback(&patch,false,&duplicate),"duplicate selector accepted");
        check(duplicate.finish().find("duplicate")!=std::string::npos,"duplicate error not sticky");
        VisionStageTrace missing;
        missing.begin(image,(root/"missing.sve").string(),(root/"missing").string()); emit(missing,patch);
        check(missing.finish().find("missing")!=std::string::npos,"missing coverage accepted");
        check(!std::filesystem::exists(root/"missing/manifest.json"),"missing capture claimed complete");

        VisionStageTrace budget(1);
        budget.begin(image,(root/"budget.sve").string(),(root/"budget").string());
        const unsigned before=copies;
        check(!VisionStageTrace::callback(&patch,false,&budget),"byte cap accepted");
        check(copies==before && budget.finish().find("budget")!=std::string::npos,"budget checked after payload read");
        for (int bad=0;bad<3;++bad) {
            VisionStageTrace invalid;
            invalid.begin(image,(root/("invalid-"+std::to_string(bad)+".sve")).string(),(root/("invalid-"+std::to_string(bad))).string());
            auto t=patch;
            if (bad==0) t.nb[1]+=4;
            if (bad==1) t.type=GGML_TYPE_F16;
            if (bad==2) t.data=nullptr;
            check(!VisionStageTrace::callback(&t,false,&invalid),"invalid meta accepted");
            check(!invalid.finish().empty(),"invalid metadata did not prevent completion");
        }
        VisionStageTrace overwrite;
        overwrite.begin(image,(root/"overwrite.sve").string(),(root/"overwrite").string());
        const auto reserved=root/"overwrite/patch_merge.f32";
        { std::ofstream old(reserved); old << "owned sentinel"; }
        const auto size=std::filesystem::file_size(reserved);
        check(!VisionStageTrace::callback(&patch,false,&overwrite),"preexisting file overwritten");
        check(std::filesystem::file_size(reserved)==size,"overwrite changed reserved file");
        check(!overwrite.finish().empty(),"write failure accepted");
        check(VisionStageTrace::exclusive_output(reserved.string())==nullptr,"SVE output overwrote existing file");
        VisionStageTrace bad_qkv;
        bad_qkv.begin(image,(root/"bad-qkv.sve").string(),(root/"bad-qkv").string()); emit(bad_qkv,patch);
        auto wrong_parent=tensor("SYNTHETIC_WRONG_PARENT",3456); wrong_parent.op=GGML_OP_MUL_MAT;
        auto q=tensor("Qcur-0",72,16); q.op=GGML_OP_VIEW; q.view_src=q.src[0]=&wrong_parent;
        check(!VisionStageTrace::callback(&q,false,&bad_qkv),"non-ADD QKV parent accepted");
        check(bad_qkv.finish().find("QKV parent")!=std::string::npos,"QKV parent failure not sticky");
        rejected=false;
        try { VisionStageTrace again; again.begin(image,(root/"unused.sve").string(),(root/"formal-1").string()); }
        catch (...) { rejected=true; }
        check(rejected,"preexisting prefix accepted");
        std::cout << "{\"status\":\"synthetic_fixture_pass\",\"real_tensor_read\":false,\"model_run\":false}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "fixture: " << e.what() << '\n'; return 1; }
}
