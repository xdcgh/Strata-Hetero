#include "strata/kernels/cpu/live_helper.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace strata::kernels::cpu {
namespace {
#if defined(_WIN32)
struct OwnedHandle {
    HANDLE value=nullptr;
    ~OwnedHandle() { if(value)CloseHandle(value); }
    OwnedHandle()=default;
    explicit OwnedHandle(HANDLE h):value(h){}
    OwnedHandle(const OwnedHandle&)=delete;
    OwnedHandle& operator=(const OwnedHandle&)=delete;
};
struct OwnedAttributes {
    LPPROC_THREAD_ATTRIBUTE_LIST value=nullptr;
    bool initialized=false;
    ~OwnedAttributes(){if(initialized)DeleteProcThreadAttributeList(value);}
};
struct Pipe {
    HANDLE process = nullptr, job = nullptr, input = nullptr, output = nullptr;
    bool poisoned = false;
    uint64_t id = 0;
    ~Pipe() { stop(); }
    void stop() {
        if (job) { CloseHandle(job); job = nullptr; }
        if (process) {
            TerminateProcess(process, 1);
            DWORD waited=WaitForSingleObject(process,5000);
            if(waited!=WAIT_OBJECT_0)
                std::fprintf(stderr,"hetero_live cleanup=unconfirmed reason=owned_process_wait code=%lu\n",(unsigned long)waited);
            CloseHandle(process); process = nullptr;
        }
        if (input) { CloseHandle(input); input = nullptr; }
        if (output) { CloseHandle(output); output = nullptr; }
    }
    void start() {
        if (poisoned) throw std::runtime_error("bridge_disabled_after_failure");
        if (process) return;
        const char* command = std::getenv("STRATA_EXPERT_HELPER_COMMAND");
        if (!command || !*command) throw std::runtime_error("missing_bridge_command");
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        OwnedHandle child_in,child_out;
        if (!CreatePipe(&child_in.value, &input, &sa, 0)) throw std::runtime_error("create_input_pipe");
        if (!CreatePipe(&output, &child_out.value, &sa, 0)) throw std::runtime_error("create_output_pipe");
        if(!SetHandleInformation(input, HANDLE_FLAG_INHERIT, 0) ||
           !SetHandleInformation(output, HANDLE_FLAG_INHERIT, 0))
            throw std::runtime_error("restrict_parent_pipe_inheritance");
        job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            throw std::runtime_error("create_owned_job");
        }
        STARTUPINFOEXA sx{}; sx.StartupInfo.cb = sizeof(sx);
        auto& si = sx.StartupInfo; si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = child_in.value; si.hStdOutput = child_out.value;
        // No inherited engine log handles. Bridge diagnostics are returned in typed errors.
        si.hStdError = child_out.value;
        SIZE_T attribute_bytes=0;
        if(InitializeProcThreadAttributeList(nullptr,1,0,&attribute_bytes) ||
           GetLastError()!=ERROR_INSUFFICIENT_BUFFER || !attribute_bytes)
            throw std::runtime_error("size_handle_list");
        std::vector<unsigned char> attributes(attribute_bytes);
        sx.lpAttributeList=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        OwnedAttributes owned_attributes;owned_attributes.value=sx.lpAttributeList;
        HANDLE inherited[2]={child_in.value,child_out.value};
        if(!InitializeProcThreadAttributeList(sx.lpAttributeList,1,0,&attribute_bytes)) {
            throw std::runtime_error("initialize_handle_list");
        }
        owned_attributes.initialized=true;
        if(!UpdateProcThreadAttribute(sx.lpAttributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                      inherited,sizeof(inherited),nullptr,nullptr)) {
            throw std::runtime_error("restrict_handle_list");
        }
        PROCESS_INFORMATION pi{};
        std::vector<char> cmd(command, command + std::strlen(command) + 1);
        BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                 CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT,
                                 nullptr, nullptr, &si, &pi);
        if (!ok) throw std::runtime_error("start_bridge");
        process = pi.hProcess;
        OwnedHandle initial_thread(pi.hThread);
        if (!AssignProcessToJobObject(job, process)) {
            throw std::runtime_error("assign_owned_job");
        }
        if(ResumeThread(initial_thread.value)==(DWORD)-1)
            throw std::runtime_error("resume_bridge_thread");
    }
    static void io(HANDLE h, void* data, size_t bytes, bool write) {
        auto* p = static_cast<unsigned char*>(data);
        while (bytes) {
            DWORD done = 0;
            BOOL ok = write ? WriteFile(h, p, (DWORD)bytes, &done, nullptr) : ReadFile(h, p, (DWORD)bytes, &done, nullptr);
            if (!ok || !done) throw std::runtime_error("bridge_pipe_io");
            p += done; bytes -= done;
        }
    }
    std::vector<unsigned char> request(std::vector<unsigned char>& payload) {
        unsigned char header[20]{};
        std::memcpy(header, "SXPU", 4);
        uint16_t version = 1, kind = 10; uint32_t size = (uint32_t)payload.size(); ++id;
        std::memcpy(header + 4, &version, 2); std::memcpy(header + 6, &kind, 2);
        std::memcpy(header + 8, &id, 8); std::memcpy(header + 16, &size, 4);
        io(input, header, 20, true); io(input, payload.data(), payload.size(), true);
        io(output, header, 20, false);
        uint16_t rv, rk; uint64_t rid; uint32_t rs;
        std::memcpy(&rv, header+4,2); std::memcpy(&rk,header+6,2); std::memcpy(&rid,header+8,8); std::memcpy(&rs,header+16,4);
        if (std::memcmp(header,"SXPU",4) || rv!=1 || rk!=0x800a || rid!=id || rs>8*1024*1024)
            throw std::runtime_error("bridge_response_header");
        std::vector<unsigned char> result(rs); io(output,result.data(),rs,false); return result;
    }
};
#endif
}
bool live_helper(const NativeFmt& f, const ExpertJobMulti& j, const float* const* raw,
                 int layer, int expert, const char* phase) noexcept {
    const char* mode = std::getenv("STRATA_EXPERT_HELPER");
    if (!mode || !*mode || !std::strcmp(mode,"off")) return false;
    // Exact frozen identity only. auto never routes until full-model quality/gain acceptance.
    if (std::strcmp(mode,"forced") || layer!=28 || expert!=288) return false;
    const auto begin=std::chrono::steady_clock::now();
    const char* reason="unsupported_platform"; bool completed=false; size_t bytes=0;
    uint64_t worker_ns=0;
    try {
        if (f.gu_type!=12 || f.d_type!=7 || f.n_embd!=2560 || f.n_ff!=640 ||
            f.up_off!=921600 || f.down_off!=1843200 || f.bytes!=3072000 || j.nt<1 || j.nt>MAXT || !j.blob)
            throw std::runtime_error("unsupported_native_contract");
        for(int t=0;t<j.nt;++t) {
            if(!raw[t] || !j.out[t]) throw std::runtime_error("missing_model_activation");
            for(int k=0;k<H;++k) if(!std::isfinite(raw[t][k])) throw std::runtime_error("nonfinite_input");
        }
#if defined(_WIN32)
        static std::mutex mutex; static Pipe pipe;
        std::lock_guard<std::mutex> lock(mutex);
        struct FailureCleanup {
            Pipe& owner; bool keep=false;
            ~FailureCleanup() { if(!keep) {owner.poisoned=true;owner.stop();} }
        } cleanup{pipe};
        // bridge kind 10: layer, expert, rows, hidden, blob_bytes followed by exact blob and LE F32 rows.
        uint32_t meta[5]={(uint32_t)layer,(uint32_t)expert,(uint32_t)j.nt,H,(uint32_t)f.bytes};
        bytes=sizeof(meta)+f.bytes+(size_t)j.nt*H*4;
        std::vector<unsigned char> payload(bytes);
        std::memcpy(payload.data(),meta,sizeof(meta)); std::memcpy(payload.data()+sizeof(meta),j.blob,f.bytes);
        for(int t=0;t<j.nt;++t) std::memcpy(payload.data()+sizeof(meta)+f.bytes+(size_t)t*H*4,raw[t],H*4);
        try { pipe.start(); } catch(...) { pipe.poisoned=true;pipe.stop();throw; }
        auto task=std::async(std::launch::async,[&]{return pipe.request(payload);});
        // IPC wait includes child initialization/compile/transfer after process
        // creation. Owned job kills bridge and its Arc child on timeout.
        if(task.wait_for(std::chrono::seconds(120))!=std::future_status::ready) {
            pipe.poisoned=true;
            // Terminate the owned process tree first. Keep pipe handles alive
            // until the blocking I/O thread has observed EOF/broken pipe.
            if(pipe.job) { CloseHandle(pipe.job);pipe.job=nullptr; }
            if(pipe.process) { TerminateProcess(pipe.process,1);WaitForSingleObject(pipe.process,5000); }
            try { task.get(); } catch(...) {}
            pipe.stop();
            throw std::runtime_error("bridge_timeout");
        }
        std::vector<unsigned char> result;
        try { result=task.get(); } catch(...) { pipe.poisoned=true; pipe.stop(); throw; }
        uint32_t status=1;
        if(result.size()<12) {pipe.poisoned=true;pipe.stop();throw std::runtime_error("short_bridge_result");}
        std::memcpy(&status,result.data(),4); std::memcpy(&worker_ns,result.data()+4,8);
        if(status) {
            std::string detail(result.begin()+12,result.end());
            for(char& c:detail) if(c=='\n'||c=='\r') c=' ';
            if(detail.size()>768)detail.resize(768);
            pipe.poisoned=true;pipe.stop();
            throw std::runtime_error(std::string("bridge_rejected_request:")+detail);
        }
        if(result.size()!=12+(size_t)j.nt*H*4) {pipe.poisoned=true;pipe.stop();throw std::runtime_error("output_shape");}
        // Validate the entire output before committing any row. CPU fallback sees untouched rows.
        for(size_t p=12;p<result.size();p+=4) {float v;std::memcpy(&v,result.data()+p,4);if(!std::isfinite(v)){pipe.poisoned=true;pipe.stop();throw std::runtime_error("nonfinite_output");}}
        for(int t=0;t<j.nt;++t) std::memcpy(j.out[t],result.data()+12+(size_t)t*H*4,H*4);
        cleanup.keep=true;
        completed=true; reason="ok";
#endif
    } catch(const std::exception& e) { reason=e.what();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::fprintf(stderr,"hetero_live phase=%s layer=%d expert=%d rows=%d attempt=1 completed=0 fallback=1 reason=%s input_bytes=%zu output_bytes=0 wall_ms=%.3f worker_ns=%llu\n",phase,layer,expert,j.nt,reason,bytes,ms,(unsigned long long)worker_ns);
        return false;
    } catch(...) { reason="unknown_error"; }
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::fprintf(stderr,"hetero_live phase=%s layer=%d expert=%d rows=%d attempt=1 completed=%d fallback=%d reason=%s input_bytes=%zu output_bytes=%zu wall_ms=%.3f worker_ns=%llu\n",phase,layer,expert,j.nt,(int)completed,(int)!completed,reason,bytes,completed?(size_t)j.nt*H*4:0,ms,(unsigned long long)worker_ns);
    return completed;
}
}
