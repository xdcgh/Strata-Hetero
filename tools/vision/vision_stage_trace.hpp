#pragma once
#include "ggml-backend.h"
#include "../hetero_native_cpu/sha256.hpp"
#include <array>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// Observation only. The selected model/source contract is checked before copying.
class VisionStageTrace {
    using Path = std::filesystem::path;
    static constexpr int64_t E = 1152;
    static constexpr const char* revision = "3cf03257f219afbe7334045ff7c6a06ac68c627d";
    const std::array<const char*,11> names{{"patch_bias","inp_pos_emb","ln1-0","ln1-1","ln1-26",
        "Qcur-0","layer_out-0","layer_out-1","layer_out-26","norm_b-27","norm_b-27 (reshaped)"}};
    const std::array<const char*,11> labels{{"patch_merge","positioned","ln1.0","ln1.1","ln1.26",
        "qkv.0","block.0","block.1","block.26","post_norm","merged"}};
    bool active_ = false, failed_ = false;
    int64_t n_ = 0;
    size_t cap_, bytes_ = 0;
    Path directory_, image_, output_;
    std::string key_, error_, records_;
    std::set<int> seen_;
    std::map<std::string,size_t> image_bytes_;
    std::map<std::string,unsigned> image_captures_;
    static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
    void fail(const char* message) noexcept {
        failed_ = true;
        try { if (error_.empty()) error_ = message; } catch (...) { }
    }
    static std::string quote(const std::string& text) {
        std::string out = "\"";
        for (unsigned char c : text) {
            if (c < 32) { char escaped[7]; std::snprintf(escaped,sizeof escaped,"\\u%04x",unsigned(c)); out += escaped; }
            else { if (c == '"' || c == '\\') out += '\\'; out += char(c); }
        }
        return out + '"';
    }
    template<class T> static void array(std::ostream& out, const T* values) {
        out << '['; for (int i=0;i<4;++i) { if (i) out << ','; out << values[i]; } out << ']';
    }
    static FILE* create_new(const Path& path) {
#ifdef _WIN32
        const int fd = _wopen(path.c_str(),_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
        FILE* stream = fd < 0 ? nullptr : _fdopen(fd,"wb");
        if (fd >= 0 && !stream) _close(fd);
#else
        const int fd = open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        FILE* stream = fd < 0 ? nullptr : fdopen(fd,"wb");
        if (fd >= 0 && !stream) close(fd);
#endif
        return stream;
    }
    static void write_new(const Path& path, const void* data, size_t size) {
        FILE* stream = create_new(path);
        require(stream != nullptr,"stage file already exists or cannot be created");
        const bool written = std::fwrite(data,1,size,stream) == size;
        const bool closed = std::fclose(stream) == 0;
        require(written && closed,"stage file short write or close failure");
    }
    void capture(ggml_tensor* selected, int index) {
        require(seen_.count(index) == 0,"duplicate stage selector");
        ggml_tensor* t = selected;
        if (index == 5) {
            t = selected->src[0];
            require(selected->op == GGML_OP_VIEW && t && selected->view_src == t && selected->view_offs == 0,
                    "Qcur-0 is not the expected combined-QKV view");
            require(t->op == GGML_OP_ADD && t->src[0] && t->src[0]->op == GGML_OP_MUL_MAT,
                    "QKV parent is not post-bias ADD after MUL_MAT");
            require(selected->type == GGML_TYPE_F32 && selected->ne[0] == 72 && selected->ne[1] == 16 &&
                    selected->ne[2] == n_ && selected->ne[3] == 1 && selected->nb[0] == 4 &&
                    selected->nb[1] == 288 && selected->nb[2] == 13824 && selected->nb[3] == size_t(n_)*13824,
                    "unexpected Qcur-0 axes or strides");
        }
        if (!n_) { require(index == 0 && (t->ne[1] == 36 || t->ne[1] == 72),"expected aligned fixture patch count"); n_ = t->ne[1]; }
        const int64_t features = index == 5 ? 3*E : (index == 10 ? 4*E : E);
        const int64_t rows = index == 10 ? n_/4 : n_;
        require(t->type == GGML_TYPE_F32 && t->ne[0] == features && t->ne[1] == rows &&
                t->ne[2] == 1 && t->ne[3] == 1,"unexpected stage dtype or axes");
        size_t stride = 4;
        for (int axis=0;axis<4;++axis) { require(t->nb[axis] == stride,"noncontiguous stage tensor"); stride *= size_t(t->ne[axis]); }
        require(t->data && (t->buffer || (t->view_src && t->view_src->buffer)),"stage tensor is not allocated");
        if (index == 10) require(t->op == GGML_OP_RESHAPE && t->view_src && t->view_offs == 0 &&
            std::string(t->view_src->name) == "norm_b-27","unexpected four-patch merger parent");
        require(stride <= cap_ && image_bytes_[key_] <= cap_-stride,"per-image stage byte budget exceeded");
        image_bytes_[key_] += stride; bytes_ += stride;  // Failed partial writes remain charged.
        std::vector<unsigned char> raw(stride);
        ggml_backend_tensor_get(t,raw.data(),0,stride);
        const Path path = directory_/(std::string(labels[index])+".f32");
        write_new(path,raw.data(),raw.size());
        std::ostringstream row;
        row << (seen_.empty() ? "" : ",") << "{\"stage\":" << quote(labels[index])
            << ",\"selector\":" << quote(selected->name) << ",\"target_name\":" << quote(t->name)
            << ",\"actual_dtype\":\"F32\",\"type_id\":" << int(t->type) << ",\"ne\":";
        array(row,t->ne); row << ",\"nb\":"; array(row,t->nb);
        row << ",\"selector_ne\":"; array(row,selected->ne); row << ",\"selector_nb\":"; array(row,selected->nb);
        row << ",\"path\":" << quote(path.generic_string()) << ",\"bytes\":" << stride
            << ",\"raw_sha256\":" << quote(hetero_native_cpu::sha256_hex(raw.data(),raw.size())) << '}';
        records_ += row.str(); seen_.insert(index);
    }
public:
    static constexpr size_t max_image_bytes = 16*1024*1024;
    explicit VisionStageTrace(size_t cap=max_image_bytes) : cap_(cap) {
        require(cap_ > 0 && cap_ <= max_image_bytes,"invalid stage byte cap");
    }
    static FILE* exclusive_output(const std::string& path) noexcept {
        try { return create_new(Path(path)); } catch (...) { return nullptr; }
    }
    void disarm() noexcept { active_=false; failed_=false; n_=0; bytes_=0; seen_.clear(); error_.clear(); records_.clear(); }
    void begin(const std::string& image, const std::string& output, const std::string& prefix) {
        disarm();
        const uint16_t endian=1; require(*reinterpret_cast<const unsigned char*>(&endian)==1,"stage export requires little endian");
        image_ = std::filesystem::weakly_canonical(image); output_ = std::filesystem::absolute(output).lexically_normal();
        key_ = image_.generic_string();
#ifdef _WIN32
        for (char& c : key_) c = char(std::tolower(static_cast<unsigned char>(c)));
#endif
        Path requested(prefix); require(!requested.filename().empty() && requested.filename() != "." && requested.filename() != "..","invalid stage prefix");
        directory_ = std::filesystem::canonical(requested.has_parent_path()?requested.parent_path():Path("."))/requested.filename();
        require(!std::filesystem::exists(output_) && !std::filesystem::is_symlink(output_),"stage SVE output already exists");
        require(image_captures_[key_] < 3,"per-image formal capture count exceeded");
        require(std::filesystem::create_directory(directory_),"stage prefix directory already exists");
        ++image_captures_[key_]; active_ = true;
    }
    std::string finish(bool encoded=true) {
        const bool was_active = active_; active_ = false;
        if (!was_active) return "";
        try {
            require(encoded,"stage image encode failed");
            require(!failed_,error_.empty()?"stage callback failed":error_.c_str());
            require(seen_.size() == names.size(),"missing stage coverage");
            require(bytes_ == size_t(13*n_*E*4),"stage byte total differs from whitelist");
            std::ostringstream manifest;
            manifest << "{\"schema_version\":1,\"status\":\"stages_captured\",\"source_revision\":" << quote(revision)
                << ",\"image\":" << quote(image_.generic_string()) << ",\"sve_output\":" << quote(output_.generic_string())
                << ",\"capture_bytes\":" << bytes_ << ",\"image_total_bytes\":" << image_bytes_[key_]
                << ",\"image_byte_cap\":" << cap_ << ",\"capture_number\":" << image_captures_[key_]
                << ",\"stages\":[" << records_ << "]}\n";
            const auto text=manifest.str(); write_new(directory_/"manifest.json",text.data(),text.size());
        } catch (const std::exception& e) { fail(e.what()); } catch (...) { fail("unknown stage completion failure"); }
        return failed_ ? "stage_taps: "+(error_.empty()?std::string("callback failure"):error_) : "";
    }
    static bool callback(ggml_tensor* t, bool ask, void* userdata) noexcept {
        if (!userdata) return false;
        auto& self=*static_cast<VisionStageTrace*>(userdata);
        try {
            if (!self.active_) return !ask;
            if (self.failed_) return false;
            require(t != nullptr,"null stage tensor");
            for (size_t index=0;index<self.names.size();++index) if (std::string(t->name)==self.names[index]) {
                if (!ask) self.capture(t,int(index));
                return true;
            }
            return !ask;
        } catch (const std::exception& e) { self.fail(e.what()); } catch (...) { self.fail("unknown stage callback failure"); }
        return false;
    }
};
