#pragma once

#include "common.h"
#include <unordered_map>
#include <variant>

namespace moe {

// GGUF v3 Metadata Value Types
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct GGUFValue {
    GGUFValueType type = GGUFValueType::UINT32;
    uint64_t u64 = 0;
    int64_t  i64 = 0;
    double   f64 = 0.0;
    bool     bval = false;
    std::string str;
    std::vector<std::string> str_arr;
    std::vector<float> f32_arr;
    std::vector<int32_t> i32_arr;
};

struct TensorDescriptor {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {1, 1, 1, 1};
    GGMLType type = GGMLType::F32;
    uint64_t offset = 0;          // Relative offset inside tensor data section (GGUF) or file offset
    uint64_t abs_file_offset = 0; // Absolute byte offset from start of file on disk
    uint64_t byte_size = 0;       // Total bytes on disk
    bool is_sparse_expert = false;// True if this tensor is an MoE routed expert weight matrix
    const void* mapped_ptr = nullptr; // Zero-copy mmap pointer for dense backbone
};

class ModelFile {
public:
    ModelFile() = default;
    ~ModelFile();

    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    // Load either a .gguf (v2/v3) or .safetensors file
    bool open(const std::string& path, std::string& err_msg);
    void close();

    bool is_open() const { return mapped_base_ != nullptr; }
    const std::string& path() const { return path_; }
    const std::string& format() const { return format_; } // "GGUFv3" or "SafeTensors"
    uint64_t file_size() const { return file_size_; }
    uint64_t data_section_offset() const { return data_offset_; }

    const std::unordered_map<std::string, GGUFValue>& metadata() const { return metadata_; }
    const std::unordered_map<std::string, TensorDescriptor>& tensors() const { return tensors_; }
    const std::vector<std::string>& tensor_names() const { return tensor_order_; }

    const TensorDescriptor* find_tensor(const std::string& name) const;
    uint32_t get_u32(const std::string& key, uint32_t def = 0) const;
    float    get_f32(const std::string& key, float def = 0.0f) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;

    const uint8_t* mapped_base() const { return mapped_base_; }

private:
    bool parse_gguf(std::string& err_msg);
    bool parse_safetensors(std::string& err_msg);

    std::string path_;
    std::string format_;
    uint64_t file_size_ = 0;
    uint64_t data_offset_ = 0;
    uint32_t alignment_ = 32;

    const uint8_t* mapped_base_ = nullptr;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE map_handle_  = nullptr;
#else
    int fd_ = -1;
#endif

    std::unordered_map<std::string, GGUFValue> metadata_;
    std::unordered_map<std::string, TensorDescriptor> tensors_;
    std::vector<std::string> tensor_order_;
};

} // namespace moe

