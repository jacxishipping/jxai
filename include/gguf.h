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

    // ---- SafeTensors + sibling config.json support ----
    // True when this file is a .safetensors model. A config.json found next
    // to the model is parsed (best-effort, no JSON library) and its fields
    // are stored as synthetic GGUF-style metadata, so the existing
    // get_u32/get_f32/get_str calls in the engine find them:
    //   general.architecture            e.g. "qwen2" (mapped from
    //                                      "Qwen2ForCausalLM", see the
    //                                      arch table in gguf.cpp)
    //   <arch>.embedding_length        hidden_size
    //   <arch>.feed_forward_length     intermediate_size
    //   <arch>.block_count             num_hidden_layers
    //   <arch>.attention.head_count    num_attention_heads
    //   <arch>.attention.head_count_kv num_key_value_heads
    //   <arch>.context_length          max_position_embeddings
    //   <arch>.attention.layer_norm_rms_epsilon  rms_norm_eps
    //   <arch>.rope.freq_base          rope_theta
    //   <arch>.expert_count / .expert_used_count  num_experts /
    //                                      num_experts_per_tok (MoE only)
    // HF tensor names are translated at parse time (see the alias table in
    // gguf.cpp), so find_tensor("blk.0.attn_q.weight") resolves a file
    // containing "model.layers.0.self_attn.q_proj.weight". 2D tensor dims
    // are stored in GGUF order [in, out] (reversed from the SafeTensors
    // [out, in] shape); the data layout is unchanged (row-major [out][in],
    // which is what matmul_quant_f32 expects).
    bool is_safetensors() const { return format_ == "SafeTensors"; }
    bool config_loaded() const { return config_loaded_; }
    const std::string& detected_arch() const { return detected_arch_; }
    const std::string& config_path() const { return config_path_; }
    // Path to a sibling tokenizer.json, if one exists next to the model.
    // Recorded only — tokenizer.json parsing belongs to the tokenizer layer.
    const std::string& tokenizer_path() const { return tokenizer_path_; }
    // True when the file contains MoE routed-expert tensors
    // (".experts." / "block_sparse_moe.experts."). Those are NOT translated:
    // the engine expects packed 3D expert tensors (ffn_gate_exps.weight),
    // which cannot be faked from per-expert SafeTensors files.
    bool safetensors_moe_unsupported() const { return st_moe_unsupported_; }
    // HF tensor names seen in the file that had no translation (e.g. fused
    // "qkv_proj" weights, MoE experts). The original names are still stored
    // and listed by tensor_names().
    const std::vector<std::string>& safetensors_untranslated() const { return st_untranslated_; }
    // SafeTensors dtypes with no GGMLType mapping (left as F32, see gguf.cpp).
    const std::vector<std::string>& safetensors_unknown_dtypes() const { return st_unknown_dtypes_; }

private:
    bool parse_gguf(std::string& err_msg);
    bool parse_safetensors(std::string& err_msg);
    // Best-effort: load sibling config.json / tokenizer.json. Never fails
    // open(); returns false when no usable config was found.
    bool load_safetensors_config();

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
    // GGUF name -> original HF name for SafeTensors files (find_tensor
    // consults this map; tensors_ itself only stores original names, so
    // byte accounting over tensors() is not double-counted).
    std::unordered_map<std::string, std::string> alias_;

    bool config_loaded_ = false;
    std::string detected_arch_;
    std::string config_path_;
    std::string tokenizer_path_;
    bool st_moe_unsupported_ = false;
    std::vector<std::string> st_untranslated_;
    std::vector<std::string> st_unknown_dtypes_;
};

} // namespace moe

