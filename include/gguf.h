#pragma once

#include "common.h"
#include <deque>
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
    // (".experts." / "block_sparse_moe.experts.") that could NOT be packed
    // into the packed 3D expert tensors the engine expects (see
    // pack_safetensors_experts()). When packing succeeds the flag is
    // cleared and blk.<l>.ffn_{gate,up,down}_exps.weight descriptors exist.
    bool safetensors_moe_unsupported() const { return st_moe_unsupported_; }
    // HF tensor names seen in the file that had no translation (e.g. fused
    // "qkv_proj" weights, MoE experts). The original names are still stored
    // and listed by tensor_names().
    const std::vector<std::string>& safetensors_untranslated() const { return st_untranslated_; }
    // SafeTensors dtypes with no GGMLType mapping (left as F32, see gguf.cpp).
    const std::vector<std::string>& safetensors_unknown_dtypes() const { return st_unknown_dtypes_; }
    // True when this file is one shard of a multi-file SafeTensors model
    // (model-0000N-of-0000M.safetensors): all shards were loaded, so
    // tensors() spans the whole model.
    bool is_sharded_safetensors() const { return st_shard_count_ > 1; }
    uint32_t safetensors_shard_count() const { return st_shard_count_; }
    // End of the KV section in a GGUF file (== start of the tensor infos).
    // Only meaningful for GGUF files; used by merge_gguf_files().
    uint64_t kv_section_end() const { return kv_end_offset_; }
    // Split a fused SafeTensors attention tensor (Phi3 qkv_proj, aliased as
    // blk.<layer>.attn_qkv.weight) into separate blk.<layer>.attn_q/k/v
    // (.bias) descriptors. q_dim = n_heads * head_dim, kv_dim =
    // n_kv_heads * head_dim. The new descriptors are views into the fused
    // tensor's mmap (no data is copied). Returns true when the split
    // tensors already exist or were just created; false with err_msg when
    // no fused tensor exists for the layer or its dims do not match
    // q_dim + 2*kv_dim. Safe to call when no fused tensor exists (then it
    // returns false and the caller keeps its previous behaviour).
    bool split_fused_qkv(uint32_t layer, uint32_t q_dim, uint32_t kv_dim,
                         std::string& err_msg);

private:
    bool parse_gguf(std::string& err_msg);
    bool parse_safetensors_shard(const uint8_t* base, uint64_t size,
                                 uint64_t& data_offset_out, std::string& err_msg);
    // Open every shard of a multi-file SafeTensors model (paths in shard
    // order). Used when open() is given a directory or one shard of a
    // model-0000N-of-0000M.safetensors group.
    bool open_safetensors_group(const std::vector<std::string>& shard_paths,
                                std::string& err_msg);
    // Post-pass after all SafeTensors shards + config.json are loaded:
    // pack per-expert tensors into contiguous packed 3D
    // blk.<l>.ffn_{gate,up,down}_exps.weight descriptors (experts stacked
    // along dim 0, one expert per contiguous byte slice — exactly the
    // layout bind_tensors() expects). Two source layouts are handled:
    // per-expert 2D tensors (mlp.experts.<i>.gate/up/down_proj.weight,
    // block_sparse_moe.experts.<i>.w1/w2/w3 — production checkpoints) and
    // already-packed 3D tensors (mlp.experts.gate_up_proj [E,2I,H] +
    // down_proj [E,H,I], transformers 5.x). The packed bytes are copied
    // once into packed_buffers_ (owned by this ModelFile); the per-expert
    // source tensors stay listed under their original names. Layers whose
    // experts are incomplete or inconsistent are skipped with a warning
    // and keep st_moe_unsupported_ set.
    bool pack_safetensors_experts();
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
    // Multi-file SafeTensors: mappings for shards beyond the first (the
    // first shard lives in mapped_base_/fd_/file_size_ above). Each shard's
    // tensors keep mapped_ptr/abs_file_offset into their own mapping.
    struct ShardMapping {
#ifdef _WIN32
        HANDLE file_handle = INVALID_HANDLE_VALUE;
        HANDLE map_handle = nullptr;
#else
        int fd = -1;
#endif
        const uint8_t* base = nullptr;
        uint64_t size = 0;
    };
    std::vector<ShardMapping> extra_shards_;
    uint32_t st_shard_count_ = 0;
    // Owned packed MoE expert buffers (see pack_safetensors_experts()).
    // deque: element addresses never move, so TensorDescriptor::mapped_ptr
    // values pointing into them stay valid while the ModelFile is open.
    std::deque<std::vector<uint8_t>> packed_buffers_;
    uint64_t kv_end_offset_ = 0;
};

// Merge sharded GGUF files (model-00001-of-00002.gguf, ...) into one valid
// GGUF file: the KV/metadata section is copied verbatim from the first
// shard, all tensors are combined in shard order with fresh aligned
// offsets. Returns false with err_msg on any problem (missing/unreadable
// shard, version/architecture mismatch, duplicate tensor name, I/O error).
// The output is re-opened and verified (tensor count + spot-checked bytes)
// before returning true.
bool merge_gguf_files(const std::vector<std::string>& shard_paths,
                      const std::string& out_path, std::string& err_msg);

// Re-quantize a GGUF file to `target_type` using the engine's own quantizers
// (quant.cpp): every 2-D+ weight tensor whose row length is a multiple of
// both the source and target block sizes is dequantized to F32 and
// re-quantized row by row; everything else is copied byte-for-byte and
// reported (1-D norm/bias tensors, tensors already in the target type,
// tensors whose source type has no dequantizer, tensors whose row length
// is not a multiple of the target block size). Nothing is ever written in
// the wrong type: a tensor that cannot be re-quantized exactly keeps its
// original type. Mixed-type targets (Q4_K_M etc.) are out of scope —
// those are per-tensor policies, not a single target type.
// The output is re-opened and verified (tensor count + types + byte sizes)
// before returning true. Input must be GGUF; SafeTensors input is refused
// (use the engine's direct SafeTensors loading instead).
bool quantize_gguf_file(const std::string& in_path, GGMLType target_type,
                        const std::string& out_path, std::string& err_msg);

// Parse a quant type name ("Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K",
// "Q4_1", "Q5_0", "Q5_1", "Q8_1", "Q2_K", "Q3_K", "Q8_K") into a GGMLType.
// Returns false for anything else — including the IQ types, which have no
// quantizer in quant.cpp, and F32 (not a quantization).
bool parse_quant_type_name(const std::string& name, GGMLType& out);

} // namespace moe

