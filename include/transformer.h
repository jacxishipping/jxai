#pragma once

#include "common.h"
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"
#include "tier.h"
#include "pilot_io.h"
#include <deque>
#include <functional>

namespace moe {

struct ModelHyperparams {
    std::string arch = "qwen2moe";
    std::string model_name = "MoE-Stream-Engine";
    uint32_t vocab_size = 512;
    uint32_t dim = 128;
    uint32_t hidden_dim = 256;
    uint32_t moe_expert_dim = 128;
    uint32_t n_layers = 4;
    uint32_t n_heads = 4;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim = 32;
    uint32_t rope_dim = 32;
    uint32_t n_experts = 8;
    uint32_t n_experts_used = 2;
    uint32_t max_seq_len = 2048;
    float    rms_norm_eps = 1e-5f;
    float    rope_freq_base = 10000.0f;
    float    yarn_scale = 1.0f;
};

struct LayerDenseWeights {
    const TensorDescriptor* attn_norm = nullptr;
    const TensorDescriptor* wq = nullptr;
    const TensorDescriptor* wk = nullptr;
    const TensorDescriptor* wv = nullptr;
    const TensorDescriptor* wo = nullptr;
    const TensorDescriptor* bq = nullptr;
    const TensorDescriptor* bk = nullptr;
    const TensorDescriptor* bv = nullptr;
    const TensorDescriptor* attn_q_norm = nullptr;
    const TensorDescriptor* attn_k_norm = nullptr;
    const TensorDescriptor* ffn_norm = nullptr;
    const TensorDescriptor* router_gate = nullptr; // [n_experts, dim]

    // Sparse experts on NVMe disk (per expert slice offsets)
    GGMLType expert_gate_type = GGMLType::Q4_0;
    GGMLType expert_up_type   = GGMLType::Q4_0;
    GGMLType expert_down_type = GGMLType::Q4_0;
    std::vector<ExpertDiskSlice> expert_slices;
};

struct KVCacheSlot {
    uint32_t slot_id = 0;
    uint32_t seq_pos = 0;
    uint32_t compactions = 0;
    std::vector<uint32_t> token_history;
    // [n_layers][max_seq_len * n_kv_heads * head_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    void init(uint32_t id, const ModelHyperparams& hp);
    void clear();
    // Compact KV cache by retaining sink tokens (first 4) + most recent window
    void compact(const ModelHyperparams& hp, uint32_t keep_recent = 128);
};

struct SamplerConfig {
    float    temperature = 0.7f;
    uint32_t top_k = 40;
    float    top_p = 0.9f;
    float    repetition_penalty = 1.10f;
    uint64_t seed = 42;
};

struct SystemOneOptionScore {
    std::string option;
    float raw_logit = 0.0f;
    float probability = 0.0f;
    uint32_t token_id = 0;
};

class MoETransformerEngine {
public:
    MoETransformerEngine() = default;
    ~MoETransformerEngine() { unload(); }

    // Load GGUF v3 or SafeTensors model file, initialize dense mmap backbone, PILOT I/O, and LFRU tier cache
    bool load_model(const std::string& path, std::string& err_msg);
    void unload() {
        std::lock_guard<std::mutex> lk(engine_mtx_);
        loaded_ = false;
        io_engine_.shutdown();
        model_file_.close();
    }

    bool is_loaded() const { return loaded_; }
    const ModelHyperparams& hparams() const { return hp_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    const HardwareSpecs& hardware() const { return hw_; }
    LFRUTierManager& tier_manager() { return tier_mgr_; }
    const LFRUTierManager& tier_manager() const { return tier_mgr_; }
    const PilotIOEngine& io_engine() const { return io_engine_; }
    const ModelFile& model_file() const { return model_file_; }

    // Math primitives exposed for unit testing & verification
    static void rms_norm(float* out, const float* x, const float* weight, int64_t size, float eps);
    static void apply_rope(
        float* q, float* k,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
        uint32_t pos, float freq_base, float yarn_scale = 1.0f, bool is_neox = true
    );
    static void softmax(float* x, int64_t size);
    static float silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Execute a single real forward pass for `token_id` at current position in `slot_id`
    // Returns vocabulary logits vector [vocab_size] and records ProfileSample
    std::vector<float> forward_token(
        uint32_t token_id,
        uint32_t slot_id,
        ProfileSample* out_profile = nullptr
    );

    // Sample next token ID from logits using temperature, top-K, top-P, repetition penalty
    uint32_t sample_next_token(
        const std::vector<float>& logits,
        const std::vector<uint32_t>& history,
        const SamplerConfig& cfg
    );

    // Stream token generation with dual-channel <think> reasoning detection
    using StreamCallback = std::function<bool(
        const std::string& token_text,
        bool is_reasoning,
        uint32_t token_id,
        const ProfileSample& profile
    )>;

    void generate_stream(
        const std::string& prompt,
        uint32_t max_new_tokens,
        uint32_t slot_id,
        bool enable_context_compaction,
        const SamplerConfig& sampler_cfg,
        StreamCallback on_token,
        bool continue_assistant = false,
        bool enable_speculative = false,
        uint32_t draft_count = 2
    );

    // Multi-Token Prediction (MTP) speculative batch verification helper
    std::vector<uint32_t> verify_speculative_batch(
        uint32_t slot_id,
        const std::vector<uint32_t>& draft_tokens,
        const SamplerConfig& sampler_cfg,
        ProfileSample* out_profile = nullptr
    );

    // SystemOne zero-token-generation categorical decision engine
    std::vector<SystemOneOptionScore> evaluate_system_one(
        const std::string& document,
        const std::string& query,
        const std::vector<std::string>& options,
        ProfileSample* out_profile = nullptr
    );

    KVCacheSlot& get_or_create_slot(uint32_t slot_id);
    void reset_slot(uint32_t slot_id);
    size_t active_slot_count() const;

    std::vector<ProfileSample> recent_profiles() const;
    uint64_t dense_backbone_bytes() const { return dense_bytes_; }
    uint64_t sparse_disk_bytes() const { return sparse_bytes_; }

private:
    bool bind_tensors(std::string& err_msg);

    bool loaded_ = false;
    HardwareSpecs hw_;
    ModelFile model_file_;
    Tokenizer tokenizer_;
    ModelHyperparams hp_;
    LFRUTierManager tier_mgr_;
    PilotIOEngine io_engine_;

    const TensorDescriptor* tok_emb_ = nullptr;
    const TensorDescriptor* out_norm_ = nullptr;
    const TensorDescriptor* lm_head_ = nullptr;
    std::vector<LayerDenseWeights> layers_;

    uint64_t dense_bytes_ = 0;
    uint64_t sparse_bytes_ = 0;

    mutable std::mutex engine_mtx_;
    mutable std::mutex profile_mtx_;
    std::unordered_map<uint32_t, KVCacheSlot> kv_slots_;
    std::deque<ProfileSample> profile_history_;
};

} // namespace moe
