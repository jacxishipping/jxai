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
    uint32_t q_dim = 128;
    uint32_t kv_dim = 64;
    uint32_t rope_dim = 32;
    uint32_t n_experts = 8;
    uint32_t n_experts_used = 2;
    uint32_t max_seq_len = 2048;
    uint32_t kv_cache_capacity = 2048;
    uint32_t ple_dim = 0;
    uint32_t sliding_window = 0;
    uint32_t shared_kv_layers = 0;
    uint32_t rope_dim_swa = 0;
    float    rope_freq_base_swa = 10000.0f;
    float    final_logit_softcapping = 0.0f;
    float    attn_logit_softcapping = 0.0f; // Gemma2/3: <arch>.attn_logit_softcapping
    bool     use_gelu = false;
    bool     scale_embeddings = false;
    bool     v_norm = false;
    float    rms_norm_eps = 1e-5f;
    float    rope_freq_base = 10000.0f;
    float    yarn_scale = 1.0f;
    // RoPE scaling (GGUF: <arch>.rope.scaling.*). yarn_scale is only set
    // from .factor when type == "yarn"; otherwise it stays 1.0.
    std::string rope_scaling_type; // "", "none", "linear", "yarn", ...
    float    rope_scaling_factor = 1.0f;
    float    yarn_log_multiplier = 0.0f;
    uint32_t rope_orig_ctx_len = 0;
    bool     rope_neox = true; // computed in bind_tensors(); Neox is the default
};

struct LayerDenseWeights {
    uint32_t q_dim = 0;
    uint32_t kv_dim = 0;
    uint32_t head_dim = 0;
    uint32_t rope_dim = 0;
    uint32_t kv_source_layer = 0;
    bool     shares_kv = false;
    float    rope_freq_base = 10000.0f;
    bool     is_swa = false;

    const TensorDescriptor* attn_norm = nullptr;
    const TensorDescriptor* wq = nullptr;
    const TensorDescriptor* wk = nullptr;
    const TensorDescriptor* wv = nullptr;
    const TensorDescriptor* wo = nullptr;
    const TensorDescriptor* bq = nullptr;
    const TensorDescriptor* bk = nullptr;
    const TensorDescriptor* bv = nullptr;
    const TensorDescriptor* bo = nullptr;
    const TensorDescriptor* attn_q_norm = nullptr;
    const TensorDescriptor* attn_k_norm = nullptr;
    const TensorDescriptor* post_attn_norm = nullptr;
    const TensorDescriptor* ffn_norm = nullptr;
    const TensorDescriptor* post_ffw_norm = nullptr;
    const TensorDescriptor* inp_gate = nullptr;
    const TensorDescriptor* ple_proj = nullptr;
    const TensorDescriptor* post_norm = nullptr;
    const TensorDescriptor* layer_output_scale = nullptr;
    const TensorDescriptor* router_gate = nullptr; // [n_experts, dim]

    // Shared Expert weights (Qwen2.5-MoE / DeepSeek-MoE)
    const TensorDescriptor* shexp_gate = nullptr;
    const TensorDescriptor* shexp_up   = nullptr;
    const TensorDescriptor* shexp_down = nullptr;
    const TensorDescriptor* shexp_gate_inp = nullptr;

    // Sparse experts on NVMe disk (per expert slice offsets) or zero-copy mapped FFN
    bool is_dense_ffn = false;
    GGMLType expert_gate_type = GGMLType::Q4_0;
    GGMLType expert_up_type   = GGMLType::Q4_0;
    GGMLType expert_down_type = GGMLType::Q4_0;
    std::vector<ExpertDiskSlice> expert_slices;
};

constexpr uint32_t KV_BLOCK_TOKENS = 16;

struct PagedBlockStats {
    size_t total_blocks = 1024;
    size_t used_blocks = 0;
    size_t free_blocks = 1024;
    size_t block_size_tokens = KV_BLOCK_TOKENS;
    size_t memory_bytes = 0;
};

struct KVSlotSummary {
    uint32_t slot_id = 0;
    uint32_t seq_pos = 0;
    uint32_t pages_allocated = 0;
    uint32_t compactions = 0;
    size_t token_count = 0;
    size_t memory_bytes = 0;
};

struct KVCacheSlot {
    uint32_t slot_id = 0;
    uint32_t seq_pos = 0;
    uint32_t compactions = 0;
    std::vector<uint32_t> token_history;
    std::vector<uint32_t> page_table;
    // FP16 KV Cache: [n_layers][max_seq_len * kv_dim]
    std::vector<std::vector<uint16_t>> k_cache;
    std::vector<std::vector<uint16_t>> v_cache;

    void init(uint32_t id, const ModelHyperparams& hp);
    void clear();
    // Compact KV cache by retaining sink tokens (first 4) + most recent window
    void compact(const ModelHyperparams& hp, uint32_t keep_recent = 128);
    void rollback(uint32_t new_seq_pos);
    uint32_t pages_allocated() const {
        return (seq_pos == 0) ? 0 : static_cast<uint32_t>((seq_pos + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS);
    }
};

struct SamplerConfig {
    float    temperature = 0.7f;
    uint32_t top_k = 40;
    float    top_p = 0.9f;
    float    repetition_penalty = 1.10f;
    uint64_t seed = 42;
};

// --- Universal-model helpers (shared by CLI chat and the HTTP server) ---
// RoPE style: Neox (rotate-half) is the default for essentially every modern
// LLM (Llama, Qwen3, Mistral, Phi3/4, ...). Only GPT-2 / GPT-J style models
// use interleaved (GPT-J) RoPE.
bool arch_uses_neox_rope(const std::string& arch);
bool arch_is_known(const std::string& arch);

struct ChatMessage {
    std::string role;
    std::string content;
};

// Render chat messages to a prompt string. Source of truth, in order:
//  1. `chat_template` from GGUF metadata (tokenizer.chat_template), rendered
//     with a minimal Jinja subset (for/endfor, if/elif/else, {{ expr }} with
//     string literals + message['role'/'content'] + bos/eos_token).
//  2. Per-architecture fallback template (ChatML / Llama3 / Mistral / Gemma / Phi).
// If source_out is non-null it receives "gguf-chat-template" or
// "fallback:<name>". BOS is NOT emitted by these templates; the caller adds
// it via encode(..., should_add_bos(...)).
std::string format_chat_messages(
    const std::vector<ChatMessage>& msgs,
    const std::string& arch,
    const std::string& chat_template,
    const Tokenizer& tok,
    bool add_generation_prompt = true,
    std::string* source_out = nullptr
);

// Wrap a raw (non-templated) user prompt as a single-user chat turn. Prompts
// that already contain template markers are returned unchanged.
std::string wrap_raw_prompt_for_chat(
    const std::string& prompt,
    const std::string& arch,
    const std::string& chat_template,
    const Tokenizer& tok
);

// BOS policy: tokenizer.ggml.add_bos_token wins when present in the GGUF;
// otherwise per-architecture defaults (true for Llama/Gemma/Mistral/Phi,
// false for Qwen/DeepSeek/GPT-2 family).
bool should_add_bos(const std::string& arch, const Tokenizer& tok);

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
        uint32_t pos, float freq_base, float yarn_scale = 1.0f, bool is_neox = true,
        const float* rope_freqs = nullptr
    );
    static void softmax(float* x, int64_t size);
    static float silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Execute a single real forward pass for `token_id` at current position in `slot_id`
    // Returns vocabulary logits vector [vocab_size] (or empty if compute_logits == false) and records ProfileSample
    std::vector<float> forward_token(
        uint32_t token_id,
        uint32_t slot_id,
        ProfileSample* out_profile = nullptr,
        bool compute_logits = true,
        std::vector<float>* out_hidden = nullptr
    );

    // Compute L2-normalized mean-pooled hidden-state embedding for `text`
    std::vector<float> compute_embeddings(const std::string& text);

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
    bool clear_slot(uint32_t slot_id);
    size_t active_slot_count() const;
    std::vector<KVSlotSummary> list_slots() const;
    PagedBlockStats paged_block_stats() const;

    std::vector<ProfileSample> recent_profiles() const;
    uint64_t dense_backbone_bytes() const { return dense_bytes_; }
    uint64_t sparse_disk_bytes() const { return sparse_bytes_; }

    // Universal-model introspection (used by `moe_engine info` and generation)
    // Stop set built from the loaded tokenizer/model: eos_id plus any token
    // found for <|end_of_text|>, <|im_end|>, <|end_of_turn|>, <|eot_id|>,
    // </s>, <|endoftext|>. Hardcoded vendor IDs are NOT used.
    std::vector<uint32_t> stop_token_ids() const;
    bool is_stop_token(uint32_t id) const;
    // "gguf-chat-template" | "fallback:<name>" | "none"
    std::string chat_template_source() const;
    const std::string& chat_template() const { return tokenizer_.chat_template(); }

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
    const TensorDescriptor* per_layer_tok_emb_ = nullptr;
    const TensorDescriptor* per_layer_model_proj_ = nullptr;
    const TensorDescriptor* per_layer_proj_norm_ = nullptr;
    const TensorDescriptor* rope_freqs_ = nullptr;
    std::vector<LayerDenseWeights> layers_;

    uint64_t dense_bytes_ = 0;
    uint64_t sparse_bytes_ = 0;

    mutable std::mutex engine_mtx_;
    mutable std::mutex profile_mtx_;
    std::unordered_map<uint32_t, KVCacheSlot> kv_slots_;
    std::deque<ProfileSample> profile_history_;
};

} // namespace moe
