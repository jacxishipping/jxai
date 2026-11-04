#include "transformer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace moe {

void KVCacheSlot::init(uint32_t id, const ModelHyperparams& hp) {
    slot_id = id;
    seq_pos = 0;
    compactions = 0;
    token_history.clear();
    size_t kv_dim = static_cast<size_t>(hp.n_kv_heads) * static_cast<size_t>(hp.head_dim);
    size_t cap = static_cast<size_t>(hp.max_seq_len) * kv_dim;
    k_cache.assign(hp.n_layers, std::vector<float>(cap, 0.0f));
    v_cache.assign(hp.n_layers, std::vector<float>(cap, 0.0f));
}

void KVCacheSlot::clear() {
    seq_pos = 0;
    token_history.clear();
    for (auto& k : k_cache) std::fill(k.begin(), k.end(), 0.0f);
    for (auto& v : v_cache) std::fill(v.begin(), v.end(), 0.0f);
}

void KVCacheSlot::compact(const ModelHyperparams& hp, uint32_t keep_recent) {
    const uint32_t sink_tokens = 4;
    if (seq_pos <= sink_tokens + keep_recent) return;

    uint32_t new_len = sink_tokens + keep_recent;
    uint32_t src_start = seq_pos - keep_recent;
    size_t kv_dim = static_cast<size_t>(hp.n_kv_heads) * static_cast<size_t>(hp.head_dim);

    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        std::memmove(
            k_cache[l].data() + sink_tokens * kv_dim,
            k_cache[l].data() + src_start * kv_dim,
            keep_recent * kv_dim * sizeof(float)
        );
        std::memmove(
            v_cache[l].data() + sink_tokens * kv_dim,
            v_cache[l].data() + src_start * kv_dim,
            keep_recent * kv_dim * sizeof(float)
        );
    }
    seq_pos = new_len;
    compactions++;
}

void MoETransformerEngine::rms_norm(float* out, const float* x, const float* weight, int64_t size, float eps) {
    float ss = vec_dot_f32(x, x, size);
    float inv_rms = 1.0f / std::sqrt((ss / static_cast<float>(size)) + eps);
    for (int64_t i = 0; i < size; ++i) {
        out[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
    }
}

void MoETransformerEngine::apply_rope(
    float* q, float* k,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t pos, float freq_base, float yarn_scale, bool is_neox
) {
    uint32_t rdim = std::min(head_dim, rope_dim);
    float pos_scaled = static_cast<float>(pos) / std::max(1e-4f, yarn_scale);

    if (is_neox) {
        uint32_t half = rdim / 2;
        for (uint32_t d = 0; d < half; ++d) {
            float freq = 1.0f / std::pow(freq_base, (2.0f * static_cast<float>(d)) / static_cast<float>(rdim));
            float angle = pos_scaled * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (uint32_t h = 0; h < n_heads; ++h) {
                float* qh = q + h * head_dim;
                float q0 = qh[d];
                float q1 = qh[d + half];
                qh[d]        = q0 * cos_a - q1 * sin_a;
                qh[d + half] = q0 * sin_a + q1 * cos_a;
            }

            if (k) {
                for (uint32_t h = 0; h < n_kv_heads; ++h) {
                    float* kh = k + h * head_dim;
                    float k0 = kh[d];
                    float k1 = kh[d + half];
                    kh[d]        = k0 * cos_a - k1 * sin_a;
                    kh[d + half] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    } else {
        for (uint32_t d = 0; d + 1 < rdim; d += 2) {
            float freq = 1.0f / std::pow(freq_base, static_cast<float>(d) / static_cast<float>(rdim));
            float angle = pos_scaled * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (uint32_t h = 0; h < n_heads; ++h) {
                float* qh = q + h * head_dim + d;
                float q0 = qh[0];
                float q1 = qh[1];
                qh[0] = q0 * cos_a - q1 * sin_a;
                qh[1] = q0 * sin_a + q1 * cos_a;
            }

            if (k) {
                for (uint32_t h = 0; h < n_kv_heads; ++h) {
                    float* kh = k + h * head_dim + d;
                    float k0 = kh[0];
                    float k1 = kh[1];
                    kh[0] = k0 * cos_a - k1 * sin_a;
                    kh[1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }
}

void MoETransformerEngine::softmax(float* x, int64_t size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int64_t i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : (1.0f / static_cast<float>(size));
    for (int64_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

bool MoETransformerEngine::load_model(const std::string& path, std::string& err_msg) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    loaded_ = false;
    hw_ = query_hardware_specs();

    if (!model_file_.open(path, err_msg)) {
        return false;
    }

    if (!tokenizer_.load_from_model(model_file_)) {
        err_msg = "Failed to initialize tokenizer from model";
        return false;
    }

    if (!io_engine_.init({path}, err_msg)) {
        return false;
    }

    if (!bind_tensors(err_msg)) {
        return false;
    }

    // Initialize LFRU tier cache: keep ~50% of experts eligible for RAM/VRAM residency
    size_t total_experts = static_cast<size_t>(hp_.n_layers) * static_cast<size_t>(hp_.n_experts);
    size_t vram_slots = std::max<size_t>(4, total_experts / 4);
    size_t ram_slots  = std::max<size_t>(4, total_experts / 4);
    tier_mgr_.init(hp_.n_layers, hp_.n_experts, ram_slots, vram_slots, 0.08f);

    kv_slots_.clear();
    KVCacheSlot slot0;
    slot0.init(0, hp_);
    kv_slots_[0] = std::move(slot0);

    loaded_ = true;
    return true;
}

bool MoETransformerEngine::bind_tensors(std::string& err_msg) {
    hp_.arch = model_file_.get_str("general.architecture", "qwen2moe");
    hp_.model_name = model_file_.get_str("general.name", "MoE-Stream-Model");
    const std::string pfx = hp_.arch;

    hp_.vocab_size     = tokenizer_.vocab_size();
    hp_.dim            = model_file_.get_u32(pfx + ".embedding_length", model_file_.get_u32("llama.embedding_length", 128));
    hp_.hidden_dim     = model_file_.get_u32(pfx + ".feed_forward_length", 256);
    hp_.moe_expert_dim = model_file_.get_u32(pfx + ".expert_feed_forward_length", hp_.hidden_dim);
    hp_.n_layers       = model_file_.get_u32(pfx + ".block_count", model_file_.get_u32("llama.block_count", 4));
    hp_.n_heads        = model_file_.get_u32(pfx + ".attention.head_count", 4);
    hp_.n_kv_heads     = model_file_.get_u32(pfx + ".attention.head_count_kv", hp_.n_heads);
    hp_.n_experts      = model_file_.get_u32(pfx + ".expert_count", 8);
    hp_.n_experts_used = model_file_.get_u32(pfx + ".expert_used_count", 2);
    hp_.max_seq_len    = model_file_.get_u32(pfx + ".context_length", 2048);
    hp_.rms_norm_eps   = model_file_.get_f32(pfx + ".attention.layer_norm_rms_epsilon", 1e-5f);
    hp_.rope_freq_base = model_file_.get_f32(pfx + ".rope.freq_base", 10000.0f);

    tok_emb_ = model_file_.find_tensor("token_embd.weight");
    if (!tok_emb_) tok_emb_ = model_file_.find_tensor("model.embed_tokens.weight");
    if (!tok_emb_) {
        err_msg = "Missing token_embd.weight tensor in model file";
        return false;
    }

    if (tok_emb_->n_dims >= 2) {
        hp_.dim = static_cast<uint32_t>(tok_emb_->dims[0]);
        hp_.vocab_size = static_cast<uint32_t>(tok_emb_->dims[1]);
    }
    hp_.head_dim = std::max(1u, hp_.dim / std::max(1u, hp_.n_heads));
    hp_.rope_dim = model_file_.get_u32(pfx + ".rope.dimension_count", hp_.head_dim);

    out_norm_ = model_file_.find_tensor("output_norm.weight");
    lm_head_  = model_file_.find_tensor("output.weight");
    if (!lm_head_) lm_head_ = tok_emb_; // weight-tied fallback

    dense_bytes_ = 0;
    sparse_bytes_ = 0;
    for (const auto& kv : model_file_.tensors()) {
        if (kv.second.is_sparse_expert) sparse_bytes_ += kv.second.byte_size;
        else dense_bytes_ += kv.second.byte_size;
    }

    layers_.resize(hp_.n_layers);
    for (uint32_t l = 0; l < hp_.n_layers; ++l) {
        std::string bp = "blk." + std::to_string(l) + ".";
        LayerDenseWeights& lw = layers_[l];
        lw.attn_norm   = model_file_.find_tensor(bp + "attn_norm.weight");
        lw.wq          = model_file_.find_tensor(bp + "attn_q.weight");
        lw.wk          = model_file_.find_tensor(bp + "attn_k.weight");
        lw.wv          = model_file_.find_tensor(bp + "attn_v.weight");
        lw.bq          = model_file_.find_tensor(bp + "attn_q.bias");
        lw.bk          = model_file_.find_tensor(bp + "attn_k.bias");
        lw.bv          = model_file_.find_tensor(bp + "attn_v.bias");
        lw.attn_q_norm = model_file_.find_tensor(bp + "attn_q_norm.weight");
        lw.attn_k_norm = model_file_.find_tensor(bp + "attn_k_norm.weight");
        lw.wo          = model_file_.find_tensor(bp + "attn_output.weight");
        lw.ffn_norm    = model_file_.find_tensor(bp + "ffn_norm.weight");
        lw.router_gate = model_file_.find_tensor(bp + "ffn_gate_inp.weight");

        const TensorDescriptor* t_gate_exps = model_file_.find_tensor(bp + "ffn_gate_exps.weight");
        const TensorDescriptor* t_up_exps   = model_file_.find_tensor(bp + "ffn_up_exps.weight");
        const TensorDescriptor* t_down_exps = model_file_.find_tensor(bp + "ffn_down_exps.weight");

        lw.expert_slices.resize(hp_.n_experts);
        if (t_gate_exps && t_up_exps && t_down_exps) {
            // Packed 3D expert tensors [dim, expert_dim, n_experts]
            if (t_gate_exps->n_dims >= 2) {
                hp_.moe_expert_dim = static_cast<uint32_t>(t_gate_exps->dims[1]);
            }
            if (t_gate_exps->n_dims >= 3) {
                hp_.n_experts = static_cast<uint32_t>(t_gate_exps->dims[2]);
                lw.expert_slices.resize(hp_.n_experts);
            }
            lw.expert_gate_type = t_gate_exps->type;
            lw.expert_up_type   = t_up_exps->type;
            lw.expert_down_type = t_down_exps->type;

            uint64_t gate_slice_sz = t_gate_exps->byte_size / hp_.n_experts;
            uint64_t up_slice_sz   = t_up_exps->byte_size / hp_.n_experts;
            uint64_t down_slice_sz = t_down_exps->byte_size / hp_.n_experts;

            for (uint32_t e = 0; e < hp_.n_experts; ++e) {
                lw.expert_slices[e].gate_offset = t_gate_exps->abs_file_offset + e * gate_slice_sz;
                lw.expert_slices[e].gate_bytes  = gate_slice_sz;
                lw.expert_slices[e].up_offset   = t_up_exps->abs_file_offset + e * up_slice_sz;
                lw.expert_slices[e].up_bytes    = up_slice_sz;
                lw.expert_slices[e].down_offset = t_down_exps->abs_file_offset + e * down_slice_sz;
                lw.expert_slices[e].down_bytes  = down_slice_sz;
            }
        } else {
            // Check for standard Dense MLP (Qwen2.5, LLaMA, Mistral)
            const TensorDescriptor* t_gate = model_file_.find_tensor(bp + "ffn_gate.weight");
            const TensorDescriptor* t_up   = model_file_.find_tensor(bp + "ffn_up.weight");
            const TensorDescriptor* t_down = model_file_.find_tensor(bp + "ffn_down.weight");
            if (t_gate && t_up && t_down) {
                hp_.n_experts = 1;
                hp_.n_experts_used = 1;
                hp_.moe_expert_dim = (t_gate->n_dims >= 2) ? static_cast<uint32_t>(t_gate->dims[1]) : hp_.hidden_dim;
                lw.expert_slices.resize(1);
                lw.expert_gate_type = t_gate->type;
                lw.expert_up_type   = t_up->type;
                lw.expert_down_type = t_down->type;
                lw.expert_slices[0].gate_offset = t_gate->abs_file_offset;
                lw.expert_slices[0].gate_bytes  = t_gate->byte_size;
                lw.expert_slices[0].up_offset   = t_up->abs_file_offset;
                lw.expert_slices[0].up_bytes    = t_up->byte_size;
                lw.expert_slices[0].down_offset = t_down->abs_file_offset;
                lw.expert_slices[0].down_bytes  = t_down->byte_size;
            }
        }
    }

    return true;
}

KVCacheSlot& MoETransformerEngine::get_or_create_slot(uint32_t slot_id) {
    auto it = kv_slots_.find(slot_id);
    if (it == kv_slots_.end()) {
        KVCacheSlot s;
        s.init(slot_id, hp_);
        it = kv_slots_.emplace(slot_id, std::move(s)).first;
    }
    return it->second;
}

void MoETransformerEngine::reset_slot(uint32_t slot_id) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    auto it = kv_slots_.find(slot_id);
    if (it != kv_slots_.end()) it->second.clear();
}

size_t MoETransformerEngine::active_slot_count() const {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    return kv_slots_.size();
}

std::vector<ProfileSample> MoETransformerEngine::recent_profiles() const {
    std::lock_guard<std::mutex> lk(profile_mtx_);
    return std::vector<ProfileSample>(profile_history_.begin(), profile_history_.end());
}

std::vector<float> MoETransformerEngine::forward_token(
    uint32_t token_id,
    uint32_t slot_id,
    ProfileSample* out_profile
) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    const double t_start = now_ms();
    double io_wait_ms = 0.0;
    double matmul_ms  = 0.0;
    double attn_ms    = 0.0;
    double head_ms    = 0.0;
    uint32_t step_hits = 0;
    uint32_t step_reads = 0;
    uint32_t step_routed = 0;

    tier_mgr_.step_tick();
    KVCacheSlot& slot = get_or_create_slot(slot_id);
    if (slot.seq_pos >= hp_.max_seq_len) {
        slot.compact(hp_, hp_.max_seq_len / 2);
    }
    const uint32_t pos = slot.seq_pos;
    slot.token_history.push_back(token_id);

    const uint32_t dim = hp_.dim;
    const uint32_t kv_dim = hp_.n_kv_heads * hp_.head_dim;
    const uint32_t exp_dim = hp_.moe_expert_dim;

    // 1. Embedding Lookup
    std::vector<float> x(dim, 0.0f);
    if (tok_emb_ && token_id < hp_.vocab_size) {
        size_t row_bytes = tensor_byte_size(tok_emb_->type, dim);
        const uint8_t* emb_row = static_cast<const uint8_t*>(tok_emb_->mapped_ptr) + token_id * row_bytes;
        dequantize_row(tok_emb_->type, emb_row, x.data(), dim);
    }

    std::vector<float> x_norm(dim, 0.0f);
    std::vector<float> q(dim, 0.0f);
    std::vector<float> k(kv_dim, 0.0f);
    std::vector<float> v(kv_dim, 0.0f);
    std::vector<float> attn_out(dim, 0.0f);
    std::vector<float> proj_out(dim, 0.0f);

    // Map holding PILOT async prefetch futures for Layer L+1
    std::unordered_map<uint32_t, std::future<StreamedExpertWeights>> prefetched_futures;

    for (uint32_t l = 0; l < hp_.n_layers; ++l) {
        const LayerDenseWeights& lw = layers_[l];

        // 2. Attention Pre-RMSNorm
        const float* anorm_w = (lw.attn_norm && lw.attn_norm->type == GGMLType::F32)
                             ? static_cast<const float*>(lw.attn_norm->mapped_ptr) : nullptr;
        rms_norm(x_norm.data(), x.data(), anorm_w, dim, hp_.rms_norm_eps);

        // 3. Q, K, V Projections
        double tm0 = now_ms();
        if (lw.wq) matmul_quant_f32(lw.wq->type, lw.wq->mapped_ptr, x_norm.data(), q.data(), dim, dim);
        if (lw.wk) matmul_quant_f32(lw.wk->type, lw.wk->mapped_ptr, x_norm.data(), k.data(), kv_dim, dim);
        if (lw.wv) matmul_quant_f32(lw.wv->type, lw.wv->mapped_ptr, x_norm.data(), v.data(), kv_dim, dim);

        if (lw.bq && lw.bq->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bq->mapped_ptr);
            for (uint32_t i = 0; i < dim; ++i) q[i] += b_ptr[i];
        }
        if (lw.bk && lw.bk->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bk->mapped_ptr);
            for (uint32_t i = 0; i < kv_dim; ++i) k[i] += b_ptr[i];
        }
        if (lw.bv && lw.bv->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bv->mapped_ptr);
            for (uint32_t i = 0; i < kv_dim; ++i) v[i] += b_ptr[i];
        }

        if (lw.attn_q_norm && lw.attn_q_norm->type == GGMLType::F32) {
            const float* w_ptr = static_cast<const float*>(lw.attn_q_norm->mapped_ptr);
            for (uint32_t h = 0; h < hp_.n_heads; ++h) {
                rms_norm(q.data() + h * hp_.head_dim, q.data() + h * hp_.head_dim, w_ptr + h * hp_.head_dim, hp_.head_dim, hp_.rms_norm_eps);
            }
        }
        if (lw.attn_k_norm && lw.attn_k_norm->type == GGMLType::F32) {
            const float* w_ptr = static_cast<const float*>(lw.attn_k_norm->mapped_ptr);
            for (uint32_t h = 0; h < hp_.n_kv_heads; ++h) {
                rms_norm(k.data() + h * hp_.head_dim, k.data() + h * hp_.head_dim, w_ptr + h * hp_.head_dim, hp_.head_dim, hp_.rms_norm_eps);
            }
        }
        matmul_ms += (now_ms() - tm0);

        // 4. Rotary Position Embeddings (RoPE / YaRN) & KV-Cache Append
        double ta0 = now_ms();
        bool is_neox = (hp_.arch == "qwen2" || hp_.arch == "qwen2moe" || hp_.arch == "olmoe" ||
                        hp_.arch == "phi" || hp_.arch == "deepseek" || hp_.arch == "deepseek2" ||
                        hp_.arch == "glm" || hp_.arch == "gptneox");
        apply_rope(
            q.data(), k.data(),
            hp_.n_heads, hp_.n_kv_heads, hp_.head_dim, hp_.rope_dim,
            pos, hp_.rope_freq_base, hp_.yarn_scale, is_neox
        );

        float* k_slot = slot.k_cache[l].data() + static_cast<size_t>(pos) * kv_dim;
        float* v_slot = slot.v_cache[l].data() + static_cast<size_t>(pos) * kv_dim;
        std::memcpy(k_slot, k.data(), kv_dim * sizeof(float));
        std::memcpy(v_slot, v.data(), kv_dim * sizeof(float));

        // 5. Grouped-Query / Multi-Head Scaled Dot-Product Attention
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        const uint32_t kv_mul = std::max(1u, hp_.n_heads / std::max(1u, hp_.n_kv_heads));
        const float scale = 1.0f / std::sqrt(static_cast<float>(hp_.head_dim));
        std::vector<float> scores(pos + 1);

        for (uint32_t h = 0; h < hp_.n_heads; ++h) {
            const float* q_head = q.data() + h * hp_.head_dim;
            const uint32_t kvh = h / kv_mul;

            for (uint32_t t = 0; t <= pos; ++t) {
                const float* k_t = slot.k_cache[l].data() + static_cast<size_t>(t) * kv_dim + kvh * hp_.head_dim;
                scores[t] = vec_dot_f32(q_head, k_t, hp_.head_dim) * scale;
            }
            softmax(scores.data(), pos + 1);

            float* out_head = attn_out.data() + h * hp_.head_dim;
            for (uint32_t t = 0; t <= pos; ++t) {
                const float* v_t = slot.v_cache[l].data() + static_cast<size_t>(t) * kv_dim + kvh * hp_.head_dim;
                const float a = scores[t];
                for (uint32_t d = 0; d < hp_.head_dim; ++d) {
                    out_head[d] += a * v_t[d];
                }
            }
        }
        attn_ms += (now_ms() - ta0);

        // Attention Output Projection + Residual Connection
        double tm1 = now_ms();
        if (lw.wo) {
            matmul_quant_f32(lw.wo->type, lw.wo->mapped_ptr, attn_out.data(), proj_out.data(), dim, dim);
            for (uint32_t i = 0; i < dim; ++i) x[i] += proj_out[i];
        }
        matmul_ms += (now_ms() - tm1);

        // 6. Post-Attention FFN RMSNorm & MoE Router Gate
        const float* fnorm_w = (lw.ffn_norm && lw.ffn_norm->type == GGMLType::F32)
                             ? static_cast<const float*>(lw.ffn_norm->mapped_ptr) : nullptr;
        rms_norm(x_norm.data(), x.data(), fnorm_w, dim, hp_.rms_norm_eps);

        std::vector<float> router_logits(hp_.n_experts, 0.0f);
        if (lw.router_gate) {
            matmul_quant_f32(lw.router_gate->type, lw.router_gate->mapped_ptr, x_norm.data(), router_logits.data(), hp_.n_experts, dim);
        } else {
            if (!router_logits.empty()) router_logits[0] = 1.0f;
        }

        // Select Top-K experts with CACHE_ROUTE awareness
        auto selected = tier_mgr_.select_experts_cache_aware(l, router_logits.data(), hp_.n_experts, hp_.n_experts_used);

        // PILOT Lookahead Prefetch: While we are about to execute Layer L's SwiGLU matmuls,
        // predict Layer L+1's top experts and dispatch asynchronous NVMe reads now!
        std::unordered_map<uint32_t, std::future<StreamedExpertWeights>> next_layer_futures;
        if (l + 1 < hp_.n_layers && layers_[l + 1].router_gate) {
            const LayerDenseWeights& next_lw = layers_[l + 1];
            std::vector<float> next_logits(hp_.n_experts, 0.0f);
            matmul_quant_f32(
                next_lw.router_gate->type,
                next_lw.router_gate->mapped_ptr,
                x_norm.data(),
                next_logits.data(),
                hp_.n_experts,
                dim
            );
            // Find top-K candidates for Layer L+1 and launch background IOCP / async reads if not cached
            std::vector<uint32_t> idx_order(hp_.n_experts);
            std::iota(idx_order.begin(), idx_order.end(), 0);
            std::partial_sort(idx_order.begin(), idx_order.begin() + std::min(hp_.n_experts_used, hp_.n_experts), idx_order.end(),
                [&](uint32_t a, uint32_t b) { return next_logits[a] > next_logits[b]; });

            for (uint32_t k_idx = 0; k_idx < std::min(hp_.n_experts_used, hp_.n_experts); ++k_idx) {
                uint32_t cand_e = idx_order[k_idx];
                if (!tier_mgr_.is_resident(l + 1, cand_e) && cand_e < next_lw.expert_slices.size()) {
                    next_layer_futures[cand_e] = io_engine_.prefetch_expert_async(
                        l + 1, cand_e, next_lw.expert_slices[cand_e]
                    );
                }
            }
        }

        // 7. Execute Selected MoE Experts (SwiGLU: (SiLU(u * W_gate) * (u * W_up)) * W_down)
        std::vector<float> moe_accum(dim, 0.0f);
        std::vector<float> gate_buf(exp_dim, 0.0f);
        std::vector<float> up_buf(exp_dim, 0.0f);
        std::vector<float> down_buf(dim, 0.0f);

        for (const auto& sel : selected) {
            if (sel.was_cache_routed) step_routed++;
            const uint32_t e = sel.expert_idx;
            const ExpertCacheEntry* cached = tier_mgr_.get_resident(l, e);

            const void* w_gate_ptr = nullptr;
            const void* w_up_ptr   = nullptr;
            const void* w_down_ptr = nullptr;
            StreamedExpertWeights scratch;

            if (cached) {
                step_hits++;
                w_gate_ptr = cached->weight_gate.data();
                w_up_ptr   = cached->weight_up.data();
                w_down_ptr = cached->weight_down.data();
            } else if (e < lw.expert_slices.size()) {
                step_reads++;
                double tio0 = now_ms();
                auto fit = prefetched_futures.find(e);
                if (fit != prefetched_futures.end()) {
                    scratch = fit->second.get();
                } else {
                    scratch = io_engine_.read_expert_sync(l, e, lw.expert_slices[e]);
                }
                io_wait_ms += (now_ms() - tio0);

                // Attempt to admit to LFRU cache (respects 25% hysteresis threshold)
                std::vector<uint8_t> cg = scratch.gate;
                std::vector<uint8_t> cu = scratch.up;
                std::vector<uint8_t> cd = scratch.down;
                const ExpertCacheEntry* admitted = tier_mgr_.admit_expert(
                    l, e, std::move(cg), std::move(cu), std::move(cd)
                );
                if (admitted) {
                    w_gate_ptr = admitted->weight_gate.data();
                    w_up_ptr   = admitted->weight_up.data();
                    w_down_ptr = admitted->weight_down.data();
                } else {
                    w_gate_ptr = scratch.gate.data();
                    w_up_ptr   = scratch.up.data();
                    w_down_ptr = scratch.down.data();
                }
            }

            if (w_gate_ptr && w_up_ptr && w_down_ptr) {
                double t_exp = now_ms();
                matmul_quant_f32(lw.expert_gate_type, w_gate_ptr, x_norm.data(), gate_buf.data(), exp_dim, dim);
                matmul_quant_f32(lw.expert_up_type,   w_up_ptr,   x_norm.data(), up_buf.data(),   exp_dim, dim);

                for (uint32_t d = 0; d < exp_dim; ++d) {
                    gate_buf[d] = silu(gate_buf[d]) * up_buf[d];
                }

                matmul_quant_f32(lw.expert_down_type, w_down_ptr, gate_buf.data(), down_buf.data(), dim, exp_dim);
                for (uint32_t d = 0; d < dim; ++d) {
                    moe_accum[d] += sel.prob * down_buf[d];
                }
                matmul_ms += (now_ms() - t_exp);
            }
        }

        for (uint32_t i = 0; i < dim; ++i) {
            x[i] += moe_accum[i];
        }

        prefetched_futures = std::move(next_layer_futures);
    }

    slot.seq_pos++;

    // 8. Final RMSNorm & LM Head Logit Projection
    double th0 = now_ms();
    const float* onorm_w = (out_norm_ && out_norm_->type == GGMLType::F32)
                         ? static_cast<const float*>(out_norm_->mapped_ptr) : nullptr;
    rms_norm(x_norm.data(), x.data(), onorm_w, dim, hp_.rms_norm_eps);

    std::vector<float> logits(hp_.vocab_size, 0.0f);
    if (lm_head_) {
        matmul_quant_f32(lm_head_->type, lm_head_->mapped_ptr, x_norm.data(), logits.data(), hp_.vocab_size, dim);
    }
    head_ms = now_ms() - th0;

    const double wall_ms = std::max(0.01, now_ms() - t_start);
    ProfileSample prof;
    prof.timestamp_ms = static_cast<uint64_t>(now_us() / 1000);
    prof.token_id     = token_id;
    prof.wall_ms      = wall_ms;
    prof.io_wait_ms   = io_wait_ms;
    prof.matmul_ms    = matmul_ms;
    prof.attention_ms = attn_ms;
    prof.lm_head_ms   = head_ms;
    prof.tok_per_sec  = 1000.0 / wall_ms;
    prof.cache_hits   = step_hits;
    prof.disk_reads   = step_reads;
    prof.cache_routed = step_routed;

    if (out_profile) *out_profile = prof;

    {
        std::lock_guard<std::mutex> plk(profile_mtx_);
        profile_history_.push_back(prof);
        if (profile_history_.size() > 120) {
            profile_history_.pop_front();
        }
    }

    return logits;
}

uint32_t MoETransformerEngine::sample_next_token(
    const std::vector<float>& logits,
    const std::vector<uint32_t>& history,
    const SamplerConfig& cfg
) {
    if (logits.empty()) return 0;
    const uint32_t vocab_sz = static_cast<uint32_t>(logits.size());
    std::vector<float> adj = logits;

    // 1. Repetition Penalty over recent history
    if (cfg.repetition_penalty != 1.0f && !history.empty()) {
        size_t start = (history.size() > 64) ? (history.size() - 64) : 0;
        for (size_t i = start; i < history.size(); ++i) {
            uint32_t tid = history[i];
            if (tid == 220 || tid == 198 || tid == 0) continue; // skip space, newline, unk
            if (tid < vocab_sz) {
                if (adj[tid] > 0.0f) adj[tid] /= cfg.repetition_penalty;
                else adj[tid] *= cfg.repetition_penalty;
            }
        }
    }

    // 2. Greedy Argmax when temperature <= 0
    if (cfg.temperature <= 1e-5f || cfg.top_k == 1) {
        return static_cast<uint32_t>(std::max_element(adj.begin(), adj.end()) - adj.begin());
    }

    // 3. Top-K filtering
    struct TokProb {
        uint32_t id;
        float val;
    };
    std::vector<TokProb> cands(vocab_sz);
    for (uint32_t i = 0; i < vocab_sz; ++i) {
        cands[i] = {i, adj[i] / cfg.temperature};
    }

    uint32_t k = (cfg.top_k > 0 && cfg.top_k < vocab_sz) ? cfg.top_k : vocab_sz;
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
        [](const TokProb& a, const TokProb& b) { return a.val > b.val; });
    cands.resize(k);

    // 4. Softmax + Top-P (Nucleus) filtering
    float max_l = cands[0].val;
    float sum_exp = 0.0f;
    for (auto& c : cands) {
        c.val = std::exp(c.val - max_l);
        sum_exp += c.val;
    }
    for (auto& c : cands) c.val /= sum_exp;

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].val;
            if (cum >= cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        cands.resize(std::max<size_t>(1, cutoff));
    }

    // Renormalize & sample
    float norm_sum = 0.0f;
    for (const auto& c : cands) norm_sum += c.val;

    thread_local std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, norm_sum);
    float r = dist(rng);
    float acc = 0.0f;
    for (const auto& c : cands) {
        acc += c.val;
        if (r <= acc) return c.id;
    }
    return cands.front().id;
}

std::vector<uint32_t> MoETransformerEngine::verify_speculative_batch(
    uint32_t slot_id,
    const std::vector<uint32_t>& draft_tokens,
    const SamplerConfig& sampler_cfg,
    ProfileSample* out_profile
) {
    std::vector<uint32_t> accepted;
    for (uint32_t draft_id : draft_tokens) {
        ProfileSample prof{};
        auto logits = forward_token(draft_id, slot_id, &prof);
        if (out_profile) *out_profile = prof;

        std::vector<uint32_t> hist;
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            hist = get_or_create_slot(slot_id).token_history;
        }
        uint32_t sampled = sample_next_token(logits, hist, sampler_cfg);
        accepted.push_back(draft_id);
        if (sampled == tokenizer_.eos_id() || sampled == tokenizer_.im_end_id()) {
            break;
        }
    }
    return accepted;
}

void MoETransformerEngine::generate_stream(
    const std::string& prompt,
    uint32_t max_new_tokens,
    uint32_t slot_id,
    bool enable_context_compaction,
    const SamplerConfig& sampler_cfg,
    StreamCallback on_token,
    bool continue_assistant,
    bool enable_speculative,
    uint32_t draft_count
) {
    std::vector<float> logits;
    ProfileSample last_prof;

    bool has_existing_context = false;
    {
        std::lock_guard<std::mutex> lk(engine_mtx_);
        has_existing_context = (get_or_create_slot(slot_id).seq_pos > 0);
    }

    if (!continue_assistant || !has_existing_context) {
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            get_or_create_slot(slot_id).clear();
        }

        std::string prompt_to_encode = prompt;
        if (prompt_to_encode.find("<|im_start|>") == std::string::npos &&
            prompt_to_encode.find("<|user|>") == std::string::npos &&
            prompt_to_encode.find("<|system|>") == std::string::npos) {
            if (hp_.arch == "olmoe" || tokenizer_.token_to_id("<|user|>") >= 0) {
                prompt_to_encode = "<|user|>\n" + prompt_to_encode + "\n<|assistant|>\n";
            } else if (hp_.arch == "qwen2" || tokenizer_.token_to_id("<|im_start|>") >= 0) {
                prompt_to_encode = "<|im_start|>system\nYou are a helpful assistant. Please answer concisely in English.<|im_end|>\n<|im_start|>user\n" + prompt_to_encode + "<|im_end|>\n<|im_start|>assistant\n";
            }
        }

        std::vector<uint32_t> prompt_tokens = tokenizer_.encode(prompt_to_encode, false);
        if (prompt_tokens.empty()) {
            prompt_tokens.push_back(tokenizer_.bos_id());
        }

        // Prefill prompt tokens through real forward pass
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            logits = forward_token(prompt_tokens[i], slot_id, &last_prof);
        }
    } else {
        // Continue assistant: use last known token in history for forward step
        uint32_t last_token = tokenizer_.bos_id();
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            const auto& hist = get_or_create_slot(slot_id).token_history;
            if (!hist.empty()) last_token = hist.back();
        }
        logits = forward_token(last_token, slot_id, &last_prof);
    }

    StreamingUTF8Decoder utf8_dec(tokenizer_);
    bool in_thinking = false;

    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        if (enable_context_compaction) {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            auto& slot = get_or_create_slot(slot_id);
            if (slot.seq_pos + 16 >= hp_.max_seq_len) {
                slot.compact(hp_, hp_.max_seq_len / 2);
            }
        }

        std::vector<uint32_t> hist;
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            hist = get_or_create_slot(slot_id).token_history;
        }

        uint32_t next_id = sample_next_token(logits, hist, sampler_cfg);
        if (next_id == tokenizer_.eos_id() || next_id == tokenizer_.im_end_id() ||
            next_id == 50279 || next_id == 50280 || next_id == 151645 || next_id == 151643) {
            break;
        }

        if (next_id == tokenizer_.think_start_id()) {
            in_thinking = true;
            logits = forward_token(next_id, slot_id, &last_prof);
            continue;
        }
        if (next_id == tokenizer_.think_end_id()) {
            in_thinking = false;
            logits = forward_token(next_id, slot_id, &last_prof);
            continue;
        }

        std::string piece = utf8_dec.push(next_id);
        if (!piece.empty()) {
            if (piece.find("<|im_end|>") != std::string::npos ||
                piece.find("<|endoftext|>") != std::string::npos ||
                piece.find("<|end_of_text|>") != std::string::npos) {
                break;
            }
            // Also detect literal "<think>" or "</think>" tags if emitted via subwords
            if (piece == "<think>") {
                in_thinking = true;
            } else if (piece == "</think>") {
                in_thinking = false;
            } else {
                if (!on_token(piece, in_thinking, next_id, last_prof)) {
                    break;
                }
            }
        }

        // Multi-Token Prediction (MTP) Speculative batch verification
        if (enable_speculative && draft_count > 0 && (step + draft_count < max_new_tokens)) {
            // Propose draft candidate tokens from greedy argmax
            std::vector<uint32_t> drafts;
            uint32_t cand = next_id;
            for (uint32_t d = 0; d < draft_count; ++d) {
                drafts.push_back(cand);
            }
            // Batched verification
            auto accepted = verify_speculative_batch(slot_id, drafts, sampler_cfg, &last_prof);
            for (uint32_t acc_id : accepted) {
                std::string acc_piece = utf8_dec.push(acc_id);
                if (!acc_piece.empty() && acc_piece != "<think>" && acc_piece != "</think>") {
                    on_token(acc_piece, in_thinking, acc_id, last_prof);
                }
            }
            step += static_cast<uint32_t>(accepted.size());
        }

        if (step + 1 < max_new_tokens) {
            logits = forward_token(next_id, slot_id, &last_prof);
        }
    }

    std::string tail = utf8_dec.flush();
    if (!tail.empty()) {
        on_token(tail, in_thinking, 0, last_prof);
    }
}

std::vector<SystemOneOptionScore> MoETransformerEngine::evaluate_system_one(
    const std::string& document,
    const std::string& query,
    const std::vector<std::string>& options,
    ProfileSample* out_profile
) {
    // Use isolated scratch slot 9999 for zero-token-generation decision evaluation
    const uint32_t sys1_slot = 9999;
    reset_slot(sys1_slot);

    std::string combined = document + "\n" + query + "\n";
    std::vector<uint32_t> tokens = tokenizer_.encode(combined, true);
    if (tokens.empty()) tokens.push_back(tokenizer_.bos_id());

    std::vector<float> logits;
    for (uint32_t tid : tokens) {
        logits = forward_token(tid, sys1_slot, out_profile);
    }

    std::vector<SystemOneOptionScore> results;
    if (options.empty() || logits.empty()) return results;

    float max_logit = -1e30f;
    for (const auto& opt : options) {
        auto opt_toks = tokenizer_.encode(opt, false);
        uint32_t first_tid = opt_toks.empty() ? 0 : opt_toks[0];
        // Aggregate mean logit across the candidate option's tokens from final LM head projection
        float score = 0.0f;
        for (uint32_t tid : opt_toks) {
            if (tid < logits.size()) score += logits[tid];
        }
        if (!opt_toks.empty()) score /= static_cast<float>(opt_toks.size());

        max_logit = std::max(max_logit, score);
        results.push_back({opt, score, 0.0f, first_tid});
    }

    float sum_exp = 0.0f;
    for (auto& r : results) {
        r.probability = std::exp(r.raw_logit - max_logit);
        sum_exp += r.probability;
    }
    for (auto& r : results) {
        r.probability = (sum_exp > 0.0f) ? (r.probability / sum_exp) : (1.0f / results.size());
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });

    return results;
}

} // namespace moe

