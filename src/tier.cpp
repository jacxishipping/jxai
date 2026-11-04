#include "tier.h"
#include <algorithm>
#include <cmath>

namespace moe {

void LFRUTierManager::init(
    uint32_t n_layers,
    uint32_t n_experts,
    size_t max_ram_slots,
    size_t max_vram_slots,
    float cache_route_eps
) {
    std::lock_guard<std::mutex> lk(mtx_);
    n_layers_ = n_layers;
    n_experts_ = n_experts;
    max_ram_slots_ = max_ram_slots;
    max_vram_slots_ = max_vram_slots;
    cache_route_eps_ = cache_route_eps;
    current_step_ = 1;
    cache_.clear();

    size_t total = static_cast<size_t>(n_layers) * static_cast<size_t>(n_experts);
    activation_counts_.assign(total, 0);
    last_step_.assign(total, 0);
    last_prob_.assign(total, 0.0f);
    total_hits_ = 0;
    total_misses_ = 0;
    total_cache_routed_ = 0;
}

uint32_t LFRUTierManager::compute_lfru_score(uint32_t frequency, uint32_t recency_val) {
    return (frequency * 256u) + std::min<uint32_t>(255u, recency_val);
}

bool LFRUTierManager::is_resident(uint32_t layer, uint32_t expert) const {
    std::lock_guard<std::mutex> lk(mtx_);
    return cache_.find(make_key(layer, expert)) != cache_.end();
}

const ExpertCacheEntry* LFRUTierManager::get_resident(uint32_t layer, uint32_t expert) {
    std::lock_guard<std::mutex> lk(mtx_);
    uint64_t k = make_key(layer, expert);
    auto it = cache_.find(k);
    if (it == cache_.end()) {
        total_misses_++;
        return nullptr;
    }
    total_hits_++;
    it->second.frequency++;
    it->second.last_access_step = current_step_;
    if (it->second.frequency >= 12) {
        it->second.pinned = true;
        it->second.tier = MemoryTier::VRAM;
    }
    return &it->second;
}

const ExpertCacheEntry* LFRUTierManager::admit_expert(
    uint32_t layer,
    uint32_t expert,
    std::vector<uint8_t>&& gate,
    std::vector<uint8_t>&& up,
    std::vector<uint8_t>&& down
) {
    std::lock_guard<std::mutex> lk(mtx_);
    uint64_t k = make_key(layer, expert);
    size_t flat_idx = static_cast<size_t>(layer) * n_experts_ + expert;
    uint32_t freq = (flat_idx < activation_counts_.size()) ? activation_counts_[flat_idx] : 1;

    auto existing = cache_.find(k);
    if (existing != cache_.end()) {
        existing->second.frequency = std::max(existing->second.frequency + 1, freq);
        existing->second.last_access_step = current_step_;
        return &existing->second;
    }

    const size_t total_capacity = max_ram_slots_ + max_vram_slots_;
    const uint32_t new_score = compute_lfru_score(freq, 255u);

    if (cache_.size() >= total_capacity && !cache_.empty()) {
        // Find lowest-scoring unpinned victim
        uint64_t victim_key = 0;
        uint32_t min_score = 0xFFFFFFFFu;
        bool found_victim = false;

        for (const auto& kv : cache_) {
            if (kv.second.pinned) continue;
            uint32_t age = (current_step_ >= kv.second.last_access_step)
                         ? (current_step_ - kv.second.last_access_step) : 0;
            uint32_t recency = (age >= 255u) ? 0u : (255u - age);
            uint32_t sc = compute_lfru_score(kv.second.frequency, recency);
            if (sc < min_score) {
                min_score = sc;
                victim_key = kv.first;
                found_victim = true;
            }
        }

        // 25% hysteresis margin to prevent cache thrashing
        if (found_victim && static_cast<double>(new_score) >= static_cast<double>(min_score) * 1.25) {
            cache_.erase(victim_key);
        } else if (cache_.size() >= total_capacity) {
            // Do not evict if new candidate fails hysteresis check; caller uses scratch buffer
            return nullptr;
        }
    }

    ExpertCacheEntry entry;
    entry.layer_idx = layer;
    entry.expert_idx = expert;
    entry.frequency = std::max(1u, freq);
    entry.last_access_step = current_step_;
    entry.pinned = (entry.frequency >= 12);
    entry.tier = (cache_.size() < max_vram_slots_) ? MemoryTier::VRAM : MemoryTier::RAM;
    entry.weight_gate = std::move(gate);
    entry.weight_up   = std::move(up);
    entry.weight_down = std::move(down);

    auto [it, _] = cache_.emplace(k, std::move(entry));
    return &it->second;
}

std::vector<ExpertSelection> LFRUTierManager::select_experts_cache_aware(
    uint32_t layer,
    const float* router_logits,
    uint32_t n_experts,
    uint32_t top_k
) {
    std::lock_guard<std::mutex> lk(mtx_);
    top_k = std::min(top_k, n_experts);

    struct Cand {
        uint32_t idx;
        float logit;
        bool resident;
    };
    std::vector<Cand> cands(n_experts);
    for (uint32_t e = 0; e < n_experts; ++e) {
        bool res = (cache_.find(make_key(layer, e)) != cache_.end());
        cands[e] = {e, router_logits[e], res};
    }

    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.logit > b.logit;
    });

    std::vector<bool> chosen(n_experts, false);
    std::vector<ExpertSelection> selected(top_k);

    for (uint32_t k = 0; k < top_k; ++k) {
        Cand pick = cands[k];
        bool routed = false;

        // CACHE_ROUTE optimization: if top candidate is not resident in RAM/VRAM,
        // check if an unselected resident expert has gate logit within cache_route_eps_
        if (!pick.resident && cache_route_eps_ > 0.0f) {
            for (uint32_t alt = top_k; alt < n_experts; ++alt) {
                if (cands[alt].resident && !chosen[cands[alt].idx]) {
                    if ((pick.logit - cands[alt].logit) <= cache_route_eps_) {
                        pick = cands[alt];
                        routed = true;
                        total_cache_routed_++;
                        break;
                    }
                }
            }
        }

        chosen[pick.idx] = true;
        selected[k].expert_idx = pick.idx;
        selected[k].raw_logit = pick.logit;
        selected[k].was_cache_hit = pick.resident;
        selected[k].was_cache_routed = routed;
    }

    // Softmax normalization over selected top-K logits
    float max_l = selected[0].raw_logit;
    for (uint32_t k = 1; k < top_k; ++k) max_l = std::max(max_l, selected[k].raw_logit);

    float sum_exp = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k) {
        selected[k].prob = std::exp(selected[k].raw_logit - max_l);
        sum_exp += selected[k].prob;
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : (1.0f / top_k);
    for (uint32_t k = 0; k < top_k; ++k) {
        selected[k].prob *= inv_sum;
        size_t flat = static_cast<size_t>(layer) * n_experts_ + selected[k].expert_idx;
        if (flat < activation_counts_.size()) {
            activation_counts_[flat]++;
            last_step_[flat] = current_step_;
            last_prob_[flat] = selected[k].prob;
        }
    }

    return selected;
}

size_t LFRUTierManager::vram_resident_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t c = 0;
    for (const auto& kv : cache_) {
        if (kv.second.tier == MemoryTier::VRAM) c++;
    }
    return c;
}

size_t LFRUTierManager::ram_resident_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t c = 0;
    for (const auto& kv : cache_) {
        if (kv.second.tier == MemoryTier::RAM) c++;
    }
    return c;
}

size_t LFRUTierManager::disk_only_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t total = static_cast<size_t>(n_layers_) * static_cast<size_t>(n_experts_);
    return (total > cache_.size()) ? (total - cache_.size()) : 0;
}

uint64_t LFRUTierManager::resident_bytes() const {
    std::lock_guard<std::mutex> lk(mtx_);
    uint64_t b = 0;
    for (const auto& kv : cache_) {
        b += kv.second.weight_gate.size() + kv.second.weight_up.size() + kv.second.weight_down.size();
    }
    return b;
}

std::vector<LFRUTierManager::ExpertTelemetry> LFRUTierManager::snapshot_telemetry() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<ExpertTelemetry> out;
    out.reserve(static_cast<size_t>(n_layers_) * static_cast<size_t>(n_experts_));

    for (uint32_t l = 0; l < n_layers_; ++l) {
        for (uint32_t e = 0; e < n_experts_; ++e) {
            size_t flat = static_cast<size_t>(l) * n_experts_ + e;
            uint32_t act = (flat < activation_counts_.size()) ? activation_counts_[flat] : 0;
            uint32_t lstep = (flat < last_step_.size()) ? last_step_[flat] : 0;
            float lprob = (flat < last_prob_.size()) ? last_prob_[flat] : 0.0f;

            uint32_t age = (current_step_ >= lstep && lstep > 0) ? (current_step_ - lstep) : 255u;
            uint32_t recency = (age >= 255u) ? 0u : (255u - age);
            uint32_t score = compute_lfru_score(act, recency);

            MemoryTier tier = MemoryTier::DISK;
            bool pinned = false;
            auto it = cache_.find(make_key(l, e));
            if (it != cache_.end()) {
                tier = it->second.tier;
                pinned = it->second.pinned;
                uint32_t cage = (current_step_ >= it->second.last_access_step)
                              ? (current_step_ - it->second.last_access_step) : 0;
                uint32_t crec = (cage >= 255u) ? 0u : (255u - cage);
                score = compute_lfru_score(it->second.frequency, crec);
            }

            out.push_back({l, e, act, score, tier, pinned, lprob});
        }
    }
    return out;
}

} // namespace moe

