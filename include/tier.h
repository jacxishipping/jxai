#pragma once

#include "common.h"
#include <vector>
#include <unordered_map>
#include <mutex>

namespace moe {

enum class MemoryTier : uint8_t {
    DISK = 0,
    RAM  = 1,
    VRAM = 2,
};

struct ExpertCacheEntry {
    uint32_t layer_idx = 0;
    uint32_t expert_idx = 0;
    uint32_t frequency = 0;
    uint32_t last_access_step = 0;
    bool     pinned = false;
    MemoryTier tier = MemoryTier::DISK;
    std::vector<uint8_t> weight_gate;
    std::vector<uint8_t> weight_up;
    std::vector<uint8_t> weight_down;
};

struct ExpertSelection {
    uint32_t expert_idx = 0;
    float    prob = 0.0f;
    float    raw_logit = 0.0f;
    bool     was_cache_hit = false;
    bool     was_cache_routed = false;
};

class LFRUTierManager {
public:
    LFRUTierManager() = default;

    void init(
        uint32_t n_layers,
        uint32_t n_experts,
        size_t max_ram_slots,
        size_t max_vram_slots,
        float cache_route_eps = 0.08f
    );

    // LFRU Score formula: Score = (Frequency * 256) + min(255, Recency)
    // where Recency = max(0, 255 - (current_step - last_access_step))
    static uint32_t compute_lfru_score(uint32_t frequency, uint32_t recency_age);

    // Check if (layer, expert) is currently resident in RAM or VRAM cache
    bool is_resident(uint32_t layer, uint32_t expert) const;

    // Retrieve cached expert entry (updates frequency & recency)
    const ExpertCacheEntry* get_resident(uint32_t layer, uint32_t expert);

    // Insert or promote streamed expert weights into LFRU cache with 25% hysteresis margin
    const ExpertCacheEntry* admit_expert(
        uint32_t layer,
        uint32_t expert,
        std::vector<uint8_t>&& gate,
        std::vector<uint8_t>&& up,
        std::vector<uint8_t>&& down
    );

    // CACHE_ROUTE: Select top-K experts from router logits, substituting a resident expert
    // when an uncached candidate and resident alternative have logits within epsilon
    std::vector<ExpertSelection> select_experts_cache_aware(
        uint32_t layer,
        const float* router_logits,
        uint32_t n_experts,
        uint32_t top_k
    );

    void step_tick() { current_step_++; }

    // Telemetry getters for /health and /experts
    uint32_t n_layers() const { return n_layers_; }
    uint32_t n_experts() const { return n_experts_; }
    size_t vram_resident_count() const;
    size_t ram_resident_count() const;
    size_t disk_only_count() const;
    uint64_t resident_bytes() const;

    uint64_t total_hits() const { return total_hits_.load(); }
    uint64_t total_misses() const { return total_misses_.load(); }
    uint64_t total_cache_routed() const { return total_cache_routed_.load(); }

    // Snapshot of activation counts & residency for /experts
    struct ExpertTelemetry {
        uint32_t layer;
        uint32_t expert;
        uint32_t activations;
        uint32_t lfru_score;
        MemoryTier tier;
        bool pinned;
        float last_prob;
    };
    std::vector<ExpertTelemetry> snapshot_telemetry() const;

private:
    uint64_t make_key(uint32_t l, uint32_t e) const {
        return (static_cast<uint64_t>(l) << 32) | static_cast<uint64_t>(e);
    }

    mutable std::mutex mtx_;
    uint32_t n_layers_ = 0;
    uint32_t n_experts_ = 0;
    size_t max_ram_slots_ = 32;
    size_t max_vram_slots_ = 16;
    float cache_route_eps_ = 0.08f;
    uint32_t current_step_ = 1;

    std::unordered_map<uint64_t, ExpertCacheEntry> cache_;
    std::vector<uint32_t> activation_counts_;
    std::vector<uint32_t> last_step_;
    std::vector<float>    last_prob_;

    std::atomic<uint64_t> total_hits_{0};
    std::atomic<uint64_t> total_misses_{0};
    std::atomic<uint64_t> total_cache_routed_{0};
};

} // namespace moe

