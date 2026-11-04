#pragma once

#include "common.h"
#include <string>
#include <vector>

namespace moe {

struct VulkanDeviceInfo {
    bool available = false;
    std::string device_name;
    uint32_t api_version_major = 0;
    uint32_t api_version_minor = 0;
    uint64_t device_memory_bytes = 0;
    bool has_subgroups = false;
    uint32_t subgroup_size = 0;
    bool has_cooperative_matrix = false;
    uint32_t max_compute_work_group_invocations = 0;
};

class VulkanComputeBackend {
public:
    static VulkanComputeBackend& instance();

    bool init(std::string& err_msg);
    void shutdown();

    bool is_available() const { return device_info_.available; }
    const VulkanDeviceInfo& device_info() const { return device_info_; }

    // Execute GEMV on GPU if available, returns true if handled, false to fallback to CPU SIMD
    bool dispatch_gemv_q4_0(
        const void* d_weights,
        const float* d_x,
        float* d_out,
        int64_t rows,
        int64_t cols
    );

    // Cross-vendor GLSL compute shader source for cooperative matrix / GEMV
    static const char* get_cooperative_matrix_glsl_source();

private:
    VulkanComputeBackend() = default;
    ~VulkanComputeBackend() { shutdown(); }

    VulkanDeviceInfo device_info_;
    void* vulkan_module_ = nullptr;
    bool initialized_ = false;
};

} // namespace moe

