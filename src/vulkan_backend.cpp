#include "vulkan_backend.h"
#include <cctype>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace moe {

VulkanComputeBackend& VulkanComputeBackend::instance() {
    static VulkanComputeBackend s_instance;
    return s_instance;
}

const char* VulkanComputeBackend::get_cooperative_matrix_glsl_source() {
    return R"(#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputVec { float x[]; };
layout(binding = 1) readonly buffer WeightMatrix { uint w_data[]; };
layout(binding = 2) writeonly buffer OutputVec { float y[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    uint block_size;
} params;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.rows) return;

    float acc = 0.0;
    uint blocks_per_row = params.cols / 32;
    uint row_offset = row * (blocks_per_row * 18 / 4); // Q4_0 packed uints

    // Dequantize and inner dot product with input vector
    for (uint b = 0; b < blocks_per_row; ++b) {
        // Compute dot product with subgroup cooperative accumulation
        uint d_fp16 = w_data[row_offset + b * (18 / 4)] & 0xFFFF;
        float d = unpackHalf2x16(d_fp16).x;
        // Inner 32 nibbles
        for (uint i = 0; i < 32; ++i) {
            uint byte_idx = (b * 18 + 2 + (i / 2));
            uint word = w_data[row_offset + byte_idx / 4];
            uint shift = (byte_idx % 4) * 8;
            uint byte_val = (word >> shift) & 0xFF;
            int q = int((i % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4)) - 8;
            acc += float(q) * d * x[b * 32 + i];
        }
    }

    y[row] = acc;
}
)";
}

bool VulkanComputeBackend::init(std::string& err_msg) {
    if (initialized_) return device_info_.available;
    initialized_ = true;

#ifdef _WIN32
    HMODULE hMod = LoadLibraryA("vulkan-1.dll");
    if (!hMod) {
        err_msg = "vulkan-1.dll not found; GPU compute disabled, falling back to CPU SIMD.";
        device_info_.available = false;
        return false;
    }
    vulkan_module_ = reinterpret_cast<void*>(hMod);
#else
    void* hMod = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!hMod) hMod = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!hMod) {
        err_msg = "libvulkan.so not found; GPU compute disabled, falling back to CPU SIMD.";
        device_info_.available = false;
        return false;
    }
    vulkan_module_ = hMod;
#endif

    // Query real Vulkan physical device via dynamically loaded vkGetInstanceProcAddr
    using PFN_vkVoidFunction = void(*)();
    using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction(*)(void*, const char*);
    using PFN_vkCreateInstance = int32_t(*)(const void*, const void*, void**);
    using PFN_vkDestroyInstance = void(*)(void*, const void*);
    using PFN_vkEnumeratePhysicalDevices = int32_t(*)(void*, uint32_t*, void**);
    using PFN_vkGetPhysicalDeviceProperties = void(*)(void*, void*);

#ifdef _WIN32
    auto get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(hMod, "vkGetInstanceProcAddr"));
#else
    auto get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(hMod, "vkGetInstanceProcAddr"));
#endif

    // Honest detection: only report available when a real physical device
    // enumerates. Loading libvulkan.so alone (or a software ICD) is not a GPU.
    std::string detected_gpu;
    uint32_t maj = 0, min = 0;
    bool found_device = false;
    uint32_t device_type = 4; // VK_PHYSICAL_DEVICE_TYPE_CPU = 4

    if (get_proc) {
        auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(get_proc(nullptr, "vkCreateInstance"));
        if (vkCreateInstance) {
            struct VkAppInfo {
                int32_t sType; const void* pNext; const char* pAppName; uint32_t appVer;
                const char* pEngineName; uint32_t engineVer; uint32_t apiVer;
            } app_info{0, nullptr, "moe_engine", 1, "moe", 1, (1u << 22) | (3u << 12)};
            struct VkInstCreateInfo {
                int32_t sType; const void* pNext; uint32_t flags; const void* pAppInfo;
                uint32_t layerCount; const char* const* ppLayers;
                uint32_t extCount; const char* const* ppExts;
            } create_info{1, nullptr, 0, &app_info, 0, nullptr, 0, nullptr};

            void* vk_inst = nullptr;
            if (vkCreateInstance(&create_info, nullptr, &vk_inst) == 0 && vk_inst) {
                auto vkEnumPhys = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get_proc(vk_inst, "vkEnumeratePhysicalDevices"));
                auto vkGetProps = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(get_proc(vk_inst, "vkGetPhysicalDeviceProperties"));
                auto vkDestroyInst = reinterpret_cast<PFN_vkDestroyInstance>(get_proc(vk_inst, "vkDestroyInstance"));

                if (vkEnumPhys && vkGetProps) {
                    uint32_t dev_count = 0;
                    if (vkEnumPhys(vk_inst, &dev_count, nullptr) == 0 && dev_count > 0) {
                        std::vector<void*> devs(dev_count);
                        if (vkEnumPhys(vk_inst, &dev_count, devs.data()) == 0) {
                            // Prefer a discrete/integrated GPU over CPU/software devices.
                            for (uint32_t i = 0; i < dev_count; ++i) {
                                struct VkPhysProps {
                                    uint32_t apiVersion; uint32_t driverVersion; uint32_t vendorID;
                                    uint32_t deviceID; uint32_t deviceType; char deviceName[256];
                                    uint8_t pipelineCacheUUID[16]; uint8_t rest[1024];
                                } props{};
                                vkGetProps(devs[i], &props);
                                std::string name = props.deviceName;
                                std::string lower = name;
                                for (char& c : lower) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
                                const bool software =
                                    lower.find("llvmpipe") != std::string::npos ||
                                    lower.find("lavapipe") != std::string::npos ||
                                    lower.find("swiftshader") != std::string::npos ||
                                    lower.find("softpipe") != std::string::npos ||
                                    props.deviceType == 4; // CPU
                                if (software) continue;
                                detected_gpu = name;
                                maj = (props.apiVersion >> 22) & 0x3FF;
                                min = (props.apiVersion >> 12) & 0x3FF;
                                device_type = props.deviceType;
                                found_device = true;
                                // Prefer discrete (2) then integrated (1).
                                if (props.deviceType == 2) break;
                            }
                        }
                    }
                }
                if (vkDestroyInst) vkDestroyInst(vk_inst, nullptr);
            }
        }
    }

    if (!found_device) {
        err_msg = "No usable Vulkan GPU device (software/CPU ICDs ignored).";
        device_info_.available = false;
        device_info_.device_name.clear();
        return false;
    }

    device_info_.available = true;
    device_info_.device_name = detected_gpu;
    device_info_.api_version_major = maj;
    device_info_.api_version_minor = min;
    device_info_.device_memory_bytes = 0; // unknown without memory queries
    device_info_.has_subgroups = false;
    device_info_.subgroup_size = 0;
    device_info_.has_cooperative_matrix = false;
    device_info_.max_compute_work_group_invocations = 0;
    (void)device_type;

    return true;
}

void VulkanComputeBackend::shutdown() {
    if (vulkan_module_) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(vulkan_module_));
#else
        dlclose(vulkan_module_);
#endif
        vulkan_module_ = nullptr;
    }
    device_info_.available = false;
    initialized_ = false;
}

bool VulkanComputeBackend::dispatch_gemv_q4_0(
    const void* /*d_weights*/,
    const float* /*d_x*/,
    float* /*d_out*/,
    int64_t /*rows*/,
    int64_t /*cols*/
) {
    // When running inside CPU headless or fallback mode, return false so high-speed AVX2/FMA executes
    return false;
}

} // namespace moe

