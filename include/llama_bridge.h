#pragma once

#include <cstdint>
#include <string>

namespace moe {

// Hybrid GPU backend via an external llama.cpp `llama-server` process.
// PCs without a usable GPU (or without a llama-server binary) keep using
// the native moe_engine CPU path — that is intentional.

enum class InferenceBackend {
    Auto,   // GPU + llama-server → llama; else native CPU
    Cpu,    // force native moe_engine
    Llama,  // require llama-server (fail if missing / no GPU unless forced)
};

struct LlamaGpuInfo {
    bool has_nvidia = false;   // nvidia-smi / NVML present
    bool has_vulkan = false;   // vulkan-1 / libvulkan present
    std::string detail;        // human-readable probe result
};

struct LlamaBridgeConfig {
    std::string server_bin;    // empty → search PATH / tools/llama / LLAMA_SERVER
    std::string model_path;
    std::string host = "127.0.0.1";
    uint16_t port = 8010;      // internal upstream port (HttpServer proxies)
    int n_gpu_layers = -1;     // -1 → 99 when GPU, 0 when CPU-only llama
    int ctx_size = 0;          // 0 → llama-server default
    std::string extra_args;    // optional raw args (advanced)
};

struct LlamaBridge {
    // Process handle (pid / PROCESS_INFORMATION bits kept opaque in .cpp).
    void* process = nullptr;
#ifdef _WIN32
    void* job = nullptr;
#endif
    int pid = 0;
    std::string server_bin;
    std::string upstream_host = "127.0.0.1";
    uint16_t upstream_port = 0;
    bool using_gpu = false;
    std::string backend_label; // e.g. "llama.cpp+CUDA" / "llama.cpp+Vulkan"
};

LlamaGpuInfo probe_llama_gpu();
std::string find_llama_server_binary(const std::string& explicit_path = "");

// true when Auto/Llama should try the llama.cpp path.
bool should_try_llama_backend(InferenceBackend mode, const LlamaGpuInfo& gpu,
                              std::string& reason);

// Spawn llama-server; wait until GET /health is ok (or timeout).
// On failure, err_msg is set and no process is left running.
bool start_llama_server(const LlamaBridgeConfig& cfg, const LlamaGpuInfo& gpu,
                        LlamaBridge& out, std::string& err_msg);

void stop_llama_server(LlamaBridge& bridge);

// Parse --backend value: auto|cpu|llama (case-insensitive).
bool parse_inference_backend(const std::string& s, InferenceBackend& out);

} // namespace moe
