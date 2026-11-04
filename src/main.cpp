#include "common.h"
#include "transformer.h"
#include "model_forge.h"
#include "server.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;
using namespace moe;

void print_banner() {
    std::cout << "\033[1;36m"
              << "===================================================================\n"
              << "  NEXT-GEN REAL-EXECUTION MoE STREAMING INFERENCE ENGINE & SUITE   \n"
              << "  Pipelined NVMe Sparse Streaming (PILOT) + LFRU Dynamic Pinning   \n"
              << "===================================================================\n"
              << "\033[0m";
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  moe_engine chat  --model <path.gguf|path.safetensors> [--prompt \"...\"] [--max-tokens 64]\n"
              << "  moe_engine serve --model <path.gguf|path.safetensors> [--port 8000] [--web-dir web/dist]\n"
              << "  moe_engine forge --out <output_model.gguf> [--format gguf|safetensors]\n"
              << "  moe_engine test  [run automated unit & verification tests]\n";
}

bool run_verification_tests() {
    std::cout << "\n\033[1;33m[TEST] Running MoE Engine Rigorous Verification Suite...\033[0m\n";

    // 1. Quantization & Dot Product Verification
    std::cout << "  [1/5] Testing Q4_0 & Q8_0 dequantize-on-the-fly dot products...";
    const int K = 128;
    std::vector<float> x(K), y(K);
    for (int i = 0; i < K; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.1f);
        y[i] = std::cos(static_cast<float>(i) * 0.2f);
    }
    float ref_dot = vec_dot_f32(x.data(), y.data(), K);

    std::vector<uint8_t> q8_buf(tensor_byte_size(GGMLType::Q8_0, K));
    quantize_row_q8_0(x.data(), q8_buf.data(), K);
    float q8_dot = vec_dot_q8_0_f32(q8_buf.data(), y.data(), K);

    std::vector<uint8_t> q4_buf(tensor_byte_size(GGMLType::Q4_0, K));
    quantize_row_q4_0(x.data(), q4_buf.data(), K);
    float q4_dot = vec_dot_q4_0_f32(q4_buf.data(), y.data(), K);

    assert(std::fabs(ref_dot - q8_dot) < 0.2f);
    assert(std::fabs(ref_dot - q4_dot) < 0.8f);
    std::cout << " \033[1;32mPASSED\033[0m (Ref: " << ref_dot << ", Q8: " << q8_dot << ", Q4: " << q4_dot << ")\n";

    // 2. RMSNorm Math Verification
    std::cout << "  [2/5] Testing RMSNorm invariance & scale...";
    std::vector<float> norm_out(K);
    MoETransformerEngine::rms_norm(norm_out.data(), x.data(), nullptr, K, 1e-5f);
    float ss = 0.0f;
    for (float v : norm_out) ss += v * v;
    float rms = std::sqrt(ss / static_cast<float>(K));
    assert(std::fabs(rms - 1.0f) < 1e-3f);
    std::cout << " \033[1;32mPASSED\033[0m (Output RMS: " << rms << ")\n";

    // 3. RoPE (Rotary Position Embedding) Verification
    std::cout << "  [3/5] Testing RoPE 2D rotation preserving norm...";
    std::vector<float> q_rope = x;
    float norm_before = vec_dot_f32(q_rope.data(), q_rope.data(), K);
    MoETransformerEngine::apply_rope(q_rope.data(), nullptr, 4, 2, 32, 32, 10, 10000.0f);
    float norm_after = vec_dot_f32(q_rope.data(), q_rope.data(), K);
    assert(std::fabs(norm_before - norm_after) < 1e-3f);
    std::cout << " \033[1;32mPASSED\033[0m (Norm before: " << norm_before << ", after: " << norm_after << ")\n";

    // 4. Model Forge GGUF Generator & Parser Verification
    std::cout << "  [4/5] Testing GGUF v3 Binary Generation & Model Parser...";
    std::string test_model_path = "test_model_verify.gguf";
    std::string err_msg;
    if (!forge_moe_gguf_model(test_model_path, err_msg)) {
        std::cerr << "Forge failed: " << err_msg << "\n";
        return false;
    }

    {
        ModelFile mf;
        if (!mf.open(test_model_path, err_msg)) {
            std::cerr << "Parse failed: " << err_msg << "\n";
            return false;
        }
        assert(mf.format() == "GGUFv3");
        assert(mf.find_tensor("token_embd.weight") != nullptr);
        assert(mf.find_tensor("blk.0.ffn_gate_exps.weight") != nullptr);
        std::cout << " \033[1;32mPASSED\033[0m (Parsed " << mf.tensors().size() << " tensors, size: " << mf.file_size() << " bytes)\n";
        mf.close();
    }

    // 5. Full Real Forward Pass & End-to-End Generation Test
    std::cout << "  [5/5] Testing Real MoE Forward Pass & Token Generation...";
    {
        MoETransformerEngine engine;
        if (!engine.load_model(test_model_path, err_msg)) {
            std::cerr << "Engine load failed: " << err_msg << "\n";
            return false;
        }

        std::string gen_output;
        bool had_reasoning = false;
        engine.generate_stream("MoE Architecture", 32, 0, true, {0.0f, 1, 0.9f, 1.0f, 42},
            [&](const std::string& piece, bool is_reasoning, uint32_t, const ProfileSample&) {
                if (is_reasoning) had_reasoning = true;
                gen_output += piece;
                return true;
            }
        );

        assert(!gen_output.empty());
        std::cout << " \033[1;32mPASSED\033[0m\n";
        std::cout << "    \033[36mGenerated token stream: \033[0m" << gen_output << "\n";
        std::cout << "    Reasoning detected: " << (had_reasoning ? "YES (<think> channel)" : "NO") << "\n";
        engine.unload();
    }

    std::cout << "\n\033[1;32m[ALL 5/5 VERIFICATION TESTS PASSED SUCCESSFULLY]\033[0m\n\n";

    std::error_code ec;
    fs::remove(test_model_path, ec);
    return true;
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "test") {
        return run_verification_tests() ? 0 : 1;
    }

    std::string model_path = "models/qwen2moe_stream_q4.gguf";
    std::string prompt = "Explain the MoE streaming and PILOT prefetch architecture.";
    uint16_t port = 8000;
    std::string web_dir = "web/dist";
    uint32_t max_tokens = 64;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoi(argv[++i]));
        else if (arg == "--web-dir" && i + 1 < argc) web_dir = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
    }

    if (mode == "forge") {
        std::string out_path = (argc > 3 && std::string(argv[2]) == "--out") ? argv[3] : model_path;
        std::cout << "[FORGE] Creating real GGUF v3 MoE model file at: " << out_path << "...\n";
        fs::create_directories(fs::path(out_path).parent_path());
        std::string err;
        if (!forge_moe_gguf_model(out_path, err)) {
            std::cerr << "Error forging model: " << err << "\n";
            return 1;
        }
        std::cout << "\033[1;32m[SUCCESS] GGUF v3 MoE Model forged successfully (" << fs::file_size(out_path) << " bytes).\033[0m\n";
        return 0;
    }

    // Auto-forge model if specified file does not exist
    if (!fs::exists(model_path)) {
        std::cout << "[NOTE] Model file '" << model_path << "' not found. Automatically forging operational MoE model...\n";
        if (fs::path(model_path).has_parent_path()) {
            fs::create_directories(fs::path(model_path).parent_path());
        }
        std::string err;
        if (!forge_moe_gguf_model(model_path, err)) {
            std::cerr << "Failed to forge initial model: " << err << "\n";
            return 1;
        }
        std::cout << "\033[1;32m[MODEL READY] " << model_path << " (" << fs::file_size(model_path) << " bytes)\033[0m\n\n";
    }

    MoETransformerEngine engine;
    std::string err_msg;
    std::cout << "[INIT] Loading Model: " << model_path << "...\n";
    if (!engine.load_model(model_path, err_msg)) {
        std::cerr << "Failed to load model: " << err_msg << "\n";
        return 1;
    }

    const auto& hw = engine.hardware();
    const auto& hp = engine.hparams();
    std::cout << "\033[1;32m[ENGINE READY]\033[0m\n"
              << "  Model Name:       " << hp.model_name << " (" << hp.arch << ")\n"
              << "  Dense Backbone:   " << (engine.dense_backbone_bytes() / 1024) << " KB (in RAM/VRAM zero-copy)\n"
              << "  Sparse NVMe MoE:  " << (engine.sparse_disk_bytes() / 1024) << " KB (streamed on-demand)\n"
              << "  Experts / Layer:  " << hp.n_experts << " (active top-" << hp.n_experts_used << ")\n"
              << "  SIMD Backend:     " << hw.simd_backend << " (" << hw.logical_cores << " threads)\n"
              << "  Async I/O:        " << hw.io_backend << "\n\n";

    if (mode == "chat") {
        std::cout << "\033[1;33mInteractive MoE Streaming Console\033[0m (Type prompt or press Enter for default, 'exit' to quit)\n";
        while (true) {
            std::cout << "\n\033[1;34mUser > \033[0m";
            std::string line;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) line = prompt;
            if (line == "exit" || line == "quit") break;

            std::cout << "\033[1;35mMoE Assistant:\033[0m\n";
            bool thinking = false;
            double t0 = now_ms();
            uint32_t count = 0;

            engine.generate_stream(line, max_tokens, 0, true, {0.7f, 40, 0.9f, 1.1f, 42},
                [&](const std::string& piece, bool is_reasoning, uint32_t, const ProfileSample&) {
                    if (is_reasoning && !thinking) {
                        std::cout << "\033[2m[Reasoning: ";
                        thinking = true;
                    } else if (!is_reasoning && thinking) {
                        std::cout << "]\033[0m\n";
                        thinking = false;
                    }
                    std::cout << piece << std::flush;
                    count++;
                    return true;
                }
            );
            if (thinking) std::cout << "]\033[0m\n";
            double elapsed = (now_ms() - t0) / 1000.0;
            std::cout << "\n\033[2m(" << count << " tokens, " << std::fixed << std::setprecision(1) << (count / std::max(0.001, elapsed))
                      << " tok/s, " << (elapsed * 1000.0) << " ms)\033[0m\n";
        }
    } else if (mode == "serve") {
        std::cout << "\033[1;32m[GATEWAY STARTING]\033[0m Port: " << port << ", Web Root: " << web_dir << "\n";
        std::cout << "  -> Web UI:               http://localhost:" << port << "/\n"
                  << "  -> OpenAI Chat Endpoint: http://localhost:" << port << "/v1/chat/completions\n"
                  << "  -> Health Endpoint:      http://localhost:" << port << "/health\n"
                  << "  -> Telemetry Profile:    http://localhost:" << port << "/profile\n"
                  << "  -> Cortex Atlas Hitmap:  http://localhost:" << port << "/experts\n"
                  << "  -> SystemOne Classifier: http://localhost:" << port << "/v1/systemone\n"
                  << "  -> Neural Latent Image:  http://localhost:" << port << "/v1/images/generations\n\n";

        HttpServer server(engine, web_dir);
        if (!server.start(port, err_msg)) {
            std::cerr << "Server error: " << err_msg << "\n";
            return 1;
        }
    } else {
        print_usage();
    }

    return 0;
}
