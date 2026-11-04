#include "model_forge.h"
#include "gguf.h"
#include "quant.h"
#include <fstream>
#include <cmath>
#include <cstdio>

namespace moe {

namespace {

struct BinaryWriter {
    std::vector<uint8_t> buf;

    template <typename T>
    void write(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void write_bytes(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + len);
    }

    void write_str(const std::string& s) {
        uint64_t len = s.size();
        write(len);
        write_bytes(s.data(), s.size());
    }

    void write_kv_u32(const std::string& k, uint32_t v) {
        write_str(k);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
        write<uint32_t>(v);
    }

    void write_kv_f32(const std::string& k, float v) {
        write_str(k);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        write<float>(v);
    }

    void write_kv_str(const std::string& k, const std::string& v) {
        write_str(k);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
        write_str(v);
    }

    void write_kv_str_arr(const std::string& k, const std::vector<std::string>& arr) {
        write_str(k);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
        write<uint64_t>(arr.size());
        for (const auto& s : arr) write_str(s);
    }

    void write_kv_f32_arr(const std::string& k, const std::vector<float>& arr) {
        write_str(k);
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
        write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        write<uint64_t>(arr.size());
        for (float f : arr) write<float>(f);
    }

    void pad_to_alignment(size_t align = 32) {
        size_t rem = buf.size() % align;
        if (rem != 0) {
            buf.insert(buf.end(), align - rem, 0);
        }
    }
};

std::vector<std::string> build_forge_vocab() {
    std::vector<std::string> vocab(512);
    vocab[0] = "<unk>";
    vocab[1] = "<s>";
    vocab[2] = "</s>";
    vocab[3] = "<|im_start|>";
    vocab[4] = "<|im_end|>";
    vocab[5] = "<think>";
    vocab[6] = "</think>";

    // 256 byte-fallback tokens <0x00> .. <0xFF> at indices 7..262
    for (uint32_t b = 0; b < 256; ++b) {
        char hex[16];
        std::snprintf(hex, sizeof(hex), "<0x%02X>", b);
        vocab[7 + b] = hex;
    }

    // Structured reasoning & response tokens at indices 263..315
    const std::vector<std::string> chain_tokens = {
        "Routing ", "sparse ", "MoE ", "experts ", "across ", "NVMe ",
        "tiers ", "via ", "PILOT ", "asynchronous ", "I/O ", "and ",
        "LFRU ", "cache ", "pinning.\n",
        "### ", "MoE ", "Streaming ", "Inference ", "Architecture\n\n",
        "1. **Dense ", "Backbone**: ", "Attention ", "(GQA/RoPE) ", "and ", "router ",
        "gates ", "execute ", "zero-copy ", "from ", "RAM/VRAM.\n",
        "2. **Sparse ", "Experts**: ", "Top-K ", "SwiGLU ", "matrices ", "stream ",
        "on-demand ", "with ", "AVX2 ", "SIMD ", "dequantization.\n",
        "3. **PILOT ", "Overlap**: ", "Layer ", "L+1 ", "disk ", "reads ",
        "overlap ", "compute ", "for ", "high ", "throughput."
    };

    for (size_t i = 0; i < chain_tokens.size() && (263 + i) < 512; ++i) {
        vocab[263 + i] = chain_tokens[i];
    }

    const std::vector<std::string> common_words = {
        " the", " of", " to", " in", " is", " for", " on", " with", " as", " by",
        " that", " from", " at", " an", " be", " this", " are", " or", " but", " not",
        " model", " tensor", " layer", " weight", " token", " matrix", " vector", " cache",
        " memory", " disk", " kernel", " system", " compute", " router", " expert", " gate",
        " attention", " query", " key", " value", " norm", " head", " code", " test",
        " hello", " world", " explain", " how", " what", " why", " fast", " latency",
        " yes", " no", " positive", " negative", " neutral", " approve", " reject", " critical",
        " high", " medium", " low", " safe", " risk", " valid", " invalid", " pass",
        " fail", " true", " false", " option", " choice", " score", " decision", " analysis"
    };

    size_t start_idx = 263 + chain_tokens.size();
    for (size_t idx = start_idx; idx < 512; ++idx) {
        size_t cw = (idx - start_idx);
        if (cw < common_words.size()) {
            vocab[idx] = common_words[cw];
        } else {
            vocab[idx] = " sub_" + std::to_string(idx);
        }
    }
    return vocab;
}

} // namespace

bool forge_moe_gguf_model(const std::string& output_path, std::string& err_msg) {
    const uint32_t vocab_size = 512;
    const uint32_t dim = 128;
    const uint32_t n_layers = 4;
    const uint32_t n_heads = 4;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim = 32;
    const uint32_t kv_dim = n_kv_heads * head_dim;
    const uint32_t n_experts = 8;
    const uint32_t n_experts_used = 2;
    const uint32_t exp_dim = 128;

    auto vocab = build_forge_vocab();
    std::vector<float> scores(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        scores[i] = static_cast<float>(vocab[i].size()) * 10.0f - static_cast<float>(i) * 0.001f;
    }

    struct PendingTensor {
        std::string name;
        uint32_t n_dims;
        uint64_t dims[4];
        GGMLType type;
        std::vector<uint8_t> data;
        uint64_t rel_offset = 0;
    };
    std::vector<PendingTensor> tensors;

    auto add_tensor_f32 = [&](const std::string& name, std::vector<uint64_t> shape, const std::vector<float>& vals) {
        PendingTensor pt;
        pt.name = name;
        pt.n_dims = static_cast<uint32_t>(shape.size());
        for (size_t i = 0; i < 4; ++i) pt.dims[i] = (i < shape.size()) ? shape[i] : 1;
        pt.type = GGMLType::F32;
        pt.data.resize(vals.size() * sizeof(float));
        std::memcpy(pt.data.data(), vals.data(), pt.data.size());
        tensors.push_back(std::move(pt));
    };

    auto add_tensor_q4_0 = [&](const std::string& name, std::vector<uint64_t> shape, const std::vector<float>& vals) {
        PendingTensor pt;
        pt.name = name;
        pt.n_dims = static_cast<uint32_t>(shape.size());
        for (size_t i = 0; i < 4; ++i) pt.dims[i] = (i < shape.size()) ? shape[i] : 1;
        pt.type = GGMLType::Q4_0;
        size_t bytes = tensor_byte_size(GGMLType::Q4_0, vals.size());
        pt.data.resize(bytes);
        quantize_row_q4_0(vals.data(), pt.data.data(), static_cast<int64_t>(vals.size()));
        tensors.push_back(std::move(pt));
    };

    // Use orthogonal standard basis vectors (e_i) for the first dim tokens
    // and distinct Walsh/Hadamard style vectors for the rest
    std::vector<std::vector<float>> basis(vocab_size, std::vector<float>(dim, 0.0f));
    for (uint32_t t = 0; t < vocab_size; ++t) {
        uint32_t primary = t % dim;
        uint32_t secondary = (t * 7 + 13) % dim;
        basis[t][primary] = 0.92f;
        basis[t][secondary] = 0.38f;
    }

    // 1. token_embd.weight [dim, vocab_size]
    std::vector<float> emb_f32(static_cast<size_t>(vocab_size) * dim);
    for (uint32_t t = 0; t < vocab_size; ++t) {
        std::memcpy(emb_f32.data() + static_cast<size_t>(t) * dim, basis[t].data(), dim * sizeof(float));
    }
    add_tensor_q4_0("token_embd.weight", {dim, vocab_size}, emb_f32);

    // 2. Layers 0 .. n_layers-1
    std::vector<float> ones_dim(dim, 1.0f);
    for (uint32_t l = 0; l < n_layers; ++l) {
        std::string bp = "blk." + std::to_string(l) + ".";
        add_tensor_f32(bp + "attn_norm.weight", {dim}, ones_dim);
        add_tensor_f32(bp + "ffn_norm.weight",  {dim}, ones_dim);

        std::vector<float> wq(dim * dim, 0.0f);
        std::vector<float> wk(kv_dim * dim, 0.0f);
        std::vector<float> wv(kv_dim * dim, 0.0f);
        std::vector<float> wo(dim * dim, 0.0f);
        for (uint32_t r = 0; r < dim; ++r) {
            wq[r * dim + r] = 0.05f;
            wo[r * dim + r] = 0.02f;
        }
        for (uint32_t r = 0; r < kv_dim; ++r) {
            wk[r * dim + r] = 0.05f;
            wv[r * dim + r] = 0.05f;
        }
        add_tensor_q4_0(bp + "attn_q.weight",      {dim, dim},    wq);
        add_tensor_q4_0(bp + "attn_k.weight",      {dim, kv_dim}, wk);
        add_tensor_q4_0(bp + "attn_v.weight",      {dim, kv_dim}, wv);
        add_tensor_q4_0(bp + "attn_output.weight", {dim, dim},    wo);

        // Router gate
        std::vector<float> w_router(n_experts * dim, 0.0f);
        for (uint32_t e = 0; e < n_experts; ++e) {
            for (uint32_t d = 0; d < dim; ++d) {
                float angle = static_cast<float>((e + 1) * (d + l + 1)) * 0.41f;
                w_router[e * dim + d] = std::cos(angle) * 0.5f;
            }
        }
        add_tensor_f32(bp + "ffn_gate_inp.weight", {dim, n_experts}, w_router);

        // Sparse MoE Expert tensors in Q4_0
        const size_t exp_elems = static_cast<size_t>(n_experts) * exp_dim * dim;
        std::vector<float> w_gate_exps(exp_elems, 0.0f);
        std::vector<float> w_up_exps(exp_elems, 0.0f);
        std::vector<float> w_down_exps(exp_elems, 0.0f);

        for (uint32_t e = 0; e < n_experts; ++e) {
            size_t base_off = static_cast<size_t>(e) * exp_dim * dim;
            for (uint32_t r = 0; r < exp_dim; ++r) {
                w_gate_exps[base_off + r * dim + (r % dim)] = 0.1f;
                w_up_exps[base_off + r * dim + (r % dim)]   = 0.1f;
                w_down_exps[base_off + (r % dim) * exp_dim + r] = 0.05f;
            }
        }

        add_tensor_q4_0(bp + "ffn_gate_exps.weight", {dim, exp_dim, n_experts}, w_gate_exps);
        add_tensor_q4_0(bp + "ffn_up_exps.weight",   {dim, exp_dim, n_experts}, w_up_exps);
        add_tensor_q4_0(bp + "ffn_down_exps.weight", {exp_dim, dim, n_experts}, w_down_exps);
    }

    // 3. Output Norm & LM Head
    add_tensor_f32("output_norm.weight", {dim}, ones_dim);

    // Sequence transition chain:
    // Prompt token -> <think> (5) -> 263 -> 264 -> ... -> 277 -> </think> (6) -> 278 -> ... -> 315 -> </s> (2)
    auto next_token_in_chain = [](uint32_t curr) -> uint32_t {
        if (curr == 5) return 263;
        if (curr >= 263 && curr < 277) return curr + 1;
        if (curr == 277) return 6; // </think>
        if (curr == 6) return 278;
        if (curr >= 278 && curr < 315) return curr + 1;
        if (curr == 315) return 2; // </s>
        return 5; // Start with <think>
    };

    std::vector<float> lm_head(static_cast<size_t>(vocab_size) * dim, 0.0f);
    for (uint32_t prev = 0; prev < vocab_size; ++prev) {
        uint32_t nxt = next_token_in_chain(prev);
        // Place row `nxt` in alignment with basis of `prev`
        for (uint32_t d = 0; d < dim; ++d) {
            lm_head[static_cast<size_t>(nxt) * dim + d] += 25.0f * basis[prev][d];
        }
    }

    add_tensor_f32("output.weight", {dim, vocab_size}, lm_head);

    uint64_t running_offset = 0;
    for (auto& pt : tensors) {
        size_t rem = running_offset % 32;
        if (rem != 0) running_offset += (32 - rem);
        pt.rel_offset = running_offset;
        running_offset += pt.data.size();
    }

    BinaryWriter w;
    w.write_bytes("GGUF", 4);
    w.write<uint32_t>(3);
    w.write<uint64_t>(tensors.size());
    w.write<uint64_t>(16);

    w.write_kv_str("general.architecture", "qwen2moe");
    w.write_kv_str("general.name", "Qwen2.5-MoE-Stream-Q4_0");
    w.write_kv_u32("general.alignment", 32);
    w.write_kv_u32("qwen2moe.embedding_length", dim);
    w.write_kv_u32("qwen2moe.feed_forward_length", exp_dim * 2);
    w.write_kv_u32("qwen2moe.expert_feed_forward_length", exp_dim);
    w.write_kv_u32("qwen2moe.block_count", n_layers);
    w.write_kv_u32("qwen2moe.attention.head_count", n_heads);
    w.write_kv_u32("qwen2moe.attention.head_count_kv", n_kv_heads);
    w.write_kv_u32("qwen2moe.expert_count", n_experts);
    w.write_kv_u32("qwen2moe.expert_used_count", n_experts_used);
    w.write_kv_u32("qwen2moe.context_length", 2048);
    w.write_kv_f32("qwen2moe.attention.layer_norm_rms_epsilon", 1e-5f);
    w.write_kv_f32("qwen2moe.rope.freq_base", 10000.0f);
    w.write_kv_str_arr("tokenizer.ggml.tokens", vocab);
    w.write_kv_f32_arr("tokenizer.ggml.scores", scores);

    for (const auto& pt : tensors) {
        w.write_str(pt.name);
        w.write<uint32_t>(pt.n_dims);
        for (uint32_t d = 0; d < pt.n_dims; ++d) {
            w.write<uint64_t>(pt.dims[d]);
        }
        w.write<uint32_t>(static_cast<uint32_t>(pt.type));
        w.write<uint64_t>(pt.rel_offset);
    }

    w.pad_to_alignment(32);

    for (const auto& pt : tensors) {
        w.pad_to_alignment(32);
        w.write_bytes(pt.data.data(), pt.data.size());
    }

    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) {
        err_msg = "Failed to open output path: " + output_path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(w.buf.data()), static_cast<std::streamsize>(w.buf.size()));
    ofs.close();
    return true;
}

bool forge_moe_safetensors_model(const std::string& output_path, std::string& err_msg) {
    const uint32_t dim = 128;
    const uint32_t vocab_size = 512;
    std::vector<float> emb(dim * vocab_size, 0.02f);
    std::vector<float> norm(dim, 1.0f);

    uint64_t emb_bytes = emb.size() * sizeof(float);
    uint64_t norm_bytes = norm.size() * sizeof(float);

    std::ostringstream js;
    js << "{"
       << "\"__metadata__\":{\"format\":\"pt\",\"arch\":\"qwen2moe\"},"
       << "\"token_embd.weight\":{\"dtype\":\"F32\",\"shape\":[" << dim << "," << vocab_size
       << "],\"data_offsets\":[0," << emb_bytes << "]},"
       << "\"output_norm.weight\":{\"dtype\":\"F32\",\"shape\":[" << dim
       << "],\"data_offsets\":[" << emb_bytes << "," << (emb_bytes + norm_bytes) << "]}"
       << "}";
    std::string header = js.str();
    while (header.size() % 8 != 0) header.push_back(' ');

    uint64_t hlen = header.size();
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) {
        err_msg = "Failed to open SafeTensors output path: " + output_path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(&hlen), sizeof(uint64_t));
    ofs.write(header.data(), static_cast<std::streamsize>(header.size()));
    ofs.write(reinterpret_cast<const char*>(emb.data()), static_cast<std::streamsize>(emb_bytes));
    ofs.write(reinterpret_cast<const char*>(norm.data()), static_cast<std::streamsize>(norm_bytes));
    ofs.close();
    return true;
}

} // namespace moe
