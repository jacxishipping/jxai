#include "gguf.h"
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace moe {

ModelFile::~ModelFile() {
    close();
}

void ModelFile::close() {
#ifdef _WIN32
    if (mapped_base_) {
        UnmapViewOfFile(mapped_base_);
        mapped_base_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(map_handle_);
        map_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mapped_base_ && file_size_ > 0) {
        munmap(const_cast<uint8_t*>(mapped_base_), file_size_);
        mapped_base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    metadata_.clear();
    tensors_.clear();
    tensor_order_.clear();
    alias_.clear();
    config_loaded_ = false;
    detected_arch_.clear();
    config_path_.clear();
    tokenizer_path_.clear();
    st_moe_unsupported_ = false;
    st_untranslated_.clear();
    st_unknown_dtypes_.clear();
    file_size_ = 0;
}

bool ModelFile::open(const std::string& path, std::string& err_msg) {
    close();
    path_ = path;

#ifdef _WIN32
    file_handle_ = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        err_msg = "Failed to open model file: " + path;
        return false;
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_handle_, &sz) || sz.QuadPart < 16) {
        err_msg = "Invalid or empty model file: " + path;
        close();
        return false;
    }
    file_size_ = static_cast<uint64_t>(sz.QuadPart);

    map_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_handle_) {
        err_msg = "CreateFileMapping failed for: " + path;
        close();
        return false;
    }

    mapped_base_ = static_cast<const uint8_t*>(MapViewOfFile(map_handle_, FILE_MAP_READ, 0, 0, 0));
    if (!mapped_base_) {
        err_msg = "MapViewOfFile failed for: " + path;
        close();
        return false;
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        err_msg = "Failed to open model file: " + path;
        return false;
    }
    struct stat st;
    if (fstat(fd_, &st) != 0 || st.st_size < 16) {
        err_msg = "Invalid or empty model file: " + path;
        close();
        return false;
    }
    file_size_ = static_cast<uint64_t>(st.st_size);
    void* ptr = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        err_msg = "mmap failed for: " + path;
        close();
        return false;
    }
    mapped_base_ = static_cast<const uint8_t*>(ptr);
#endif

    // Check magic bytes for GGUF ("GGUF" = 0x46554747)
    if (std::memcmp(mapped_base_, "GGUF", 4) == 0) {
        format_ = "GGUFv3";
        return parse_gguf(err_msg);
    } else {
        format_ = "SafeTensors";
        if (!parse_safetensors(err_msg)) return false;
        // Best-effort only: a missing/unparseable config.json must not fail
        // the open — the engine then falls back to its previous defaults
        // (and the caller can check config_loaded() / detected_arch()).
        load_safetensors_config();
        return true;
    }
}

namespace {

struct BinaryCursor {
    const uint8_t* base;
    uint64_t size;
    uint64_t pos = 0;

    bool can_read(uint64_t n) const { return pos + n <= size; }

    template <typename T>
    bool read(T& out) {
        if (!can_read(sizeof(T))) return false;
        std::memcpy(&out, base + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_string(std::string& out) {
        uint64_t len = 0;
        if (!read(len)) return false;
        if (!can_read(len)) return false;
        out.assign(reinterpret_cast<const char*>(base + pos), static_cast<size_t>(len));
        pos += len;
        return true;
    }
};

bool is_expert_tensor_name(const std::string& name) {
    return name.find("ffn_gate_exps") != std::string::npos ||
           name.find("ffn_up_exps")   != std::string::npos ||
           name.find("ffn_down_exps") != std::string::npos ||
           name.find(".experts.")     != std::string::npos;
}

// ---------------------------------------------------------------------------
// SafeTensors helpers: minimal flat-JSON field extraction (no JSON library),
// HF architecture mapping, and HF -> GGUF tensor-name translation.
// ---------------------------------------------------------------------------

// First string value for a flat JSON key, e.g. "torch_dtype": "bfloat16".
bool cfg_get_string(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    size_t i = q1 + 1;
    out.clear();
    while (i < json.size() && json[i] != '"') {
        if (json[i] == '\\' && i + 1 < json.size()) ++i;
        out.push_back(json[i++]);
    }
    return i < json.size();
}

bool cfg_get_number(const std::string& json, const std::string& key, double& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    size_t start = i;
    while (i < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[i])) || json[i] == '-' ||
            json[i] == '+' || json[i] == '.' || json[i] == 'e' || json[i] == 'E')) ++i;
    if (i == start) return false;
    try { out = std::stod(json.substr(start, i - start)); } catch (...) { return false; }
    return true;
}

bool cfg_get_bool(const std::string& json, const std::string& key, bool& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    std::string rest = json.substr(i, 5);
    if (rest.rfind("true", 0) == 0) { out = true; return true; }
    if (rest.rfind("false", 0) == 0) { out = false; return true; }
    return false;
}

// First string element of an array value, e.g. "architectures": ["Qwen2..."].
bool cfg_get_first_array_string(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t br = json.find('[', k + pat.size());
    if (br == std::string::npos) return false;
    size_t q1 = json.find('"', br + 1);
    if (q1 == std::string::npos) return false;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    out = json.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

// HF transformers class name -> GGUF architecture string used by the engine
// (general.architecture, and the "<arch>.*" metadata prefix).
std::string hf_arch_to_gguf(const std::string& hf_class) {
    static const std::pair<const char*, const char*> kArch[] = {
        {"Qwen2ForCausalLM", "qwen2"},   {"Qwen2MoeForCausalLM", "qwen2moe"},
        {"Qwen3ForCausalLM", "qwen3"},   {"Qwen3MoeForCausalLM", "qwen3moe"},
        {"LlamaForCausalLM", "llama"},   {"MistralForCausalLM", "mistral"},
        {"MixtralForCausalLM", "mixtral"},
        {"GemmaForCausalLM", "gemma"},   {"Gemma2ForCausalLM", "gemma2"},
        {"Gemma3ForCausalLM", "gemma3"},
        {"PhiForCausalLM", "phi"},       {"Phi3ForCausalLM", "phi3"},
        {"Phi3SmallForCausalLM", "phi3"},
        {"DeepseekForCausalLM", "deepseek"},
        {"DeepseekV2ForCausalLM", "deepseek2"}, {"DeepseekV3ForCausalLM", "deepseek2"},
        {"FalconForCausalLM", "falcon"}, {"GPTNeoXForCausalLM", "gptneox"},
        {"GPTJForCausalLM", "gptj"},     {"OlmoForCausalLM", "olmo"},
        {"OlmoeForCausalLM", "olmoe"},   {"ChatGLMForCausalLM", "glm"},
        {"GLMForCausalLM", "glm"},       {"InternLM2ForCausalLM", "internlm2"},
        {"StableLmForCausalLM", "stablelm"},
        {"CohereForCausalLM", "cohere"}, {"DbrxForCausalLM", "dbrx"},
    };
    for (const auto& [hf, gguf] : kArch) {
        if (hf_class == hf) return gguf;
    }
    return "";
}

bool is_known_gguf_arch(const std::string& arch) {
    static const char* kKnown[] = {
        "qwen2", "qwen2moe", "qwen3", "qwen3moe", "llama", "mistral", "mixtral",
        "gemma", "gemma2", "gemma3", "gemma4", "phi", "phi3", "deepseek",
        "deepseek2", "falcon", "gptneox", "gptj", "olmo", "olmoe", "glm",
        "internlm2", "stablelm", "cohere", "dbrx",
    };
    for (const char* k : kKnown) if (arch == k) return true;
    return false;
}

std::vector<std::string> split_dots(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t dot = s.find('.', start);
        if (dot == std::string::npos) { parts.push_back(s.substr(start)); break; }
        parts.push_back(s.substr(start, dot - start));
        start = dot + 1;
    }
    return parts;
}

// Translate one HF (transformers) tensor name to its GGUF equivalent.
// Returns "" when there is no translation (unknown pattern, MoE experts,
// fused projections). The caller records those in st_untranslated_.
std::string translate_hf_tensor_name(const std::string& hf_name) {
    std::vector<std::string> p = split_dots(hf_name);
    if (!p.empty() && p[0] == "model") p.erase(p.begin());
    if (p.size() == 2 && p[0] == "embed_tokens" && p[1] == "weight") return "token_embd.weight";
    if (p.size() == 2 && p[0] == "norm" && p[1] == "weight") return "output_norm.weight";
    if (p.size() == 2 && p[0] == "lm_head" && p[1] == "weight") return "output.weight";
    if (p.size() < 4 || p[0] != "layers") return "";
    const std::string& layer = p[1];
    const std::string blk = "blk." + layer + ".";
    if (p[2] == "input_layernorm" && p[3] == "weight") return blk + "attn_norm.weight";
    if (p[2] == "post_attention_layernorm" && p[3] == "weight") return blk + "ffn_norm.weight";
    if (p[2] == "self_attn" && p.size() >= 4) {
        std::string suffix = (p.size() >= 5) ? p[4] : "weight"; // "weight" or "bias"
        if (suffix != "weight" && suffix != "bias") return "";
        if (p[3] == "q_proj") return blk + "attn_q." + suffix;
        if (p[3] == "k_proj") return blk + "attn_k." + suffix;
        if (p[3] == "v_proj") return blk + "attn_v." + suffix;
        if (p[3] == "o_proj") return blk + "attn_output." + suffix;
        if (p[3] == "q_norm") return blk + "attn_q_norm.weight";
        if (p[3] == "k_norm") return blk + "attn_k_norm.weight";
        return ""; // qkv_proj (fused) and anything else: no translation
    }
    if (p[2] == "mlp" && p.size() >= 4) {
        if (p[3] == "gate_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_gate.weight";
        if (p[3] == "up_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_up.weight";
        if (p[3] == "down_proj" && p.size() >= 5 && p[4] == "weight") return blk + "ffn_down.weight";
        // Qwen2.5-MoE / DeepSeek shared expert (dense tensors, translatable).
        if (p[3] == "shared_expert" && p.size() >= 6 && p[5] == "weight") {
            if (p[4] == "gate_proj") return blk + "ffn_gate_shexp.weight";
            if (p[4] == "up_proj") return blk + "ffn_up_shexp.weight";
            if (p[4] == "down_proj") return blk + "ffn_down_shexp.weight";
        }
        return ""; // routed experts ("mlp.experts.<i>.*") and the rest
    }
    return "";
}

bool read_gguf_value(BinaryCursor& cur, GGUFValueType vtype, GGUFValue& val) {
    val.type = vtype;
    switch (vtype) {
        case GGUFValueType::UINT8: {
            uint8_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT8: {
            int8_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::UINT16: {
            uint16_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT16: {
            int16_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::UINT32: {
            uint32_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = v; return true;
        }
        case GGUFValueType::INT32: {
            int32_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::FLOAT32: {
            float v; if (!cur.read(v)) return false;
            val.f64 = v; return true;
        }
        case GGUFValueType::BOOL: {
            uint8_t v; if (!cur.read(v)) return false;
            val.bval = (v != 0); val.u64 = val.bval ? 1 : 0; return true;
        }
        case GGUFValueType::STRING: {
            return cur.read_string(val.str);
        }
        case GGUFValueType::UINT64: {
            uint64_t v; if (!cur.read(v)) return false;
            val.u64 = v; val.i64 = static_cast<int64_t>(v); return true;
        }
        case GGUFValueType::INT64: {
            int64_t v; if (!cur.read(v)) return false;
            val.i64 = v; val.u64 = static_cast<uint64_t>(v); return true;
        }
        case GGUFValueType::FLOAT64: {
            double v; if (!cur.read(v)) return false;
            val.f64 = v; return true;
        }
        case GGUFValueType::ARRAY: {
            uint32_t elem_type_raw = 0;
            uint64_t arr_len = 0;
            if (!cur.read(elem_type_raw) || !cur.read(arr_len)) return false;
            if (arr_len > 10000000ULL) return false;
            auto etype = static_cast<GGUFValueType>(elem_type_raw);
            if (etype == GGUFValueType::STRING) {
                val.str_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read_string(val.str_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::FLOAT32) {
                val.f32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read(val.f32_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::INT32 || etype == GGUFValueType::UINT32) {
                val.i32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!cur.read(val.i32_arr[static_cast<size_t>(i)])) return false;
                }
            } else if (etype == GGUFValueType::BOOL || etype == GGUFValueType::UINT8 || etype == GGUFValueType::INT8) {
                val.i32_arr.resize(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; ++i) {
                    uint8_t b = 0;
                    if (!cur.read(b)) return false;
                    val.i32_arr[static_cast<size_t>(i)] = static_cast<int32_t>(b);
                }
            } else {
                for (uint64_t i = 0; i < arr_len; ++i) {
                    GGUFValue dummy;
                    if (!read_gguf_value(cur, etype, dummy)) return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

} // namespace

bool ModelFile::parse_gguf(std::string& err_msg) {
    BinaryCursor cur{mapped_base_, file_size_, 4}; // skip "GGUF"
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;

    if (!cur.read(version) || (version != 2 && version != 3)) {
        err_msg = "Unsupported GGUF version: " + std::to_string(version);
        return false;
    }
    if (!cur.read(n_tensors) || !cur.read(n_kv)) {
        err_msg = "Truncated GGUF header";
        return false;
    }

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t vtype_raw = 0;
        if (!cur.read_string(key) || !cur.read(vtype_raw)) {
            err_msg = "Failed reading GGUF KV entry " + std::to_string(i);
            return false;
        }
        GGUFValue val;
        if (!read_gguf_value(cur, static_cast<GGUFValueType>(vtype_raw), val)) {
            err_msg = "Failed reading GGUF KV value for key: " + key;
            return false;
        }
        if (key == "general.alignment" && val.u64 > 0) {
            alignment_ = static_cast<uint32_t>(val.u64);
        }
        metadata_[key] = std::move(val);
    }

    std::vector<TensorDescriptor> temp_tensors;
    temp_tensors.reserve(static_cast<size_t>(n_tensors));

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorDescriptor td;
        if (!cur.read_string(td.name) || !cur.read(td.n_dims) || td.n_dims > 4) {
            err_msg = "Invalid GGUF tensor header at index " + std::to_string(i);
            return false;
        }
        uint64_t ne = 1;
        for (uint32_t d = 0; d < td.n_dims; ++d) {
            if (!cur.read(td.dims[d])) return false;
            ne *= td.dims[d];
        }
        uint32_t type_raw = 0;
        if (!cur.read(type_raw) || !cur.read(td.offset)) {
            err_msg = "Truncated GGUF tensor descriptor: " + td.name;
            return false;
        }
        td.type = static_cast<GGMLType>(type_raw);
        td.byte_size = tensor_byte_size(td.type, ne);
        td.is_sparse_expert = is_expert_tensor_name(td.name);
        temp_tensors.push_back(std::move(td));
    }

    // Align cursor position to general.alignment (default 32 bytes)
    uint64_t rem = cur.pos % alignment_;
    data_offset_ = (rem == 0) ? cur.pos : (cur.pos + (alignment_ - rem));

    for (auto& td : temp_tensors) {
        td.abs_file_offset = data_offset_ + td.offset;
        if (td.abs_file_offset + td.byte_size > file_size_) {
            err_msg = "Tensor exceeds file bounds: " + td.name +
                      " (abs_offset: " + std::to_string(td.abs_file_offset) +
                      ", byte_size: " + std::to_string(td.byte_size) +
                      ", type: " + std::to_string(static_cast<int>(td.type)) +
                      ", file_size: " + std::to_string(file_size_) + ")";
            return false;
        }
        td.mapped_ptr = mapped_base_ + td.abs_file_offset;
        tensor_order_.push_back(td.name);
        tensors_[td.name] = std::move(td);
    }

    return true;
}

// Native SafeTensors binary + JSON header parser
bool ModelFile::parse_safetensors(std::string& err_msg) {
    if (file_size_ < 8) {
        err_msg = "SafeTensors file too small";
        return false;
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, mapped_base_, sizeof(uint64_t));
    if (header_len == 0 || 8 + header_len > file_size_) {
        err_msg = "Invalid SafeTensors JSON header size";
        return false;
    }

    data_offset_ = 8 + header_len;
    std::string json(reinterpret_cast<const char*>(mapped_base_ + 8), static_cast<size_t>(header_len));

    // Parse top-level tensor objects in SafeTensors JSON header:
    // { "tensor_name": { "dtype": "F16", "shape": [r, c], "data_offsets": [start, end] }, "__metadata__": {...} }
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == ',')) pos++;
    };
    auto parse_str = [&](std::string& s) -> bool {
        skip_ws();
        if (pos >= json.size() || json[pos] != '"') return false;
        pos++;
        s.clear();
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) pos++;
            s.push_back(json[pos++]);
        }
        if (pos >= json.size()) return false;
        pos++; // closing quote
        return true;
    };

    skip_ws();
    if (pos >= json.size() || json[pos] != '{') {
        err_msg = "SafeTensors header does not start with '{'";
        return false;
    }
    pos++;

    while (pos < json.size()) {
        skip_ws();
        if (pos >= json.size() || json[pos] == '}') break;
        std::string key;
        if (!parse_str(key)) break;
        skip_ws();
        if (pos >= json.size() || json[pos] != ':') break;
        pos++;
        skip_ws();
        if (pos >= json.size() || json[pos] != '{') break;

        // Find matching '}' for inner object
        size_t obj_start = pos;
        int depth = 0;
        while (pos < json.size()) {
            if (json[pos] == '{') depth++;
            else if (json[pos] == '}') {
                depth--;
                if (depth == 0) { pos++; break; }
            }
            pos++;
        }
        std::string obj = json.substr(obj_start, pos - obj_start);

        if (key == "__metadata__") {
            // Previously ignored entirely. Store string pairs as metadata
            // (usually just {"format": "pt"}); the engine does not read them,
            // but they are now visible via metadata() instead of dropped.
            size_t scan = 0;
            while (scan < obj.size()) {
                size_t k1 = obj.find('"', scan);
                if (k1 == std::string::npos) break;
                size_t k2 = obj.find('"', k1 + 1);
                if (k2 == std::string::npos) break;
                std::string mk = obj.substr(k1 + 1, k2 - k1 - 1);
                size_t colon = obj.find(':', k2 + 1);
                if (colon == std::string::npos) break;
                size_t v1 = obj.find('"', colon + 1);
                if (v1 == std::string::npos) break;
                size_t v2 = obj.find('"', v1 + 1);
                if (v2 == std::string::npos) break;
                GGUFValue mv;
                mv.type = GGUFValueType::STRING;
                mv.str = obj.substr(v1 + 1, v2 - v1 - 1);
                metadata_["safetensors.metadata." + mk] = std::move(mv);
                scan = v2 + 1;
            }
            continue;
        }

        TensorDescriptor td;
        td.name = key;
        td.is_sparse_expert = is_expert_tensor_name(td.name);

        // Extract dtype. Unknown dtypes keep the previous F32 default but
        // are recorded (and warned about once) instead of silently
        // misinterpreting the bytes.
        size_t dt_pos = obj.find("\"dtype\"");
        if (dt_pos != std::string::npos) {
            size_t q1 = obj.find('"', dt_pos + 7);
            size_t q2 = (q1 != std::string::npos) ? obj.find('"', q1 + 1) : std::string::npos;
            if (q1 != std::string::npos && q2 != std::string::npos) {
                std::string dt = obj.substr(q1 + 1, q2 - q1 - 1);
                static const std::pair<const char*, GGMLType> kDtypes[] = {
                    {"F32", GGMLType::F32},     {"F16", GGMLType::F16},
                    {"BF16", GGMLType::BF16},   {"F64", GGMLType::F64},
                    {"I8", GGMLType::I8},       {"I16", GGMLType::I16},
                    {"I32", GGMLType::I32},     {"I64", GGMLType::I64},
                    {"F8_E4M3", GGMLType::FP8_E4M3},
                    {"Q4_0", GGMLType::Q4_0},   {"Q8_0", GGMLType::Q8_0},
                    {"Q4_K", GGMLType::Q4_K},   {"Q5_K", GGMLType::Q5_K},
                    {"Q6_K", GGMLType::Q6_K},   {"Q8_K", GGMLType::Q8_K},
                    {"IQ4_NL", GGMLType::IQ4_NL},
                };
                bool known = false;
                for (const auto& [name, type] : kDtypes) {
                    if (dt == name) { td.type = type; known = true; break; }
                }
                if (!known &&
                    std::find(st_unknown_dtypes_.begin(), st_unknown_dtypes_.end(), dt) ==
                        st_unknown_dtypes_.end()) {
                    st_unknown_dtypes_.push_back(dt);
                    std::cerr << "[SAFETENSORS] WARNING: unknown dtype '" << dt
                              << "' for tensor '" << key << "' — treating as F32.\n";
                }
            }
        }

        // Extract shape. SafeTensors shapes are PyTorch order [out, in],
        // but this engine's GGUF convention is dims = [in, out]
        // (dims[0] = input cols, dims[1] = output rows — see bind_tensors(),
        // which reads q_dim from dims[1] and dim/vocab from the embed
        // tensor's dims[0]/dims[1], and model_forge.cpp, which writes
        // token_embd as {dim, vocab_size}). The ST bytes themselves are
        // already row-major [out][in], which is exactly the layout
        // matmul_quant_f32 expects (one output row per row of data), so
        // only the dims are reversed — never the data.
        size_t sh_pos = obj.find("\"shape\"");
        if (sh_pos != std::string::npos) {
            size_t b1 = obj.find('[', sh_pos);
            size_t b2 = (b1 != std::string::npos) ? obj.find(']', b1) : std::string::npos;
            if (b1 != std::string::npos && b2 != std::string::npos) {
                std::string arr = obj.substr(b1 + 1, b2 - b1 - 1);
                std::stringstream ss(arr);
                std::string item;
                uint32_t d = 0;
                while (std::getline(ss, item, ',') && d < 4) {
                    if (!item.empty()) {
                        td.dims[d++] = std::stoull(item);
                    }
                }
                td.n_dims = d;
                if (td.n_dims == 2) std::swap(td.dims[0], td.dims[1]);
            }
        }

        // Extract data_offsets
        size_t do_pos = obj.find("\"data_offsets\"");
        if (do_pos != std::string::npos) {
            size_t b1 = obj.find('[', do_pos);
            size_t b2 = (b1 != std::string::npos) ? obj.find(']', b1) : std::string::npos;
            if (b1 != std::string::npos && b2 != std::string::npos) {
                std::string arr = obj.substr(b1 + 1, b2 - b1 - 1);
                size_t comma = arr.find(',');
                if (comma != std::string::npos) {
                    uint64_t start_off = std::stoull(arr.substr(0, comma));
                    uint64_t end_off   = std::stoull(arr.substr(comma + 1));
                    td.offset = start_off;
                    td.abs_file_offset = data_offset_ + start_off;
                    td.byte_size = end_off - start_off;
                    if (td.abs_file_offset + td.byte_size <= file_size_) {
                        td.mapped_ptr = mapped_base_ + td.abs_file_offset;
                    }
                }
            }
        }

        // HF -> GGUF name translation. Routed MoE expert tensors and fused
        // projections have no translation: the engine expects packed 3D
        // expert tensors / separate Q,K,V, so faking an alias would load
        // garbage. They stay under their original names and are flagged.
        if (td.name.find(".experts.") != std::string::npos) {
            st_moe_unsupported_ = true;
            st_untranslated_.push_back(td.name);
        } else if (td.name.find("qkv_proj") != std::string::npos) {
            st_untranslated_.push_back(td.name);
        } else {
            std::string gguf_name = translate_hf_tensor_name(td.name);
            if (!gguf_name.empty() && gguf_name != td.name) {
                alias_[gguf_name] = td.name;
            }
        }

        tensor_order_.push_back(td.name);
        tensors_[td.name] = std::move(td);
    }

    return !tensors_.empty();
}

// Load a sibling config.json (and note a sibling tokenizer.json) for a
// SafeTensors model. Best-effort: returns false when no usable config was
// found, and open() treats that as "load without config", not an error.
bool ModelFile::load_safetensors_config() {
    namespace fs = std::filesystem;
    fs::path dir = fs::path(path_).parent_path();
    std::error_code ec;

    fs::path tok_path = dir / "tokenizer.json";
    if (fs::exists(tok_path, ec) && !ec) {
        tokenizer_path_ = tok_path.string();
        std::cout << "[SAFETENSORS] tokenizer.json found: " << tokenizer_path_
                  << " (parsing it belongs to the tokenizer layer).\n";
    }

    fs::path cfg_path = dir / "config.json";
    if (!fs::exists(cfg_path, ec) || ec) return false;
    std::ifstream in(cfg_path);
    if (!in) return false;
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (json.empty()) return false;

    std::string arch_class, model_type;
    cfg_get_first_array_string(json, "architectures", arch_class);
    cfg_get_string(json, "model_type", model_type);

    std::string arch = hf_arch_to_gguf(arch_class);
    if (arch.empty() && is_known_gguf_arch(model_type)) arch = model_type;
    if (arch.empty()) {
        std::cout << "[SAFETENSORS] WARNING: config.json architecture '" << arch_class
                  << "' (model_type '" << model_type << "') is not mapped — "
                  << "no hparams were loaded; the engine will use its defaults.\n";
        return false;
    }

    double hidden = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0;
    double ctx = 0, vocab = 0, eps = 0, rope_theta = 0;
    bool have_hidden = cfg_get_number(json, "hidden_size", hidden);
    bool have_layers = cfg_get_number(json, "num_hidden_layers", layers);
    if (!have_hidden || hidden <= 0 || !have_layers || layers <= 0) {
        std::cout << "[SAFETENSORS] WARNING: config.json for '" << arch
                  << "' has no usable hidden_size/num_hidden_layers — no hparams loaded.\n";
        return false;
    }

    auto set_u32 = [&](const std::string& k, uint32_t v) {
        GGUFValue val; val.type = GGUFValueType::UINT32; val.u64 = v; val.i64 = v;
        metadata_[k] = std::move(val);
    };
    auto set_f32 = [&](const std::string& k, float v) {
        GGUFValue val; val.type = GGUFValueType::FLOAT32; val.f64 = v;
        metadata_[k] = std::move(val);
    };
    auto set_str = [&](const std::string& k, const std::string& v) {
        GGUFValue val; val.type = GGUFValueType::STRING; val.str = v;
        metadata_[k] = std::move(val);
    };
    auto set_bool = [&](const std::string& k, bool v) {
        GGUFValue val; val.type = GGUFValueType::BOOL; val.bval = v; val.u64 = v ? 1 : 0;
        metadata_[k] = std::move(val);
    };

    set_str("general.architecture", arch);
    set_str("general.name", arch_class.empty() ? model_type : arch_class);
    set_u32(arch + ".embedding_length", static_cast<uint32_t>(hidden));
    if (cfg_get_number(json, "intermediate_size", intermediate) && intermediate > 0) {
        set_u32(arch + ".feed_forward_length", static_cast<uint32_t>(intermediate));
    }
    set_u32(arch + ".block_count", static_cast<uint32_t>(layers));
    if (cfg_get_number(json, "num_attention_heads", heads) && heads > 0) {
        set_u32(arch + ".attention.head_count", static_cast<uint32_t>(heads));
    }
    if (cfg_get_number(json, "num_key_value_heads", kv_heads) && kv_heads > 0) {
        set_u32(arch + ".attention.head_count_kv", static_cast<uint32_t>(kv_heads));
    } else if (heads > 0) {
        set_u32(arch + ".attention.head_count_kv", static_cast<uint32_t>(heads));
    }
    if (cfg_get_number(json, "max_position_embeddings", ctx) && ctx > 0) {
        set_u32(arch + ".context_length", static_cast<uint32_t>(ctx));
    }
    if (cfg_get_number(json, "vocab_size", vocab) && vocab > 0) {
        set_u32(arch + ".vocab_size", static_cast<uint32_t>(vocab));
    }
    if (cfg_get_number(json, "rms_norm_eps", eps) && eps > 0) {
        set_f32(arch + ".attention.layer_norm_rms_epsilon", static_cast<float>(eps));
    }
    if (cfg_get_number(json, "rope_theta", rope_theta) && rope_theta > 0) {
        set_f32(arch + ".rope.freq_base", static_cast<float>(rope_theta));
    }
    double v = 0;
    if (cfg_get_number(json, "head_dim", v) && v > 0) {
        set_u32(arch + ".attention.key_length", static_cast<uint32_t>(v));
    }
    if (cfg_get_number(json, "sliding_window", v) && v > 0) {
        set_u32(arch + ".attention.sliding_window", static_cast<uint32_t>(v));
    }
    if (cfg_get_number(json, "num_experts", v) && v > 0) {
        set_u32(arch + ".expert_count", static_cast<uint32_t>(v));
    }
    if (cfg_get_number(json, "num_experts_per_tok", v) && v > 0) {
        set_u32(arch + ".expert_used_count", static_cast<uint32_t>(v));
    }
    bool b = false;
    if (cfg_get_bool(json, "tie_word_embeddings", b)) {
        set_bool(arch + ".tie_word_embeddings", b);
    }
    std::string dtype;
    if (cfg_get_string(json, "torch_dtype", dtype)) {
        set_str("general.torch_dtype", dtype);
    }

    detected_arch_ = arch;
    config_path_ = cfg_path.string();
    config_loaded_ = true;
    std::cout << "[SAFETENSORS] config.json loaded: arch=" << arch
              << " layers=" << static_cast<uint32_t>(layers)
              << " hidden=" << static_cast<uint32_t>(hidden)
              << " heads=" << static_cast<uint32_t>(heads)
              << " kv_heads=" << static_cast<uint32_t>(kv_heads > 0 ? kv_heads : heads)
              << " (" << config_path_ << ")\n";
    return true;
}

const TensorDescriptor* ModelFile::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return &it->second;
    // SafeTensors: try the HF -> GGUF alias table (e.g. "blk.0.attn_q.weight"
    // resolves "model.layers.0.self_attn.q_proj.weight"). The alias map is
    // empty for GGUF files, so their lookup behaviour is unchanged.
    auto alias = alias_.find(name);
    if (alias != alias_.end()) {
        auto jt = tensors_.find(alias->second);
        if (jt != tensors_.end()) return &jt->second;
    }
    return nullptr;
}

uint32_t ModelFile::get_u32(const std::string& key, uint32_t def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? static_cast<uint32_t>(it->second.u64) : def;
}

float ModelFile::get_f32(const std::string& key, float def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? static_cast<float>(it->second.f64) : def;
}

std::string ModelFile::get_str(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? it->second.str : def;
}

} // namespace moe

