#include "gguf.h"
#include <algorithm>

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
        return parse_safetensors(err_msg);
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

        if (key == "__metadata__") continue;

        TensorDescriptor td;
        td.name = key;
        td.is_sparse_expert = is_expert_tensor_name(td.name);

        // Extract dtype
        size_t dt_pos = obj.find("\"dtype\"");
        if (dt_pos != std::string::npos) {
            size_t q1 = obj.find('"', dt_pos + 7);
            size_t q2 = (q1 != std::string::npos) ? obj.find('"', q1 + 1) : std::string::npos;
            if (q1 != std::string::npos && q2 != std::string::npos) {
                std::string dt = obj.substr(q1 + 1, q2 - q1 - 1);
                if (dt == "F32") td.type = GGMLType::F32;
                else if (dt == "F16") td.type = GGMLType::F16;
                else if (dt == "BF16") td.type = GGMLType::BF16;
                else if (dt == "F8_E4M3") td.type = GGMLType::FP8_E4M3;
                else if (dt == "Q4_0") td.type = GGMLType::Q4_0;
                else if (dt == "Q8_0") td.type = GGMLType::Q8_0;
            }
        }

        // Extract shape
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

        tensor_order_.push_back(td.name);
        tensors_[td.name] = std::move(td);
    }

    return !tensors_.empty();
}

const TensorDescriptor* ModelFile::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
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

