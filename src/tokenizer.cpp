#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <fstream>

namespace moe {

bool Tokenizer::load_from_model(const ModelFile& mf) {
    vocab_.clear();
    scores_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    has_byte_fallback_ = false;

    const auto& meta = mf.metadata();
    auto it_tok = meta.find("tokenizer.ggml.tokens");
    if (it_tok != meta.end() && !it_tok->second.str_arr.empty()) {
        vocab_ = it_tok->second.str_arr;
    }

    auto it_scores = meta.find("tokenizer.ggml.scores");
    if (it_scores != meta.end() && !it_scores->second.f32_arr.empty()) {
        scores_ = it_scores->second.f32_arr;
    }

    auto it_merges = meta.find("tokenizer.ggml.merges");
    if (it_merges != meta.end() && !it_merges->second.str_arr.empty()) {
        const auto& m = it_merges->second.str_arr;
        for (size_t i = 0; i < m.size(); ++i) {
            merge_ranks_[m[i]] = static_cast<int32_t>(i);
        }
    }

    // If SafeTensors or missing GGUF vocab, build default byte-level + common BPE vocab
    if (vocab_.empty()) {
        uint32_t target_vocab = 512;
        const auto* emb = mf.find_tensor("token_embd.weight");
        if (emb && emb->n_dims >= 2) {
            target_vocab = static_cast<uint32_t>(emb->dims[1]);
        }
        vocab_.resize(target_vocab);
        vocab_[0] = "<unk>";
        vocab_[1] = "<s>";
        vocab_[2] = "</s>";
        vocab_[3] = "<|im_start|>";
        vocab_[4] = "<|im_end|>";
        vocab_[5] = "<think>";
        vocab_[6] = "</think>";
        for (uint32_t b = 0; b < 256 && (7 + b) < target_vocab; ++b) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            vocab_[7 + b] = buf;
        }
        for (uint32_t i = 263; i < target_vocab; ++i) {
            vocab_[i] = "tok_" + std::to_string(i);
        }
    }

    if (scores_.size() < vocab_.size()) {
        scores_.resize(vocab_.size(), 0.0f);
        for (size_t i = 0; i < vocab_.size(); ++i) {
            scores_[i] = static_cast<float>(vocab_[i].size()) * 10.0f - static_cast<float>(i) * 0.001f;
        }
    }

    for (uint32_t i = 0; i < vocab_.size(); ++i) {
        const std::string& t = vocab_[i];
        token_to_id_[t] = i;

        if (t == "<s>" || t == "<|begin_of_text|>") bos_id_ = i;
        else if (t == "</s>" || t == "<|endoftext|>" || t == "<|end_of_text|>") eos_id_ = i;
        else if (t == "<|im_start|>") im_start_id_ = i;
        else if (t == "<|im_end|>") im_end_id_ = i;
        else if (t == "<think>") think_start_id_ = i;
        else if (t == "</think>") think_end_id_ = i;

        // Detect <0xXX> byte fallback tokens
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && (t[2] == 'x' || t[2] == 'X') && t[5] == '>') {
            unsigned int byte_val = 0;
            if (std::sscanf(t.c_str(), "<0x%02X>", &byte_val) == 1 ||
                std::sscanf(t.c_str(), "<0x%02x>", &byte_val) == 1) {
                if (byte_val < 256) {
                    byte_fallback_[byte_val] = i;
                    has_byte_fallback_ = true;
                }
            }
        }

        // Register special control tokens for exact prefix matching
        if (t.size() >= 3 && t.front() == '<' && t.back() == '>' && t.substr(0, 3) != "<0x") {
            special_tokens_.emplace_back(t, i);
        }
    }

    // Sort special tokens longest-first so greedy matching prefers longer tags
    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    bos_id_ = mf.get_u32("tokenizer.ggml.bos_token_id", bos_id_);
    eos_id_ = mf.get_u32("tokenizer.ggml.eos_token_id", eos_id_);

    return true;
}

bool Tokenizer::load_tokenizer_json(const std::string& json_path) {
    std::ifstream ifs(json_path);
    return ifs.good();
}

int32_t Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return (it != token_to_id_.end()) ? static_cast<int32_t>(it->second) : -1;
}

std::vector<uint32_t> Tokenizer::encode_segment_bpe(const std::string& segment) const {
    if (segment.empty()) return {};

    // Initialize symbols as individual UTF-8 characters (or single bytes if not valid UTF-8)
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < segment.size()) {
        unsigned char c = static_cast<unsigned char>(segment[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        if (i + char_len > segment.size()) char_len = 1;
        symbols.push_back(segment.substr(i, char_len));
        i += char_len;
    }

    // Iteratively merge adjacent symbol pairs using BPE merge ranks or vocabulary scores
    while (symbols.size() > 1) {
        int best_idx = -1;
        float best_score = -1e30f;
        int32_t best_rank = 0x7FFFFFFF;

        for (size_t j = 0; j + 1 < symbols.size(); ++j) {
            std::string merged = symbols[j] + symbols[j + 1];
            if (!merge_ranks_.empty()) {
                std::string pair_key = symbols[j] + " " + symbols[j + 1];
                auto rit = merge_ranks_.find(pair_key);
                if (rit != merge_ranks_.end() && rit->second < best_rank) {
                    best_rank = rit->second;
                    best_idx = static_cast<int>(j);
                }
            } else {
                auto vit = token_to_id_.find(merged);
                if (vit != token_to_id_.end()) {
                    float sc = scores_[vit->second];
                    if (sc > best_score) {
                        best_score = sc;
                        best_idx = static_cast<int>(j);
                    }
                }
            }
        }

        if (best_idx < 0) break;
        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // Map final symbols to token IDs, falling back to <0xXX> byte tokens when needed
    std::vector<uint32_t> out;
    out.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (has_byte_fallback_) {
            for (unsigned char b : sym) {
                out.push_back(byte_fallback_[b]);
            }
        } else {
            out.push_back(0); // <unk>
        }
    }
    return out;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<uint32_t> result;
    if (add_bos) result.push_back(bos_id_);
    if (text.empty()) return result;

    size_t pos = 0;
    size_t seg_start = 0;

    while (pos < text.size()) {
        int matched_special = -1;
        size_t matched_len = 0;
        if (text[pos] == '<') {
            for (const auto& sp : special_tokens_) {
                if (pos + sp.first.size() <= text.size() &&
                    text.compare(pos, sp.first.size(), sp.first) == 0) {
                    matched_special = static_cast<int>(sp.second);
                    matched_len = sp.first.size();
                    break;
                }
            }
        }

        if (matched_special >= 0) {
            if (pos > seg_start) {
                auto sub = encode_segment_bpe(text.substr(seg_start, pos - seg_start));
                result.insert(result.end(), sub.begin(), sub.end());
            }
            result.push_back(static_cast<uint32_t>(matched_special));
            pos += matched_len;
            seg_start = pos;
        } else {
            pos++;
        }
    }

    if (seg_start < text.size()) {
        auto sub = encode_segment_bpe(text.substr(seg_start));
        result.insert(result.end(), sub.begin(), sub.end());
    }

    return result;
}

std::string Tokenizer::decode_token_raw(uint32_t token_id) const {
    if (token_id >= vocab_.size()) return "";
    const std::string& t = vocab_[token_id];

    // Check <0xXX> byte fallback token
    if (t.size() == 6 && t[0] == '<' && t[1] == '0' && (t[2] == 'x' || t[2] == 'X') && t[5] == '>') {
        unsigned int byte_val = 0;
        if (std::sscanf(t.c_str(), "<0x%02X>", &byte_val) == 1 ||
            std::sscanf(t.c_str(), "<0x%02x>", &byte_val) == 1) {
            char c = static_cast<char>(byte_val & 0xFF);
            return std::string(1, c);
        }
    }

    // Replace SentencePiece "▁" (0xE2 0x96 0x81) with space ' '
    std::string out;
    out.reserve(t.size());
    for (size_t i = 0; i < t.size(); ) {
        if (i + 3 <= t.size() &&
            static_cast<unsigned char>(t[i])   == 0xE2 &&
            static_cast<unsigned char>(t[i+1]) == 0x96 &&
            static_cast<unsigned char>(t[i+2]) == 0x81) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(t[i]);
            i++;
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    StreamingUTF8Decoder dec(*this);
    std::string out;
    for (uint32_t id : ids) {
        out += dec.push(id);
    }
    out += dec.flush();
    return out;
}

std::string StreamingUTF8Decoder::push(uint32_t token_id) {
    pending_bytes_ += tok_.decode_token_raw(token_id);
    if (pending_bytes_.empty()) return "";

    // Find how many bytes from the start form complete UTF-8 sequences
    size_t valid_end = 0;
    size_t i = 0;
    const size_t n = pending_bytes_.size();

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(pending_bytes_[i]);
        size_t seq_len = 1;
        if ((c & 0x80) == 0x00) {
            seq_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            seq_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            seq_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            seq_len = 4;
        } else {
            // Invalid leading byte, emit as-is
            seq_len = 1;
        }

        if (i + seq_len > n) {
            // Incomplete multi-byte UTF-8 sequence at end of buffer — wait for next token!
            break;
        }
        i += seq_len;
        valid_end = i;
    }

    if (valid_end == 0) return "";
    std::string ready = pending_bytes_.substr(0, valid_end);
    pending_bytes_.erase(0, valid_end);
    return ready;
}

std::string StreamingUTF8Decoder::flush() {
    std::string rem = std::move(pending_bytes_);
    pending_bytes_.clear();
    return rem;
}

} // namespace moe

