#pragma once

#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace moe {

class Tokenizer {
public:
    Tokenizer() = default;

    // Initialize from loaded GGUF ModelFile metadata (or fallback byte-BPE table if SafeTensors)
    bool load_from_model(const ModelFile& mf);

    // Load additional rules from external tokenizer.json if present
    bool load_tokenizer_json(const std::string& json_path);

    // Encode UTF-8 text into token IDs
    std::vector<uint32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode a single token ID into raw bytes
    std::string decode_token_raw(uint32_t token_id) const;

    // Decode a sequence of token IDs into valid UTF-8 text
    std::string decode(const std::vector<uint32_t>& ids) const;

    uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }
    uint32_t bos_id() const { return bos_id_; }
    uint32_t eos_id() const { return eos_id_; }
    uint32_t im_start_id() const { return im_start_id_; }
    uint32_t im_end_id() const { return im_end_id_; }
    uint32_t think_start_id() const { return think_start_id_; }
    uint32_t think_end_id() const { return think_end_id_; }
    bool is_byte_level_bpe() const { return is_byte_level_bpe_; }
    // GGUF tokenizer metadata (empty when the model does not carry them)
    const std::string& chat_template() const { return chat_template_; }
    const std::string& tokenizer_model() const { return tokenizer_model_; }
    // tokenizer.ggml.add_bos_token / add_eos_token. has_* is false when the
    // GGUF does not carry the key, so callers can fall back to per-arch rules.
    bool has_add_bos_token() const { return has_add_bos_token_; }
    bool add_bos_token() const { return add_bos_token_; }
    bool has_add_eos_token() const { return has_add_eos_token_; }
    bool add_eos_token() const { return add_eos_token_; }

    const std::vector<std::string>& vocab() const { return vocab_; }
    const std::unordered_map<std::string, int32_t>& merges() const { return merge_ranks_; }
    int32_t token_to_id(const std::string& tok) const;

private:
    std::vector<uint32_t> encode_segment_bpe(const std::string& segment) const;
    // (Re)build token_to_id_, special_tokens_ and byte-fallback tables from
    // vocab_, and (re)detect bos/eos/im_* ids from token spellings. Called by
    // load_from_model() and load_tokenizer_json().
    void rebuild_token_index();

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<std::pair<std::string, uint32_t>> special_tokens_;
    uint32_t byte_fallback_[256] = {0};
    bool has_byte_fallback_ = false;
    bool is_byte_level_bpe_ = false;
    bool is_sentencepiece_ = false;

    uint32_t bos_id_ = 1;
    uint32_t eos_id_ = 2;
    uint32_t im_start_id_ = 3;
    uint32_t im_end_id_ = 4;
    uint32_t think_start_id_ = 5;
    uint32_t think_end_id_ = 6;

    std::string chat_template_;
    std::string tokenizer_model_;
    bool has_add_bos_token_ = false;
    bool add_bos_token_ = false;
    bool has_add_eos_token_ = false;
    bool add_eos_token_ = false;
};

// Streaming UTF-8 byte accumulator for token-by-token SSE emission
class StreamingUTF8Decoder {
public:
    explicit StreamingUTF8Decoder(const Tokenizer& tok) : tok_(tok) {}

    // Feed next token ID; returns complete UTF-8 codepoints ready to emit
    std::string push(uint32_t token_id);

    // Flush any remaining buffered bytes at end of stream
    std::string flush();

private:
    const Tokenizer& tok_;
    std::string pending_bytes_;
};

} // namespace moe

