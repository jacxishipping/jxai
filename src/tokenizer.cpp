#include "tokenizer.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace moe {

static const uint16_t kByteToUnicode[256] = {
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 
    0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x010D, 0x010E, 0x010F, 
    0x0110, 0x0111, 0x0112, 0x0113, 0x0114, 0x0115, 0x0116, 0x0117, 
    0x0118, 0x0119, 0x011A, 0x011B, 0x011C, 0x011D, 0x011E, 0x011F, 
    0x0120, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027, 
    0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F, 
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 
    0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F, 
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047, 
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F, 
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057, 
    0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F, 
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 
    0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F, 
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077, 
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x0121, 
    0x0122, 0x0123, 0x0124, 0x0125, 0x0126, 0x0127, 0x0128, 0x0129, 
    0x012A, 0x012B, 0x012C, 0x012D, 0x012E, 0x012F, 0x0130, 0x0131, 
    0x0132, 0x0133, 0x0134, 0x0135, 0x0136, 0x0137, 0x0138, 0x0139, 
    0x013A, 0x013B, 0x013C, 0x013D, 0x013E, 0x013F, 0x0140, 0x0141, 
    0x0142, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7, 
    0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x0143, 0x00AE, 0x00AF, 
    0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7, 
    0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF, 
    0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7, 
    0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF, 
    0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7, 
    0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF, 
    0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7, 
    0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF, 
    0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7, 
    0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF
};

static int g_unicode_to_byte[0x200];
static bool g_table_init = false;

static void ensure_byte_unicode_tables() {
    if (g_table_init) return;
    for (int i = 0; i < 0x200; ++i) g_unicode_to_byte[i] = -1;
    for (int b = 0; b < 256; ++b) {
        uint16_t cp = kByteToUnicode[b];
        if (cp < 0x200) {
            g_unicode_to_byte[cp] = b;
        }
    }
    g_table_init = true;
}

// Maximum tokenizer vocabulary this engine will build: a vocab entry
// with an id at or above this is rejected (the load fails and the caller
// falls back to the synthetic vocab) instead of allocating a vector of
// that size. Real vocabularies top out at 256k (Gemma); 1M is generous.
constexpr uint32_t kMaxTokenizerVocab = 1u << 20;

std::string codepoint_to_utf8(uint32_t cp) {
    // Surrogates and values above U+10FFFF have no UTF-8 encoding: emit
    // U+FFFD instead of the invalid bytes the old code produced (a lone
    // surrogate used to encode as ED A0 xx — invalid UTF-8 that the BPE
    // decoder then split unpredictably).
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

uint32_t utf8_next_codepoint(const std::string& str, size_t& i) {
    if (i >= str.size()) return 0;
    unsigned char c = static_cast<unsigned char>(str[i]);
    if ((c & 0x80) == 0x00) {
        i += 1;
        return c;
    }
    // Continuation byte k (1-based) or -1 when missing/invalid.
    auto cont = [&](size_t k) -> int {
        if (i + k >= str.size()) return -1;
        unsigned char b = static_cast<unsigned char>(str[i + k]);
        return (b & 0xC0) == 0x80 ? (b & 0x3F) : -1;
    };
    // Any invalid sequence (bad continuation byte, overlong form,
    // surrogate, > U+10FFFF, or a truncated sequence at the end of the
    // string) yields U+FFFD and advances 1 byte. The old code accepted
    // all of these and returned a bogus codepoint — e.g. a truncated
    // 0xE2 at end-of-string came back as codepoint 226, and 0xC0 0x80
    // decoded as U+0000. Valid UTF-8 decodes exactly as before.
    if ((c & 0xE0) == 0xC0) {
        int c1 = cont(1);
        if (c1 < 0) { i += 1; return 0xFFFD; }
        uint32_t cp = ((c & 0x1F) << 6) | static_cast<uint32_t>(c1);
        if (cp < 0x80) { i += 1; return 0xFFFD; } // overlong
        i += 2;
        return cp;
    } else if ((c & 0xF0) == 0xE0) {
        int c1 = cont(1), c2 = cont(2);
        if (c1 < 0 || c2 < 0) { i += 1; return 0xFFFD; }
        uint32_t cp = ((c & 0x0F) << 12) | (static_cast<uint32_t>(c1) << 6) |
                      static_cast<uint32_t>(c2);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) { i += 1; return 0xFFFD; }
        i += 3;
        return cp;
    } else if ((c & 0xF8) == 0xF0) {
        int c1 = cont(1), c2 = cont(2), c3 = cont(3);
        if (c1 < 0 || c2 < 0 || c3 < 0) { i += 1; return 0xFFFD; }
        uint32_t cp = ((c & 0x07) << 18) | (static_cast<uint32_t>(c1) << 12) |
                      (static_cast<uint32_t>(c2) << 6) | static_cast<uint32_t>(c3);
        if (cp < 0x10000 || cp > 0x10FFFF) { i += 1; return 0xFFFD; }
        i += 4;
        return cp;
    }
    i += 1;
    return 0xFFFD;
}

// Validate a GGUF chat template's Jinja syntax. The renderer
// (transformer.cpp render_jinja_segment) only understands {% if/elif/else/
// endif %}, {% for/endfor %} and {% set %} — any other {% %} statement
// is skipped and left LITERAL in the prompt, an unbalanced {% for %}
// renders the rest of the template as literal text, and an unclosed
// {{ / {% / {# is silently truncated. So a malformed template produces
// wrong prompts instead of an error. This check runs at load; on failure
// the caller clears the template (per-arch fallback) and warns. Returns
// false + a reason on the first problem found.
bool validate_chat_template(const std::string& tpl, std::string& reason) {
    size_t pos = 0;
    int for_depth = 0, if_depth = 0;
    while (pos < tpl.size()) {
        size_t v = tpl.find("{{", pos);
        size_t s = tpl.find("{%", pos);
        size_t c = tpl.find("{#", pos);
        size_t tag = std::string::npos;
        int kind = 0; // 1 = {{ }}, 2 = {% %}, 3 = {# #}
        if (v != std::string::npos) { tag = v; kind = 1; }
        if (s != std::string::npos && s < tag) { tag = s; kind = 2; }
        if (c != std::string::npos && c < tag) { tag = c; kind = 3; }
        if (tag == std::string::npos) break;
        if (kind == 1) {
            size_t en = tpl.find("}}", tag + 2);
            if (en == std::string::npos) { reason = "unclosed '{{'"; return false; }
            pos = en + 2;
        } else if (kind == 3) {
            size_t en = tpl.find("#}", tag + 2);
            if (en == std::string::npos) { reason = "unclosed '{# comment'"; return false; }
            pos = en + 2;
        } else {
            size_t en = tpl.find("%}", tag + 2);
            if (en == std::string::npos) { reason = "unclosed '{%'"; return false; }
            std::string body = tpl.substr(tag + 2, en - (tag + 2));
            // First word, ignoring whitespace and the '-' markers.
            size_t b = body.find_first_not_of(" \t\r\n-");
            if (b == std::string::npos) { reason = "empty {% %} statement"; return false; }
            size_t e = body.find_first_of(" \t\r\n-", b);
            std::string kw = body.substr(b, e == std::string::npos ? e : e - b);
            if (kw == "for") ++for_depth;
            else if (kw == "endfor") {
                if (--for_depth < 0) { reason = "{% endfor %} without {% for %}"; return false; }
            } else if (kw == "if") ++if_depth;
            else if (kw == "endif") {
                if (--if_depth < 0) { reason = "{% endif %} without {% if %}"; return false; }
            } else if (kw == "elif" || kw == "else") {
                if (if_depth == 0) { reason = "{% " + kw + " %} outside {% if %}"; return false; }
            } else if (kw == "set") {
                // ok
            } else {
                reason = "unsupported statement '{% " + kw +
                         " %}' — the renderer would leave it literal in the prompt";
                return false;
            }
            pos = en + 2;
        }
    }
    if (for_depth > 0) { reason = "unclosed {% for %}"; return false; }
    if (if_depth > 0) { reason = "unclosed {% if %}"; return false; }
    return true;
}

bool Tokenizer::load_from_model(const ModelFile& mf) {
    ensure_byte_unicode_tables();
    vocab_.clear();
    scores_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    has_byte_fallback_ = false;
    is_byte_level_bpe_ = false;
    is_sentencepiece_ = false;
    chat_template_.clear();
    tokenizer_model_.clear();
    has_add_bos_token_ = false;
    add_bos_token_ = false;
    has_add_eos_token_ = false;
    add_eos_token_ = false;

    const auto& meta = mf.metadata();
    chat_template_ = mf.get_str("tokenizer.chat_template", "");
    if (!chat_template_.empty()) {
        std::string tpl_reason;
        if (!validate_chat_template(chat_template_, tpl_reason)) {
            std::cerr << "[TOKENIZER] WARNING: GGUF chat template is malformed (" << tpl_reason
                      << ") — ignoring it and using the per-architecture fallback template.\n";
            chat_template_.clear();
        }
    }
    // Apply a metadata BOS/EOS id only when it is in range; otherwise
    // keep the id found by scanning the vocabulary and warn. An
    // out-of-range id used to be stored and later emitted by encode(),
    // where forward_token either rejected the token or the embedding
    // lookup silently produced zeros.
    auto apply_meta_id = [&](const char* key, uint32_t& current, const char* what) {
        auto it = meta.find(key);
        if (it == meta.end()) return; // absent: keep the scanned id
        uint32_t v = static_cast<uint32_t>(it->second.u64);
        if (v < vocab_.size()) { current = v; return; }
        std::cerr << "[TOKENIZER] WARNING: metadata " << what << " (" << v
                  << ") is outside the vocabulary (" << vocab_.size()
                  << " tokens) — keeping the scanned id " << current << ".\n";
    };
    {
        auto it_bos = meta.find("tokenizer.ggml.add_bos_token");
        if (it_bos != meta.end()) { has_add_bos_token_ = true; add_bos_token_ = it_bos->second.bval; }
        auto it_eos = meta.find("tokenizer.ggml.add_eos_token");
        if (it_eos != meta.end()) { has_add_eos_token_ = true; add_eos_token_ = it_eos->second.bval; }
    }
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

    std::string tok_model = mf.get_str("tokenizer.ggml.model", "");
    tokenizer_model_ = tok_model;
    if (tok_model == "gpt2") {
        is_byte_level_bpe_ = true;
    } else if (tok_model == "llama" || tok_model == "gemma" || tok_model == "gemma4") {
        is_sentencepiece_ = true;
    } else {
        for (const auto& m : merge_ranks_) {
            if (m.first.find("\xC4\xA0") != std::string::npos) {
                is_byte_level_bpe_ = true;
                break;
            }
        }
    }

    // If the GGUF carries no vocab (e.g. SafeTensors models), try a sibling
    // tokenizer.json next to the model file before falling back to a
    // synthetic byte-level vocab (which can only produce toy output).
    if (vocab_.empty()) {
        std::string model_path = mf.path();
        size_t slash = model_path.find_last_of("/\\");
        std::string dir = (slash == std::string::npos) ? "" : model_path.substr(0, slash + 1);
        {
            std::string tj_path = dir + "tokenizer.json";
            std::ifstream probe(tj_path);
            if (probe.good()) {
                probe.close();
                if (load_tokenizer_json(tj_path) && !vocab_.empty()) {
                    apply_meta_id("tokenizer.ggml.bos_token_id", bos_id_, "bos_token_id");
                    apply_meta_id("tokenizer.ggml.eos_token_id", eos_id_, "eos_token_id");
                    if (bos_id_ >= vocab_.size()) bos_id_ = 0;
                    if (eos_id_ >= vocab_.size()) eos_id_ = 0;
                    return true;
                }
                // JSON load failed: fall through to the synthetic vocab below.
                vocab_.clear();
                token_to_id_.clear();
                special_tokens_.clear();
            }
        }
        uint32_t target_vocab = 512;
        const auto* emb = mf.find_tensor("token_embd.weight");
        // Bounded: a corrupt embedding dims[1] used to resize the vocab
        // to an arbitrary (possibly truncated, possibly zero) size — zero
        // made the vocab_[0] write below out of bounds.
        if (emb && emb->n_dims >= 2 && emb->dims[1] > 0 && emb->dims[1] <= kMaxTokenizerVocab) {
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

    rebuild_token_index();

    apply_meta_id("tokenizer.ggml.bos_token_id", bos_id_, "bos_token_id");
    apply_meta_id("tokenizer.ggml.eos_token_id", eos_id_, "eos_token_id");

    // Final safety net: the vocabulary must be non-empty and BOS/EOS
    // must be valid token ids even for a tiny vocabulary or absent
    // metadata (the member defaults 1/2 are out of range for a vocab
    // smaller than 3 entries).
    if (vocab_.empty()) {
        std::cerr << "[TOKENIZER] ERROR: no usable vocabulary after all fallbacks\n";
        return false;
    }
    if (bos_id_ >= vocab_.size()) bos_id_ = 0;
    if (eos_id_ >= vocab_.size()) eos_id_ = 0;

    return true;
}

void Tokenizer::rebuild_token_index() {
    token_to_id_.clear();
    special_tokens_.clear();
    has_byte_fallback_ = false;
    for (uint32_t i = 0; i < vocab_.size(); ++i) {
        const std::string& t = vocab_[i];
        token_to_id_[t] = i;

        if (t == "<s>" || t == "<|begin_of_text|>" || t == "<bos>") bos_id_ = i;
        else if (t == "</s>" || t == "<|endoftext|>" || t == "<|end_of_text|>" || t == "<eos>" || t == "<turn|>") eos_id_ = i;
        else if (t == "<|im_start|>") im_start_id_ = i;
        else if (t == "<|im_end|>") im_end_id_ = i;
        else if (t == "<think>" || t == "<|channel>") think_start_id_ = i;
        else if (t == "</think>" || t == "<channel|>") think_end_id_ = i;

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

        // Register special control tokens for exact prefix matching.
        // Angle-bracket tags (<s>, <|im_start|>, ...) and square-bracket
        // tags ([INST], [/INST], [TOOL_CALLS], [control_N], ...) are both
        // control tokens. The bracket form is required by Mistral: without
        // it [INST] / [/INST] are BPE-split into "[", "INST", "]", the model
        // never sees the turn boundary, and it echoes template tokens
        // instead of answering (verified 2026-11-17 on
        // Mistral-7B-Instruct-v0.3 Q2_K — both RoPE modes gave
        // "\n\n[INST] 4 [" before this fix).
        if ((t.size() >= 3 && t.front() == '<' && t.back() == '>' && t.substr(0, 3) != "<0x") ||
            (t.size() >= 3 && t.front() == '[' && t.back() == ']')) {
            special_tokens_.emplace_back(t, i);
        }
    }

    // Sort special tokens longest-first so greedy matching prefers longer tags
    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

// --- Minimal JSON helpers for tokenizer.json (HF format) ---
// Only what a tokenizer.json needs: string values with escapes, numbers,
// and the "vocab" / "merges" / "added_tokens" structures. No full JSON lib.
static void tj_skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) ++pos;
}

// Parse a JSON string starting at s[pos] == '"'. Advances pos past the closing quote.
static bool tj_parse_string(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"') return true;
        if (c == '\\' && pos < s.size()) {
            char e = s[pos++];
            switch (e) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case '/': out.push_back('/'); break;
                case '\\': out.push_back('\\'); break;
                case '"': out.push_back('"'); break;
                case 'u': {
                    if (pos + 4 > s.size()) return false;
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[pos++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else return false;
                    }
                    // JSON \u escapes come in surrogate pairs for astral
                    // characters: a lone surrogate has no UTF-8 encoding
                    // (the old code emitted the invalid ED A0 xx bytes).
                    // Combine a high surrogate with a following low one;
                    // anything else unpaired becomes U+FFFD.
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        bool paired = false;
                        if (pos + 6 <= s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                            unsigned lo = 0;
                            bool ok = true;
                            for (int k = 0; k < 4; ++k) {
                                char h = s[pos + 2 + k];
                                lo <<= 4;
                                if (h >= '0' && h <= '9') lo |= static_cast<unsigned>(h - '0');
                                else if (h >= 'a' && h <= 'f') lo |= static_cast<unsigned>(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') lo |= static_cast<unsigned>(h - 'A' + 10);
                                else { ok = false; break; }
                            }
                            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                                pos += 6;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                paired = true;
                            }
                        }
                        if (!paired) cp = 0xFFFD;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD; // lone low surrogate
                    }
                    out += codepoint_to_utf8(cp);
                    break;
                }
                default: out.push_back(e); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

// Find `"key"` at any depth and leave pos just past the ':' that follows it.
static bool tj_find_key(const std::string& s, const std::string& key, size_t& pos_out) {
    std::string pat = "\"" + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    size_t colon = s.find(':', p + pat.size());
    if (colon == std::string::npos) return false;
    pos_out = colon + 1;
    return true;
}

// Strict non-negative decimal parse: no sign, no junk, no overflow.
// std::stoul throws out_of_range on huge inputs and wraps a leading '-'.
static bool tj_parse_u32_strict(const std::string& s, uint32_t& out) {
    if (s.empty()) return false;
    uint64_t v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        v = v * 10 + static_cast<uint64_t>(ch - '0');
        if (v > UINT32_MAX) return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

bool Tokenizer::load_tokenizer_json(const std::string& json_path) {
    std::ifstream ifs(json_path, std::ios::binary);
    if (!ifs.good()) return false;
    std::string js((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (js.empty()) return false;

    // Byte-level BPE when the file uses a ByteLevel pre-tokenizer / decoder.
    if (js.find("ByteLevel") != std::string::npos) {
        is_byte_level_bpe_ = true;
        is_sentencepiece_ = false;
    } else if (js.find("SentencePiece") != std::string::npos || js.find("Unigram") != std::string::npos) {
        is_sentencepiece_ = true;
        is_byte_level_bpe_ = false;
    }
    if (tokenizer_model_.empty()) {
        // "model": {"type": "BPE" | "Unigram" | ...}
        size_t mp = 0;
        if (tj_find_key(js, "type", mp)) {
            tj_skip_ws(js, mp);
            std::string t;
            if (tj_parse_string(js, mp, t)) tokenizer_model_ = t;
        }
    }

    // --- "vocab": { "<token>": id, ... } ---
    size_t vp = 0;
    if (!tj_find_key(js, "vocab", vp)) return false;
    tj_skip_ws(js, vp);
    if (vp >= js.size() || js[vp] != '{') return false;
    ++vp;
    std::vector<std::pair<uint32_t, std::string>> by_id;
    while (true) {
        tj_skip_ws(js, vp);
        if (vp >= js.size()) return false;
        if (js[vp] == '}') { ++vp; break; }
        std::string tok;
        if (!tj_parse_string(js, vp, tok)) return false;
        tj_skip_ws(js, vp);
        if (vp >= js.size() || js[vp] != ':') return false;
        ++vp;
        tj_skip_ws(js, vp);
        size_t num_start = vp;
        // Token ids are non-negative JSON numbers: a '-' (or '+') here is
        // malformed, and std::stoul used to wrap a negative id into a
        // huge uint32 (then vocab_.assign(max_id + 1) tried to allocate
        // gigabytes). Strict-parse instead.
        while (vp < js.size() && js[vp] >= '0' && js[vp] <= '9') ++vp;
        if (num_start == vp) return false;
        uint32_t id = 0;
        if (!tj_parse_u32_strict(js.substr(num_start, vp - num_start), id)) return false;
        if (id >= kMaxTokenizerVocab) return false;
        by_id.emplace_back(id, tok);
        tj_skip_ws(js, vp);
        if (vp < js.size() && js[vp] == ',') { ++vp; continue; }
        if (vp < js.size() && js[vp] == '}') { ++vp; break; }
    }
    if (by_id.empty()) return false;
    uint32_t max_id = 0;
    for (const auto& p : by_id) max_id = std::max(max_id, p.first);
    vocab_.assign(max_id + 1, "<unk>");
    for (const auto& p : by_id) vocab_[p.first] = p.second;

    // --- "merges": ["a b", ...] ---
    size_t mg = 0;
    if (tj_find_key(js, "merges", mg)) {
        tj_skip_ws(js, mg);
        if (mg < js.size() && js[mg] == '[') {
            ++mg;
            merge_ranks_.clear();
            int32_t rank = 0;
            while (true) {
                tj_skip_ws(js, mg);
                if (mg >= js.size()) break;
                if (js[mg] == ']') { ++mg; break; }
                std::string m;
                if (!tj_parse_string(js, mg, m)) break;
                merge_ranks_[m] = rank++;
                tj_skip_ws(js, mg);
                if (mg < js.size() && js[mg] == ',') { ++mg; continue; }
                if (mg < js.size() && js[mg] == ']') { ++mg; break; }
            }
        }
    }

    // --- "added_tokens": [{"id": n, "content": "...", "special": true}, ...] ---
    size_t ap = 0;
    if (tj_find_key(js, "added_tokens", ap)) {
        tj_skip_ws(js, ap);
        if (ap < js.size() && js[ap] == '[') {
            ++ap;
            while (true) {
                tj_skip_ws(js, ap);
                if (ap >= js.size()) break;
                if (js[ap] == ']') { ++ap; break; }
                if (js[ap] != '{') break;
                // Scan one object for "id" and "content".
                size_t obj_end = js.find('}', ap);
                if (obj_end == std::string::npos) break;
                std::string obj = js.substr(ap, obj_end - ap + 1);
                size_t ip = 0;
                uint32_t id = 0;
                std::string content;
                bool have_id = false, have_content = false;
                if (tj_find_key(obj, "id", ip)) {
                    tj_skip_ws(obj, ip);
                    size_t ns = ip;
                    while (ip < obj.size() && obj[ip] >= '0' && obj[ip] <= '9') ++ip;
                    if (ip > ns && tj_parse_u32_strict(obj.substr(ns, ip - ns), id)) have_id = true;
                }
                size_t cp = 0;
                if (tj_find_key(obj, "content", cp)) {
                    tj_skip_ws(obj, cp);
                    if (tj_parse_string(obj, cp, content)) have_content = true;
                }
                if (have_id && have_content) {
                    // Bounded: an added-token id above the vocab cap is
                    // skipped instead of resizing the vocab to id + 1.
                    if (id < kMaxTokenizerVocab) {
                        if (id >= vocab_.size()) vocab_.resize(id + 1, "<unk>");
                        vocab_[id] = content;
                    }
                }
                ap = obj_end + 1;
                tj_skip_ws(js, ap);
                if (ap < js.size() && js[ap] == ',') { ++ap; continue; }
            }
        }
    }

    scores_.assign(vocab_.size(), 0.0f);
    rebuild_token_index();
    return !vocab_.empty();
}

int32_t Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return (it != token_to_id_.end()) ? static_cast<int32_t>(it->second) : -1;
}

static std::vector<std::string> split_gpt2_pretokens(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    const size_t n = text.size();

    auto is_ascii_letter = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
    };
    auto is_ascii_digit = [](unsigned char c) {
        return (c >= '0' && c <= '9');
    };
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };

    while (i < n) {
        size_t start = i;
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Attach a single leading space to a following word/number/punctuation token (GPT-2 / Qwen2 rule)
        if (c == ' ' && i + 1 < n && !is_ws(static_cast<unsigned char>(text[i + 1]))) {
            i++;
            c = static_cast<unsigned char>(text[i]);
        }

        if (is_ascii_letter(c)) {
            while (i < n && is_ascii_letter(static_cast<unsigned char>(text[i]))) i++;
        } else if (is_ascii_digit(c)) {
            while (i < n && is_ascii_digit(static_cast<unsigned char>(text[i]))) i++;
        } else if (is_ws(c)) {
            if (c == '\n' || c == '\r') {
                while (i < n && (text[i] == '\n' || text[i] == '\r')) i++;
            } else {
                while (i < n && (text[i] == ' ' || text[i] == '\t')) {
                    // Leave the last space for the next non-ws token if followed by non-ws
                    if (i > start && i + 1 < n && !is_ws(static_cast<unsigned char>(text[i + 1]))) {
                        break;
                    }
                    i++;
                }
            }
        } else {
            while (i < n) {
                unsigned char nc = static_cast<unsigned char>(text[i]);
                if (is_ascii_letter(nc) || is_ascii_digit(nc) || is_ws(nc)) break;
                i++;
            }
        }
        words.push_back(text.substr(start, i - start));
    }
    return words;
}

std::vector<uint32_t> Tokenizer::encode_segment_bpe(const std::string& segment) const {
    if (segment.empty()) return {};

    if (is_byte_level_bpe_ && segment.size() > 1) {
        auto chunks = split_gpt2_pretokens(segment);
        if (chunks.size() > 1) {
            std::vector<uint32_t> combined;
            combined.reserve(segment.size());
            for (const auto& ch : chunks) {
                auto sub = encode_segment_bpe(ch);
                combined.insert(combined.end(), sub.begin(), sub.end());
            }
            return combined;
        }
    }

    std::string text_for_bpe;
    if (is_byte_level_bpe_) {
        text_for_bpe.reserve(segment.size() * 2);
        for (unsigned char b : segment) {
            text_for_bpe += codepoint_to_utf8(kByteToUnicode[b]);
        }
    } else if (is_sentencepiece_) {
        text_for_bpe.reserve(segment.size() * 2);
        for (char ch : segment) {
            if (ch == ' ') {
                text_for_bpe += "\xE2\x96\x81";
            } else {
                text_for_bpe.push_back(ch);
            }
        }
    } else {
        text_for_bpe = segment;
    }

    // Initialize symbols as individual UTF-8 characters (or single bytes if not valid UTF-8)
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < text_for_bpe.size()) {
        size_t start = i;
        utf8_next_codepoint(text_for_bpe, i);
        if (i > start) {
            symbols.push_back(text_for_bpe.substr(start, i - start));
        } else {
            symbols.push_back(text_for_bpe.substr(start, 1));
            i++;
        }
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
            // If unknown multi-char symbol, attempt character-level fallback
            bool any_mapped = false;
            size_t cidx = 0;
            while (cidx < sym.size()) {
                size_t cstart = cidx;
                utf8_next_codepoint(sym, cidx);
                std::string ch = sym.substr(cstart, cidx - cstart);
                auto cit = token_to_id_.find(ch);
                if (cit != token_to_id_.end()) {
                    out.push_back(cit->second);
                    any_mapped = true;
                }
            }
            if (!any_mapped) {
                out.push_back(0); // <unk>
            }
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
        // Special tokens start with '<' (e.g. <s>, <|im_start|>) or '['
        // (e.g. [INST], [/INST]). The '[' case is required by Mistral:
        // without it [INST]/[/INST] are BPE-split and the turn boundary
        // is lost even though the tokens are registered as special.
        if (text[pos] == '<' || text[pos] == '[') {
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

    if (is_byte_level_bpe_) {
        std::string out;
        size_t idx = 0;
        while (idx < t.size()) {
            uint32_t cp = utf8_next_codepoint(t, idx);
            if (cp < 0x200 && g_unicode_to_byte[cp] >= 0) {
                out.push_back(static_cast<char>(g_unicode_to_byte[cp]));
            } else {
                out += codepoint_to_utf8(cp);
            }
        }
        return out;
    }

    // Replace SentencePiece " " (0xE2 0x96 0x81) with space ' '
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

