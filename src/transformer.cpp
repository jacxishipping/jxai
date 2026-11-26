#include "transformer.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace moe {

// ---- Workstream C: validation + concurrency model -----------------------
// Concurrency model (documented in docs/SUPPORT_MATRIX.md):
//  * engine_mtx_ (a plain std::mutex, NOT recursive) serializes every
//    forward_token() — the only function that mutates slot KV state and
//    the expert tier manager. get_or_create_slot() must only be called
//    with engine_mtx_ held; every internal caller does.
//  * generate_stream() / verify_speculative_batch() hold the lock only
//    around slot-state accesses, so two generations on DIFFERENT slots
//    interleave safely (each forward is atomic), but two generations on
//    the SAME slot must not run concurrently — the caller (HTTP server)
//    serializes per slot. Slot state is never shared between slots.
//  * Slot ids are bounded (below): a client must not be able to force
//    unbounded KV allocation by naming arbitrary slot ids. Internal
//    scratch slots 9998 (embeddings) / 9999 (SystemOne) are exempt.
//    Defense in depth: server.cpp also rejects cache_slot outside [0, 63]
//    with HTTP 400 before generation starts.
constexpr uint32_t kMaxUserSlot = 63;
constexpr uint32_t kScratchSlotEmbeddings = 9998;
constexpr uint32_t kScratchSlotSystemOne = 9999;
constexpr uint32_t kScratchSlotDiffusion = 8888; // diffusion.cpp image path (internal; not user-reachable via server cache_slot)
constexpr uint32_t kMaxNewTokens = 16384;
constexpr uint32_t kMaxDraftTokens = 64;

bool is_valid_slot_id(uint32_t slot_id) {
    return slot_id <= kMaxUserSlot || slot_id == kScratchSlotEmbeddings ||
           slot_id == kScratchSlotSystemOne || slot_id == kScratchSlotDiffusion;
}

// Validate one bound tensor structurally: data present, dimensions sane,
// byte size exactly what its dtype+shape require. `expected_numel == 0`
// means "any element count, just check internal consistency"; otherwise
// the element count must match exactly (used for biases/norms whose
// length the forward pass trusts without re-checking). A tensor that
// fails this check must never reach a matmul: the matmul kernels do not
// null-check or bounds-check their weight pointers.
bool validate_bound_tensor(const TensorDescriptor* td, const char* name,
                           uint64_t expected_numel, std::string& err_msg) {
    if (!td) return true; // optional tensor: absent is fine
    if (!td->mapped_ptr) {
        err_msg = std::string("tensor '") + name + "' has no data pointer (truncated file?)";
        return false;
    }
    if (td->n_dims == 0 || td->n_dims > 4) {
        err_msg = std::string("tensor '") + name + "' has an invalid dimension count " +
                  std::to_string(td->n_dims);
        return false;
    }
    uint64_t numel = 1;
    for (uint32_t d = 0; d < td->n_dims; ++d) {
        if (td->dims[d] == 0) {
            err_msg = std::string("tensor '") + name + "' has a zero dimension";
            return false;
        }
        if (numel > UINT64_MAX / td->dims[d]) {
            err_msg = std::string("tensor '") + name + "' dimensions overflow";
            return false;
        }
        numel *= td->dims[d];
    }
    if (expected_numel != 0 && numel != expected_numel) {
        err_msg = std::string("tensor '") + name + "' has " + std::to_string(numel) +
                  " elements, expected " + std::to_string(expected_numel);
        return false;
    }
    uint64_t expected_bytes = 0;
    if (!checked_tensor_bytes(td->type, numel, expected_bytes)) {
        err_msg = std::string("tensor '") + name + "' byte size overflows for dtype " +
                  ggml_type_name(td->type);
        return false;
    }
    if (expected_bytes != td->byte_size) {
        err_msg = std::string("tensor '") + name + "' byte size mismatch: file says " +
                  std::to_string(td->byte_size) + " bytes but dtype " +
                  ggml_type_name(td->type) + " x " + std::to_string(numel) +
                  " elements needs " + std::to_string(expected_bytes);
        return false;
    }
    return true;
}

// Validate sampler parameters. Returns false + a human-readable reason on
// the first invalid value. This is the authoritative check, called by the
// public entry points (generate_stream, verify_speculative_batch); the
// HTTP server MUST also validate (server.cpp is workstream B's file) so a
// bad request is rejected with a 400 before it reaches the engine.
bool validate_sampler_config(const SamplerConfig& cfg, std::string& err_msg) {
    if (std::isnan(cfg.temperature) || std::isinf(cfg.temperature) || cfg.temperature < 0.0f) {
        err_msg = "temperature must be a finite value >= 0 (0 = greedy)";
        return false;
    }
    if (std::isnan(cfg.top_p) || cfg.top_p < 0.0f || cfg.top_p > 1.0f) {
        err_msg = "top_p must be in [0, 1]";
        return false;
    }
    if (std::isnan(cfg.repetition_penalty) || std::isinf(cfg.repetition_penalty) ||
        cfg.repetition_penalty <= 0.0f) {
        err_msg = "repetition_penalty must be finite and > 0 (1.0 = disabled)";
        return false;
    }
    return true;
}

void KVCacheSlot::init(uint32_t id, const ModelHyperparams& hp) {
    slot_id = id;
    seq_pos = 0;
    compactions = 0;
    token_history.clear();
    size_t kv_dim = static_cast<size_t>(hp.kv_dim);
    uint32_t alloc_seq = std::max(1u, hp.kv_cache_capacity);
    // Checked allocation: kv_dim/capacity come from model metadata via
    // bind_tensors() (validated + RAM-guarded there), but init() must not
    // trust them blindly — a wrapped size_t here used to allocate a
    // too-small cache that forward_token would then write past.
    if (kv_dim == 0 || hp.n_layers == 0 ||
        alloc_seq > (SIZE_MAX / 2) / std::max<size_t>(kv_dim, 1)) {
        throw std::runtime_error("KVCacheSlot::init: unreasonable KV dimensions (kv_dim=" +
                                 std::to_string(kv_dim) + ", capacity=" +
                                 std::to_string(alloc_seq) + ", layers=" +
                                 std::to_string(hp.n_layers) + ")");
    }
    size_t cap = static_cast<size_t>(alloc_seq) * kv_dim;
    k_cache.assign(hp.n_layers, std::vector<uint16_t>(cap, 0));
    v_cache.assign(hp.n_layers, std::vector<uint16_t>(cap, 0));
}

void KVCacheSlot::clear() {
    seq_pos = 0;
    token_history.clear();
    for (auto& k : k_cache) std::fill(k.begin(), k.end(), static_cast<uint16_t>(0));
    for (auto& v : v_cache) std::fill(v.begin(), v.end(), static_cast<uint16_t>(0));
}

void KVCacheSlot::rollback(uint32_t new_seq_pos) {
    if (new_seq_pos < seq_pos) {
        seq_pos = new_seq_pos;
    }
    if (token_history.size() > new_seq_pos) {
        token_history.resize(new_seq_pos);
    }
}

void KVCacheSlot::compact(const ModelHyperparams& hp, uint32_t keep_recent) {
    const uint32_t sink_tokens = 4;
    // Clamp keep_recent to what actually exists: a caller passing
    // max_seq_len/2 while the cache capacity is much smaller (RAM-guarded
    // models) used to make `seq_pos - keep_recent` underflow to a huge
    // uint32, and the memmove below then read/wrote far out of bounds.
    // (generate_stream used to pass hp_.max_seq_len / 2; it now passes
    // kv_cache_capacity / 2, and this clamp is the backstop.)
    if (seq_pos > sink_tokens && keep_recent > seq_pos - sink_tokens) {
        keep_recent = seq_pos - sink_tokens;
    }
    if (seq_pos <= sink_tokens + keep_recent) return;
    if (k_cache.size() < hp.n_layers || v_cache.size() < hp.n_layers) return;

    uint32_t new_len = sink_tokens + keep_recent;
    uint32_t src_start = seq_pos - keep_recent;
    size_t kv_dim = static_cast<size_t>(hp.kv_dim);

    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        std::memmove(
            k_cache[l].data() + sink_tokens * kv_dim,
            k_cache[l].data() + src_start * kv_dim,
            keep_recent * kv_dim * sizeof(uint16_t)
        );
        std::memmove(
            v_cache[l].data() + sink_tokens * kv_dim,
            v_cache[l].data() + src_start * kv_dim,
            keep_recent * kv_dim * sizeof(uint16_t)
        );
    }
    seq_pos = new_len;
    // Keep token_history consistent with the KV window: the old code kept
    // the FIRST new_len tokens (resize), while the KV cache kept the sink
    // + the most RECENT tokens — so history and cache disagreed after a
    // compaction (repetition penalty scored dropped tokens; the LCP
    // prefix-reuse check could match tokens no longer cached).
    if (token_history.size() > new_len) {
        size_t hist_src = token_history.size() - keep_recent;
        if (hist_src < sink_tokens) hist_src = sink_tokens;
        token_history.erase(token_history.begin() + sink_tokens,
                            token_history.begin() + hist_src);
    }
    compactions++;
}

void MoETransformerEngine::rms_norm(float* out, const float* x, const float* weight, int64_t size, float eps) {
    float ss = vec_dot_f32(x, x, size);
    float inv_rms = 1.0f / std::sqrt((ss / static_cast<float>(size)) + eps);
    for (int64_t i = 0; i < size; ++i) {
        out[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
    }
}

void MoETransformerEngine::apply_rope(
    float* q, float* k,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t pos, float freq_base, float yarn_scale, bool is_neox,
    const float* rope_freqs
) {
    uint32_t rdim = std::min(head_dim, rope_dim);
    float pos_scaled = static_cast<float>(pos) / std::max(1e-4f, yarn_scale);

    if (is_neox) {
        uint32_t half = rdim / 2;
        for (uint32_t d = 0; d < half; ++d) {
            float freq = 1.0f / std::pow(freq_base, (2.0f * static_cast<float>(d)) / static_cast<float>(rdim));
            if (rope_freqs) freq /= rope_freqs[d];
            float angle = pos_scaled * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (uint32_t h = 0; h < n_heads; ++h) {
                float* qh = q + h * head_dim;
                float q0 = qh[d];
                float q1 = qh[d + half];
                qh[d]        = q0 * cos_a - q1 * sin_a;
                qh[d + half] = q0 * sin_a + q1 * cos_a;
            }

            if (k) {
                for (uint32_t h = 0; h < n_kv_heads; ++h) {
                    float* kh = k + h * head_dim;
                    float k0 = kh[d];
                    float k1 = kh[d + half];
                    kh[d]        = k0 * cos_a - k1 * sin_a;
                    kh[d + half] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    } else {
        for (uint32_t d = 0; d + 1 < rdim; d += 2) {
            float freq = 1.0f / std::pow(freq_base, static_cast<float>(d) / static_cast<float>(rdim));
            float angle = pos_scaled * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (uint32_t h = 0; h < n_heads; ++h) {
                float* qh = q + h * head_dim + d;
                float q0 = qh[0];
                float q1 = qh[1];
                qh[0] = q0 * cos_a - q1 * sin_a;
                qh[1] = q0 * sin_a + q1 * cos_a;
            }

            if (k) {
                for (uint32_t h = 0; h < n_kv_heads; ++h) {
                    float* kh = k + h * head_dim + d;
                    float k0 = kh[0];
                    float k1 = kh[1];
                    kh[0] = k0 * cos_a - k1 * sin_a;
                    kh[1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }
}

bool arch_uses_neox_rope(const std::string& arch) {
    // Empirical per-arch RoPE convention for THIS engine — verified by
    // direct A/B on real models (2026-11-17), do NOT "fix" this to match
    // llama.cpp's Neox-everywhere assumption:
    //   Qwen2.5-0.5B-Instruct Q4_0 + Neox     -> "The answer to 2+2 is 4."
    //   Qwen2.5-0.5B-Instruct Q4_0 + interleaved -> garbage ("I'm not sure
    //     what you're 100% of the 19756348593478659")
    //   Llama-3.2-1B-Instruct Q4_K_M + interleaved -> "The answer to 2+2
    //     is 4. Would you like to know more..."
    //   Llama-3.2-1B-Instruct Q4_K_M + Neox -> garbage ("`` `` and and the
    //     ` \"System (in to to (")
    //   Qwen3-0.6B-Instruct Q4_K_M + interleaved -> garbage ("<|im_start|>
    //     [Reasoning: ] anic / anism" — fragmented tokens, no English)
    //   Qwen3-0.6B-Instruct Q4_K_M + Neox -> coherent thinking + "the
    //     answer is simply 4." / "2 + 2 equals 4" (254 tokens, temp 0)
    //   (verified 2026-11-17; Qwen3 thinks before answering, so it needs
    //   >128 max-tokens to reach the final answer.)
    //   Phi-3-mini-4k-instruct Q2_K + interleaved (fused gate_up SwiGLU +
    //     fixed Q2_K/Q3_K structs) -> coherent English but factually
    //     confused: 'The expression "2 + 2 equals 4. The answer is 5.
    //     In Python code snippet of code in python...'
    //   Phi-3-mini-4k-instruct Q2_K + Neox (same) -> clean: 'The answer
    //     to the question "What is 2+2?" is straightforward arithmetic.
    //     The calculation of 2 + 2 equals...' (32 tokens, temp 0)
    //   NOTE: the Phi-3 RoPE A/B was confounded until the Q2_K/Q3_K
    //   block-struct fix (include/quant.h) and the fused gate_up SwiGLU
    //   fix landed — before those, both modes gave <unk> (broken FFN)
    //   and then the same "age/measurement" word salad. The Neox verdict
    //   above holds only with those two fixes in place.
    // The effective convention depends on how each family's GGUF weights
    // are laid out (llama.cpp's converter permutes Q/K weights for some
    // families), so it is arch-specific here. Any arch not in the Neox
    // list below is UNVERIFIED in this engine — test it against a real
    // model before moving it, and never default the whole table to Neox.
    return (arch == "qwen2" || arch == "qwen2moe" || arch == "olmoe" ||
            arch == "phi" || arch == "phi3" || arch == "deepseek" || arch == "deepseek2" ||
            arch == "glm" || arch == "gptneox" ||
            arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4" ||
            arch == "qwen3"); // qwen3moe NOT verified: same family, different arch — test before moving
}

bool arch_is_known(const std::string& arch) {
    static const char* kKnown[] = {
        "llama", "llama3", "llama4", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
        "mistral", "mixtral", "gemma", "gemma2", "gemma3", "gemma4",
        "phi", "phi3", "phi4", "deepseek", "deepseek2", "olmoe", "glm",
        "gptneox", "gpt2", "gptj", "smollm", "stablelm", "starcoder2",
        "exaone", "command-r", "falcon", "mamba", "bert", "nomic-bert"
    };
    for (const char* k : kKnown) if (arch == k) return true;
    return false;
}

bool should_add_bos(const std::string& arch, const Tokenizer& tok) {
    if (tok.has_add_bos_token()) return tok.add_bos_token();
    // GGUF carries no add_bos_token key: per-architecture defaults.
    if (arch == "qwen2" || arch == "qwen2moe" || arch == "qwen3" || arch == "qwen3moe" ||
        arch == "deepseek" || arch == "deepseek2" || arch == "olmoe" ||
        arch == "gpt2" || arch == "gptj" || arch == "gpt-j" || arch == "gptneox" || arch == "glm") {
        return false;
    }
    return true; // Llama, Gemma, Mistral, Phi, SmolLM, StableLM, unknown: BOS
}

namespace {

// --- Minimal Jinja subset for GGUF tokenizer.chat_template ---
static std::string tj_trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Unquote a Jinja string literal ('...' or "..."), honouring \n \t \\ \' \".
static std::string jinja_unquote(const std::string& s) {
    std::string t = tj_trim(s);
    if (t.size() >= 2 && ((t.front() == '\'' && t.back() == '\'') ||
                          (t.front() == '"' && t.back() == '"'))) {
        std::string out;
        for (size_t i = 1; i + 1 < t.size(); ++i) {
            char c = t[i];
            if (c == '\\' && i + 2 < t.size()) {
                char n = t[++i];
                if (n == 'n') out.push_back('\n');
                else if (n == 't') out.push_back('\t');
                else if (n == 'r') out.push_back('\r');
                else out.push_back(n);
            } else {
                out.push_back(c);
            }
        }
        return out;
    }
    return t;
}

static std::string jinja_token_text(const Tokenizer& tok, uint32_t id) {
    if (id < tok.vocab().size()) return tok.vocab()[id];
    return "";
}

// Split an expression on top-level '+' (outside string literals).
static std::vector<std::string> jinja_split_plus(const std::string& expr) {
    std::vector<std::string> parts;
    std::string cur;
    char quote = 0;
    for (size_t i = 0; i < expr.size(); ++i) {
        char c = expr[i];
        if (quote) {
            cur.push_back(c);
            if (c == '\\' && i + 1 < expr.size()) cur.push_back(expr[++i]);
            else if (c == quote) quote = 0;
        } else if (c == '\'' || c == '"') {
            quote = c;
            cur.push_back(c);
        } else if (c == '+') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    return parts;
}

static std::string jinja_field(const ChatMessage* msg, const std::string& term) {
    if (!msg) return "";
    if (term == "message['role']" || term == "message[\"role\"]" || term == "message.role" || term == "role") return msg->role;
    if (term == "message['content']" || term == "message[\"content\"]" || term == "message.content" || term == "content") return msg->content;
    return "";
}

// --- {% set %} variable environment -------------------------------------
// (forward declarations: these helpers live below the expression
// evaluators in this file.)
static std::string jinja_rstrip_ws(const std::string& s);
static std::string jinja_lstrip_ws(const std::string& s);
static int eval_jinja_index_expr(const std::string& raw, int loop_idx);
struct JinjaEnv;
static bool eval_jinja_cond(const std::string& cond, const ChatMessage* msg, bool add_gen,
                            const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                            JinjaEnv& env);

// Real chat templates rely on {% set %} (Qwen3, Mistral v0.3, DeepSeek):
//   {% set content = message.content %}              -> scalar string var
//   {% set system_message = messages[0]["content"] %} -> scalar string var
//   {% set loop_messages = messages[1:] %}           -> message-list alias
//   {% set ns = namespace(a=true, b=...) %}          -> namespace object;
//       {% set ns.attr = ... %} mutates it, and namespace attributes are
//       the ONLY set state that persists across for-loop iterations (plain
//       set vars are scoped to the current iteration, as in Jinja2).
//   {% set tools = none %}                           -> the value `none`
// Without this, Mistral rendered only bos_token (its message loop iterates
// a set alias) and lost its system prompt, and Qwen3's content variable
// only worked by accident. Unsupported set expressions leave the previous
// value untouched instead of clobbering it with "".
struct JinjaEnv {
    std::map<std::string, std::string> vars;    // scalar string vars
    std::map<std::string, bool> is_none;        // var explicitly set to `none`
    std::map<std::string, std::vector<ChatMessage>> lists; // message-list vars
    // Namespace attributes, keyed "ns.attr": shared across for-loop
    // iterations by design. Copying a JinjaEnv copies the shared_ptr, so
    // loop bodies get private vars/lists but the same namespace map.
    std::shared_ptr<std::map<std::string, std::string>> ns =
        std::make_shared<std::map<std::string, std::string>>();
    // Tool definitions bound to the template's `tools` variable (workstream
    // G, 2026-11-17). `tools_present` is true only when the caller passed a
    // tool list at all (even an empty one) — it drives `tools is defined`.
    // `cur_tool_idx` is the index of the tool bound to the loop variable
    // inside {% for tool in tools %} (-1 outside that loop); an index (not
    // a pointer) so per-iteration env copies stay valid.
    std::vector<ToolDefinition> tools;
    bool tools_present = false;
    int cur_tool_idx = -1;
};

// Find the ')' matching the '(' at `open`, skipping string literals.
static size_t jinja_matching_paren(const std::string& s, size_t open) {
    int depth = 0;
    char quote = 0;
    for (size_t i = open; i < s.size(); ++i) {
        char c = s[i];
        if (quote) {
            if (c == '\\' && i + 1 < s.size()) ++i;
            else if (c == quote) quote = 0;
        } else if (c == '\'' || c == '"') {
            quote = c;
        } else if (c == '(') {
            ++depth;
        } else if (c == ')') {
            if (--depth == 0) return i;
        }
    }
    return std::string::npos;
}

// Strip ONE pair of parentheses that encloses the whole string, e.g.
// "(a) != (b)" is left alone (the parens do not enclose the whole atom)
// while "((a == b))" becomes "a == b".
static std::string jinja_strip_outer_parens(const std::string& raw) {
    std::string s = tj_trim(raw);
    while (s.size() >= 2 && s.front() == '(' &&
           jinja_matching_paren(s, 0) == s.size() - 1) {
        s = tj_trim(s.substr(1, s.size() - 2));
    }
    return s;
}

// Remove Jinja comments {# ... #} (with { #- / -#} whitespace control).
// Without this, comment text leaks verbatim into the rendered prompt
// (Mistral v0.3's template carries a long alternation-check comment).
static std::string jinja_strip_comments(const std::string& s) {
    std::string out;
    size_t pos = 0;
    while (true) {
        size_t st = s.find("{#", pos);
        if (st == std::string::npos) { out += s.substr(pos); break; }
        size_t en = s.find("#}", st + 2);
        if (en == std::string::npos) { out += s.substr(pos); break; }
        std::string before = s.substr(pos, st - pos);
        bool trim_before = (st + 2 < s.size() && s[st + 2] == '-');
        bool trim_after = (en > st + 2 && s[en - 1] == '-');
        if (trim_before) before = jinja_rstrip_ws(before);
        out += before;
        pos = en + 2;
        if (trim_after) {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                                      s[pos] == '\r' || s[pos] == '\n')) ++pos;
        }
    }
    return out;
}

// Resolve a bare name: loop counters, scalar vars, namespace attributes,
// then message fields. `found` is false for unknown names.
static bool jinja_lookup_value(const std::string& name, const ChatMessage* msg,
                               int loop_idx, JinjaEnv& env, std::string& out) {
    if (name == "loop.index") { out = std::to_string(loop_idx + 1); return true; }
    if (name == "loop.index0") { out = std::to_string(loop_idx); return true; }
    // `tools` bare name: "true" when at least one tool is bound, "false"
    // otherwise — this is what {% if tools %} evaluates through the
    // bare-variable truthiness path in eval_jinja_atom.
    if (name == "tools") {
        out = (!env.tools.empty() && !env.is_none.count("tools")) ? "true" : "false";
        return true;
    }
    // The loop variable of {% for tool in tools %}: tool / tool.type /
    // tool.function(.name/.description/.parameters). Resolved from the
    // index so per-iteration env copies stay valid.
    if (name == "tool" || name.rfind("tool.", 0) == 0) {
        if (env.cur_tool_idx < 0 ||
            env.cur_tool_idx >= static_cast<int>(env.tools.size())) return false;
        const ToolDefinition& t = env.tools[static_cast<size_t>(env.cur_tool_idx)];
        if (name == "tool") { out = t.raw_json; return true; }
        if (name == "tool.type") { out = "function"; return true; }
        if (name == "tool.function") { out = t.function_json; return true; }
        if (name == "tool.function.name") { out = t.name; return true; }
        if (name == "tool.function.description") { out = t.description; return true; }
        if (name == "tool.function.parameters") { out = t.parameters_json; return true; }
        return false;
    }
    auto vi = env.vars.find(name);
    if (vi != env.vars.end()) { out = vi->second; return true; }
    auto ni = env.ns->find(name);
    if (ni != env.ns->end()) { out = ni->second; return true; }
    std::string f = jinja_field(msg, name);
    if (!f.empty()) { out = f; return true; }
    // jinja_field returns "" both for "empty field" and "unknown name";
    // treat the known message fields as found even when empty.
    if (name == "message['role']" || name == "message[\"role\"]" ||
        name == "message.role" || name == "role" ||
        name == "message['content']" || name == "message[\"content\"]" ||
        name == "message.content" || name == "content") {
        out = "";
        return true;
    }
    return false;
}

// Integer expression evaluator: literals, loop.index/loop.index0,
// messages|length, variables / namespace attrs holding integers, and
// + - * % with parentheses. Returns false for non-integer expressions.
struct JinjaIntParser {
    const std::string& s;
    size_t pos = 0;
    const ChatMessage* msg;
    const std::vector<ChatMessage>& msgs;
    int loop_idx;
    JinjaEnv& env;
    bool ok = true;

    void skip_ws() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
    }
    long long parse_expr() {
        long long v = parse_term();
        while (true) {
            skip_ws();
            if (pos < s.size() && s[pos] == '+') { ++pos; v += parse_term(); }
            else if (pos < s.size() && s[pos] == '-') { ++pos; v -= parse_term(); }
            else break;
        }
        return v;
    }
    long long parse_term() {
        long long v = parse_factor();
        while (true) {
            skip_ws();
            if (pos < s.size() && s[pos] == '*') { ++pos; v *= parse_factor(); }
            else if (pos < s.size() && s[pos] == '%') {
                ++pos;
                long long d = parse_factor();
                if (d == 0) { ok = false; return 0; }
                v %= d;
            } else break;
        }
        return v;
    }
    long long parse_factor() {
        skip_ws();
        if (pos >= s.size()) { ok = false; return 0; }
        if (s[pos] == '(') {
            ++pos;
            long long v = parse_expr();
            skip_ws();
            if (pos < s.size() && s[pos] == ')') ++pos; else ok = false;
            return v;
        }
        if (s[pos] == '-') { ++pos; return -parse_factor(); }
        if (s[pos] >= '0' && s[pos] <= '9') {
            size_t j = pos;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
            long long v = 0;
            try { v = std::stoll(s.substr(pos, j - pos)); } catch (...) { ok = false; }
            pos = j;
            return v;
        }
        if (std::isalpha(static_cast<unsigned char>(s[pos])) || s[pos] == '_') {
            size_t j = pos;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) ||
                                    s[j] == '_' || s[j] == '.')) ++j;
            std::string name = s.substr(pos, j - pos);
            pos = j;
            skip_ws();
            // `X|length`: length of the looked-up string value.
            if (pos < s.size() && s[pos] == '|') {
                ++pos;
                skip_ws();
                if (s.compare(pos, 6, "length") == 0) {
                    pos += 6;
                    std::string val;
                    if (name == "messages") return static_cast<long long>(msgs.size());
                    if (name == "tools") return static_cast<long long>(env.tools.size());
                    if (!jinja_lookup_value(name, msg, loop_idx, env, val)) { ok = false; return 0; }
                    return static_cast<long long>(val.size());
                }
                ok = false;
                return 0;
            }
            if (name == "messages") { ok = false; return 0; } // bare list, not an int
            std::string val;
            if (!jinja_lookup_value(name, msg, loop_idx, env, val)) { ok = false; return 0; }
            try {
                size_t used = 0;
                long long v = std::stoll(val, &used);
                if (used != val.size()) { ok = false; return 0; }
                return v;
            } catch (...) { ok = false; return 0; }
        }
        ok = false;
        return 0;
    }
    bool at_end() { skip_ws(); return pos >= s.size(); }
};

static bool jinja_eval_int(const std::string& raw, const ChatMessage* msg,
                           const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                           JinjaEnv& env, long long& out) {
    (void)loop_total;
    std::string s = tj_trim(raw);
    if (s.empty()) return false;
    JinjaIntParser p{s, 0, msg, msgs, loop_idx, env, true};
    long long v = p.parse_expr();
    if (!p.ok || !p.at_end()) return false;
    out = v;
    return true;
}

// Parse a slice spec "[a:b]", "[a:]", "[:b]", "[::step]", "[i]" after a
// list name. `single` is set for "[i]" (one element). Returns false when
// the bracket content is not a slice/index (e.g. ["content"]).
static bool jinja_parse_slice(const std::string& bracket, long long n,
                              long long& start, long long& stop, long long& step,
                              bool& single) {
    std::string b = tj_trim(bracket);
    if (b.empty() || b.front() != '[' || b.back() != ']') return false;
    std::string inner = tj_trim(b.substr(1, b.size() - 2));
    if (inner.find('\'') != std::string::npos || inner.find('"') != std::string::npos) return false;
    if (inner.find(':') == std::string::npos) {
        // [i]: single index.
        try {
            long long i = std::stoll(inner);
            if (i < 0) i += n;
            if (i < 0 || i >= n) return false;
            start = i; stop = i + 1; step = 1; single = true;
            return true;
        } catch (...) { return false; }
    }
    single = false;
    auto parse_part = [](const std::string& t, long long def, long long n, bool is_stop) -> long long {
        std::string tt = tj_trim(t);
        if (tt.empty()) return def;
        try {
            long long v = std::stoll(tt);
            if (v < 0) v += n;
            if (is_stop) return std::max(0LL, std::min(n, v));
            return std::max(0LL, std::min(n - 1, v));
        } catch (...) { return def; }
    };
    std::vector<std::string> parts;
    size_t p0 = 0;
    while (true) {
        size_t c = inner.find(':', p0);
        parts.push_back(inner.substr(p0, c == std::string::npos ? std::string::npos : c - p0));
        if (c == std::string::npos) break;
        p0 = c + 1;
    }
    if (parts.size() > 3) return false;
    step = 1;
    if (parts.size() == 3 && !tj_trim(parts[2]).empty()) {
        try { step = std::stoll(tj_trim(parts[2])); } catch (...) { return false; }
        if (step == 0) return false;
    }
    if (step > 0) {
        start = parse_part(parts[0], 0, n, false);
        stop = parse_part(parts[1], n, n, true);
    } else {
        start = parse_part(parts[0], n - 1, n, false);
        stop = parse_part(parts[1], -1, n, true);
    }
    return true;
}

// Resolve a message-list expression: "messages", "messages[1:]",
// "messages[::-1]", a list variable, or "<list> | selectattr(...) | list".
// Returns false for anything that is not a message list.
static bool jinja_resolve_list(const std::string& raw,
                               const std::vector<ChatMessage>& msgs, JinjaEnv& env,
                               std::vector<ChatMessage>& out) {
    std::string s = tj_trim(raw);
    // Strip a trailing "| list" filter.
    if (s.size() > 6 && s.compare(s.size() - 6, 6, "| list") == 0) {
        s = tj_trim(s.substr(0, s.size() - 6));
    }
    // "<base> | selectattr("role", "equalto", "user")": filter by role.
    std::string filter_role;
    size_t sel = s.find("| selectattr(");
    if (sel != std::string::npos) {
        size_t close = s.find(')', sel);
        if (close == std::string::npos) return false;
        std::string args = s.substr(sel + 14, close - (sel + 14));
        // args: "role", "equalto", "user"
        std::vector<std::string> parts;
        size_t pp = 0;
        bool inq = false;
        char q = 0;
        std::string cur;
        for (size_t i = 0; i <= args.size(); ++i) {
            char c = (i < args.size()) ? args[i] : ',';
            if (inq) {
                cur.push_back(c);
                if (c == q) inq = false;
            } else if (c == '\'' || c == '"') {
                inq = true; q = c; cur.push_back(c);
            } else if (c == ',') {
                parts.push_back(tj_trim(cur)); cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (parts.size() != 3) return false;
        if (jinja_unquote(parts[0]) != "role" || jinja_unquote(parts[1]) != "equalto") return false;
        filter_role = jinja_unquote(parts[2]);
        s = tj_trim(s.substr(0, sel));
    }
    std::vector<ChatMessage> base;
    if (s == "messages") {
        base = msgs;
    } else if (s.rfind("messages[", 0) == 0) {
        size_t close = s.find(']');
        if (close == std::string::npos || close != s.size() - 1) return false;
        long long st, sp, step;
        bool single = false;
        if (!jinja_parse_slice(s.substr(8, close - 8 + 1), static_cast<long long>(msgs.size()),
                               st, sp, step, single)) return false;
        if (single) {
            base.push_back(msgs[static_cast<size_t>(st)]);
        } else {
            for (long long i = st; step > 0 ? i < sp : i > sp; i += step) {
                if (i >= 0 && i < static_cast<long long>(msgs.size()))
                    base.push_back(msgs[static_cast<size_t>(i)]);
            }
        }
    } else {
        auto it = env.lists.find(s);
        if (it == env.lists.end()) return false;
        base = it->second;
    }
    if (!filter_role.empty()) {
        std::vector<ChatMessage> filtered;
        for (const auto& m : base) if (m.role == filter_role) filtered.push_back(m);
        base.swap(filtered);
    }
    out.swap(base);
    return true;
}

// Evaluate a string expression: a base value (string literal, message
// field, variable, namespace attr, messages[N]["content"]) followed by
// .method(args) calls and [index]/[start:end] subscripts. Methods:
// split, strip/lstrip/rstrip, upper/lower, startswith/endswith (yield
// "true"/"false"). Returns false when the base cannot be resolved or an
// operation is unsupported — callers leave the previous value untouched.
static bool jinja_eval_chain(const std::string& raw, const ChatMessage* msg,
                             const std::vector<ChatMessage>& msgs, int loop_idx, JinjaEnv& env,
                             std::string& out) {
    std::string s = tj_trim(raw);
    if (s.empty()) return false;
    std::string cur_str;
    std::vector<std::string> cur_list;
    bool is_list = false;
    size_t pos = 0;

    // --- base value ---
    if (s[pos] == '\'' || s[pos] == '"') {
        char q = s[pos];
        size_t j = pos + 1;
        while (j < s.size() && s[j] != q) {
            if (s[j] == '\\' && j + 1 < s.size()) ++j;
            ++j;
        }
        if (j >= s.size()) return false;
        cur_str = jinja_unquote(s.substr(pos, j - pos + 1));
        pos = j + 1;
    } else if (s.rfind("messages[", 0) == 0) {
        // messages[N]["content"] / messages[N].role — index via the
        // existing index evaluator (handles loop.index0 arithmetic).
        size_t close = s.find(']');
        if (close == std::string::npos) return false;
        int idx = eval_jinja_index_expr(s.substr(9, close - 9), loop_idx);
        if (idx < 0 || idx >= static_cast<int>(msgs.size())) return false;
        std::string rest = tj_trim(s.substr(close + 1));
        size_t consumed = 0;
        if (rest.rfind("['role']", 0) == 0 || rest.rfind("[\"role\"]", 0) == 0) {
            cur_str = msgs[static_cast<size_t>(idx)].role;
            consumed = 8;
        } else if (rest.rfind(".role", 0) == 0) {
            cur_str = msgs[static_cast<size_t>(idx)].role;
            consumed = 5;
        } else if (rest.rfind("['content']", 0) == 0 || rest.rfind("[\"content\"]", 0) == 0) {
            cur_str = msgs[static_cast<size_t>(idx)].content;
            consumed = 11;
        } else if (rest.rfind(".content", 0) == 0) {
            cur_str = msgs[static_cast<size_t>(idx)].content;
            consumed = 8;
        } else {
            return false;
        }
        pos = close + 1 + consumed;
    } else {
        // Identifier path: name[.name...] — but a ".method(" starts a call.
        size_t j = pos;
        std::string base;
        while (true) {
            size_t k = j;
            while (k < s.size() && (std::isalnum(static_cast<unsigned char>(s[k])) ||
                                    s[k] == '_')) ++k;
            if (k == j) return false;
            std::string ident = s.substr(j, k - j);
            if (!base.empty()) base += ".";
            base += ident;
            j = k;
            if (j < s.size() && s[j] == '.') {
                // Peek: identifier followed by '(' means a method call.
                size_t m = j + 1;
                while (m < s.size() && (std::isalnum(static_cast<unsigned char>(s[m])) ||
                                        s[m] == '_')) ++m;
                if (m < s.size() && s[m] == '(') break;
                if (m == j + 1) return false;
                j = j + 1; // skip the '.'; next iteration reads the field name
                continue;
            }
            if (j < s.size() && s[j] == '[') {
                // message['role'] / message["content"] field access.
                size_t close = s.find(']', j);
                if (close == std::string::npos) return false;
                std::string field = jinja_unquote(tj_trim(s.substr(j + 1, close - j - 1)));
                if (field != "role" && field != "content") return false;
                base += (field == "role") ? ".role" : ".content";
                j = close + 1;
            }
            break;
        }
        if (!jinja_lookup_value(base, msg, loop_idx, env, cur_str)) return false;
        pos = j;
    }

    // --- suffix chain: .method(args) / [index] / [a:b] ---
    while (pos < s.size()) {
        if (s[pos] == '.') {
            size_t m = pos + 1;
            while (m < s.size() && (std::isalnum(static_cast<unsigned char>(s[m])) ||
                                    s[m] == '_')) ++m;
            std::string method = s.substr(pos + 1, m - pos - 1);
            if (m >= s.size() || s[m] != '(') return false;
            size_t close = jinja_matching_paren(s, m);
            if (close == std::string::npos) return false;
            std::string arg = tj_trim(s.substr(m + 1, close - m - 1));
            if (is_list) return false; // no methods on a split list
            if (method == "split") {
                std::string sep = jinja_unquote(arg);
                cur_list.clear();
                if (sep.empty()) {
                    cur_list.push_back(cur_str);
                } else {
                    size_t from = 0;
                    while (true) {
                        size_t f = cur_str.find(sep, from);
                        if (f == std::string::npos) {
                            cur_list.push_back(cur_str.substr(from));
                            break;
                        }
                        cur_list.push_back(cur_str.substr(from, f - from));
                        from = f + sep.size();
                    }
                }
                is_list = true;
            } else if (method == "strip" || method == "lstrip" || method == "rstrip") {
                std::string chars = arg.empty() ? " \t\r\n" : jinja_unquote(arg);
                auto is_strip = [&](char c) { return chars.find(c) != std::string::npos; };
                if (method != "lstrip") {
                    while (!cur_str.empty() && is_strip(cur_str.back())) cur_str.pop_back();
                }
                if (method != "rstrip") {
                    size_t b = 0;
                    while (b < cur_str.size() && is_strip(cur_str[b])) ++b;
                    cur_str.erase(0, b);
                }
            } else if (method == "upper" || method == "lower") {
                for (char& c : cur_str) {
                    c = (method == "upper")
                        ? static_cast<char>(std::toupper(static_cast<unsigned char>(c)))
                        : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            } else if (method == "startswith" || method == "endswith") {
                std::string pre = jinja_unquote(arg);
                bool r = (method == "startswith")
                    ? (cur_str.size() >= pre.size() && cur_str.compare(0, pre.size(), pre) == 0)
                    : (cur_str.size() >= pre.size() &&
                       cur_str.compare(cur_str.size() - pre.size(), pre.size(), pre) == 0);
                cur_str = r ? "true" : "false";
            } else {
                return false;
            }
            pos = close + 1;
        } else if (s[pos] == '[') {
            size_t close = s.find(']', pos);
            if (close == std::string::npos) return false;
            std::string inner = s.substr(pos, close - pos + 1);
            if (is_list) {
                long long st, sp, step;
                bool single = false;
                if (!jinja_parse_slice(inner, static_cast<long long>(cur_list.size()),
                                       st, sp, step, single)) return false;
                if (single) {
                    cur_str = cur_list[static_cast<size_t>(st)];
                    is_list = false;
                } else {
                    return false; // list slices are not rendered as strings
                }
            } else {
                long long st, sp, step;
                bool single = false;
                if (!jinja_parse_slice(inner, static_cast<long long>(cur_str.size()),
                                       st, sp, step, single)) return false;
                if (single) {
                    cur_str = std::string(1, cur_str[static_cast<size_t>(st)]);
                } else {
                    std::string sliced;
                    for (long long i = st; step > 0 ? i < sp : i > sp; i += step) {
                        if (i >= 0 && i < static_cast<long long>(cur_str.size()))
                            sliced.push_back(cur_str[static_cast<size_t>(i)]);
                    }
                    cur_str = sliced;
                }
            }
            pos = close + 1;
        } else {
            return false; // trailing junk
        }
    }
    if (is_list) return false;
    out = cur_str;
    return true;
}

// Apply one {% set ... %} statement to env. Returns false when the
// statement is unsupported — the caller then leaves env unchanged.
static bool jinja_apply_set(const std::string& content, const ChatMessage* msg,
                            const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                            JinjaEnv& env) {
    std::string rest = tj_trim(content.substr(4)); // after "set "
    // Split target = rhs on the first top-level '=' (not '==' etc.).
    size_t eq = std::string::npos;
    {
        int depth = 0;
        char quote = 0;
        for (size_t i = 0; i < rest.size(); ++i) {
            char c = rest[i];
            if (quote) {
                if (c == '\\' && i + 1 < rest.size()) ++i;
                else if (c == quote) quote = 0;
            } else if (c == '\'' || c == '"') {
                quote = c;
            } else if (c == '(' || c == '[') {
                ++depth;
            } else if (c == ')' || c == ']') {
                --depth;
            } else if (c == '=' && depth == 0 &&
                       (i + 1 >= rest.size() || rest[i + 1] != '=')) {
                eq = i;
                break;
            }
        }
    }
    if (eq == std::string::npos) return false;
    std::string target = tj_trim(rest.substr(0, eq));
    std::string rhs = tj_trim(rest.substr(eq + 1));
    if (target.empty() || rhs.empty()) return false;
    bool is_ns_attr = (target.find('.') != std::string::npos);
    auto store_str = [&](const std::string& v) {
        if (is_ns_attr) {
            (*env.ns)[target] = v;
        } else {
            env.vars[target] = v;
            env.is_none.erase(target);
            env.lists.erase(target);
        }
    };

    // namespace(...) / namespace(): initialise namespace attributes.
    if (rhs.rfind("namespace(", 0) == 0 && rhs.back() == ')') {
        std::string args = rhs.substr(10, rhs.size() - 11);
        if (!tj_trim(args).empty()) {
            // Split kwargs on top-level commas.
            std::vector<std::string> kwargs;
            {
                int depth = 0;
                char quote = 0;
                size_t start = 0;
                for (size_t i = 0; i <= args.size(); ++i) {
                    char c = (i < args.size()) ? args[i] : ',';
                    if (quote) {
                        if (c == '\\' && i + 1 < args.size()) ++i;
                        else if (c == quote) quote = 0;
                    } else if (c == '\'' || c == '"') {
                        quote = c;
                    } else if (c == '(' || c == '[') {
                        ++depth;
                    } else if (c == ')' || c == ']') {
                        --depth;
                    } else if (c == ',' && depth == 0) {
                        kwargs.push_back(args.substr(start, i - start));
                        start = i + 1;
                    }
                }
            }
            for (const auto& kw : kwargs) {
                size_t e = kw.find('=');
                if (e == std::string::npos) return false;
                std::string k = tj_trim(kw.substr(0, e));
                std::string v = tj_trim(kw.substr(e + 1));
                if (v == "true" || v == "false") {
                    (*env.ns)[target + "." + k] = v;
                } else if (!v.empty() && (v.front() == '\'' || v.front() == '"')) {
                    (*env.ns)[target + "." + k] = jinja_unquote(v);
                } else {
                    long long iv = 0;
                    if (!jinja_eval_int(v, msg, msgs, loop_idx, loop_total, env, iv)) return false;
                    (*env.ns)[target + "." + k] = std::to_string(iv);
                }
            }
        }
        return true;
    }
    if (rhs == "none") {
        if (is_ns_attr) return false;
        env.is_none[target] = true;
        env.vars[target] = "";
        env.lists.erase(target);
        return true;
    }
    if (rhs == "true" || rhs == "false") {
        store_str(rhs);
        return true;
    }
    if (rhs.front() == '\'' || rhs.front() == '"') {
        store_str(jinja_unquote(rhs));
        return true;
    }
    // Message-list assignment: {% set xs = messages[1:] %} etc.
    std::vector<ChatMessage> list;
    if (jinja_resolve_list(rhs, msgs, env, list)) {
        if (is_ns_attr) return false;
        env.lists[target] = std::move(list);
        env.vars.erase(target);
        env.is_none.erase(target);
        return true;
    }
    // Integer expression: {% set index = (messages|length - 1) - loop.index0 %}
    long long iv = 0;
    if (jinja_eval_int(rhs, msg, msgs, loop_idx, loop_total, env, iv)) {
        store_str(std::to_string(iv));
        return true;
    }
    // String concatenation: {% set x = messages[0]['content'] + '\n\n' %}
    // (Gemma3's first_user_prefix). Split on top-level '+' and evaluate
    // each part as a literal, chain, or integer.
    if (rhs.find('+') != std::string::npos) {
        std::string joined;
        bool okcat = true;
        for (const auto& part : jinja_split_plus(rhs)) {
            std::string p = tj_trim(part);
            std::string pv;
            if (!p.empty() && (p.front() == '\'' || p.front() == '"')) {
                pv = jinja_unquote(p);
            } else if (jinja_eval_chain(p, msg, msgs, loop_idx, env, pv)) {
                // pv already set
            } else {
                long long piv = 0;
                if (jinja_eval_int(p, msg, msgs, loop_idx, loop_total, env, piv)) {
                    pv = std::to_string(piv);
                } else {
                    okcat = false;
                    break;
                }
            }
            joined += pv;
        }
        if (okcat) {
            store_str(joined);
            return true;
        }
        // Otherwise fall through: leave env unchanged.
    }
    // String / chain expression: message.content, content.split(...)[-1], ...
    std::string sv;
    if (jinja_eval_chain(rhs, msg, msgs, loop_idx, env, sv)) {
        store_str(sv);
        return true;
    }
    return false;
}

// Evaluate a {{ ... }} expression: string literals and message fields joined
// JSON-encode a plain string as a quoted JSON string (for the `tojson`
// filter applied to a non-tool value, e.g. {{ tool.function.name | tojson }}
// -> "get_weather"). Tool objects / function / parameters are already JSON
// and are returned verbatim instead — see eval_jinja_expr.
static std::string jinja_json_quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out += "\"";
    return out;
}

// by '+', with an optional `| trim` filter. Variables set by {% set %} and
// namespace attributes resolve from `env` first; message fields are the
// fallback (so {{ content }} still works when a template never set it).
// Unknown variables yield "". Also handles inline conditionals
// (A if COND else B), used by Gemma3 for the system prefix.
static std::string eval_jinja_expr(const std::string& expr, const ChatMessage* msg,
                                   bool add_gen, const Tokenizer& tok,
                                   const std::vector<ChatMessage>& msgs, int loop_idx,
                                   int loop_total, JinjaEnv& env) {
    std::string out;
    for (const auto& raw_term : jinja_split_plus(expr)) {
        std::string term = tj_trim(raw_term);
        bool do_trim = false;
        // A Jinja filter '|' must be OUTSIDE string literals: '<|im_start|>'
        // and friends contain '|' characters that are literal text, not
        // filters. (This was truncating every ChatML marker to "'<" .)
        size_t pipe = std::string::npos;
        {
            char quote = 0;
            for (size_t i = 0; i < term.size(); ++i) {
                char c = term[i];
                if (quote) {
                    if (c == '\\' && i + 1 < term.size()) ++i;
                    else if (c == quote) quote = 0;
                } else if (c == '\'' || c == '"') {
                    quote = c;
                } else if (c == '|') {
                    pipe = i;
                    break;
                }
            }
        }
        std::string filter;
        if (pipe != std::string::npos) {
            filter = term.substr(pipe);
            if (filter.find("trim") != std::string::npos) do_trim = true;
            term = tj_trim(term.substr(0, pipe));
        }
        std::string val;
        if (!term.empty() && (term.front() == '\'' || term.front() == '"')) {
            val = jinja_unquote(term);
        } else if (term == "bos_token") {
            val = jinja_token_text(tok, tok.bos_id());
        } else if (term == "eos_token") {
            val = jinja_token_text(tok, tok.eos_id());
        } else if (term == "add_generation_prompt") {
            val = "";
        } else if (term == "messages|length") {
            val = std::to_string(msgs.size());
        } else if (term == "tools|length") {
            val = std::to_string(env.tools.size());
        } else if (term.find(" if ") != std::string::npos &&
                   term.find(" else ") != std::string::npos) {
            // Inline conditional: (A if COND else B).
            std::string t = tj_trim(term);
            if (t.size() >= 2 && t.front() == '(' && t.back() == ')') {
                t = tj_trim(t.substr(1, t.size() - 2));
            }
            size_t if_pos = t.find(" if ");
            size_t else_pos = (if_pos == std::string::npos)
                ? std::string::npos : t.find(" else ", if_pos + 4);
            if (if_pos != std::string::npos && else_pos != std::string::npos) {
                std::string a = tj_trim(t.substr(0, if_pos));
                std::string cond = tj_trim(t.substr(if_pos + 4, else_pos - (if_pos + 4)));
                std::string b = tj_trim(t.substr(else_pos + 6));
                std::string chosen =
                    eval_jinja_cond(cond, msg, add_gen, msgs, loop_idx, loop_total, env) ? a : b;
                if (!chosen.empty() && (chosen.front() == '\'' || chosen.front() == '"')) {
                    val = jinja_unquote(chosen);
                } else if (!jinja_eval_chain(chosen, msg, msgs, loop_idx, env, val)) {
                    val = jinja_field(msg, chosen);
                }
            }
        } else if (jinja_eval_chain(term, msg, msgs, loop_idx, env, val)) {
            // Resolved via a {% set %} variable, namespace attribute,
            // message field, or method chain (e.g. content.strip('\n')).
        } else {
            val = jinja_field(msg, term);
        }
        if (do_trim) val = tj_trim(val);
        if (filter.find("length") != std::string::npos) val = std::to_string(val.size());
        if (filter.find("tojson") != std::string::npos) {
            // `{{ tool | tojson }}` (Qwen2.5/3, Mistral, Llama3 templates):
            // tool / tool.function / tool.function.parameters already hold
            // raw JSON — emit verbatim. Anything else is JSON-quoted as a
            // string (e.g. tool.function.name -> "get_weather").
            if (term != "tool" && term != "tool.function" &&
                term != "tool.function.parameters") {
                val = jinja_json_quote(val);
            }
        }
        out += val;
    }
    return out;
}

// Evaluate a messages[...] index expression: a plain integer, or
// loop.index / loop.index0 with +/- arithmetic (e.g. loop.index0 - 1).
static int eval_jinja_index_expr(const std::string& raw, int loop_idx) {
    std::string s = tj_trim(raw);
    auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
        size_t p = 0;
        while ((p = str.find(from, p)) != std::string::npos) {
            str.replace(p, from.size(), to);
            p += to.size();
        }
    };
    // loop.index0 first (it contains "loop.index" as a prefix).
    replace_all(s, "loop.index0", std::to_string(loop_idx));
    replace_all(s, "loop.index", std::to_string(loop_idx + 1));
    // Evaluate a sum of integers joined by + / -.
    int total = 0;
    int sign = 1;
    size_t i = 0;
    bool any = false;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (i >= s.size()) break;
        if (s[i] == '+') { sign = 1; ++i; continue; }
        if (s[i] == '-') { sign = -1; ++i; continue; }
        if (s[i] < '0' || s[i] > '9') return -1;
        size_t j = i;
        while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        int v = 0;
        try { v = std::stoi(s.substr(i, j - i)); } catch (...) { return -1; }
        total += sign * v;
        sign = 1;
        any = true;
        i = j;
    }
    return any ? total : -1;
}

// Evaluate one Jinja condition atom (no and/or at this level; `not` and
// parentheses are handled here per-atom).
static bool eval_jinja_cond(const std::string& cond, const ChatMessage* msg, bool add_gen,
                            const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                            JinjaEnv& env);
static bool eval_jinja_atom(const std::string& raw, const ChatMessage* msg, bool add_gen,
                            const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                            JinjaEnv& env) {
    std::string c = tj_trim(raw);
    // `not (...)` / `not X`: negate the inner condition.
    if (c.rfind("not(", 0) == 0 || c.rfind("not (", 0) == 0) {
        size_t open = c.find('(');
        size_t close = jinja_matching_paren(c, open);
        if (close != std::string::npos) {
            return !eval_jinja_cond(c.substr(open + 1, close - open - 1), msg, add_gen,
                                    msgs, loop_idx, loop_total, env);
        }
        return false;
    }
    if (c.rfind("not ", 0) == 0) {
        return !eval_jinja_atom(c.substr(4), msg, add_gen, msgs, loop_idx, loop_total, env);
    }
    // Strip ONE pair of parentheses only when it encloses the whole atom.
    // (Stripping all leading/trailing parens mangled "(a) != (b)".)
    c = jinja_strip_outer_parens(c);
    // `X is defined` / `X is not defined`: add_generation_prompt, loop.*,
    // message and messages are defined, message.role/content are defined
    // when a message is in scope, {% set %} variables (including `none`)
    // are defined once set. message.tool_calls / message.reasoning_content
    // and any other message.X are never defined (no tools are ever passed).
    if (c.find(" is defined") != std::string::npos) {
        bool neg = (c.find(" is not defined") != std::string::npos);
        std::string var = tj_trim(c.substr(0, c.find(" is ")));
        bool defined = false;
        if (var == "add_generation_prompt" || var == "message" || var == "messages" ||
            var.rfind("loop.", 0) == 0) {
            defined = true;
        } else if (var == "tools") {
            // Defined only when the caller actually passed a tool list.
            defined = env.tools_present;
        } else if (var == "message.role" || var == "message.content" ||
                   var == "message['role']" || var == "message[\"role\"]" ||
                   var == "message['content']" || var == "message[\"content\"]") {
            defined = (msg != nullptr);
        } else if (var.rfind("messages[", 0) == 0) {
            // messages[N]['content'] / messages[N].role: defined when the
            // index is in range and the field resolves.
            std::string v;
            defined = jinja_eval_chain(var, msg, msgs, loop_idx, env, v);
        } else if (env.vars.count(var) || env.lists.count(var) ||
                   env.ns->count(var) || env.is_none.count(var)) {
            defined = true;
        }
        return neg ? !defined : defined;
    }
    // `X is string` / `X is not string`: message fields and {% set %}
    // string variables are strings; `none`, tool_calls and unknown names
    // are not. (Qwen3 gates its {% set content = ... %} on this.)
    if (c.find(" is string") != std::string::npos ||
        c.find(" is not string") != std::string::npos) {
        bool neg = (c.find(" is not string") != std::string::npos);
        std::string var = tj_trim(c.substr(0, c.find(" is ")));
        bool is_str = false;
        if (var == "message.role" || var == "message.content" ||
            var == "message['role']" || var == "message[\"role\"]" ||
            var == "message['content']" || var == "message[\"content\"]" ||
            var == "content" || var == "role") {
            is_str = (msg != nullptr);
        } else if (var.rfind("messages[", 0) == 0) {
            // messages[N]['content']: a string when it resolves (Gemma3
            // gates its first_user_prefix set on this; without it the
            // system message was silently dropped).
            std::string v;
            is_str = jinja_eval_chain(var, msg, msgs, loop_idx, env, v);
        } else if (!var.empty() && (var.front() == '\'' || var.front() == '"')) {
            is_str = true;
        } else if (env.vars.count(var) && !env.is_none.count(var)) {
            is_str = true;
        }
        return neg ? !is_str : is_str;
    }
    // `X is none` / `X is not none`: only an explicit {% set x = none %}.
    if (c.find(" is none") != std::string::npos ||
        c.find(" is not none") != std::string::npos) {
        bool neg = (c.find(" is not none") != std::string::npos);
        std::string var = tj_trim(c.substr(0, c.find(" is ")));
        bool is_none = (var == "none") || (env.is_none.count(var) != 0);
        return neg ? !is_none : is_none;
    }
    if (c == "add_generation_prompt") return add_gen;
    if (c == "loop.first") return loop_idx == 0;
    if (c == "loop.last") return loop_total > 0 && loop_idx == loop_total - 1;
    if (c == "messages") return !msgs.empty();
    if (c == "message") return msg != nullptr;
    if (c == "message['content']" || c == "message[\"content\"]" || c == "message.content") {
        return msg && !msg->content.empty();
    }
    // messages[EXPR]['role'|'content'] == 'v' / != 'v'  (EXPR may use loop.index0)
    if (c.rfind("messages[", 0) == 0) {
        size_t close = c.find(']');
        if (close != std::string::npos) {
            int idx = eval_jinja_index_expr(c.substr(9, close - 9), loop_idx);
            std::string rest = tj_trim(c.substr(close + 1));
            std::string field;
            if (rest.rfind("['role']", 0) == 0 || rest.rfind("[\"role\"]", 0) == 0 || rest.rfind(".role", 0) == 0) {
                field = (idx >= 0 && idx < static_cast<int>(msgs.size())) ? msgs[static_cast<size_t>(idx)].role : "";
            } else if (rest.rfind("['content']", 0) == 0 || rest.rfind("[\"content\"]", 0) == 0 || rest.rfind(".content", 0) == 0) {
                field = (idx >= 0 && idx < static_cast<int>(msgs.size())) ? msgs[static_cast<size_t>(idx)].content : "";
            } else {
                return false;
            }
            if (rest.find("!=") != std::string::npos) {
                return field != jinja_unquote(rest.substr(rest.find("!=") + 2));
            }
            if (rest.find("==") != std::string::npos) {
                return field == jinja_unquote(rest.substr(rest.find("==") + 2));
            }
            return !field.empty();
        }
        return false;
    }
    // '<lit>' in <string expr>: substring test (Qwen3: '</think>' in content).
    {
        size_t in_pos = c.find(" in ");
        if (in_pos != std::string::npos) {
            std::string lhs = tj_trim(c.substr(0, in_pos));
            std::string rhs = tj_trim(c.substr(in_pos + 4));
            if (lhs.size() >= 2 && lhs.front() == lhs.back() &&
                (lhs.front() == '\'' || lhs.front() == '"') &&
                !rhs.empty() && rhs.front() != '[') {
                std::string needle = jinja_unquote(lhs);
                std::string hay;
                if (jinja_eval_chain(rhs, msg, msgs, loop_idx, env, hay)) {
                    return hay.find(needle) != std::string::npos;
                }
                return false;
            }
        }
    }
    // message['role'] in ['a', 'b'] / not in [...]
    {
        size_t in_pos = c.find(" in ");
        size_t not_in = c.find(" not in ");
        if (in_pos != std::string::npos || not_in != std::string::npos) {
            bool neg = (not_in != std::string::npos);
            size_t op = neg ? not_in : in_pos;
            std::string lhs = jinja_field(msg, tj_trim(c.substr(0, op)));
            std::string list = c.substr(op + (neg ? 8 : 4));
            bool found = false;
            size_t lb = list.find('['), rb = list.rfind(']');
            if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
                std::string inner = list.substr(lb + 1, rb - lb - 1);
                size_t p = 0;
                while (p < inner.size()) {
                    size_t comma = inner.find(',', p);
                    std::string item = jinja_unquote(tj_trim(inner.substr(p, comma == std::string::npos ? std::string::npos : comma - p)));
                    if (item == lhs) { found = true; break; }
                    if (comma == std::string::npos) break;
                    p = comma + 1;
                }
            }
            return neg ? !found : found;
        }
    }
    // message['role'] == 'v' / != 'v' — generalised: integer comparison
    // first (loop.index, ns counters, `ns.index % 2 == 0`), then a
    // parenthesised boolean sub-expression on either side
    // ((role == "user") != (ns.index % 2 == 0)), then string comparison.
    {
        size_t op_pos = std::string::npos;
        bool is_ne = false;
        size_t ne = c.find("!=");
        size_t eq = c.find("==");
        if (ne != std::string::npos && (eq == std::string::npos || ne < eq)) {
            op_pos = ne;
            is_ne = true;
        } else if (eq != std::string::npos) {
            op_pos = eq;
            is_ne = false;
        }
        if (op_pos != std::string::npos) {
            std::string lhs = tj_trim(c.substr(0, op_pos));
            std::string rhs = tj_trim(c.substr(op_pos + 2));
            auto eval_bool_side = [&](const std::string& side, bool& b) -> bool {
                std::string t = tj_trim(side);
                if (t.size() >= 2 && t.front() == '(' &&
                    jinja_matching_paren(t, 0) == t.size() - 1) {
                    b = eval_jinja_cond(t.substr(1, t.size() - 2), msg, add_gen, msgs,
                                        loop_idx, loop_total, env);
                    return true;
                }
                return false;
            };
            bool lb = false, rb = false;
            bool l_is_bool = eval_bool_side(lhs, lb);
            bool r_is_bool = eval_bool_side(rhs, rb);
            if (l_is_bool || r_is_bool) {
                if (!l_is_bool) {
                    if (lhs == "true") lb = true;
                    else if (lhs == "false") lb = false;
                    else return false;
                }
                if (!r_is_bool) {
                    if (rhs == "true") rb = true;
                    else if (rhs == "false") rb = false;
                    else return false;
                }
                return is_ne ? (lb != rb) : (lb == rb);
            }
            long long li = 0, ri = 0;
            if (jinja_eval_int(lhs, msg, msgs, loop_idx, loop_total, env, li) &&
                jinja_eval_int(rhs, msg, msgs, loop_idx, loop_total, env, ri)) {
                return is_ne ? (li != ri) : (li == ri);
            }
            std::string lv, rv;
            if (!jinja_eval_chain(lhs, msg, msgs, loop_idx, env, lv)) lv = jinja_field(msg, lhs);
            if (!jinja_eval_chain(rhs, msg, msgs, loop_idx, env, rv)) {
                if (!rhs.empty() && (rhs.front() == '\'' || rhs.front() == '"')) {
                    rv = jinja_unquote(rhs);
                } else {
                    rv = jinja_field(msg, rhs);
                }
            }
            return is_ne ? (lv != rv) : (lv == rv);
        }
    }
    // Integer ordering comparisons: loop.index0 > ns.last_query_index etc.
    // (loop.index == N is already handled by the == block above.)
    {
        static const char* kOps[] = {">=", "<=", ">", "<"};
        for (const char* op : kOps) {
            std::string ops(op);
            size_t p = c.find(ops);
            if (p == std::string::npos) continue;
            std::string lhs = tj_trim(c.substr(0, p));
            std::string rhs = tj_trim(c.substr(p + ops.size()));
            long long li = 0, ri = 0;
            if (jinja_eval_int(lhs, msg, msgs, loop_idx, loop_total, env, li) &&
                jinja_eval_int(rhs, msg, msgs, loop_idx, loop_total, env, ri)) {
                if (ops == ">=") return li >= ri;
                if (ops == "<=") return li <= ri;
                if (ops == ">") return li > ri;
                return li < ri;
            }
            return false;
        }
    }
    // message.content.startswith('x') / .endswith('x') as a bare atom.
    if (c.find(".startswith(") != std::string::npos ||
        c.find(".endswith(") != std::string::npos) {
        std::string v;
        if (jinja_eval_chain(c, msg, msgs, loop_idx, env, v)) return v == "true";
        return false;
    }
    // Bare variable / namespace truthiness: {% if ns.multi_step_tool %}.
    {
        std::string v;
        if (jinja_lookup_value(c, msg, loop_idx, env, v)) {
            return !v.empty() && v != "false";
        }
        if (env.is_none.count(c)) return false;
    }
    // Unknown / undefined variable: falsy in Jinja (e.g. `tools` when no tools
    // were passed). Never silently render a branch we cannot evaluate.
    return false;
}

static bool eval_jinja_cond(const std::string& cond, const ChatMessage* msg, bool add_gen,
                            const std::vector<ChatMessage>& msgs, int loop_idx, int loop_total,
                            JinjaEnv& env) {
    // `not` is handled per-atom (Jinja: `not A and B` == `(not A) and B`).
    // Split on top-level " or " first (lower precedence), then " and ".
    // Splits are paren-aware: "not(A and B)" must stay one atom (Qwen3's
    // tool-response guard), and quoted ' and ' inside string literals
    // must not split either.
    auto split_top = [](const std::string& s, const std::string& delim) {
        std::vector<std::string> parts;
        int depth = 0;
        char quote = 0;
        size_t start = 0;
        for (size_t i = 0; i + delim.size() <= s.size(); ++i) {
            char ch = s[i];
            if (quote) {
                if (ch == '\\' && i + 1 < s.size()) ++i;
                else if (ch == quote) quote = 0;
            } else if (ch == '\'' || ch == '"') {
                quote = ch;
            } else if (ch == '(' || ch == '[') {
                ++depth;
            } else if (ch == ')' || ch == ']') {
                --depth;
            } else if (depth == 0 && s.compare(i, delim.size(), delim) == 0) {
                parts.push_back(s.substr(start, i - start));
                i += delim.size() - 1;
                start = i + 1;
            }
        }
        parts.push_back(s.substr(start));
        return parts;
    };
    std::string c = jinja_strip_outer_parens(cond);
    bool result = false;
    bool first_or = true;
    for (const auto& or_part : split_top(c, " or ")) {
        bool and_result = true;
        for (const auto& atom : split_top(or_part, " and ")) {
            if (!eval_jinja_atom(atom, msg, add_gen, msgs, loop_idx, loop_total, env)) {
                and_result = false;
                break;
            }
        }
        if (first_or) { result = and_result; first_or = false; }
        else result = result || and_result;
    }
    return result;
}

// Find a {% ... %} or {{ ... }} tag starting at or after `from`. Returns
// false when none. `is_var` distinguishes {{ }} from {% %}; content is the
// tag body with optional '-' whitespace markers stripped.
struct JinjaTag {
    size_t start = 0;
    size_t end = 0; // one past %}
    bool is_var = false;
    std::string content;
    // Jinja whitespace control: {%- / -%} strip adjacent whitespace.
    // Real chat templates rely on this heavily; without it the rendered
    // prompt is full of stray blank lines and spaces (which measurably
    // changes what a small model generates).
    bool trim_before = false;
    bool trim_after = false;
};

static bool find_jinja_tag(const std::string& s, size_t from, bool is_var, JinjaTag& tag) {
    std::string open = is_var ? "{{" : "{%";
    size_t st = s.find(open, from);
    if (st == std::string::npos) return false;
    std::string close = is_var ? "}}" : "%}";
    size_t en = s.find(close, st + 2);
    if (en == std::string::npos) return false;
    tag.start = st;
    tag.end = en + 2;
    tag.is_var = is_var;
    tag.trim_before = false;
    tag.trim_after = false;
    tag.content = tj_trim(s.substr(st + 2, en - (st + 2)));
    if (!tag.content.empty() && tag.content.front() == '-') {
        tag.trim_before = true;
        tag.content = tj_trim(tag.content.substr(1));
    }
    if (!tag.content.empty() && tag.content.back() == '-') {
        tag.trim_after = true;
        tag.content = tj_trim(tag.content.substr(0, tag.content.size() - 1));
    }
    return true;
}

static std::string jinja_rstrip_ws(const std::string& s) {
    size_t e = s.find_last_not_of(" \t\r\n");
    return (e == std::string::npos) ? "" : s.substr(0, e + 1);
}

static std::string jinja_lstrip_ws(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    return (b == std::string::npos) ? "" : s.substr(b);
}

// Plain {{ expr }} substitution (no {% %} statement handling), honouring
// {{- / -}} whitespace control.
static std::string render_jinja_vars(const std::string& seg, const ChatMessage* msg, bool add_gen,
                                     const Tokenizer& tok,
                                     const std::vector<ChatMessage>& msgs, int loop_idx,
                                     int loop_total, JinjaEnv& env) {
    std::string out;
    size_t pos = 0;
    bool skip_leading_ws = false;
    JinjaTag vt;
    while (find_jinja_tag(seg, pos, true, vt)) {
        std::string lit = seg.substr(pos, vt.start - pos);
        if (skip_leading_ws) lit = jinja_lstrip_ws(lit);
        if (vt.trim_before) lit = jinja_rstrip_ws(lit);
        out += lit;
        out += eval_jinja_expr(vt.content, msg, add_gen, tok, msgs, loop_idx, loop_total, env);
        skip_leading_ws = vt.trim_after;
        pos = vt.end;
    }
    std::string tail = seg.substr(pos);
    if (skip_leading_ws) tail = jinja_lstrip_ws(tail);
    out += tail;
    return out;
}

// Render a template segment: handles {% if/elif/else/endif %},
// {% for x in y %}/{% endfor %} and {% set ... %} (recursively, with
// nesting-aware matching). `msg` is the current loop message (or nullptr).
// For-loops over anything other than `messages` (tools, tool_calls) render
// zero iterations: this renderer is never passed tools.
// `env` carries {% set %} state; `{% set messages = ... %}` rebinds the
// message list for the rest of the segment (effective list `eff`).
static std::string render_jinja_segment(const std::string& seg, const ChatMessage* msg, bool add_gen,
                                        const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                        int loop_idx, int loop_total, JinjaEnv& env) {
    auto mi = env.lists.find("messages");
    const std::vector<ChatMessage>& eff = (mi != env.lists.end()) ? mi->second : msgs;
    // Find the first {% if %}, {% for %} or {% set %} statement tag.
    JinjaTag if_tag;
    size_t search = 0;
    bool have_stmt = false;
    while (find_jinja_tag(seg, search, false, if_tag)) {
        if (if_tag.content.rfind("if ", 0) == 0 ||
            if_tag.content.rfind("for ", 0) == 0 ||
            if_tag.content.rfind("set ", 0) == 0) {
            have_stmt = true;
            break;
        }
        search = if_tag.end;
    }
    if (!have_stmt) {
        return render_jinja_vars(seg, msg, add_gen, tok, eff, loop_idx, loop_total, env);
    }

    // {% set ... %}: apply the assignment to env (unsupported statements
    // leave env unchanged), then render the rest with the new state.
    if (if_tag.content.rfind("set ", 0) == 0) {
        jinja_apply_set(if_tag.content, msg, eff, loop_idx, loop_total, env);
        std::string before = seg.substr(0, if_tag.start);
        if (if_tag.trim_before) before = jinja_rstrip_ws(before);
        std::string rest = seg.substr(if_tag.end);
        if (if_tag.trim_after) rest = jinja_lstrip_ws(rest);
        return render_jinja_vars(before, msg, add_gen, tok, eff, loop_idx, loop_total, env) +
               render_jinja_segment(rest, msg, add_gen, tok, eff, loop_idx, loop_total, env);
    }

    // {% for VAR in ITER %} ... {% endfor %}
    if (if_tag.content.rfind("for ", 0) == 0) {
        std::string decl = tj_trim(if_tag.content.substr(4));
        size_t in_pos = decl.find(" in ");
        std::string iter = (in_pos == std::string::npos) ? "" : tj_trim(decl.substr(in_pos + 4));
        // Find the matching {% endfor %} with nesting-aware depth counting.
        int depth = 1;
        size_t pos = if_tag.end;
        size_t end_start = std::string::npos, end_end = std::string::npos;
        JinjaTag t;
        JinjaTag end_tag;
        while (find_jinja_tag(seg, pos, false, t)) {
            if (t.content.rfind("for ", 0) == 0) {
                ++depth;
            } else if (t.content == "endfor") {
                --depth;
                if (depth == 0) { end_start = t.start; end_end = t.end; end_tag = t; break; }
            }
            pos = t.end;
        }
        std::string before = seg.substr(0, if_tag.start);
        if (if_tag.trim_before) before = jinja_rstrip_ws(before);
        before = render_jinja_vars(before, msg, add_gen, tok, eff, loop_idx, loop_total, env);
        std::string after;
        std::string loop_out;
        if (end_start == std::string::npos) {
            // Unbalanced template: treat the rest as literal text.
            std::string rest = seg.substr(if_tag.end);
            if (if_tag.trim_after) rest = jinja_lstrip_ws(rest);
            after = render_jinja_vars(rest, msg, add_gen, tok, eff, loop_idx, loop_total, env);
            return before + after;
        }
        {
            std::string rest = seg.substr(end_end);
            if (end_tag.trim_after) rest = jinja_lstrip_ws(rest);
            after = render_jinja_segment(rest, msg, add_gen, tok, eff, loop_idx, loop_total, env);
        }
        // Resolve the iterable: "messages", a slice ("messages[::-1]"), or
        // a {% set %} list alias ("loop_messages"). {% for tool in tools %}
        // iterates the request's tool definitions instead (the loop variable
        // is bound by index in the per-iteration env copy). Anything else
        // (message.tool_calls) is empty here.
        std::vector<ChatMessage> iter_owned;
        const std::vector<ChatMessage>* iter_list = nullptr;
        if (jinja_resolve_list(iter, eff, env, iter_owned)) {
            iter_list = &iter_owned;
        }
        if (iter_list != nullptr) {
            std::string body = seg.substr(if_tag.end, end_start - if_tag.end);
            if (if_tag.trim_after) body = jinja_lstrip_ws(body);
            if (end_tag.trim_before) body = jinja_rstrip_ws(body);
            for (size_t i = 0; i < iter_list->size(); ++i) {
                // Per-iteration env copy: plain {% set %} vars are scoped
                // to the iteration (Jinja2); namespace attrs are shared
                // (the shared_ptr in JinjaEnv), so ns.index accumulates.
                JinjaEnv iter_env = env;
                loop_out += render_jinja_segment(body, &(*iter_list)[i], add_gen, tok, eff,
                                                 static_cast<int>(i),
                                                 static_cast<int>(iter_list->size()), iter_env);
            }
        } else if (iter == "tools") {
            // The loop variable name (decl before " in ") is expected to be
            // "tool"; any other name still binds the same way, since the
            // body can only address the current tool as `tool`.
            std::string body = seg.substr(if_tag.end, end_start - if_tag.end);
            if (if_tag.trim_after) body = jinja_lstrip_ws(body);
            if (end_tag.trim_before) body = jinja_rstrip_ws(body);
            for (size_t i = 0; i < env.tools.size(); ++i) {
                JinjaEnv iter_env = env;
                iter_env.cur_tool_idx = static_cast<int>(i);
                loop_out += render_jinja_segment(body, nullptr, add_gen, tok, eff,
                                                 static_cast<int>(i),
                                                 static_cast<int>(env.tools.size()), iter_env);
            }
        }
        return before + loop_out + after;
    }

    if (if_tag.content.rfind("if ", 0) == 0) {
        // Find matching {% endif %} accounting for nesting, and collect the
        // top-level {% elif %}/{% else %} branch ranges.
        struct Branch {
            std::string cond;
            bool is_else = false;
            size_t body_start = 0;
            size_t body_end = 0;
            bool trim_after_open = false;  // opening tag had -%}
            bool trim_before_close = false; // closing tag had {%- 
        };
        std::vector<Branch> branches;
        branches.push_back({if_tag.content.substr(3), false, if_tag.end, std::string::npos,
                            if_tag.trim_after, false});
        int depth = 1;
        size_t pos = if_tag.end;
        size_t endif_start = std::string::npos, endif_end = std::string::npos;
        JinjaTag t;
        JinjaTag endif_tag;
        while (find_jinja_tag(seg, pos, false, t)) {
            if (t.content.rfind("if ", 0) == 0) {
                ++depth;
            } else if (t.content == "endif") {
                --depth;
                if (depth == 0) {
                    branches.back().body_end = t.start;
                    branches.back().trim_before_close = t.trim_before;
                    endif_start = t.start; endif_end = t.end; endif_tag = t;
                    break;
                }
            } else if (depth == 1 && t.content.rfind("elif ", 0) == 0) {
                branches.back().body_end = t.start;
                branches.back().trim_before_close = t.trim_before;
                branches.push_back({t.content.substr(5), false, t.end, std::string::npos,
                                    t.trim_after, false});
            } else if (depth == 1 && t.content == "else") {
                branches.back().body_end = t.start;
                branches.back().trim_before_close = t.trim_before;
                branches.push_back({"", true, t.end, std::string::npos,
                                    t.trim_after, false});
            }
            pos = t.end;
        }
        if (endif_start == std::string::npos) {
            // Unbalanced template: drop the if-tag and render the rest.
            return render_jinja_segment(seg.substr(0, if_tag.start) + seg.substr(if_tag.end),
                                        msg, add_gen, tok, eff, loop_idx, loop_total, env);
        }
        // Evaluate branches in order; the first true one wins, "else" always wins.
        std::string chosen;
        bool have_choice = false;
        for (const auto& b : branches) {
            bool take = b.is_else ||
                        eval_jinja_cond(b.cond, msg, add_gen, eff, loop_idx, loop_total, env);
            if (take) {
                std::string body = seg.substr(b.body_start, b.body_end - b.body_start);
                if (b.trim_after_open) body = jinja_lstrip_ws(body);
                if (b.trim_before_close) body = jinja_rstrip_ws(body);
                chosen = render_jinja_segment(body, msg, add_gen, tok, eff, loop_idx, loop_total, env);
                have_choice = true;
                break;
            }
        }
        std::string before = seg.substr(0, if_tag.start);
        if (if_tag.trim_before) before = jinja_rstrip_ws(before);
        before = render_jinja_segment(before, msg, add_gen, tok, eff, loop_idx, loop_total, env);
        std::string after = seg.substr(endif_end);
        if (endif_tag.trim_after) after = jinja_lstrip_ws(after);
        after = render_jinja_segment(after, msg, add_gen, tok, eff, loop_idx, loop_total, env);
        return before + (have_choice ? chosen : "") + after;
    }

    // Unreachable: have_stmt guarantees if/for/set, all handled above.
    return render_jinja_vars(seg, msg, add_gen, tok, eff, loop_idx, loop_total, env);
}

static std::string render_chat_template_jinja(const std::string& tmpl,
                                              const std::vector<ChatMessage>& msgs,
                                              bool add_gen, const Tokenizer& tok,
                                              const std::vector<ToolDefinition>* tools = nullptr) {
    // The unified recursive renderer handles the whole template: the
    // leading {% if tools %}...{% else %}...{% endif %}, the
    // {% for message in messages %} loop (with its nested tool_call loop,
    // which renders zero iterations — tool_calls are never passed in a
    // request), and the trailing {% if add_generation_prompt %}. Jinja
    // comments are stripped first (they would otherwise leak verbatim into
    // the prompt), and a fresh {% set %} environment is created per render.
    JinjaEnv env;
    if (tools != nullptr) {
        env.tools = *tools;
        env.tools_present = true;
    }
    return render_jinja_segment(jinja_strip_comments(tmpl), nullptr, add_gen, tok, msgs, -1,
                                static_cast<int>(msgs.size()), env);
}

} // namespace

std::string format_chat_messages(
    const std::vector<ChatMessage>& msgs,
    const std::string& arch,
    const std::string& chat_template,
    const Tokenizer& tok,
    bool add_generation_prompt,
    std::string* source_out,
    const std::vector<ToolDefinition>* tools
) {
    if (!chat_template.empty()) {
        if (source_out) *source_out = "gguf-chat-template";
        return render_chat_template_jinja(chat_template, msgs, add_generation_prompt, tok, tools);
    }

    // --- Per-architecture fallback templates (no chat_template in the GGUF) ---
    // With tools, a Qwen-style <tools> system block is injected ahead of the
    // turns (merged into an existing leading system message when there is
    // one), so tool-capable models still see the definitions and the
    // <tool_call> instruction even without a GGUF template.
    std::vector<ChatMessage> eff_msgs = msgs;
    if (tools != nullptr && !tools->empty()) {
        std::ostringstream tb;
        tb << "You may call one or more functions to assist with the user query.\n\n"
           << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : *tools) tb << "\n" << t.raw_json;
        tb << "\n</tools>\n\nFor each function call, return a json object with function name "
           << "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
           << "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        if (!eff_msgs.empty() && eff_msgs.front().role == "system") {
            eff_msgs.front().content += "\n\n" + tb.str();
        } else {
            eff_msgs.insert(eff_msgs.begin(), {"system", tb.str()});
        }
    }
    const std::vector<ChatMessage>& use_msgs = eff_msgs;
    std::string name;
    std::ostringstream out;
    auto is_llama = (arch == "llama" || arch == "llama3" || arch == "llama4");
    auto is_gemma = (arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4");
    auto is_mistral = (arch == "mistral" || arch == "mixtral");
    auto is_phi = (arch == "phi" || arch == "phi3" || arch == "phi4");

    if (is_llama) {
        name = "llama3";
        for (const auto& m : use_msgs) {
            out << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n"
                << m.content << "<|eot_id|>";
        }
        if (add_generation_prompt) out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (is_mistral) {
        name = "mistral";
        std::string sys;
        for (const auto& m : use_msgs) {
            if (m.role == "system") { sys = m.content; continue; }
            if (m.role == "user") {
                out << "[INST] ";
                if (!sys.empty()) out << "<<SYS>>\n" << sys << "\n<</SYS>>\n\n";
                sys.clear();
                out << m.content << " [/INST]";
            } else if (m.role == "assistant") {
                out << " " << m.content << "</s>";
            } else {
                out << "[INST] " << m.content << " [/INST]";
            }
        }
        // Generation prompt is the trailing [/INST] when the last turn is a
        // user turn; nothing extra is needed.
    } else if (is_gemma) {
        name = "gemma";
        for (const auto& m : use_msgs) {
            std::string role = (m.role == "assistant") ? "model" : m.role;
            out << "<start_of_turn>" << role << "\n" << m.content << "<end_of_turn>\n";
        }
        if (add_generation_prompt) out << "<start_of_turn>model\n";
    } else if (is_phi) {
        name = (arch == "phi") ? "phi" : "phi3";
        if (arch == "phi") {
            for (const auto& m : use_msgs) {
                if (m.role == "user") out << "<|user|>\n" << m.content << "\n";
                else if (m.role == "assistant") out << "<|assistant|>\n" << m.content << "\n";
                else out << "<|user|>\n" << m.content << "\n";
            }
            if (add_generation_prompt) out << "<|assistant|>\n";
        } else {
            for (const auto& m : use_msgs) {
                if (m.role == "system") out << "<|system|>\n" << m.content << "<|end|>\n";
                else if (m.role == "user") out << "<|user|>\n" << m.content << "<|end|>\n";
                else if (m.role == "assistant") out << "<|assistant|>\n" << m.content << "<|end|>\n";
                else out << "<|user|>\n" << m.content << "<|end|>\n";
            }
            if (add_generation_prompt) out << "<|assistant|>\n";
        }
    } else {
        // ChatML: Qwen2/3, DeepSeek, OLMoE, SmolLM, GPT-2 family, and the
        // default for any unknown architecture.
        name = "chatml";
        for (const auto& m : use_msgs) {
            out << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
        }
        if (add_generation_prompt) out << "<|im_start|>assistant\n";
    }

    if (source_out) *source_out = "fallback:" + name;
    return out.str();
}

std::string wrap_raw_prompt_for_chat(
    const std::string& prompt,
    const std::string& arch,
    const std::string& chat_template,
    const Tokenizer& tok
) {
    static const char* kMarkers[] = {
        "<|im_start|>", "<|user|>", "<|system|>", "<|turn>",
        "<|start_header_id|>", "[INST]", "<start_of_turn>",
        "<|begin_of_text|>", "<|assistant|>"
    };
    for (const char* m : kMarkers) {
        if (prompt.find(m) != std::string::npos) return prompt;
    }
    return format_chat_messages({{"user", prompt}}, arch, chat_template, tok, true, nullptr);
}

void MoETransformerEngine::softmax(float* x, int64_t size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int64_t i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : (1.0f / static_cast<float>(size));
    for (int64_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

bool MoETransformerEngine::load_model(const std::string& path, std::string& err_msg) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    loaded_ = false;
    hw_ = query_hardware_specs();

    if (!model_file_.open(path, err_msg)) {
        return false;
    }

    if (!tokenizer_.load_from_model(model_file_)) {
        err_msg = "Failed to initialize tokenizer from model";
        return false;
    }

    if (!io_engine_.init({path}, err_msg)) {
        return false;
    }

    if (!bind_tensors(err_msg)) {
        return false;
    }

    // Initialize LFRU tier cache: keep ~50% of experts eligible for RAM/VRAM residency (or 100% for dense 1-expert models)
    size_t total_experts = static_cast<size_t>(hp_.n_layers) * static_cast<size_t>(hp_.n_experts);
    size_t vram_slots = (hp_.n_experts <= 1) ? total_experts : std::max<size_t>(4, total_experts / 4);
    size_t ram_slots  = (hp_.n_experts <= 1) ? total_experts : std::max<size_t>(4, total_experts / 4);
    tier_mgr_.init(hp_.n_layers, hp_.n_experts, ram_slots, vram_slots, 0.08f);

    kv_slots_.clear();
    KVCacheSlot slot0;
    slot0.init(0, hp_);
    kv_slots_[0] = std::move(slot0);

    loaded_ = true;
    return true;
}

bool MoETransformerEngine::bind_tensors(std::string& err_msg) {
    // --- Workstream C: architecture validation ---
    // A missing general.architecture used to silently default to
    // "qwen2moe" and load whatever tensors happened to be present, so a
    // mislabeled or non-transformer file produced plausible-looking
    // garbage instead of an error.
    if (model_file_.metadata().find("general.architecture") == model_file_.metadata().end()) {
        err_msg = "Model metadata is missing general.architecture — cannot determine the architecture (refusing to guess)";
        return false;
    }
    hp_.arch = model_file_.get_str("general.architecture", "qwen2moe");
    hp_.model_name = model_file_.get_str("general.name", "MoE-Stream-Model");
    const std::string pfx = hp_.arch;

    if (!arch_is_known(hp_.arch)) {
        err_msg = "Unsupported architecture '" + hp_.arch + "' — this engine only runs decoder "
                  "transformers (see docs/SUPPORT_MATRIX.md for the tested list)";
        return false;
    }
    // Known GGUF archs this decoder-only engine cannot run: mamba is a
    // state-space model (no attention), bert/nomic-bert are encoders (no
    // causal LM head). Loading one used to "succeed" and generate noise.
    if (hp_.arch == "mamba" || hp_.arch == "bert" || hp_.arch == "nomic-bert") {
        err_msg = "Architecture '" + hp_.arch + "' is not a causal decoder transformer — "
                  "this engine cannot run it (see docs/SUPPORT_MATRIX.md)";
        return false;
    }
    // Known but never chat-verified against a real model in this engine:
    // load, but say so loudly. (Verified: qwen2, llama, qwen3, gemma3,
    // phi3, mistral. True deepseek/deepseek2 arch is UNVERIFIED —
    // DeepSeek-LLM-7B reports arch "llama" and runs as Llama.)
    {
        static const char* kVerified[] = {"qwen2", "qwen2moe", "llama", "qwen3", "gemma3", "phi3", "mistral"};
        bool verified = false;
        for (const char* v : kVerified) if (hp_.arch == v) { verified = true; break; }
        if (!verified) {
            std::cerr << "[BIND] WARNING: architecture '" << hp_.arch << "' is known but "
                      << "UNVERIFIED in this engine — output correctness has not been "
                      << "checked against a real model of this arch (see docs/SUPPORT_MATRIX.md).\n";
        }
    }

    hp_.vocab_size     = tokenizer_.vocab_size();
    hp_.dim            = model_file_.get_u32(pfx + ".embedding_length", model_file_.get_u32("llama.embedding_length", 128));
    hp_.hidden_dim     = model_file_.get_u32(pfx + ".feed_forward_length", 256);
    hp_.moe_expert_dim = model_file_.get_u32(pfx + ".expert_feed_forward_length", hp_.hidden_dim);
    hp_.n_layers       = model_file_.get_u32(pfx + ".block_count", model_file_.get_u32("llama.block_count", 4));
    hp_.n_heads        = model_file_.get_u32(pfx + ".attention.head_count", 4);
    hp_.n_kv_heads     = model_file_.get_u32(pfx + ".attention.head_count_kv", hp_.n_heads);
    hp_.n_experts      = model_file_.get_u32(pfx + ".expert_count", 8);
    hp_.n_experts_used = model_file_.get_u32(pfx + ".expert_used_count", 2);
    hp_.max_seq_len    = model_file_.get_u32(pfx + ".context_length", 2048);
    // kv_cache_capacity is computed below, once kv_dim is final (it needs a
    // RAM guard; see after rope_dim).
    hp_.rms_norm_eps   = model_file_.get_f32(pfx + ".attention.layer_norm_rms_epsilon", 1e-5f);
    hp_.rope_freq_base = model_file_.get_f32(pfx + ".rope.freq_base", 10000.0f);
    hp_.rope_freq_base_swa = model_file_.get_f32(pfx + ".rope.freq_base_swa", hp_.rope_freq_base);
    hp_.sliding_window = model_file_.get_u32(pfx + ".attention.sliding_window", 0);
    hp_.shared_kv_layers = model_file_.get_u32(pfx + ".attention.shared_kv_layers", 0);
    hp_.ple_dim        = model_file_.get_u32(pfx + ".embedding_length_per_layer_input", 0);
    hp_.rope_dim_swa   = model_file_.get_u32(pfx + ".rope.dimension_count_swa", 0);
    hp_.final_logit_softcapping = model_file_.get_f32(pfx + ".final_logit_softcapping", 0.0f);
    hp_.attn_logit_softcapping = model_file_.get_f32(pfx + ".attn_logit_softcapping", 0.0f);
    // RoPE scaling (GGUF: <arch>.rope.scaling.*). Only "yarn" sets yarn_scale;
    // "linear"/"none"/absent leave it at 1.0 (apply_rope divides pos by it).
    hp_.rope_scaling_type = model_file_.get_str(pfx + ".rope.scaling.type", "");
    hp_.rope_scaling_factor = model_file_.get_f32(pfx + ".rope.scaling.factor", 1.0f);
    hp_.yarn_log_multiplier = model_file_.get_f32(pfx + ".rope.scaling.yarn_log_multiplier", 0.0f);
    hp_.rope_orig_ctx_len = model_file_.get_u32(pfx + ".rope.scaling.original_context_length", 0);
    if (hp_.rope_scaling_type == "yarn" && hp_.rope_scaling_factor > 0.0f) {
        hp_.yarn_scale = hp_.rope_scaling_factor;
    } else {
        hp_.yarn_scale = 1.0f;
    }
    hp_.rope_neox = arch_uses_neox_rope(hp_.arch);

    // --- Workstream C: hyperparameter range validation ---
    // Metadata values come from the file (or from silent defaults above);
    // an out-of-range value used to flow into layers_.resize(),
    // KVCacheSlot::init(), or the KV-capacity estimate and either throw
    // bad_alloc or silently allocate a wrong-sized cache. Reject before
    // anything is allocated. Bounds are generous (far above any real
    // model) — they only catch corrupt/ hostile metadata.
    {
        auto bad = [&](const char* what, const std::string& detail) {
            err_msg = std::string("Invalid hyperparameter ") + what + ": " + detail;
            return false;
        };
        if (hp_.dim == 0 || hp_.dim > 65536) return bad("embedding_length", std::to_string(hp_.dim));
        if (hp_.hidden_dim == 0 || hp_.hidden_dim > (1u << 20)) return bad("feed_forward_length", std::to_string(hp_.hidden_dim));
        if (hp_.moe_expert_dim == 0 || hp_.moe_expert_dim > (1u << 20)) return bad("expert_feed_forward_length", std::to_string(hp_.moe_expert_dim));
        if (hp_.n_layers == 0 || hp_.n_layers > 512) return bad("block_count", std::to_string(hp_.n_layers));
        if (hp_.n_heads == 0 || hp_.n_heads > 256) return bad("attention.head_count", std::to_string(hp_.n_heads));
        if (hp_.n_kv_heads == 0 || hp_.n_kv_heads > hp_.n_heads) return bad("attention.head_count_kv", std::to_string(hp_.n_kv_heads) + " (must be 1.." + std::to_string(hp_.n_heads) + ")");
        if (hp_.n_experts == 0 || hp_.n_experts > 1024) return bad("expert_count", std::to_string(hp_.n_experts));
        if (hp_.n_experts_used == 0 || hp_.n_experts_used > hp_.n_experts) return bad("expert_used_count", std::to_string(hp_.n_experts_used) + " (must be 1.." + std::to_string(hp_.n_experts) + ")");
        if (hp_.max_seq_len == 0 || hp_.max_seq_len > (1u << 20)) return bad("context_length", std::to_string(hp_.max_seq_len));
        if (!(hp_.rms_norm_eps > 0.0f) || !std::isfinite(hp_.rms_norm_eps)) return bad("attention.layer_norm_rms_epsilon", "must be finite and > 0");
        if (!(hp_.rope_freq_base > 0.0f) || !std::isfinite(hp_.rope_freq_base)) return bad("rope.freq_base", "must be finite and > 0");
        if (!std::isfinite(hp_.final_logit_softcapping) || !std::isfinite(hp_.attn_logit_softcapping)) return bad("logit softcapping", "must be finite");
        if (hp_.ple_dim > 65536) return bad("embedding_length_per_layer_input", std::to_string(hp_.ple_dim));
    }

    bool is_gemma_family = (hp_.arch == "gemma" || hp_.arch == "gemma2" || hp_.arch == "gemma3" || hp_.arch == "gemma4");
    hp_.use_gelu = is_gemma_family;
    hp_.scale_embeddings = is_gemma_family;
    hp_.v_norm = (hp_.arch == "gemma4");

    tok_emb_ = model_file_.find_tensor("token_embd.weight");
    if (!tok_emb_) tok_emb_ = model_file_.find_tensor("model.embed_tokens.weight");
    if (!tok_emb_) {
        err_msg = "Missing token_embd.weight tensor in model file";
        return false;
    }

    if (tok_emb_->n_dims >= 2) {
        hp_.dim = static_cast<uint32_t>(tok_emb_->dims[0]);
        hp_.vocab_size = static_cast<uint32_t>(tok_emb_->dims[1]);
    }
    if (!validate_bound_tensor(tok_emb_, "token_embd.weight", 0, err_msg)) return false;
    if (tok_emb_->n_dims != 2 || tok_emb_->dims[0] == 0 || tok_emb_->dims[1] == 0) {
        err_msg = "token_embd.weight must be a 2D tensor [dim, vocab]";
        return false;
    }
    // Re-check dim AFTER the embedding override: the metadata value was
    // validated above, but the embedding dims win and a huge dim here
    // would truncate in the uint32 cast and mis-size every activation.
    if (tok_emb_->dims[0] > 65536 || tok_emb_->dims[1] > (1u << 20)) {
        err_msg = "token_embd.weight dims out of range: [" + std::to_string(tok_emb_->dims[0]) +
                  ", " + std::to_string(tok_emb_->dims[1]) + "]";
        return false;
    }
    // The embedding table defines the valid token-id range. A tokenizer
    // larger than the table used to let encode() produce ids that read
    // past the embedding (forward_token guarded with token_id <
    // vocab_size and silently produced a zero embedding instead).
    if (hp_.vocab_size == 0 || hp_.vocab_size > (1u << 20)) {
        err_msg = "Invalid vocabulary size " + std::to_string(hp_.vocab_size);
        return false;
    }
    if (tokenizer_.vocab_size() > hp_.vocab_size) {
        err_msg = "Tokenizer vocabulary (" + std::to_string(tokenizer_.vocab_size()) +
                  ") is larger than the embedding table (" + std::to_string(hp_.vocab_size) +
                  " rows) — token ids from the tokenizer would read past the embedding";
        return false;
    }
    if (tokenizer_.vocab_size() < hp_.vocab_size) {
        std::cerr << "[BIND] WARNING: tokenizer vocabulary (" << tokenizer_.vocab_size()
                  << ") is smaller than the embedding table (" << hp_.vocab_size
                  << " rows) — the extra rows are unreachable.\n";
    }
    hp_.head_dim = model_file_.get_u32(
        pfx + ".attention.key_length",
        std::max(1u, hp_.dim / std::max(1u, hp_.n_heads))
    );
    hp_.q_dim  = hp_.n_heads * hp_.head_dim;
    hp_.kv_dim = hp_.n_kv_heads * hp_.head_dim;
    if (const auto* wq0 = model_file_.find_tensor("blk.0.attn_q.weight")) {
        if (wq0->n_dims >= 2 && wq0->dims[1] > 0) {
            hp_.q_dim = static_cast<uint32_t>(wq0->dims[1]);
            hp_.head_dim = std::max(1u, hp_.q_dim / std::max(1u, hp_.n_heads));
        }
    }
    if (const auto* wk0 = model_file_.find_tensor("blk.0.attn_k.weight")) {
        if (wk0->n_dims >= 2 && wk0->dims[1] > 0) {
            hp_.kv_dim = static_cast<uint32_t>(wk0->dims[1]);
        }
    }
    // q_dim/kv_dim must split evenly across heads: head_dim is recomputed
    // by integer division, so a non-divisible q_dim used to silently
    // truncate and the attention loop then read stale values from a
    // previous layer's buffer (in-bounds of the buffer, wrong output).
    if (hp_.q_dim % hp_.n_heads != 0 || hp_.kv_dim % hp_.n_kv_heads != 0) {
        err_msg = "Attention dims do not split evenly across heads: q_dim=" +
                  std::to_string(hp_.q_dim) + " / n_heads=" + std::to_string(hp_.n_heads) +
                  ", kv_dim=" + std::to_string(hp_.kv_dim) + " / n_kv_heads=" +
                  std::to_string(hp_.n_kv_heads);
        return false;
    }
    hp_.rope_dim = model_file_.get_u32(pfx + ".rope.dimension_count", hp_.head_dim);

    // KV cache capacity: up to min(max_seq_len, 32768), but never let the
    // FP16 K+V cache (n_layers * capacity * kv_dim * 2 arrays * 2 bytes)
    // exceed 25% of available RAM. Halve until it fits (floor 256 tokens).
    // (n_layers/kv_dim are range-validated above, so the estimate below
    // cannot overflow uint64.)
    {
        uint32_t cap = std::min<uint32_t>(hp_.max_seq_len, 32768);
        if (cap == 0) cap = 2048;
        uint64_t avail = hw_.avail_ram_bytes;
        if (avail == 0) {
            // query_hardware_specs() could not determine RAM: assume a
            // conservative 2 GiB instead of skipping the guard entirely
            // (the old code allocated the full 32768-token cache, which
            // is ~10 GB for an 80-layer kv_dim-1024 model).
            std::cerr << "[BIND] WARNING: available RAM unknown — assuming 2 GiB for the KV-cache budget.\n";
            avail = 2ull << 30;
        }
        if (hp_.n_layers > 0 && hp_.kv_dim > 0) {
            // Large models leave less headroom for page cache + activations —
            // keep KV under 20% of RAM when the dense backbone exceeds 1.5 GiB.
            const uint64_t dense_est = model_file_.file_size();
            uint64_t budget = (dense_est >= (1536ull << 20)) ? (avail / 5) : (avail / 4);
            while (cap > 256) {
                uint64_t est = static_cast<uint64_t>(hp_.n_layers) * cap *
                               static_cast<uint64_t>(hp_.kv_dim) * 2ull * 2ull;
                if (est <= budget) break;
                cap /= 2;
            }
        }
        hp_.kv_cache_capacity = cap;
    }

    out_norm_ = model_file_.find_tensor("output_norm.weight");
    lm_head_  = model_file_.find_tensor("output.weight");
    if (!lm_head_) lm_head_ = tok_emb_; // weight-tied fallback
    // Output norm / LM head: the LM-head matmul trusts vocab_size x dim.
    if (!validate_bound_tensor(out_norm_, "output_norm.weight", hp_.dim, err_msg)) return false;
    if (out_norm_ && out_norm_->type != GGMLType::F32) {
        std::cerr << "[BIND] WARNING: output_norm.weight is " << ggml_type_name(out_norm_->type)
                  << ", not F32 — forward ignores non-F32 norms.\n";
    }
    if (lm_head_ && lm_head_ != tok_emb_) {
        if (!validate_bound_tensor(lm_head_, "output.weight", 0, err_msg)) return false;
        if (lm_head_->n_dims != 2 || lm_head_->dims[0] != hp_.dim ||
            lm_head_->dims[1] != hp_.vocab_size) {
            err_msg = "output.weight must be [dim, vocab]";
            return false;
        }
    }
    per_layer_tok_emb_    = model_file_.find_tensor("per_layer_token_embd.weight");
    per_layer_model_proj_ = model_file_.find_tensor("per_layer_model_proj.weight");
    per_layer_proj_norm_  = model_file_.find_tensor("per_layer_proj_norm.weight");
    rope_freqs_           = model_file_.find_tensor("rope_freqs.weight");
    // Gemma4 per-layer-embedding tensors: byte consistency always; the
    // proj-norm length only matters when PLE is configured (ple_dim > 0 —
    // when it is 0 the expected count is 0, i.e. "any").
    if (!validate_bound_tensor(per_layer_tok_emb_, "per_layer_token_embd.weight", 0, err_msg) ||
        !validate_bound_tensor(per_layer_model_proj_, "per_layer_model_proj.weight", 0, err_msg) ||
        !validate_bound_tensor(per_layer_proj_norm_, "per_layer_proj_norm.weight", hp_.ple_dim, err_msg) ||
        !validate_bound_tensor(rope_freqs_, "rope_freqs.weight", 0, err_msg)) {
        return false;
    }

    std::vector<int32_t> swa_pattern;
    auto swa_it = model_file_.metadata().find(pfx + ".attention.sliding_window_pattern");
    if (swa_it != model_file_.metadata().end()) {
        swa_pattern = swa_it->second.i32_arr;
    }

    dense_bytes_ = 0;
    sparse_bytes_ = 0;
    for (const auto& kv : model_file_.tensors()) {
        if (kv.second.is_sparse_expert) sparse_bytes_ += kv.second.byte_size;
        else dense_bytes_ += kv.second.byte_size;
    }

    uint32_t last_swa_kv_layer = 0;
    uint32_t last_full_kv_layer = 0;
    // Workstream C: MoE layers must agree on the expert count (dense
    // layers in a mixed model, e.g. DeepSeek's first layer, do not affect
    // this — only MoE layers are tracked).
    uint32_t moe_experts_seen = 0;
    const uint32_t first_shared_kv = (hp_.shared_kv_layers > 0 && hp_.n_layers > hp_.shared_kv_layers)
                                   ? (hp_.n_layers - hp_.shared_kv_layers) : hp_.n_layers;

    layers_.resize(hp_.n_layers);
    for (uint32_t l = 0; l < hp_.n_layers; ++l) {
        std::string bp = "blk." + std::to_string(l) + ".";
        LayerDenseWeights& lw = layers_[l];
        lw.attn_norm   = model_file_.find_tensor(bp + "attn_norm.weight");
        lw.wq          = model_file_.find_tensor(bp + "attn_q.weight");
        lw.wk          = model_file_.find_tensor(bp + "attn_k.weight");
        lw.wv          = model_file_.find_tensor(bp + "attn_v.weight");
        lw.bq          = model_file_.find_tensor(bp + "attn_q.bias");
        lw.bk          = model_file_.find_tensor(bp + "attn_k.bias");
        lw.bv          = model_file_.find_tensor(bp + "attn_v.bias");
        // Fused SafeTensors attention (Phi3 qkv_proj): when the split Q/K/V
        // tensors are absent but the fused blk.<l>.attn_qkv.weight exists,
        // split it into Q/K/V views (head counts give q_dim/kv_dim). The
        // views share the fused tensor's mmap — no data is copied.
        if ((!lw.wq || !lw.wk || !lw.wv) &&
            model_file_.find_tensor(bp + "attn_qkv.weight")) {
            std::string split_err;
            uint32_t q_dim = hp_.n_heads * hp_.head_dim;
            uint32_t kv_dim = hp_.n_kv_heads * hp_.head_dim;
            if (!model_file_.split_fused_qkv(l, q_dim, kv_dim, split_err)) {
                err_msg = "Layer " + std::to_string(l) + ": cannot split fused qkv_proj: " + split_err;
                return false;
            }
            lw.wq = model_file_.find_tensor(bp + "attn_q.weight");
            lw.wk = model_file_.find_tensor(bp + "attn_k.weight");
            lw.wv = model_file_.find_tensor(bp + "attn_v.weight");
            lw.bq = model_file_.find_tensor(bp + "attn_q.bias");
            lw.bk = model_file_.find_tensor(bp + "attn_k.bias");
            lw.bv = model_file_.find_tensor(bp + "attn_v.bias");
        }
        lw.attn_q_norm = model_file_.find_tensor(bp + "attn_q_norm.weight");
        lw.attn_k_norm = model_file_.find_tensor(bp + "attn_k_norm.weight");
        lw.wo          = model_file_.find_tensor(bp + "attn_output.weight");
        lw.bo          = model_file_.find_tensor(bp + "attn_output.bias");
        lw.post_attn_norm = model_file_.find_tensor(bp + "post_attention_norm.weight");
        lw.ffn_norm    = model_file_.find_tensor(bp + "ffn_norm.weight");
        lw.post_ffw_norm  = model_file_.find_tensor(bp + "post_ffw_norm.weight");
        lw.inp_gate       = model_file_.find_tensor(bp + "inp_gate.weight");
        lw.ple_proj       = model_file_.find_tensor(bp + "proj.weight");
        lw.post_norm      = model_file_.find_tensor(bp + "post_norm.weight");
        lw.layer_output_scale = model_file_.find_tensor(bp + "layer_output_scale.weight");
        lw.router_gate = model_file_.find_tensor(bp + "ffn_gate_inp.weight");

        lw.q_dim = (lw.wq && lw.wq->n_dims >= 2 && lw.wq->dims[1] > 0)
                 ? static_cast<uint32_t>(lw.wq->dims[1]) : hp_.q_dim;
        lw.kv_dim = (lw.wk && lw.wk->n_dims >= 2 && lw.wk->dims[1] > 0)
                  ? static_cast<uint32_t>(lw.wk->dims[1]) : hp_.kv_dim;
        lw.head_dim = std::max(1u, lw.q_dim / std::max(1u, hp_.n_heads));
        lw.is_swa = (l < swa_pattern.size()) ? (swa_pattern[l] != 0)
                  : (hp_.sliding_window > 0 && lw.head_dim < hp_.head_dim);
        lw.rope_freq_base = lw.is_swa ? hp_.rope_freq_base_swa : hp_.rope_freq_base;
        lw.rope_dim = (lw.is_swa && hp_.rope_dim_swa > 0) ? hp_.rope_dim_swa
                    : std::min(lw.head_dim, hp_.rope_dim > 0 ? hp_.rope_dim : lw.head_dim);

        // --- Workstream C: per-layer attention/norm/bias validation ---
        // A missing attention weight used to reach matmul_quant_f32 as a
        // null pointer (the kernels do not null-check) — a segfault. A
        // short/wrong-sized bias used to be read out of bounds by the
        // bias-add loops in forward_token. Non-F32 norms/biases are
        // silently IGNORED by forward (it only applies F32 weights), so
        // warn: an all-BF16 SafeTensors model runs with identity norms.
        if (!lw.wq || !lw.wk || !lw.wv || !lw.wo) {
            err_msg = "Layer " + std::to_string(l) + ": missing attention weight tensor(s):" +
                      (!lw.wq ? " attn_q" : "") + (!lw.wk ? " attn_k" : "") +
                      (!lw.wv ? " attn_v" : "") + (!lw.wo ? " attn_output" : "");
            return false;
        }
        if (!validate_bound_tensor(lw.wq, "attn_q.weight", 0, err_msg) ||
            !validate_bound_tensor(lw.wk, "attn_k.weight", 0, err_msg) ||
            !validate_bound_tensor(lw.wv, "attn_v.weight", 0, err_msg) ||
            !validate_bound_tensor(lw.wo, "attn_output.weight", 0, err_msg)) {
            err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
            return false;
        }
        if (lw.q_dim % hp_.n_heads != 0 || lw.kv_dim % hp_.n_kv_heads != 0) {
            err_msg = "Layer " + std::to_string(l) + ": q_dim/kv_dim do not split evenly across heads";
            return false;
        }
        if (lw.wq->n_dims != 2 || lw.wq->dims[0] != hp_.dim || lw.wq->dims[1] != lw.q_dim ||
            lw.wk->n_dims != 2 || lw.wk->dims[0] != hp_.dim || lw.wk->dims[1] != lw.kv_dim ||
            lw.wv->n_dims != 2 || lw.wv->dims[0] != hp_.dim || lw.wv->dims[1] != lw.kv_dim ||
            lw.wo->n_dims != 2 || lw.wo->dims[0] != lw.q_dim || lw.wo->dims[1] != hp_.dim) {
            err_msg = "Layer " + std::to_string(l) + ": attention weight shapes do not match dim/q_dim/kv_dim";
            return false;
        }
        // Biases: the forward bias-add loops trust the length, so require
        // the exact element count (any shape with the right numel is OK).
        if (!validate_bound_tensor(lw.bq, "attn_q.bias", lw.q_dim, err_msg) ||
            !validate_bound_tensor(lw.bk, "attn_k.bias", lw.kv_dim, err_msg) ||
            !validate_bound_tensor(lw.bv, "attn_v.bias", lw.kv_dim, err_msg) ||
            !validate_bound_tensor(lw.bo, "attn_output.bias", hp_.dim, err_msg)) {
            err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
            return false;
        }
        // Norms: exact length; warn when not F32 (ignored by forward).
        if (!validate_bound_tensor(lw.attn_norm, "attn_norm.weight", hp_.dim, err_msg) ||
            !validate_bound_tensor(lw.ffn_norm, "ffn_norm.weight", hp_.dim, err_msg) ||
            !validate_bound_tensor(lw.post_attn_norm, "post_attention_norm.weight", hp_.dim, err_msg) ||
            !validate_bound_tensor(lw.post_ffw_norm, "post_ffw_norm.weight", hp_.dim, err_msg) ||
            !validate_bound_tensor(lw.post_norm, "post_norm.weight", hp_.dim, err_msg) ||
            !validate_bound_tensor(lw.attn_q_norm, "attn_q_norm.weight", lw.head_dim, err_msg) ||
            !validate_bound_tensor(lw.attn_k_norm, "attn_k_norm.weight", lw.head_dim, err_msg)) {
            err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
            return false;
        }
        {
            auto warn_non_f32 = [&](const TensorDescriptor* td, const char* name) {
                if (td && td->type != GGMLType::F32) {
                    std::cerr << "[BIND] WARNING: layer " << l << " " << name << " is "
                              << ggml_type_name(td->type) << ", not F32 — forward ignores "
                              << "non-F32 norms/biases, so this weight has no effect.\n";
                }
            };
            warn_non_f32(lw.attn_norm, "attn_norm.weight");
            warn_non_f32(lw.ffn_norm, "ffn_norm.weight");
            warn_non_f32(lw.post_attn_norm, "post_attention_norm.weight");
            warn_non_f32(lw.post_ffw_norm, "post_ffw_norm.weight");
            warn_non_f32(lw.post_norm, "post_norm.weight");
            warn_non_f32(lw.attn_q_norm, "attn_q_norm.weight");
            warn_non_f32(lw.attn_k_norm, "attn_k_norm.weight");
            warn_non_f32(lw.bq, "attn_q.bias");
            warn_non_f32(lw.bk, "attn_k.bias");
            warn_non_f32(lw.bv, "attn_v.bias");
            warn_non_f32(lw.bo, "attn_output.bias");
        }

        if (l < first_shared_kv) {
            lw.shares_kv = false;
            lw.kv_source_layer = l;
            if (lw.is_swa) last_swa_kv_layer = l;
            else last_full_kv_layer = l;
        } else {
            lw.shares_kv = true;
            lw.kv_source_layer = lw.is_swa ? last_swa_kv_layer : last_full_kv_layer;
        }

        hp_.q_dim  = std::max(hp_.q_dim, lw.q_dim);
        hp_.kv_dim = std::max(hp_.kv_dim, lw.kv_dim);

        // Shared expert tensors (Qwen2.5-MoE / DeepSeek-MoE)
        lw.shexp_gate     = model_file_.find_tensor(bp + "ffn_gate_shexp.weight");
        lw.shexp_up       = model_file_.find_tensor(bp + "ffn_up_shexp.weight");
        lw.shexp_down     = model_file_.find_tensor(bp + "ffn_down_shexp.weight");
        lw.shexp_gate_inp = model_file_.find_tensor(bp + "ffn_gate_inp_shexp.weight");
        // Shared experts are all-or-nothing: a partially present trio
        // used to be silently skipped, leaving the layer without the
        // shared expert every other layer has.
        {
            int shexp_present = (lw.shexp_gate ? 1 : 0) + (lw.shexp_up ? 1 : 0) +
                                (lw.shexp_down ? 1 : 0);
            if (shexp_present != 0 && shexp_present != 3) {
                err_msg = "Layer " + std::to_string(l) + ": partial shared-expert tensors "
                          "(ffn_gate/up/down_shexp must be all present or all absent)";
                return false;
            }
        }
        if (lw.shexp_gate) {
            if (!validate_bound_tensor(lw.shexp_gate, "ffn_gate_shexp.weight", 0, err_msg) ||
                !validate_bound_tensor(lw.shexp_up, "ffn_up_shexp.weight", 0, err_msg) ||
                !validate_bound_tensor(lw.shexp_down, "ffn_down_shexp.weight", 0, err_msg)) {
                err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                return false;
            }
            if (lw.shexp_gate->n_dims != 2 || lw.shexp_up->n_dims != 2 || lw.shexp_down->n_dims != 2 ||
                lw.shexp_gate->dims[0] != hp_.dim || lw.shexp_up->dims[0] != hp_.dim ||
                lw.shexp_gate->dims[1] != lw.shexp_up->dims[1] ||
                lw.shexp_down->dims[0] != lw.shexp_gate->dims[1] ||
                lw.shexp_down->dims[1] != hp_.dim) {
                err_msg = "Layer " + std::to_string(l) + ": shared-expert tensor shapes are inconsistent";
                return false;
            }
        }

        const TensorDescriptor* t_gate_exps = model_file_.find_tensor(bp + "ffn_gate_exps.weight");
        const TensorDescriptor* t_up_exps   = model_file_.find_tensor(bp + "ffn_up_exps.weight");
        const TensorDescriptor* t_down_exps = model_file_.find_tensor(bp + "ffn_down_exps.weight");

        lw.expert_slices.resize(hp_.n_experts);
        // Workstream C: every layer must end up with a working FFN (MoE
        // or dense). A layer whose experts could not be packed
        // (SafeTensors warning in gguf.cpp) and that has no dense FFN
        // used to run with a zero FFN — silent degradation, not an error.
        bool layer_has_ffn = false;
        if (t_gate_exps && t_up_exps && t_down_exps) {
            // Packed 3D expert tensors [dim, expert_dim, n_experts]
            if (t_gate_exps->n_dims >= 2) {
                hp_.moe_expert_dim = static_cast<uint32_t>(t_gate_exps->dims[1]);
            }
            if (t_gate_exps->n_dims >= 3) {
                hp_.n_experts = static_cast<uint32_t>(t_gate_exps->dims[2]);
                lw.expert_slices.resize(hp_.n_experts);
            }
            lw.expert_gate_type = t_gate_exps->type;
            lw.expert_up_type   = t_up_exps->type;
            lw.expert_down_type = t_down_exps->type;

            // --- Workstream C: expert tensor validation ---
            // A non-divisible byte size used to give every expert a
            // wrong-sized slice (the last slice ran past the tensor);
            // n_experts == 0 divided by zero below.
            if (!validate_bound_tensor(t_gate_exps, "ffn_gate_exps.weight", 0, err_msg) ||
                !validate_bound_tensor(t_up_exps, "ffn_up_exps.weight", 0, err_msg) ||
                !validate_bound_tensor(t_down_exps, "ffn_down_exps.weight", 0, err_msg)) {
                err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                return false;
            }
            if (t_gate_exps->n_dims != 3 || t_up_exps->n_dims != 3 || t_down_exps->n_dims != 3) {
                err_msg = "Layer " + std::to_string(l) + ": packed expert tensors must be 3D [dim, expert_dim, n_experts]";
                return false;
            }
            if (hp_.n_experts == 0 || hp_.n_experts > 1024 ||
                t_gate_exps->dims[1] == 0 || t_gate_exps->dims[1] > (1u << 20)) {
                err_msg = "Layer " + std::to_string(l) + ": invalid packed expert dims (n_experts=" +
                          std::to_string(hp_.n_experts) + ", expert_dim=" +
                          std::to_string(t_gate_exps->dims[1]) + ")";
                return false;
            }
            if (moe_experts_seen == 0) moe_experts_seen = hp_.n_experts;
            else if (moe_experts_seen != hp_.n_experts) {
                err_msg = "Layer " + std::to_string(l) + ": expert count " +
                          std::to_string(hp_.n_experts) + " differs from earlier MoE layers (" +
                          std::to_string(moe_experts_seen) + ")";
                return false;
            }
            if (t_gate_exps->byte_size % hp_.n_experts != 0 ||
                t_up_exps->byte_size % hp_.n_experts != 0 ||
                t_down_exps->byte_size % hp_.n_experts != 0) {
                err_msg = "Layer " + std::to_string(l) + ": packed expert tensor byte sizes are not divisible by n_experts=" +
                          std::to_string(hp_.n_experts);
                return false;
            }
            // The three projections must agree on the expert count and
            // the expert width, and on dim; otherwise slices from
            // different tensors describe different experts.
            if (t_up_exps->dims[2] != hp_.n_experts || t_down_exps->dims[2] != hp_.n_experts ||
                t_gate_exps->dims[1] != t_up_exps->dims[1] ||
                t_down_exps->dims[0] != t_gate_exps->dims[1] ||
                t_gate_exps->dims[0] != hp_.dim || t_up_exps->dims[0] != hp_.dim ||
                t_down_exps->dims[1] != hp_.dim) {
                err_msg = "Layer " + std::to_string(l) + ": packed expert tensor dims are inconsistent";
                return false;
            }
            // The router is required for MoE: without it forward routes
            // every token to expert 0 (see forward_token) — silent.
            if (!lw.router_gate) {
                err_msg = "Layer " + std::to_string(l) + ": MoE layer has packed expert tensors but no ffn_gate_inp.weight router — cannot route tokens";
                return false;
            }
            {
                uint64_t router_numel = static_cast<uint64_t>(hp_.dim) * hp_.n_experts;
                if (!validate_bound_tensor(lw.router_gate, "ffn_gate_inp.weight", router_numel, err_msg)) {
                    err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                    return false;
                }
            }
            layer_has_ffn = true;

            uint64_t gate_slice_sz = t_gate_exps->byte_size / hp_.n_experts;
            uint64_t up_slice_sz   = t_up_exps->byte_size / hp_.n_experts;
            uint64_t down_slice_sz = t_down_exps->byte_size / hp_.n_experts;

            for (uint32_t e = 0; e < hp_.n_experts; ++e) {
                lw.expert_slices[e].gate_offset = t_gate_exps->abs_file_offset + e * gate_slice_sz;
                lw.expert_slices[e].gate_bytes  = gate_slice_sz;
                lw.expert_slices[e].up_offset   = t_up_exps->abs_file_offset + e * up_slice_sz;
                lw.expert_slices[e].up_bytes    = up_slice_sz;
                lw.expert_slices[e].down_offset = t_down_exps->abs_file_offset + e * down_slice_sz;
                lw.expert_slices[e].down_bytes  = down_slice_sz;
                if (t_gate_exps->mapped_ptr) {
                    lw.expert_slices[e].gate_mapped_ptr = static_cast<const uint8_t*>(t_gate_exps->mapped_ptr) + e * gate_slice_sz;
                }
                if (t_up_exps->mapped_ptr) {
                    lw.expert_slices[e].up_mapped_ptr = static_cast<const uint8_t*>(t_up_exps->mapped_ptr) + e * up_slice_sz;
                }
                if (t_down_exps->mapped_ptr) {
                    lw.expert_slices[e].down_mapped_ptr = static_cast<const uint8_t*>(t_down_exps->mapped_ptr) + e * down_slice_sz;
                }
            }
            // Every slice must be readable: via its mmap pointer, or via
            // a file offset inside the model file. (Packed SafeTensors
            // experts are synthetic — abs_file_offset 0 with a valid
            // mapped_ptr — so neither check alone suffices. A slice that
            // fails both would make read_expert_sync read garbage or
            // short, and the old code admitted the result unchecked.)
            {
                uint64_t fsize = model_file_.file_size();
                for (uint32_t e = 0; e < hp_.n_experts; ++e) {
                    const auto& s = lw.expert_slices[e];
                    auto slice_ok = [&](const void* mp, uint64_t off, uint64_t nbytes) {
                        if (mp) return true;
                        return off <= fsize && nbytes <= fsize - off;
                    };
                    if (!slice_ok(s.gate_mapped_ptr, s.gate_offset, s.gate_bytes) ||
                        !slice_ok(s.up_mapped_ptr, s.up_offset, s.up_bytes) ||
                        !slice_ok(s.down_mapped_ptr, s.down_offset, s.down_bytes)) {
                        err_msg = "Layer " + std::to_string(l) + ": expert " + std::to_string(e) +
                                  " slice is unreadable (no data pointer and offset outside the file)";
                        return false;
                    }
                }
            }
        } else {
            // Check for standard Dense MLP (Qwen2.5, LLaMA, Mistral)
            const TensorDescriptor* t_gate = model_file_.find_tensor(bp + "ffn_gate.weight");
            const TensorDescriptor* t_up   = model_file_.find_tensor(bp + "ffn_up.weight");
            const TensorDescriptor* t_down = model_file_.find_tensor(bp + "ffn_down.weight");
            if (t_gate && t_up && t_down) {
                if (!validate_bound_tensor(t_gate, "ffn_gate.weight", 0, err_msg) ||
                    !validate_bound_tensor(t_up, "ffn_up.weight", 0, err_msg) ||
                    !validate_bound_tensor(t_down, "ffn_down.weight", 0, err_msg)) {
                    err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                    return false;
                }
                if (t_gate->n_dims != 2 || t_up->n_dims != 2 || t_down->n_dims != 2 ||
                    t_gate->dims[0] != hp_.dim || t_up->dims[0] != hp_.dim ||
                    t_gate->dims[1] != t_up->dims[1] || t_gate->dims[1] > (1u << 20) ||
                    t_down->dims[0] != t_gate->dims[1] ||
                    t_down->dims[1] != hp_.dim) {
                    err_msg = "Layer " + std::to_string(l) + ": dense FFN tensor shapes are inconsistent";
                    return false;
                }
                layer_has_ffn = true;
                hp_.n_experts = 1;
                hp_.n_experts_used = 1;
                hp_.moe_expert_dim = (t_gate->n_dims >= 2) ? static_cast<uint32_t>(t_gate->dims[1]) : hp_.hidden_dim;
                lw.is_dense_ffn = true;
                lw.expert_slices.resize(1);
                lw.expert_gate_type = t_gate->type;
                lw.expert_up_type   = t_up->type;
                lw.expert_down_type = t_down->type;
                lw.expert_slices[0].gate_offset = t_gate->abs_file_offset;
                lw.expert_slices[0].gate_bytes  = t_gate->byte_size;
                lw.expert_slices[0].up_offset   = t_up->abs_file_offset;
                lw.expert_slices[0].up_bytes    = t_up->byte_size;
                lw.expert_slices[0].down_offset = t_down->abs_file_offset;
                lw.expert_slices[0].down_bytes  = t_down->byte_size;
                lw.expert_slices[0].gate_mapped_ptr = t_gate->mapped_ptr;
                lw.expert_slices[0].up_mapped_ptr   = t_up->mapped_ptr;
                lw.expert_slices[0].down_mapped_ptr = t_down->mapped_ptr;
            } else if (t_up && t_down && t_up->n_dims >= 2 && t_down->n_dims >= 1 &&
                       t_up->dims[1] == 2 * t_down->dims[0]) {
                // Fused gate_up_proj (Phi-3: gate_up_proj [2*intermediate,
                // hidden] stored as a single ffn_up tensor; rows
                // [0:intermediate] = gate, [intermediate:2*intermediate]
                // = up). This is a SwiGLU (SiLU) MLP, NOT a 2-matrix GELU
                // MLP. Verified against microsoft/Phi-3-mini-4k-instruct
                // SafeTensors (gate_up_proj [16384, 3072], down_proj
                // [3072, 8192]): GGUF ffn_up row 0 correlates with ST
                // row 0 (gate) and row 8192 with ST row 8192 (up), not
                // swapped; the swapped order gives punctuation garbage.
                // Gate/up are views into the two halves of the fused
                // tensor; the standard dense SwiGLU path runs them (SiLU:
                // hp_.use_gelu is false for Phi-3). Requires the Q2_K/Q3_K
                // block-struct fix (include/quant.h) to work — before that
                // fix this model gave <unk> in both RoPE modes.
                if (!validate_bound_tensor(t_up, "ffn_up.weight", 0, err_msg) ||
                    !validate_bound_tensor(t_down, "ffn_down.weight", 0, err_msg)) {
                    err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                    return false;
                }
                if (t_down->dims[0] == 0 || t_down->dims[0] > (1u << 20) ||
                    (t_down->n_dims >= 2 && t_down->dims[1] != hp_.dim)) {
                    err_msg = "Layer " + std::to_string(l) + ": fused ffn_down shape is inconsistent";
                    return false;
                }
                uint32_t inter = static_cast<uint32_t>(t_down->dims[0]);
                uint64_t up_row_bytes = 0;
                if (!checked_tensor_bytes(t_up->type, t_up->dims[0], up_row_bytes) || up_row_bytes == 0) {
                    err_msg = "Layer " + std::to_string(l) + ": fused ffn_up row size overflows";
                    return false;
                }
                // The two halves of the fused tensor must exactly cover
                // it: a short tensor used to make the up-half view read
                // past the tensor into the next one.
                if (up_row_bytes > UINT64_MAX / (2ull * inter) ||
                    up_row_bytes * inter * 2 != t_up->byte_size) {
                    err_msg = "Layer " + std::to_string(l) + ": fused ffn_up byte size does not match 2*intermediate rows";
                    return false;
                }
                layer_has_ffn = true;
                hp_.n_experts = 1;
                hp_.n_experts_used = 1;
                hp_.moe_expert_dim = inter;
                lw.is_dense_ffn = true;
                lw.ffn_gelu_only = false;
                lw.expert_slices.resize(1);
                lw.expert_gate_type = t_up->type;
                lw.expert_up_type   = t_up->type;
                lw.expert_down_type = t_down->type;
                lw.expert_slices[0].gate_offset = t_up->abs_file_offset;
                lw.expert_slices[0].gate_bytes  = up_row_bytes * inter;
                lw.expert_slices[0].up_offset   = t_up->abs_file_offset + up_row_bytes * inter;
                lw.expert_slices[0].up_bytes    = up_row_bytes * inter;
                lw.expert_slices[0].down_offset = t_down->abs_file_offset;
                lw.expert_slices[0].down_bytes  = t_down->byte_size;
                lw.expert_slices[0].gate_mapped_ptr = t_up->mapped_ptr;
                lw.expert_slices[0].up_mapped_ptr   = t_up->mapped_ptr
                    ? static_cast<const uint8_t*>(t_up->mapped_ptr) + up_row_bytes * inter : nullptr;
                lw.expert_slices[0].down_mapped_ptr = t_down->mapped_ptr;
            } else if (t_up && t_down) {
                // 2-matrix GELU MLP (up + down, no gate projection).
                // Without this the layer kept the default n_experts=8 and
                // the router selected experts with no backing tensors, so
                // the FFN contributed zero every layer and the model
                // generated only <unk> in both RoPE modes (verified
                // 2026-11-17 on Phi-3-mini-4k-instruct Q2_K, before the
                // fused gate_up form above was identified as the real
                // Phi-3 layout).
                if (!validate_bound_tensor(t_up, "ffn_up.weight", 0, err_msg) ||
                    !validate_bound_tensor(t_down, "ffn_down.weight", 0, err_msg)) {
                    err_msg = "Layer " + std::to_string(l) + ": " + err_msg;
                    return false;
                }
                if (t_up->n_dims != 2 || t_down->n_dims != 2 ||
                    t_up->dims[0] != hp_.dim || t_up->dims[1] != t_down->dims[0] ||
                    t_down->dims[0] == 0 || t_down->dims[0] > (1u << 20) ||
                    t_down->dims[1] != hp_.dim) {
                    err_msg = "Layer " + std::to_string(l) + ": GELU FFN tensor shapes are inconsistent";
                    return false;
                }
                layer_has_ffn = true;
                hp_.n_experts = 1;
                hp_.n_experts_used = 1;
                // NOTE: take the intermediate size from ffn_down's input
                // width (dims[0]), NOT from ffn_up's dims[1]. A fused
                // gate_up tensor has twice the rows of the intermediate
                // size (see the branch above); using ffn_up's dims[1]
                // would make the down matmul read past each row into the
                // next tensor.
                hp_.moe_expert_dim = (t_down->n_dims >= 1) ? static_cast<uint32_t>(t_down->dims[0]) : hp_.hidden_dim;
                lw.is_dense_ffn = true;
                lw.ffn_gelu_only = true;
                lw.expert_slices.resize(1);
                lw.expert_up_type   = t_up->type;
                lw.expert_down_type = t_down->type;
                lw.expert_slices[0].up_offset = t_up->abs_file_offset;
                lw.expert_slices[0].up_bytes  = t_up->byte_size;
                lw.expert_slices[0].down_offset = t_down->abs_file_offset;
                lw.expert_slices[0].down_bytes  = t_down->byte_size;
                lw.expert_slices[0].up_mapped_ptr   = t_up->mapped_ptr;
                lw.expert_slices[0].down_mapped_ptr = t_down->mapped_ptr;
            }
        }
        if (!layer_has_ffn) {
            err_msg = "Layer " + std::to_string(l) + ": no usable FFN tensors (no packed MoE experts "
                      "and no dense ffn_gate/up/down) — the layer would contribute nothing";
            return false;
        }
    }

    // Re-check the KV-capacity guard with the FINAL kv_dim: per-layer
    // overrides above can raise kv_dim above the layer-0 value the first
    // guard used, so the slots would allocate more than the budget.
    {
        uint64_t avail = hw_.avail_ram_bytes;
        if (avail == 0) avail = 2ull << 30; // same fallback as above
        uint64_t budget = avail / 4;
        uint32_t cap = hp_.kv_cache_capacity;
        while (cap > 256) {
            uint64_t est = static_cast<uint64_t>(hp_.n_layers) * cap *
                           static_cast<uint64_t>(hp_.kv_dim) * 2ull * 2ull;
            if (est <= budget) break;
            cap /= 2;
        }
        hp_.kv_cache_capacity = cap;
    }
    if (hp_.n_experts_used > hp_.n_experts) {
        err_msg = "expert_used_count (" + std::to_string(hp_.n_experts_used) +
                  ") exceeds expert_count (" + std::to_string(hp_.n_experts) +
                  ") after layer binding";
        return false;
    }

    return true;
}

KVCacheSlot& MoETransformerEngine::get_or_create_slot(uint32_t slot_id) {
    auto it = kv_slots_.find(slot_id);
    if (it == kv_slots_.end()) {
        KVCacheSlot s;
        s.init(slot_id, hp_);
        it = kv_slots_.emplace(slot_id, std::move(s)).first;
    }
    return it->second;
}

void MoETransformerEngine::reset_slot(uint32_t slot_id) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    auto it = kv_slots_.find(slot_id);
    if (it != kv_slots_.end()) it->second.clear();
}

bool MoETransformerEngine::clear_slot(uint32_t slot_id) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    auto it = kv_slots_.find(slot_id);
    if (it != kv_slots_.end()) {
        it->second.clear();
        kv_slots_.erase(it);
        return true;
    }
    return false;
}

size_t MoETransformerEngine::active_slot_count() const {
    std::unique_lock<std::mutex> lk(engine_mtx_, std::try_to_lock);
    if (!lk.owns_lock()) return 1;
    return kv_slots_.size();
}

std::vector<KVSlotSummary> MoETransformerEngine::list_slots() const {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    std::vector<KVSlotSummary> out;
    size_t kv_dim = static_cast<size_t>(hp_.n_kv_heads) * static_cast<size_t>(hp_.head_dim);
    for (const auto& [id, slot] : kv_slots_) {
        KVSlotSummary sum;
        sum.slot_id = id;
        sum.seq_pos = slot.seq_pos;
        sum.pages_allocated = slot.pages_allocated();
        sum.compactions = slot.compactions;
        sum.token_count = slot.token_history.size();
        sum.memory_bytes = static_cast<size_t>(slot.pages_allocated()) * KV_BLOCK_TOKENS * hp_.kv_dim * sizeof(uint16_t) * hp_.n_layers * 2;
        out.push_back(sum);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.slot_id < b.slot_id; });
    return out;
}

PagedBlockStats MoETransformerEngine::paged_block_stats() const {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    PagedBlockStats st;
    st.total_blocks = 2048;
    st.used_blocks = 0;
    size_t kv_dim = static_cast<size_t>(hp_.n_kv_heads) * static_cast<size_t>(hp_.head_dim);
    for (const auto& [id, slot] : kv_slots_) {
        st.used_blocks += slot.pages_allocated();
    }
    st.free_blocks = (st.total_blocks > st.used_blocks) ? (st.total_blocks - st.used_blocks) : 0;
    st.block_size_tokens = KV_BLOCK_TOKENS;
    st.memory_bytes = st.used_blocks * KV_BLOCK_TOKENS * hp_.kv_dim * sizeof(uint16_t) * hp_.n_layers * 2;
    return st;
}

std::vector<ProfileSample> MoETransformerEngine::recent_profiles() const {
    std::lock_guard<std::mutex> lk(profile_mtx_);
    return std::vector<ProfileSample>(profile_history_.begin(), profile_history_.end());
}

std::vector<float> MoETransformerEngine::forward_token(
    uint32_t token_id,
    uint32_t slot_id,
    ProfileSample* out_profile,
    bool compute_logits,
    std::vector<float>* out_hidden
) {
    std::lock_guard<std::mutex> lk(engine_mtx_);
    // --- Workstream C: entry guards. An empty return means "no logits"
    // (not loaded / bad slot / bad token / no KV room). Callers MUST
    // check for empty logits (generate_stream does) instead of sampling
    // token 0 from them.
    if (!loaded_) {
        std::cerr << "[FORWARD] rejected: engine not loaded\n";
        return {};
    }
    if (!is_valid_slot_id(slot_id)) {
        std::cerr << "[FORWARD] rejected: slot " << slot_id << " out of range (user slots 0.."
                  << kMaxUserSlot << ", scratch 8888/9998/9999)\n";
        return {};
    }
    if (token_id >= hp_.vocab_size) {
        std::cerr << "[FORWARD] rejected: token_id " << token_id << " >= vocab_size "
                  << hp_.vocab_size << "\n";
        return {};
    }
    const double t_start = now_ms();
    double io_wait_ms = 0.0;
    double matmul_ms  = 0.0;
    double attn_ms    = 0.0;
    double head_ms    = 0.0;
    uint32_t step_hits = 0;
    uint32_t step_reads = 0;
    uint32_t step_routed = 0;

    tier_mgr_.step_tick();
    KVCacheSlot& slot = get_or_create_slot(slot_id);
    if (slot.seq_pos >= hp_.kv_cache_capacity) {
        slot.compact(hp_, hp_.kv_cache_capacity / 2);
    }
    // Compaction must make room; if it could not, running would write
    // past the KV cache (pos is used as a direct index into k/v_cache).
    if (slot.seq_pos >= hp_.kv_cache_capacity) {
        std::cerr << "[FORWARD] rejected: slot " << slot_id << " seq_pos " << slot.seq_pos
                  << " >= KV capacity " << hp_.kv_cache_capacity << " after compaction\n";
        return {};
    }
    const uint32_t pos = slot.seq_pos;
    slot.token_history.push_back(token_id);

    const uint32_t dim = hp_.dim;
    const uint32_t max_q_dim = hp_.q_dim;
    const uint32_t max_kv_dim = hp_.kv_dim;
    const uint32_t exp_dim = hp_.moe_expert_dim;

    // Grow once, reuse every token — critical for large models (36–80 layers)
    // where per-call vector construction dominated allocator traffic.
    scratch_.ensure(dim, max_q_dim, max_kv_dim, exp_dim, hp_.n_experts,
                    hp_.vocab_size);
    auto& x = scratch_.x;
    auto& x_norm = scratch_.x_norm;
    auto& q = scratch_.q;
    auto& k = scratch_.k;
    auto& v = scratch_.v;
    auto& attn_out = scratch_.attn_out;
    auto& proj_out = scratch_.proj_out;
    auto& router_logits = scratch_.router_logits;
    auto& moe_accum = scratch_.moe_accum;
    auto& gate_buf = scratch_.gate_buf;
    auto& down_buf = scratch_.down_buf;

    // 1. Embedding Lookup
    std::fill(x.begin(), x.begin() + dim, 0.0f);
    if (tok_emb_ && token_id < hp_.vocab_size) {
        size_t row_bytes = tensor_byte_size(tok_emb_->type, dim);
        const uint8_t* emb_row = static_cast<const uint8_t*>(tok_emb_->mapped_ptr) + token_id * row_bytes;
        dequantize_row(tok_emb_->type, emb_row, x.data(), dim);
        if (hp_.scale_embeddings) {
            const float emb_scale = std::sqrt(static_cast<float>(dim));
            for (uint32_t i = 0; i < dim; ++i) x[i] *= emb_scale;
        }
    }

    // Optional Gemma4 Per-Layer Input (PLE) pre-projection
    std::vector<float> ple_all;
    if (hp_.ple_dim > 0 && per_layer_tok_emb_ && per_layer_model_proj_ &&
        per_layer_tok_emb_->mapped_ptr && per_layer_model_proj_->mapped_ptr &&
        token_id < hp_.vocab_size) {
        const size_t total_ple = static_cast<size_t>(hp_.n_layers) * hp_.ple_dim;
        ple_all.resize(total_ple, 0.0f);
        std::vector<float> ple_model(total_ple, 0.0f);
        size_t row_bytes = tensor_byte_size(per_layer_tok_emb_->type, total_ple);
        const uint8_t* ple_row = static_cast<const uint8_t*>(per_layer_tok_emb_->mapped_ptr) + token_id * row_bytes;
        dequantize_row(per_layer_tok_emb_->type, ple_row, ple_all.data(), static_cast<int64_t>(total_ple));

        const float ple_tok_scale = std::sqrt(static_cast<float>(hp_.ple_dim));
        const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(dim));
        const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
        std::vector<float> x_ple_in(dim);
        for (uint32_t i = 0; i < dim; ++i) x_ple_in[i] = x[i] * inv_sqrt_dim;
        matmul_quant_f32(per_layer_model_proj_->type, per_layer_model_proj_->mapped_ptr, x_ple_in.data(), ple_model.data(), static_cast<int64_t>(total_ple), dim);

        const float* pnorm_w = (per_layer_proj_norm_ && per_layer_proj_norm_->type == GGMLType::F32)
                             ? static_cast<const float*>(per_layer_proj_norm_->mapped_ptr) : nullptr;
        std::vector<float> ple_normed(hp_.ple_dim);
        for (uint32_t l = 0; l < hp_.n_layers; ++l) {
            float* m_slice = ple_model.data() + static_cast<size_t>(l) * hp_.ple_dim;
            float* t_slice = ple_all.data()   + static_cast<size_t>(l) * hp_.ple_dim;
            rms_norm(ple_normed.data(), m_slice, pnorm_w, hp_.ple_dim, hp_.rms_norm_eps);
            for (uint32_t d = 0; d < hp_.ple_dim; ++d) {
                t_slice[d] = (t_slice[d] * ple_tok_scale + ple_normed[d]) * inv_sqrt_2;
            }
        }
    }

    // Map holding PILOT async prefetch futures for Layer L+1
    std::unordered_map<uint32_t, std::future<StreamedExpertWeights>> prefetched_futures;

    for (uint32_t l = 0; l < hp_.n_layers; ++l) {
        const LayerDenseWeights& lw = layers_[l];
        const uint32_t q_dim = lw.q_dim > 0 ? lw.q_dim : max_q_dim;
        const uint32_t kv_dim = lw.kv_dim > 0 ? lw.kv_dim : max_kv_dim;
        const uint32_t head_dim = lw.head_dim > 0 ? lw.head_dim : hp_.head_dim;

        // 2. Attention Pre-RMSNorm
        const float* anorm_w = (lw.attn_norm && lw.attn_norm->type == GGMLType::F32)
                             ? static_cast<const float*>(lw.attn_norm->mapped_ptr) : nullptr;
        rms_norm(x_norm.data(), x.data(), anorm_w, dim, hp_.rms_norm_eps);

        // 3. Q, K, V Projections (Fused single OpenMP region with shared Q8_0 activation)
        double tm0 = now_ms();
        matmul_qkv_quant_f32(
            lw.wq ? lw.wq->type : GGMLType::F32, lw.wq ? lw.wq->mapped_ptr : nullptr, q_dim,  q.data(),
            lw.wk ? lw.wk->type : GGMLType::F32, lw.wk ? lw.wk->mapped_ptr : nullptr, kv_dim, k.data(),
            lw.wv ? lw.wv->type : GGMLType::F32, lw.wv ? lw.wv->mapped_ptr : nullptr, kv_dim, v.data(),
            x_norm.data(), dim
        );

        if (lw.bq && lw.bq->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bq->mapped_ptr);
            for (uint32_t i = 0; i < q_dim; ++i) q[i] += b_ptr[i];
        }
        if (lw.bk && lw.bk->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bk->mapped_ptr);
            for (uint32_t i = 0; i < kv_dim; ++i) k[i] += b_ptr[i];
        }
        if (lw.bv && lw.bv->type == GGMLType::F32) {
            const float* b_ptr = static_cast<const float*>(lw.bv->mapped_ptr);
            for (uint32_t i = 0; i < kv_dim; ++i) v[i] += b_ptr[i];
        }

        if (lw.attn_q_norm && lw.attn_q_norm->type == GGMLType::F32) {
            const float* w_ptr = static_cast<const float*>(lw.attn_q_norm->mapped_ptr);
            const bool shared_head_norm = (lw.attn_q_norm->dims[0] == head_dim);
            for (uint32_t h = 0; h < hp_.n_heads; ++h) {
                const float* hw = shared_head_norm ? w_ptr : (w_ptr + h * head_dim);
                rms_norm(q.data() + h * head_dim, q.data() + h * head_dim, hw, head_dim, hp_.rms_norm_eps);
            }
        }
        if (lw.attn_k_norm && lw.attn_k_norm->type == GGMLType::F32) {
            const float* w_ptr = static_cast<const float*>(lw.attn_k_norm->mapped_ptr);
            const bool shared_head_norm = (lw.attn_k_norm->dims[0] == head_dim);
            for (uint32_t h = 0; h < hp_.n_kv_heads; ++h) {
                const float* hw = shared_head_norm ? w_ptr : (w_ptr + h * head_dim);
                rms_norm(k.data() + h * head_dim, k.data() + h * head_dim, hw, head_dim, hp_.rms_norm_eps);
            }
        }
        if (hp_.v_norm) {
            for (uint32_t h = 0; h < hp_.n_kv_heads; ++h) {
                rms_norm(v.data() + h * head_dim, v.data() + h * head_dim, nullptr, head_dim, hp_.rms_norm_eps);
            }
        }
        matmul_ms += (now_ms() - tm0);

        // 4. Rotary Position Embeddings (RoPE / YaRN) & KV-Cache Append
        double ta0 = now_ms();
        // Neox is the default for modern LLMs (see arch_uses_neox_rope);
        // only GPT-2 / GPT-J use interleaved RoPE.
        bool is_neox = hp_.rope_neox;
        const float* rf_ptr = (!lw.is_swa && rope_freqs_ && rope_freqs_->type == GGMLType::F32 &&
                               rope_freqs_->dims[0] >= lw.rope_dim / 2)
                            ? static_cast<const float*>(rope_freqs_->mapped_ptr) : nullptr;
        apply_rope(
            q.data(), k.data(),
            hp_.n_heads, hp_.n_kv_heads, head_dim, lw.rope_dim,
            pos, lw.rope_freq_base, hp_.yarn_scale, is_neox, rf_ptr
        );

        const uint32_t kv_layer = lw.kv_source_layer;
        if (!lw.shares_kv) {
            uint16_t* k_slot = slot.k_cache[kv_layer].data() + static_cast<size_t>(pos) * max_kv_dim;
            uint16_t* v_slot = slot.v_cache[kv_layer].data() + static_cast<size_t>(pos) * max_kv_dim;
            for (uint32_t i = 0; i < kv_dim; ++i) {
                k_slot[i] = fp32_to_fp16(k[i]);
                v_slot[i] = fp32_to_fp16(v[i]);
            }
        }

        // 5. Grouped-Query / Multi-Head Scaled Dot-Product Attention (FP16 KV Cache, Head-Parallel, SWA-aware)
        std::fill(attn_out.begin(), attn_out.begin() + q_dim, 0.0f);
        const uint32_t kv_mul = std::max(1u, hp_.n_heads / std::max(1u, hp_.n_kv_heads));
        const float scale = (hp_.arch == "gemma4") ? 1.0f : (1.0f / std::sqrt(static_cast<float>(head_dim)));
        const uint32_t t_start_win = (lw.is_swa && hp_.sliding_window > 0 && pos >= hp_.sliding_window)
                                   ? (pos - hp_.sliding_window + 1) : 0;

#if defined(_OPENMP)
        // Parallelize attention earlier: waiting until pos>=32 left the first
        // ~32 decode steps single-threaded on multi-head models (stutter at
        // the start of a reply).
        #pragma omp parallel for if(hp_.n_heads >= 4 && pos >= 8) schedule(static)
#endif
        for (int h_i = 0; h_i < static_cast<int>(hp_.n_heads); ++h_i) {
            const uint32_t h = static_cast<uint32_t>(h_i);
            const float* q_head = q.data() + h * head_dim;
            const uint32_t kvh = h / kv_mul;
            const uint32_t win_len = pos - t_start_win + 1;
            // Thread-local scores: OpenMP heads must not share one buffer,
            // and reallocating per head per layer stalls large models.
            thread_local std::vector<float> tls_scores;
            if (tls_scores.size() < win_len) tls_scores.resize(win_len);
            float* scores = tls_scores.data();

            for (uint32_t t = t_start_win; t <= pos; ++t) {
                const uint16_t* k_t = slot.k_cache[kv_layer].data() + static_cast<size_t>(t) * max_kv_dim + kvh * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * fp16_to_fp32(k_t[d]);
                }
                scores[t - t_start_win] = dot * scale;
            }
            // Gemma2/3 attention logit softcapping (same tanh form as the
            // final-logit softcap, applied before softmax, only when > 0).
            if (hp_.attn_logit_softcapping > 0.0f) {
                const float cap = hp_.attn_logit_softcapping;
                const float inv_cap = 1.0f / cap;
                for (uint32_t s = 0; s < win_len; ++s) {
                    scores[s] = cap * std::tanh(scores[s] * inv_cap);
                }
            }
            softmax(scores, win_len);

            float* out_head = attn_out.data() + h * head_dim;
            for (uint32_t t = t_start_win; t <= pos; ++t) {
                const uint16_t* v_t = slot.v_cache[kv_layer].data() + static_cast<size_t>(t) * max_kv_dim + kvh * head_dim;
                const float a = scores[t - t_start_win];
                for (uint32_t d = 0; d < head_dim; ++d) {
                    out_head[d] += a * fp16_to_fp32(v_t[d]);
                }
            }
        }
        attn_ms += (now_ms() - ta0);

        // Attention Output Projection + Optional Post-Attn Norm + Residual Connection
        double tm1 = now_ms();
        if (lw.wo) {
            matmul_quant_f32(lw.wo->type, lw.wo->mapped_ptr, attn_out.data(), proj_out.data(), dim, q_dim);
            if (lw.bo && lw.bo->type == GGMLType::F32) {
                const float* bo_ptr = static_cast<const float*>(lw.bo->mapped_ptr);
                for (uint32_t i = 0; i < dim; ++i) proj_out[i] += bo_ptr[i];
            }
            if (lw.post_attn_norm && lw.post_attn_norm->type == GGMLType::F32) {
                rms_norm(proj_out.data(), proj_out.data(), static_cast<const float*>(lw.post_attn_norm->mapped_ptr), dim, hp_.rms_norm_eps);
            }
            for (uint32_t i = 0; i < dim; ++i) x[i] += proj_out[i];
        }
        matmul_ms += (now_ms() - tm1);

        // 6. Post-Attention FFN RMSNorm & MoE Router Gate
        const float* fnorm_w = (lw.ffn_norm && lw.ffn_norm->type == GGMLType::F32)
                             ? static_cast<const float*>(lw.ffn_norm->mapped_ptr) : nullptr;
        rms_norm(x_norm.data(), x.data(), fnorm_w, dim, hp_.rms_norm_eps);

        std::fill(router_logits.begin(), router_logits.end(), 0.0f);
        if (lw.router_gate) {
            matmul_quant_f32(lw.router_gate->type, lw.router_gate->mapped_ptr, x_norm.data(), router_logits.data(), hp_.n_experts, dim);
        } else {
            if (!router_logits.empty()) router_logits[0] = 1.0f;
        }

        // Select Top-K experts with CACHE_ROUTE awareness
        auto selected = tier_mgr_.select_experts_cache_aware(l, router_logits.data(), hp_.n_experts, hp_.n_experts_used);

        // PILOT Lookahead Prefetch: While we are about to execute Layer L's SwiGLU matmuls,
        // predict Layer L+1's top experts and dispatch asynchronous NVMe reads now!
        std::unordered_map<uint32_t, std::future<StreamedExpertWeights>> next_layer_futures;
        if (l + 1 < hp_.n_layers && layers_[l + 1].router_gate) {
            const LayerDenseWeights& next_lw = layers_[l + 1];
            std::vector<float> next_logits(hp_.n_experts, 0.0f);
            matmul_quant_f32(
                next_lw.router_gate->type,
                next_lw.router_gate->mapped_ptr,
                x_norm.data(),
                next_logits.data(),
                hp_.n_experts,
                dim
            );
            // Find top-K candidates for Layer L+1 and launch background IOCP / async reads if not cached
            std::vector<uint32_t> idx_order(hp_.n_experts);
            std::iota(idx_order.begin(), idx_order.end(), 0);
            std::partial_sort(idx_order.begin(), idx_order.begin() + std::min(hp_.n_experts_used, hp_.n_experts), idx_order.end(),
                [&](uint32_t a, uint32_t b) { return next_logits[a] > next_logits[b]; });

            for (uint32_t k_idx = 0; k_idx < std::min(hp_.n_experts_used, hp_.n_experts); ++k_idx) {
                uint32_t cand_e = idx_order[k_idx];
                if (!tier_mgr_.is_resident(l + 1, cand_e) && cand_e < next_lw.expert_slices.size() &&
                    !next_lw.expert_slices[cand_e].gate_mapped_ptr) {
                    next_layer_futures[cand_e] = io_engine_.prefetch_expert_async(
                        l + 1, cand_e, next_lw.expert_slices[cand_e]
                    );
                }
            }
        }

        // 7. Execute Selected MoE Experts (SwiGLU: (SiLU(u * W_gate) * (u * W_up)) * W_down)
        std::fill(moe_accum.begin(), moe_accum.end(), 0.0f);

        for (const auto& sel : selected) {
            if (sel.was_cache_routed) step_routed++;
            const uint32_t e = sel.expert_idx;
            // 2-matrix GELU MLP (Phi-3): down(GELU(up(x))). No gate
            // matrix, no routing, no cache — bypass the SwiGLU path
            // entirely (its gate pointer would be null and the layer
            // would contribute zero). GELU is the tanh approximation,
            // matching Phi-3's activation. gate_buf is reused as the
            // GELU intermediate (same size: exp_dim).
            if (lw.ffn_gelu_only && e < lw.expert_slices.size()) {
                const auto& slice = lw.expert_slices[e];
                if (slice.up_mapped_ptr && slice.down_mapped_ptr) {
                    double t_exp = now_ms();
                    matmul_quant_f32(lw.expert_up_type, slice.up_mapped_ptr,
                                     x_norm.data(), gate_buf.data(), exp_dim, dim);
                    for (uint32_t r = 0; r < exp_dim; ++r) {
                        float u = gate_buf[r];
                        constexpr float kSqrt2OverPi = 0.7978845608028654f;
                        gate_buf[r] = 0.5f * u * (1.0f + std::tanh(kSqrt2OverPi * (u + 0.044715f * u * u * u)));
                    }
                    matmul_quant_f32(lw.expert_down_type, slice.down_mapped_ptr,
                                     gate_buf.data(), down_buf.data(), dim, exp_dim);
                    for (uint32_t d = 0; d < dim; ++d) {
                        moe_accum[d] += sel.prob * down_buf[d];
                    }
                    matmul_ms += (now_ms() - t_exp);
                }
                continue;
            }
            const ExpertCacheEntry* cached = lw.is_dense_ffn ? nullptr : tier_mgr_.get_resident(l, e);

            const void* w_gate_ptr = nullptr;
            const void* w_up_ptr   = nullptr;
            const void* w_down_ptr = nullptr;
            StreamedExpertWeights scratch;

            if (lw.is_dense_ffn && e < lw.expert_slices.size() &&
                lw.expert_slices[e].gate_mapped_ptr &&
                lw.expert_slices[e].up_mapped_ptr &&
                lw.expert_slices[e].down_mapped_ptr) {
                // Zero-copy direct memory-mapped execution for dense FFN backbone
                step_hits++;
                w_gate_ptr = lw.expert_slices[e].gate_mapped_ptr;
                w_up_ptr   = lw.expert_slices[e].up_mapped_ptr;
                w_down_ptr = lw.expert_slices[e].down_mapped_ptr;
            } else if (cached) {
                step_hits++;
                w_gate_ptr = !cached->weight_gate.empty() ? cached->weight_gate.data() : lw.expert_slices[e].gate_mapped_ptr;
                w_up_ptr   = !cached->weight_up.empty()   ? cached->weight_up.data()   : lw.expert_slices[e].up_mapped_ptr;
                w_down_ptr = !cached->weight_down.empty() ? cached->weight_down.data() : lw.expert_slices[e].down_mapped_ptr;
            } else if (e < lw.expert_slices.size()) {
                const auto& slice = lw.expert_slices[e];
                if (slice.gate_mapped_ptr && slice.up_mapped_ptr && slice.down_mapped_ptr && prefetched_futures.find(e) == prefetched_futures.end()) {
                    // Zero-copy mmap fast path: admit lightweight cache entry without duplicating multi-MB weight buffers
                    step_hits++;
                    tier_mgr_.admit_expert(l, e, {}, {}, {});
                    w_gate_ptr = slice.gate_mapped_ptr;
                    w_up_ptr   = slice.up_mapped_ptr;
                    w_down_ptr = slice.down_mapped_ptr;
                } else {
                    step_reads++;
                    double tio0 = now_ms();
                    auto fit = prefetched_futures.find(e);
                    if (fit != prefetched_futures.end()) {
                        scratch = fit->second.get();
                    } else {
                        scratch = io_engine_.read_expert_sync(l, e, slice);
                    }
                    io_wait_ms += (now_ms() - tio0);

                    // A short/failed disk read must not be admitted as
                    // valid weights. read_expert_sync() reports it via
                    // StreamedExpertWeights.ok (pilot_io loops pread to
                    // completion and returns false on EOF/error); the size
                    // check is kept as a second line of defense. A failed
                    // read falls back to the slice's mmap pointer (or the
                    // expert is skipped when there is none). Admitting a
                    // short buffer would poison the cache — the
                    // admitted-entry path below takes .data() without an
                    // .empty() check.
                    bool scratch_ok = scratch.ok &&
                                      (scratch.gate.size() == slice.gate_bytes &&
                                       scratch.up.size() == slice.up_bytes &&
                                       scratch.down.size() == slice.down_bytes);
                    if (!scratch_ok) {
                        std::cerr << "[FORWARD] WARNING: expert read for layer " << l << " expert " << e
                                  << " failed or returned a short buffer (gate " << scratch.gate.size() << "/"
                                  << slice.gate_bytes << " bytes) — discarding, using mmap fallback\n";
                    }
                    const ExpertCacheEntry* admitted = nullptr;
                    if (scratch_ok) {
                        // Attempt to admit to LFRU cache (respects 25% hysteresis threshold)
                        admitted = tier_mgr_.admit_expert(
                            l, e, std::move(scratch.gate), std::move(scratch.up), std::move(scratch.down)
                        );
                    }
                    if (admitted) {
                        w_gate_ptr = admitted->weight_gate.data();
                        w_up_ptr   = admitted->weight_up.data();
                        w_down_ptr = admitted->weight_down.data();
                    } else {
                        w_gate_ptr = slice.gate_mapped_ptr;
                        w_up_ptr   = slice.up_mapped_ptr;
                        w_down_ptr = slice.down_mapped_ptr;
                    }
                }
            }

            if (w_gate_ptr && w_up_ptr && w_down_ptr) {
                double t_exp = now_ms();
                matmul_swiglu_quant_f32(
                    lw.expert_gate_type, w_gate_ptr,
                    lw.expert_up_type,   w_up_ptr,
                    x_norm.data(), gate_buf.data(), exp_dim, dim, hp_.use_gelu
                );

                matmul_quant_f32(lw.expert_down_type, w_down_ptr, gate_buf.data(), down_buf.data(), dim, exp_dim);
                for (uint32_t d = 0; d < dim; ++d) {
                    moe_accum[d] += sel.prob * down_buf[d];
                }
                matmul_ms += (now_ms() - t_exp);
            }
        }

        // 7b. Shared Expert Execution (Qwen2.5-MoE / DeepSeek-MoE)
        if (lw.shexp_gate && lw.shexp_up && lw.shexp_down &&
            lw.shexp_gate->mapped_ptr && lw.shexp_up->mapped_ptr && lw.shexp_down->mapped_ptr) {
            double t_sh = now_ms();
            uint32_t sh_dim = (lw.shexp_gate->n_dims >= 2) ? static_cast<uint32_t>(lw.shexp_gate->dims[1]) : exp_dim;
            std::vector<float> sh_buf(sh_dim, 0.0f);
            matmul_swiglu_quant_f32(
                lw.shexp_gate->type, lw.shexp_gate->mapped_ptr,
                lw.shexp_up->type,   lw.shexp_up->mapped_ptr,
                x_norm.data(), sh_buf.data(), sh_dim, dim
            );
            matmul_quant_f32(lw.shexp_down->type, lw.shexp_down->mapped_ptr, sh_buf.data(), down_buf.data(), dim, sh_dim);
            float sh_scale = 1.0f;
            if (lw.shexp_gate_inp && lw.shexp_gate_inp->mapped_ptr) {
                float g_val = vec_dot_quant_f32(lw.shexp_gate_inp->type, lw.shexp_gate_inp->mapped_ptr, x_norm.data(), dim);
                sh_scale = 1.0f / (1.0f + std::exp(-g_val));
            }
            for (uint32_t d = 0; d < dim; ++d) {
                moe_accum[d] += sh_scale * down_buf[d];
            }
            matmul_ms += (now_ms() - t_sh);
        }

        if (lw.post_ffw_norm && lw.post_ffw_norm->type == GGMLType::F32) {
            rms_norm(moe_accum.data(), moe_accum.data(), static_cast<const float*>(lw.post_ffw_norm->mapped_ptr), dim, hp_.rms_norm_eps);
        }

        for (uint32_t i = 0; i < dim; ++i) {
            x[i] += moe_accum[i];
        }

        // Gemma4 Per-Layer Embedding (PLE) residual + layer_output_scale
        if (!ple_all.empty() && lw.inp_gate && lw.ple_proj &&
            lw.inp_gate->mapped_ptr && lw.ple_proj->mapped_ptr) {
            std::vector<float> g_ple(hp_.ple_dim, 0.0f);
            matmul_quant_f32(lw.inp_gate->type, lw.inp_gate->mapped_ptr, x.data(), g_ple.data(), hp_.ple_dim, dim);
            const float* ple_l = ple_all.data() + static_cast<size_t>(l) * hp_.ple_dim;
            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            for (uint32_t d = 0; d < hp_.ple_dim; ++d) {
                float gv = g_ple[d];
                float gelu_g = 0.5f * gv * (1.0f + std::tanh(kSqrt2OverPi * (gv + 0.044715f * gv * gv * gv)));
                g_ple[d] = gelu_g * ple_l[d];
            }
            matmul_quant_f32(lw.ple_proj->type, lw.ple_proj->mapped_ptr, g_ple.data(), proj_out.data(), dim, hp_.ple_dim);
            if (lw.post_norm && lw.post_norm->type == GGMLType::F32) {
                rms_norm(proj_out.data(), proj_out.data(), static_cast<const float*>(lw.post_norm->mapped_ptr), dim, hp_.rms_norm_eps);
            }
            for (uint32_t i = 0; i < dim; ++i) x[i] += proj_out[i];
        }
        if (lw.layer_output_scale && lw.layer_output_scale->type == GGMLType::F32 && lw.layer_output_scale->mapped_ptr) {
            float sc = *static_cast<const float*>(lw.layer_output_scale->mapped_ptr);
            for (uint32_t i = 0; i < dim; ++i) x[i] *= sc;
        }

        prefetched_futures = std::move(next_layer_futures);
    }

    slot.seq_pos++;

    // 8. Final RMSNorm & LM Head Logit Projection (skipped for intermediate prefill tokens)
    double th0 = now_ms();
    std::vector<float> logits;
    if (compute_logits || out_hidden != nullptr) {
        const float* onorm_w = (out_norm_ && out_norm_->type == GGMLType::F32)
                             ? static_cast<const float*>(out_norm_->mapped_ptr) : nullptr;
        rms_norm(x_norm.data(), x.data(), onorm_w, dim, hp_.rms_norm_eps);
        if (out_hidden) {
            out_hidden->assign(x_norm.begin(), x_norm.begin() + dim);
        }
        if (compute_logits) {
            // Do not zero-fill: matmul_quant_f32 writes every vocab row.
            // Write into reused scratch, then copy out so the API still
            // returns an owned vector without reallocating the scratch.
            float* logit_dst = scratch_.logits.data();
            if (lm_head_) {
                matmul_quant_f32(lm_head_->type, lm_head_->mapped_ptr, x_norm.data(), logit_dst, hp_.vocab_size, dim);
                if (hp_.final_logit_softcapping > 0.0f) {
                    const float cap = hp_.final_logit_softcapping;
                    const float inv_cap = 1.0f / cap;
                    for (uint32_t i = 0; i < hp_.vocab_size; ++i) {
                        logit_dst[i] = cap * std::tanh(logit_dst[i] * inv_cap);
                    }
                }
            }
            logits.assign(scratch_.logits.begin(),
                          scratch_.logits.begin() + hp_.vocab_size);
        }
    }
    head_ms = now_ms() - th0;

    const double wall_ms = std::max(0.01, now_ms() - t_start);
    ProfileSample prof;
    prof.timestamp_ms = static_cast<uint64_t>(now_us() / 1000);
    prof.token_id     = token_id;
    prof.wall_ms      = wall_ms;
    prof.io_wait_ms   = io_wait_ms;
    prof.matmul_ms    = matmul_ms;
    prof.attention_ms = attn_ms;
    prof.lm_head_ms   = head_ms;
    prof.tok_per_sec  = 1000.0 / wall_ms;
    prof.cache_hits   = step_hits;
    prof.disk_reads   = step_reads;
    prof.cache_routed = step_routed;

    if (out_profile) *out_profile = prof;

    {
        std::lock_guard<std::mutex> plk(profile_mtx_);
        profile_history_.push_back(prof);
        if (profile_history_.size() > 120) {
            profile_history_.pop_front();
        }
    }

    return logits;
}

namespace {
// Sampler RNG: thread-local, (re)seeded once per generate_stream() call.
// An explicit non-zero seed reproduces the same output every run; seed 0
// draws fresh entropy (random_device + time) per generation.
thread_local std::mt19937_64 g_sampler_rng;
thread_local bool g_sampler_seeded = false;
} // namespace

static void reseed_sampler_for_generation(uint64_t seed) {
    if (seed != 0) {
        g_sampler_rng.seed(seed);
    } else {
        std::random_device rd;
        uint64_t s = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        s ^= static_cast<uint64_t>(now_us());
        s ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        g_sampler_rng.seed(s);
    }
    g_sampler_seeded = true;
}

std::vector<uint32_t> MoETransformerEngine::stop_token_ids() const {
    std::vector<uint32_t> ids;
    auto add = [&ids](int32_t id) {
        if (id < 0) return;
        uint32_t u = static_cast<uint32_t>(id);
        if (std::find(ids.begin(), ids.end(), u) == ids.end()) ids.push_back(u);
    };
    add(static_cast<int32_t>(tokenizer_.eos_id()));
    add(static_cast<int32_t>(tokenizer_.im_end_id()));
    // Generic stop markers resolved against THIS model's vocab. A marker
    // only stops generation when it is a real token in the vocabulary, so
    // no vendor-specific ID is ever hardcoded. Note "<end_of_turn>" (no
    // pipes) is Gemma's turn-end token; "<|end_of_turn|>" is the ChatML
    // variant — both are needed, and the token form is what actually
    // stops Gemma (verified 2026-11-17: without it generation ran past
    // <end_of_turn> into a second turn).
    static const char* kStopMarkers[] = {
        "<|end_of_text|>", "<|im_end|>", "<|end_of_turn|>", "<end_of_turn>",
        "<|eot_id|>", "</s>", "<|endoftext|>", "<eos>", "<|end|>"
    };
    for (const char* m : kStopMarkers) add(tokenizer_.token_to_id(m));
    return ids;
}

bool MoETransformerEngine::is_stop_token(uint32_t id) const {
    const auto ids = stop_token_ids();
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

std::string MoETransformerEngine::chat_template_source() const {
    if (!tokenizer_.chat_template().empty()) return "gguf-chat-template";
    std::string src;
    format_chat_messages({}, hp_.arch, "", tokenizer_, true, &src);
    return src; // "fallback:<name>"
}

uint32_t MoETransformerEngine::sample_next_token(
    const std::vector<float>& logits,
    const std::vector<uint32_t>& history,
    const SamplerConfig& cfg
) {
    if (logits.empty()) return 0;
    // Defensive sanitization (the authoritative check is
    // validate_sampler_config(), called by generate_stream /
    // verify_speculative_batch — but this function is also callable
    // directly, and a NaN temperature or a zero repetition_penalty here
    // used to produce NaN logits / a division by zero with no error).
    // Invalid values fall back to the safe equivalent here; the entry
    // points reject them instead of silently fixing them.
    SamplerConfig safe = cfg;
    if (std::isnan(safe.temperature) || std::isinf(safe.temperature) || safe.temperature < 0.0f) safe.temperature = 0.0f; // greedy
    if (std::isnan(safe.top_p) || safe.top_p < 0.0f || safe.top_p > 1.0f) safe.top_p = 1.0f; // nucleus off
    if (std::isnan(safe.repetition_penalty) || std::isinf(safe.repetition_penalty) || safe.repetition_penalty <= 0.0f) safe.repetition_penalty = 1.0f;
    const uint32_t vocab_sz = static_cast<uint32_t>(logits.size());
    std::vector<float> adj = logits;
    // A NaN logit (corrupt weights / overflowed activations) used to make
    // max_element / partial_sort / the softmax produce unspecified
    // results. Treat NaN as -inf so it can never be sampled; if every
    // logit is NaN there is no valid token to return.
    for (auto& v : adj) if (std::isnan(v)) v = -std::numeric_limits<float>::infinity();
    if (std::isinf(*std::max_element(adj.begin(), adj.end()))) return 0;

    // 1. Repetition Penalty over recent history
    if (safe.repetition_penalty != 1.0f && !history.empty()) {
        size_t start = (history.size() > 64) ? (history.size() - 64) : 0;
        for (size_t i = start; i < history.size(); ++i) {
            uint32_t tid = history[i];
            if (tid == 220 || tid == 198 || tid == 0) continue; // skip space, newline, unk
            if (tid < vocab_sz) {
                if (adj[tid] > 0.0f) adj[tid] /= safe.repetition_penalty;
                else adj[tid] *= safe.repetition_penalty;
            }
        }
    }

    // 2. Greedy Argmax when temperature <= 0
    if (safe.temperature <= 1e-5f || safe.top_k == 1) {
        return static_cast<uint32_t>(std::max_element(adj.begin(), adj.end()) - adj.begin());
    }

    // 3. Top-K filtering — O(V log k) min-heap instead of allocating and
    // partial_sorting a TokProb for every vocab row (V≈150k on Qwen).
    struct TokProb {
        uint32_t id;
        float val;
    };
    const uint32_t k = (safe.top_k > 0 && safe.top_k < vocab_sz) ? safe.top_k : vocab_sz;
    const float inv_temp = 1.0f / safe.temperature;
    std::vector<TokProb> cands;
    cands.reserve(k);
    auto worse = [](const TokProb& a, const TokProb& b) { return a.val > b.val; }; // min-heap by score
    for (uint32_t i = 0; i < vocab_sz; ++i) {
        const float v = adj[i] * inv_temp;
        if (cands.size() < k) {
            cands.push_back({i, v});
            if (cands.size() == k) std::make_heap(cands.begin(), cands.end(), worse);
        } else if (v > cands.front().val) {
            std::pop_heap(cands.begin(), cands.end(), worse);
            cands.back() = {i, v};
            std::push_heap(cands.begin(), cands.end(), worse);
        }
    }
    std::sort_heap(cands.begin(), cands.end(), worse); // ascending; reverse for softmax max
    std::reverse(cands.begin(), cands.end());

    // 4. Softmax + Top-P (Nucleus) filtering
    float max_l = cands[0].val;
    float sum_exp = 0.0f;
    for (auto& c : cands) {
        c.val = std::exp(c.val - max_l);
        sum_exp += c.val;
    }
    for (auto& c : cands) c.val /= sum_exp;

    if (safe.top_p > 0.0f && safe.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].val;
            if (cum >= cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        cands.resize(std::max<size_t>(1, cutoff));
    }

    // Renormalize & sample. The RNG is seeded per generation (see
    // reseed_sampler_for_generation); a direct call without a generation
    // lazily seeds from cfg.seed (or fresh entropy when seed is 0).
    float norm_sum = 0.0f;
    for (const auto& c : cands) norm_sum += c.val;

    if (!g_sampler_seeded) reseed_sampler_for_generation(cfg.seed);
    std::uniform_real_distribution<float> dist(0.0f, norm_sum);
    float r = dist(g_sampler_rng);
    float acc = 0.0f;
    for (const auto& c : cands) {
        acc += c.val;
        if (r <= acc) return c.id;
    }
    return cands.front().id;
}

std::vector<uint32_t> MoETransformerEngine::verify_speculative_batch(
    uint32_t slot_id,
    const std::vector<uint32_t>& draft_tokens,
    const SamplerConfig& sampler_cfg,
    ProfileSample* out_profile
) {
    std::vector<uint32_t> accepted;
    if (draft_tokens.empty()) return accepted;
    // Workstream C: reject invalid inputs instead of verifying garbage.
    if (!is_valid_slot_id(slot_id)) {
        std::cerr << "[SPECULATIVE] rejected: slot " << slot_id << " out of range\n";
        return accepted;
    }
    {
        std::string cfg_err;
        if (!validate_sampler_config(sampler_cfg, cfg_err)) {
            std::cerr << "[SPECULATIVE] rejected: invalid sampler config: " << cfg_err << "\n";
            return accepted;
        }
    }
    if (draft_tokens.size() > kMaxDraftTokens) {
        std::cerr << "[SPECULATIVE] rejected: " << draft_tokens.size()
                  << " draft tokens exceeds the maximum of " << kMaxDraftTokens << "\n";
        return accepted;
    }

    uint32_t base_seq_pos = 0;
    {
        std::lock_guard<std::mutex> lk(engine_mtx_);
        base_seq_pos = get_or_create_slot(slot_id).seq_pos;
    }

    // Verify each draft token strictly against the target model's sampled prediction
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        uint32_t draft_id = draft_tokens[i];
        ProfileSample prof{};
        auto logits = forward_token(draft_id, slot_id, &prof);
        if (logits.empty()) {
            // Out-of-vocab draft token (or another forward failure):
            // do not accept the draft, stop verification here.
            std::cerr << "[SPECULATIVE] stopping: forward failed for draft token " << draft_id << "\n";
            return accepted;
        }
        if (out_profile) *out_profile = prof;
        accepted.push_back(draft_id);

        std::vector<uint32_t> hist;
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            hist = get_or_create_slot(slot_id).token_history;
        }
        uint32_t target_next = sample_next_token(logits, hist, sampler_cfg);
        if (is_stop_token(target_next)) {
            break;
        }
        if (i + 1 < draft_tokens.size() && target_next != draft_tokens[i + 1]) {
            // Mismatch on next draft token: replace suffix with target_next (no extra rollback needed)
            accepted.push_back(target_next);
            break;
        }
    }
    (void)base_seq_pos;
    return accepted;
}

void MoETransformerEngine::generate_stream(
    const std::string& prompt,
    uint32_t max_new_tokens,
    uint32_t slot_id,
    bool enable_context_compaction,
    const SamplerConfig& sampler_cfg,
    StreamCallback on_token,
    bool continue_assistant,
    bool enable_speculative,
    uint32_t draft_count
) {
    // Workstream C: reject invalid inputs up front — a bad slot used to
    // allocate an unbounded KV cache, a NaN temperature used to produce
    // NaN logits, and a huge max_new_tokens ran until EOS (or forever).
    // The HTTP server must ALSO validate (server.cpp, workstream B) so a
    // bad request gets a 400 instead of an empty stream.
    if (!loaded_) {
        std::cerr << "[GENERATE] rejected: engine not loaded\n";
        return;
    }
    if (!is_valid_slot_id(slot_id)) {
        std::cerr << "[GENERATE] rejected: slot " << slot_id << " out of range (user slots 0.."
                  << kMaxUserSlot << ", scratch 8888/9998/9999)\n";
        return;
    }
    {
        std::string cfg_err;
        if (!validate_sampler_config(sampler_cfg, cfg_err)) {
            std::cerr << "[GENERATE] rejected: invalid sampler config: " << cfg_err << "\n";
            return;
        }
    }
    if (max_new_tokens > kMaxNewTokens) {
        std::cerr << "[GENERATE] rejected: max_new_tokens " << max_new_tokens
                  << " exceeds the maximum of " << kMaxNewTokens << "\n";
        return;
    }
    if (enable_speculative && draft_count > kMaxDraftTokens) {
        std::cerr << "[GENERATE] rejected: draft_count " << draft_count
                  << " exceeds the maximum of " << kMaxDraftTokens << "\n";
        return;
    }
    if (!on_token) {
        std::cerr << "[GENERATE] rejected: no token callback\n";
        return;
    }
    std::vector<float> logits;
    ProfileSample last_prof;
    bool cancelled = false; // set when on_token returns false; suppresses the tail flush

    // (Re)seed the sampler for this generation: an explicit non-zero seed
    // is deterministic, seed 0 draws fresh entropy.
    reseed_sampler_for_generation(sampler_cfg.seed);

    bool has_existing_context = false;
    {
        std::lock_guard<std::mutex> lk(engine_mtx_);
        has_existing_context = (get_or_create_slot(slot_id).seq_pos > 0);
    }

    if (!continue_assistant || !has_existing_context) {
        // Single shared chat-template path (identical to the HTTP server):
        // GGUF tokenizer.chat_template wins, else a per-arch fallback.
        // Raw prompts become one user turn; already-templated prompts
        // pass through unchanged. No default system prompt is injected.
        std::string prompt_to_encode = wrap_raw_prompt_for_chat(
            prompt, hp_.arch, tokenizer_.chat_template(), tokenizer_);

        bool add_bos = should_add_bos(hp_.arch, tokenizer_);
        if (add_bos && tokenizer_.bos_id() < tokenizer_.vocab_size()) {
            // The rendered template may already start with the BOS token
            // text (e.g. Llama3 {{ bos_token }}); do not emit it twice.
            const std::string& bos_text = tokenizer_.vocab()[tokenizer_.bos_id()];
            if (!bos_text.empty() && prompt_to_encode.rfind(bos_text, 0) == 0) add_bos = false;
        }
        std::vector<uint32_t> prompt_tokens = tokenizer_.encode(prompt_to_encode, add_bos);
        if (prompt_tokens.empty()) {
            prompt_tokens.push_back(tokenizer_.bos_id());
        }

        // Reuse Longest Common Prefix (LCP) from existing KV cache in this slot
        size_t lcp = 0;
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            auto& slot = get_or_create_slot(slot_id);
            if (slot.compactions == 0 && slot.seq_pos == slot.token_history.size()) {
                while (lcp < slot.seq_pos &&
                       lcp + 1 < prompt_tokens.size() &&
                       slot.token_history[lcp] == prompt_tokens[lcp]) {
                    lcp++;
                }
            }
            if (lcp > 0) {
                slot.seq_pos = static_cast<uint32_t>(lcp);
                slot.token_history.resize(lcp);
            } else {
                slot.clear();
            }
        }

        // Prefill remaining prompt tokens through real forward pass (only compute LM head logits on final prompt token)
        for (size_t i = lcp; i < prompt_tokens.size(); ++i) {
            const bool is_last = (i + 1 == prompt_tokens.size());
            logits = forward_token(prompt_tokens[i], slot_id, &last_prof, is_last);
        }
    } else {
        // Continue assistant: use last known token in history for forward step
        uint32_t last_token = tokenizer_.bos_id();
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            const auto& hist = get_or_create_slot(slot_id).token_history;
            if (!hist.empty()) last_token = hist.back();
        }
        logits = forward_token(last_token, slot_id, &last_prof);
    }

    if (logits.empty()) {
        // Prefill failed — there is nothing to sample from. Return here
        // instead of sampling token 0 in a loop (the old behaviour when
        // a forward returned empty logits).
        std::cerr << "[GENERATE] stopping: prefill produced no logits\n";
        return;
    }

    StreamingUTF8Decoder utf8_dec(tokenizer_);
    bool in_thinking = false;

    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        if (enable_context_compaction) {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            auto& slot = get_or_create_slot(slot_id);
            // Compact against the KV-cache capacity, not max_seq_len:
            // the capacity is what the slots actually allocate (RAM
            // guard can halve it far below max_seq_len), and passing
            // max_seq_len/2 as keep_recent used to underflow inside
            // compact() when the capacity was smaller.
            if (slot.seq_pos + 16 >= hp_.kv_cache_capacity) {
                slot.compact(hp_, hp_.kv_cache_capacity / 2);
            }
        }

        std::vector<uint32_t> hist;
        {
            std::lock_guard<std::mutex> lk(engine_mtx_);
            hist = get_or_create_slot(slot_id).token_history;
        }

        if (logits.empty()) {
            // A forward inside the loop failed: stop instead of sampling
            // token 0 from empty logits until max_new_tokens.
            break;
        }

        uint32_t next_id = sample_next_token(logits, hist, sampler_cfg);
        if (is_stop_token(next_id)) {
            break;
        }

        if (next_id == tokenizer_.think_start_id()) {
            in_thinking = true;
            logits = forward_token(next_id, slot_id, &last_prof);
            continue;
        }
        if (next_id == tokenizer_.think_end_id()) {
            in_thinking = false;
            logits = forward_token(next_id, slot_id, &last_prof);
            continue;
        }

        std::string piece = utf8_dec.push(next_id);
        if (!piece.empty()) {
            if (piece.find("<|im_end|>") != std::string::npos ||
                piece.find("<|endoftext|>") != std::string::npos ||
                piece.find("<|end_of_text|>") != std::string::npos ||
                piece.find("<|eot_id|>") != std::string::npos ||
                piece.find("<|end_of_turn|>") != std::string::npos ||
                piece.find("<end_of_turn>") != std::string::npos ||
                piece.find("<turn|>") != std::string::npos ||
                piece.find("<eos>") != std::string::npos) {
                break;
            }
            // Also detect literal "<think>" or "</think>" tags if emitted via subwords
            if (piece == "<think>" || piece == "<|channel>") {
                in_thinking = true;
            } else if (piece == "</think>" || piece == "<channel|>") {
                in_thinking = false;
            } else {
                if (!on_token(piece, in_thinking, next_id, last_prof)) {
                    cancelled = true;
                    break;
                }
            }
        }

        // Multi-Token Prediction (MTP) Speculative Prompt-Lookup N-Gram Verification
        if (enable_speculative && draft_count > 0 && (step + 1 < max_new_tokens)) {
            logits = forward_token(next_id, slot_id, &last_prof);

            std::vector<uint32_t> cur_hist;
            {
                std::lock_guard<std::mutex> lk(engine_mtx_);
                cur_hist = get_or_create_slot(slot_id).token_history;
            }

            // Prompt-lookup 2-gram / 1-gram draft proposal from recent history
            std::vector<uint32_t> drafts;
            if (cur_hist.size() >= 4) {
                for (size_t idx = cur_hist.size() - 2; idx >= 1; --idx) {
                    if (cur_hist[idx] == next_id && cur_hist[idx - 1] == cur_hist[cur_hist.size() - 2]) {
                        for (uint32_t d = 1; d <= draft_count && (idx + d) < cur_hist.size() - 1; ++d) {
                            drafts.push_back(cur_hist[idx + d]);
                        }
                        break;
                    }
                }
            }

            if (!drafts.empty()) {
                for (size_t d = 0; d < drafts.size() && (step + 1 < max_new_tokens); ++d) {
                    uint32_t target_tok = sample_next_token(logits, cur_hist, sampler_cfg);
                    if (target_tok != drafts[d]) {
                        // Draft rejected: stop speculative run cleanly without corrupting KV cache
                        break;
                    }
                    if (is_stop_token(target_tok) ||
                        target_tok == tokenizer_.think_start_id() || target_tok == tokenizer_.think_end_id()) {
                        break;
                    }
                    std::string acc_piece = utf8_dec.push(target_tok);
                    if (!acc_piece.empty() && acc_piece != "<think>" && acc_piece != "</think>") {
                        if (!on_token(acc_piece, in_thinking, target_tok, last_prof)) {
                            cancelled = true;
                            step = max_new_tokens;
                            break;
                        }
                    }
                    step++;
                    logits = forward_token(target_tok, slot_id, &last_prof);
                    cur_hist.push_back(target_tok);
                }
            }
        } else if (step + 1 < max_new_tokens) {
            logits = forward_token(next_id, slot_id, &last_prof);
        }
    }

    std::string tail = utf8_dec.flush();
    // After a cancellation the caller asked for no more tokens — do not
    // flush the decoder tail into the callback (the old code did).
    if (!cancelled && !tail.empty()) {
        on_token(tail, in_thinking, 0, last_prof);
    }
}

std::vector<SystemOneOptionScore> MoETransformerEngine::evaluate_system_one(
    const std::string& document,
    const std::string& query,
    const std::vector<std::string>& options,
    ProfileSample* out_profile
) {
    // Use isolated scratch slot 9999 for zero-token-generation decision evaluation
    const uint32_t sys1_slot = 9999;
    reset_slot(sys1_slot);

    std::string combined = document + "\n" + query + "\n";
    std::vector<uint32_t> tokens = tokenizer_.encode(combined, true);
    if (tokens.empty()) tokens.push_back(tokenizer_.bos_id());

    std::vector<float> logits;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool is_last = (i + 1 == tokens.size());
        logits = forward_token(tokens[i], sys1_slot, out_profile, is_last);
    }

    std::vector<SystemOneOptionScore> results;
    if (options.empty() || logits.empty()) return results;

    float max_logit = -1e30f;
    for (const auto& opt : options) {
        auto opt_toks = tokenizer_.encode(opt, false);
        uint32_t first_tid = opt_toks.empty() ? 0 : opt_toks[0];
        // Aggregate mean logit across the candidate option's tokens from final LM head projection
        float score = 0.0f;
        for (uint32_t tid : opt_toks) {
            if (tid < logits.size()) score += logits[tid];
        }
        if (!opt_toks.empty()) score /= static_cast<float>(opt_toks.size());

        max_logit = std::max(max_logit, score);
        results.push_back({opt, score, 0.0f, first_tid});
    }

    float sum_exp = 0.0f;
    for (auto& r : results) {
        r.probability = std::exp(r.raw_logit - max_logit);
        sum_exp += r.probability;
    }
    for (auto& r : results) {
        r.probability = (sum_exp > 0.0f) ? (r.probability / sum_exp) : (1.0f / results.size());
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });

    return results;
}

std::vector<float> MoETransformerEngine::compute_embeddings(const std::string& text) {
    const uint32_t emb_slot = 9998;
    reset_slot(emb_slot);

    std::vector<uint32_t> tokens = tokenizer_.encode(text, true);
    if (tokens.empty()) tokens.push_back(tokenizer_.bos_id());

    std::vector<float> pooled(hp_.dim, 0.0f);
    std::vector<float> hidden;
    for (uint32_t tid : tokens) {
        forward_token(tid, emb_slot, nullptr, false, &hidden);
        for (uint32_t d = 0; d < hp_.dim && d < hidden.size(); ++d) {
            pooled[d] += hidden[d];
        }
    }

    const float inv_n = 1.0f / static_cast<float>(tokens.size());
    float norm_sq = 0.0f;
    for (float& v : pooled) {
        v *= inv_n;
        norm_sq += v * v;
    }
    const float inv_norm = (norm_sq > 1e-12f) ? (1.0f / std::sqrt(norm_sq)) : 1.0f;
    for (float& v : pooled) v *= inv_norm;

    return pooled;
}

} // namespace moe

