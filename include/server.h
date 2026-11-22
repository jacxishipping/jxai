#pragma once

#include "transformer.h"
#include "diffusion.h"
#include <string>
#include <atomic>
#include <mutex>

namespace moe {

// ---------------------------------------------------------------------------
// JSON helpers (defined in server.cpp). These are unit-testable on purpose:
// the server's old substring extractors could be fooled by a key appearing
// inside a string value or a nested object, so every extractor below only
// matches keys at depth 1 of the outermost JSON object, and strings are
// decoded with full JSON escapes (including \uXXXX -> UTF-8). Numbers use
// strict JSON grammar (NaN/Infinity rejected).
// ---------------------------------------------------------------------------

// Strict JSON validation: true when `text` parses as a JSON object
// (recursive descent, depth-limited). Used to reject malformed request
// bodies with a 400 before any handler runs.
bool json_is_valid_object(const std::string& text);

// Top-level-only field extractors. Missing / wrong-typed / malformed
// fields return `def`.
std::string extract_json_string(const std::string& body, const std::string& key,
                                const std::string& def = "");
double extract_json_number(const std::string& body, const std::string& key, double def);
bool extract_json_bool(const std::string& body, const std::string& key, bool def);
std::vector<std::string> extract_json_string_array(const std::string& body, const std::string& key);

// Parse the OpenAI-style "messages" array into {role, content} pairs.
// Roles/contents are taken from each message object's own depth-1 fields,
// so a "role" string embedded in message content cannot smuggle a role.
std::vector<ChatMessage> parse_chat_messages(const std::string& body);

// ---------------------------------------------------------------------------
// Auth / CORS / validation helpers (pure, unit-testable).
// ---------------------------------------------------------------------------

// Mutating routes that require a key whenever one is configured.
bool route_requires_auth(const std::string& method, const std::string& path);

// Check Authorization: Bearer <key> or x-api-key: <key> against `key`.
bool request_authorized(const std::string& headers, const std::string& key);

// Case-insensitive single header lookup in a raw header block. Matches
// header names at line starts only, so a header VALUE containing
// "x-api-key: ..." can never be mistaken for the real header.
std::string header_value(const std::string& headers, const std::string& name);

// CORS: returns `origin` when it is in the comma-separated allow-list,
// otherwise "". An empty allow-list means same-origin only (no
// Access-Control-Allow-Origin header is emitted at all); the project UI is
// served from the engine itself (or via the vite dev proxy), so it never
// needs a wildcard. Configure with MOE_CORS_ORIGINS (env) or "cors_origins"
// in moe_config.json.
std::string cors_allowed_origin(const std::string& origin,
                                const std::string& allowed_origins_csv);

// Sampler validation: false + err on NaN/Inf or out-of-range input.
// Ranges: temperature [0, 5], top_p (0, 1], top_k [0, 1000],
// repetition_penalty [0, 2].
bool validate_sampler_params(double temperature, double top_p, double top_k,
                             double repetition_penalty, std::string& err);

// Server-config validation for POST /v1/config (same bounds the config
// writer enforces). api_host must be a plain hostname / IP literal.
bool validate_server_config(int paged_blocks, int active_slots, int max_context,
                            int api_port, int spec_tokens, double temperature,
                            double top_p, const std::string& api_host,
                            std::string& err);

class HttpServer {
public:
    HttpServer(MoETransformerEngine& engine, const std::string& web_root = "web/dist");
    ~HttpServer();

    // Start listening on `port` (blocking loop until stop() is called).
    // Binds to `host` (default 127.0.0.1). A non-loopback host with no API
    // key configured is refused unless `allow_insecure_remote` is true.
    bool start(uint16_t port, std::string& err_msg,
               const std::string& host = "127.0.0.1",
               bool allow_insecure_remote = false);
    void stop();

private:
    void handle_client(uintptr_t client_sock);

    MoETransformerEngine& engine_;
    std::string web_root_;
    uint16_t port_ = 8000;
    std::string host_ = "127.0.0.1";
    // Empty = auth disabled (localhost-only use). Set from MOE_API_KEY env
    // or moe_config.json "api_key" at start().
    std::string api_key_;
    // Comma-separated CORS allow-list (MOE_CORS_ORIGINS env or
    // moe_config.json "cors_origins"). Empty = same-origin only.
    std::string cors_origins_;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> active_requests_{0};
    // One mutex per user KV slot (0-63). generate_stream() reuses a slot's
    // KV cache across its LCP computation, prefill and decode, and assumes
    // the caller serializes generations on the same slot; two concurrent
    // chats on one slot interleaved their slot state and both came back
    // garbled (verified live: parallel "5+5?" / "7+7?" on the default slot
    // returned each other's fragments). The chat handler holds the slot's
    // mutex for the whole generation; different slots still run in
    // parallel. Lock order is always slot-mutex -> engine_mtx_ (the engine
    // takes its own mutex internally), so no deadlock cycle exists.
    std::mutex slot_mutexes_[64];
    uintptr_t listen_sock_ = ~static_cast<uintptr_t>(0);
};

} // namespace moe

