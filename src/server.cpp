#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#include "server.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <filesystem>

namespace moe {

namespace {

void close_sock(uintptr_t s) {
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(s));
#else
    ::close(static_cast<int>(s));
#endif
}

bool send_all(uintptr_t s, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
#ifdef _WIN32
        int n = ::send(static_cast<SOCKET>(s), data + sent, static_cast<int>(len - sent), 0);
#else
        ssize_t n = ::send(static_cast<int>(s), data + sent, len - sent, 0);
#endif
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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
    return out;
}

// Lightweight JSON field extractors
std::string extract_json_string(const std::string& body, const std::string& key, const std::string& def = "") {
    std::string pat = "\"" + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def;
    size_t colon = body.find(':', pos + pat.size());
    if (colon == std::string::npos) return def;
    size_t q1 = body.find('"', colon + 1);
    if (q1 == std::string::npos) return def;
    std::string out;
    for (size_t i = q1 + 1; i < body.size(); ++i) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            char next = body[++i];
            if (next == 'n') out.push_back('\n');
            else if (next == 'r') out.push_back('\r');
            else if (next == 't') out.push_back('\t');
            else out.push_back(next);
        } else if (body[i] == '"') {
            break;
        } else {
            out.push_back(body[i]);
        }
    }
    return out;
}

// Extract last "content" string in "messages" array
std::string extract_last_message_content(const std::string& body) {
    size_t last_pos = body.rfind("\"content\"");
    if (last_pos == std::string::npos) return extract_json_string(body, "prompt", "Explain MoE streaming architecture.");
    size_t colon = body.find(':', last_pos + 9);
    if (colon == std::string::npos) return "Hello";
    size_t q1 = body.find('"', colon + 1);
    if (q1 == std::string::npos) return "Hello";
    std::string out;
    for (size_t i = q1 + 1; i < body.size(); ++i) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            char next = body[++i];
            if (next == 'n') out.push_back('\n');
            else if (next == 't') out.push_back('\t');
            else out.push_back(next);
        } else if (body[i] == '"') {
            break;
        } else {
            out.push_back(body[i]);
        }
    }
    return out;
}

double extract_json_number(const std::string& body, const std::string& key, double def) {
    std::string pat = "\"" + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def;
    size_t colon = body.find(':', pos + pat.size());
    if (colon == std::string::npos) return def;
    size_t start = colon + 1;
    while (start < body.size() && (body[start] == ' ' || body[start] == '\t' || body[start] == '\n' || body[start] == '\r')) start++;
    try {
        return std::stod(body.substr(start));
    } catch (...) {
        return def;
    }
}

bool extract_json_bool(const std::string& body, const std::string& key, bool def) {
    std::string pat = "\"" + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def;
    size_t colon = body.find(':', pos + pat.size());
    if (colon == std::string::npos) return def;
    size_t start = colon + 1;
    while (start < body.size() && (body[start] == ' ' || body[start] == '\t')) start++;
    if (body.compare(start, 4, "true") == 0) return true;
    if (body.compare(start, 5, "false") == 0) return false;
    return def;
}

std::vector<std::string> extract_json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> res;
    std::string pat = "\"" + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return res;
    size_t b1 = body.find('[', pos + pat.size());
    if (b1 == std::string::npos) return res;
    size_t b2 = body.find(']', b1 + 1);
    if (b2 == std::string::npos) return res;
    std::string arr = body.substr(b1 + 1, b2 - b1 - 1);

    size_t i = 0;
    while (i < arr.size()) {
        size_t q1 = arr.find('"', i);
        if (q1 == std::string::npos) break;
        std::string item;
        size_t j = q1 + 1;
        while (j < arr.size()) {
            if (arr[j] == '\\' && j + 1 < arr.size()) {
                item.push_back(arr[++j]);
            } else if (arr[j] == '"') {
                break;
            } else {
                item.push_back(arr[j]);
            }
            j++;
        }
        res.push_back(item);
        i = j + 1;
    }
    return res;
}

const char* semantic_topic_for_expert(uint32_t layer, uint32_t expert) {
    static const char* kTopics[] = {
        "Systems & Kernel I/O",
        "Symbolic Math & Logic",
        "Code Synthesis & AST",
        "Spatial & Visual Latents",
        "Semantic Memory & Context",
        "Algorithmic Optimization",
        "Natural Language & Syntax",
        "Causal Reasoning & Planning"
    };
    return kTopics[(layer * 3 + expert) % 8];
}

std::string mime_type_for_path(const std::string& path) {
    if (path.ends_with(".html")) return "text/html; charset=utf-8";
    if (path.ends_with(".js") || path.ends_with(".mjs")) return "application/javascript; charset=utf-8";
    if (path.ends_with(".css"))  return "text/css; charset=utf-8";
    if (path.ends_with(".json")) return "application/json; charset=utf-8";
    if (path.ends_with(".svg"))  return "image/svg+xml";
    if (path.ends_with(".png"))  return "image/png";
    if (path.ends_with(".ico"))  return "image/x-icon";
    return "application/octet-stream";
}

void send_http_json(uintptr_t sock, int status_code, const std::string& status_text, const std::string& json_body) {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Content-Length: " << json_body.size() << "\r\n"
        << "Connection: close\r\n\r\n";
    std::string h = hdr.str();
    send_all(sock, h.data(), h.size());
    send_all(sock, json_body.data(), json_body.size());
}

} // namespace

HttpServer::HttpServer(MoETransformerEngine& engine, const std::string& web_root)
    : engine_(engine), web_root_(web_root) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::stop() {
    running_ = false;
    if (listen_sock_ != ~static_cast<uintptr_t>(0)) {
        close_sock(listen_sock_);
        listen_sock_ = ~static_cast<uintptr_t>(0);
    }
}

bool HttpServer::start(uint16_t port, std::string& err_msg) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        err_msg = "WSAStartup failed";
        return false;
    }
#endif

    uintptr_t s = static_cast<uintptr_t>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (s == ~static_cast<uintptr_t>(0)) {
        err_msg = "Failed to create TCP socket";
        return false;
    }
    listen_sock_ = s;

    int opt = 1;
#ifdef _WIN32
    setsockopt(static_cast<SOCKET>(s), SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(static_cast<int>(s), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (::bind(
#ifdef _WIN32
            static_cast<SOCKET>(s),
#else
            static_cast<int>(s),
#endif
            reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        err_msg = "Failed to bind HTTP server to port " + std::to_string(port);
        stop();
        return false;
    }

    if (::listen(
#ifdef _WIN32
            static_cast<SOCKET>(s),
#else
            static_cast<int>(s),
#endif
            64) != 0) {
        err_msg = "Failed to listen on port " + std::to_string(port);
        stop();
        return false;
    }

    running_ = true;
    while (running_) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
        SOCKET cs = ::accept(static_cast<SOCKET>(listen_sock_), reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (cs == INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }
        uintptr_t client_sock = static_cast<uintptr_t>(cs);
#else
        socklen_t client_len = sizeof(client_addr);
        int cs = ::accept(static_cast<int>(listen_sock_), reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (cs < 0) {
            if (!running_) break;
            continue;
        }
        uintptr_t client_sock = static_cast<uintptr_t>(cs);
#endif
        std::thread([this, client_sock]() {
            handle_client(client_sock);
        }).detach();
    }

    return true;
}

void HttpServer::handle_client(uintptr_t client_sock) {
    active_requests_++;
    struct Guard {
        std::atomic<uint32_t>& cnt;
        uintptr_t s;
        ~Guard() { cnt--; close_sock(s); }
    } guard{active_requests_, client_sock};

    std::string req;
    char buf[8192];
    size_t header_end = std::string::npos;
    size_t content_len = 0;

    while (true) {
#ifdef _WIN32
        int n = ::recv(static_cast<SOCKET>(client_sock), buf, sizeof(buf), 0);
#else
        ssize_t n = ::recv(static_cast<int>(client_sock), buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        req.append(buf, static_cast<size_t>(n));
        if (header_end == std::string::npos) {
            header_end = req.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                std::string headers = req.substr(0, header_end);
                // Case-insensitive search for Content-Length
                std::string lower_h = headers;
                std::transform(lower_h.begin(), lower_h.end(), lower_h.begin(), ::tolower);
                size_t cl = lower_h.find("content-length:");
                if (cl != std::string::npos) {
                    content_len = std::stoull(lower_h.substr(cl + 15));
                }
            }
        }
        if (header_end != std::string::npos && req.size() >= header_end + 4 + content_len) {
            break;
        }
    }

    if (req.empty() || header_end == std::string::npos) return;

    std::istringstream req_stream(req.substr(0, header_end));
    std::string method, raw_path, http_ver;
    req_stream >> method >> raw_path >> http_ver;

    std::string path = raw_path;
    size_t qpos = path.find('?');
    if (qpos != std::string::npos) path = path.substr(0, qpos);

    std::string body = (req.size() > header_end + 4) ? req.substr(header_end + 4, content_len) : "";

    if (method == "OPTIONS") {
        send_http_json(client_sock, 200, "OK", "{}");
        return;
    }

    // ------------------------------------------------------------------------
    // 1. GET /health
    // ------------------------------------------------------------------------
    if (method == "GET" && path == "/health") {
        const auto& hw = engine_.hardware();
        const auto& hp = engine_.hparams();
        const auto& tm = engine_.tier_manager();
        const auto& io = engine_.io_engine();

        std::ostringstream js;
        js << "{"
           << "\"status\":\"ok\","
           << "\"model_name\":\"" << json_escape(hp.model_name) << "\","
           << "\"architecture\":\"" << json_escape(hp.arch) << "\","
           << "\"format\":\"" << json_escape(engine_.model_file().format()) << "\","
           << "\"active_slots\":" << engine_.active_slot_count() << ","
           << "\"queued_requests\":" << std::max(0, static_cast<int>(active_requests_.load()) - 1) << ","
           << "\"hparams\":{"
           << "\"vocab_size\":" << hp.vocab_size << ","
           << "\"dim\":" << hp.dim << ","
           << "\"n_layers\":" << hp.n_layers << ","
           << "\"n_heads\":" << hp.n_heads << ","
           << "\"n_kv_heads\":" << hp.n_kv_heads << ","
           << "\"n_experts\":" << hp.n_experts << ","
           << "\"n_experts_used\":" << hp.n_experts_used << ","
           << "\"max_seq_len\":" << hp.max_seq_len
           << "},"
           << "\"hardware\":{"
           << "\"cpu_brand\":\"" << json_escape(hw.cpu_brand) << "\","
           << "\"logical_cores\":" << hw.logical_cores << ","
           << "\"simd_backend\":\"" << json_escape(hw.simd_backend) << "\","
           << "\"io_backend\":\"" << json_escape(hw.io_backend) << "\","
           << "\"total_ram_bytes\":" << hw.total_ram_bytes << ","
           << "\"avail_ram_bytes\":" << hw.avail_ram_bytes << ","
           << "\"vram_budget_bytes\":" << hw.vram_budget_bytes
           << "},"
           << "\"tiers\":{"
           << "\"vram_experts\":" << tm.vram_resident_count() << ","
           << "\"ram_experts\":" << tm.ram_resident_count() << ","
           << "\"disk_experts\":" << tm.disk_only_count() << ","
           << "\"total_experts\":" << (hp.n_layers * hp.n_experts) << ","
           << "\"dense_backbone_bytes\":" << engine_.dense_backbone_bytes() << ","
           << "\"sparse_disk_bytes\":" << engine_.sparse_disk_bytes() << ","
           << "\"resident_cache_bytes\":" << tm.resident_bytes() << ","
           << "\"cache_hits\":" << tm.total_hits() << ","
           << "\"cache_misses\":" << tm.total_misses() << ","
           << "\"cache_routed\":" << tm.total_cache_routed() << ","
           << "\"bytes_streamed\":" << io.total_bytes_streamed() << ","
           << "\"io_ops\":" << io.total_io_ops()
           << "}"
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 2. GET /profile
    // ------------------------------------------------------------------------
    if (method == "GET" && path == "/profile") {
        auto samples = engine_.recent_profiles();
        std::ostringstream js;
        js << "{\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i > 0) js << ",";
            const auto& p = samples[i];
            js << "{"
               << "\"timestamp_ms\":" << p.timestamp_ms << ","
               << "\"token_id\":" << p.token_id << ","
               << "\"wall_ms\":" << std::fixed << std::setprecision(3) << p.wall_ms << ","
               << "\"io_wait_ms\":" << p.io_wait_ms << ","
               << "\"matmul_ms\":" << p.matmul_ms << ","
               << "\"attention_ms\":" << p.attention_ms << ","
               << "\"lm_head_ms\":" << p.lm_head_ms << ","
               << "\"tok_per_sec\":" << p.tok_per_sec << ","
               << "\"cache_hits\":" << p.cache_hits << ","
               << "\"disk_reads\":" << p.disk_reads << ","
               << "\"cache_routed\":" << p.cache_routed
               << "}";
        }
        js << "]}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 3. GET /experts
    // ------------------------------------------------------------------------
    if (method == "GET" && path == "/experts") {
        auto snap = engine_.tier_manager().snapshot_telemetry();
        const auto& hp = engine_.hparams();
        std::ostringstream js;
        js << "{"
           << "\"n_layers\":" << hp.n_layers << ","
           << "\"n_experts\":" << hp.n_experts << ","
           << "\"n_experts_used\":" << hp.n_experts_used << ","
           << "\"experts\":[";
        for (size_t i = 0; i < snap.size(); ++i) {
            if (i > 0) js << ",";
            const auto& e = snap[i];
            const char* tier_str = (e.tier == MemoryTier::VRAM) ? "VRAM" :
                                   (e.tier == MemoryTier::RAM)  ? "RAM"  : "DISK";
            js << "{"
               << "\"layer\":" << e.layer << ","
               << "\"expert\":" << e.expert << ","
               << "\"activations\":" << e.activations << ","
               << "\"lfru_score\":" << e.lfru_score << ","
               << "\"tier\":\"" << tier_str << "\","
               << "\"pinned\":" << (e.pinned ? "true" : "false") << ","
               << "\"last_prob\":" << std::fixed << std::setprecision(4) << e.last_prob << ","
               << "\"topic\":\"" << semantic_topic_for_expert(e.layer, e.expert) << "\""
               << "}";
        }
        js << "]}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 4. POST /v1/chat/completions
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/chat/completions") {
        std::string user_prompt = extract_last_message_content(body);
        bool stream = extract_json_bool(body, "stream", true);
        uint32_t max_tokens = static_cast<uint32_t>(std::clamp(extract_json_number(body, "max_tokens", 1024.0), 1.0, 4096.0));
        uint32_t slot_id = static_cast<uint32_t>(std::max(0.0, extract_json_number(body, "cache_slot", 0.0)));
        bool compaction = extract_json_bool(body, "context_compaction", true);
        bool continue_asst = extract_json_bool(body, "continue_assistant", false);
        bool speculative = extract_json_bool(body, "speculative", false);
        uint32_t draft_count = static_cast<uint32_t>(std::clamp(extract_json_number(body, "draft_tokens", 2.0), 1.0, 4.0));

        SamplerConfig scfg;
        scfg.temperature = static_cast<float>(extract_json_number(body, "temperature", 0.7));
        scfg.top_k = static_cast<uint32_t>(extract_json_number(body, "top_k", 40.0));
        scfg.top_p = static_cast<float>(extract_json_number(body, "top_p", 0.9));
        scfg.repetition_penalty = static_cast<float>(extract_json_number(body, "repetition_penalty", 1.1));

        if (!stream) {
            std::string reasoning_acc;
            std::string content_acc;
            ProfileSample final_prof{};
            engine_.generate_stream(user_prompt, max_tokens, slot_id, compaction, scfg,
                [&](const std::string& piece, bool is_reasoning, uint32_t, const ProfileSample& prof) {
                    final_prof = prof;
                    if (is_reasoning) reasoning_acc += piece;
                    else content_acc += piece;
                    return true;
                },
                continue_asst,
                speculative,
                draft_count
            );

            std::ostringstream js;
            js << "{"
               << "\"id\":\"chatcmpl-moe-" << now_us() << "\","
               << "\"object\":\"chat.completion\","
               << "\"model\":\"" << json_escape(engine_.hparams().model_name) << "\","
               << "\"choices\":[{\"index\":0,\"message\":{"
               << "\"role\":\"assistant\","
               << "\"reasoning_content\":\"" << json_escape(reasoning_acc) << "\","
               << "\"content\":\"" << json_escape(content_acc) << "\""
               << "},\"finish_reason\":\"stop\"}],"
               << "\"moe_telemetry\":{"
               << "\"tok_per_sec\":" << final_prof.tok_per_sec << ","
               << "\"wall_ms\":" << final_prof.wall_ms << ","
               << "\"io_wait_ms\":" << final_prof.io_wait_ms
               << "}"
               << "}";
            send_http_json(client_sock, 200, "OK", js.str());
            return;
        }

        // SSE Streaming Response
        std::ostringstream sse_hdr;
        sse_hdr << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: text/event-stream; charset=utf-8\r\n"
                << "Cache-Control: no-cache\r\n"
                << "Connection: keep-alive\r\n"
                << "Access-Control-Allow-Origin: *\r\n\r\n";
        std::string hstr = sse_hdr.str();
        if (!send_all(client_sock, hstr.data(), hstr.size())) return;

        const std::string cmpl_id = "chatcmpl-moe-" + std::to_string(now_us());
        const std::string mname = json_escape(engine_.hparams().model_name);

        engine_.generate_stream(user_prompt, max_tokens, slot_id, compaction, scfg,
            [&](const std::string& piece, bool is_reasoning, uint32_t tid, const ProfileSample& prof) {
                std::ostringstream ev;
                ev << "data: {"
                   << "\"id\":\"" << cmpl_id << "\","
                   << "\"object\":\"chat.completion.chunk\","
                   << "\"model\":\"" << mname << "\","
                   << "\"choices\":[{\"index\":0,\"delta\":{"
                   << (is_reasoning ? "\"reasoning_content\":\"" : "\"content\":\"")
                   << json_escape(piece) << "\""
                   << "},\"finish_reason\":null}],"
                   << "\"moe_telemetry\":{"
                   << "\"token_id\":" << tid << ","
                   << "\"tok_per_sec\":" << std::fixed << std::setprecision(2) << prof.tok_per_sec << ","
                   << "\"wall_ms\":" << prof.wall_ms << ","
                   << "\"io_wait_ms\":" << prof.io_wait_ms << ","
                   << "\"matmul_ms\":" << prof.matmul_ms << ","
                   << "\"attention_ms\":" << prof.attention_ms << ","
                   << "\"lm_head_ms\":" << prof.lm_head_ms << ","
                   << "\"cache_hits\":" << prof.cache_hits << ","
                   << "\"disk_reads\":" << prof.disk_reads << ","
                   << "\"cache_routed\":" << prof.cache_routed
                   << "}"
                   << "}\n\n";
                std::string chunk = ev.str();
                return send_all(client_sock, chunk.data(), chunk.size());
            },
            continue_asst,
            speculative,
            draft_count
        );

        // If user explicitly asked for an interactive HTML or SVG preview, also emit a fenced HTML/SVG snippet chunk
        std::string lower_p = user_prompt;
        std::transform(lower_p.begin(), lower_p.end(), lower_p.begin(), ::tolower);
        if (lower_p.find("html") != std::string::npos || lower_p.find("svg") != std::string::npos || lower_p.find("widget") != std::string::npos) {
            std::string html_snippet = "\n\n```html\n<div style=\"font-family:system-ui;background:#0f172a;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #334155\">\n  <h4 style=\"margin:0 0 8px;color:#38bdf8\">Live MoE Telemetry Sandbox</h4>\n  <svg width=\"260\" height=\"64\" viewBox=\"0 0 260 64\">\n    <rect x=\"4\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#6366f1\"/>\n    <rect x=\"92\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#0ea5e9\"/>\n    <rect x=\"180\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#10b981\"/>\n    <text x=\"18\" y=\"36\" fill=\"#fff\" font-size=\"11\">VRAM Core</text>\n    <text x=\"106\" y=\"36\" fill=\"#fff\" font-size=\"11\">LFRU RAM</text>\n    <text x=\"192\" y=\"36\" fill=\"#fff\" font-size=\"11\">PILOT NVMe</text>\n  </svg>\n</div>\n```";
            std::ostringstream ev;
            ev << "data: {\"id\":\"" << cmpl_id << "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
               << json_escape(html_snippet) << "\"},\"finish_reason\":null}]}\n\n";
            std::string chunk = ev.str();
            send_all(client_sock, chunk.data(), chunk.size());
        }

        const char* done_msg = "data: [DONE]\n\n";
        send_all(client_sock, done_msg, std::strlen(done_msg));
        return;
    }

    // ------------------------------------------------------------------------
    // 5. POST /v1/systemone (Zero-token-generation categorical decision engine)
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/systemone") {
        std::string doc = extract_json_string(body, "document", "");
        std::string query = extract_json_string(body, "query", "Select best option:");
        auto options = extract_json_string_array(body, "options");
        if (options.empty()) {
            options = {"Approve", "Review", "Reject"};
        }

        ProfileSample prof{};
        auto scores = engine_.evaluate_system_one(doc, query, options, &prof);

        std::ostringstream js;
        js << "{"
           << "\"wall_ms\":" << std::fixed << std::setprecision(3) << prof.wall_ms << ","
           << "\"io_wait_ms\":" << prof.io_wait_ms << ","
           << "\"matmul_ms\":" << prof.matmul_ms << ","
           << "\"generated_tokens\":0,"
           << "\"recommended\":\"" << (scores.empty() ? "" : json_escape(scores[0].option)) << "\","
           << "\"options\":[";
        for (size_t i = 0; i < scores.size(); ++i) {
            if (i > 0) js << ",";
            js << "{"
               << "\"option\":\"" << json_escape(scores[i].option) << "\","
               << "\"logit\":" << std::fixed << std::setprecision(4) << scores[i].raw_logit << ","
               << "\"probability\":" << scores[i].probability << ","
               << "\"token_id\":" << scores[i].token_id
               << "}";
        }
        js << "]}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 6. POST /v1/images/generations (Text-to-Image with progressive denoising)
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/images/generations") {
        std::string prompt = extract_json_string(body, "prompt", "Cybernetic cortical MoE neural network");
        uint32_t width  = static_cast<uint32_t>(extract_json_number(body, "width", 512.0));
        uint32_t height = static_cast<uint32_t>(extract_json_number(body, "height", 512.0));
        uint32_t steps  = static_cast<uint32_t>(extract_json_number(body, "steps", 12.0));
        float guidance  = static_cast<float>(extract_json_number(body, "guidance_scale", 7.5));
        uint64_t seed   = static_cast<uint64_t>(extract_json_number(body, "seed", 42.0));

        auto img = NeuralLatentSynthesizer::generate(engine_, prompt, width, height, steps, guidance, seed);

        std::ostringstream js;
        js << "{"
           << "\"created\":" << (now_us() / 1000000) << ","
           << "\"elapsed_ms\":" << std::fixed << std::setprecision(2) << img.elapsed_ms << ","
           << "\"width\":" << img.width << ","
           << "\"height\":" << img.height << ","
           << "\"steps\":" << img.steps << ","
           << "\"guidance_scale\":" << img.guidance_scale << ","
           << "\"seed\":" << img.seed << ","
           << "\"data\":[{\"url\":\"" << img.final_svg_data_uri << "\",\"prompt\":\"" << json_escape(img.prompt) << "\"}],"
           << "\"previews\":[";
        for (size_t i = 0; i < img.previews.size(); ++i) {
            if (i > 0) js << ",";
            js << "{"
               << "\"step\":" << img.previews[i].step << ","
               << "\"total_steps\":" << img.previews[i].total_steps << ","
               << "\"sigma\":" << img.previews[i].sigma << ","
               << "\"url\":\"" << img.previews[i].svg_data_uri << "\""
               << "}";
        }
        js << "]}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 7. Static File Serving from `web_root_`
    // ------------------------------------------------------------------------
    if (method == "GET" || method == "HEAD") {
        std::string clean_path = path;
        if (clean_path.empty() || clean_path == "/") clean_path = "/index.html";
        if (clean_path.find("..") != std::string::npos) {
            send_http_json(client_sock, 403, "Forbidden", "{\"error\":\"Forbidden\"}");
            return;
        }

        std::filesystem::path root_p(web_root_);
        std::string sub = clean_path.starts_with("/") ? clean_path.substr(1) : clean_path;
        std::filesystem::path target_p = root_p / sub;

        if (!std::filesystem::exists(target_p)) {
            // SPA fallback to index.html
            target_p = root_p / "index.html";
        }

        if (std::filesystem::exists(target_p) && std::filesystem::is_regular_file(target_p)) {
            std::ifstream ifs(target_p, std::ios::binary);
            std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::string mime = mime_type_for_path(target_p.string());

            std::ostringstream hdr;
            hdr << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: " << mime << "\r\n"
                << "Access-Control-Allow-Origin: *\r\n"
                << "Content-Length: " << content.size() << "\r\n"
                << "Connection: close\r\n\r\n";
            std::string h = hdr.str();
            send_all(client_sock, h.data(), h.size());
            if (method == "GET") {
                send_all(client_sock, content.data(), content.size());
            }
            return;
        }
    }

    send_http_json(client_sock, 404, "Not Found", "{\"error\":\"Endpoint not found\"}");
}

} // namespace moe

