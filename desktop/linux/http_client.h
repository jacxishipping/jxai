#pragma once
// POSIX HTTP + SSE client for the JXAI native engine.
// Mirrors desktop/src/api_client.h without WinSock.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace jxai::term {

struct HealthInfo {
    bool online = false;
    std::string model_name = "—";
    std::string architecture = "—";
    std::string simd = "—";
    std::string io_backend = "—";
    std::string format;
    uint32_t active_slots = 0;
    uint32_t n_layers = 0;
    uint32_t dim = 0;
    uint32_t max_seq_len = 0;
};

struct StreamTelem {
    double tok_per_sec = 0.0;
    double wall_ms = 0.0;
    double io_wait_ms = 0.0;
    double matmul_ms = 0.0;
    double attention_ms = 0.0;
    double lm_head_ms = 0.0;
    uint32_t cache_hits = 0;
    uint32_t disk_reads = 0;
};

struct ChatTurn {
    std::string role;    // "system" | "user" | "assistant"
    std::string content;
};

inline std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

inline std::string json_unescape_string(const std::string& src, size_t start) {
    std::string out;
    size_t q = start;
    while (q < src.size()) {
        if (src[q] == '\\' && q + 1 < src.size()) {
            char nx = src[++q];
            if (nx == 'n') out += '\n';
            else if (nx == 'r') out += '\r';
            else if (nx == 't') out += '\t';
            else if (nx == '"' || nx == '\\' || nx == '/') out += nx;
            else if (nx == 'u' && q + 4 < src.size()) {
                // Minimal \uXXXX — only handle ASCII-range for display.
                unsigned code = 0;
                bool ok = true;
                for (int i = 1; i <= 4; ++i) {
                    char h = src[q + i];
                    code <<= 4;
                    if (h >= '0' && h <= '9') code |= (h - '0');
                    else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                    else { ok = false; break; }
                }
                if (ok) {
                    q += 4;
                    if (code < 0x80) out += static_cast<char>(code);
                    else out += '?';
                } else {
                    out += nx;
                }
            } else {
                out += nx;
            }
        } else if (src[q] == '"') {
            break;
        } else {
            out += src[q];
        }
        ++q;
    }
    return out;
}

inline bool extract_json_string(const std::string& hay, const std::string& key, std::string& out) {
    const std::string pat = "\"" + key + "\":\"";
    size_t p = hay.find(pat);
    if (p == std::string::npos) return false;
    out = json_unescape_string(hay, p + pat.size());
    return true;
}

inline bool extract_json_number(const std::string& hay, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\":";
    size_t p = hay.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    while (p < hay.size() && (hay[p] == ' ' || hay[p] == '\t')) ++p;
    try {
        size_t n = 0;
        out = std::stod(hay.substr(p), &n);
        return n > 0;
    } catch (...) {
        return false;
    }
}

class EngineClient {
public:
    using TokenCb = std::function<void(const std::string& piece, bool reasoning, const StreamTelem& telem)>;
    using DoneCb = std::function<void(bool ok, const std::string& err)>;

    EngineClient(std::string host = "127.0.0.1", uint16_t port = 8001)
        : host_(std::move(host)), port_(port) {}

    ~EngineClient() { stop(); }

    void set_endpoint(const std::string& host, uint16_t port) {
        host_ = host;
        port_ = port;
    }

    std::string host() const { return host_; }
    uint16_t port() const { return port_; }

    bool streaming() const { return streaming_.load(); }

    void stop() {
        stop_ = true;
        int fd = active_fd_.exchange(-1);
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
        if (worker_.joinable()) worker_.join();
        streaming_ = false;
    }

    HealthInfo health() {
        HealthInfo h;
        std::string resp;
        if (!http_get("/health", resp) || resp.find("200") == std::string::npos) {
            h.online = false;
            return h;
        }
        h.online = true;
        extract_json_string(resp, "model_name", h.model_name);
        extract_json_string(resp, "architecture", h.architecture);
        extract_json_string(resp, "format", h.format);
        extract_json_string(resp, "simd_backend", h.simd);
        extract_json_string(resp, "io_backend", h.io_backend);
        double v = 0;
        if (extract_json_number(resp, "active_slots", v)) h.active_slots = static_cast<uint32_t>(v);
        if (extract_json_number(resp, "n_layers", v)) h.n_layers = static_cast<uint32_t>(v);
        if (extract_json_number(resp, "dim", v)) h.dim = static_cast<uint32_t>(v);
        if (extract_json_number(resp, "max_seq_len", v)) h.max_seq_len = static_cast<uint32_t>(v);
        // nested hparams fallback
        if (h.n_layers == 0 && extract_json_number(resp, "n_layers", v)) h.n_layers = static_cast<uint32_t>(v);
        return h;
    }

    // Clear a KV cache slot on the engine (POST /v1/slots/clear).
    bool clear_slot(uint32_t slot_id, const std::string& api_key, std::string& err) {
        const std::string body = "{\"slot_id\":" + std::to_string(slot_id) + "}";
        std::string resp;
        int status = 0;
        if (!http_post_json("/v1/slots/clear", body, api_key, resp, status)) {
            err = "cannot reach engine at " + host_ + ":" + std::to_string(port_);
            return false;
        }
        if (status != 200) {
            err = "clear slot HTTP " + std::to_string(status);
            std::string msg;
            if (extract_json_string(resp, "message", msg) || extract_json_string(resp, "error", msg))
                err += ": " + msg;
            return false;
        }
        return true;
    }

    // Hot-swap the active GGUF (POST /v1/models/load). Path must be under CWD.
    bool load_model(const std::string& model_path, const std::string& api_key, std::string& err) {
        const std::string body = "{\"model_path\":\"" + json_escape(model_path) + "\"}";
        std::string resp;
        int status = 0;
        if (!http_post_json("/v1/models/load", body, api_key, resp, status)) {
            err = "cannot reach engine at " + host_ + ":" + std::to_string(port_);
            return false;
        }
        if (status != 200) {
            err = "load model HTTP " + std::to_string(status);
            std::string msg;
            if (extract_json_string(resp, "error", msg) || extract_json_string(resp, "message", msg))
                err += ": " + msg;
            return false;
        }
        return true;
    }

    void stream_chat_async(const std::vector<ChatTurn>& messages,
                           uint32_t max_tokens,
                           float temperature,
                           uint32_t slot_id,
                           const std::string& api_key,
                           TokenCb on_token,
                           DoneCb on_done) {
        stop();
        stop_ = false;
        streaming_ = true;
        worker_ = std::thread([=, this]() {
            std::string err;
            bool ok = stream_chat_sync(messages, max_tokens, temperature, slot_id, api_key, on_token, err);
            streaming_ = false;
            if (on_done) on_done(ok, err);
        });
    }

private:
    int connect_fd() {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            // try localhost fallback for weird hosts
            if (::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1) {
                ::close(fd);
                return -1;
            }
        }
        // connect timeout via nonblock + select
        int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc < 0 && errno != EINPROGRESS) {
            ::close(fd);
            return -1;
        }
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);
        timeval tv{2, 0};
        rc = ::select(fd + 1, nullptr, &wfds, nullptr, &tv);
        if (rc <= 0) {
            ::close(fd);
            return -1;
        }
        int soerr = 0;
        socklen_t sl = sizeof(soerr);
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
        if (soerr != 0) {
            ::close(fd);
            return -1;
        }
        ::fcntl(fd, F_SETFL, flags); // back to blocking
        // read timeout so a stalled engine cannot hang the TUI forever
        timeval rcv{120, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
        return fd;
    }

    bool http_get(const std::string& path, std::string& out) {
        int fd = connect_fd();
        if (fd < 0) return false;
        std::ostringstream req;
        req << "GET " << path << " HTTP/1.1\r\n"
            << "Host: " << host_ << "\r\n"
            << "Connection: close\r\n\r\n";
        std::string r = req.str();
        if (::send(fd, r.data(), r.size(), 0) < 0) {
            ::close(fd);
            return false;
        }
        out.clear();
        char buf[4096];
        while (true) {
            ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            out.append(buf, static_cast<size_t>(n));
        }
        ::close(fd);
        return true;
    }

    bool http_post_json(const std::string& path, const std::string& body,
                        const std::string& api_key, std::string& out, int& status) {
        status = 0;
        int fd = connect_fd();
        if (fd < 0) return false;
        std::ostringstream req;
        req << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << host_ << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n";
        if (!api_key.empty()) {
            req << "Authorization: Bearer " << api_key << "\r\n";
        }
        req << "\r\n" << body;
        std::string r = req.str();
        if (::send(fd, r.data(), r.size(), 0) < 0) {
            ::close(fd);
            return false;
        }
        out.clear();
        char buf[4096];
        while (true) {
            ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            out.append(buf, static_cast<size_t>(n));
        }
        ::close(fd);
        if (out.rfind("HTTP/", 0) == 0) {
            size_t sp = out.find(' ');
            if (sp != std::string::npos) {
                try { status = std::stoi(out.substr(sp + 1, 3)); } catch (...) {}
            }
        }
        return true;
    }

    bool stream_chat_sync(const std::vector<ChatTurn>& messages,
                          uint32_t max_tokens,
                          float temperature,
                          uint32_t slot_id,
                          const std::string& api_key,
                          TokenCb on_token,
                          std::string& err) {
        int fd = connect_fd();
        if (fd < 0) {
            err = "cannot reach engine at " + host_ + ":" + std::to_string(port_);
            return false;
        }
        active_fd_ = fd;

        std::ostringstream body;
        body << "{\"messages\":[";
        for (size_t i = 0; i < messages.size(); ++i) {
            if (i) body << ',';
            body << "{\"role\":\"" << json_escape(messages[i].role)
                 << "\",\"content\":\"" << json_escape(messages[i].content) << "\"}";
        }
        body << "],\"stream\":true"
             << ",\"max_tokens\":" << max_tokens
             << ",\"temperature\":" << temperature
             << ",\"cache_slot\":" << slot_id
             << ",\"context_compaction\":true"
             << "}";
        std::string b = body.str();

        std::ostringstream req;
        req << "POST /v1/chat/completions HTTP/1.1\r\n"
            << "Host: " << host_ << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << b.size() << "\r\n"
            << "Connection: close\r\n";
        if (!api_key.empty()) {
            req << "Authorization: Bearer " << api_key << "\r\n";
        }
        req << "\r\n" << b;
        std::string r = req.str();
        if (::send(fd, r.data(), r.size(), 0) < 0) {
            err = "failed to send chat request";
            ::close(fd);
            active_fd_ = -1;
            return false;
        }

        std::string buffer;
        char chunk[4096];
        bool saw_http = false;
        int http_status = 0;

        while (!stop_.load()) {
            ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
            if (n <= 0) break;
            buffer.append(chunk, static_cast<size_t>(n));

            if (!saw_http) {
                size_t hdr = buffer.find("\r\n\r\n");
                if (hdr == std::string::npos) continue;
                // status line
                if (buffer.rfind("HTTP/", 0) == 0) {
                    size_t sp = buffer.find(' ');
                    if (sp != std::string::npos) {
                        try { http_status = std::stoi(buffer.substr(sp + 1, 3)); } catch (...) {}
                    }
                }
                buffer.erase(0, hdr + 4);
                saw_http = true;
                if (http_status != 0 && http_status != 200) {
                    err = "engine HTTP " + std::to_string(http_status);
                    // keep reading a bit of body for context
                    if (buffer.size() > 200) err += ": " + buffer.substr(0, 200);
                    ::close(fd);
                    active_fd_ = -1;
                    return false;
                }
            }

            size_t line_end;
            while ((line_end = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, line_end);
                buffer.erase(0, line_end + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("data:", 0) != 0) continue;
                std::string data = line.substr(5);
                while (!data.empty() && data[0] == ' ') data.erase(data.begin());
                if (data == "[DONE]") {
                    ::close(fd);
                    active_fd_ = -1;
                    return true;
                }

                std::string piece;
                bool reasoning = false;
                size_t p = data.find("\"content\":\"");
                if (p != std::string::npos) {
                    piece = json_unescape_string(data, p + 11);
                }
                size_t pr = data.find("\"reasoning_content\":\"");
                if (pr != std::string::npos) {
                    reasoning = true;
                    piece = json_unescape_string(data, pr + 21);
                }

                StreamTelem telem{};
                extract_json_number(data, "tok_per_sec", telem.tok_per_sec);
                extract_json_number(data, "wall_ms", telem.wall_ms);
                extract_json_number(data, "io_wait_ms", telem.io_wait_ms);
                extract_json_number(data, "matmul_ms", telem.matmul_ms);
                extract_json_number(data, "attention_ms", telem.attention_ms);
                extract_json_number(data, "lm_head_ms", telem.lm_head_ms);
                double v = 0;
                if (extract_json_number(data, "cache_hits", v)) telem.cache_hits = static_cast<uint32_t>(v);
                if (extract_json_number(data, "disk_reads", v)) telem.disk_reads = static_cast<uint32_t>(v);

                if (on_token && !piece.empty()) on_token(piece, reasoning, telem);
            }
        }

        ::close(fd);
        active_fd_ = -1;
        return !stop_.load();
    }

    std::string host_;
    uint16_t port_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> stop_{false};
    std::atomic<int> active_fd_{-1};
    std::thread worker_;
};

} // namespace jxai::term
