#pragma once

#include "transformer.h"
#include "diffusion.h"
#include <string>
#include <atomic>

namespace moe {

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
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> active_requests_{0};
    uintptr_t listen_sock_ = ~static_cast<uintptr_t>(0);
};

} // namespace moe

