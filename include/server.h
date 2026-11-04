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

    // Start listening on `port` (blocking loop until stop() is called)
    bool start(uint16_t port, std::string& err_msg);
    void stop();

private:
    void handle_client(uintptr_t client_sock);

    MoETransformerEngine& engine_;
    std::string web_root_;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> active_requests_{0};
    uintptr_t listen_sock_ = ~static_cast<uintptr_t>(0);
};

} // namespace moe

