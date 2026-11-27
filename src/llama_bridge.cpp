#include "llama_bridge.h"
#include "vulkan_backend.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace moe {

namespace {

bool file_is_executable(const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) return false;
#ifdef _WIN32
    auto ext = p.extension().string();
    for (char& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return ext == ".exe";
#else
    return ::access(p.c_str(), X_OK) == 0;
#endif
}

std::string http_get(const std::string& host, uint16_t port, const std::string& path,
                     int timeout_ms) {
#ifdef _WIN32
    WSADATA wsa{};
    static bool once = (::WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    if (!once) return "";
    SOCKET fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) return "";
    DWORD tv = static_cast<DWORD>(timeout_ms);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
#ifdef _WIN32
        ::closesocket(fd);
#else
        ::close(fd);
#endif
        return "";
    }
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
#ifdef _WIN32
        ::closesocket(fd);
#else
        ::close(fd);
#endif
        return "";
    }
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\nHost: " << host << "\r\nConnection: close\r\n\r\n";
    std::string r = req.str();
#ifdef _WIN32
    if (::send(fd, r.data(), static_cast<int>(r.size()), 0) <= 0) {
        ::closesocket(fd);
        return "";
    }
#else
    if (::send(fd, r.data(), r.size(), 0) <= 0) {
        ::close(fd);
        return "";
    }
#endif
    std::string resp;
    char buf[4096];
    while (true) {
#ifdef _WIN32
        int n = ::recv(fd, buf, sizeof(buf), 0);
#else
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        resp.append(buf, static_cast<size_t>(n));
        if (resp.size() > 1 << 20) break;
    }
#ifdef _WIN32
    ::closesocket(fd);
#else
    ::close(fd);
#endif
    return resp;
}

bool nvidia_smi_ok() {
#ifdef _WIN32
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    std::string cmd = "nvidia-smi -L";
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');
    if (!::CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE,
                          CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        return false;
    }
    ::WaitForSingleObject(pi.hProcess, 5000);
    DWORD code = 1;
    ::GetExitCodeProcess(pi.hProcess, &code);
    ::CloseHandle(pi.hProcess);
    ::CloseHandle(pi.hThread);
    return code == 0;
#else
    int rc = ::system("nvidia-smi -L >/dev/null 2>&1");
    return rc == 0;
#endif
}

} // namespace

LlamaGpuInfo probe_llama_gpu() {
    LlamaGpuInfo info;
    info.has_nvidia = nvidia_smi_ok();
    std::string vk_err;
    info.has_vulkan = VulkanComputeBackend::instance().init(vk_err);
    std::ostringstream d;
    if (info.has_nvidia) d << "NVIDIA GPU detected (nvidia-smi)";
    if (info.has_vulkan) {
        if (!d.str().empty()) d << "; ";
        d << "Vulkan: " << VulkanComputeBackend::instance().device_info().device_name;
    }
    if (d.str().empty()) d << "no GPU backend detected";
    info.detail = d.str();
    return info;
}

std::string find_llama_server_binary(const std::string& explicit_path) {
    if (!explicit_path.empty() && file_is_executable(explicit_path)) return explicit_path;
    if (const char* env = std::getenv("LLAMA_SERVER")) {
        if (file_is_executable(env)) return env;
    }
    const char* names[] = {
#ifdef _WIN32
        "llama-server.exe", "llama-server"
#else
        "llama-server", "llama-server.exe"
#endif
    };
    std::vector<fs::path> dirs;
    dirs.push_back(fs::current_path());
    dirs.push_back(fs::current_path() / "tools" / "llama");
    dirs.push_back(fs::current_path() / "tools" / "llama" / "bin");
#ifdef _WIN32
    char mod[MAX_PATH];
    if (::GetModuleFileNameA(nullptr, mod, MAX_PATH) > 0) {
        dirs.push_back(fs::path(mod).parent_path());
        dirs.push_back(fs::path(mod).parent_path() / "tools" / "llama");
    }
#else
    // Best-effort: argv0 not available here; also search PATH via which-like walk.
#endif
    if (const char* path = std::getenv("PATH")) {
        std::string p = path;
        size_t start = 0;
        while (start <= p.size()) {
#ifdef _WIN32
            size_t semi = p.find(';', start);
#else
            size_t semi = p.find(':', start);
#endif
            std::string part = p.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
            if (!part.empty()) dirs.push_back(part);
            if (semi == std::string::npos) break;
            start = semi + 1;
        }
    }
    for (const auto& dir : dirs) {
        for (const char* name : names) {
            fs::path cand = dir / name;
            if (file_is_executable(cand)) return cand.string();
        }
    }
    return "";
}

bool should_try_llama_backend(InferenceBackend mode, const LlamaGpuInfo& gpu,
                              std::string& reason) {
    if (mode == InferenceBackend::Cpu) {
        reason = "backend=cpu (native moe_engine)";
        return false;
    }
    if (mode == InferenceBackend::Llama) {
        reason = "backend=llama (forced)";
        return true;
    }
    // Auto: only switch when a GPU is present. No-GPU PCs stay on moe_engine.
    if (gpu.has_nvidia || gpu.has_vulkan) {
        reason = "backend=auto + GPU present → llama.cpp";
        return true;
    }
    reason = "backend=auto + no GPU → native moe_engine CPU";
    return false;
}

bool parse_inference_backend(const std::string& s, InferenceBackend& out) {
    std::string v = s;
    for (char& c : v) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    if (v == "auto") { out = InferenceBackend::Auto; return true; }
    if (v == "cpu" || v == "native" || v == "moe") { out = InferenceBackend::Cpu; return true; }
    if (v == "llama" || v == "llamacpp" || v == "llama.cpp") {
        out = InferenceBackend::Llama;
        return true;
    }
    return false;
}

bool start_llama_server(const LlamaBridgeConfig& cfg, const LlamaGpuInfo& gpu,
                        LlamaBridge& out, std::string& err_msg) {
    stop_llama_server(out);
    out = LlamaBridge{};

    std::string bin = find_llama_server_binary(cfg.server_bin);
    if (bin.empty()) {
        err_msg = "llama-server not found. Install a llama.cpp build into tools/llama/ "
                  "or set LLAMA_SERVER / pass --llama-server. "
                  "See scripts/fetch_llama_server.sh (or .ps1).";
        return false;
    }
    if (!fs::exists(cfg.model_path)) {
        err_msg = "Model not found for llama-server: " + cfg.model_path;
        return false;
    }

    const bool want_gpu = gpu.has_nvidia || gpu.has_vulkan;
    int ngl = cfg.n_gpu_layers;
    if (ngl < 0) ngl = want_gpu ? 99 : 0;

    out.server_bin = bin;
    out.upstream_host = cfg.host;
    out.upstream_port = cfg.port;
    out.using_gpu = want_gpu && ngl > 0;
    if (gpu.has_nvidia && out.using_gpu) out.backend_label = "llama.cpp+CUDA/GPU";
    else if (gpu.has_vulkan && out.using_gpu) out.backend_label = "llama.cpp+Vulkan";
    else out.backend_label = "llama.cpp+CPU";

    std::ostringstream cmd;
#ifdef _WIN32
    cmd << "\"" << bin << "\"";
#else
    cmd << bin;
#endif
    cmd << " -m \"" << cfg.model_path << "\""
        << " --host " << cfg.host
        << " --port " << cfg.port
        << " -ngl " << ngl
        << " --jinja";
    if (cfg.ctx_size > 0) cmd << " -c " << cfg.ctx_size;
    if (!cfg.extra_args.empty()) cmd << " " << cfg.extra_args;

    std::cout << "[HYBRID] Starting " << out.backend_label << "\n"
              << "  binary:  " << bin << "\n"
              << "  model:   " << cfg.model_path << "\n"
              << "  upstream http://" << cfg.host << ":" << cfg.port << "\n"
              << "  ngl:     " << ngl << "\n";

#ifdef _WIN32
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::string cmdline = cmd.str();
    std::vector<char> buf(cmdline.begin(), cmdline.end());
    buf.push_back('\0');
    DWORD flags = CREATE_NEW_PROCESS_GROUP;
    if (!::CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, flags,
                          nullptr, nullptr, &si, &pi)) {
        err_msg = "CreateProcess failed for llama-server (err=" +
                  std::to_string(::GetLastError()) + ")";
        return false;
    }
    out.process = pi.hProcess;
    out.pid = static_cast<int>(pi.dwProcessId);
    ::CloseHandle(pi.hThread);
#else
    pid_t pid = ::fork();
    if (pid < 0) {
        err_msg = "fork() failed for llama-server";
        return false;
    }
    if (pid == 0) {
        // Child
        std::vector<std::string> args_store;
        args_store.push_back(bin);
        args_store.push_back("-m");
        args_store.push_back(cfg.model_path);
        args_store.push_back("--host");
        args_store.push_back(cfg.host);
        args_store.push_back("--port");
        args_store.push_back(std::to_string(cfg.port));
        args_store.push_back("-ngl");
        args_store.push_back(std::to_string(ngl));
        args_store.push_back("--jinja");
        if (cfg.ctx_size > 0) {
            args_store.push_back("-c");
            args_store.push_back(std::to_string(cfg.ctx_size));
        }
        std::vector<char*> argv;
        for (auto& s : args_store) argv.push_back(s.data());
        argv.push_back(nullptr);
        ::execv(bin.c_str(), argv.data());
        ::_exit(127);
    }
    out.pid = static_cast<int>(pid);
    out.process = reinterpret_cast<void*>(static_cast<intptr_t>(pid));
#endif

    // Wait for health (model load can take a while for large GGUFs).
    const int timeout_ms = 180000;
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        if (elapsed > timeout_ms) {
            err_msg = "llama-server did not become healthy within timeout";
            stop_llama_server(out);
            return false;
        }
#ifdef _WIN32
        DWORD code = STILL_ACTIVE;
        if (out.process && ::GetExitCodeProcess(static_cast<HANDLE>(out.process), &code) &&
            code != STILL_ACTIVE) {
            err_msg = "llama-server exited early (code " + std::to_string(code) + ")";
            stop_llama_server(out);
            return false;
        }
#else
        int status = 0;
        pid_t w = ::waitpid(out.pid, &status, WNOHANG);
        if (w == out.pid) {
            err_msg = "llama-server exited early";
            stop_llama_server(out);
            return false;
        }
#endif
        std::string resp = http_get(cfg.host, cfg.port, "/health", 2000);
        if (resp.find("200") != std::string::npos &&
            (resp.find("\"ok\"") != std::string::npos ||
             resp.find("\"status\"") != std::string::npos)) {
            std::cout << "[HYBRID] llama-server healthy after " << elapsed << " ms\n";
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
}

void stop_llama_server(LlamaBridge& bridge) {
    if (bridge.pid == 0 && bridge.process == nullptr) return;
#ifdef _WIN32
    if (bridge.process) {
        ::TerminateProcess(static_cast<HANDLE>(bridge.process), 1);
        ::WaitForSingleObject(static_cast<HANDLE>(bridge.process), 5000);
        ::CloseHandle(static_cast<HANDLE>(bridge.process));
    }
#else
    if (bridge.pid > 0) {
        ::kill(bridge.pid, SIGTERM);
        for (int i = 0; i < 50; ++i) {
            int status = 0;
            pid_t w = ::waitpid(bridge.pid, &status, WNOHANG);
            if (w == bridge.pid || w < 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ::kill(bridge.pid, SIGKILL);
        int status = 0;
        ::waitpid(bridge.pid, &status, 0);
    }
#endif
    bridge = LlamaBridge{};
}

} // namespace moe
