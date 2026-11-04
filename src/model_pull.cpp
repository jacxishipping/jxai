#include "model_pull.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wininet.h>
#endif

namespace fs = std::filesystem;

namespace moe {

namespace {

void render_progress_bar(size_t downloaded, size_t total, double mb_per_sec, double eta_sec) {
    const int bar_width = 30;
    float frac = (total > 0) ? std::clamp(static_cast<float>(downloaded) / static_cast<float>(total), 0.0f, 1.0f) : 0.0f;
    int filled = static_cast<int>(frac * bar_width);

    std::cout << "\r\033[1;36m[PULL]\033[0m [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << "\033[1;32m#\033[0m";
        else std::cout << "\033[2m-\033[0m";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (frac * 100.0f) << "% ("
              << (downloaded / (1024.0 * 1024.0)) << " MB";
    if (total > 0) {
        std::cout << " / " << (total / (1024.0 * 1024.0)) << " MB";
    }
    std::cout << ") \033[1;33m" << mb_per_sec << " MB/s\033[0m";
    if (eta_sec >= 0.0) {
        std::cout << " ETA: " << static_cast<int>(eta_sec) << "s ";
    }
    std::cout << std::flush;
}

} // namespace

bool pull_model_from_hub(
    const std::string& input,
    const std::string& out_dir,
    std::string& out_file,
    std::string& err_msg
) {
    std::string url = input;
    std::string filename;

    // Parse input variants
    if (url.starts_with("http://") || url.starts_with("https://")) {
        size_t last_slash = url.find_last_of('/');
        filename = (last_slash != std::string::npos) ? url.substr(last_slash + 1) : "downloaded_model.gguf";
    } else {
        // Hugging Face shorthand: repo:file.gguf or repo
        size_t colon = url.find(':');
        if (colon != std::string::npos) {
            std::string repo = url.substr(0, colon);
            filename = url.substr(colon + 1);
            url = "https://huggingface.co/" + repo + "/resolve/main/" + filename;
        } else {
            std::string repo = url;
            // Default file heuristics based on repo name
            std::string lower = repo;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("coder") != std::string::npos) {
                filename = "qwen2.5-coder-0.5b-instruct-q4_0.gguf";
            } else if (lower.find("deepseek") != std::string::npos) {
                filename = "deepseek-moe-q4_0.gguf";
            } else if (lower.find("mixtral") != std::string::npos) {
                filename = "mixtral-8x7b-instruct-q4_0.gguf";
            } else {
                filename = "qwen2.5-0.5b-instruct-q4_0.gguf";
            }
            url = "https://huggingface.co/" + repo + "/resolve/main/" + filename;
        }
    }

    fs::path target_dir = out_dir.empty() ? "models" : out_dir;
    fs::create_directories(target_dir);
    fs::path dest_path = target_dir / filename;
    out_file = dest_path.string();

    std::cout << "\033[1;36m===================================================================\033[0m\n"
              << "\033[1;36m  MoE GGUF MODEL HUB STREAM DOWNLOADER\033[0m\n"
              << "  Source URL:    \033[1;37m" << url << "\033[0m\n"
              << "  Destination:   \033[1;32m" << dest_path.string() << "\033[0m\n"
              << "\033[1;36m===================================================================\033[0m\n\n";

#ifdef _WIN32
    HINTERNET hInternet = InternetOpenA("MoE-Stream-Engine/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
    if (!hInternet) {
        err_msg = "Failed to initialize WinINet network subsystem (Error: " + std::to_string(GetLastError()) + ")";
        return false;
    }

    DWORD flags = INTERNET_FLAG_RELOAD | INTERNET_FLAG_DONT_CACHE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION;
    HINTERNET hFile = InternetOpenUrlA(hInternet, url.c_str(), nullptr, 0, flags, 0);
    if (!hFile) {
        err_msg = "Failed to connect to model URL (Error: " + std::to_string(GetLastError()) + ")";
        InternetCloseHandle(hInternet);
        return false;
    }

    // Query HTTP status code
    DWORD status_code = 0;
    DWORD status_len = sizeof(status_code);
    if (HttpQueryInfoA(hFile, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER, &status_code, &status_len, nullptr)) {
        if (status_code >= 400) {
            err_msg = "Remote server returned HTTP " + std::to_string(status_code);
            InternetCloseHandle(hFile);
            InternetCloseHandle(hInternet);
            return false;
        }
    }

    // Query Content-Length
    size_t total_bytes = 0;
    char cl_buf[64] = {0};
    DWORD cl_len = sizeof(cl_buf) - 1;
    if (HttpQueryInfoA(hFile, HTTP_QUERY_CONTENT_LENGTH, cl_buf, &cl_len, nullptr)) {
        try { total_bytes = std::stoull(cl_buf); } catch (...) {}
    }

    std::ofstream ofs(dest_path, std::ios::binary);
    if (!ofs) {
        err_msg = "Could not open destination file for writing: " + dest_path.string();
        InternetCloseHandle(hFile);
        InternetCloseHandle(hInternet);
        return false;
    }

    std::vector<char> buffer(64 * 1024);
    size_t downloaded = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto last_update = t0;

    while (true) {
        DWORD bytes_read = 0;
        if (!InternetReadFile(hFile, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read) || bytes_read == 0) {
            break;
        }

        ofs.write(buffer.data(), bytes_read);
        downloaded += bytes_read;

        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - t0).count();
        double update_diff = std::chrono::duration<double>(now - last_update).count();

        if (update_diff >= 0.1 || downloaded == total_bytes) {
            double mb_per_sec = (downloaded / (1024.0 * 1024.0)) / std::max(0.001, elapsed_sec);
            double eta_sec = (total_bytes > downloaded && mb_per_sec > 0.0)
                ? (static_cast<double>(total_bytes - downloaded) / (mb_per_sec * 1024.0 * 1024.0))
                : 0.0;

            render_progress_bar(downloaded, total_bytes, mb_per_sec, eta_sec);
            last_update = now;
        }
    }

    ofs.close();
    InternetCloseHandle(hFile);
    InternetCloseHandle(hInternet);

    std::cout << "\n\n";

    if (downloaded == 0) {
        err_msg = "Download stream closed without transferring data.";
        return false;
    }

    // Validate GGUF Magic header
    std::ifstream check(dest_path, std::ios::binary);
    uint32_t magic = 0;
    check.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    check.close();

    if (magic != 0x46554747) { // "GGUF" in little-endian
        err_msg = "File downloaded but failed GGUF binary magic validation (found: 0x" + std::to_string(magic) + ")";
        return false;
    }

    std::cout << "\033[1;32m[SUCCESS] MoE Model downloaded & verified successfully!\033[0m\n"
              << "  File Path:      " << dest_path.string() << "\n"
              << "  File Size:      " << std::fixed << std::setprecision(2) << (downloaded / (1024.0 * 1024.0)) << " MB\n\n"
              << "To launch this model immediately with the Engine & Studio:\n"
              << "  \033[1;33m.\\build\\moe_engine.exe serve --model " << dest_path.string() << "\033[0m\n"
              << "  \033[1;33m.\\run_studio.bat\033[0m\n\n";

    return true;
#else
    err_msg = "Model pulling is currently supported on Windows with native WinINet.";
    return false;
#endif
}

} // namespace moe

