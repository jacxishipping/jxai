#include "model_pull.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wininet.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace moe {

namespace {

// ---------------------------------------------------------------------------
// Validation helpers. Nothing that reaches a shell or a URL is trusted:
// the download path below never invokes a shell at all (fork/exec curl on
// POSIX, WinINet on Windows), and these checks reject hostile input early
// with a clear error instead of passing it on.
// ---------------------------------------------------------------------------

// Characters that must never appear in a user-supplied URL / repo / filename.
bool has_shell_meta(const std::string& s) {
    static const char* kBad = "'\"`$;&| \t\r\n\\<>?*~#!(){}";
    for (char c : s) {
        if (static_cast<unsigned char>(c) < 0x20) return true;
        if (std::strchr(kBad, c) != nullptr) return true;
    }
    return false;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return to_lower(s.substr(s.size() - suffix.size())) == to_lower(suffix);
}

std::string base_name(const std::string& path) {
    size_t p = path.find_last_of('/');
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

// HF repo ids are exactly "owner/name" with no ".." segments.
bool valid_repo_id(const std::string& repo, std::string& err_msg) {
    if (repo.empty() || repo.size() > 200) {
        err_msg = "Invalid Hugging Face repo id (empty or too long): '" + repo + "'";
        return false;
    }
    if (has_shell_meta(repo)) {
        err_msg = "Invalid Hugging Face repo id: contains shell metacharacters or whitespace: '" + repo + "'";
        return false;
    }
    size_t slash = repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash == repo.size() - 1 ||
        repo.find('/', slash + 1) != std::string::npos) {
        err_msg = "Invalid Hugging Face repo id (expected 'owner/name'): '" + repo + "'";
        return false;
    }
    if (repo.find("..") != std::string::npos) {
        err_msg = "Invalid Hugging Face repo id (contains '..'): '" + repo + "'";
        return false;
    }
    for (char c : repo) {
        unsigned char u = static_cast<unsigned char>(c);
        bool ok = (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
                  (u >= '0' && u <= '9') || c == '.' || c == '_' || c == '-' || c == '/';
        if (!ok) {
            err_msg = "Invalid Hugging Face repo id (bad character '" + std::string(1, c) + "'): '" + repo + "'";
            return false;
        }
    }
    return true;
}

// A downloadable model file: a plain filename, .gguf or .safetensors only.
bool valid_model_filename(const std::string& fn, std::string& err_msg) {
    if (fn.empty() || fn.find('/') != std::string::npos || fn.find('\\') != std::string::npos) {
        err_msg = "Invalid model filename (must be a plain filename, no directories): '" + fn + "'";
        return false;
    }
    if (has_shell_meta(fn)) {
        err_msg = "Invalid model filename (shell metacharacters or whitespace): '" + fn + "'";
        return false;
    }
    if (!ends_with_ci(fn, ".gguf") && !ends_with_ci(fn, ".safetensors")) {
        err_msg = "Unsupported model file (only .gguf and .safetensors are supported): '" + fn + "'";
        return false;
    }
    return true;
}

bool valid_url(const std::string& url, std::string& err_msg) {
    if (!url.starts_with("http://") && !url.starts_with("https://")) {
        err_msg = "URL must start with http:// or https://: '" + url + "'";
        return false;
    }
    if (has_shell_meta(url)) {
        err_msg = "URL contains shell metacharacters, whitespace or a query string (?): '" + url + "'";
        return false;
    }
    size_t host_begin = url.find("://") + 3;
    if (host_begin >= url.size() || url.find('/', host_begin) == std::string::npos) {
        err_msg = "URL has no path (a model file is required): '" + url + "'";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shard / picking helpers
// ---------------------------------------------------------------------------

bool parse_sharded_gguf_name_impl(const std::string& filename, std::string& group,
                                  uint32_t& index, uint32_t& total) {
    if (!ends_with_ci(filename, ".gguf")) return false;
    std::string stem = filename.substr(0, filename.size() - 5); // strip ".gguf"
    size_t of = stem.rfind("-of-");
    if (of == std::string::npos) return false;
    std::string total_s = stem.substr(of + 4);
    if (total_s.empty() ||
        !std::all_of(total_s.begin(), total_s.end(), ::isdigit)) return false;
    size_t dash = stem.rfind('-', of - 1);
    if (dash == std::string::npos || dash + 1 >= of) return false;
    std::string index_s = stem.substr(dash + 1, of - dash - 1);
    if (index_s.empty() ||
        !std::all_of(index_s.begin(), index_s.end(), ::isdigit)) return false;
    try {
        total = static_cast<uint32_t>(std::stoul(total_s));
        index = static_cast<uint32_t>(std::stoul(index_s));
    } catch (...) {
        return false;
    }
    if (total < 2 || index < 1 || index > total) return false;
    group = stem.substr(0, dash) + ".gguf";
    return true;
}

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

// ---------------------------------------------------------------------------
// Minimal JSON extraction (no JSON library in this codebase).
// These helpers scan for a top-level key and parse the value that follows.
// ---------------------------------------------------------------------------

// Extract the first string value for "key" (skips nested objects by taking
// the first occurrence; good enough for the flat fields we need).
bool json_get_string(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    size_t i = q1 + 1;
    out.clear();
    while (i < json.size() && json[i] != '"') {
        if (json[i] == '\\' && i + 1 < json.size()) {
            ++i;
            switch (json[i]) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                default:  out.push_back(json[i]); break;
            }
            ++i;
        } else {
            out.push_back(json[i++]);
        }
    }
    return i < json.size();
}

bool json_get_number(const std::string& json, const std::string& key, double& out) {
    std::string pat = "\"" + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return false;
    size_t colon = json.find(':', k + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    size_t start = i;
    while (i < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[i])) || json[i] == '-' ||
            json[i] == '+' || json[i] == '.' || json[i] == 'e' || json[i] == 'E')) ++i;
    if (i == start) return false;
    try {
        out = std::stod(json.substr(start, i - start));
    } catch (...) {
        return false;
    }
    return true;
}

// Parse one top-level object of the tree/main array: returns path/size/type.
// "size" is the FIRST size field in the object (the top-level one; the
// nested lfs object, if any, comes later and repeats it).
bool parse_tree_entry(const std::string& obj, std::string& path, uint64_t& size, std::string& type) {
    if (!json_get_string(obj, "path", path)) return false;
    double sz = 0;
    if (!json_get_number(obj, "size", sz)) return false;
    size = static_cast<uint64_t>(sz);
    if (!json_get_string(obj, "type", type)) type = "file";
    return true;
}

void parse_hf_tree(const std::string& body, std::vector<HfFileInfo>& gguf_files,
                   std::vector<HfFileInfo>& st_files) {
    // Split the top-level JSON array into its object elements by brace depth.
    size_t pos = 0;
    while (pos < body.size()) {
        size_t start = body.find('{', pos);
        if (start == std::string::npos) break;
        int depth = 0;
        size_t i = start;
        bool in_str = false;
        for (; i < body.size(); ++i) {
            char c = body[i];
            if (in_str) {
                if (c == '\\') { ++i; continue; }
                if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') {
                if (--depth == 0) { ++i; break; }
            }
        }
        std::string obj = body.substr(start, i - start);
        pos = i;
        std::string path, type;
        uint64_t size = 0;
        if (!parse_tree_entry(obj, path, size, type)) continue;
        if (type != "file") continue;
        HfFileInfo info;
        info.filename = path;
        info.size = size;
        std::string group;
        uint32_t idx = 0, tot = 0;
        if (parse_sharded_gguf_name_impl(base_name(path), group, idx, tot)) {
            info.is_shard = true;
            info.shard_index = idx;
            info.shard_total = tot;
            info.shard_group = group;
        }
        if (ends_with_ci(path, ".gguf")) gguf_files.push_back(std::move(info));
        else if (ends_with_ci(path, ".safetensors")) st_files.push_back(std::move(info));
    }
}

// ---------------------------------------------------------------------------
// HTTP via curl (POSIX, no shell) / WinINet (Windows)
// ---------------------------------------------------------------------------

#ifndef _WIN32

// Run curl with an argv vector (never a shell string). Captures stdout.
// On success returns true and fills http_code (from -w) when requested.
bool run_curl(const std::vector<std::string>& args, std::string& stdout_text,
              int& exit_code, bool capture_code) {
    std::vector<std::string> full = args;
    if (capture_code) {
        full.push_back("-w");
        full.push_back("\n%{http_code}");
    }
    int pipefd[2];
    if (::pipe(pipefd) != 0) return false;
    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return false;
    }
    if (pid == 0) {
        // Child: only async-signal-safe calls before exec.
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        std::vector<char*> cargs;
        cargs.reserve(full.size() + 1);
        for (auto& a : full) cargs.push_back(const_cast<char*>(a.c_str()));
        cargs.push_back(nullptr);
        ::execvp("curl", cargs.data());
        ::_exit(127); // curl not found
    }
    ::close(pipefd[1]);
    stdout_text.clear();
    char buf[16384];
    ssize_t n;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
        stdout_text.append(buf, static_cast<size_t>(n));
    }
    ::close(pipefd[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) exit_code = 128 + WTERMSIG(status);
    else exit_code = -1;
    return true;
}

// Extra curl args for HF auth when the user exported HF_TOKEN (needed for
// gated repos). The token is passed as an argv element, never via a shell.
std::vector<std::string> hf_auth_args() {
    const char* tok = std::getenv("HF_TOKEN");
    if (tok != nullptr && tok[0] != '\0') {
        return {"-H", std::string("Authorization: Bearer ") + tok};
    }
    return {};
}

bool hf_api_get(const std::string& url, std::string& body, int& http_code, std::string& err_msg) {
    std::vector<std::string> args = {"curl", "-sS", "--max-time", "30"};
    auto auth = hf_auth_args();
    args.insert(args.end(), auth.begin(), auth.end());
    args.push_back("-H");
    args.push_back("Accept: application/json");
    args.push_back(url);
    std::string out;
    int exit_code = -1;
    if (!run_curl(args, out, exit_code, true)) {
        err_msg = "Failed to run curl (is curl installed?) for " + url;
        return false;
    }
    http_code = 0;
    size_t nl = out.rfind('\n');
    if (nl != std::string::npos) {
        try { http_code = std::stoi(out.substr(nl + 1)); } catch (...) { http_code = 0; }
        body = out.substr(0, nl);
    } else {
        body = out;
    }
    if (exit_code == 127) {
        err_msg = "curl not found in PATH (required for Hugging Face downloads)";
        return false;
    }
    if (exit_code != 0 && http_code == 0) {
        err_msg = "HF API request failed (curl exit " + std::to_string(exit_code) + ") for " + url;
        return false;
    }
    if (http_code != 0 && http_code != 200) {
        err_msg = "Hugging Face API returned HTTP " + std::to_string(http_code) + " for " + url +
                  (http_code == 401 ? " (repo not found, private, or gated — set HF_TOKEN for gated repos)"
                                    : http_code == 404 ? " (repo not found)"
                                                       : "");
        return false;
    }
    return true;
}

// HEAD request for the remote Content-Length (used to recognise an
// already-complete file when the caller has no expected size, e.g. an
// explicit URL or repo:file). Returns false when the length is unknown.
bool http_content_length(const std::string& url, uint64_t& length) {
    std::vector<std::string> args = {"curl", "-sI", "-L", "--max-time", "15"};
    auto auth = hf_auth_args();
    args.insert(args.end(), auth.begin(), auth.end());
    args.push_back(url);
    std::string out;
    int exit_code = -1;
    if (!run_curl(args, out, exit_code, false) || exit_code != 0) return false;
    std::istringstream iss(out);
    std::string line, last_cl;
    while (std::getline(iss, line)) {
        if (to_lower(line).rfind("content-length:", 0) == 0) last_cl = line.substr(15);
    }
    if (last_cl.empty()) return false;
    try { length = std::stoull(last_cl); } catch (...) { return false; }
    return length > 0;
}

// Download one file with resume (-C -) and a response-header file for the
// Content-Length size check. expected_size>0 (from the HF API) lets us skip
// an already-complete file. Returns false with err_msg on any failure.
bool curl_download(const std::string& url, const fs::path& dest, uint64_t expected_size,
                   bool show_progress, PullProgressCallback on_progress, std::string& err_msg) {
    std::error_code ec;
    // No expected size (explicit URL / repo:file): ask the server first so
    // an already-complete file is recognised instead of re-downloaded.
    if (expected_size == 0 && fs::exists(dest, ec) && !ec) {
        uint64_t have = fs::file_size(dest, ec);
        uint64_t remote = 0;
        if (!ec && have > 0 && http_content_length(url, remote)) {
            if (have == remote) {
                std::cout << "[PULL] Already present and complete, skipping download: " << dest.string() << "\n";
                if (on_progress) on_progress(have, remote, 0.0, 0.0);
                return true;
            }
            if (have > remote) {
                err_msg = "Existing file is larger than the remote file (" + std::to_string(have) +
                          " > " + std::to_string(remote) + " bytes): " + dest.string() +
                          " — delete it and retry";
                return false;
            }
            expected_size = remote; // enables the resume/size messages below
        }
    }
    if (expected_size > 0 && fs::exists(dest, ec) && !ec) {
        uint64_t have = fs::file_size(dest, ec);
        if (!ec && have == expected_size) {
            std::cout << "[PULL] Already present and complete, skipping download: " << dest.string() << "\n";
            if (on_progress) on_progress(have, expected_size, 0.0, 0.0);
            return true;
        }
        if (!ec && have > expected_size) {
            err_msg = "Existing file is larger than expected (" + std::to_string(have) +
                      " > " + std::to_string(expected_size) + " bytes): " + dest.string() +
                      " — delete it and retry";
            return false;
        }
        if (!ec && have > 0) {
            std::cout << "[PULL] Resuming partial download (" << have << " / " << expected_size
                      << " bytes): " << dest.string() << "\n";
        }
    }

    fs::path headers_path = dest;
    headers_path += ".headers";
    std::vector<std::string> args = {"curl", "-L", "--fail", "-C", "-", "-o", dest.string(),
                                     "-D", headers_path.string()};
    auto auth = hf_auth_args();
    args.insert(args.end(), auth.begin(), auth.end());
    if (show_progress) args.push_back("--progress-bar");
    else { args.push_back("--silent"); args.push_back("--show-error"); }
    args.push_back(url);

    std::string out;
    int exit_code = -1;
    if (!run_curl(args, out, exit_code, false)) {
        err_msg = "Failed to run curl (is curl installed?)";
        return false;
    }
    if (exit_code == 127) {
        err_msg = "curl not found in PATH (required for downloads)";
        return false;
    }
    if (exit_code != 0) {
        err_msg = "Download failed (curl exit " + std::to_string(exit_code) +
                  (exit_code == 22 ? " — HTTP error, e.g. file not found or gated repo" : "") +
                  (exit_code == 33 ? " — server does not support byte ranges, so a partial file cannot be resumed; delete the partial file and retry" : "") +
                  ") for " + url;
        return false;
    }
    if (!fs::exists(dest, ec) || ec || fs::file_size(dest, ec) == 0) {
        err_msg = "Downloaded file does not exist or is empty: " + dest.string();
        return false;
    }
    uint64_t downloaded = fs::file_size(dest, ec);

    // Size check against the Content-Length the server reported.
    uint64_t content_length = 0;
    {
        std::ifstream hf(headers_path);
        std::string line, last_cl;
        while (std::getline(hf, line)) {
            std::string low = to_lower(line);
            if (low.rfind("content-length:", 0) == 0) last_cl = line.substr(15);
        }
        if (!last_cl.empty()) {
            try { content_length = std::stoull(last_cl); } catch (...) {}
        }
    }
    fs::remove(headers_path, ec);
    uint64_t expected = (expected_size > 0) ? expected_size : content_length;
    if (expected > 0 && downloaded != expected) {
        err_msg = "Size mismatch for " + dest.string() + ": downloaded " +
                  std::to_string(downloaded) + " bytes, expected " + std::to_string(expected) +
                  " bytes — download incomplete, retry to resume";
        return false;
    }
    if (on_progress) on_progress(downloaded, expected, 0.0, 0.0);
    return true;
}

#else // _WIN32

bool hf_api_get(const std::string& url, std::string& body, int& http_code, std::string& err_msg) {
    HINTERNET hNet = InternetOpenA("MoE-Stream-Engine/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
    if (!hNet) {
        err_msg = "Failed to initialize WinINet";
        return false;
    }
    std::string headers = "Accept: application/json\r\n";
    char tok_buf[512] = {0};
    if (GetEnvironmentVariableA("HF_TOKEN", tok_buf, sizeof(tok_buf)) > 0) {
        headers += "Authorization: Bearer " + std::string(tok_buf) + "\r\n";
    }
    HINTERNET hFile = InternetOpenUrlA(hNet, url.c_str(), headers.c_str(),
                                       static_cast<DWORD>(headers.size()),
                                       INTERNET_FLAG_RELOAD | INTERNET_FLAG_DONT_CACHE, 0);
    if (!hFile) {
        err_msg = "Failed to connect to Hugging Face API (Error: " + std::to_string(GetLastError()) + ")";
        InternetCloseHandle(hNet);
        return false;
    }
    DWORD status = 0, slen = sizeof(status);
    http_code = 0;
    if (HttpQueryInfoA(hFile, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER, &status, &slen, nullptr)) {
        http_code = static_cast<int>(status);
    }
    body.clear();
    char buf[16384];
    DWORD read = 0;
    while (InternetReadFile(hFile, buf, sizeof(buf), &read) && read > 0) {
        body.append(buf, read);
    }
    InternetCloseHandle(hFile);
    InternetCloseHandle(hNet);
    if (http_code != 200) {
        err_msg = "Hugging Face API returned HTTP " + std::to_string(http_code) + " for " + url +
                  (http_code == 401 ? " (repo not found, private, or gated — set HF_TOKEN for gated repos)" : "");
        return false;
    }
    return true;
}

#endif // _WIN32

// Validate a downloaded model file by its header, not just its extension.
bool validate_model_file(const fs::path& path, const std::string& filename, std::string& err_msg) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err_msg = "Cannot open downloaded file for validation: " + path.string();
        return false;
    }
    if (ends_with_ci(filename, ".gguf")) {
        char magic[4] = {0};
        in.read(magic, 4);
        if (in.gcount() != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
            err_msg = "File downloaded but failed GGUF magic validation: " + path.string() +
                      " (not a GGUF file — check the repo/file name)";
            return false;
        }
        return true;
    }
    if (ends_with_ci(filename, ".safetensors")) {
        uint64_t header_len = 0;
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (in.gcount() != 8 || header_len == 0) {
            err_msg = "File downloaded but failed SafeTensors header validation: " + path.string();
            return false;
        }
        uint64_t fsize = fs::file_size(path);
        if (8 + header_len > fsize) {
            err_msg = "File downloaded but SafeTensors header length exceeds file size: " + path.string();
            return false;
        }
        return true;
    }
    std::cout << "[PULL] WARNING: no header validation for '" << filename << "' (unknown extension)\n";
    return true;
}

// Validate a downloaded sibling file (config.json / tokenizer.json).
// Catches the common failure modes: empty files, HTML error pages, and
// the 39-byte proxy-CONNECT stub a misconfigured proxy returns instead of
// the real file. Returns false with a reason in `reason` when invalid.
bool validate_sibling_file(const fs::path& dest, const std::string& sibling,
                           std::string& reason) {
    std::error_code ec;
    uint64_t size = fs::file_size(dest, ec);
    if (ec || size <= 100) {
        reason = "file is missing or suspiciously small (" + std::to_string(size) + " bytes)";
        return false;
    }
    std::ifstream in(dest, std::ios::binary);
    if (!in) {
        reason = "cannot open downloaded file for validation";
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t first = content.find_first_not_of(" \t\r\n");
    size_t last = content.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || content[first] != '{' || content[last] != '}') {
        reason = "content is not a JSON object (proxy stub / HTML error page?)";
        return false;
    }
    if (sibling == "config.json") {
        // A real HF config.json names the architecture. A stub never does.
        if (content.find("\"architectures\"") == std::string::npos &&
            content.find("\"model_type\"") == std::string::npos) {
            reason = "config.json contains neither \"architectures\" nor \"model_type\"";
            return false;
        }
    }
    return true;
}

// Fetch a small sibling file (config.json / tokenizer.json) next to a
// downloaded .safetensors file. A sibling that is genuinely absent (404,
// unreachable repo) stays best-effort: the pull continues with a NOTE and
// the engine falls back to architecture defaults. But a download that
// *returns bytes* which fail validation (proxy-CONNECT stub, empty file,
// HTML error page) is retried once and then fails LOUDLY — returning false
// so the pull aborts instead of silently leaving a corrupt file behind.
// Returns true on success or on genuinely-absent (best-effort) siblings.
bool fetch_safetensors_sibling(const std::string& base_url, const std::string& sibling,
                               const fs::path& dest_dir, std::string& err_msg) {
    fs::path dest = dest_dir / sibling;
    std::error_code ec;
    if (fs::exists(dest, ec) && !ec) return true; // never clobber an existing file
    std::string url = base_url + "/" + sibling;
    std::string err;
#ifdef _WIN32
    (void)url; (void)err;
    std::cout << "[PULL] NOTE: sibling '" << sibling << "' not auto-fetched on Windows — download it manually from the repo.\n";
    return true;
#else
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!curl_download(url, dest, 0, false, nullptr, err)) {
            if (attempt == 0) {
                // First attempt failed to fetch anything: sibling may simply
                // not exist in the repo. Best-effort — do not fail the pull.
                std::cout << "[PULL] NOTE: sibling '" << sibling << "' not available (" << err << ") — "
                          << "the engine will load the SafeTensors file without a config.json "
                          << "(architecture defaults apply).\n";
                fs::remove(dest, ec); // drop any partial file curl left behind
                return true;
            }
            err_msg = "Sibling '" + sibling + "' failed validation on first download and the retry download failed (" + err + ")";
            std::cerr << "[PULL] ERROR: " << err_msg << "\n";
            fs::remove(dest, ec);
            return false;
        }
        std::string reason;
        if (validate_sibling_file(dest, sibling, reason)) {
            std::cout << "[PULL] Fetched sibling file: " << dest.string() << "\n";
            return true;
        }
        // Bytes came back but they are not a real sibling file (proxy stub,
        // empty file, HTML error page). Delete and retry once.
        std::cerr << "[PULL] WARNING: downloaded sibling '" << sibling << "' failed validation ("
                  << reason << ") — retrying once.\n";
        fs::remove(dest, ec);
    }
    err_msg = "Sibling '" + sibling + "' downloaded but failed validation twice (proxy stub / empty file / HTML error page?) — refusing to continue with a corrupt sibling. Check your proxy and retry the pull.";
    std::cerr << "[PULL] ERROR: " << err_msg << "\n";
    fs::remove(dest, ec);
    return false;
#endif
}

} // namespace

bool parse_sharded_gguf_name(const std::string& filename, std::string& group,
                             uint32_t& index, uint32_t& total) {
    return parse_sharded_gguf_name_impl(filename, group, index, total);
}

int gguf_pick_rank(const std::string& filename) {
    std::string up = filename;
    std::transform(up.begin(), up.end(), up.begin(), ::toupper);
    // Order matters: check the specific tags before the generic ones.
    static const std::pair<const char*, int> kRanks[] = {
        {"Q4_K_M", 0}, {"Q4_0", 1}, {"Q5_K_M", 2}, {"Q4_K_S", 3},
        {"Q5_K_S", 4}, {"Q6_K", 5},  {"Q8_0", 6},  {"Q4_K", 7},
        {"Q5_K", 7},   {"Q3_K_M", 8}, {"Q5_0", 8}, {"Q2_K", 9},
        {"FP16", 10},  {"F16", 10},
    };
    for (const auto& [tag, rank] : kRanks) {
        if (up.find(tag) != std::string::npos) return rank;
    }
    return 11; // unrecognised quant: last resort
}

bool list_hf_model_files(const std::string& repo,
                         std::vector<HfFileInfo>& gguf_files,
                         std::vector<HfFileInfo>& st_files,
                         bool& gated,
                         std::string& err_msg) {
    gguf_files.clear();
    st_files.clear();
    gated = false;
    if (!valid_repo_id(repo, err_msg)) return false;

    std::string tree_body;
    int code = 0;
    if (!hf_api_get("https://huggingface.co/api/models/" + repo + "/tree/main", tree_body, code, err_msg)) {
        return false;
    }
    parse_hf_tree(tree_body, gguf_files, st_files);

    // Gated flag comes from the model endpoint (best-effort; the file list
    // above is what matters, and tree/main answers 200 even for gated repos).
    std::string meta_body;
    int meta_code = 0;
    std::string meta_err;
    if (hf_api_get("https://huggingface.co/api/models/" + repo, meta_body, meta_code, meta_err)) {
        // "gated" is a JSON boolean or the strings "manual"/"auto"/false, so
        // read the raw token after the key (a string helper would misparse
        // the boolean case by grabbing the next quoted key).
        size_t k = meta_body.find("\"gated\"");
        if (k != std::string::npos) {
            size_t colon = meta_body.find(':', k);
            std::string rest = (colon == std::string::npos) ? "" : meta_body.substr(colon + 1, 12);
            gated = (rest.find("true") != std::string::npos ||
                     rest.find("manual") != std::string::npos ||
                     rest.find("auto") != std::string::npos);
        }
    }

    // Best pick first: by quant rank, then keep the parts of one shard group
    // together in part order (a size sort would put a tiny tail shard first),
    // then by size for single files of equal rank.
    auto by_pick = [](const HfFileInfo& a, const HfFileInfo& b) {
        int ra = gguf_pick_rank(a.filename), rb = gguf_pick_rank(b.filename);
        if (ra != rb) return ra < rb;
        if (a.is_shard && b.is_shard && a.shard_group == b.shard_group) {
            return a.shard_index < b.shard_index;
        }
        std::string ka = a.is_shard ? a.shard_group : a.filename;
        std::string kb = b.is_shard ? b.shard_group : b.filename;
        if (ka != kb) return ka < kb;
        if (a.size != b.size) return a.size < b.size;
        return a.filename < b.filename;
    };
    std::sort(gguf_files.begin(), gguf_files.end(), by_pick);
    std::sort(st_files.begin(), st_files.end(),
              [](const HfFileInfo& a, const HfFileInfo& b) { return a.filename < b.filename; });
    return true;
}

bool pull_model_from_hub(
    const std::string& input,
    const std::string& out_dir,
    std::string& out_file,
    std::string& err_msg,
    PullProgressCallback on_progress
) {
    // ---- Resolve the input into one or more (url, filename, expected_size) ----
    struct DownloadItem { std::string url; std::string filename; uint64_t size = 0; };
    std::vector<DownloadItem> items;
    bool is_sharded_pick = false;
    uint32_t shard_total = 0;

    if (input.starts_with("http://") || input.starts_with("https://")) {
        if (!valid_url(input, err_msg)) return false;
        std::string filename = base_name(input);
        if (!valid_model_filename(filename, err_msg)) return false;
        items.push_back({input, filename, 0});
    } else if (input.find(':') != std::string::npos) {
        // "owner/name:file.gguf" — explicit file, no API lookup needed.
        size_t colon = input.find(':');
        std::string repo = input.substr(0, colon);
        std::string filename = input.substr(colon + 1);
        if (!valid_repo_id(repo, err_msg)) return false;
        if (!valid_model_filename(filename, err_msg)) return false;
        items.push_back({"https://huggingface.co/" + repo + "/resolve/main/" + filename, filename, 0});
    } else {
        // Bare repo id: resolve via the HF API. The old filename-guessing
        // heuristics are gone — a wrong guess downloaded a wrong model.
        std::string repo = input;
        if (!valid_repo_id(repo, err_msg)) return false;
        std::vector<HfFileInfo> gguf_files, st_files;
        bool gated = false;
        if (!list_hf_model_files(repo, gguf_files, st_files, gated, err_msg)) return false;
        if (gated) {
            std::cout << "[PULL] WARNING: repo '" << repo << "' is gated (access approval required). "
                      << "Downloads will fail with HTTP 401 unless HF_TOKEN is set to a token "
                      << "with access to this repo.\n";
        }
        if (gguf_files.empty()) {
            if (!st_files.empty()) {
                std::ostringstream oss;
                oss << "Repo '" << repo << "' has no .gguf files, only SafeTensors:\n";
                for (const auto& f : st_files) oss << "  - " << f.filename << "\n";
                oss << "Pull a single SafeTensors file explicitly, e.g.:\n"
                    << "  moe_engine pull " << repo << ":model.safetensors\n"
                    << "Note: also fetch config.json / tokenizer.json from the repo so the "
                    << "engine can read the architecture (they are fetched automatically "
                    << "when pulling a .safetensors file from the Hub).";
                err_msg = oss.str();
            } else {
                err_msg = "Repo '" + repo + "' has no .gguf or .safetensors model files.";
            }
            return false;
        }
        // Rule 1: prefer a single-file .gguf. Only when the repo has no
        // single-file .gguf at all do we take a sharded group.
        std::vector<HfFileInfo> single;
        for (const auto& f : gguf_files) if (!f.is_shard) single.push_back(f);
        if (!single.empty()) {
            const HfFileInfo& pick = single.front(); // already sorted by rank/size
            std::cout << "[PULL] Picked " << pick.filename << " (" << std::fixed << std::setprecision(1)
                      << (pick.size / (1024.0 * 1024.0)) << " MB) from " << gguf_files.size()
                      << " .gguf file(s) in '" << repo << "'.\n";
            items.push_back({"https://huggingface.co/" + repo + "/resolve/main/" + pick.filename,
                             base_name(pick.filename), pick.size});
        } else {
            // All .gguf files are shards: group by shard_group, rank groups by
            // the pick rank of the group name, take the best complete group.
            std::sort(gguf_files.begin(), gguf_files.end(),
                      [](const HfFileInfo& a, const HfFileInfo& b) {
                          if (a.shard_group != b.shard_group) {
                              int ra = gguf_pick_rank(a.shard_group), rb = gguf_pick_rank(b.shard_group);
                              if (ra != rb) return ra < rb;
                              return a.shard_group < b.shard_group;
                          }
                          return a.shard_index < b.shard_index;
                      });
            std::string group = gguf_files.front().shard_group;
            std::vector<HfFileInfo> parts;
            for (const auto& f : gguf_files) if (f.shard_group == group) parts.push_back(f);
            uint32_t total = parts.front().shard_total;
            bool complete = (parts.size() == total);
            for (uint32_t i = 1; i <= total && complete; ++i) {
                bool found = false;
                for (const auto& p : parts) if (p.shard_index == i) found = true;
                if (!found) complete = false;
            }
            if (!complete) {
                err_msg = "Sharded model '" + group + "' in repo '" + repo +
                          "' is incomplete on the Hub (found " + std::to_string(parts.size()) +
                          " of " + std::to_string(total) + " parts) — cannot download.";
                return false;
            }
            is_sharded_pick = true;
            shard_total = total;
            std::cout << "[PULL] Repo has only sharded .gguf files. Picked group '" << group
                      << "' (" << total << " parts).\n";
            for (const auto& p : parts) {
                items.push_back({"https://huggingface.co/" + repo + "/resolve/main/" + p.filename,
                                 base_name(p.filename), p.size});
            }
        }
    }

    fs::path target_dir = out_dir.empty() ? "models" : out_dir;
    {
        std::error_code ec;
        fs::create_directories(target_dir, ec);
        if (ec) {
            err_msg = "Cannot create output directory: " + target_dir.string();
            return false;
        }
    }

    std::cout << "\033[1;36m===================================================================\033[0m\n"
              << "\033[1;36m  MoE GGUF MODEL HUB STREAM DOWNLOADER\033[0m\n";
    for (const auto& it : items) {
        std::cout << "  Source URL:    \033[1;37m" << it.url << "\033[0m\n"
                  << "  Destination:   \033[1;32m" << (target_dir / it.filename).string() << "\033[0m\n";
    }
    std::cout << "\033[1;36m===================================================================\033[0m\n\n";

#ifdef _WIN32
    // Windows: WinINet download (no shell involved). One file at a time.
    for (const auto& it : items) {
        fs::path dest_path = target_dir / it.filename;
        out_file = dest_path.string();
        HINTERNET hInternet = InternetOpenA("MoE-Stream-Engine/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
        if (!hInternet) {
            err_msg = "Failed to initialize WinINet network subsystem (Error: " + std::to_string(GetLastError()) + ")";
            return false;
        }
        DWORD flags = INTERNET_FLAG_RELOAD | INTERNET_FLAG_DONT_CACHE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION;
        HINTERNET hFile = InternetOpenUrlA(hInternet, it.url.c_str(), nullptr, 0, flags, 0);
        if (!hFile) {
            err_msg = "Failed to connect to model URL (Error: " + std::to_string(GetLastError()) + ")";
            InternetCloseHandle(hInternet);
            return false;
        }
        DWORD status_code = 0;
        DWORD status_len = sizeof(status_code);
        if (HttpQueryInfoA(hFile, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER, &status_code, &status_len, nullptr)) {
            if (status_code >= 400) {
                err_msg = "Remote server returned HTTP " + std::to_string(status_code) + " for " + it.url;
                InternetCloseHandle(hFile);
                InternetCloseHandle(hInternet);
                return false;
            }
        }
        size_t total_bytes = it.size;
        char cl_buf[64] = {0};
        DWORD cl_len = sizeof(cl_buf) - 1;
        if (total_bytes == 0 && HttpQueryInfoA(hFile, HTTP_QUERY_CONTENT_LENGTH, cl_buf, &cl_len, nullptr)) {
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
            if (!InternetReadFile(hFile, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read) || bytes_read == 0) break;
            ofs.write(buffer.data(), bytes_read);
            downloaded += bytes_read;
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - t0).count();
            double update_diff = std::chrono::duration<double>(now - last_update).count();
            if (update_diff >= 0.1 || downloaded == total_bytes) {
                double mb_per_sec = (downloaded / (1024.0 * 1024.0)) / std::max(0.001, elapsed_sec);
                double eta_sec = (total_bytes > downloaded && mb_per_sec > 0.0)
                    ? (static_cast<double>(total_bytes - downloaded) / (mb_per_sec * 1024.0 * 1024.0)) : 0.0;
                if (on_progress) on_progress(downloaded, total_bytes, mb_per_sec, eta_sec);
                else render_progress_bar(downloaded, total_bytes, mb_per_sec, eta_sec);
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
        if (total_bytes > 0 && downloaded != total_bytes) {
            err_msg = "Size mismatch for " + dest_path.string() + ": downloaded " +
                      std::to_string(downloaded) + " bytes, expected " + std::to_string(total_bytes) +
                      " bytes — download incomplete, retry.";
            return false;
        }
        if (!validate_model_file(dest_path, it.filename, err_msg)) return false;
    }
#else
    // POSIX: curl via fork/exec (no shell), resume, Content-Length check.
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        fs::path dest_path = target_dir / it.filename;
        if (i == 0) out_file = dest_path.string();
        bool show_progress = (on_progress == nullptr);
        if (!curl_download(it.url, dest_path, it.size, show_progress,
                           (items.size() == 1) ? on_progress : nullptr, err_msg)) {
            return false;
        }
        if (!validate_model_file(dest_path, it.filename, err_msg)) return false;
        // SafeTensors needs its siblings to be usable: fetch config.json and
        // tokenizer.json from the same base URL. Absent siblings are
        // best-effort; corrupt ones fail the pull (see fetch_safetensors_sibling).
        if (ends_with_ci(it.filename, ".safetensors") &&
            (it.url.starts_with("https://huggingface.co/") || it.url.starts_with("http://"))) {
            size_t slash = it.url.find_last_of('/');
            std::string base_url = (slash == std::string::npos) ? it.url : it.url.substr(0, slash);
            // A corrupt sibling (proxy stub / empty file) fails the pull
            // loudly; a genuinely absent sibling is best-effort (see above).
            if (!fetch_safetensors_sibling(base_url, "config.json", target_dir, err_msg)) return false;
            if (!fetch_safetensors_sibling(base_url, "tokenizer.json", target_dir, err_msg)) return false;
        }
    }
#endif

    uint64_t total_downloaded = 0;
    for (const auto& it : items) {
        std::error_code ec;
        fs::path p = target_dir / it.filename;
        if (fs::exists(p, ec) && !ec) total_downloaded += fs::file_size(p, ec);
    }
    std::cout << "\n\033[1;32m[SUCCESS] MoE Model downloaded & verified successfully!\033[0m\n"
              << "  File Path:      " << out_file << "\n"
              << "  Total Size:     " << std::fixed << std::setprecision(2)
              << (total_downloaded / (1024.0 * 1024.0)) << " MB\n\n";
    if (is_sharded_pick) {
        std::cout << "\033[1;33m[WARNING] SHARDED MODEL — " << shard_total << " parts downloaded.\033[0m\n"
                  << "  The engine CANNOT load a single shard as a model. Merge the parts first:\n"
                  << "    ./build/moe_engine convert --merge " << out_file;
        for (size_t i = 1; i < items.size(); ++i) {
            std::cout << " " << (target_dir / items[i].filename).string();
        }
        std::cout << " -o " << (target_dir / "merged.gguf").string() << "\n\n";
    } else {
        std::cout << "To launch this model immediately with the Engine & Studio:\n"
                  << "  \033[1;33m./build/moe_engine serve --model " << out_file << "\033[0m\n\n";
    }
    return true;
}

} // namespace moe
