#include "model_pull.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
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
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
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
    // HTTPS only. The old code accepted http://, which sent HF_TOKEN over
    // plaintext whenever a token was configured.
    if (!url.starts_with("https://")) {
        err_msg = "URL must use https:// (plain http is not accepted for model downloads): '" + url + "'";
        return false;
    }
    return pull_url_is_safe(url, err_msg);
}

// Extract the host (without port / brackets) from an https URL.
// Returns false when the URL has no usable authority section.
bool url_host(const std::string& url, std::string& host_out) {
    size_t scheme = url.find("://");
    if (scheme == std::string::npos) return false;
    size_t begin = scheme + 3;
    size_t end = url.find_first_of("/?#", begin);
    std::string auth = url.substr(begin, end == std::string::npos ? end : end - begin);
    if (auth.empty()) return false;
    // Userinfo (user:pass@host) is never allowed in a download URL.
    if (auth.find('@') != std::string::npos) return false;
    if (!auth.empty() && auth.front() == '[') {
        size_t close = auth.find(']');
        if (close == std::string::npos) return false;
        host_out = auth.substr(1, close - 1);
        return true;
    }
    size_t colon = auth.find(':');
    host_out = (colon == std::string::npos) ? auth : auth.substr(0, colon);
    return !host_out.empty();
}

bool is_public_ipv4(uint32_t ip_host_order) {
    uint8_t a = static_cast<uint8_t>((ip_host_order >> 24) & 0xFF);
    uint8_t b = static_cast<uint8_t>((ip_host_order >> 16) & 0xFF);
    if (a == 0 || a == 10 || a == 127) return false;                    // 0/8, 10/8, 127/8
    if (a == 100 && b >= 64 && b <= 127) return false;                 // 100.64/10 CGNAT
    if (a == 169 && b == 254) return false;                            // 169.254/16 link-local
    if (a == 172 && b >= 16 && b <= 31) return false;                  // 172.16/12
    if (a == 192 && b == 0) return false;                              // 192.0/24 (incl. 192.0.0.0/24, 192.0.2/24 TEST-NET-1)
    if (a == 192 && b == 88) return false;                             // 192.88/16 (deprecated 6to4 relay)
    if (a == 192 && b == 168) return false;                            // 192.168/16
    if (a == 198 && (b == 18 || b == 19)) return false;                // 198.18/15 benchmarking
    if (a >= 224) return false;                                        // 224/4 multicast + 240/4 reserved
    if (a == 192 && b == 31) return false;                             // 192.31.196/24 (AS112, non-unicast use)
    return true;
}

bool is_public_ipv6(const uint8_t* addr) {
    // ::1, ::, fe80::/10 link-local, fc00::/7 unique-local, ff00::/8
    // multicast, 2001:db8::/32 documentation, 100::/64 discard.
    if (addr[0] == 0 && addr[1] == 0 && addr[2] == 0 && addr[3] == 0 &&
        addr[4] == 0 && addr[5] == 0 && addr[6] == 0 && addr[7] == 0 &&
        addr[8] == 0 && addr[9] == 0 && addr[10] == 0 && addr[11] == 0 &&
        addr[12] == 0 && addr[13] == 0 && addr[14] == 0 &&
        (addr[15] == 0 || addr[15] == 1)) return false;
    if (addr[0] == 0xFE && (addr[1] & 0xC0) == 0x80) return false;      // fe80::/10
    if ((addr[0] & 0xFE) == 0xFC) return false;                         // fc00::/7
    if (addr[0] == 0xFF) return false;                                  // ff00::/8
    if (addr[0] == 0x20 && addr[1] == 0x01 && addr[2] == 0x0D && addr[3] == 0xB8) return false; // 2001:db8::/32
    if (addr[0] == 0x01 && addr[1] == 0x00) return false;               // 100::/64
    return true;
}

// Syntax-only URL policy (no DNS): HTTPS, no userinfo, host present, and
// a literal-IP host must be public unicast. Hostnames pass this stage;
// pull_url_host_is_public() checks their resolution at download time.
bool pull_url_is_safe_impl(const std::string& url, std::string& err_msg) {
    if (!url.starts_with("https://")) {
        err_msg = "URL must use https://: '" + url + "'";
        return false;
    }
    if (has_shell_meta(url)) {
        err_msg = "URL contains shell metacharacters, whitespace or a query string (?): '" + url + "'";
        return false;
    }
    std::string host;
    if (!url_host(url, host)) {
        err_msg = "URL has no usable host: '" + url + "'";
        return false;
    }
    size_t host_begin = url.find("://") + 3;
    if (host_begin >= url.size() || url.find('/', host_begin) == std::string::npos) {
        err_msg = "URL has no path (a model file is required): '" + url + "'";
        return false;
    }
    // Literal IPv4?
    struct in_addr v4 {};
    if (::inet_pton(AF_INET, host.c_str(), &v4) == 1) {
        if (!is_public_ipv4(ntohl(v4.s_addr))) {
            err_msg = "URL host is not a public address (loopback/private/link-local rejected): '" + host + "'";
            return false;
        }
        return true;
    }
    // Literal IPv6?
    struct in6_addr v6 {};
    if (::inet_pton(AF_INET6, host.c_str(), &v6) == 1) {
        if (!is_public_ipv6(v6.s6_addr)) {
            err_msg = "URL host is not a public address (loopback/private/link-local rejected): '" + host + "'";
            return false;
        }
        return true;
    }
    // Hostname: syntactic checks only here (DNS stage is separate).
    if (host.size() > 253) {
        err_msg = "URL host is too long: '" + host + "'";
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

// Position just after the ':' of a depth-1 `"key":` in a JSON object, or
// npos. String-aware: keys inside string literals or nested objects never
// match. (The old helpers took the first occurrence anywhere, so a nested
// "size" or "oid" could shadow the top-level field.)
size_t find_top_level_key(const std::string& json, const std::string& key) {
    size_t i = 0;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t' || json[i] == '\n' || json[i] == '\r')) ++i;
    if (i >= json.size() || json[i] != '{') return std::string::npos;
    int depth = 0;
    bool in_str = false;
    for (size_t j = i; j < json.size(); ++j) {
        char c = json[j];
        if (in_str) {
            if (c == '\\' && j + 1 < json.size()) { ++j; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') {
            if (depth == 1) {
                // A string at depth 1 is a key or a value; read it and
                // compare only when it is followed by ':' (a key).
                size_t k = j + 1;
                std::string found;
                while (k < json.size() && json[k] != '"') {
                    if (json[k] == '\\' && k + 1 < json.size()) {
                        found.push_back(json[k]);
                        found.push_back(json[k + 1]);
                        k += 2;
                    } else {
                        found.push_back(json[k]);
                        ++k;
                    }
                }
                if (k >= json.size()) return std::string::npos;
                if (found == key) {
                    size_t p = k + 1;
                    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) ++p;
                    if (p < json.size() && json[p] == ':') return p + 1;
                    return std::string::npos;
                }
                j = k; // value string, not our key: skip it
                continue;
            }
            in_str = true;
        } else if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') {
            if (--depth == 0) break;
        }
    }
    return std::string::npos;
}

// Extract the string value for "key" at depth 1 of the given JSON object.
bool json_get_string(const std::string& json, const std::string& key, std::string& out) {
    size_t k = find_top_level_key(json, key);
    if (k == std::string::npos) return false;
    size_t i = k;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    if (i >= json.size() || json[i] != '"') return false;
    size_t j = i + 1;
    out.clear();
    while (j < json.size() && json[j] != '"') {
        if (json[j] == '\\' && j + 1 < json.size()) {
            ++j;
            switch (json[j]) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                default:  out.push_back(json[j]); break;
            }
            ++j;
        } else {
            out.push_back(json[j++]);
        }
    }
    return j < json.size();
}

bool json_get_number(const std::string& json, const std::string& key, double& out) {
    size_t k = find_top_level_key(json, key);
    if (k == std::string::npos) return false;
    size_t i = k;
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

// The Hub tree API reports a file's content hash in the nested lfs object:
// "lfs": {"oid": "sha256:<hex>", "size": N, ...}. The top-level "oid" is
// the git blob id, NOT the content hash — so the lfs object must be found
// first, then its oid. Returns false when there is no usable lfs oid.
bool lfs_sha256(const std::string& obj, std::string& sha_out) {
    size_t lfs = obj.find("\"lfs\"");
    if (lfs == std::string::npos) return false;
    size_t brace = obj.find('{', lfs);
    if (brace == std::string::npos) return false;
    int depth = 0;
    bool in_str = false;
    size_t end = std::string::npos;
    for (size_t j = brace; j < obj.size(); ++j) {
        char c = obj[j];
        if (in_str) {
            if (c == '\\' && j + 1 < obj.size()) ++j;
            else if (c == '"') in_str = false;
        } else if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            if (--depth == 0) { end = j; break; }
        }
    }
    if (end == std::string::npos) return false;
    std::string lfs_obj = obj.substr(brace, end - brace + 1);
    std::string oid;
    if (!json_get_string(lfs_obj, "oid", oid)) return false;
    const std::string prefix = "sha256:";
    if (oid.rfind(prefix, 0) != 0) return false;
    std::string hex = oid.substr(prefix.size());
    if (hex.size() != 64) return false;
    for (char c : hex) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    }
    sha_out = hex;
    return true;
}

// Parse one top-level object of the tree/main array: returns path/size/type
// and the LFS SHA-256 when the Hub reported one.
bool parse_tree_entry(const std::string& obj, std::string& path, uint64_t& size,
                      std::string& type, std::string& sha256) {
    if (!json_get_string(obj, "path", path)) return false;
    double sz = 0;
    if (!json_get_number(obj, "size", sz)) return false;
    size = static_cast<uint64_t>(sz);
    if (!json_get_string(obj, "type", type)) type = "file";
    (void)lfs_sha256(obj, sha256); // absent = no checksum available
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
        std::string path, type, sha256;
        uint64_t size = 0;
        if (!parse_tree_entry(obj, path, size, type, sha256)) continue;
        if (type != "file") continue;
        HfFileInfo info;
        info.filename = path;
        info.size = size;
        info.sha256 = sha256;
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
// max_stdout_bytes caps captured output (0 = unlimited); the HF tree API
// can return megabytes for repos with thousands of files, so API calls
// cap it instead of buffering unboundedly.
bool run_curl(const std::vector<std::string>& args, std::string& stdout_text,
              int& exit_code, bool capture_code, size_t max_stdout_bytes = 0) {
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
    bool capped = false;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
        if (max_stdout_bytes > 0 && stdout_text.size() + static_cast<size_t>(n) > max_stdout_bytes) {
            stdout_text.append(buf, max_stdout_bytes - stdout_text.size());
            capped = true;
            break;
        }
        stdout_text.append(buf, static_cast<size_t>(n));
    }
    ::close(pipefd[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (capped) {
        // Truncated output is unusable (e.g. a cut-off tree JSON): report
        // failure instead of letting the caller parse a partial document.
        // The child may still be writing; reap it without blocking.
        ::kill(pid, SIGTERM);
        while (::waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) break;
        }
        exit_code = -2;
        return false;
    }
    if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) exit_code = 128 + WTERMSIG(status);
    else exit_code = -1;
    return true;
}

// Extra curl args for HF auth when the user exported HF_TOKEN (needed for
// gated repos). The token is passed as an argv element, never via a shell,
// and ONLY to huggingface.co hosts: the old code attached it to every
// request, including downloads from arbitrary user-supplied URLs (and over
// plain http). curl itself strips Authorization on cross-host redirects
// (verified: a redirect to a different host/port arrives without the
// header), so the token never reaches a redirect target either.
std::vector<std::string> hf_auth_args_for(const std::string& url) {
    if (!url_is_huggingface(url)) return {};
    const char* tok = std::getenv("HF_TOKEN");
    if (tok != nullptr && tok[0] != '\0') {
        return {"-H", std::string("Authorization: Bearer ") + tok};
    }
    return {};
}

bool hf_api_get(const std::string& url, std::string& body, int& http_code, std::string& err_msg) {
    std::vector<std::string> args = {"curl", "-sS", "--max-time", "30", "--max-redirs", "5"};
    auto auth = hf_auth_args_for(url);
    args.insert(args.end(), auth.begin(), auth.end());
    args.push_back("-H");
    args.push_back("Accept: application/json");
    args.push_back(url);
    std::string out;
    int exit_code = -1;
    if (!run_curl(args, out, exit_code, true, 16 * 1024 * 1024)) {
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
// The effective URL after redirects is validated the same way as a
// download's: a HEAD that lands on a private host reports "unknown"
// instead of trusting the response.
bool http_content_length(const std::string& url, uint64_t& length) {
    std::vector<std::string> args = {"curl", "-sI", "-L", "--max-time", "15", "--max-redirs", "5",
                                     "-w", "\n%{url_effective}"};
    auto auth = hf_auth_args_for(url);
    args.insert(args.end(), auth.begin(), auth.end());
    args.push_back(url);
    std::string out;
    int exit_code = -1;
    if (!run_curl(args, out, exit_code, false) || exit_code != 0) return false;
    {
        std::string effective_url;
        size_t nl = out.rfind('\n');
        if (nl != std::string::npos) effective_url = out.substr(nl + 1);
        while (!effective_url.empty() && (effective_url.back() == '\r' || effective_url.back() == '\n')) {
            effective_url.pop_back();
        }
        if (!effective_url.empty() && effective_url != url) {
            std::string err;
            if (!effective_download_url_is_safe(effective_url, err)) return false;
        }
    }
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
// an already-complete file. expected_sha256 (from the Hub tree API "lfs"
// oid) is verified after the download when present; a mismatch deletes
// the file and fails — this is the cryptographic integrity check, distinct
// from the size + header checks below.
// Safety: the download goes to "<dest>.part" and is renamed to <dest> only
// after size + header + (when available) SHA-256 verification pass, so a
// failed or interrupted download never leaves a partial file at the model
// path. Timeouts: --connect-timeout 15 plus a stall abort (speed < 1 KiB/s
// for 60 s); there is no absolute --max-time because a legitimate multi-GB
// model can take hours. Redirects are capped at 5 and the effective URL
// curl ended on is validated (https + public host) before promotion.
// Returns false with err_msg on any failure.
bool curl_download(const std::string& url, const fs::path& dest, uint64_t expected_size,
                   const std::string& expected_sha256,
                   bool show_progress, PullProgressCallback on_progress, std::string& err_msg) {
    if (expected_size > kMaxDownloadBytes) {
        err_msg = "Refusing to download " + dest.string() + ": expected size " +
                  std::to_string(expected_size) + " bytes exceeds the " +
                  std::to_string(kMaxDownloadBytes) + "-byte limit";
        return false;
    }
    if (!pull_url_is_safe(url, err_msg)) return false;
    if (!pull_url_host_is_public(url, err_msg)) return false;

    std::error_code ec;
    fs::path tmp_path = dest;
    tmp_path += ".part";
    // A promoted file that is already complete is skipped. A partial
    // promoted file from an older version (which downloaded straight to
    // <dest>) is moved to the .part path so the resume below picks it up.
    if (fs::exists(dest, ec) && !ec) {
        uint64_t have = fs::file_size(dest, ec);
        if (!ec && expected_size > 0 && have == expected_size) {
            std::cout << "[PULL] Already present and complete, skipping download: " << dest.string() << "\n";
            if (on_progress) on_progress(have, expected_size, 0.0, 0.0);
            return true;
        }
        if (!ec && expected_size > 0 && have > expected_size) {
            err_msg = "Existing file is larger than expected (" + std::to_string(have) +
                      " > " + std::to_string(expected_size) + " bytes): " + dest.string() +
                      " — delete it and retry";
            return false;
        }
        if (!ec && have > 0 && !fs::exists(tmp_path, ec)) {
            std::cout << "[PULL] Moving partial promoted file to resume state: " << dest.string() << "\n";
            fs::rename(dest, tmp_path, ec);
            if (ec) {
                err_msg = "Cannot move partial file to resume state: " + dest.string();
                return false;
            }
        }
    }
    // No expected size (explicit URL / repo:file): ask the server first so
    // an already-complete file is recognised instead of re-downloaded.
    if (expected_size == 0) {
        uint64_t remote = 0;
        if (http_content_length(url, remote)) {
            expected_size = remote;
            if (fs::exists(tmp_path, ec) && !ec) {
                uint64_t have = fs::file_size(tmp_path, ec);
                if (!ec && have > 0 && have == remote) {
                    // Already complete: validate and promote, do not download.
                    if (!validate_model_file(tmp_path, tmp_path.filename().string(), err_msg)) {
                        fs::remove(tmp_path, ec);
                        return false;
                    }
                    fs::rename(tmp_path, dest, ec);
                    if (ec) {
                        err_msg = "Could not promote downloaded file to " + dest.string();
                        return false;
                    }
                    std::cout << "[PULL] Already present and complete, skipping download: " << dest.string() << "\n";
                    if (on_progress) on_progress(have, remote, 0.0, 0.0);
                    return true;
                }
                if (!ec && have > remote) {
                    err_msg = "Existing file is larger than the remote file (" + std::to_string(have) +
                              " > " + std::to_string(remote) + " bytes): " + dest.string() +
                              " — delete it and retry";
                    return false;
                }
            }
        }
    }
    {
        uint64_t have = 0;
        if (fs::exists(tmp_path, ec) && !ec) have = fs::file_size(tmp_path, ec);
        if (!ec && have > 0 && expected_size > 0) {
            std::cout << "[PULL] Resuming partial download (" << have << " / " << expected_size
                      << " bytes): " << dest.string() << "\n";
        }
    }

    fs::path headers_path = tmp_path;
    headers_path += ".headers";
    std::vector<std::string> args = {"curl", "-L", "--fail", "-C", "-", "-o", tmp_path.string(),
                                     "-D", headers_path.string(),
                                     "--connect-timeout", "15",
                                     "--speed-limit", "1024", "--speed-time", "60",
                                     "--max-redirs", "5",
                                     "--max-filesize", std::to_string(kMaxDownloadBytes),
                                     "-w", "\n%{url_effective}"};
    auto auth = hf_auth_args_for(url);
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
                  (exit_code == 63 ? " — download exceeded the 200 GiB size limit" : "") +
                  ") for " + url;
        return false;
    }
    // The -w output is the last line of stdout: the effective URL after
    // redirects. A redirect chain that landed on a non-public host (SSRF)
    // or a non-HTTPS URL fails here and the file is deleted, not promoted.
    {
        std::string effective_url;
        size_t nl = out.rfind('\n');
        if (nl != std::string::npos) effective_url = out.substr(nl + 1);
        // Strip a trailing carriage return (curl -w output quirk).
        while (!effective_url.empty() && (effective_url.back() == '\r' || effective_url.back() == '\n')) {
            effective_url.pop_back();
        }
        if (effective_url.empty() || effective_url == url) {
            // No redirect happened (or curl gave nothing back): validate
            // the original URL, which was already checked above.
        } else if (!effective_download_url_is_safe(effective_url, err_msg)) {
            err_msg = "Download redirected to an unsafe URL (" + effective_url + "): " + err_msg;
            fs::remove(tmp_path, ec);
            fs::remove(headers_path, ec);
            return false;
        }
    }
    if (!fs::exists(tmp_path, ec) || ec || fs::file_size(tmp_path, ec) == 0) {
        err_msg = "Downloaded file does not exist or is empty: " + dest.string();
        return false;
    }
    uint64_t downloaded = fs::file_size(tmp_path, ec);

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
    // Cryptographic integrity check when the Hub reported a checksum.
    // A missing checksum is NOT a pass: it is reported as header+size
    // only (see pull_model_from_hub's success message).
    if (!expected_sha256.empty()) {
        std::string actual;
        if (!sha256_file_hex(tmp_path.string(), actual)) {
            err_msg = "Could not hash downloaded file for integrity check: " + dest.string();
            fs::remove(tmp_path, ec);
            return false;
        }
        if (to_lower(actual) != to_lower(expected_sha256)) {
            err_msg = "SHA-256 mismatch for " + dest.string() + ": expected " + expected_sha256 +
                      ", got " + actual + " — download corrupt or tampered, deleting file";
            fs::remove(tmp_path, ec);
            return false;
        }
        std::cout << "[PULL] Integrity verified (SHA-256): " << dest.string() << "\n";
    }
    if (!validate_model_file(tmp_path, tmp_path.filename().string(), err_msg)) {
        fs::remove(tmp_path, ec);
        return false;
    }
    // Atomic promotion: only a fully verified file ever appears at <dest>.
    fs::rename(tmp_path, dest, ec);
    if (ec) {
        err_msg = "Could not promote downloaded file to " + dest.string();
        fs::remove(tmp_path, ec);
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

// (validate_model_file now lives at moe scope, with structural validation;
// see below the anonymous namespace.)

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
        if (!curl_download(url, dest, 0, "", false, nullptr, err)) {
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

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained so integrity verification works on
// every platform without an external tool. Verified against the standard
// test vectors in the unit tests (see run_verification_tests, section 11).
// ---------------------------------------------------------------------------
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        h_[0] = 0x6a09e667; h_[1] = 0xbb67ae85; h_[2] = 0x3c6ef372; h_[3] = 0xa54ff53a;
        h_[4] = 0x510e527f; h_[5] = 0x9b05688c; h_[6] = 0x1f83d9ab; h_[7] = 0x5be0cd19;
        total_len_ = 0;
        buf_len_ = 0;
    }
    void update(const uint8_t* data, size_t len) {
        total_len_ += len;
        while (len > 0) {
            size_t take = std::min(len, sizeof(buf_) - buf_len_);
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            len -= take;
            if (buf_len_ == sizeof(buf_)) {
                compress(buf_);
                buf_len_ = 0;
            }
        }
    }
    void finish(uint8_t out[32]) {
        const uint64_t bits = total_len_ * 8; // captured before padding
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (buf_len_ != 56) update(&zero, 1);
        uint8_t len_bytes[8];
        for (int k = 0; k < 8; ++k) {
            len_bytes[k] = static_cast<uint8_t>((bits >> (56 - 8 * k)) & 0xFF);
        }
        update(len_bytes, 8);
        for (int k = 0; k < 8; ++k) {
            out[4 * k] = static_cast<uint8_t>((h_[k] >> 24) & 0xFF);
            out[4 * k + 1] = static_cast<uint8_t>((h_[k] >> 16) & 0xFF);
            out[4 * k + 2] = static_cast<uint8_t>((h_[k] >> 8) & 0xFF);
            out[4 * k + 3] = static_cast<uint8_t>(h_[k] & 0xFF);
        }
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void compress(const uint8_t* block) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t w[64];
        for (int t = 0; t < 16; ++t) {
            w[t] = (static_cast<uint32_t>(block[4 * t]) << 24) |
                   (static_cast<uint32_t>(block[4 * t + 1]) << 16) |
                   (static_cast<uint32_t>(block[4 * t + 2]) << 8) |
                   static_cast<uint32_t>(block[4 * t + 3]);
        }
        for (int t = 16; t < 64; ++t) {
            uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[t] + w[t];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_ = 0;
    uint64_t total_len_ = 0;
};

} // namespace

// ---------------------------------------------------------------------------
// Public download-safety helpers (declared in model_pull.h).
// ---------------------------------------------------------------------------

bool pull_url_is_safe(const std::string& url, std::string& err_msg) {
    return pull_url_is_safe_impl(url, err_msg);
}

bool url_is_huggingface(const std::string& url) {
    std::string host;
    if (!url_host(url, host)) return false;
    std::string h = to_lower(host);
    return h == "huggingface.co" || (h.size() > 15 && h.ends_with(".huggingface.co"));
}

bool pull_url_host_is_public(const std::string& url, std::string& err_msg) {
    std::string host;
    if (!url_host(url, host)) {
        err_msg = "URL has no usable host: '" + url + "'";
        return false;
    }
    // Literal IPs were already checked by pull_url_is_safe(); a hostname
    // needs a real DNS lookup. Fail closed: unresolvable == rejected.
    // (DNS rebinding is the reason this runs even for huggingface.co:
    // a poisoned resolver could otherwise point the trusted name at a
    // private address and receive the request — and the token.)
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    int gai = ::getaddrinfo(host.c_str(), "443", &hints, &res);
    if (gai != 0 || res == nullptr) {
        err_msg = "Cannot resolve download host (refusing to download): '" + host + "'";
        return false;
    }
    bool ok = true;
    for (struct addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        if (ai->ai_family == AF_INET && ai->ai_addr != nullptr) {
            auto* sin = reinterpret_cast<struct sockaddr_in*>(ai->ai_addr);
            if (!is_public_ipv4(ntohl(sin->sin_addr.s_addr))) { ok = false; break; }
        } else if (ai->ai_family == AF_INET6 && ai->ai_addr != nullptr) {
            auto* sin6 = reinterpret_cast<struct sockaddr_in6*>(ai->ai_addr);
            if (!is_public_ipv6(sin6->sin6_addr.s6_addr)) { ok = false; break; }
        }
    }
    ::freeaddrinfo(res);
    if (!ok) {
        err_msg = "Download host resolves to a non-public address (refusing to download): '" + host + "'";
        return false;
    }
    return true;
}

bool effective_download_url_is_safe(const std::string& url, std::string& err_msg) {
    // Hugging Face CDN redirects (cdn-lfs / *.cdn.hf.co / Xet) use signed
    // URLs with ?Expires=&Signature=… query strings. Caller-supplied pull
    // URLs still reject `?` (pull_url_is_safe), but the post-redirect
    // effective URL must allow a query/fragment while keeping the same
    // HTTPS + public-host rules on the authority/path.
    std::string base = url;
    const size_t cut = base.find_first_of("?#");
    if (cut != std::string::npos) base.resize(cut);
    if (!pull_url_is_safe(base, err_msg)) return false;
    return pull_url_host_is_public(url, err_msg);
}

std::string sha256_hex(const uint8_t* data, size_t len) {
    Sha256 h;
    h.update(data, len);
    uint8_t digest[32];
    h.finish(digest);
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) {
        out.push_back(kHex[(digest[i] >> 4) & 0xF]);
        out.push_back(kHex[digest[i] & 0xF]);
    }
    return out;
}

bool sha256_file_hex(const std::string& path, std::string& out_hex) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    Sha256 h;
    std::vector<uint8_t> buf(1 << 20);
    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        std::streamsize n = in.gcount();
        if (n > 0) h.update(buf.data(), static_cast<size_t>(n));
    }
    uint8_t digest[32];
    h.finish(digest);
    static const char* kHex = "0123456789abcdef";
    out_hex.clear();
    out_hex.reserve(64);
    for (int i = 0; i < 32; ++i) {
        out_hex.push_back(kHex[(digest[i] >> 4) & 0xF]);
        out_hex.push_back(kHex[digest[i] & 0xF]);
    }
    return true;
}

// Validate a SafeTensors header JSON document (already read into `header`).
// Checks: it is a JSON object; every tensor entry (except "__metadata__")
// has a known dtype and data_offsets [begin, end] with
// 0 <= begin <= end <= data_len. The full tensor metadata is parsed again
// by the loader; this is the download-time structural gate, not a
// replacement for it.
bool validate_safetensors_header(const std::string& header, uint64_t data_len,
                                 std::string& err_msg) {
    size_t first = header.find_first_not_of(" \t\r\n");
    size_t last = header.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || header[first] != '{' || header[last] != '}') {
        err_msg = "SafeTensors header is not a JSON object";
        return false;
    }
    static const char* kDtypes[] = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16",
                                    "I8", "U8", "U16", "U32", "U64", "BOOL",
                                    "F8_E4M3", "F8_E5M2"};
    size_t i = first + 1;
    while (i < last) {
        while (i < last && (header[i] == ' ' || header[i] == '\t' || header[i] == '\n' ||
                            header[i] == '\r' || header[i] == ',')) ++i;
        if (i >= last) break;
        if (header[i] != '"') {
            err_msg = "SafeTensors header has a malformed entry (expected a tensor name)";
            return false;
        }
        size_t name_end = header.find('"', i + 1);
        if (name_end == std::string::npos) {
            err_msg = "SafeTensors header has an unterminated tensor name";
            return false;
        }
        std::string name = header.substr(i + 1, name_end - i - 1);
        size_t obj_start = header.find('{', name_end);
        if (obj_start == std::string::npos) {
            err_msg = "SafeTensors header entry '" + name + "' has no object";
            return false;
        }
        int depth = 0;
        bool in_str = false;
        size_t obj_end = std::string::npos;
        for (size_t j = obj_start; j <= last; ++j) {
            char c = header[j];
            if (in_str) {
                if (c == '\\' && j + 1 <= last) ++j;
                else if (c == '"') in_str = false;
            } else if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') {
                if (--depth == 0) { obj_end = j; break; }
            }
        }
        if (obj_end == std::string::npos) {
            err_msg = "SafeTensors header entry '" + name + "' is unbalanced";
            return false;
        }
        if (name != "__metadata__") {
            std::string entry = header.substr(obj_start, obj_end - obj_start + 1);
            std::string dtype;
            if (!json_get_string(entry, "dtype", dtype)) {
                err_msg = "SafeTensors tensor '" + name + "' has no dtype";
                return false;
            }
            bool known = false;
            for (const char* d : kDtypes) if (dtype == d) { known = true; break; }
            if (!known) {
                err_msg = "SafeTensors tensor '" + name + "' has unknown dtype '" + dtype + "'";
                return false;
            }
            size_t dpos = entry.find("\"data_offsets\"");
            if (dpos == std::string::npos) {
                err_msg = "SafeTensors tensor '" + name + "' has no data_offsets";
                return false;
            }
            size_t b1 = entry.find('[', dpos);
            size_t comma = (b1 == std::string::npos) ? std::string::npos : entry.find(',', b1);
            size_t b2 = (comma == std::string::npos) ? std::string::npos : entry.find(']', comma);
            if (b1 == std::string::npos || comma == std::string::npos || b2 == std::string::npos) {
                err_msg = "SafeTensors tensor '" + name + "' has malformed data_offsets";
                return false;
            }
            auto parse_u64 = [](const std::string& s, uint64_t& out) {
                size_t a = s.find_first_not_of(" \t");
                if (a == std::string::npos) return false;
                size_t b = s.find_last_not_of(" \t");
                std::string t = s.substr(a, b - a + 1);
                if (t.empty() || !std::all_of(t.begin(), t.end(), ::isdigit)) return false;
                try { out = std::stoull(t); } catch (...) { return false; }
                return true;
            };
            uint64_t begin = 0, end = 0;
            if (!parse_u64(entry.substr(b1 + 1, comma - b1 - 1), begin) ||
                !parse_u64(entry.substr(comma + 1, b2 - comma - 1), end)) {
                err_msg = "SafeTensors tensor '" + name + "' has non-numeric data_offsets";
                return false;
            }
            if (begin > end || end > data_len) {
                err_msg = "SafeTensors tensor '" + name + "' data_offsets [" +
                          std::to_string(begin) + ", " + std::to_string(end) +
                          "] are outside the file data (" + std::to_string(data_len) + " bytes)";
                return false;
            }
        }
        i = obj_end + 1;
    }
    return true;
}

bool validate_model_file(const fs::path& path, const std::string& filename, std::string& err_msg) {
    std::error_code ec;
    uint64_t fsize = fs::file_size(path, ec);
    if (ec) {
        err_msg = "Cannot open downloaded file for validation: " + path.string();
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err_msg = "Cannot open downloaded file for validation: " + path.string();
        return false;
    }
    // GGUF integers are little-endian, matching the loader's assumption on
    // x86-64 / ARM64 hosts.
    auto u32le = [](const char* p) {
        return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
               (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
               (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
    };
    auto u64le = [](const char* p) {
        uint64_t v = 0;
        for (int k = 0; k < 8; ++k) {
            v |= static_cast<uint64_t>(static_cast<uint8_t>(p[k])) << (8 * k);
        }
        return v;
    };
    if (ends_with_ci(filename, ".gguf")) {
        if (fsize < 24) {
            err_msg = "File too small to be a GGUF file (24-byte header required, got " +
                      std::to_string(fsize) + " bytes): " + path.string();
            return false;
        }
        char hdr[24] = {0};
        in.read(hdr, 24);
        if (in.gcount() != 24 || std::memcmp(hdr, "GGUF", 4) != 0) {
            err_msg = "File downloaded but failed GGUF magic validation: " + path.string() +
                      " (not a GGUF file — check the repo/file name)";
            return false;
        }
        uint32_t version = u32le(hdr + 4);
        if (version != 2 && version != 3) {
            err_msg = "Unsupported GGUF version " + std::to_string(version) +
                      " (expected 2 or 3): " + path.string();
            return false;
        }
        uint64_t tensor_count = u64le(hdr + 8);
        uint64_t metadata_count = u64le(hdr + 16);
        if (tensor_count == 0 || tensor_count > 1000000 || metadata_count > 1000000) {
            err_msg = "GGUF header counts are implausible (tensor_count=" +
                      std::to_string(tensor_count) + ", metadata_count=" +
                      std::to_string(metadata_count) + ") — corrupt file: " + path.string();
            return false;
        }
        return true;
    }
    if (ends_with_ci(filename, ".safetensors")) {
        if (fsize < 8) {
            err_msg = "File too small to be a SafeTensors file (8-byte header length required): " +
                      path.string();
            return false;
        }
        uint64_t header_len = 0;
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (in.gcount() != 8 || header_len == 0) {
            err_msg = "File downloaded but failed SafeTensors header validation: " + path.string();
            return false;
        }
        if (header_len > 64 * 1024 * 1024) {
            err_msg = "SafeTensors header length implausibly large (" +
                      std::to_string(header_len) + " bytes): " + path.string();
            return false;
        }
        if (8 + header_len > fsize) {
            err_msg = "File downloaded but SafeTensors header length exceeds file size: " + path.string();
            return false;
        }
        std::string header(static_cast<size_t>(header_len), '\0');
        in.read(header.data(), static_cast<std::streamsize>(header_len));
        if (in.gcount() != static_cast<std::streamsize>(header_len)) {
            err_msg = "Short read on SafeTensors header: " + path.string();
            return false;
        }
        if (!validate_safetensors_header(header, fsize - 8 - header_len, err_msg)) {
            err_msg += ": " + path.string();
            return false;
        }
        return true;
    }
    std::cout << "[PULL] WARNING: no header validation for '" << filename << "' (unknown extension)\n";
    return true;
}

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
    struct DownloadItem {
        std::string url;
        std::string filename;
        uint64_t size = 0;
        std::string sha256; // Hub lfs oid when available; "" = no checksum
    };
    std::vector<DownloadItem> items;
    bool is_sharded_pick = false;
    uint32_t shard_total = 0;

    if (input.starts_with("http://") || input.starts_with("https://")) {
        if (!valid_url(input, err_msg)) return false;
        std::string filename = base_name(input);
        if (!valid_model_filename(filename, err_msg)) return false;
        items.push_back({input, filename, 0, ""});
    } else if (input.find(':') != std::string::npos) {
        // "owner/name:file.gguf" — explicit file, no API lookup needed.
        size_t colon = input.find(':');
        std::string repo = input.substr(0, colon);
        std::string filename = input.substr(colon + 1);
        if (!valid_repo_id(repo, err_msg)) return false;
        if (!valid_model_filename(filename, err_msg)) return false;
        items.push_back({"https://huggingface.co/" + repo + "/resolve/main/" + filename, filename, 0, ""});
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
                             base_name(pick.filename), pick.size, pick.sha256});
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
                                 base_name(p.filename), p.size, p.sha256});
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
    // Same safety policy as the POSIX path: URL validation, temp file +
    // atomic promotion, size cap, receive timeout, SHA-256 when the Hub
    // reported one. (WinINet follows redirects internally, so the
    // effective-URL check the POSIX path does after curl is not available
    // here; the initial URL is validated before any bytes are fetched.)
    for (const auto& it : items) {
        if (it.size > kMaxDownloadBytes) {
            err_msg = "Refusing to download " + it.filename + ": expected size " +
                      std::to_string(it.size) + " bytes exceeds the " +
                      std::to_string(kMaxDownloadBytes) + "-byte limit";
            return false;
        }
        if (!pull_url_is_safe(it.url, err_msg)) return false;
        if (!pull_url_host_is_public(it.url, err_msg)) return false;
        fs::path dest_path = target_dir / it.filename;
        fs::path tmp_path = dest_path;
        tmp_path += ".part";
        out_file = dest_path.string();
        HINTERNET hInternet = InternetOpenA("MoE-Stream-Engine/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
        if (!hInternet) {
            err_msg = "Failed to initialize WinINet network subsystem (Error: " + std::to_string(GetLastError()) + ")";
            return false;
        }
        DWORD flags = INTERNET_FLAG_RELOAD | INTERNET_FLAG_DONT_CACHE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION;
        // HF_TOKEN is attached only to huggingface.co hosts (same policy
        // as the POSIX path); other hosts never see it.
        std::string win_headers;
        if (url_is_huggingface(it.url)) {
            char tok_buf[512] = {0};
            if (GetEnvironmentVariableA("HF_TOKEN", tok_buf, sizeof(tok_buf)) > 0) {
                win_headers = "Authorization: Bearer " + std::string(tok_buf) + "\r\n";
            }
        }
        HINTERNET hFile = InternetOpenUrlA(hInternet, it.url.c_str(),
                                           win_headers.empty() ? nullptr : win_headers.c_str(),
                                           static_cast<DWORD>(win_headers.size()),
                                           flags, 0);
        if (!hFile) {
            err_msg = "Failed to connect to model URL (Error: " + std::to_string(GetLastError()) + ")";
            InternetCloseHandle(hInternet);
            return false;
        }
        DWORD recv_timeout_ms = 60000; // stall abort: 60 s without data
        InternetSetOptionA(hFile, INTERNET_OPTION_RECEIVE_TIMEOUT, &recv_timeout_ms, sizeof(recv_timeout_ms));
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
        if (total_bytes > kMaxDownloadBytes) {
            err_msg = "Refusing to download " + it.filename + ": remote size " +
                      std::to_string(total_bytes) + " bytes exceeds the " +
                      std::to_string(kMaxDownloadBytes) + "-byte limit";
            InternetCloseHandle(hFile);
            InternetCloseHandle(hInternet);
            return false;
        }
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs) {
            err_msg = "Could not open destination file for writing: " + tmp_path.string();
            InternetCloseHandle(hFile);
            InternetCloseHandle(hInternet);
            return false;
        }
        std::vector<char> buffer(64 * 1024);
        size_t downloaded = 0;
        auto t0 = std::chrono::steady_clock::now();
        auto last_update = t0;
        bool over_cap = false;
        while (true) {
            DWORD bytes_read = 0;
            if (!InternetReadFile(hFile, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read) || bytes_read == 0) break;
            ofs.write(buffer.data(), bytes_read);
            downloaded += bytes_read;
            if (downloaded > kMaxDownloadBytes) { over_cap = true; break; }
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
        if (over_cap) {
            std::error_code ec;
            fs::remove(tmp_path, ec);
            err_msg = "Download exceeded the 200 GiB size limit: " + dest_path.string();
            return false;
        }
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
        if (!it.sha256.empty()) {
            std::string actual;
            if (!sha256_file_hex(tmp_path.string(), actual) || to_lower(actual) != to_lower(it.sha256)) {
                std::error_code ec;
                fs::remove(tmp_path, ec);
                err_msg = "SHA-256 mismatch for " + dest_path.string() + " — download corrupt or tampered, deleting file";
                return false;
            }
            std::cout << "[PULL] Integrity verified (SHA-256): " << dest_path.string() << "\n";
        }
        if (!validate_model_file(tmp_path, it.filename, err_msg)) {
            std::error_code ec;
            fs::remove(tmp_path, ec);
            return false;
        }
        {
            std::error_code ec;
            fs::rename(tmp_path, dest_path, ec);
            if (ec) {
                err_msg = "Could not promote downloaded file to " + dest_path.string();
                fs::remove(tmp_path, ec);
                return false;
            }
        }
    }
#else
    // POSIX: curl via fork/exec (no shell), resume, Content-Length check.
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        fs::path dest_path = target_dir / it.filename;
        if (i == 0) out_file = dest_path.string();
        bool show_progress = (on_progress == nullptr);
        if (!curl_download(it.url, dest_path, it.size, it.sha256, show_progress,
                           (items.size() == 1) ? on_progress : nullptr, err_msg)) {
            return false;
        }
        if (!validate_model_file(dest_path, it.filename, err_msg)) return false;
        // SafeTensors needs its siblings to be usable: fetch config.json and
        // tokenizer.json from the same base URL. Absent siblings are
        // best-effort; corrupt ones fail the pull (see fetch_safetensors_sibling).
        // Siblings are only fetched from Hugging Face, never from an
        // arbitrary user-supplied host.
        if (ends_with_ci(it.filename, ".safetensors") &&
            it.url.starts_with("https://huggingface.co/")) {
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
              << (total_downloaded / (1024.0 * 1024.0)) << " MB\n";
    {
        // Say plainly which files got cryptographic verification and which
        // did not: a SHA-256 match against the Hub's lfs oid is integrity
        // verification; size + header checks alone are not, and must never
        // be reported as if they were.
        size_t with_sha = 0;
        for (const auto& it : items) if (!it.sha256.empty()) ++with_sha;
        if (with_sha == items.size()) {
            std::cout << "  Integrity:      SHA-256 verified against the Hugging Face Hub for all "
                      << items.size() << " file(s).\n";
        } else if (with_sha > 0) {
            std::cout << "  Integrity:      SHA-256 verified for " << with_sha << " of " << items.size()
                      << " file(s); the rest had no checksum from the Hub (size + header checks only).\n";
        } else {
            std::cout << "  Integrity:      NO SHA-256 checksum was available for these files "
                      << "(explicit URL / repo:file) — verified by size + header checks only, "
                      << "not cryptographic integrity.\n";
        }
    }
    std::cout << "\n";
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
