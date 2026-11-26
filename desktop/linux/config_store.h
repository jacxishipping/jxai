#pragma once
// XDG-ish config for jxai_term. HF token lives in a 0600 file, never in the repo.

#include <sys/stat.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace jxai::term {

namespace fs = std::filesystem;

inline std::string home_dir() {
    const char* h = std::getenv("HOME");
    return h && *h ? h : ".";
}

inline fs::path config_dir() {
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) return fs::path(xdg) / "jxai";
    return fs::path(home_dir()) / ".config" / "jxai";
}

inline fs::path term_config_path() { return config_dir() / "term.json"; }
inline fs::path hf_token_path() { return config_dir() / "hf_token"; }

inline bool ensure_config_dir() {
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    return !ec;
}

inline std::string read_file(const fs::path& p) {
    std::ifstream in(p);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline bool write_file(const fs::path& p, const std::string& body, mode_t mode = 0644) {
    if (!ensure_config_dir()) return false;
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << body;
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) return false;
    ::chmod(p.c_str(), mode);
    return true;
}

inline std::string json_get_string(const std::string& json, const std::string& key, const std::string& def = {}) {
    const std::string pat = "\"" + key + "\":\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) {
        // also allow without quotes for bools/numbers callers
        return def;
    }
    p += pat.size();
    std::string out;
    while (p < json.size()) {
        char c = json[p++];
        if (c == '\\' && p < json.size()) {
            char n = json[p++];
            if (n == 'n') out += '\n';
            else if (n == '"') out += '"';
            else if (n == '\\') out += '\\';
            else out += n;
        } else if (c == '"') break;
        else out += c;
    }
    return out;
}

inline bool json_get_bool(const std::string& json, const std::string& key, bool def) {
    const std::string pat = "\"" + key + "\":";
    size_t p = json.find(pat);
    if (p == std::string::npos) return def;
    p += pat.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
    if (json.compare(p, 4, "true") == 0) return true;
    if (json.compare(p, 5, "false") == 0) return false;
    return def;
}

struct TermConfig {
    bool first_run_completed = false;
    std::string model_path;
    std::string host = "127.0.0.1";
    uint16_t port = 8001;
    float temperature = 0.7f;
    uint32_t max_tokens = 512;
    std::string system_prompt;
};

inline TermConfig load_term_config() {
    TermConfig c;
    auto raw = read_file(term_config_path());
    if (raw.empty()) return c;
    c.first_run_completed = json_get_bool(raw, "first_run_completed", false);
    c.model_path = json_get_string(raw, "model_path");
    c.host = json_get_string(raw, "host", c.host);
    c.system_prompt = json_get_string(raw, "system_prompt");
    std::string port_s = json_get_string(raw, "port");
    if (!port_s.empty()) {
        try { c.port = static_cast<uint16_t>(std::stoi(port_s)); } catch (...) {}
    } else {
        // numeric form "port": 8001
        const std::string pat = "\"port\":";
        size_t p = raw.find(pat);
        if (p != std::string::npos) {
            try { c.port = static_cast<uint16_t>(std::stoi(raw.substr(p + pat.size()))); } catch (...) {}
        }
    }
    return c;
}

inline bool save_term_config(const TermConfig& c) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"first_run_completed\": " << (c.first_run_completed ? "true" : "false") << ",\n"
       << "  \"model_path\": \"" << c.model_path << "\",\n"
       << "  \"host\": \"" << c.host << "\",\n"
       << "  \"port\": " << c.port << ",\n"
       << "  \"temperature\": " << c.temperature << ",\n"
       << "  \"max_tokens\": " << c.max_tokens << ",\n"
       << "  \"system_prompt\": \"" << c.system_prompt << "\"\n"
       << "}\n";
    return write_file(term_config_path(), ss.str(), 0644);
}

inline std::string load_hf_token() {
    auto t = read_file(hf_token_path());
    // trim
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
    while (!t.empty() && (t.front() == ' ' || t.front() == '\n' || t.front() == '\r')) t.erase(t.begin());
    if (t.empty()) {
        const char* env = std::getenv("HF_TOKEN");
        if (env && *env) return env;
    }
    return t;
}

inline bool save_hf_token(const std::string& token) {
    if (token.empty()) {
        std::error_code ec;
        fs::remove(hf_token_path(), ec);
        return true;
    }
    return write_file(hf_token_path(), token + "\n", 0600);
}

// Keep moe_config.json in sync so ./start_jxai.sh picks up the same model.
inline bool update_moe_config_model(const std::string& model_path,
                                    const std::string& host,
                                    uint16_t port) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"first_run_completed\": true,\n"
       << "  \"model_path\": \"" << model_path << "\",\n"
       << "  \"api_host\": \"" << host << "\",\n"
       << "  \"api_port\": " << port << ",\n"
       << "  \"temperature\": 0.75,\n"
       << "  \"top_p\": 0.95,\n"
       << "  \"max_context\": 32768,\n"
       << "  \"active_slots\": 4,\n"
       << "  \"paged_attention_blocks\": 2048,\n"
       << "  \"bypass_io\": true,\n"
       << "  \"direct_dma\": true,\n"
       << "  \"engine\": \"MoE-Stream-Engine-v2\"\n"
       << "}\n";
    std::ofstream out("moe_config.json", std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << ss.str();
    return true;
}

inline bool needs_first_run_wizard(bool force_wizard) {
    if (force_wizard) return true;
    auto c = load_term_config();
    if (!c.first_run_completed) return true;
    if (c.model_path.empty()) return true;
    std::error_code ec;
    if (!fs::exists(c.model_path, ec) || fs::file_size(c.model_path, ec) == 0) return true;
    return false;
}

} // namespace jxai::term
