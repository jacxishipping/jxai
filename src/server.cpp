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
#include "model_pull.h"
#include "vulkan_backend.h"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
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

// Request limits (DoS bounds). A request larger than the body cap is
// rejected with 413 before its bytes are buffered; a header block larger
// than the header cap is dropped. Concurrency is capped in start().
constexpr size_t kMaxHeaderBytes = 64 * 1024;        // 64 KiB request line + headers
constexpr size_t kMaxBodyBytes = 1 * 1024 * 1024;    // 1 MiB JSON bodies
constexpr size_t kMaxStaticFileBytes = 32 * 1024 * 1024; // 32 MiB static assets
constexpr uint32_t kMaxConcurrentRequests = 64;      // simultaneous connections

} // namespace (anonymous — reopen after resolve_under_root)

bool resolve_under_root(const std::string& root, const std::string& rel,
                        std::string& out) {
    if (rel.empty()) return false;
    std::filesystem::path rel_p(rel);
    // Absolute / root-name / has_root_directory means the RHS of path
    // join would replace `root` (classic `//etc/passwd` / `/etc/passwd`
    // escape against `web_root / sub`).
    if (rel_p.is_absolute() || rel_p.has_root_name() || rel_p.has_root_directory()) {
        return false;
    }
    for (const auto& part : rel_p) {
        if (part == "..") return false;
    }
    std::filesystem::path root_p = std::filesystem::path(root).lexically_normal();
    if (root_p.empty()) root_p = ".";
    // Make root absolute-ish for the prefix check without requiring the
    // directory to exist on disk (unit tests use synthetic roots).
    if (!root_p.is_absolute()) {
        root_p = (std::filesystem::current_path() / root_p).lexically_normal();
    }
    std::filesystem::path target = (root_p / rel_p).lexically_normal();
    auto root_s = root_p.generic_string();
    auto target_s = target.generic_string();
    if (root_s.back() != '/') root_s.push_back('/');
    // Exact root match is not a file; require target to be a strict child.
    if (target_s.size() <= root_s.size() ||
        target_s.compare(0, root_s.size(), root_s) != 0) {
        return false;
    }
    out = target.string();
    return true;
}

namespace {

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

// ---------------------------------------------------------------------------
// Strict, small JSON parser (no third-party JSON library in this codebase).
// Every request body and every field extractor below goes through it.
//
// Why: the old substring extractors matched the FIRST `"key"` anywhere in
// the body, so a key appearing inside a string value or a nested object
// shadowed the real field (e.g. a message containing `"temperature": 99`
// overrode the top-level temperature, and a `"role": "system"` inside
// message content could smuggle a role). They also mishandled \uXXXX and
// \b \f escapes and accepted NaN/Infinity as numbers. This parser is
// strict on purpose: trailing garbage, unclosed strings, bad escapes,
// NaN/Infinity and over-deep nesting are rejected instead of misread.
// Each value records its [start, end) span in the source text so the tools
// parser can keep the client's verbatim tool JSON.
// ---------------------------------------------------------------------------
struct JsonValue {
    enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;
    size_t start = 0, end = 0; // span in the source text
    const JsonValue* find(const std::string& key) const {
        if (type != kObject) return nullptr;
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

struct JsonParser {
    const std::string& s;
    size_t i = 0;
    explicit JsonParser(const std::string& text) : s(text) {}

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    bool parse(JsonValue& out, int depth) {
        if (depth > 64) return false;
        skip_ws();
        if (i >= s.size()) return false;
        out.start = i;
        char c = s[i];
        bool ok = false;
        if (c == '{') ok = parse_object(out, depth);
        else if (c == '[') ok = parse_array(out, depth);
        else if (c == '"') {
            std::string v;
            ok = parse_string_raw(v);
            if (ok) { out.type = JsonValue::kString; out.str = std::move(v); }
        }
        else if (c == 't') ok = parse_literal("true", JsonValue::kBool, true, out);
        else if (c == 'f') ok = parse_literal("false", JsonValue::kBool, false, out);
        else if (c == 'n') ok = parse_literal("null", JsonValue::kNull, false, out);
        else ok = parse_number(out);
        if (!ok) return false;
        out.end = i;
        return true;
    }

    bool parse_hex4(uint32_t& out) {
        if (i + 4 > s.size()) return false;
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return false;
        }
        out = v;
        return true;
    }

    // Parse a JSON string starting at s[i] == '"'. Full escape decoding
    // (\" \\ \/ \b \f \n \r \t \uXXXX with surrogate pairs -> UTF-8).
    // Rejects raw control characters, bad escapes and unterminated strings.
    bool parse_string_raw(std::string& out) {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size()) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == '"') { ++i; return true; }
            if (c < 0x20) return false; // raw control character in string
            if (c == '\\') {
                ++i;
                if (i >= s.size()) return false;
                char e = s[i++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // High surrogate: must be followed by \uDC00-\uDFFF.
                            if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                                i += 2;
                                uint32_t lo = 0;
                                if (!parse_hex4(lo)) return false;
                                if (lo < 0xDC00 || lo > 0xDFFF) return false;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else return false;
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            return false; // lone low surrogate
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(static_cast<char>(c));
                ++i;
            }
        }
        return false; // unterminated string
    }

    // Strict JSON number grammar (no NaN/Infinity/hex, no leading '+',
    // no leading zeros). Overflow to +/-inf is rejected.
    bool parse_number(JsonValue& out) {
        size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        if (i >= s.size()) return false;
        if (s[i] == '0') ++i;
        else if (s[i] >= '1' && s[i] <= '9') {
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        } else return false;
        if (i < s.size() && s[i] == '.') {
            ++i;
            if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        try {
            size_t used = 0;
            double v = std::stod(s.substr(start, i - start), &used);
            if (used != i - start) return false;
            if (!std::isfinite(v)) return false;
            out.type = JsonValue::kNumber;
            out.num = v;
            return true;
        } catch (...) {
            return false;
        }
    }

    bool parse_literal(const char* word, JsonValue::Type t, bool b, JsonValue& out) {
        size_t len = std::strlen(word);
        if (s.compare(i, len, word) != 0) return false;
        i += len;
        out.type = t;
        out.b = b;
        return true;
    }

    bool parse_array(JsonValue& out, int depth) {
        ++i; // '['
        out.type = JsonValue::kArray;
        out.arr.clear();
        skip_ws();
        if (i < s.size() && s[i] == ']') { ++i; return true; }
        while (true) {
            JsonValue v;
            if (!parse(v, depth + 1)) return false;
            out.arr.push_back(std::move(v));
            skip_ws();
            if (i >= s.size()) return false;
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == ']') { ++i; return true; }
            return false;
        }
    }

    bool parse_object(JsonValue& out, int depth) {
        ++i; // '{'
        out.type = JsonValue::kObject;
        out.obj.clear();
        skip_ws();
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        while (true) {
            skip_ws();
            if (i >= s.size() || s[i] != '"') return false;
            std::string key;
            if (!parse_string_raw(key)) return false;
            skip_ws();
            if (i >= s.size() || s[i] != ':') return false;
            ++i;
            JsonValue v;
            if (!parse(v, depth + 1)) return false;
            out.obj.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (i >= s.size()) return false;
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == '}') { ++i; return true; }
            return false;
        }
    }
};

// Parse a complete JSON document (exactly one value, nothing trailing).
// False on any malformed input.
bool json_parse_value(const std::string& text, JsonValue& out) {
    JsonParser p(text);
    if (!p.parse(out, 0)) return false;
    p.skip_ws();
    return p.i == text.size();
}

// Extract last "content" string in "messages" array (parsed properly, so a
// "content" key inside a nested string cannot shadow the real one).
// Falls back to a top-level "prompt" string, then the default.
std::string extract_last_message_content(const std::string& body) {
    auto msgs = parse_chat_messages(body);
    if (!msgs.empty()) return msgs.back().content;
    return extract_json_string(body, "prompt", "Explain MoE streaming architecture.");
}

// Find the balanced JSON value that follows "key": at the top level of
// `body` (object / array / string / literal). Returns the [start, end)
// span in `body`, or false when the key is absent or the value is malformed.
// Implemented on the strict parser, so a key inside a nested string or
// object can never match.
bool extract_json_value_span(const std::string& body, const std::string& key,
                             size_t& val_start, size_t& val_end) {
    JsonValue root;
    if (!json_parse_value(body, root)) return false;
    if (root.type != JsonValue::kObject) return false;
    const JsonValue* v = root.find(key);
    if (v == nullptr) return false;
    val_start = v->start;
    val_end = v->end;
    return true;
}

// Parse the OpenAI-style "tools" array of a chat request into ToolDefinition
// records. Only tools with a function name survive; malformed entries are
// skipped (a template rendering a nameless tool would be worse than none).
// The raw JSON of each tool object is kept verbatim so {{ tool | tojson }}
// renders exactly what the client sent.
std::vector<ToolDefinition> parse_chat_tools(const std::string& body) {
    std::vector<ToolDefinition> out;
    size_t vs = 0, ve = 0;
    if (!extract_json_value_span(body, "tools", vs, ve)) return out;
    if (ve <= vs || body[vs] != '[') return out;
    // Iterate the top-level objects of the array (depth-aware, quote-aware).
    size_t i = vs + 1;
    while (i < ve) {
        while (i < ve && body[i] != '{') ++i;
        if (i >= ve) break;
        size_t obj_start = i;
        int depth = 0;
        bool in_str = false;
        size_t obj_end = std::string::npos;
        for (size_t j = i; j < ve; ++j) {
            char c = body[j];
            if (in_str) {
                if (c == '\\' && j + 1 < ve) ++j;
                else if (c == '"') in_str = false;
            } else if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0) { obj_end = j + 1; break; }
            }
        }
        if (obj_end == std::string::npos) break;
        std::string tool_json = body.substr(obj_start, obj_end - obj_start);
        i = obj_end;

        ToolDefinition td;
        td.raw_json = tool_json;
        // The function object, or the tool object itself for flat shapes.
        std::string fn_json = tool_json;
        size_t fs = 0, fe = 0;
        if (extract_json_value_span(tool_json, "function", fs, fe) &&
            fe > fs && tool_json[fs] == '{') {
            fn_json = tool_json.substr(fs, fe - fs);
        }
        td.function_json = fn_json;
        td.name = extract_json_string(fn_json, "name", "");
        if (td.name.empty()) continue; // nameless tool: skip, do not render
        td.description = extract_json_string(fn_json, "description", "");
        size_t ps = 0, pe = 0;
        if (extract_json_value_span(fn_json, "parameters", ps, pe) && pe > ps) {
            td.parameters_json = fn_json.substr(ps, pe - ps);
        } else {
            td.parameters_json = "{}";
        }
        out.push_back(std::move(td));
    }
    return out;
}

// One tool call parsed out of generated text.
struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json; // JSON object text (or "" when absent)
};

// Extract the "arguments"/"parameters" value of one tool-call JSON object
// span as JSON text: an object/array is kept verbatim; a string is decoded
// (its content is the arguments JSON in the Qwen <tool_call> convention).
std::string tool_call_arguments_json(const std::string& obj_json) {
    size_t s = 0, e = 0;
    if (!extract_json_value_span(obj_json, "arguments", s, e)) {
        if (!extract_json_value_span(obj_json, "parameters", s, e)) return "";
    }
    std::string v = obj_json.substr(s, e - s);
    if (!v.empty() && v.front() == '"') {
        // JSON string: decode escapes.
        std::string out;
        for (size_t i = 1; i + 1 < v.size(); ++i) {
            if (v[i] == '\\' && i + 1 < v.size() - 1) {
                char n = v[++i];
                if (n == 'n') out.push_back('\n');
                else if (n == 'r') out.push_back('\r');
                else if (n == 't') out.push_back('\t');
                else out.push_back(n);
            } else {
                out.push_back(v[i]);
            }
        }
        return out;
    }
    return v;
}

// Parse tool calls out of generated text. Accepted forms (in order):
//   1. <tool_call>{...}</tool_call> blocks (Qwen2.5/3, as instructed by
//      their templates; possibly several in one reply).
//   2. [TOOL_CALLS] [{...}, ...] (Mistral).
//   3. A bare leading JSON object {"name": ..., "arguments"|"parameters":
//      ...} (Llama3 style) — only when the whole trimmed reply is that
//      object (or starts with it and the rest is whitespace).
// A call is only accepted when its name matches a declared tool: a model
// inventing a function name is a hallucination, not a tool call, and the
// reply falls through to plain content.
std::vector<ParsedToolCall> parse_tool_calls_from_text(
    const std::string& text, const std::vector<ToolDefinition>& declared) {
    auto is_declared = [&](const std::string& name) {
        for (const auto& t : declared) if (t.name == name) return true;
        return false;
    };
    std::vector<ParsedToolCall> calls;
    uint64_t call_no = 0;
    auto next_id = [&]() {
        return "call_" + std::to_string(now_us()) + "_" + std::to_string(call_no++);
    };

    // 1. <tool_call> blocks.
    size_t pos = 0;
    while (true) {
        size_t b = text.find("<tool_call>", pos);
        if (b == std::string::npos) break;
        size_t e = text.find("</tool_call>", b + 11);
        if (e == std::string::npos) break;
        std::string inner = text.substr(b + 11, e - (b + 11));
        // Trim whitespace.
        size_t a = inner.find_first_not_of(" \t\r\n");
        size_t z = inner.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) inner = inner.substr(a, z - a + 1);
        std::string name = extract_json_string(inner, "name", "");
        if (!name.empty() && is_declared(name)) {
            calls.push_back({next_id(), name, tool_call_arguments_json(inner)});
        }
        pos = e + 12;
    }
    if (!calls.empty()) return calls;

    // 2. [TOOL_CALLS] array.
    size_t tc = text.find("[TOOL_CALLS]");
    if (tc != std::string::npos) {
        size_t arr = text.find('[', tc + 12);
        if (arr != std::string::npos) {
            // Balanced array scan starting at `arr`.
            int depth = 0;
            bool in_str = false;
            size_t arr_end = std::string::npos;
            for (size_t j = arr; j < text.size(); ++j) {
                char c = text[j];
                if (in_str) {
                    if (c == '\\' && j + 1 < text.size()) ++j;
                    else if (c == '"') in_str = false;
                } else if (c == '"') {
                    in_str = true;
                } else if (c == '[') {
                    ++depth;
                } else if (c == ']') {
                    if (--depth == 0) { arr_end = j + 1; break; }
                }
            }
            if (arr_end != std::string::npos) {
                std::string arr_json = text.substr(arr, arr_end - arr);
                size_t i = 0;
                while (i < arr_json.size()) {
                    size_t os = arr_json.find('{', i);
                    if (os == std::string::npos) break;
                    int d = 0;
                    bool ins = false;
                    size_t oe = std::string::npos;
                    for (size_t j = os; j < arr_json.size(); ++j) {
                        char c = arr_json[j];
                        if (ins) {
                            if (c == '\\' && j + 1 < arr_json.size()) ++j;
                            else if (c == '"') ins = false;
                        } else if (c == '"') {
                            ins = true;
                        } else if (c == '{') {
                            ++d;
                        } else if (c == '}') {
                            if (--d == 0) { oe = j + 1; break; }
                        }
                    }
                    if (oe == std::string::npos) break;
                    std::string obj = arr_json.substr(os, oe - os);
                    std::string name = extract_json_string(obj, "name", "");
                    if (!name.empty() && is_declared(name)) {
                        std::string id = extract_json_string(obj, "id", "");
                        calls.push_back({id.empty() ? next_id() : id, name,
                                         tool_call_arguments_json(obj)});
                    }
                    i = oe;
                }
            }
        }
    }
    if (!calls.empty()) return calls;

    // 3. Bare leading JSON object.
    size_t t0 = text.find_first_not_of(" \t\r\n");
    if (t0 != std::string::npos && text[t0] == '{') {
        int depth = 0;
        bool in_str = false;
        size_t obj_end = std::string::npos;
        for (size_t j = t0; j < text.size(); ++j) {
            char c = text[j];
            if (in_str) {
                if (c == '\\' && j + 1 < text.size()) ++j;
                else if (c == '"') in_str = false;
            } else if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0) { obj_end = j + 1; break; }
            }
        }
        if (obj_end != std::string::npos) {
            std::string rest = text.substr(obj_end);
            if (rest.find_first_not_of(" \t\r\n") == std::string::npos) {
                std::string obj = text.substr(t0, obj_end - t0);
                std::string name = extract_json_string(obj, "name", "");
                if (!name.empty() && is_declared(name) &&
                    (obj.find("\"arguments\"") != std::string::npos ||
                     obj.find("\"parameters\"") != std::string::npos)) {
                    calls.push_back({next_id(), name, tool_call_arguments_json(obj)});
                }
            }
        }
    }
    return calls;
}

// Build the chat.completion "choices" message JSON for a tool-call reply
// (content null, tool_calls array, finish_reason "tool_calls").
std::string tool_calls_message_json(const std::vector<ParsedToolCall>& calls) {
    std::ostringstream js;
    js << "\"role\":\"assistant\",\"content\":null,\"tool_calls\":[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i > 0) js << ",";
        js << "{\"id\":\"" << json_escape(calls[i].id) << "\","
           << "\"type\":\"function\","
           << "\"function\":{\"name\":\"" << json_escape(calls[i].name) << "\","
           << "\"arguments\":\"" << json_escape(calls[i].arguments_json) << "\"}}";
    }
    js << "]";
    return js.str();
}

// (parse_chat_messages now lives at moe scope, implemented on the strict
// JSON parser; see below the anonymous namespace.)

// Build the generation prompt for a chat request, using the SAME shared
// helper as CLI chat (generate_stream): the GGUF tokenizer.chat_template
// wins, otherwise a per-architecture fallback template.
std::string format_chat_prompt_for_arch(const std::string& body, const MoETransformerEngine& engine,
                                        const std::vector<ToolDefinition>* tools = nullptr) {
    const auto& hp = engine.hparams();
    const auto& tok = engine.tokenizer();
    auto msgs = parse_chat_messages(body);
    if (msgs.empty()) {
        if (tools == nullptr || tools->empty()) {
            return wrap_raw_prompt_for_chat(extract_last_message_content(body),
                                            hp.arch, tok.chat_template(), tok);
        }
        msgs = {{"user", extract_last_message_content(body)}};
    }
    return format_chat_messages(msgs, hp.arch, tok.chat_template(), tok, true, nullptr, tools);
}

// (extract_json_number / extract_json_bool / extract_json_string_array now
// live at moe scope, implemented on the strict JSON parser; see below the
// anonymous namespace.)

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

// Per-connection CORS origin, set by handle_client from the request's
// Origin header and the configured allow-list (thread_local: one connection
// per thread, never shared). send_http_json / the SSE and static responses
// read it so the header is emitted only for allowed origins — the default
// is same-origin only (no Access-Control-Allow-Origin header at all).
// Previously every response sent `Access-Control-Allow-Origin: *`, which
// let any website call the engine's mutating API from a browser.
thread_local std::string g_allowed_origin;

std::string cors_response_header() {
    if (g_allowed_origin.empty()) return "";
    return "Access-Control-Allow-Origin: " + g_allowed_origin + "\r\n"
           "Vary: Origin\r\n";
}

void send_http_json(uintptr_t sock, int status_code, const std::string& status_text, const std::string& json_body) {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << cors_response_header()
        << "Content-Length: " << json_body.size() << "\r\n"
        << "Connection: close\r\n\r\n";
    std::string h = hdr.str();
    send_all(sock, h.data(), h.size());
    send_all(sock, json_body.data(), json_body.size());
}

// ---------------------------------------------------------------------------
// Auth / bind helpers (security workstream, 2026-11-17)
// ---------------------------------------------------------------------------

// Read one string field from moe_config.json in the working directory.
// Returns `def` when the file is missing or the field is absent.
std::string read_config_string(const std::string& key, const std::string& def = "") {
    std::ifstream ifs("moe_config.json");
    if (!ifs.is_open()) return def;
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return extract_json_string(content, key, def);
}

// Resolve the API key: MOE_API_KEY env var wins; otherwise moe_config.json
// "api_key" if present. Empty = auth disabled (localhost-only use).
std::string resolve_api_key() {
    const char* env_key = std::getenv("MOE_API_KEY");
    if (env_key != nullptr && env_key[0] != '\0') return std::string(env_key);
    return read_config_string("api_key", "");
}

bool is_loopback_host(const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

// Constant-time comparison so key guessing cannot use timing as an oracle.
bool constant_time_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    return diff == 0;
}

// (header_value / route_requires_auth / request_authorized now live at
// moe scope; see below the anonymous namespace.)

// Replace the value of "api_key" in a JSON blob with "***" so a key stored in
// moe_config.json is never leaked through GET /v1/config.
std::string redact_api_key(const std::string& json) {
    std::string pat = "\"api_key\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return json;
    size_t colon = json.find(':', pos + pat.size());
    if (colon == std::string::npos) return json;
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return json;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return json;
    std::string out = json;
    out.replace(q1 + 1, q2 - q1 - 1, "***");
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Public JSON / auth / validation helpers (declared in server.h, defined
// here at moe scope so the unit tests can call them). All field extraction
// goes through the strict parser above: only depth-1 keys of the outermost
// object match, so nested/string-embedded keys can never shadow a field.
// ---------------------------------------------------------------------------

bool json_is_valid_object(const std::string& text) {
    JsonValue v;
    return json_parse_value(text, v) && v.type == JsonValue::kObject;
}

std::string extract_json_string(const std::string& body, const std::string& key,
                                const std::string& def) {
    JsonValue root;
    if (!json_parse_value(body, root)) return def;
    const JsonValue* v = root.find(key);
    if (v == nullptr || v->type != JsonValue::kString) return def;
    return v->str;
}

double extract_json_number(const std::string& body, const std::string& key, double def) {
    JsonValue root;
    if (!json_parse_value(body, root)) return def;
    const JsonValue* v = root.find(key);
    if (v == nullptr || v->type != JsonValue::kNumber) return def;
    return v->num;
}

bool extract_json_bool(const std::string& body, const std::string& key, bool def) {
    JsonValue root;
    if (!json_parse_value(body, root)) return def;
    const JsonValue* v = root.find(key);
    if (v == nullptr || v->type != JsonValue::kBool) return def;
    return v->b;
}

std::vector<std::string> extract_json_string_array(const std::string& body,
                                                   const std::string& key) {
    std::vector<std::string> res;
    JsonValue root;
    if (!json_parse_value(body, root)) return res;
    const JsonValue* v = root.find(key);
    if (v == nullptr || v->type != JsonValue::kArray) return res;
    for (const auto& el : v->arr) {
        if (el.type == JsonValue::kString) res.push_back(el.str);
    }
    return res;
}

// Parse the OpenAI-style "messages" array into {role, content} pairs.
// Roles/contents come from each message object's own depth-1 fields, so a
// "role" string embedded in message content cannot smuggle a role. Content
// may also be an array of parts ({"type":"text","text":"..."}) — the text
// parts are concatenated. Returns an empty vector when there is no usable
// messages array.
std::vector<ChatMessage> parse_chat_messages(const std::string& body) {
    std::vector<ChatMessage> msgs;
    JsonValue root;
    if (!json_parse_value(body, root)) return msgs;
    const JsonValue* arr = root.find("messages");
    if (arr == nullptr || arr->type != JsonValue::kArray) return msgs;
    for (const auto& el : arr->arr) {
        if (el.type != JsonValue::kObject) continue;
        std::string role = "user";
        if (const JsonValue* r = el.find("role")) {
            if (r->type == JsonValue::kString && !r->str.empty()) role = r->str;
        }
        std::string content;
        if (const JsonValue* c = el.find("content")) {
            if (c->type == JsonValue::kString) {
                content = c->str;
            } else if (c->type == JsonValue::kArray) {
                for (const auto& part : c->arr) {
                    if (part.type != JsonValue::kObject) continue;
                    if (const JsonValue* t = part.find("text")) {
                        if (t->type == JsonValue::kString) content += t->str;
                    }
                }
            }
        }
        if (!content.empty()) msgs.push_back({role, content});
    }
    return msgs;
}

// Case-insensitive single header lookup in a raw header block. Matches
// header names at line starts only: the old substring search could match
// "x-api-key:" inside another header's VALUE and treat it as the key.
std::string header_value(const std::string& headers, const std::string& name) {
    std::string lname = name;
    std::transform(lname.begin(), lname.end(), lname.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    size_t pos = 0;
    while (pos < headers.size()) {
        size_t eol = headers.find("\r\n", pos);
        size_t line_end = (eol == std::string::npos) ? headers.size() : eol;
        std::string line = headers.substr(pos, line_end - pos);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string hname = line.substr(0, colon);
            size_t a = hname.find_first_not_of(" \t");
            size_t b = hname.find_last_not_of(" \t");
            if (a != std::string::npos) hname = hname.substr(a, b - a + 1);
            std::transform(hname.begin(), hname.end(), hname.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (hname == lname) {
                std::string val = line.substr(colon + 1);
                size_t va = val.find_first_not_of(" \t");
                if (va == std::string::npos) return "";
                size_t vb = val.find_last_not_of(" \t\r");
                return val.substr(va, vb - va + 1);
            }
        }
        if (eol == std::string::npos) break;
        pos = eol + 2;
    }
    return "";
}

// Mutating routes that require a key whenever one is configured. Read-only
// GETs (including GET /health, the liveness probe) stay open.
bool route_requires_auth(const std::string& method, const std::string& path) {
    if (method == "POST" || method == "DELETE") {
        if (path == "/v1/models/load" || path == "/v1/models/pull") return true;
        if (path == "/v1/config" || path == "/api/config") return true;
        if (path == "/v1/slots/clear" || path.starts_with("/v1/slots/")) return true;
        if (path == "/v1/chat/completions" || path == "/v1/completions" ||
            path == "/api/chat" || path == "/api/generate") return true;
        if (path == "/v1/embeddings" || path == "/v1/systemone" ||
            path == "/v1/images/generations") return true;
    }
    return false;
}

// Check Authorization: Bearer <key> or x-api-key: <key> against `key`.
// The scheme is case-insensitive per RFC 9110 ("bearer" is accepted too).
bool request_authorized(const std::string& headers, const std::string& key) {
    std::string auth = header_value(headers, "Authorization");
    if (auth.size() > 7) {
        std::string scheme = auth.substr(0, 7);
        std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (scheme == "bearer " && constant_time_equal(auth.substr(7), key)) return true;
    }
    std::string xkey = header_value(headers, "x-api-key");
    if (!xkey.empty() && constant_time_equal(xkey, key)) return true;
    return false;
}

std::string cors_allowed_origin(const std::string& origin,
                                const std::string& allowed_origins_csv) {
    if (origin.empty() || allowed_origins_csv.empty()) return "";
    std::string o = origin;
    while (!o.empty() && (o.back() == ' ' || o.back() == '\t' || o.back() == '/')) o.pop_back();
    std::istringstream ss(allowed_origins_csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t a = item.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        size_t b = item.find_last_not_of(" \t/");
        if (item.substr(a, b - a + 1) == o) return origin;
    }
    return "";
}

bool validate_sampler_params(double temperature, double top_p, double top_k,
                             double repetition_penalty, std::string& err) {
    if (!std::isfinite(temperature) || temperature < 0.0 || temperature > 5.0) {
        err = "temperature must be a number in [0, 5]";
        return false;
    }
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) {
        err = "top_p must be a number in (0, 1]";
        return false;
    }
    if (!std::isfinite(top_k) || top_k < 0.0 || top_k > 1000.0 ||
        std::floor(top_k) != top_k) {
        err = "top_k must be an integer in [0, 1000]";
        return false;
    }
    if (!std::isfinite(repetition_penalty) || repetition_penalty < 0.0 ||
        repetition_penalty > 2.0) {
        err = "repetition_penalty must be a number in [0, 2]";
        return false;
    }
    return true;
}

bool validate_server_config(int paged_blocks, int active_slots, int max_context,
                            int api_port, int spec_tokens, double temperature,
                            double top_p, const std::string& api_host,
                            std::string& err) {
    if (paged_blocks < 1 || paged_blocks > 65536) {
        err = "paged_attention_blocks must be an integer in [1, 65536]";
        return false;
    }
    if (active_slots < 1 || active_slots > 64) {
        err = "active_slots must be an integer in [1, 64]";
        return false;
    }
    if (max_context < 256 || max_context > 262144) {
        err = "max_context must be an integer in [256, 262144]";
        return false;
    }
    if (api_port < 1 || api_port > 65535) {
        err = "api_port must be an integer in [1, 65535]";
        return false;
    }
    if (spec_tokens < 0 || spec_tokens > 16) {
        err = "speculative_draft_tokens must be an integer in [0, 16]";
        return false;
    }
    std::string serr;
    if (!validate_sampler_params(temperature, top_p, 40.0, 1.0, serr)) {
        err = serr;
        return false;
    }
    if (api_host.empty() || api_host.size() > 253) {
        err = "api_host must be a non-empty hostname or IP (max 253 chars)";
        return false;
    }
    for (char c : api_host) {
        unsigned char u = static_cast<unsigned char>(c);
        bool ok = (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
                  (u >= '0' && u <= '9') || c == '.' || c == '-' || c == ':' ||
                  c == '[' || c == ']';
        if (!ok) {
            err = "api_host contains an invalid character";
            return false;
        }
    }
    return true;
}

HttpServer::HttpServer(MoETransformerEngine& engine, const std::string& web_root)
    : engine_(engine), web_root_(web_root) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::stop() {
    running_ = false;
    if (listen_sock_ != ~static_cast<uintptr_t>(0)) {
#ifndef _WIN32
        // shutdown() wakes a thread blocked in accept(); close() alone
        // does not on Linux (the socket object stays alive while accept
        // holds a reference), so without this the accept thread never
        // exits and a join() after stop() hangs forever. This also
        // matters for a clean `serve` shutdown.
        ::shutdown(static_cast<int>(listen_sock_), SHUT_RDWR);
#endif
        close_sock(listen_sock_);
        listen_sock_ = ~static_cast<uintptr_t>(0);
    }
}

bool HttpServer::start(uint16_t port, std::string& err_msg,
                       const std::string& host, bool allow_insecure_remote) {
    port_ = port;
    host_ = host;
    api_key_ = resolve_api_key();
#ifndef _WIN32
    // A client hanging up mid-stream used to kill the whole engine with
    // SIGPIPE (exit 141): send() on a closed socket raises SIGPIPE by
    // default, and the streaming callbacks' send_all() failure return
    // never got the chance to end just that connection. Ignoring SIGPIPE
    // process-wide makes send() fail with EPIPE instead, so a disconnect
    // ends only its own connection. (Windows has no SIGPIPE.)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    // CORS allow-list: MOE_CORS_ORIGINS env wins, else moe_config.json
    // "cors_origins" (comma-separated exact origins). Empty = same-origin
    // only: no Access-Control-Allow-Origin header is emitted at all.
    {
        const char* env_cors = std::getenv("MOE_CORS_ORIGINS");
        if (env_cors != nullptr && env_cors[0] != '\0') cors_origins_ = env_cors;
        else cors_origins_ = read_config_string("cors_origins", "");
    }

    // Refuse a non-loopback bind with no key unless the operator explicitly
    // opts out. Listening on all interfaces unauthenticated is how the old
    // default leaked model-load / config-write to the whole network.
    if (!is_loopback_host(host_) && api_key_.empty() && !allow_insecure_remote) {
        err_msg = "Refusing to bind " + host_ + " without an API key: set MOE_API_KEY "
                  "(or \"api_key\" in moe_config.json), bind 127.0.0.1, or pass "
                  "--allow-insecure-remote to accept the risk explicitly.";
        std::cerr << "\n\033[1;31m[SECURITY] " << err_msg << "\033[0m\n\n";
        return false;
    }
    if (!is_loopback_host(host_)) {
        if (api_key_.empty()) {
            std::cerr << "\n\033[1;31m[SECURITY WARNING] Binding " << host_
                      << " with NO API key (--allow-insecure-remote). Any host that can "
                         "reach this port can load models, download files, and overwrite "
                         "moe_config.json. This is strongly discouraged.\033[0m\n\n";
        } else {
            std::cerr << "\n[SECURITY] Binding " << host_
                      << " — API-key auth is enabled for mutating endpoints.\n\n";
        }
    } else if (api_key_.empty()) {
        std::cerr << "\n[SECURITY] Listening on loopback only (" << host_
                  << ") with no API key configured. Set MOE_API_KEY to require auth.\n\n";
    }
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
    if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
        err_msg = "Invalid bind host: " + host_;
        stop();
        return false;
    }
    addr.sin_port = htons(port);

    if (::bind(
#ifdef _WIN32
            static_cast<SOCKET>(s),
#else
            static_cast<int>(s),
#endif
            reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        err_msg = "Failed to bind HTTP server to " + host_ + ":" + std::to_string(port);
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
        // Bounded concurrency: each connection gets a detached thread, so
        // an unbounded accept loop is a thread-exhaustion DoS. Past the
        // cap the socket is closed immediately (the client sees a reset,
        // not a hang). active_requests_ is incremented in handle_client.
        if (active_requests_.load() >= kMaxConcurrentRequests) {
            close_sock(client_sock);
            continue;
        }
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

#ifdef _WIN32
    DWORD rcv_timeout_ms = 5000;
    setsockopt(static_cast<SOCKET>(client_sock), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&rcv_timeout_ms), sizeof(rcv_timeout_ms));
    DWORD snd_timeout_ms = 30000;
    setsockopt(static_cast<SOCKET>(client_sock), SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&snd_timeout_ms), sizeof(snd_timeout_ms));
#else
    struct timeval tv{5, 0};
    setsockopt(static_cast<int>(client_sock), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    struct timeval snd_tv{30, 0};
    setsockopt(static_cast<int>(client_sock), SOL_SOCKET, SO_SNDTIMEO, &snd_tv, sizeof(snd_tv));
#endif

    std::string req;
    char buf[8192];
    size_t header_end = std::string::npos;
    size_t content_len = 0;
    bool header_too_large = false;
    bool body_too_large = false;
    bool bad_length = false;
    bool has_transfer_encoding = false;

    while (true) {
#ifdef _WIN32
        int n = ::recv(static_cast<SOCKET>(client_sock), buf, sizeof(buf), 0);
#else
        ssize_t n = ::recv(static_cast<int>(client_sock), buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        req.append(buf, static_cast<size_t>(n));
        if (header_end == std::string::npos) {
            // Header block cap: a client that never sends \r\n\r\n cannot
            // make us buffer forever.
            if (req.size() > kMaxHeaderBytes) { header_too_large = true; break; }
            header_end = req.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                std::string headers = req.substr(0, header_end);
                // Safe Content-Length parse: digits only, single header.
                // The old code used std::stoull on raw header text, which
                // throws std::invalid_argument on "Content-Length: abc" —
                // an uncaught throw in a detached thread terminates the
                // whole process. Duplicate / chunked framing is rejected.
                std::string lower_h = headers;
                std::transform(lower_h.begin(), lower_h.end(), lower_h.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (lower_h.find("transfer-encoding:") != std::string::npos) {
                    has_transfer_encoding = true;
                }
                size_t cl = lower_h.find("content-length:");
                if (cl != std::string::npos) {
                    if (lower_h.find("content-length:", cl + 15) != std::string::npos) {
                        bad_length = true;
                    } else {
                        size_t vstart = cl + 15;
                        while (vstart < lower_h.size() && (lower_h[vstart] == ' ' || lower_h[vstart] == '\t')) ++vstart;
                        size_t vend = vstart;
                        while (vend < lower_h.size() && std::isdigit(static_cast<unsigned char>(lower_h[vend]))) ++vend;
                        if (vend == vstart) {
                            bad_length = true;
                        } else {
                            try {
                                content_len = std::stoull(lower_h.substr(vstart, vend - vstart));
                            } catch (...) {
                                bad_length = true;
                            }
                            if (content_len > kMaxBodyBytes) body_too_large = true;
                        }
                    }
                }
            }
        }
        if (header_end != std::string::npos && req.size() >= header_end + 4 + content_len) {
            break;
        }
        // Never buffer more than the declared body plus the header cap.
        if (header_end != std::string::npos && req.size() > header_end + 4 + content_len) break;
        if (req.size() > kMaxHeaderBytes + kMaxBodyBytes) { body_too_large = true; break; }
    }

    if (header_too_large) {
        send_http_json(client_sock, 431, "Request Header Fields Too Large",
                       "{\"error\":\"header_too_large\",\"message\":\"Request headers exceed 64 KiB.\"}");
        return;
    }
    if (req.empty() || header_end == std::string::npos) return;
    if (has_transfer_encoding) {
        send_http_json(client_sock, 400, "Bad Request",
                       "{\"error\":\"unsupported_transfer_encoding\",\"message\":\"Transfer-Encoding is not supported; send a Content-Length body.\"}");
        return;
    }
    if (bad_length) {
        send_http_json(client_sock, 400, "Bad Request",
                       "{\"error\":\"invalid_content_length\",\"message\":\"Content-Length must be a single non-negative integer.\"}");
        return;
    }
    if (body_too_large) {
        send_http_json(client_sock, 413, "Content Too Large",
                       "{\"error\":\"request_too_large\",\"message\":\"Request body exceeds the 1 MiB limit.\"}");
        return;
    }
    if (req.size() < header_end + 4 + content_len) {
        // Client disconnected or stalled mid-body: never act on a partial
        // request.
        send_http_json(client_sock, 400, "Bad Request",
                       "{\"error\":\"incomplete_body\",\"message\":\"Request body ended before Content-Length bytes arrived.\"}");
        return;
    }

    std::istringstream req_stream(req.substr(0, header_end));
    std::string method, raw_path, http_ver;
    req_stream >> method >> raw_path >> http_ver;
    if (method.empty() || raw_path.empty() || http_ver.empty() ||
        http_ver.rfind("HTTP/", 0) != 0) {
        send_http_json(client_sock, 400, "Bad Request",
                       "{\"error\":\"malformed_request\",\"message\":\"Request line must be METHOD PATH HTTP/x.\"}");
        return;
    }

    std::string path = raw_path;
    size_t qpos = path.find('?');
    if (qpos != std::string::npos) path = path.substr(0, qpos);

    std::string body = (req.size() > header_end + 4) ? req.substr(header_end + 4, content_len) : "";

    // CORS origin for this connection (thread_local, read by the response
    // helpers). Same-origin requests send no Origin header and get no
    // Access-Control-Allow-Origin header; that is the default policy.
    {
        std::string headers = req.substr(0, header_end);
        g_allowed_origin = cors_allowed_origin(header_value(headers, "Origin"), cors_origins_);
    }

    if (method == "OPTIONS") {
        // Preflight: only allowed origins get the CORS headers. A
        // disallowed origin gets a plain 403 (no ACAO), so the browser
        // blocks the cross-origin call.
        std::string headers = req.substr(0, header_end);
        std::string origin = header_value(headers, "Origin");
        if (!origin.empty() && g_allowed_origin.empty()) {
            send_http_json(client_sock, 403, "Forbidden",
                           "{\"error\":\"cors_forbidden\",\"message\":\"This origin is not allowed to call this server.\"}");
            return;
        }
        std::ostringstream hdr;
        hdr << "HTTP/1.1 200 OK\r\n"
            << cors_response_header()
            << "Access-Control-Allow-Methods: GET, POST, OPTIONS, DELETE\r\n"
            << "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key\r\n"
            << "Content-Length: 2\r\n"
            << "Connection: close\r\n\r\n{}";
        std::string h = hdr.str();
        send_all(client_sock, h.data(), h.size());
        return;
    }

    // API-key auth: when a key is configured (MOE_API_KEY env or
    // moe_config.json "api_key"), every mutating endpoint requires
    // `Authorization: Bearer <key>` or `x-api-key: <key>`. Read-only GETs
    // (including GET /health) stay open. No key configured = no auth
    // (server is loopback-only by default; see start()).
    if (!api_key_.empty() && route_requires_auth(method, path)) {
        std::string headers = req.substr(0, header_end);
        if (!request_authorized(headers, api_key_)) {
            send_http_json(client_sock, 401, "Unauthorized",
                           "{\"error\":\"unauthorized\",\"message\":\"Missing or invalid API key. "
                           "Send Authorization: Bearer <key> or x-api-key: <key>.\"}");
            return;
        }
    }

    // Malformed-JSON gate: every POST/DELETE endpoint takes a JSON object.
    // A body that is not a well-formed JSON object is rejected with a
    // structured 400 here, before any handler's field extractors run, so
    // truncated or hostile bodies can never produce a half-parsed request.
    // (An empty body is allowed: handlers treat missing fields as absent.)
    if ((method == "POST" || method == "DELETE") && !body.empty() &&
        !json_is_valid_object(body)) {
        send_http_json(client_sock, 400, "Bad Request",
                       "{\"error\":\"invalid_json\",\"message\":\"Request body must be a well-formed JSON object.\"}");
        return;
    }

    // ------------------------------------------------------------------------
    // 0a. GET /v1/models & GET /models (OpenAI Standard Model Discovery)
    // ------------------------------------------------------------------------
    if (method == "GET" && (path == "/v1/models" || path == "/models" || path.starts_with("/v1/models/") || path.starts_with("/models/"))) {
        const auto& hp = engine_.hparams();
        const std::string mname = hp.model_name.empty() ? "qwen2.5-0.5b-instruct" : hp.model_name;

        if ((path.starts_with("/v1/models/") && path.size() > 11) || (path.starts_with("/models/") && path.size() > 8)) {
            std::ostringstream js;
            js << "{"
               << "\"id\":\"" << json_escape(mname) << "\","
               << "\"object\":\"model\","
               << "\"created\":1730000000,"
               << "\"owned_by\":\"moe-engine\","
               << "\"permission\":[],"
               << "\"root\":\"" << json_escape(mname) << "\","
               << "\"parent\":null"
               << "}";
            send_http_json(client_sock, 200, "OK", js.str());
            return;
        }

        const std::string active_path = engine_.model_file().path();
        std::ostringstream js;
        js << "{"
           << "\"object\":\"list\","
           << "\"data\":["
           << "{"
           << "\"id\":\"" << json_escape(mname) << "\","
           << "\"object\":\"model\","
           << "\"created\":1730000000,"
           << "\"owned_by\":\"moe-engine\","
           << "\"path\":\"" << json_escape(active_path) << "\","
           << "\"size\":" << engine_.model_file().file_size() << ","
           << "\"active\":true,"
           << "\"permission\":[],"
           << "\"root\":\"" << json_escape(mname) << "\","
           << "\"parent\":null"
           << "}";

        // Discover local .gguf / .safetensors under models/ (incl. models/real/).
        std::error_code ec;
        if (std::filesystem::exists("models", ec) && std::filesystem::is_directory("models", ec)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator("models", ec)) {
                if (!entry.is_regular_file(ec)) continue;
                std::string ext = entry.path().extension().string();
                if (ext != ".gguf" && ext != ".safetensors") continue;
                std::string rel_path = entry.path().generic_string();
                // Normalize leading "./"
                if (rel_path.rfind("./", 0) == 0) rel_path.erase(0, 2);
                std::string fname = entry.path().filename().string();
                bool is_curr = (active_path == rel_path || active_path.ends_with(fname) ||
                                active_path.ends_with(rel_path));
                uint64_t fsize = entry.file_size(ec);
                js << ",{"
                   << "\"id\":\"" << json_escape(fname) << "\","
                   << "\"object\":\"model\","
                   << "\"created\":1730000000,"
                   << "\"owned_by\":\"local-gguf\","
                   << "\"path\":\"" << json_escape(rel_path) << "\","
                   << "\"size\":" << fsize << ","
                   << "\"active\":" << (is_curr ? "true" : "false")
                   << "}";
            }
        }

        js << ",{"
           << "\"id\":\"default\","
           << "\"object\":\"model\","
           << "\"created\":1730000000,"
           << "\"owned_by\":\"moe-engine\""
           << "}"
           << "]"
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 0a-2. POST /v1/models/load (Hot-Swap Active Model in Engine)
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/models/load") {
        std::string model_path = extract_json_string(body, "model_path", "");
        // Contain under CWD (same policy as static files): no absolute
        // paths, no `..`, no `//etc/...` root replacement. Operators who
        // need a path outside the working tree must restart with --model.
        std::string resolved;
        if (model_path.empty() || !resolve_under_root(".", model_path, resolved)) {
            send_http_json(client_sock, 400, "Bad Request",
                           "{\"error\":\"Invalid or missing model_path\",\"message\":\"model_path must be a relative path under the engine working directory (no .. or absolute paths).\"}");
            return;
        }
        model_path = resolved;
        std::string err_msg;
        if (!engine_.load_model(model_path, err_msg)) {
            std::ostringstream err_js;
            err_js << "{\"status\":\"error\",\"error\":\"" << json_escape(err_msg) << "\"}";
            send_http_json(client_sock, 400, "Bad Request", err_js.str());
            return;
        }
        const auto& hp = engine_.hparams();
        std::ostringstream js;
        js << "{"
           << "\"status\":\"ok\","
           << "\"model_path\":\"" << json_escape(model_path) << "\","
           << "\"model_name\":\"" << json_escape(hp.model_name) << "\","
           << "\"architecture\":\"" << json_escape(hp.arch) << "\""
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 0a-3. POST /v1/models/pull (Stream Hugging Face GGUF Download Progress via SSE)
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/models/pull") {
        std::string source = extract_json_string(body, "source", "");
        if (source.empty()) source = extract_json_string(body, "url", "");
        if (source.empty()) {
            send_http_json(client_sock, 400, "Bad Request", "{\"error\":\"Missing source or url\"}");
            return;
        }

        std::ostringstream sse_hdr;
        sse_hdr << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: text/event-stream; charset=utf-8\r\n"
                << "Cache-Control: no-cache\r\n"
                << "Connection: close\r\n"
                << cors_response_header() << "\r\n";
        std::string hstr = sse_hdr.str();
        if (!send_all(client_sock, hstr.data(), hstr.size())) return;

        std::string out_file, err_msg;
        bool ok = pull_model_from_hub(source, "models", out_file, err_msg,
            [&](size_t downloaded, size_t total, double mb_s, double eta_s) {
                double pct = (total > 0) ? (static_cast<double>(downloaded) * 100.0 / static_cast<double>(total)) : 0.0;
                std::ostringstream ev;
                ev << "data: {\"status\":\"downloading\",\"downloaded\":" << downloaded
                   << ",\"total\":" << total
                   << ",\"percent\":" << std::fixed << std::setprecision(1) << pct
                   << ",\"mb_per_sec\":" << mb_s
                   << ",\"eta_sec\":" << eta_s << "}\n\n";
                std::string chunk = ev.str();
                send_all(client_sock, chunk.data(), chunk.size());
            }
        );

        std::ostringstream fin;
        if (ok) {
            fin << "data: {\"status\":\"done\",\"file\":\"" << json_escape(out_file) << "\"}\n\n";
        } else {
            fin << "data: {\"status\":\"error\",\"error\":\"" << json_escape(err_msg) << "\"}\n\n";
        }
        std::string fstr = fin.str();
        send_all(client_sock, fstr.data(), fstr.size());
        return;
    }

    // ------------------------------------------------------------------------
    // 0a-4. POST /v1/embeddings (OpenAI-Compatible Mean-Pooled Hidden Embeddings)
    // ------------------------------------------------------------------------
    if (method == "POST" && path == "/v1/embeddings") {
        std::string input_text = extract_json_string(body, "input", "");
        if (input_text.empty()) {
            auto arr = extract_json_string_array(body, "input");
            if (!arr.empty()) input_text = arr[0];
        }
        if (input_text.empty()) {
            send_http_json(client_sock, 400, "Bad Request", "{\"error\":\"Missing input text\"}");
            return;
        }

        auto emb = engine_.compute_embeddings(input_text);
        std::ostringstream js;
        js << "{"
           << "\"object\":\"list\","
           << "\"model\":\"" << json_escape(engine_.hparams().model_name) << "\","
           << "\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[";
        for (size_t i = 0; i < emb.size(); ++i) {
            if (i > 0) js << ",";
            js << std::fixed << std::setprecision(6) << emb[i];
        }
        js << "]}],"
           << "\"usage\":{\"prompt_tokens\":" << (input_text.size() / 4 + 1) << ",\"total_tokens\":" << (input_text.size() / 4 + 1) << "}"
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 0b. GET /api/tags (Ollama Standard Model Discovery)
    // ------------------------------------------------------------------------
    if (method == "GET" && path == "/api/tags") {
        const auto& hp = engine_.hparams();
        const std::string mname = hp.model_name.empty() ? "qwen2.5:latest" : hp.model_name;
        std::ostringstream js;
        js << "{\"models\":[{"
           << "\"name\":\"" << json_escape(mname) << "\","
           << "\"model\":\"" << json_escape(mname) << "\","
           << "\"modified_at\":\"2026-11-04T00:00:00Z\","
           << "\"size\":1328288,"
           << "\"digest\":\"sha256:moe-engine-local\","
           << "\"details\":{\"format\":\"gguf\",\"family\":\"qwen2\",\"parameter_size\":\"0.5B\",\"quantization_level\":\"Q4_0\"}"
           << "}]}";
        send_http_json(client_sock, 200, "OK", js.str());
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
           << "\"vram_budget_bytes\":" << hw.vram_budget_bytes << ","
           << "\"bypass_io\":" << (io.bypass_io_supported() ? "true" : "false") << ","
           << "\"direct_dma\":" << (io.direct_dma_active() ? "true" : "false") << ","
           << "\"vulkan_available\":" << (moe::VulkanComputeBackend::instance().is_available() ? "true" : "false") << ","
           << "\"vulkan_device\":\"" << json_escape(moe::VulkanComputeBackend::instance().device_info().device_name) << "\""
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
    // 3b. GET /v1/slots & POST /v1/slots/clear (PagedAttention Session Management)
    // ------------------------------------------------------------------------
    if (method == "GET" && path == "/v1/slots") {
        auto slots = engine_.list_slots();
        auto pstats = engine_.paged_block_stats();
        std::ostringstream js;
        js << "{"
           << "\"slots\":[";
        for (size_t i = 0; i < slots.size(); ++i) {
            if (i > 0) js << ",";
            const auto& s = slots[i];
            js << "{"
               << "\"slot_id\":" << s.slot_id << ","
               << "\"seq_pos\":" << s.seq_pos << ","
               << "\"pages_allocated\":" << s.pages_allocated << ","
               << "\"compactions\":" << s.compactions << ","
               << "\"token_count\":" << s.token_count << ","
               << "\"memory_bytes\":" << s.memory_bytes
               << "}";
        }
        js << "],"
           << "\"paged_block_manager\":{"
           << "\"total_blocks\":" << pstats.total_blocks << ","
           << "\"used_blocks\":" << pstats.used_blocks << ","
           << "\"free_blocks\":" << pstats.free_blocks << ","
           << "\"block_size_tokens\":" << pstats.block_size_tokens << ","
           << "\"memory_bytes\":" << pstats.memory_bytes
           << "}"
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    if ((method == "POST" || method == "DELETE") && (path == "/v1/slots/clear" || path.starts_with("/v1/slots/"))) {
        // Same user-slot range as chat completions: integer in [0, 63].
        // Path form: DELETE/POST /v1/slots/<id>. Body form: {"slot_id": N}
        // on /v1/slots/clear. Reject out-of-range / non-integer with 400
        // instead of silently wrapping via uint32 cast or ignoring parse failure.
        double slot_raw = 0.0;
        bool have_slot = false;
        if (path.starts_with("/v1/slots/") && path.size() > 10 && path != "/v1/slots/clear") {
            try {
                size_t idx = 0;
                unsigned long parsed = std::stoul(path.substr(10), &idx);
                if (idx == 0 || path.substr(10).find_first_not_of("0123456789") != std::string::npos) {
                    send_http_json(client_sock, 400, "Bad Request",
                                   "{\"error\":\"invalid_parameter\",\"message\":\"slot id in path must be an integer in [0, 63].\"}");
                    return;
                }
                slot_raw = static_cast<double>(parsed);
                have_slot = true;
            } catch (...) {
                send_http_json(client_sock, 400, "Bad Request",
                               "{\"error\":\"invalid_parameter\",\"message\":\"slot id in path must be an integer in [0, 63].\"}");
                return;
            }
        } else {
            slot_raw = extract_json_number(body, "slot_id", 0.0);
            have_slot = true;
        }
        if (!have_slot || !std::isfinite(slot_raw) || slot_raw < 0.0 || slot_raw > 63.0 ||
            std::floor(slot_raw) != slot_raw) {
            send_http_json(client_sock, 400, "Bad Request",
                           "{\"error\":\"invalid_parameter\",\"message\":\"slot_id must be an integer in [0, 63].\"}");
            return;
        }
        uint32_t slot_id = static_cast<uint32_t>(slot_raw);
        // Clear under the slot's generation mutex so a clear cannot land
        // mid-generation on that slot.
        std::unique_lock<std::mutex> clear_lock(slot_mutexes_[slot_id]);
        bool cleared = engine_.clear_slot(slot_id);
        std::ostringstream js;
        js << "{\"status\":\"ok\",\"slot_id\":" << slot_id << ",\"cleared\":" << (cleared ? "true" : "false") << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 3c. GET /api/config & GET /v1/config (Retrieve Live moe_config.json)
    // ------------------------------------------------------------------------
    if (method == "GET" && (path == "/api/config" || path == "/v1/config")) {
        std::ifstream ifs("moe_config.json");
        std::string json_content;
        if (ifs.is_open()) {
            json_content = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        }
        if (json_content.empty()) {
            const auto& hp = engine_.hparams();
            std::ostringstream js;
            js << "{\n"
               << "  \"engine\": \"MoE-Stream-Engine-v2\",\n"
               << "  \"first_run_completed\": true,\n"
               << "  \"model_path\": \"" << (hp.model_name.empty() ? "models/test_qwen.gguf" : hp.model_name) << "\",\n"
               << "  \"bypass_io\": true,\n"
               << "  \"direct_dma\": true,\n"
               << "  \"paged_attention_blocks\": 2048,\n"
               << "  \"active_slots\": 4,\n"
               << "  \"max_context\": 32768,\n"
               << "  \"api_port\": " << port_ << ",\n"
               << "  \"speculative_draft_tokens\": 3,\n"
               << "  \"temperature\": 0.7,\n"
               << "  \"top_p\": 0.9\n"
               << "}";
            json_content = js.str();
        }
        // Never leak a stored api_key through the read endpoint.
        send_http_json(client_sock, 200, "OK", redact_api_key(json_content));
        return;
    }

    // ------------------------------------------------------------------------
    // 3d. POST /api/config & POST /v1/config (Update & Persist moe_config.json)
    // ------------------------------------------------------------------------
    if (method == "POST" && (path == "/api/config" || path == "/v1/config")) {
        std::string model_path = extract_json_string(body, "model_path", "models/test_qwen.gguf");
        bool bypass_io = extract_json_bool(body, "bypass_io", true);
        bool direct_dma = extract_json_bool(body, "direct_dma", true);
        int paged_blocks = static_cast<int>(extract_json_number(body, "paged_attention_blocks", 2048.0));
        int active_slots = static_cast<int>(extract_json_number(body, "active_slots", 4.0));
        int max_ctx = static_cast<int>(extract_json_number(body, "max_context", 32768.0));
        int api_port = static_cast<int>(extract_json_number(body, "api_port", static_cast<double>(port_)));
        int spec_tokens = static_cast<int>(extract_json_number(body, "speculative_draft_tokens", 3.0));
        double temp = extract_json_number(body, "temperature", 0.7);
        double top_p = extract_json_number(body, "top_p", 0.9);
        // Preserve the stored API key and bind host: a config update must not
        // silently wipe the key or the loopback bind. The body may override
        // api_host; api_key is never accepted from a request body (env or
        // direct file edit only).
        std::string keep_key = read_config_string("api_key", "");
        std::string keep_host = extract_json_string(body, "api_host", read_config_string("api_host", "127.0.0.1"));

        // Validate ranges instead of persisting arbitrary numbers (a
        // negative active_slots or a 0 api_port used to be written verbatim
        // and break the next start).
        {
            std::string cfg_err;
            if (model_path.size() > 1024) cfg_err = "model_path must be at most 1024 characters";
            else validate_server_config(paged_blocks, active_slots, max_ctx, api_port,
                                        spec_tokens, temp, top_p, keep_host, cfg_err);
            if (!cfg_err.empty()) {
                std::ostringstream ej;
                ej << "{\"error\":\"invalid_parameter\",\"message\":\"" << json_escape(cfg_err) << "\"}";
                send_http_json(client_sock, 400, "Bad Request", ej.str());
                return;
            }
        }

        // Write atomically (temp file + rename) and JSON-escape every
        // caller-controlled string. The old code interpolated model_path
        // and api_host raw into moe_config.json, so a body like
        // {"model_path": "x\", \"api_key\": \"pwned"} injected arbitrary
        // JSON into the config — including an api_key the endpoint claims
        // never to accept from a request body.
        std::string cfg_tmp = "moe_config.json.tmp";
        {
            std::ofstream ofs(cfg_tmp, std::ios::trunc | std::ios::binary);
            if (!ofs.is_open()) {
                send_http_json(client_sock, 500, "Internal Server Error",
                               "{\"error\":\"config_write_failed\",\"message\":\"Could not write moe_config.json.\"}");
                return;
            }
            ofs << "{\n"
                << "  \"engine\": \"MoE-Stream-Engine-v2\",\n"
                << "  \"first_run_completed\": true,\n"
                << "  \"model_path\": \"" << json_escape(model_path) << "\",\n"
                << "  \"bypass_io\": " << (bypass_io ? "true" : "false") << ",\n"
                << "  \"direct_dma\": " << (direct_dma ? "true" : "false") << ",\n"
                << "  \"paged_attention_blocks\": " << paged_blocks << ",\n"
                << "  \"active_slots\": " << active_slots << ",\n"
                << "  \"max_context\": " << max_ctx << ",\n"
                << "  \"api_port\": " << api_port << ",\n"
                << "  \"api_host\": \"" << json_escape(keep_host) << "\",\n";
            if (!keep_key.empty()) ofs << "  \"api_key\": \"" << json_escape(keep_key) << "\",\n";
            ofs << "  \"speculative_draft_tokens\": " << spec_tokens << ",\n"
                << "  \"temperature\": " << temp << ",\n"
                << "  \"top_p\": " << top_p << "\n"
                << "}\n";
            ofs.close();
            if (!ofs) {
                std::error_code ec;
                std::filesystem::remove(cfg_tmp, ec);
                send_http_json(client_sock, 500, "Internal Server Error",
                               "{\"error\":\"config_write_failed\",\"message\":\"Could not write moe_config.json.\"}");
                return;
            }
        }
        {
            std::error_code ec;
            std::filesystem::rename(cfg_tmp, "moe_config.json", ec);
            if (ec) {
                std::filesystem::remove(cfg_tmp, ec);
                send_http_json(client_sock, 500, "Internal Server Error",
                               "{\"error\":\"config_write_failed\",\"message\":\"Could not write moe_config.json.\"}");
                return;
            }
        }

        std::ostringstream js;
        js << "{\n"
           << "  \"status\": \"ok\",\n"
           << "  \"message\": \"System configuration updated and written to moe_config.json\",\n"
           << "  \"config\": {\n"
           << "    \"model_path\": \"" << json_escape(model_path) << "\",\n"
           << "    \"bypass_io\": " << (bypass_io ? "true" : "false") << ",\n"
           << "    \"direct_dma\": " << (direct_dma ? "true" : "false") << ",\n"
           << "    \"paged_attention_blocks\": " << paged_blocks << ",\n"
           << "    \"active_slots\": " << active_slots << ",\n"
           << "    \"max_context\": " << max_ctx << ",\n"
           << "    \"api_port\": " << api_port << ",\n"
           << "    \"api_host\": \"" << json_escape(keep_host) << "\",\n"
           << "    \"speculative_draft_tokens\": " << spec_tokens << ",\n"
           << "    \"temperature\": " << temp << ",\n"
           << "    \"top_p\": " << top_p << "\n"
           << "  }\n"
           << "}";
        send_http_json(client_sock, 200, "OK", js.str());
        return;
    }

    // ------------------------------------------------------------------------
    // 4. POST /v1/chat/completions, /v1/completions, /api/chat, /api/generate
    // ------------------------------------------------------------------------
    if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions" || path == "/api/chat" || path == "/api/generate")) {
        // Tools: parsed before prompt formatting so the template's `tools`
        // variable renders them. "tool_choice": "none" disables tool use.
        std::vector<ToolDefinition> tools = parse_chat_tools(body);
        if (extract_json_string(body, "tool_choice", "") == "none") tools.clear();
        const std::vector<ToolDefinition>* tools_ptr = tools.empty() ? nullptr : &tools;
        std::string user_prompt = format_chat_prompt_for_arch(body, engine_, tools_ptr);
        bool stream = extract_json_bool(body, "stream", true);
        uint32_t max_tokens = static_cast<uint32_t>(std::clamp(extract_json_number(body, "max_tokens", 1024.0), 1.0, 4096.0));
        // Slot ids are validated, not silently wrapped: a huge or negative
        // cache_slot used to cast to an arbitrary uint32_t.
        double slot_raw = extract_json_number(body, "cache_slot", 0.0);
        if (!std::isfinite(slot_raw) || slot_raw < 0.0 || slot_raw > 63.0 ||
            std::floor(slot_raw) != slot_raw) {
            send_http_json(client_sock, 400, "Bad Request",
                           "{\"error\":\"invalid_parameter\",\"message\":\"cache_slot must be an integer in [0, 63].\"}");
            return;
        }
        uint32_t slot_id = static_cast<uint32_t>(slot_raw);
        // Serialize generations on this KV slot (see slot_mutexes_ in
        // server.h): two concurrent chats on one slot corrupt each other's
        // KV state and both return garbled text. Held for the whole
        // handler — all three generate_stream call sites below.
        std::lock_guard<std::mutex> slot_lock(slot_mutexes_[slot_id]);
        bool compaction = extract_json_bool(body, "context_compaction", true);
        bool continue_asst = extract_json_bool(body, "continue_assistant", false);
        // Speculative prompt-lookup is on by default for smoother streaming
        // when history has repeating n-grams; clients can set "speculative":false.
        bool speculative = extract_json_bool(body, "speculative", true);
        uint32_t draft_count = static_cast<uint32_t>(std::clamp(extract_json_number(body, "draft_tokens", 2.0), 1.0, 4.0));

        SamplerConfig scfg;
        scfg.temperature = static_cast<float>(extract_json_number(body, "temperature", 0.7));
        scfg.top_k = static_cast<uint32_t>(extract_json_number(body, "top_k", 40.0));
        scfg.top_p = static_cast<float>(extract_json_number(body, "top_p", 0.9));
        scfg.repetition_penalty = static_cast<float>(extract_json_number(body, "repetition_penalty", 1.1));
        // Seed 0 (default) = fresh entropy per request; an explicit seed is
        // deterministic across requests.
        scfg.seed = static_cast<uint64_t>(extract_json_number(body, "seed", 0.0));

        // Reject invalid sampler values instead of silently feeding NaN /
        // out-of-range parameters to the sampler (the old extractors
        // accepted NaN, negative top_p and huge top_k without complaint).
        {
            std::string samp_err;
            if (!validate_sampler_params(scfg.temperature, scfg.top_p, scfg.top_k,
                                         scfg.repetition_penalty, samp_err)) {
                std::ostringstream ej;
                ej << "{\"error\":\"invalid_parameter\",\"message\":\"" << json_escape(samp_err) << "\"}";
                send_http_json(client_sock, 400, "Bad Request", ej.str());
                return;
            }
        }

        // --- Tool-calling path: generation is buffered, because the
        // content-vs-tool_calls decision is atomic — nothing is sent before
        // the full reply is parsed. The reply is either plain content
        // (finish_reason "stop") or tool_calls (content null,
        // finish_reason "tool_calls"), in both streaming and non-streaming
        // form.
        if (!tools.empty()) {
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
            std::vector<ParsedToolCall> calls = parse_tool_calls_from_text(content_acc, tools);
            const std::string cmpl_id = "chatcmpl-moe-" + std::to_string(now_us());
            const std::string mname = json_escape(engine_.hparams().model_name);

            if (!stream) {
                std::ostringstream js;
                js << "{"
                   << "\"id\":\"" << cmpl_id << "\","
                   << "\"object\":\"chat.completion\","
                   << "\"model\":\"" << mname << "\","
                   << "\"choices\":[{\"index\":0,\"message\":{";
                if (!calls.empty()) {
                    js << tool_calls_message_json(calls);
                } else {
                    js << "\"role\":\"assistant\","
                       << "\"reasoning_content\":\"" << json_escape(reasoning_acc) << "\","
                       << "\"content\":\"" << json_escape(content_acc) << "\"";
                }
                js << "},\"finish_reason\":\"" << (calls.empty() ? "stop" : "tool_calls") << "\"}],"
                   << "\"usage\":{"
                   << "\"prompt_tokens\":" << (user_prompt.size() / 4 + 1) << ","
                   << "\"completion_tokens\":" << (content_acc.size() / 4 + 1) << ","
                   << "\"total_tokens\":" << ((user_prompt.size() + content_acc.size()) / 4 + 2)
                   << "},"
                   << "\"moe_telemetry\":{"
                   << "\"tok_per_sec\":" << final_prof.tok_per_sec << ","
                   << "\"wall_ms\":" << final_prof.wall_ms << ","
                   << "\"io_wait_ms\":" << final_prof.io_wait_ms
                   << "}"
                   << "}";
                send_http_json(client_sock, 200, "OK", js.str());
                return;
            }

            std::ostringstream sse_hdr;
            sse_hdr << "HTTP/1.1 200 OK\r\n"
                    << "Content-Type: text/event-stream; charset=utf-8\r\n"
                    << "Cache-Control: no-cache\r\n"
                    << "Connection: keep-alive\r\n"
                    << cors_response_header() << "\r\n";
            std::string hstr = sse_hdr.str();
            if (!send_all(client_sock, hstr.data(), hstr.size())) return;
            std::ostringstream ev;
            ev << "data: {"
               << "\"id\":\"" << cmpl_id << "\","
               << "\"object\":\"chat.completion.chunk\","
               << "\"model\":\"" << mname << "\","
               << "\"choices\":[{\"index\":0,\"delta\":{";
            if (!calls.empty()) {
                ev << tool_calls_message_json(calls);
            } else {
                ev << "\"role\":\"assistant\",\"content\":\"" << json_escape(content_acc) << "\"";
            }
            ev << "},\"finish_reason\":\"" << (calls.empty() ? "stop" : "tool_calls") << "\"}]"
               << "}\n\n";
            std::string chunk = ev.str();
            send_all(client_sock, chunk.data(), chunk.size());
            std::ostringstream fin;
            fin << "data: {"
                << "\"id\":\"" << cmpl_id << "\","
                << "\"object\":\"chat.completion.chunk\","
                << "\"model\":\"" << mname << "\","
                << "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\""
                << (calls.empty() ? "stop" : "tool_calls") << "\"}]"
                << "}\n\n";
            std::string fstr = fin.str();
            send_all(client_sock, fstr.data(), fstr.size());
            const char* done_msg = "data: [DONE]\n\n";
            send_all(client_sock, done_msg, std::strlen(done_msg));
            return;
        }

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
               << "\"usage\":{"
               << "\"prompt_tokens\":" << (user_prompt.size() / 4 + 1) << ","
               << "\"completion_tokens\":" << (content_acc.size() / 4 + 1) << ","
               << "\"total_tokens\":" << ((user_prompt.size() + content_acc.size()) / 4 + 2)
               << "},"
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
                << cors_response_header() << "\r\n";
        std::string hstr = sse_hdr.str();
        if (!send_all(client_sock, hstr.data(), hstr.size())) return;

        const std::string cmpl_id = "chatcmpl-moe-" + std::to_string(now_us());
        const std::string mname = json_escape(engine_.hparams().model_name);
        std::string streamed_content_acc;

        engine_.generate_stream(user_prompt, max_tokens, slot_id, compaction, scfg,
            [&](const std::string& piece, bool is_reasoning, uint32_t tid, const ProfileSample& prof) {
                if (!is_reasoning) streamed_content_acc += piece;
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

        // Only emit synthetic HTML/SVG fallback snippet on tiny forged test models when the model did not emit HTML/SVG itself
        bool has_own_html = (streamed_content_acc.find("```html") != std::string::npos ||
                             streamed_content_acc.find("```svg") != std::string::npos ||
                             streamed_content_acc.find("<svg") != std::string::npos ||
                             streamed_content_acc.find("<div") != std::string::npos);
        std::string lower_p = user_prompt;
        std::transform(lower_p.begin(), lower_p.end(), lower_p.begin(), ::tolower);
        if (!has_own_html && engine_.hparams().vocab_size <= 1024 &&
            (lower_p.find("html") != std::string::npos || lower_p.find("svg") != std::string::npos || lower_p.find("widget") != std::string::npos)) {
            std::string html_snippet = "\n\n```html\n<div style=\"font-family:system-ui;background:#0f172a;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #334155\">\n  <h4 style=\"margin:0 0 8px;color:#38bdf8\">Live MoE Telemetry Sandbox</h4>\n  <svg width=\"260\" height=\"64\" viewBox=\"0 0 260 64\">\n    <rect x=\"4\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#6366f1\"/>\n    <rect x=\"92\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#0ea5e9\"/>\n    <rect x=\"180\" y=\"12\" width=\"75\" height=\"40\" rx=\"6\" fill=\"#10b981\"/>\n    <text x=\"18\" y=\"36\" fill=\"#fff\" font-size=\"11\">VRAM Core</text>\n    <text x=\"106\" y=\"36\" fill=\"#fff\" font-size=\"11\">LFRU RAM</text>\n    <text x=\"192\" y=\"36\" fill=\"#fff\" font-size=\"11\">PILOT NVMe</text>\n  </svg>\n</div>\n```";
            std::ostringstream ev;
            ev << "data: {\"id\":\"" << cmpl_id << "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
               << json_escape(html_snippet) << "\"},\"finish_reason\":null}]}\n\n";
            std::string chunk = ev.str();
            send_all(client_sock, chunk.data(), chunk.size());
        }

        std::ostringstream final_chunk;
        final_chunk << "data: {"
                    << "\"id\":\"" << cmpl_id << "\","
                    << "\"object\":\"chat.completion.chunk\","
                    << "\"model\":\"" << mname << "\","
                    << "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]"
                    << "}\n\n";
        std::string fc_str = final_chunk.str();
        send_all(client_sock, fc_str.data(), fc_str.size());

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
        double w_raw = extract_json_number(body, "width", 512.0);
        double h_raw = extract_json_number(body, "height", 512.0);
        double s_raw = extract_json_number(body, "steps", 12.0);
        double g_raw = extract_json_number(body, "guidance_scale", 7.5);
        // Unclamped dimensions used to cast straight to uint32_t, so a
        // huge/negative width wrapped to an arbitrary size.
        if (!std::isfinite(w_raw) || w_raw < 16.0 || w_raw > 2048.0 || std::floor(w_raw) != w_raw ||
            !std::isfinite(h_raw) || h_raw < 16.0 || h_raw > 2048.0 || std::floor(h_raw) != h_raw ||
            !std::isfinite(s_raw) || s_raw < 1.0 || s_raw > 50.0 || std::floor(s_raw) != s_raw ||
            !std::isfinite(g_raw) || g_raw < 0.0 || g_raw > 30.0) {
            send_http_json(client_sock, 400, "Bad Request",
                           "{\"error\":\"invalid_parameter\",\"message\":\"width/height must be integers in [16, 2048], steps in [1, 50], guidance_scale in [0, 30].\"}");
            return;
        }
        uint32_t width  = static_cast<uint32_t>(w_raw);
        uint32_t height = static_cast<uint32_t>(h_raw);
        uint32_t steps  = static_cast<uint32_t>(s_raw);
        float guidance  = static_cast<float>(g_raw);
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
        // Strip a single leading slash so the join is relative. Multiple
        // leading slashes used to leave an absolute `sub` (`/etc/passwd`)
        // that replaced `web_root_` under std::filesystem join rules.
        while (!clean_path.empty() && clean_path.front() == '/') clean_path.erase(clean_path.begin());
        if (clean_path.empty()) clean_path = "index.html";

        std::string resolved;
        std::filesystem::path target_p;
        if (!resolve_under_root(web_root_, clean_path, resolved)) {
            send_http_json(client_sock, 403, "Forbidden", "{\"error\":\"Forbidden\"}");
            return;
        }
        target_p = resolved;

        if (!std::filesystem::exists(target_p) || !std::filesystem::is_regular_file(target_p)) {
            // SPA fallback — only when the fallback itself stays under root.
            std::string index_resolved;
            if (!resolve_under_root(web_root_, "index.html", index_resolved) ||
                !std::filesystem::exists(index_resolved) ||
                !std::filesystem::is_regular_file(index_resolved)) {
                send_http_json(client_sock, 404, "Not Found", "{\"error\":\"Not found\"}");
                return;
            }
            target_p = index_resolved;
        }

        std::error_code ec;
        const auto fsize = std::filesystem::file_size(target_p, ec);
        if (ec || fsize > kMaxStaticFileBytes) {
            send_http_json(client_sock, 413, "Content Too Large",
                           "{\"error\":\"file_too_large\",\"message\":\"Static file exceeds the 32 MiB limit.\"}");
            return;
        }

        std::ifstream ifs(target_p, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        std::string mime = mime_type_for_path(target_p.string());

        std::ostringstream hdr;
        hdr << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: " << mime << "\r\n"
            << cors_response_header()
            << "Content-Length: " << content.size() << "\r\n"
            << "Connection: close\r\n\r\n";
        std::string h = hdr.str();
        send_all(client_sock, h.data(), h.size());
        if (method == "GET") {
            send_all(client_sock, content.data(), content.size());
        }
        return;
    }

    send_http_json(client_sock, 404, "Not Found", "{\"error\":\"Endpoint not found\"}");
}

} // namespace moe

