// jxai_term — full-screen Linux terminal client for the JXAI native engine.
//
// Talks to `moe_engine serve` over HTTP/SSE (default 127.0.0.1:8001).
// No ncurses dependency: termios + ANSI/VT100 only.
//
// Build: produced as `build/jxai_term` via CMake (non-Windows).
// Run:   ./build/jxai_term [--host 127.0.0.1] [--port 8001]
//        ./start_jxai_term.sh   # starts engine if needed, then the TUI

#include "http_client.h"
#include "config_store.h"
#include "wizard.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Shared with wizard.h
volatile std::sig_atomic_t g_winch = 0;
volatile std::sig_atomic_t g_interrupt = 0;

namespace {

using jxai::term::ChatTurn;
using jxai::term::EngineClient;
using jxai::term::HealthInfo;
using jxai::term::StreamTelem;

// ── palette (ANSI) ──────────────────────────────────────────────────────────
// Deep ink + copper accent. Avoid purple / neon-glow tropes.
constexpr const char* RESET   = "\033[0m";
constexpr const char* BOLD    = "\033[1m";
constexpr const char* DIM     = "\033[2m";
constexpr const char* ITALIC  = "\033[3m";
constexpr const char* INVERSE = "\033[7m";
constexpr const char* FG_BRAND  = "\033[38;2;232;168;92m";   // copper
constexpr const char* FG_INK    = "\033[38;2;220;214;201m";   // warm paper
constexpr const char* FG_MUTED  = "\033[38;2;140;134;122m";
constexpr const char* FG_USER   = "\033[38;2;122;168;196m";   // steel blue
constexpr const char* FG_OK     = "\033[38;2;122;176;120m";
constexpr const char* FG_BAD    = "\033[38;2;196;96;84m";
constexpr const char* FG_META   = "\033[38;2;168;156;132m";
constexpr const char* BG_BAR    = "\033[48;2;28;26;22m";
constexpr const char* BG_INPUT  = "\033[48;2;36;33;28m";

void on_winch(int) { g_winch = 1; }
void on_int(int) { g_interrupt = 1; }

struct Winsize {
    int rows = 24;
    int cols = 80;
};

Winsize term_size() {
    Winsize w;
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0 && ws.ws_col > 0) {
        w.rows = ws.ws_row;
        w.cols = ws.ws_col;
    }
    return w;
}

struct TermiosGuard {
    termios orig{};
    bool ok = false;
    bool mouse = false;
    TermiosGuard() {
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return;
        if (tcgetattr(STDIN_FILENO, &orig) != 0) return;
        termios raw = orig;
        cfmakeraw(&raw);
        raw.c_lflag |= ISIG;          // keep Ctrl-C
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;          // 100ms poll
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return;
        // Alt screen + hide cursor. Mouse reporting is deferred until the
        // chat UI — enabling it during the wizard made clicks emit ESC
        // sequences that the wizard treated as quit.
        std::fputs("\033[?1049h\033[?25l\033[2J\033[H", stdout);
        std::fflush(stdout);
        ok = true;
    }
    void enable_mouse() {
        if (!ok || mouse) return;
        // SGR mouse (1006) + basic press/release (1000) for wheel scroll.
        std::fputs("\033[?1000h\033[?1006h", stdout);
        std::fflush(stdout);
        mouse = true;
    }
    void disable_mouse() {
        if (!ok || !mouse) return;
        std::fputs("\033[?1006l\033[?1000l", stdout);
        std::fflush(stdout);
        mouse = false;
    }
    ~TermiosGuard() {
        if (!ok) return;
        if (mouse) std::fputs("\033[?1006l\033[?1000l", stdout);
        std::fputs("\033[?25h\033[?1049l", stdout);
        std::fflush(stdout);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    }
};

enum class Role { System, User, Assistant, Error, Info };

struct Bubble {
    Role role = Role::Info;
    std::string text;
    std::string meta; // e.g. "42 tok · 9.1 t/s"
};

struct AppConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 8001;
    uint32_t max_tokens = 512;
    float temperature = 0.7f;
    uint32_t slot = 0;
    std::string system_prompt;
    std::string api_key;
    bool force_wizard = false;
    bool skip_wizard = false;
    std::string model_path; // from wizard / config
};

// Spawn moe_engine serve if nothing healthy is listening. Returns true when
// /health becomes ok (reused or freshly started).
bool ensure_engine(const AppConfig& cfg, pid_t& child_pid, std::string& err) {
    child_pid = -1;
    EngineClient probe(cfg.host, cfg.port);
    if (probe.health().online) return true;

    std::string model = cfg.model_path;
    if (model.empty()) {
        auto tc = jxai::term::load_term_config();
        model = tc.model_path;
    }
    if (model.empty()) {
        err = "no model_path configured — run the first-run wizard";
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(model, ec) || std::filesystem::file_size(model, ec) == 0) {
        err = "model file missing: " + model;
        return false;
    }
    if (!std::filesystem::exists("build/moe_engine", ec)) {
        err = "build/moe_engine not found — build the engine first";
        return false;
    }

    // Load HF token into env for any future pulls the engine may do.
    auto tok = jxai::term::load_hf_token();
    if (!tok.empty()) setenv("HF_TOKEN", tok.c_str(), 1);

    std::filesystem::create_directories("build", ec);
    pid_t pid = fork();
    if (pid == 0) {
        int logfd = ::open("build/jxai-engine.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (logfd >= 0) {
            dup2(logfd, STDOUT_FILENO);
            dup2(logfd, STDERR_FILENO);
            ::close(logfd);
        }
        std::string port = std::to_string(cfg.port);
        const char* web = std::filesystem::exists("web/dist/index.html") ? "web/dist" : nullptr;
        if (web) {
            execl("./build/moe_engine",
                  "./build/moe_engine",
                  "serve",
                  "--model", model.c_str(),
                  "--host", cfg.host.c_str(),
                  "--port", port.c_str(),
                  "--web-dir", web,
                  static_cast<char*>(nullptr));
        } else {
            execl("./build/moe_engine",
                  "./build/moe_engine",
                  "serve",
                  "--model", model.c_str(),
                  "--host", cfg.host.c_str(),
                  "--port", port.c_str(),
                  static_cast<char*>(nullptr));
        }
        _exit(127);
    }
    if (pid < 0) {
        err = "fork failed while starting engine";
        return false;
    }
    child_pid = pid;
    for (int i = 0; i < 90; ++i) {
        if (probe.health().online) return true;
        int st = 0;
        if (waitpid(pid, &st, WNOHANG) == pid) {
            err = "engine exited while starting — see build/jxai-engine.log";
            child_pid = -1;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    err = "engine did not become healthy within 45s";
    return false;
}

struct App {
    AppConfig cfg;
    EngineClient client;
    HealthInfo health;
    std::deque<Bubble> bubbles;
    std::vector<ChatTurn> history; // for API multi-turn
    std::vector<std::string> input_hist; // submitted composer lines (↑/↓ recall)
    int input_hist_idx = -1;             // -1 = editing a fresh line
    std::string input_draft;             // saved when browsing history
    std::string input;
    size_t cursor = 0;
    int scroll = 0; // lines from bottom (0 = pinned to latest)
    bool dirty = true;
    bool running = true;
    bool streaming = false;
    bool thinking = false;
    bool request_setup = false; // /setup — handled in main (needs TermiosGuard)
    std::string status_left;
    std::string status_right;
    double last_tps = 0.0;
    uint32_t stream_toks = 0;
    std::mutex mu;
    std::string pending_assistant; // live buffer while streaming

    explicit App(AppConfig c)
        : cfg(std::move(c)), client(cfg.host, cfg.port) {
        const char* env_key = std::getenv("MOE_API_KEY");
        if (env_key && *env_key && cfg.api_key.empty()) cfg.api_key = env_key;
        refresh_health();
        push_info(
            "JXAI terminal — local inference over the native engine.\n"
            "Enter sends · ↑/↓ history · PgUp scroll · Esc aborts · /help for commands.");
        if (!health.online) {
            push_error("Engine offline at " + cfg.host + ":" + std::to_string(cfg.port) +
                       ".\nStart it with ./start_jxai.sh  (or ./start_jxai_term.sh).");
        } else {
            push_info("Connected · " + health.model_name + " · " + health.architecture +
                      (health.n_layers ? (" · " + std::to_string(health.n_layers) + "L") : ""));
        }
    }

    void remember_input(const std::string& line) {
        if (line.empty()) return;
        if (!input_hist.empty() && input_hist.back() == line) {
            input_hist_idx = -1;
            input_draft.clear();
            return;
        }
        input_hist.push_back(line);
        if (input_hist.size() > 200) {
            input_hist.erase(input_hist.begin(), input_hist.begin() + static_cast<long>(input_hist.size() - 200));
        }
        input_hist_idx = -1;
        input_draft.clear();
    }

    void history_up() {
        if (input_hist.empty()) return;
        if (input_hist_idx < 0) {
            input_draft = input;
            input_hist_idx = static_cast<int>(input_hist.size()) - 1;
        } else if (input_hist_idx > 0) {
            --input_hist_idx;
        } else {
            return;
        }
        input = input_hist[static_cast<size_t>(input_hist_idx)];
        cursor = input.size();
        dirty = true;
    }

    void history_down() {
        if (input_hist_idx < 0) return;
        if (input_hist_idx + 1 < static_cast<int>(input_hist.size())) {
            ++input_hist_idx;
            input = input_hist[static_cast<size_t>(input_hist_idx)];
        } else {
            input_hist_idx = -1;
            input = input_draft;
            input_draft.clear();
        }
        cursor = input.size();
        dirty = true;
    }

    void refresh_health() {
        health = client.health();
    }

    void push_info(const std::string& t) {
        bubbles.push_back({Role::Info, t, {}});
        scroll = 0;
        dirty = true;
    }
    void push_error(const std::string& t) {
        bubbles.push_back({Role::Error, t, {}});
        scroll = 0;
        dirty = true;
    }

    void begin_assistant() {
        std::lock_guard<std::mutex> lk(mu);
        pending_assistant.clear();
        stream_toks = 0;
        last_tps = 0;
        thinking = false;
        bubbles.push_back({Role::Assistant, "", ""});
        scroll = 0;
        dirty = true;
    }

    void append_assistant(const std::string& piece, bool reasoning, const StreamTelem& telem) {
        std::lock_guard<std::mutex> lk(mu);
        if (bubbles.empty() || bubbles.back().role != Role::Assistant) {
            bubbles.push_back({Role::Assistant, "", ""});
        }
        if (reasoning) {
            if (!thinking) {
                bubbles.back().text += (bubbles.back().text.empty() ? "" : "\n") + std::string("[thinking] ");
                thinking = true;
            }
        } else if (thinking) {
            bubbles.back().text += " [/thinking]\n";
            thinking = false;
        }
        bubbles.back().text += piece;
        pending_assistant = bubbles.back().text;
        ++stream_toks;
        if (telem.tok_per_sec > 0) last_tps = telem.tok_per_sec;
        char meta[96];
        std::snprintf(meta, sizeof(meta), "%u tok · %.1f t/s", stream_toks, last_tps);
        bubbles.back().meta = meta;
        scroll = 0;
        dirty = true;
    }

    void finish_assistant(bool ok, const std::string& err) {
        std::lock_guard<std::mutex> lk(mu);
        streaming = false;
        if (thinking && !bubbles.empty() && bubbles.back().role == Role::Assistant) {
            bubbles.back().text += " [/thinking]";
            thinking = false;
        }
        if (!ok) {
            if (!err.empty()) {
                if (!bubbles.empty() && bubbles.back().role == Role::Assistant && bubbles.back().text.empty()) {
                    bubbles.pop_back();
                }
                bubbles.push_back({Role::Error, err, {}});
            }
        } else if (!pending_assistant.empty()) {
            history.push_back({"assistant", pending_assistant});
        }
        pending_assistant.clear();
        dirty = true;
    }
};

// ── text helpers ────────────────────────────────────────────────────────────

int display_width(const std::string& s) {
    // Count UTF-8 codepoints roughly as width-1 (good enough for Latin + most CJK
    // will be slightly under-counted; acceptable for wrapping).
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { ++w; ++i; }
        else if ((c >> 5) == 0x6) { ++w; i += 2; }
        else if ((c >> 4) == 0xE) { ++w; i += 3; }
        else if ((c >> 3) == 0x1E) { ++w; i += 4; }
        else { ++w; ++i; }
    }
    return w;
}

std::vector<std::string> wrap_text(const std::string& text, int width) {
    std::vector<std::string> lines;
    if (width < 8) width = 8;
    std::string cur;
    auto flush = [&]() {
        lines.push_back(cur);
        cur.clear();
    };
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '\n') {
            flush();
            ++i;
            continue;
        }
        // take next UTF-8 char
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c >> 5) == 0x6) len = 2;
        else if ((c >> 4) == 0xE) len = 3;
        else if ((c >> 3) == 0x1E) len = 4;
        if (i + len > text.size()) len = 1;
        if (display_width(cur) + 1 > width) flush();
        cur.append(text, i, len);
        i += len;
    }
    flush();
    if (lines.empty()) lines.push_back("");
    return lines;
}

std::string pad_or_trim(const std::string& s, int width) {
    int w = display_width(s);
    if (w == width) return s;
    if (w < width) return s + std::string(static_cast<size_t>(width - w), ' ');
    // trim by codepoints
    std::string out;
    int used = 0;
    for (size_t i = 0; i < s.size() && used < width - 1;) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(s[i]);
        if ((c >> 5) == 0x6) len = 2;
        else if ((c >> 4) == 0xE) len = 3;
        else if ((c >> 3) == 0x1E) len = 4;
        if (i + len > s.size()) len = 1;
        out.append(s, i, len);
        ++used;
        i += len;
    }
    out += "…";
    while (display_width(out) < width) out += ' ';
    return out;
}

// ── rendering ───────────────────────────────────────────────────────────────

void move_to(int row, int col) {
    std::printf("\033[%d;%dH", row, col);
}

void draw_app(App& app) {
    Winsize ws = term_size();
    const int rows = std::max(8, ws.rows);
    const int cols = std::max(40, ws.cols);
    const int header_h = 2;
    const int footer_h = 3;
    const int body_h = rows - header_h - footer_h;
    const int body_top = header_h + 1;

    // Build body lines from bubbles
    struct Line { std::string ansi; };
    std::vector<Line> body;
    {
        std::lock_guard<std::mutex> lk(app.mu);
        for (const auto& b : app.bubbles) {
            const char* role_fg = FG_MUTED;
            std::string label;
            switch (b.role) {
                case Role::User:      role_fg = FG_USER; label = "you"; break;
                case Role::Assistant: role_fg = FG_BRAND; label = "jxai"; break;
                case Role::Error:     role_fg = FG_BAD; label = "error"; break;
                case Role::System:    role_fg = FG_META; label = "system"; break;
                case Role::Info:      role_fg = FG_MUTED; label = ""; break;
            }
            if (!label.empty()) {
                std::string head = std::string(BOLD) + role_fg + label + RESET;
                if (!b.meta.empty() && b.role == Role::Assistant) {
                    head += std::string(DIM) + FG_MUTED + "  " + b.meta + RESET;
                }
                body.push_back({pad_or_trim("", cols)}); // spacer feel via empty then label
                // replace last empty with label line
                body.back().ansi = head + std::string(std::max(0, cols - display_width(label) - (b.meta.empty() ? 0 : display_width(b.meta) + 2)), ' ');
                // Actually pad carefully — ANSI messes width. Just print label left.
                body.back().ansi = std::string(BOLD) + role_fg + label + RESET;
                if (!b.meta.empty()) {
                    body.back().ansi += std::string("  ") + DIM + FG_MUTED + b.meta + RESET;
                }
            }
            auto wrapped = wrap_text(b.text, cols - 2);
            for (const auto& wl : wrapped) {
                std::string line = std::string("  ") + FG_INK + wl + RESET;
                body.push_back({line});
            }
        }
    }

    // scroll: 0 means show bottom
    int total = static_cast<int>(body.size());
    int max_scroll = std::max(0, total - body_h);
    if (app.scroll > max_scroll) app.scroll = max_scroll;
    if (app.scroll < 0) app.scroll = 0;
    int start = std::max(0, total - body_h - app.scroll);

    // ── header ──
    move_to(1, 1);
    std::string online = app.health.online ? (std::string(FG_OK) + "● online") : (std::string(FG_BAD) + "● offline");
    std::ostringstream hdr;
    hdr << BG_BAR << BOLD << FG_BRAND << " JXAI " << RESET << BG_BAR << FG_MUTED << "· "
        << FG_INK << (app.health.online ? app.health.model_name : "no engine")
        << FG_MUTED << " · " << online << RESET << BG_BAR;
    if (app.health.online && app.health.architecture != "—") {
        hdr << FG_MUTED << " · " << app.health.architecture;
        if (app.health.n_layers) hdr << " " << app.health.n_layers << "L/" << app.health.dim << "d";
        hdr << RESET << BG_BAR;
    }
    if (app.streaming && app.last_tps > 0) {
        char tps[32];
        std::snprintf(tps, sizeof(tps), " · %.1f t/s", app.last_tps);
        hdr << FG_BRAND << tps << RESET << BG_BAR;
    }
    std::string hdr_s = hdr.str();
    // Clear line with bar bg
    std::printf("%s", (BG_BAR + std::string(cols, ' ') + RESET).c_str());
    move_to(1, 1);
    std::fputs(hdr_s.c_str(), stdout);
    std::fputs(RESET, stdout);

    move_to(2, 1);
    {
        std::ostringstream meta;
        meta << "  " << app.cfg.host << ":" << app.cfg.port
             << "  temp " << std::fixed << std::setprecision(2) << app.cfg.temperature
             << "  max " << app.cfg.max_tokens
             << "  slot " << app.cfg.slot;
        if (app.scroll) meta << "  ↑" << app.scroll;
        std::printf("%s%s", FG_MUTED, pad_or_trim(meta.str(), cols).c_str());
        std::fputs(RESET, stdout);
    }

    // ── body ──
    for (int i = 0; i < body_h; ++i) {
        move_to(body_top + i, 1);
        std::fputs("\033[2K", stdout); // erase line
        int idx = start + i;
        if (idx >= 0 && idx < total) {
            std::fputs(body[static_cast<size_t>(idx)].ansi.c_str(), stdout);
        }
    }

    // ── composer ──
    int input_row = rows - 2;
    move_to(input_row, 1);
    std::string prompt = app.streaming ? " … " : " › ";
    std::string shown = app.input;
    // show a single visual line; if longer, window around cursor
    int avail = cols - display_width(prompt) - 1;
    if (avail < 8) avail = 8;
    std::string view = shown;
    if (display_width(view) > avail) {
        // keep end visible when cursor at end
        while (display_width(view) > avail && !view.empty()) {
            // drop first UTF-8 char
            unsigned char c = static_cast<unsigned char>(view[0]);
            size_t len = 1;
            if ((c >> 5) == 0x6) len = 2;
            else if ((c >> 4) == 0xE) len = 3;
            else if ((c >> 3) == 0x1E) len = 4;
            if (len > view.size()) len = 1;
            view.erase(0, len);
        }
    }
    std::printf("%s%s%s%s%s%s", BG_INPUT, BOLD, FG_BRAND, prompt.c_str(), RESET, BG_INPUT);
    std::printf("%s%s", FG_INK, pad_or_trim(view, avail).c_str());
    std::fputs(RESET, stdout);

    // ── footer hints ──
    move_to(rows, 1);
    std::string hints = app.streaming
        ? "  Esc abort stream  ·  Ctrl+C quit"
        : "  Enter send  ·  ↑/↓ history  ·  PgUp scroll  ·  /help  ·  Ctrl+C quit";
    std::printf("%s%s%s", DIM, FG_MUTED, pad_or_trim(hints, cols).c_str());
    std::fputs(RESET, stdout);

    // place cursor in input
    int cursor_col = display_width(prompt) + 1 + std::min(avail, display_width(shown));
    // better: approximate cursor within view
    cursor_col = display_width(prompt) + 1 + display_width(view);
    if (cursor_col > cols) cursor_col = cols;
    move_to(input_row, cursor_col);
    std::fputs("\033[?25h", stdout); // show cursor
    std::fflush(stdout);
    app.dirty = false;
}

// ── commands & send ─────────────────────────────────────────────────────────

void send_user(App& app, const std::string& text) {
    if (text.empty()) return;
    if (app.streaming) {
        app.push_error("Already streaming — press Esc to abort.");
        return;
    }
    if (!app.health.online) {
        app.refresh_health();
        if (!app.health.online) {
            app.push_error("Engine offline. Start ./start_jxai.sh first.");
            return;
        }
    }

    app.bubbles.push_back({Role::User, text, {}});
    app.history.push_back({"user", text});
    app.scroll = 0;

    std::vector<ChatTurn> msgs;
    if (!app.cfg.system_prompt.empty()) {
        msgs.push_back({"system", app.cfg.system_prompt});
    }
    // keep a reasonable history window
    const size_t keep = 24;
    size_t start = app.history.size() > keep ? app.history.size() - keep : 0;
    for (size_t i = start; i < app.history.size(); ++i) msgs.push_back(app.history[i]);

    app.streaming = true;
    app.begin_assistant();

    app.client.stream_chat_async(
        msgs, app.cfg.max_tokens, app.cfg.temperature, app.cfg.slot, app.cfg.api_key,
        [&app](const std::string& piece, bool reasoning, const StreamTelem& telem) {
            app.append_assistant(piece, reasoning, telem);
        },
        [&app](bool ok, const std::string& err) {
            app.finish_assistant(ok, err);
            app.refresh_health();
            app.dirty = true;
        });
}

bool handle_slash(App& app, const std::string& line) {
    if (line.empty() || line[0] != '/') return false;
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;
    auto rest = [&]() {
        std::string r;
        std::getline(iss, r);
        if (!r.empty() && r[0] == ' ') r.erase(0, 1);
        return r;
    };

    if (cmd == "/help" || cmd == "/?") {
        app.push_info(
            "Commands:\n"
            "  /help                 this help\n"
            "  /clear                clear chat + engine KV slot\n"
            "  /health               refresh engine status\n"
            "  /model                show loaded model\n"
            "  /temp <0-2>           sampling temperature\n"
            "  /tokens <n>           max new tokens\n"
            "  /slot <0-63>          KV cache slot\n"
            "  /system <text>        set system prompt (empty clears)\n"
            "  /setup                re-run setup wizard (HF token + model)\n"
            "  /retry                resend last user message\n"
            "  /abort                stop current stream\n"
            "  /quit                 exit\n"
            "Keys: ↑/↓ recall input · PgUp/PgDn/wheel scroll transcript");
        return true;
    }
    if (cmd == "/setup" || cmd == "/wizard") {
        if (app.streaming) {
            app.client.stop();
            app.streaming = false;
        }
        app.request_setup = true;
        app.push_info("Opening setup wizard…");
        return true;
    }
    if (cmd == "/quit" || cmd == "/exit" || cmd == "/q") {
        app.running = false;
        return true;
    }
    if (cmd == "/clear") {
        if (app.streaming) {
            app.client.stop();
            app.streaming = false;
        }
        app.bubbles.clear();
        app.history.clear();
        std::string err;
        if (app.client.clear_slot(app.cfg.slot, app.cfg.api_key, err)) {
            app.push_info("Cleared chat and KV slot " + std::to_string(app.cfg.slot) + ".");
        } else {
            app.push_info("Cleared local chat. Engine KV clear failed: " + err);
        }
        return true;
    }
    if (cmd == "/health") {
        app.refresh_health();
        if (app.health.online) {
            app.push_info("Online · " + app.health.model_name + " · " + app.health.architecture +
                          " · " + app.health.simd);
        } else {
            app.push_error("Offline at " + app.cfg.host + ":" + std::to_string(app.cfg.port));
        }
        return true;
    }
    if (cmd == "/model") {
        app.refresh_health();
        app.push_info(app.health.online
            ? ("model " + app.health.model_name + " (" + app.health.architecture + ", " +
               app.health.format + ")")
            : "no model — engine offline");
        return true;
    }
    if (cmd == "/temp") {
        std::string v = rest();
        if (v.empty()) {
            app.push_info("temperature = " + std::to_string(app.cfg.temperature));
        } else {
            try {
                float t = std::stof(v);
                if (t < 0.f || t > 2.f || !std::isfinite(t)) throw std::runtime_error("range");
                app.cfg.temperature = t;
                app.push_info("temperature → " + v);
            } catch (...) {
                app.push_error("usage: /temp <0-2>");
            }
        }
        return true;
    }
    if (cmd == "/tokens") {
        std::string v = rest();
        if (v.empty()) {
            app.push_info("max_tokens = " + std::to_string(app.cfg.max_tokens));
        } else {
            try {
                int n = std::stoi(v);
                if (n < 1 || n > 8192) throw std::runtime_error("range");
                app.cfg.max_tokens = static_cast<uint32_t>(n);
                app.push_info("max_tokens → " + v);
            } catch (...) {
                app.push_error("usage: /tokens <1-8192>");
            }
        }
        return true;
    }
    if (cmd == "/slot") {
        std::string v = rest();
        if (v.empty()) {
            app.push_info("slot = " + std::to_string(app.cfg.slot));
        } else {
            try {
                int n = std::stoi(v);
                if (n < 0 || n > 63) throw std::runtime_error("range");
                app.cfg.slot = static_cast<uint32_t>(n);
                app.push_info("slot → " + v);
            } catch (...) {
                app.push_error("usage: /slot <0-63>");
            }
        }
        return true;
    }
    if (cmd == "/system") {
        app.cfg.system_prompt = rest();
        app.push_info(app.cfg.system_prompt.empty() ? "system prompt cleared"
                                                    : ("system prompt set (" +
                                                       std::to_string(app.cfg.system_prompt.size()) +
                                                       " chars)"));
        return true;
    }
    if (cmd == "/abort") {
        if (app.streaming) {
            app.client.stop();
            app.push_info("Aborted.");
        } else {
            app.push_info("Nothing to abort.");
        }
        return true;
    }
    if (cmd == "/retry") {
        std::string last;
        for (auto it = app.history.rbegin(); it != app.history.rend(); ++it) {
            if (it->role == "user") { last = it->content; break; }
        }
        if (last.empty()) {
            app.push_error("No user message to retry.");
        } else {
            // drop last assistant if present
            if (!app.history.empty() && app.history.back().role == "assistant") app.history.pop_back();
            if (!app.history.empty() && app.history.back().role == "user") app.history.pop_back();
            send_user(app, last);
        }
        return true;
    }
    app.push_error("Unknown command: " + cmd + "  (try /help)");
    return true;
}

// ── input reading ───────────────────────────────────────────────────────────

enum class Key {
    None, Char, Enter, Backspace, Escape, CtrlJ, CtrlC, CtrlL, CtrlU,
    Up, Down, Left, Right, Home, End, PgUp, PgDn, WheelUp, WheelDown
};

struct KeyEvent {
    Key key = Key::None;
    char32_t ch = 0;
    std::string utf8;
};

// Byte buffer so a multi-char paste ("/health\r") is not truncated to '/'.
std::string& key_buf() {
    static std::string b;
    return b;
}

void fill_key_buf() {
    unsigned char tmp[256];
    ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
    if (n > 0) key_buf().append(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(tmp) + n);
}

KeyEvent read_key() {
    KeyEvent ev;
    auto& buf = key_buf();
    if (buf.empty()) fill_key_buf();
    if (buf.empty()) return ev;

    auto take = [&](size_t n) {
        buf.erase(0, n);
    };

    // Mouse SGR: ESC [ < b ; x ; y M/m  (clicks ignored; wheel → scroll)
    if (buf.size() >= 3 && buf[0] == 0x1b && buf[1] == '[' && buf[2] == '<') {
        size_t end = buf.find_first_of("Mm");
        if (end == std::string::npos) {
            fill_key_buf();
            end = buf.find_first_of("Mm");
            if (end == std::string::npos) {
                // Still incomplete — wait; do NOT treat as Esc (that quit the app).
                if (buf.size() > 64) take(buf.size()); // corrupt / truncated: drop
                return ev;
            }
        }
        int button = 0, x = 0, y = 0;
        char kind = 0;
        if (std::sscanf(buf.c_str(), "\033[<%d;%d;%d%c", &button, &x, &y, &kind) >= 4) {
            if (button == 64) ev.key = Key::WheelUp;
            else if (button == 65) ev.key = Key::WheelDown;
            // button 0/1/2 (left/mid/right) and releases: ignore
        }
        take(end + 1);
        return ev;
    }

    // Legacy X10 mouse: ESC [ M Cb Cx Cy — ignore entirely (no wheel encoding here).
    if (buf.size() >= 3 && buf[0] == 0x1b && buf[1] == '[' && buf[2] == 'M') {
        if (buf.size() < 6) {
            fill_key_buf();
            if (buf.size() < 6) return ev;
        }
        take(6);
        return ev;
    }

    if (static_cast<unsigned char>(buf[0]) == 0x1b) {
        if (buf.size() == 1) {
            // Could be a real Esc or the start of a CSI — peek briefly.
            fill_key_buf();
            if (buf.size() == 1) { ev.key = Key::Escape; take(1); return ev; }
        }
        if (buf.size() >= 3 && buf[1] == '[') {
            char c = buf[2];
            if (c == 'A') { ev.key = Key::Up; take(3); return ev; }
            if (c == 'B') { ev.key = Key::Down; take(3); return ev; }
            if (c == 'C') { ev.key = Key::Right; take(3); return ev; }
            if (c == 'D') { ev.key = Key::Left; take(3); return ev; }
            if (c == 'H') { ev.key = Key::Home; take(3); return ev; }
            if (c == 'F') { ev.key = Key::End; take(3); return ev; }
            if (buf.size() >= 4 && c == '5' && buf[3] == '~') { ev.key = Key::PgUp; take(4); return ev; }
            if (buf.size() >= 4 && c == '6' && buf[3] == '~') { ev.key = Key::PgDn; take(4); return ev; }
            // Unknown CSI — consume through the final byte (0x40–0x7E), never
            // emit Escape (mouse / focus / keypad leftovers used to quit).
            size_t i = 2;
            while (i < buf.size()) {
                unsigned char ch = static_cast<unsigned char>(buf[i]);
                if (ch >= 0x40 && ch <= 0x7E) { take(i + 1); return ev; }
                ++i;
            }
            fill_key_buf();
            return ev; // incomplete CSI; wait
        }
        ev.key = Key::Escape;
        take(1);
        return ev;
    }

    unsigned char b0 = static_cast<unsigned char>(buf[0]);
    if (b0 == 0x03) { ev.key = Key::CtrlC; take(1); return ev; }
    if (b0 == 0x0d || b0 == 0x0a) { ev.key = Key::Enter; take(1); return ev; }
    if (b0 == 0x0c) { ev.key = Key::CtrlL; take(1); return ev; }
    if (b0 == 0x15) { ev.key = Key::CtrlU; take(1); return ev; }
    if (b0 == 0x7f || b0 == 0x08) { ev.key = Key::Backspace; take(1); return ev; }
    if (b0 == 0x0e) { ev.key = Key::CtrlJ; take(1); return ev; }

    // UTF-8 codepoint
    size_t len = 1;
    if ((b0 >> 5) == 0x6) len = 2;
    else if ((b0 >> 4) == 0xE) len = 3;
    else if ((b0 >> 3) == 0x1E) len = 4;
    if (buf.size() < len) {
        fill_key_buf();
        if (buf.size() < len) return ev;
    }
    ev.key = Key::Char;
    ev.utf8 = buf.substr(0, len);
    take(len);
    return ev;
}

void insert_text(App& app, const std::string& utf8) {
    if (app.input_hist_idx >= 0) {
        app.input_hist_idx = -1;
        app.input_draft.clear();
    }
    if (app.cursor > app.input.size()) app.cursor = app.input.size();
    app.input.insert(app.cursor, utf8);
    app.cursor += utf8.size();
    app.dirty = true;
}

void backspace(App& app) {
    if (app.cursor == 0) return;
    if (app.input_hist_idx >= 0) {
        app.input_hist_idx = -1;
        app.input_draft.clear();
    }
    // delete one UTF-8 codepoint before cursor
    size_t i = app.cursor;
    do { --i; } while (i > 0 && (static_cast<unsigned char>(app.input[i]) & 0xC0) == 0x80);
    app.input.erase(i, app.cursor - i);
    app.cursor = i;
    app.dirty = true;
}

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [--host ADDR] [--port N] [--temp F] [--tokens N] [--slot N]\n"
        << "                       [--system TEXT] [--api-key KEY] [--wizard] [--skip-wizard]\n"
        << "\n"
        << "Full-screen terminal client for the JXAI native engine.\n"
        << "On first run, a setup wizard asks for an HF token and model.\n"
        << "Default endpoint: 127.0.0.1:8001  (same as ./start_jxai.sh)\n"
        << "Set MOE_API_KEY if the engine requires auth.\n";
}

AppConfig parse_args(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else if (a == "--host") cfg.host = need("--host");
        else if (a == "--port") cfg.port = static_cast<uint16_t>(std::stoi(need("--port")));
        else if (a == "--temp") cfg.temperature = std::stof(need("--temp"));
        else if (a == "--tokens") cfg.max_tokens = static_cast<uint32_t>(std::stoi(need("--tokens")));
        else if (a == "--slot") cfg.slot = static_cast<uint32_t>(std::stoi(need("--slot")));
        else if (a == "--system") cfg.system_prompt = need("--system");
        else if (a == "--api-key") cfg.api_key = need("--api-key");
        else if (a == "--wizard") cfg.force_wizard = true;
        else if (a == "--skip-wizard") cfg.skip_wizard = true;
        else {
            std::cerr << "unknown argument: " << a << "\n";
            usage(argv[0]);
            std::exit(2);
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    AppConfig cfg = parse_args(argc, argv);

    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        std::cerr << "jxai_term needs an interactive TTY.\n"
                  << "For scripting, use curl against /v1/chat/completions instead.\n";
        return 1;
    }

    std::signal(SIGWINCH, on_winch);
    std::signal(SIGINT, on_int);

    // Prefill from saved term config / HF token when present.
    {
        auto tc = jxai::term::load_term_config();
        if (!tc.host.empty()) cfg.host = tc.host;
        if (tc.port) cfg.port = tc.port;
        if (!tc.model_path.empty()) cfg.model_path = tc.model_path;
        if (tc.max_tokens) cfg.max_tokens = tc.max_tokens;
        cfg.temperature = tc.temperature;
        if (!tc.system_prompt.empty()) cfg.system_prompt = tc.system_prompt;
        auto tok = jxai::term::load_hf_token();
        if (!tok.empty()) setenv("HF_TOKEN", tok.c_str(), 0);
    }

    TermiosGuard guard;
    if (!guard.ok) {
        std::cerr << "failed to enter raw terminal mode\n";
        return 1;
    }

    pid_t engine_pid = -1;

    const bool run_wizard = !cfg.skip_wizard &&
        jxai::term::needs_first_run_wizard(cfg.force_wizard);
    if (run_wizard) {
        jxai::term::WizardResult wr;
        if (!jxai::term::run_first_run_wizard(wr)) {
            return 1; // user aborted
        }
        cfg.model_path = wr.model_path;
        cfg.host = wr.host;
        cfg.port = wr.port;
        // Clear screen before chat UI
        std::fputs("\033[2J\033[H", stdout);
        std::fflush(stdout);
    }

    // Mouse only in the chat UI (wheel scroll). Enabling earlier made
    // terminal clicks look like Esc and quit the wizard / main loop.
    guard.enable_mouse();

    // Bring up the engine if needed (wizard may have just downloaded a model).
    {
        std::string eng_err;
        if (!ensure_engine(cfg, engine_pid, eng_err)) {
            // Still open the TUI so the user can see the error /health offline.
            // Message is pushed after App construction below via health check.
            (void)eng_err;
        }
    }

    App app(cfg);
    if (!cfg.model_path.empty() && !app.health.online) {
        app.push_info("Starting engine with " + cfg.model_path + " …");
        // one more poll after a short wait
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        app.refresh_health();
    }
    draw_app(app);

    auto handle_event = [&](const KeyEvent& ev) -> bool {
        // returns false when the main loop should stop
        if (ev.key == Key::None) return true;
        if (ev.key == Key::CtrlC) {
            app.running = false;
            return false;
        }
        if (ev.key == Key::Escape) {
            if (app.streaming) {
                app.client.stop();
                app.push_info("Aborted.");
            }
            return true;
        }
        if (ev.key == Key::CtrlL) { app.dirty = true; return true; }
        if (ev.key == Key::CtrlU) {
            app.input.clear();
            app.cursor = 0;
            app.dirty = true;
            return true;
        }
        if (ev.key == Key::Up) {
            app.history_up();
            return true;
        }
        if (ev.key == Key::Down) {
            app.history_down();
            return true;
        }
        if (ev.key == Key::PgUp || ev.key == Key::WheelUp) {
            app.scroll += (ev.key == Key::PgUp ? 8 : 1);
            app.dirty = true;
            return true;
        }
        if (ev.key == Key::PgDn || ev.key == Key::WheelDown) {
            app.scroll -= (ev.key == Key::PgDn ? 8 : 1);
            app.dirty = true;
            return true;
        }
        if (ev.key == Key::Left) {
            if (app.cursor > 0) {
                do { --app.cursor; } while (app.cursor > 0 &&
                    (static_cast<unsigned char>(app.input[app.cursor]) & 0xC0) == 0x80);
                app.dirty = true;
            }
            return true;
        }
        if (ev.key == Key::Right) {
            if (app.cursor < app.input.size()) {
                unsigned char c = static_cast<unsigned char>(app.input[app.cursor]);
                size_t len = 1;
                if ((c >> 5) == 0x6) len = 2;
                else if ((c >> 4) == 0xE) len = 3;
                else if ((c >> 3) == 0x1E) len = 4;
                app.cursor = std::min(app.input.size(), app.cursor + len);
                app.dirty = true;
            }
            return true;
        }
        if (ev.key == Key::Home) { app.cursor = 0; app.dirty = true; return true; }
        if (ev.key == Key::End) { app.cursor = app.input.size(); app.dirty = true; return true; }
        if (ev.key == Key::Backspace) { backspace(app); return true; }
        if (ev.key == Key::CtrlJ) { insert_text(app, "\n"); return true; }
        if (ev.key == Key::Enter) {
            std::string line = app.input;
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
                line.pop_back();
            app.input.clear();
            app.cursor = 0;
            app.dirty = true;
            if (!line.empty()) {
                app.remember_input(line);
                if (line[0] == '/') handle_slash(app, line);
                else send_user(app, line);
            }
            return true;
        }
        if (ev.key == Key::Char && !ev.utf8.empty()) {
            if (static_cast<unsigned char>(ev.utf8[0]) >= 0x20) insert_text(app, ev.utf8);
        }
        return true;
    };

    while (app.running && !g_interrupt) {
        if (app.request_setup) {
            app.request_setup = false;
            guard.disable_mouse();
            std::fputs("\033[2J\033[H", stdout);
            std::fflush(stdout);
            jxai::term::WizardResult wr;
            const bool ok = jxai::term::run_first_run_wizard(wr);
            std::fputs("\033[2J\033[H", stdout);
            std::fflush(stdout);
            guard.enable_mouse();
            if (!ok || !wr.completed) {
                app.push_info("Setup cancelled.");
                app.dirty = true;
                continue;
            }
            const std::string prev_model = app.cfg.model_path;
            app.cfg.model_path = wr.model_path;
            app.cfg.host = wr.host;
            app.cfg.port = wr.port;
            app.client.set_endpoint(app.cfg.host, app.cfg.port);
            app.bubbles.clear();
            app.history.clear();
            app.input.clear();
            app.cursor = 0;
            app.scroll = 0;
            {
                std::string cerr;
                (void)app.client.clear_slot(app.cfg.slot, app.cfg.api_key, cerr);
            }
            if (!app.health.online) {
                std::string eng_err;
                if (!ensure_engine(app.cfg, engine_pid, eng_err)) {
                    app.push_error("Could not start engine: " + eng_err);
                }
            } else if (!wr.model_path.empty() && wr.model_path != prev_model) {
                app.push_info("Loading " + wr.model_path + " …");
                draw_app(app);
                std::string lerr;
                if (!app.client.load_model(wr.model_path, app.cfg.api_key, lerr)) {
                    app.push_error("Model load failed: " + lerr);
                }
            }
            app.refresh_health();
            if (app.health.online) {
                app.push_info("Setup done · " + app.health.model_name + " · " +
                              app.health.architecture);
            } else {
                app.push_error("Engine offline after setup.");
            }
            app.dirty = true;
            continue;
        }

        if (g_winch) {
            g_winch = 0;
            app.dirty = true;
        }
        if (app.dirty) {
            std::fputs("\033[?25l", stdout);
            draw_app(app);
        }

        // Block for the first event, then drain any paste burst before redrawing.
        KeyEvent ev = read_key();
        if (ev.key == Key::None) continue;
        if (!handle_event(ev)) break;
        while (!key_buf().empty()) {
            KeyEvent more = read_key();
            if (more.key == Key::None) break;
            if (!handle_event(more)) { app.running = false; break; }
        }
    }

    app.client.stop();
    if (engine_pid > 0) {
        kill(engine_pid, SIGTERM);
        for (int i = 0; i < 10; ++i) {
            int st = 0;
            if (waitpid(engine_pid, &st, WNOHANG) == engine_pid) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        kill(engine_pid, SIGKILL);
        waitpid(engine_pid, nullptr, WNOHANG);
    }
    return 0;
}
