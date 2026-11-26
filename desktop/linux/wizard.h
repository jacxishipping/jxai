#pragma once
// First-run wizard for jxai_term: HF token → model select → download → save.
// Uses the same termios/ANSI style as the main TUI (caller owns raw mode).

#include "config_store.h"
#include "http_client.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Set by jxai_term.cpp signal handler.
extern volatile std::sig_atomic_t g_interrupt;

namespace jxai::term {

namespace fs = std::filesystem;

struct WizardModel {
    const char* key;
    const char* title;
    const char* blurb;
    const char* hf_repo;       // null = local-only option
    const char* hf_filename;   // null = local-only
    const char* local_rel;     // relative path under repo, e.g. models/real/...
    uint64_t approx_bytes;
};

inline const WizardModel kWizardModels[] = {
    {
        "qwen25-05b",
        "Qwen2.5-0.5B Instruct Q4_0",
        "Fast starter · ~409 MB · fits 8–16 GB RAM VMs",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_0.gguf",
        "models/real/qwen2.5-0.5b-instruct-q4_0.gguf",
        428730208ULL,
    },
    {
        "qwen25-15b",
        "Qwen2.5-1.5B Instruct Q4_0",
        "Balanced quality · ~1.0 GB",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_0.gguf",
        "models/real/qwen2.5-1.5b-instruct-q4_0.gguf",
        1050000000ULL,
    },
    {
        "qwen25-3b",
        "Qwen2.5-3B Instruct Q4_0",
        "Stronger answers · ~1.9 GB · needs ~4 GB free RAM",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_0.gguf",
        "models/real/qwen2.5-3b-instruct-q4_0.gguf",
        1997532128ULL,
    },
    {
        "local",
        "Local GGUF on disk",
        "Point at an existing .gguf file (no download)",
        nullptr,
        nullptr,
        nullptr,
        0,
    },
};

inline constexpr int kWizardModelCount = static_cast<int>(sizeof(kWizardModels) / sizeof(kWizardModels[0]));

struct WizardResult {
    bool completed = false;
    std::string model_path;
    std::string hf_token;
    std::string host = "127.0.0.1";
    uint16_t port = 8001;
};

enum class WizardStep {
    Welcome = 0,
    HfToken = 1,
    ModelSelect = 2,
    LocalPath = 3,
    Download = 4,
    Finish = 5,
};

struct WizardState {
    WizardStep step = WizardStep::Welcome;
    std::string hf_token;
    bool token_skip = false;
    int model_index = 0;
    std::string local_path;
    std::string status;
    std::string error;
    bool downloading = false;
    std::atomic<uint64_t> dl_have{0};
    std::atomic<uint64_t> dl_total{0};
    std::atomic<bool> dl_done{false};
    std::atomic<bool> dl_ok{false};
    std::string dl_err;
    std::string resolved_model;
    bool dirty = true;
    bool quit = false;
    // input buffer for token / path
    std::string field;
    size_t cursor = 0;
    bool mask_field = false;
};

// Forward decls of helpers expected from jxai_term.cpp (palette + input).
// To keep the wizard self-contained we re-declare a minimal draw/input API here.

namespace wizard_ui {

constexpr const char* RESET   = "\033[0m";
constexpr const char* BOLD    = "\033[1m";
constexpr const char* DIM     = "\033[2m";
constexpr const char* FG_BRAND  = "\033[38;2;232;168;92m";
constexpr const char* FG_INK    = "\033[38;2;220;214;201m";
constexpr const char* FG_MUTED  = "\033[38;2;140;134;122m";
constexpr const char* FG_OK     = "\033[38;2;122;176;120m";
constexpr const char* FG_BAD    = "\033[38;2;196;96;84m";
constexpr const char* FG_USER   = "\033[38;2;122;168;196m";
constexpr const char* BG_BAR    = "\033[48;2;28;26;22m";
constexpr const char* BG_INPUT  = "\033[48;2;36;33;28m";
constexpr const char* BG_CARD   = "\033[48;2;32;29;24m";

inline void move_to(int row, int col) { std::printf("\033[%d;%dH", row, col); }

inline int cols() {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    return 80;
}
inline int rows() {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return ws.ws_row;
    return 24;
}

inline std::string pad(const std::string& s, int width) {
    int w = static_cast<int>(s.size()); // ASCII-heavy UI
    if (w >= width) return s.substr(0, static_cast<size_t>(width));
    return s + std::string(static_cast<size_t>(width - w), ' ');
}

inline void clear_screen() {
    std::fputs("\033[2J\033[H\033[?25l", stdout);
}

inline std::string step_name(WizardStep s) {
    switch (s) {
        case WizardStep::Welcome:     return "Welcome";
        case WizardStep::HfToken:     return "HF token";
        case WizardStep::ModelSelect: return "Model";
        case WizardStep::LocalPath:   return "Local path";
        case WizardStep::Download:    return "Download";
        case WizardStep::Finish:      return "Finish";
    }
    return "?";
}

inline int step_index(WizardStep s) {
    return static_cast<int>(s);
}

inline void draw_chrome(WizardState& st) {
    const int R = rows();
    const int C = cols();
    clear_screen();

    // header
    move_to(1, 1);
    std::printf("%s%s", BG_BAR, BOLD);
    std::printf("%s", pad(std::string(" JXAI  ·  first-run setup"), C).c_str());
    std::fputs(RESET, stdout);

    // step pills
    move_to(3, 2);
    const WizardStep pills[] = {
        WizardStep::Welcome, WizardStep::HfToken, WizardStep::ModelSelect,
        WizardStep::Download, WizardStep::Finish
    };
    for (auto s : pills) {
        bool cur = (s == st.step) ||
                   (st.step == WizardStep::LocalPath && s == WizardStep::ModelSelect);
        bool done = step_index(st.step) > step_index(s) &&
                    !(st.step == WizardStep::LocalPath && s == WizardStep::Download);
        if (st.step == WizardStep::LocalPath && s == WizardStep::ModelSelect) cur = true;
        if (done) std::printf("%s✓ %s  %s", FG_OK, step_name(s).c_str(), FG_MUTED);
        else if (cur) std::printf("%s%s● %s  %s", BOLD, FG_BRAND, step_name(s).c_str(), RESET);
        else std::printf("%s○ %s  ", FG_MUTED, step_name(s).c_str());
    }
    std::fputs(RESET, stdout);

    move_to(R, 1);
    std::printf("%s%s", DIM, FG_MUTED);
    std::printf("%s", pad(" Enter continue · Backspace edit · Esc quit · ↑↓ / numbers select", C).c_str());
    std::fputs(RESET, stdout);
    (void)R;
}

inline void draw_field(int row, const std::string& label, const std::string& value,
                       size_t cursor, bool mask, bool focused) {
    const int C = cols();
    move_to(row, 2);
    std::printf("%s%s%s", FG_MUTED, label.c_str(), RESET);
    move_to(row + 1, 2);
    std::string shown;
    if (mask) shown.assign(value.size(), '*');
    else shown = value;
    int width = std::max(20, C - 4);
    if (static_cast<int>(shown.size()) > width) {
        shown = shown.substr(shown.size() - static_cast<size_t>(width));
    }
    std::printf("%s%s %s%s", BG_INPUT, focused ? FG_BRAND : FG_INK,
                pad(shown, width - 1).c_str(), RESET);
    move_to(row + 1, 4 + static_cast<int>(std::min(shown.size(), static_cast<size_t>(width - 1))));
    std::fputs("\033[?25h", stdout);
    (void)cursor;
}

inline void draw_welcome(WizardState&) {
    move_to(6, 4);
    std::printf("%s%sWelcome to JXAI%s", BOLD, FG_BRAND, RESET);
    move_to(8, 4);
    std::printf("%sThis wizard sets up local inference on your machine.%s", FG_INK, RESET);
    move_to(10, 4);
    std::printf("%sYou will:%s", FG_MUTED, RESET);
    move_to(11, 6);
    std::printf("%s1.%s  Save a Hugging Face token (optional, for gated/private models)", FG_BRAND, FG_INK);
    move_to(12, 6);
    std::printf("%s2.%s  Pick a GGUF model to download — or point at one you already have", FG_BRAND, FG_INK);
    move_to(13, 6);
    std::printf("%s3.%s  Write config and open the terminal chat UI", FG_BRAND, FG_INK);
    move_to(15, 4);
    std::printf("%sToken is stored in ~/.config/jxai/hf_token (mode 0600), never in the repo.%s",
                FG_MUTED, RESET);
    move_to(18, 4);
    std::printf("%s%sPress Enter to begin →%s", BOLD, FG_BRAND, RESET);
}

inline void draw_token(WizardState& st) {
    move_to(6, 4);
    std::printf("%s%sHugging Face token%s", BOLD, FG_BRAND, RESET);
    move_to(8, 4);
    std::printf("%sNeeded for gated repos and higher rate limits. Leave blank to skip.%s", FG_MUTED, RESET);
    move_to(9, 4);
    std::printf("%sCreate one at https://huggingface.co/settings/tokens%s", FG_MUTED, RESET);

    st.mask_field = true;
    draw_field(12, "HF token (paste, then Enter)", st.field, st.cursor, true, true);

    move_to(16, 4);
    if (!st.error.empty()) std::printf("%s%s%s", FG_BAD, st.error.c_str(), RESET);
    else if (st.token_skip) std::printf("%sSkipping token — public GGUFs still work.%s", FG_OK, RESET);
    else {
        auto existing = load_hf_token();
        if (!existing.empty() && st.field.empty()) {
            std::printf("%sA token is already saved. Enter keeps it; type a new one to replace; /skip clears.%s",
                        FG_OK, RESET);
        } else {
            std::printf("%sTip: type  /skip  to continue without a token.%s", FG_MUTED, RESET);
        }
    }
}

inline void draw_models(WizardState& st) {
    move_to(6, 4);
    std::printf("%s%sChoose a model%s", BOLD, FG_BRAND, RESET);
    move_to(7, 4);
    std::printf("%s↑↓ or number keys · Enter to confirm%s", FG_MUTED, RESET);

    for (int i = 0; i < kWizardModelCount; ++i) {
        const auto& m = kWizardModels[i];
        bool sel = (i == st.model_index);
        int row = 9 + i * 3;
        move_to(row, 4);
        std::printf("%s%s%s %d. %s%s",
                    sel ? BOLD : "",
                    sel ? FG_BRAND : FG_INK,
                    sel ? "●" : "○",
                    i + 1,
                    m.title,
                    RESET);
        move_to(row + 1, 8);
        std::printf("%s%s%s", FG_MUTED, m.blurb, RESET);
        if (m.local_rel) {
            std::error_code ec;
            if (fs::exists(m.local_rel, ec) && fs::file_size(m.local_rel, ec) > 0) {
                std::printf("  %s[already on disk]%s", FG_OK, RESET);
            }
        }
    }
    if (!st.error.empty()) {
        move_to(9 + kWizardModelCount * 3, 4);
        std::printf("%s%s%s", FG_BAD, st.error.c_str(), RESET);
    }
}

inline void draw_local(WizardState& st) {
    move_to(6, 4);
    std::printf("%s%sPath to a local .gguf%s", BOLD, FG_BRAND, RESET);
    st.mask_field = false;
    draw_field(9, "Absolute or repo-relative path", st.field, st.cursor, false, true);
    move_to(13, 4);
    if (!st.error.empty()) std::printf("%s%s%s", FG_BAD, st.error.c_str(), RESET);
    else std::printf("%sExample: models/real/my-model.gguf%s", FG_MUTED, RESET);
}

inline void draw_download(WizardState& st) {
    move_to(6, 4);
    std::printf("%s%sFetching model%s", BOLD, FG_BRAND, RESET);
    move_to(8, 4);
    std::printf("%s%s%s", FG_INK, st.status.c_str(), RESET);

    uint64_t have = st.dl_have.load();
    uint64_t total = st.dl_total.load();
    move_to(11, 4);
    if (total > 0) {
        double pct = 100.0 * static_cast<double>(have) / static_cast<double>(total);
        int bar_w = std::max(10, cols() - 20);
        int fill = static_cast<int>(bar_w * pct / 100.0);
        std::string bar;
        bar.reserve(static_cast<size_t>(bar_w));
        for (int i = 0; i < bar_w; ++i) bar += (i < fill) ? '#' : '.';
        std::printf("%s[%s]%s %5.1f%%  %.1f / %.1f MB",
                    FG_BRAND, bar.c_str(), RESET, pct,
                    have / 1e6, total / 1e6);
    } else if (st.downloading) {
        std::printf("%sDownloading… %.1f MB received%s", FG_BRAND, have / 1e6, RESET);
    }

    move_to(14, 4);
    if (!st.dl_err.empty()) std::printf("%s%s%s", FG_BAD, st.dl_err.c_str(), RESET);
    else if (st.dl_done.load() && st.dl_ok.load()) {
        std::printf("%sDownload complete. Press Enter →%s", FG_OK, RESET);
    } else if (!st.downloading && st.dl_done.load() && !st.dl_ok.load()) {
        std::printf("%sFailed. Fix the issue and press Enter to retry, or Esc to quit.%s", FG_BAD, RESET);
    }
}

inline void draw_finish(WizardState& st) {
    move_to(6, 4);
    std::printf("%s%sYou're set%s", BOLD, FG_OK, RESET);
    move_to(8, 4);
    std::printf("%sModel:%s  %s", FG_MUTED, FG_INK, st.resolved_model.c_str());
    move_to(9, 4);
    std::printf("%sConfig:%s %s", FG_MUTED, FG_INK, term_config_path().string().c_str());
    move_to(10, 4);
    auto tok = load_hf_token();
    std::printf("%sHF token:%s %s", FG_MUTED, FG_INK,
                tok.empty() ? "(none)" : "saved (~/.config/jxai/hf_token)");
    move_to(12, 4);
    std::printf("%sEndpoint:%s http://127.0.0.1:8001", FG_MUTED, FG_INK);
    move_to(15, 4);
    std::printf("%s%sPress Enter to open the terminal chat →%s", BOLD, FG_BRAND, RESET);
    if (!st.error.empty()) {
        move_to(17, 4);
        std::printf("%s%s%s", FG_BAD, st.error.c_str(), RESET);
    }
}

inline void draw(WizardState& st) {
    draw_chrome(st);
    switch (st.step) {
        case WizardStep::Welcome:     draw_welcome(st); break;
        case WizardStep::HfToken:     draw_token(st); break;
        case WizardStep::ModelSelect: draw_models(st); break;
        case WizardStep::LocalPath:   draw_local(st); break;
        case WizardStep::Download:    draw_download(st); break;
        case WizardStep::Finish:      draw_finish(st); break;
    }
    std::fflush(stdout);
    st.dirty = false;
}

} // namespace wizard_ui

inline bool curl_download(const std::string& url,
                          const std::string& dest,
                          const std::string& token,
                          uint64_t expected,
                          WizardState& st) {
    std::error_code ec;
    fs::create_directories(fs::path(dest).parent_path(), ec);
    std::string tmp = dest + ".partial";
    fs::remove(tmp, ec);

    st.downloading = true;
    st.dl_done = false;
    st.dl_ok = false;
    st.dl_have = 0;
    st.dl_total = expected;
    st.dl_err.clear();
    st.status = "Contacting Hugging Face…";
    st.dirty = true;

    std::thread progress([&]() {
        while (st.downloading && !st.dl_done.load()) {
            std::error_code e2;
            if (fs::exists(tmp, e2)) {
                st.dl_have = fs::file_size(tmp, e2);
                st.dirty = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    // Build curl argv via shell-safe system call using exec-style list would be better;
    // use posix_spawn-like via fork/exec.
    pid_t pid = fork();
    if (pid == 0) {
        std::string auth = "Authorization: Bearer " + token;
        std::vector<const char*> argv;
        argv.push_back("curl");
        argv.push_back("-L");
        argv.push_back("--fail");
        argv.push_back("--retry");
        argv.push_back("3");
        argv.push_back("--retry-delay");
        argv.push_back("2");
        argv.push_back("-o");
        argv.push_back(tmp.c_str());
        if (!token.empty()) {
            argv.push_back("-H");
            argv.push_back(auth.c_str());
        }
        argv.push_back(url.c_str());
        argv.push_back(nullptr);
        // silence curl meter — we show our own
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        execvp("curl", const_cast<char* const*>(argv.data()));
        _exit(127);
    }
    if (pid < 0) {
        st.downloading = false;
        st.dl_done = true;
        st.dl_ok = false;
        st.dl_err = "fork failed";
        progress.join();
        return false;
    }

    st.status = "Downloading " + dest + " …";
    int status = 0;
    while (true) {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::error_code e2;
        if (fs::exists(tmp, e2)) st.dl_have = fs::file_size(tmp, e2);
        wizard_ui::draw(st);
    }

    st.downloading = false;
    progress.join();

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        st.dl_ok = false;
        st.dl_done = true;
        st.dl_err = "curl failed (HTTP error or network). Check HF token / disk space.";
        fs::remove(tmp, ec);
        return false;
    }

    auto sz = fs::file_size(tmp, ec);
    if (ec || sz < 1024) {
        st.dl_ok = false;
        st.dl_done = true;
        st.dl_err = "downloaded file too small — likely an HTML error page";
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, dest, ec);
    if (ec) {
        st.dl_ok = false;
        st.dl_done = true;
        st.dl_err = "failed to finalize download: " + ec.message();
        return false;
    }
    st.dl_have = sz;
    st.dl_ok = true;
    st.dl_done = true;
    st.status = "Saved " + dest;
    st.resolved_model = dest;
    st.dirty = true;
    return true;
}

// Key reading with a sticky byte buffer so "/skip\r" yields Char then Enter.
enum class WKey { None, Enter, Esc, Up, Down, Backspace, Char, CtrlC };

struct WKeyEv { WKey key = WKey::None; std::string utf8; };

inline std::string& wizard_key_buf() {
    static std::string b;
    return b;
}

inline WKeyEv wizard_read_key() {
    WKeyEv ev;
    auto& buf = wizard_key_buf();
    if (buf.empty()) {
        unsigned char tmp[128];
        ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
        if (n > 0) buf.append(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(tmp) + n);
    }
    if (buf.empty()) return ev;

    auto take = [&](size_t n) { buf.erase(0, n); };

    unsigned char b0 = static_cast<unsigned char>(buf[0]);
    if (b0 == 0x1b) {
        if (buf.size() == 1) {
            // peek
            unsigned char tmp[16];
            ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
            if (n > 0) buf.append(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(tmp) + n);
            if (buf.size() == 1) { ev.key = WKey::Esc; take(1); return ev; }
        }
        // SGR mouse (ESC [ < … M/m) — ignore. Clicks used to become Esc and quit.
        if (buf.size() >= 3 && buf[1] == '[' && buf[2] == '<') {
            size_t end = buf.find_first_of("Mm");
            if (end == std::string::npos) {
                unsigned char tmp[64];
                ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
                if (n > 0) buf.append(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(tmp) + n);
                end = buf.find_first_of("Mm");
            }
            if (end != std::string::npos) take(end + 1);
            else if (buf.size() > 64) take(buf.size());
            return ev;
        }
        // X10 mouse: ESC [ M Cb Cx Cy
        if (buf.size() >= 3 && buf[1] == '[' && buf[2] == 'M') {
            if (buf.size() < 6) {
                unsigned char tmp[16];
                ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
                if (n > 0) buf.append(reinterpret_cast<char*>(tmp), reinterpret_cast<char*>(tmp) + n);
            }
            if (buf.size() >= 6) take(6);
            return ev;
        }
        if (buf.size() >= 3 && buf[1] == '[') {
            if (buf[2] == 'A') { ev.key = WKey::Up; take(3); return ev; }
            if (buf[2] == 'B') { ev.key = WKey::Down; take(3); return ev; }
            // Other CSI (not bare Esc) — consume terminator, do not quit.
            size_t i = 2;
            while (i < buf.size()) {
                unsigned char ch = static_cast<unsigned char>(buf[i]);
                if (ch >= 0x40 && ch <= 0x7E) { take(i + 1); return ev; }
                ++i;
            }
            return ev;
        }
        ev.key = WKey::Esc;
        take(1);
        return ev;
    }
    if (b0 == 0x03) { ev.key = WKey::CtrlC; take(1); return ev; }
    if (b0 == 0x0d || b0 == 0x0a) { ev.key = WKey::Enter; take(1); return ev; }
    if (b0 == 0x7f || b0 == 0x08) { ev.key = WKey::Backspace; take(1); return ev; }
    if (b0 < 0x20) { take(1); return ev; }

    // One printable UTF-8 codepoint per event (paste drains across loops).
    size_t len = 1;
    if ((b0 >> 5) == 0x6) len = 2;
    else if ((b0 >> 4) == 0xE) len = 3;
    else if ((b0 >> 3) == 0x1E) len = 4;
    if (buf.size() < len) return ev;
    ev.key = WKey::Char;
    ev.utf8 = buf.substr(0, len);
    take(len);
    return ev;
}

inline void field_backspace(WizardState& st) {
    if (st.field.empty()) return;
    size_t i = st.field.size();
    do { --i; } while (i > 0 && (static_cast<unsigned char>(st.field[i]) & 0xC0) == 0x80);
    st.field.erase(i);
    st.cursor = st.field.size();
    st.dirty = true;
}

inline bool commit_model_selection(WizardState& st) {
    st.error.clear();
    const auto& m = kWizardModels[st.model_index];
    if (!m.hf_repo) {
        st.step = WizardStep::LocalPath;
        st.field.clear();
        st.cursor = 0;
        st.dirty = true;
        return true;
    }
    // already on disk?
    std::error_code ec;
    if (m.local_rel && fs::exists(m.local_rel, ec) && fs::file_size(m.local_rel, ec) > 0) {
        st.resolved_model = m.local_rel;
        st.status = std::string("Using existing ") + m.local_rel;
        st.step = WizardStep::Finish;
        st.dirty = true;
        return true;
    }
    st.step = WizardStep::Download;
    st.dirty = true;
    return true;
}

inline bool start_selected_download(WizardState& st) {
    const auto& m = kWizardModels[st.model_index];
    if (!m.hf_repo || !m.hf_filename || !m.local_rel) {
        st.dl_err = "internal: not a downloadable model";
        return false;
    }
    std::string url = std::string("https://huggingface.co/") + m.hf_repo +
                      "/resolve/main/" + m.hf_filename;
    std::string token = st.hf_token;
    if (token.empty()) token = load_hf_token();
    // Export for child curl / later engine pulls
    if (!token.empty()) setenv("HF_TOKEN", token.c_str(), 1);
    return curl_download(url, m.local_rel, token, m.approx_bytes, st);
}

inline bool persist_wizard(WizardState& st, WizardResult& out) {
    if (st.resolved_model.empty()) {
        st.error = "No model path resolved";
        return false;
    }
    std::error_code ec;
    if (!fs::exists(st.resolved_model, ec) || fs::file_size(st.resolved_model, ec) == 0) {
        st.error = "Model file missing: " + st.resolved_model;
        return false;
    }

    if (!st.token_skip && !st.hf_token.empty()) {
        if (!save_hf_token(st.hf_token)) {
            st.error = "Could not write ~/.config/jxai/hf_token";
            return false;
        }
        setenv("HF_TOKEN", st.hf_token.c_str(), 1);
    } else if (st.token_skip) {
        // leave any existing token; /skip with empty means don't require one
    }

    TermConfig cfg;
    cfg.first_run_completed = true;
    cfg.model_path = st.resolved_model;
    cfg.host = "127.0.0.1";
    cfg.port = 8001;
    if (!save_term_config(cfg)) {
        st.error = "Could not write " + term_config_path().string();
        return false;
    }
    update_moe_config_model(cfg.model_path, cfg.host, cfg.port);

    out.completed = true;
    out.model_path = cfg.model_path;
    out.hf_token = load_hf_token();
    out.host = cfg.host;
    out.port = cfg.port;
    return true;
}

// Returns true when wizard finished successfully.
inline bool run_first_run_wizard(WizardResult& out) {
    WizardState st;
    // Prefill token field empty; we keep existing on Enter-with-empty.
    auto existing = load_hf_token();
    wizard_ui::draw(st);

    while (!st.quit && !g_interrupt) {
        if (st.dirty) wizard_ui::draw(st);

        // Auto-start download when entering Download step
        if (st.step == WizardStep::Download && !st.downloading && !st.dl_done.load()) {
            wizard_ui::draw(st);
            start_selected_download(st);
            st.dirty = true;
            continue;
        }

        auto handle = [&](const WKeyEv& ev) -> bool {
            // returns false to quit wizard unsuccessfully
            if (ev.key == WKey::None) return true;
            if (ev.key == WKey::CtrlC || ev.key == WKey::Esc) {
                st.quit = true;
                return false;
            }

            if (st.step == WizardStep::Welcome) {
                if (ev.key == WKey::Enter) {
                    st.step = WizardStep::HfToken;
                    st.field.clear();
                    st.dirty = true;
                }
                return true;
            }

            if (st.step == WizardStep::HfToken) {
                if (ev.key == WKey::Backspace) { field_backspace(st); return true; }
                if (ev.key == WKey::Char) {
                    st.field += ev.utf8;
                    st.cursor = st.field.size();
                    st.dirty = true;
                    return true;
                }
                if (ev.key == WKey::Enter) {
                    std::string v = st.field;
                    while (!v.empty() && v.back() == ' ') v.pop_back();
                    if (v == "/skip" || v == "skip") {
                        st.token_skip = true;
                        st.hf_token.clear();
                    } else if (v.empty()) {
                        st.hf_token = existing;
                        st.token_skip = existing.empty();
                    } else {
                        if (v.size() < 8) {
                            st.error = "Token looks too short — paste a full HF token or /skip";
                            st.dirty = true;
                            return true;
                        }
                        st.hf_token = v;
                        st.token_skip = false;
                        save_hf_token(v);
                        setenv("HF_TOKEN", v.c_str(), 1);
                    }
                    st.step = WizardStep::ModelSelect;
                    st.error.clear();
                    st.dirty = true;
                }
                return true;
            }

            if (st.step == WizardStep::ModelSelect) {
                if (ev.key == WKey::Up) {
                    st.model_index = (st.model_index + kWizardModelCount - 1) % kWizardModelCount;
                    st.dirty = true;
                } else if (ev.key == WKey::Down) {
                    st.model_index = (st.model_index + 1) % kWizardModelCount;
                    st.dirty = true;
                } else if (ev.key == WKey::Char && !ev.utf8.empty() && ev.utf8[0] >= '1' &&
                           ev.utf8[0] < '1' + kWizardModelCount) {
                    st.model_index = ev.utf8[0] - '1';
                    st.dirty = true;
                } else if (ev.key == WKey::Enter) {
                    commit_model_selection(st);
                }
                return true;
            }

            if (st.step == WizardStep::LocalPath) {
                if (ev.key == WKey::Backspace) { field_backspace(st); return true; }
                if (ev.key == WKey::Char) {
                    st.field += ev.utf8;
                    st.cursor = st.field.size();
                    st.dirty = true;
                    return true;
                }
                if (ev.key == WKey::Enter) {
                    std::string p = st.field;
                    while (!p.empty() && p.back() == ' ') p.pop_back();
                    std::error_code ec;
                    if (p.empty() || !fs::exists(p, ec) || fs::file_size(p, ec) == 0) {
                        st.error = "File not found or empty — enter a valid .gguf path";
                        st.dirty = true;
                        return true;
                    }
                    st.resolved_model = p;
                    st.step = WizardStep::Finish;
                    st.error.clear();
                    st.dirty = true;
                }
                return true;
            }

            if (st.step == WizardStep::Download) {
                if (ev.key == WKey::Enter) {
                    if (st.dl_done.load() && st.dl_ok.load()) {
                        st.step = WizardStep::Finish;
                        st.dirty = true;
                    } else if (st.dl_done.load() && !st.dl_ok.load()) {
                        st.dl_done = false;
                        st.dl_err.clear();
                        st.dirty = true;
                    }
                }
                return true;
            }

            if (st.step == WizardStep::Finish) {
                if (ev.key == WKey::Enter) {
                    if (persist_wizard(st, out)) {
                        st.quit = true;
                        // signal success via out.completed
                    } else {
                        st.dirty = true;
                    }
                }
            }
            return true;
        };

        WKeyEv ev = wizard_read_key();
        if (ev.key == WKey::None) {
            if (st.dirty) continue;
            continue;
        }
        if (!handle(ev)) break;
        while (!wizard_key_buf().empty()) {
            WKeyEv more = wizard_read_key();
            if (more.key == WKey::None) break;
            if (!handle(more)) { st.quit = true; break; }
        }
        if (out.completed) return true;
    }
    return out.completed;
}

} // namespace jxai::term
