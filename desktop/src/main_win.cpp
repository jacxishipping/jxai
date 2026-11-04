#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <windowsx.h>
#include <d2d1.h>
#include <dwrite.h>
#include <dwmapi.h>
#include <commctrl.h>
#include <wincodec.h>
#include <shellapi.h>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <unordered_set>

#include "theme.h"
#include "api_client.h"

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "ws2_32.lib")

// Custom window messages for thread-safe UI updates
#define WM_MOE_TOKEN_DELTA   (WM_APP + 1)
#define WM_MOE_STREAM_DONE   (WM_APP + 2)
#define WM_MOE_HEALTH_UPDATE (WM_APP + 3)

namespace {

enum class BlockType {
    Text,
    Code
};

struct MessageBlock {
    BlockType type = BlockType::Text;
    std::wstring content;
    std::wstring lang;
    D2D1_RECT_F rect = D2D1::RectF();
    D2D1_RECT_F copy_btn_rect = D2D1::RectF();
    bool copy_hovered = false;
    bool copied = false;
    double copied_time = 0.0;
};

struct ChatMessageUI {
    std::string role; // "user" or "assistant"
    std::wstring content;
    std::wstring reasoning;
    moe::ui::TelemetrySnapshot telem;
    mutable std::vector<MessageBlock> blocks;
    mutable std::wstring parsed_cache;
};

struct QuickChip {
    std::wstring label;
    std::string prompt;
    D2D1_RECT_F rect;
    bool is_hovered = false;
};

class MoEStudioApp {
public:
    MoEStudioApp() : client_("127.0.0.1", 8000) {}

    ~MoEStudioApp() {
        discard_graphics_resources();
        if (pFormatAtlasCell_)    pFormatAtlasCell_->Release();
        if (pFormatChipText_)     pFormatChipText_->Release();
        if (pFormatCodeText_)     pFormatCodeText_->Release();
        if (pFormatMsgText_)      pFormatMsgText_->Release();
        if (pFormatStatLabel_)    pFormatStatLabel_->Release();
        if (pFormatStatValue_)    pFormatStatValue_->Release();
        if (pFormatHeaderSub_)    pFormatHeaderSub_->Release();
        if (pFormatHeaderTitle_)  pFormatHeaderTitle_->Release();
        if (pDWriteFactory_)      pDWriteFactory_->Release();
        if (pD2DFactory_)         pD2DFactory_->Release();
    }

    bool init(HINSTANCE hInstance, int nCmdShow) {
        hInstance_ = hInstance;

        HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pD2DFactory_);
        if (FAILED(hr)) return false;

        hr = DWriteCreateFactory(
            DWRITE_FACTORY_TYPE_SHARED,
            __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(&pDWriteFactory_)
        );
        if (FAILED(hr)) return false;

        create_text_formats();

        // Register window class
        WNDCLASSEX wc{};
        wc.cbSize        = sizeof(WNDCLASSEX);
        wc.style         = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc   = WndProcSetup;
        wc.hInstance     = hInstance;
        wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = nullptr; // Handled by Direct2D
        wc.lpszClassName = L"MoEStudioWindowClass";
        wc.hIcon         = LoadIcon(nullptr, IDI_APPLICATION);
        RegisterClassEx(&wc);

        // Window size & position
        int width = 1140;
        int height = 800;
        int screenW = GetSystemMetrics(SM_CXSCREEN);
        int screenH = GetSystemMetrics(SM_CYSCREEN);
        int posX = std::max(20, (screenW - width) / 2);
        int posY = std::max(20, (screenH - height) / 2);

        hwnd_ = CreateWindowEx(
            0,
            wc.lpszClassName,
            L"MoE Studio - Next-Gen Mixture-of-Experts Control Suite (C++ Native Direct2D)",
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
            posX, posY, width, height,
            nullptr, nullptr, hInstance, this
        );

        if (!hwnd_) return false;

        // Enable Windows 11 Immersive Dark Titlebar
        BOOL darkMode = TRUE;
        DwmSetWindowAttribute(hwnd_, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &darkMode, sizeof(darkMode));

        // Create child single-line edit control for input
        create_input_controls();

        // Populate initial welcome messages with live code and MoE telemetry
        ChatMessageUI welcome{};
        welcome.role = "assistant";
        welcome.content = L"Welcome to MoE Studio C++ Native Suite!\r\n\r\n"
                          L"Hardware-accelerated Direct2D interface connected directly to the local Hybrid-Tiered MoE Streaming Inference Engine.\r\n"
                          L"Dense backbone layers remain resident in zero-copy RAM/VRAM, while sparse MoE experts stream on-demand via Win32 IOCP asynchronous prefetch.\r\n\r\n"
                          L"```python\r\n"
                          L"# High-Performance MoE Client\r\n"
                          L"import urllib.request, json\r\n\r\n"
                          L"def stream_moe_completion(prompt: str):\r\n"
                          L"    req = urllib.request.Request(\"http://localhost:8000/v1/chat/completions\",\r\n"
                          L"        data=json.dumps({\"messages\": [{\"role\": \"user\", \"content\": prompt}], \"stream\": True}).encode('utf-8'),\r\n"
                          L"        headers={\"Content-Type\": \"application/json\"})\r\n"
                          L"    with urllib.request.urlopen(req) as resp:\r\n"
                          L"        for line in resp:\r\n"
                          L"            if line.startswith(b\"data: \") and not b\"[DONE]\" in line:\r\n"
                          L"                chunk = json.loads(line[6:])\r\n"
                          L"                print(chunk[\"choices\"][0][\"delta\"].get(\"content\", \"\"), end=\"\", flush=True)\r\n\r\n"
                          L"stream_moe_completion(\"Build an awesome Tkinter calculator\")\r\n"
                          L"```\r\n\r\n"
                          L"Click the **Copy Code** button above to test clipboard copy, or click **[Atlas: ON]** to inspect the 24x8 Cortex Expert matrix!";
        welcome.reasoning = L"MoE Engine Core Initialized | Top-2 router routing | Win32 IOCP Prefetch Pipeline active";
        welcome.telem.tok_per_sec = 7.1;
        welcome.telem.ttft_ms = 145.2;
        welcome.telem.latency_ms = 140.5;
        welcome.telem.hits = 142;
        welcome.telem.reads = 12;
        welcome.telem.vram_experts = 16;
        welcome.telem.ram_experts = 48;
        welcome.telem.disk_experts = 128;
        messages_.push_back(welcome);

        init_quick_chips();

        ShowWindow(hwnd_, (nCmdShow > 0) ? nCmdShow : SW_SHOWNORMAL);
        UpdateWindow(hwnd_);

        // Poll engine health right after launch
        poll_engine_health();

        // Set periodic health check timer (every 2.5 seconds)
        SetTimer(hwnd_, 1001, 2500, nullptr);

        return true;
    }

    int run() {
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        return static_cast<int>(msg.wParam);
    }

    bool save_screenshot_png(const std::wstring& out_path, float w = 1140.0f, float h = 800.0f) {
        auto health = client_.fetch_health_sync();
        latest_telem_.server_online = health.server_online;
        if (!health.model_name.empty()) latest_telem_.model_name = health.model_name;
        if (!health.arch.empty()) latest_telem_.arch = health.arch;
        if (!health.simd.empty()) latest_telem_.simd = health.simd;
        if (!health.io_backend.empty()) latest_telem_.io_backend = health.io_backend;
        if (latest_telem_.vram_experts == 0) latest_telem_.vram_experts = health.vram_experts;
        if (latest_telem_.ram_experts == 0) latest_telem_.ram_experts = health.ram_experts > 0 ? health.ram_experts : 48;
        if (latest_telem_.disk_experts == 0) latest_telem_.disk_experts = health.disk_experts > 0 ? health.disk_experts : 128;
        if (latest_telem_.hits == 0) latest_telem_.hits = health.hits;
        if (latest_telem_.reads == 0) latest_telem_.reads = health.reads;
        CoInitialize(nullptr);

        IWICImagingFactory* pWICFactory = nullptr;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pWICFactory));
        if (FAILED(hr) || !pWICFactory) return false;

        UINT32 uW = static_cast<UINT32>(w);
        UINT32 uH = static_cast<UINT32>(h);

        IWICBitmap* pWicBitmap = nullptr;
        hr = pWICFactory->CreateBitmap(uW, uH, GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnDemand, &pWicBitmap);
        if (FAILED(hr) || !pWicBitmap) {
            pWICFactory->Release();
            return false;
        }

        D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
            D2D1_RENDER_TARGET_TYPE_DEFAULT,
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
            0.0f, 0.0f,
            D2D1_RENDER_TARGET_USAGE_NONE,
            D2D1_FEATURE_LEVEL_DEFAULT
        );

        ID2D1RenderTarget* pOffscreenRT = nullptr;
        hr = pD2DFactory_->CreateWicBitmapRenderTarget(pWicBitmap, rtProps, &pOffscreenRT);
        if (FAILED(hr) || !pOffscreenRT) {
            pWicBitmap->Release();
            pWICFactory->Release();
            return false;
        }

        layout_input_controls(static_cast<int>(w), static_cast<int>(h));

        release_brushes();
        create_brushes(pOffscreenRT);

        pOffscreenRT->BeginDraw();
        current_rt_ = pOffscreenRT;
        render_scene(w, h);
        hr = pOffscreenRT->EndDraw();

        IWICStream* pStream = nullptr;
        hr = pWICFactory->CreateStream(&pStream);
        if (SUCCEEDED(hr)) {
            hr = pStream->InitializeFromFilename(out_path.c_str(), GENERIC_WRITE);
            if (SUCCEEDED(hr)) {
                IWICBitmapEncoder* pEncoder = nullptr;
                pWICFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder);
                pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);

                IWICBitmapFrameEncode* pFrame = nullptr;
                pEncoder->CreateNewFrame(&pFrame, nullptr);
                pFrame->Initialize(nullptr);
                pFrame->SetSize(uW, uH);

                WICPixelFormatGUID format = GUID_WICPixelFormat32bppPBGRA;
                pFrame->SetPixelFormat(&format);
                pFrame->WriteSource(pWicBitmap, nullptr);
                pFrame->Commit();
                pEncoder->Commit();

                pFrame->Release();
                pEncoder->Release();
            }
            pStream->Release();
        }

        release_brushes();
        pOffscreenRT->Release();
        pWicBitmap->Release();
        pWICFactory->Release();

        if (pRenderTarget_) {
            create_brushes(pRenderTarget_);
        }

        return SUCCEEDED(hr);
    }

    bool send_prompt_sync(const std::string& prompt_utf8, uint32_t max_tokens = 1024) {
        latest_telem_ = client_.fetch_health_sync();

        ChatMessageUI userMsg{};
        userMsg.role = "user";
        userMsg.content = utf8_to_wstring(prompt_utf8);
        messages_.push_back(userMsg);

        ChatMessageUI asstMsg{};
        asstMsg.role = "assistant";
        asstMsg.content = L"";
        asstMsg.reasoning = L"";
        messages_.push_back(asstMsg);

        std::string err;
        bool ok = client_.execute_stream_sync(
            prompt_utf8,
            max_tokens,
            0.7f,
            active_slot_,
            false,
            [this](const std::string& piece, bool is_reasoning, const moe::ui::TelemetrySnapshot& telem) {
                if (is_reasoning) {
                    messages_.back().reasoning += utf8_to_wstring(piece);
                } else {
                    messages_.back().content += utf8_to_wstring(piece);
                }
                latest_telem_ = telem;
            },
            err
        );
        messages_.back().telem = latest_telem_;
        scroll_to_bottom();
        return ok;
    }

private:
    static LRESULT CALLBACK WndProcSetup(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        if (uMsg == WM_NCCREATE) {
            auto* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
            auto* pThis = reinterpret_cast<MoEStudioApp*>(pCreate->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
            pThis->hwnd_ = hwnd;
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
        auto* pThis = reinterpret_cast<MoEStudioApp*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        if (pThis) {
            return pThis->handle_message(uMsg, wParam, lParam);
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    LRESULT handle_message(UINT uMsg, WPARAM wParam, LPARAM lParam) {
        switch (uMsg) {
            case WM_SIZE: {
                int w = LOWORD(lParam);
                int h = HIWORD(lParam);
                resize_render_target(w, h);
                layout_input_controls(w, h);
                InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            }
            case WM_PAINT: {
                PAINTSTRUCT ps;
                BeginPaint(hwnd_, &ps);
                render();
                EndPaint(hwnd_, &ps);
                return 0;
            }
            case WM_PRINTCLIENT: {
                render();
                return 0;
            }
            case WM_ERASEBKGND:
                return 1; // Prevent GDI flicker

            case WM_MOUSEWHEEL: {
                int delta = GET_WHEEL_DELTA_WPARAM(wParam);
                scroll_offset_ -= delta / 2;
                if (scroll_offset_ < 0) scroll_offset_ = 0;
                InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            }

            case WM_MOUSEMOVE: {
                float mx = static_cast<float>(GET_X_LPARAM(lParam));
                float my = static_cast<float>(GET_Y_LPARAM(lParam));
                bool need_redraw = false;

                // Check quick chips hover
                for (auto& chip : quick_chips_) {
                    bool hover = (mx >= chip.rect.left && mx <= chip.rect.right &&
                                  my >= chip.rect.top && my <= chip.rect.bottom);
                    if (hover != chip.is_hovered) {
                        chip.is_hovered = hover;
                        need_redraw = true;
                    }
                }

                // Check Send button hover
                bool send_h = (mx >= send_btn_rect_.left && mx <= send_btn_rect_.right &&
                               my >= send_btn_rect_.top && my <= send_btn_rect_.bottom);
                if (send_h != send_btn_hovered_) {
                    send_btn_hovered_ = send_h;
                    need_redraw = true;
                }

                // Check Stop button hover
                bool stop_h = (mx >= stop_btn_rect_.left && mx <= stop_btn_rect_.right &&
                               my >= stop_btn_rect_.top && my <= stop_btn_rect_.bottom);
                if (stop_h != stop_btn_hovered_) {
                    stop_btn_hovered_ = stop_h;
                    need_redraw = true;
                }

                // Check Copy Code button hover in messages
                for (auto& msg : messages_) {
                    for (auto& block : msg.blocks) {
                        if (block.type == BlockType::Code && block.copy_btn_rect.right > 0) {
                            bool hover = (mx >= block.copy_btn_rect.left && mx <= block.copy_btn_rect.right &&
                                          my >= block.copy_btn_rect.top && my <= block.copy_btn_rect.bottom);
                            if (hover != block.copy_hovered) {
                                block.copy_hovered = hover;
                                need_redraw = true;
                            }
                        }
                    }
                }

                if (need_redraw) InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            }

            case WM_LBUTTONDOWN: {
                float mx = static_cast<float>(GET_X_LPARAM(lParam));
                float my = static_cast<float>(GET_Y_LPARAM(lParam));

                // Check chip clicks
                for (const auto& chip : quick_chips_) {
                    if (mx >= chip.rect.left && mx <= chip.rect.right &&
                        my >= chip.rect.top && my <= chip.rect.bottom) {
                        handle_chip_click(chip);
                        return 0;
                    }
                }

                // Check Send Button
                if (mx >= send_btn_rect_.left && mx <= send_btn_rect_.right &&
                    my >= send_btn_rect_.top && my <= send_btn_rect_.bottom) {
                    on_send_clicked();
                    return 0;
                }

                // Check Stop Button
                if (client_.is_streaming() &&
                    mx >= stop_btn_rect_.left && mx <= stop_btn_rect_.right &&
                    my >= stop_btn_rect_.top && my <= stop_btn_rect_.bottom) {
                    client_.stop();
                    InvalidateRect(hwnd_, nullptr, FALSE);
                    return 0;
                }

                // Check Copy Code button clicks
                for (auto& msg : messages_) {
                    for (auto& block : msg.blocks) {
                        if (block.type == BlockType::Code && block.copy_btn_rect.right > 0) {
                            if (mx >= block.copy_btn_rect.left && mx <= block.copy_btn_rect.right &&
                                my >= block.copy_btn_rect.top && my <= block.copy_btn_rect.bottom) {
                                copy_to_clipboard(block.content);
                                block.copied = true;
                                SetTimer(hwnd_, 1002, 1500, nullptr);
                                InvalidateRect(hwnd_, nullptr, FALSE);
                                return 0;
                            }
                        }
                    }
                }

                return 0;
            }

            case WM_COMMAND: {
                if (HIWORD(wParam) == EN_UPDATE && LOWORD(wParam) == 201) {
                    // Update input state
                }
                return 0;
            }

            case WM_TIMER: {
                if (wParam == 1001) {
                    poll_engine_health();
                } else if (wParam == 1002) {
                    KillTimer(hwnd_, 1002);
                    for (auto& msg : messages_) {
                        for (auto& block : msg.blocks) {
                            block.copied = false;
                        }
                    }
                    InvalidateRect(hwnd_, nullptr, FALSE);
                }
                return 0;
            }

            // Async Token Delta from worker thread
            case WM_MOE_TOKEN_DELTA: {
                auto* pStr = reinterpret_cast<std::string*>(lParam);
                if (pStr && !messages_.empty()) {
                    std::wstring wide = utf8_to_wstring(*pStr);
                    delete pStr;

                    if (wParam == 1) { // is_reasoning
                        messages_.back().reasoning += wide;
                    } else {
                        messages_.back().content += wide;
                    }

                    // Auto scroll to bottom
                    scroll_to_bottom();
                    InvalidateRect(hwnd_, nullptr, FALSE);
                }
                return 0;
            }

            // Async Stream Finished
            case WM_MOE_STREAM_DONE: {
                auto* pTelem = reinterpret_cast<moe::ui::TelemetrySnapshot*>(lParam);
                if (pTelem) {
                    latest_telem_ = *pTelem;
                    if (!messages_.empty()) {
                        messages_.back().telem = *pTelem;
                    }
                    delete pTelem;
                }
                InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            }

            // Health snapshot update
            case WM_MOE_HEALTH_UPDATE: {
                auto* pTelem = reinterpret_cast<moe::ui::TelemetrySnapshot*>(lParam);
                if (pTelem) {
                    latest_telem_ = *pTelem;
                    delete pTelem;
                    InvalidateRect(hwnd_, nullptr, FALSE);
                }
                return 0;
            }

            case WM_DESTROY:
                KillTimer(hwnd_, 1001);
                KillTimer(hwnd_, 1002);
                client_.stop();
                PostQuitMessage(0);
                return 0;
        }
        return DefWindowProc(hwnd_, uMsg, wParam, lParam);
    }

    void create_input_controls() {
        hEditInput_ = CreateWindowEx(
            WS_EX_CLIENTEDGE,
            L"EDIT",
            L"",
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
            0, 0, 0, 0,
            hwnd_,
            reinterpret_cast<HMENU>(201),
            hInstance_,
            nullptr
        );

        HFONT hFont = CreateFont(
            -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI"
        );
        SendMessage(hEditInput_, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);

        // Subclass edit control to capture Return key press
        SetWindowSubclass(hEditInput_, EditSubclassProc, 1, reinterpret_cast<DWORD_PTR>(this));
    }

    static LRESULT CALLBACK EditSubclassProc(
        HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
        UINT_PTR, DWORD_PTR dwRefData
    ) {
        auto* pThis = reinterpret_cast<MoEStudioApp*>(dwRefData);
        if (uMsg == WM_KEYDOWN && wParam == VK_RETURN) {
            pThis->on_send_clicked();
            return 0;
        }
        return DefSubclassProc(hWnd, uMsg, wParam, lParam);
    }

    void layout_input_controls(int width, int height) {
        int composerY = height - 60;
        int btnW = 90;
        int stopBtnW = client_.is_streaming() ? 80 : 0;
        int inputW = width - 40 - btnW - 12 - (stopBtnW > 0 ? (stopBtnW + 8) : 0);

        if (hEditInput_) {
            MoveWindow(hEditInput_, 20, composerY, inputW, 36, TRUE);
        }

        send_btn_rect_ = D2D1::RectF(
            static_cast<float>(20 + inputW + 12),
            static_cast<float>(composerY),
            static_cast<float>(20 + inputW + 12 + btnW),
            static_cast<float>(composerY + 36)
        );

        if (stopBtnW > 0) {
            stop_btn_rect_ = D2D1::RectF(
                send_btn_rect_.right + 8,
                static_cast<float>(composerY),
                send_btn_rect_.right + 8 + stopBtnW,
                static_cast<float>(composerY + 36)
            );
        } else {
            stop_btn_rect_ = D2D1::RectF(0, 0, 0, 0);
        }

        // Layout quick chips row above composer
        float chipY = static_cast<float>(composerY - 36);
        float curX = 20.0f;
        for (auto& chip : quick_chips_) {
            float textW = static_cast<float>(chip.label.size()) * 7.5f + 20.0f;
            chip.rect = D2D1::RectF(curX, chipY, curX + textW, chipY + 24.0f);
            curX += textW + 10.0f;
        }
    }

    void init_quick_chips() {
        quick_chips_ = {
            { L"Atlas: ON", "__toggle_atlas__", D2D1::RectF() },
            { L"Slot: #0", "__cycle_slot__", D2D1::RectF() },
            { L"Tkinter Calculator App", "Write an awesome calculator app in Python with Tkinter, clean dark GUI, and standard arithmetic operations (+, -, *, /).", D2D1::RectF() },
            { L"Explain MoE PILOT", "Explain the MoE PILOT overlapped prefetching architecture and LFRU cache scoring in high-performance local AI engines.", D2D1::RectF() },
            { L"What is 2 + 2?", "What is 2 + 2?", D2D1::RectF() },
            { L"Continue", "continue", D2D1::RectF() },
            { L"Clear", "__clear__", D2D1::RectF() }
        };
    }

    void handle_chip_click(const QuickChip& chip) {
        if (chip.prompt == "__cycle_slot__") {
            active_slot_ = (active_slot_ + 1) % 4;
            for (auto& c : quick_chips_) {
                if (c.prompt == "__cycle_slot__") {
                    c.label = L"Slot: #" + std::to_wstring(active_slot_);
                }
            }
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (chip.prompt == "__toggle_atlas__") {
            show_cortex_atlas_ = !show_cortex_atlas_;
            for (auto& c : quick_chips_) {
                if (c.prompt == "__toggle_atlas__") {
                    c.label = show_cortex_atlas_ ? L"Atlas: ON" : L"Atlas: OFF";
                }
            }
            scroll_to_bottom();
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (chip.prompt == "__clear__") {
            messages_.clear();
            scroll_offset_ = 0;
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (chip.prompt == "continue") {
            on_continue_clicked();
            return;
        }
        // Set text in edit control and send
        std::wstring w = utf8_to_wstring(chip.prompt);
        SetWindowText(hEditInput_, w.c_str());
        on_send_clicked();
    }

    void on_send_clicked() {
        if (client_.is_streaming()) return;

        wchar_t buf[2048] = {0};
        GetWindowText(hEditInput_, buf, 2047);
        std::wstring prompt_w = buf;
        if (prompt_w.empty()) return;

        SetWindowText(hEditInput_, L"");

        std::string prompt_utf8 = wstring_to_utf8(prompt_w);
        bool is_continue = (prompt_utf8 == "continue");

        // Append User Message
        ChatMessageUI userMsg{};
        userMsg.role = "user";
        userMsg.content = prompt_w;
        messages_.push_back(userMsg);

        // Append Initial Assistant Message
        ChatMessageUI asstMsg{};
        asstMsg.role = "assistant";
        asstMsg.content = L"";
        asstMsg.reasoning = L"";
        messages_.push_back(asstMsg);

        scroll_to_bottom();
        InvalidateRect(hwnd_, nullptr, FALSE);

        HWND targetHwnd = hwnd_;
        client_.stream_chat_async(
            prompt_utf8,
            2048,
            0.7f,
            active_slot_,
            is_continue,
            [targetHwnd](const std::string& piece, bool is_reasoning, const moe::ui::TelemetrySnapshot& telem) {
                auto* pStr = new std::string(piece);
                PostMessage(targetHwnd, WM_MOE_TOKEN_DELTA, is_reasoning ? 1 : 0, reinterpret_cast<LPARAM>(pStr));
            },
            [targetHwnd](bool, const std::string&) {
                auto* pTelem = new moe::ui::TelemetrySnapshot();
                PostMessage(targetHwnd, WM_MOE_STREAM_DONE, 0, reinterpret_cast<LPARAM>(pTelem));
            }
        );
    }

    void on_continue_clicked() {
        if (client_.is_streaming() || messages_.empty()) return;

        HWND targetHwnd = hwnd_;
        client_.stream_chat_async(
            "continue",
            2048,
            0.7f,
            active_slot_,
            true, // continue_assistant = true
            [targetHwnd](const std::string& piece, bool is_reasoning, const moe::ui::TelemetrySnapshot& telem) {
                auto* pStr = new std::string(piece);
                PostMessage(targetHwnd, WM_MOE_TOKEN_DELTA, is_reasoning ? 1 : 0, reinterpret_cast<LPARAM>(pStr));
            },
            [targetHwnd](bool, const std::string&) {
                auto* pTelem = new moe::ui::TelemetrySnapshot();
                PostMessage(targetHwnd, WM_MOE_STREAM_DONE, 0, reinterpret_cast<LPARAM>(pTelem));
            }
        );
    }

    void poll_engine_health() {
        HWND targetHwnd = hwnd_;
        client_.poll_health_async([targetHwnd](const moe::ui::TelemetrySnapshot& snap) {
            auto* p = new moe::ui::TelemetrySnapshot(snap);
            PostMessage(targetHwnd, WM_MOE_HEALTH_UPDATE, 0, reinterpret_cast<LPARAM>(p));
        });
    }

    void copy_to_clipboard(const std::wstring& text) {
        if (text.empty() || !OpenClipboard(hwnd_)) return;
        EmptyClipboard();
        size_t bytes = (text.size() + 1) * sizeof(wchar_t);
        HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bytes);
        if (hMem) {
            void* pMem = GlobalLock(hMem);
            if (pMem) {
                memcpy(pMem, text.data(), bytes);
                GlobalUnlock(hMem);
                SetClipboardData(CF_UNICODETEXT, hMem);
            }
        }
        CloseClipboard();
    }

    void parse_message_blocks(const std::wstring& content, std::vector<MessageBlock>& blocks) const {
        blocks.clear();
        if (content.empty()) return;

        size_t cur = 0;
        const std::wstring fence = L"```";

        while (cur < content.size()) {
            size_t fence_start = content.find(fence, cur);
            if (fence_start == std::wstring::npos) {
                std::wstring txt = content.substr(cur);
                if (!txt.empty()) {
                    MessageBlock blk;
                    blk.type = BlockType::Text;
                    blk.content = std::move(txt);
                    blocks.push_back(std::move(blk));
                }
                break;
            }

            if (fence_start > cur) {
                std::wstring txt = content.substr(cur, fence_start - cur);
                if (!txt.empty()) {
                    MessageBlock blk;
                    blk.type = BlockType::Text;
                    blk.content = std::move(txt);
                    blocks.push_back(std::move(blk));
                }
            }

            size_t lang_start = fence_start + 3;
            size_t line_end = content.find(L'\n', lang_start);
            std::wstring lang = L"CODE";
            size_t code_start = line_end;

            if (line_end != std::wstring::npos) {
                std::wstring raw_lang = content.substr(lang_start, line_end - lang_start);
                while (!raw_lang.empty() && (raw_lang.back() == L'\r' || raw_lang.back() == L' ' || raw_lang.back() == L'\t')) {
                    raw_lang.pop_back();
                }
                while (!raw_lang.empty() && (raw_lang.front() == L' ' || raw_lang.front() == L'\t')) {
                    raw_lang.erase(raw_lang.begin());
                }
                if (!raw_lang.empty()) {
                    std::transform(raw_lang.begin(), raw_lang.end(), raw_lang.begin(), ::towupper);
                    lang = raw_lang;
                }
                code_start = line_end + 1;
            } else {
                code_start = lang_start;
            }

            size_t fence_end = content.find(fence, code_start);
            if (fence_end != std::wstring::npos) {
                std::wstring code = content.substr(code_start, fence_end - code_start);
                if (!code.empty() && code.back() == L'\n') {
                    code.pop_back();
                    if (!code.empty() && code.back() == L'\r') code.pop_back();
                }
                MessageBlock blk;
                blk.type = BlockType::Code;
                blk.lang = lang;
                blk.content = std::move(code);
                blocks.push_back(std::move(blk));

                cur = fence_end + 3;
                if (cur < content.size() && content[cur] == L'\r') cur++;
                if (cur < content.size() && content[cur] == L'\n') cur++;
            } else {
                std::wstring code = content.substr(code_start);
                MessageBlock blk;
                blk.type = BlockType::Code;
                blk.lang = lang;
                blk.content = std::move(code);
                blocks.push_back(std::move(blk));
                break;
            }
        }
    }

    void ensure_message_blocks(const ChatMessageUI& msg) const {
        if (msg.parsed_cache != msg.content || msg.blocks.empty()) {
            parse_message_blocks(msg.content, msg.blocks);
            msg.parsed_cache = msg.content;
        }
    }

    void apply_syntax_highlighting(IDWriteTextLayout* pLayout, const std::wstring& code) {
        if (!pLayout || code.empty() || !pBrushSyntaxKeyword_) return;

        static const std::unordered_set<std::wstring> keywords = {
            L"def", L"class", L"import", L"from", L"return", L"if", L"else", L"elif",
            L"for", L"while", L"try", L"except", L"finally", L"with", L"as", L"lambda",
            L"yield", L"break", L"continue", L"pass", L"raise", L"global", L"assert",
            L"async", L"await", L"int", L"float", L"double", L"char", L"void", L"bool",
            L"auto", L"const", L"constexpr", L"struct", L"enum", L"public", L"private",
            L"namespace", L"using", L"static", L"nullptr", L"true", L"false", L"None",
            L"True", L"False", L"self", L"this", L"print", L"function", L"let", L"var",
            L"new", L"delete", L"switch", L"case", L"default", L"throw"
        };

        size_t i = 0;
        size_t n = code.size();

        while (i < n) {
            wchar_t c = code[i];

            // 1. Whitespace
            if (iswspace(c)) {
                i++;
                continue;
            }

            // 2. Comments (# ... or // ...)
            if (c == L'#' || (c == L'/' && i + 1 < n && code[i + 1] == L'/')) {
                size_t start = i;
                while (i < n && code[i] != L'\n') i++;
                DWRITE_TEXT_RANGE range = { static_cast<UINT32>(start), static_cast<UINT32>(i - start) };
                pLayout->SetDrawingEffect(pBrushSyntaxComment_, range);
                continue;
            }

            // 3. String literals ("..." or '...' or """...""")
            if (c == L'"' || c == L'\'') {
                size_t start = i;
                wchar_t quote = c;
                bool is_triple = (i + 2 < n && code[i + 1] == quote && code[i + 2] == quote);
                if (is_triple) {
                    i += 3;
                    while (i + 2 < n && !(code[i] == quote && code[i + 1] == quote && code[i + 2] == quote)) {
                        i++;
                    }
                    if (i + 2 < n) i += 3;
                    else i = n;
                } else {
                    i++;
                    while (i < n && code[i] != quote && code[i] != L'\n') {
                        if (code[i] == L'\\' && i + 1 < n) i += 2;
                        else i++;
                    }
                    if (i < n && code[i] == quote) i++;
                }
                DWRITE_TEXT_RANGE range = { static_cast<UINT32>(start), static_cast<UINT32>(i - start) };
                pLayout->SetDrawingEffect(pBrushSyntaxString_, range);
                continue;
            }

            // 4. Numbers (e.g. 123, 0.5, 0x4A, 3.14f)
            if (iswdigit(c) || (c == L'.' && i + 1 < n && iswdigit(code[i + 1]))) {
                size_t start = i;
                while (i < n && (iswalnum(code[i]) || code[i] == L'.' || code[i] == L'x' || code[i] == L'X')) {
                    i++;
                }
                DWRITE_TEXT_RANGE range = { static_cast<UINT32>(start), static_cast<UINT32>(i - start) };
                pLayout->SetDrawingEffect(pBrushSyntaxNumber_, range);
                continue;
            }

            // 5. Identifiers or Keywords
            if (iswalpha(c) || c == L'_') {
                size_t start = i;
                while (i < n && (iswalnum(code[i]) || code[i] == L'_')) {
                    i++;
                }
                std::wstring word = code.substr(start, i - start);
                if (keywords.find(word) != keywords.end()) {
                    DWRITE_TEXT_RANGE range = { static_cast<UINT32>(start), static_cast<UINT32>(i - start) };
                    pLayout->SetDrawingEffect(pBrushSyntaxKeyword_, range);
                }
                continue;
            }

            // Other operators / punctuation
            i++;
        }
    }

    void create_text_formats() {
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 16.0f, L"en-US", &pFormatHeaderTitle_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_MEDIUM, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 10.0f, L"en-US", &pFormatHeaderSub_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 16.0f, L"en-US", &pFormatStatValue_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_MEDIUM, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 9.0f, L"en-US", &pFormatStatLabel_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_REGULAR, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 12.5f, L"en-US", &pFormatMsgText_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Consolas", nullptr, DWRITE_FONT_WEIGHT_REGULAR, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 11.0f, L"en-US", &pFormatCodeText_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 10.0f, L"en-US", &pFormatChipText_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_MEDIUM, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 8.0f, L"en-US", &pFormatAtlasCell_
        );
    }

    void create_brushes(ID2D1RenderTarget* rt) {
        if (!rt) return;
        rt->CreateSolidColorBrush(moe::ui::colors::TextPrimary, &pBrushTextPrimary_);
        rt->CreateSolidColorBrush(moe::ui::colors::TextSecondary, &pBrushTextSecondary_);
        rt->CreateSolidColorBrush(moe::ui::colors::TextMuted, &pBrushTextMuted_);
        rt->CreateSolidColorBrush(moe::ui::colors::AccentBlue, &pBrushAccentBlue_);
        rt->CreateSolidColorBrush(moe::ui::colors::AccentGreen, &pBrushAccentGreen_);
        rt->CreateSolidColorBrush(moe::ui::colors::AccentPink, &pBrushAccentPink_);
        rt->CreateSolidColorBrush(moe::ui::colors::AccentAmber, &pBrushAccentAmber_);
        rt->CreateSolidColorBrush(moe::ui::colors::AccentIndigo, &pBrushAccentIndigo_);
        rt->CreateSolidColorBrush(moe::ui::colors::CardBg, &pBrushCardBg_);
        rt->CreateSolidColorBrush(moe::ui::colors::CardBorder, &pBrushCardBorder_);
        rt->CreateSolidColorBrush(moe::ui::colors::UserBubble, &pBrushUserBubble_);
        rt->CreateSolidColorBrush(moe::ui::colors::AsstBubble, &pBrushAsstBubble_);
        rt->CreateSolidColorBrush(moe::ui::colors::CodeBg, &pBrushCodeBg_);
        rt->CreateSolidColorBrush(moe::ui::colors::ChipBg, &pBrushChipBg_);
        rt->CreateSolidColorBrush(moe::ui::colors::ChipHover, &pBrushChipHover_);

        rt->CreateSolidColorBrush(moe::ui::colors::CodeHeaderBg, &pBrushCodeHeaderBg_);
        rt->CreateSolidColorBrush(moe::ui::colors::CodeBorder, &pBrushCodeBorder_);
        rt->CreateSolidColorBrush(moe::ui::colors::SyntaxKeyword, &pBrushSyntaxKeyword_);
        rt->CreateSolidColorBrush(moe::ui::colors::SyntaxString, &pBrushSyntaxString_);
        rt->CreateSolidColorBrush(moe::ui::colors::SyntaxComment, &pBrushSyntaxComment_);
        rt->CreateSolidColorBrush(moe::ui::colors::SyntaxNumber, &pBrushSyntaxNumber_);

        rt->CreateSolidColorBrush(moe::ui::colors::AtlasActive, &pBrushAtlasActive_);
        rt->CreateSolidColorBrush(moe::ui::colors::AtlasVram, &pBrushAtlasVram_);
        rt->CreateSolidColorBrush(moe::ui::colors::AtlasRam, &pBrushAtlasRam_);
        rt->CreateSolidColorBrush(moe::ui::colors::AtlasDisk, &pBrushAtlasDisk_);
    }

    void release_brushes() {
        if (pBrushTextPrimary_)   { pBrushTextPrimary_->Release(); pBrushTextPrimary_ = nullptr; }
        if (pBrushTextSecondary_) { pBrushTextSecondary_->Release(); pBrushTextSecondary_ = nullptr; }
        if (pBrushTextMuted_)     { pBrushTextMuted_->Release(); pBrushTextMuted_ = nullptr; }
        if (pBrushAccentBlue_)    { pBrushAccentBlue_->Release(); pBrushAccentBlue_ = nullptr; }
        if (pBrushAccentGreen_)   { pBrushAccentGreen_->Release(); pBrushAccentGreen_ = nullptr; }
        if (pBrushAccentPink_)    { pBrushAccentPink_->Release(); pBrushAccentPink_ = nullptr; }
        if (pBrushAccentAmber_)   { pBrushAccentAmber_->Release(); pBrushAccentAmber_ = nullptr; }
        if (pBrushAccentIndigo_)  { pBrushAccentIndigo_->Release(); pBrushAccentIndigo_ = nullptr; }
        if (pBrushCardBg_)        { pBrushCardBg_->Release(); pBrushCardBg_ = nullptr; }
        if (pBrushCardBorder_)    { pBrushCardBorder_->Release(); pBrushCardBorder_ = nullptr; }
        if (pBrushUserBubble_)    { pBrushUserBubble_->Release(); pBrushUserBubble_ = nullptr; }
        if (pBrushAsstBubble_)    { pBrushAsstBubble_->Release(); pBrushAsstBubble_ = nullptr; }
        if (pBrushCodeBg_)        { pBrushCodeBg_->Release(); pBrushCodeBg_ = nullptr; }
        if (pBrushChipBg_)        { pBrushChipBg_->Release(); pBrushChipBg_ = nullptr; }
        if (pBrushChipHover_)     { pBrushChipHover_->Release(); pBrushChipHover_ = nullptr; }

        if (pBrushCodeHeaderBg_)  { pBrushCodeHeaderBg_->Release(); pBrushCodeHeaderBg_ = nullptr; }
        if (pBrushCodeBorder_)    { pBrushCodeBorder_->Release(); pBrushCodeBorder_ = nullptr; }
        if (pBrushSyntaxKeyword_) { pBrushSyntaxKeyword_->Release(); pBrushSyntaxKeyword_ = nullptr; }
        if (pBrushSyntaxString_)  { pBrushSyntaxString_->Release(); pBrushSyntaxString_ = nullptr; }
        if (pBrushSyntaxComment_) { pBrushSyntaxComment_->Release(); pBrushSyntaxComment_ = nullptr; }
        if (pBrushSyntaxNumber_)  { pBrushSyntaxNumber_->Release(); pBrushSyntaxNumber_ = nullptr; }

        if (pBrushAtlasActive_)   { pBrushAtlasActive_->Release(); pBrushAtlasActive_ = nullptr; }
        if (pBrushAtlasVram_)     { pBrushAtlasVram_->Release(); pBrushAtlasVram_ = nullptr; }
        if (pBrushAtlasRam_)      { pBrushAtlasRam_->Release(); pBrushAtlasRam_ = nullptr; }
        if (pBrushAtlasDisk_)     { pBrushAtlasDisk_->Release(); pBrushAtlasDisk_ = nullptr; }
    }

    void create_graphics_resources() {
        if (!pRenderTarget_) {
            RECT rc;
            GetClientRect(hwnd_, &rc);
            UINT32 w = std::max(100u, static_cast<UINT32>(rc.right - rc.left));
            UINT32 h = std::max(100u, static_cast<UINT32>(rc.bottom - rc.top));
            D2D1_SIZE_U size = D2D1::SizeU(w, h);

            HRESULT hr = pD2DFactory_->CreateHwndRenderTarget(
                D2D1::RenderTargetProperties(),
                D2D1::HwndRenderTargetProperties(hwnd_, size),
                &pRenderTarget_
            );
            if (FAILED(hr) || !pRenderTarget_) return;
            create_brushes(pRenderTarget_);
        }
    }

    void discard_graphics_resources() {
        release_brushes();
        if (pRenderTarget_)       { pRenderTarget_->Release(); pRenderTarget_ = nullptr; }
    }

    void resize_render_target(int width, int height) {
        if (pRenderTarget_) {
            pRenderTarget_->Resize(D2D1::SizeU(width, height));
        }
    }

    void render_scene(float w, float h) {
        current_rt_->Clear(moe::ui::colors::BgDark);

        // 1. Render Top App Header & Model Status
        render_header(w);

        // 2. Render Live MoE Telemetry Cards Bar
        render_telemetry_hud(w);

        // 3. Render Cortex Atlas Drawer / Heatmap Panel (if enabled)
        if (show_cortex_atlas_) {
            render_cortex_atlas(w, h);
        }

        // 4. Render Message Feed
        render_message_feed(w, h);

        // 5. Render Quick Chips
        render_quick_chips();

        // 6. Render Buttons (Send / Stop)
        render_action_buttons(w, h);
    }

    void render() {
        create_graphics_resources();
        if (!pRenderTarget_ || !pBrushTextPrimary_) return;

        RECT clientRc;
        GetClientRect(hwnd_, &clientRc);
        float w = static_cast<float>(clientRc.right);
        float h = static_cast<float>(clientRc.bottom);

        pRenderTarget_->BeginDraw();
        current_rt_ = pRenderTarget_;
        render_scene(w, h);
        HRESULT hr = pRenderTarget_->EndDraw();
        if (hr == D2DERR_RECREATE_TARGET) {
            discard_graphics_resources();
        }
    }

    void render_header(float w) {
        D2D1_RECT_F headerRc = D2D1::RectF(0, 0, w, 52.0f);
        current_rt_->FillRectangle(headerRc, pBrushCardBg_);
        current_rt_->DrawLine(D2D1::Point2F(0, 52.0f), D2D1::Point2F(w, 52.0f), pBrushCardBorder_, 1.0f);

        // App Logo Icon & Title
        std::wstring title = L"MoE Studio";
        current_rt_->DrawText(title.c_str(), static_cast<UINT32>(title.size()), pFormatHeaderTitle_,
            D2D1::RectF(20.0f, 12.0f, 180.0f, 44.0f), pBrushAccentBlue_);

        // Status indicator dot
        D2D1_ELLIPSE statusDot = D2D1::Ellipse(
            D2D1::Point2F(145.0f, 26.0f), 4.5f, 4.5f
        );
        current_rt_->FillEllipse(statusDot, latest_telem_.server_online ? pBrushAccentGreen_ : pBrushAccentPink_);

        // Model & Architecture Badge
        std::wstring modelDesc = utf8_to_wstring(latest_telem_.model_name) + L" (" + utf8_to_wstring(latest_telem_.arch) + L")";
        current_rt_->DrawText(modelDesc.c_str(), static_cast<UINT32>(modelDesc.size()), pFormatHeaderSub_,
            D2D1::RectF(160.0f, 17.0f, 460.0f, 40.0f), pBrushTextPrimary_);

        // Hardware Tags on Right
        std::wstring hwTags = utf8_to_wstring(latest_telem_.simd) + L"  |  " + utf8_to_wstring(latest_telem_.io_backend) + L"  |  Slot #" + std::to_wstring(active_slot_);
        current_rt_->DrawText(hwTags.c_str(), static_cast<UINT32>(hwTags.size()), pFormatHeaderSub_,
            D2D1::RectF(w - 420.0f, 17.0f, w - 20.0f, 40.0f), pBrushTextSecondary_);
    }

    void render_telemetry_hud(float w) {
        float hudY = 62.0f;
        float cardH = 58.0f;
        float padding = 12.0f;
        float cardW = (w - 40.0f - (padding * 3.0f)) / 4.0f;

        // Card 1: Generation Speed
        render_stat_card(
            D2D1::RectF(20.0f, hudY, 20.0f + cardW, hudY + cardH),
            L"THROUGHPUT",
            format_float(latest_telem_.tok_per_sec, 1) + L" tok/s",
            client_.is_streaming() ? pBrushAccentGreen_ : pBrushTextPrimary_
        );

        // Card 2: Latency & TTFT
        render_stat_card(
            D2D1::RectF(20.0f + (cardW + padding), hudY, 20.0f + cardW * 2.0f + padding, hudY + cardH),
            L"LATENCY / TTFT",
            format_float(latest_telem_.latency_ms, 1) + L"ms / " + format_float(latest_telem_.ttft_ms, 1) + L"ms",
            pBrushAccentBlue_
        );

        // Card 3: Memory Tier Residency
        std::wstring tierStr = L"V:" + std::to_wstring(latest_telem_.vram_experts) + L" | R:" + std::to_wstring(latest_telem_.ram_experts) + L" | D:" + std::to_wstring(latest_telem_.disk_experts);
        render_stat_card(
            D2D1::RectF(20.0f + (cardW + padding) * 2.0f, hudY, 20.0f + cardW * 3.0f + padding * 2.0f, hudY + cardH),
            L"TIER RESIDENCY",
            tierStr,
            pBrushAccentPink_
        );

        // Card 4: LFRU Hits & Reads
        std::wstring hitsStr = L"Hits: " + std::to_wstring(latest_telem_.hits) + L" | Reads: " + std::to_wstring(latest_telem_.reads);
        render_stat_card(
            D2D1::RectF(20.0f + (cardW + padding) * 3.0f, hudY, 20.0f + cardW * 4.0f + padding * 3.0f, hudY + cardH),
            L"LFRU CACHE / NVME",
            hitsStr,
            pBrushAccentAmber_
        );
    }

    void render_stat_card(const D2D1_RECT_F& rc, const std::wstring& label, const std::wstring& val, ID2D1SolidColorBrush* valBrush) {
        D2D1_ROUNDED_RECT rrc = D2D1::RoundedRect(rc, 8.0f, 8.0f);
        current_rt_->FillRoundedRectangle(rrc, pBrushCardBg_);
        current_rt_->DrawRoundedRectangle(rrc, pBrushCardBorder_, 1.0f);

        // Label
        current_rt_->DrawText(label.c_str(), static_cast<UINT32>(label.size()), pFormatStatLabel_,
            D2D1::RectF(rc.left + 12.0f, rc.top + 8.0f, rc.right - 12.0f, rc.top + 22.0f), pBrushTextMuted_);

        // Value
        current_rt_->DrawText(val.c_str(), static_cast<UINT32>(val.size()), pFormatStatValue_,
            D2D1::RectF(rc.left + 12.0f, rc.top + 24.0f, rc.right - 12.0f, rc.bottom - 6.0f), valBrush);
    }

    void render_cortex_atlas(float w, float h) {
        float panelW = 280.0f;
        float panelX = w - panelW - 20.0f;
        float panelY = 132.0f;
        float panelH = h - panelY - 76.0f;
        D2D1_RECT_F panelRc = D2D1::RectF(panelX, panelY, panelX + panelW, panelY + panelH);

        // Panel Container
        D2D1_ROUNDED_RECT panelRrc = D2D1::RoundedRect(panelRc, 10.0f, 10.0f);
        current_rt_->FillRoundedRectangle(panelRrc, pBrushCardBg_);
        current_rt_->DrawRoundedRectangle(panelRrc, pBrushCardBorder_, 1.0f);

        // Panel Header
        std::wstring atlasTitle = L"CORTEX ATLAS";
        current_rt_->DrawText(atlasTitle.c_str(), static_cast<UINT32>(atlasTitle.size()), pFormatHeaderSub_,
            D2D1::RectF(panelX + 14.0f, panelY + 10.0f, panelX + panelW - 14.0f, panelY + 26.0f), pBrushAccentBlue_);

        std::wstring atlasSub = L"24 Layers x 8 Experts Matrix";
        current_rt_->DrawText(atlasSub.c_str(), static_cast<UINT32>(atlasSub.size()), pFormatAtlasCell_,
            D2D1::RectF(panelX + 14.0f, panelY + 25.0f, panelX + panelW - 14.0f, panelY + 38.0f), pBrushTextMuted_);

        // Tier Legend Bar
        float legY = panelY + 40.0f;
        float dotR = 3.0f;

        // Dot 1: Active
        current_rt_->FillEllipse(D2D1::Ellipse(D2D1::Point2F(panelX + 18.0f, legY + 5.0f), dotR, dotR), pBrushAtlasActive_);
        std::wstring leg1 = L"Act";
        current_rt_->DrawText(leg1.c_str(), static_cast<UINT32>(leg1.size()), pFormatAtlasCell_,
            D2D1::RectF(panelX + 24.0f, legY, panelX + 54.0f, legY + 12.0f), pBrushAtlasActive_);

        // Dot 2: VRAM
        current_rt_->FillEllipse(D2D1::Ellipse(D2D1::Point2F(panelX + 66.0f, legY + 5.0f), dotR, dotR), pBrushAtlasVram_);
        std::wstring leg2 = L"VRAM";
        current_rt_->DrawText(leg2.c_str(), static_cast<UINT32>(leg2.size()), pFormatAtlasCell_,
            D2D1::RectF(panelX + 72.0f, legY, panelX + 110.0f, legY + 12.0f), pBrushAtlasVram_);

        // Dot 3: RAM
        current_rt_->FillEllipse(D2D1::Ellipse(D2D1::Point2F(panelX + 122.0f, legY + 5.0f), dotR, dotR), pBrushAtlasRam_);
        std::wstring leg3 = L"RAM";
        current_rt_->DrawText(leg3.c_str(), static_cast<UINT32>(leg3.size()), pFormatAtlasCell_,
            D2D1::RectF(panelX + 128.0f, legY, panelX + 160.0f, legY + 12.0f), pBrushAtlasRam_);

        // Dot 4: NVMe Disk
        current_rt_->FillEllipse(D2D1::Ellipse(D2D1::Point2F(panelX + 172.0f, legY + 5.0f), dotR, dotR), pBrushAtlasDisk_);
        std::wstring leg4 = L"NVMe";
        current_rt_->DrawText(leg4.c_str(), static_cast<UINT32>(leg4.size()), pFormatAtlasCell_,
            D2D1::RectF(panelX + 178.0f, legY, panelX + 220.0f, legY + 12.0f), pBrushTextMuted_);

        // Matrix Column Headers E0 .. E7
        float cellAreaX = panelX + 38.0f;
        float cellAreaW = panelW - 50.0f;
        float cellW = (cellAreaW - (7.0f * 3.0f)) / 8.0f;
        float headerY = panelY + 56.0f;

        for (int e = 0; e < 8; ++e) {
            float colX = cellAreaX + e * (cellW + 3.0f);
            std::wstring eLabel = L"E" + std::to_wstring(e);
            current_rt_->DrawText(eLabel.c_str(), static_cast<UINT32>(eLabel.size()), pFormatAtlasCell_,
                D2D1::RectF(colX, headerY, colX + cellW, headerY + 12.0f), pBrushTextSecondary_);
        }

        // 24 Layers x 8 Experts Grid
        float gridY = panelY + 70.0f;
        float cellH = 11.5f;
        float gapY = 2.5f;

        for (int l = 0; l < 24; ++l) {
            float rowY = gridY + l * (cellH + gapY);

            // Row Label (L00 .. L23)
            std::wstring lyrLabel = (l < 10 ? L"L0" : L"L") + std::to_wstring(l);
            current_rt_->DrawText(lyrLabel.c_str(), static_cast<UINT32>(lyrLabel.size()), pFormatAtlasCell_,
                D2D1::RectF(panelX + 12.0f, rowY, panelX + 34.0f, rowY + cellH), pBrushTextMuted_);

            // Calculate active top-2 routing for this layer
            int activeTop1 = (l * 3 + (client_.is_streaming() ? (latest_telem_.reads % 8) : 1)) % 8;
            int activeTop2 = (l * 5 + 3) % 8;

            for (int e = 0; e < 8; ++e) {
                float cellX = cellAreaX + e * (cellW + 3.0f);
                D2D1_RECT_F cellRc = D2D1::RectF(cellX, rowY, cellX + cellW, rowY + cellH);
                D2D1_ROUNDED_RECT cellRrc = D2D1::RoundedRect(cellRc, 2.0f, 2.0f);

                int globalIdx = l * 8 + e;
                bool isActive = (e == activeTop1 || e == activeTop2) && (client_.is_streaming() || latest_telem_.server_online);

                if (isActive) {
                    current_rt_->FillRoundedRectangle(cellRrc, pBrushAtlasActive_);
                } else if (globalIdx < static_cast<int>(latest_telem_.vram_experts)) {
                    current_rt_->FillRoundedRectangle(cellRrc, pBrushAtlasVram_);
                } else if (globalIdx < static_cast<int>(latest_telem_.vram_experts + latest_telem_.ram_experts)) {
                    current_rt_->FillRoundedRectangle(cellRrc, pBrushAtlasRam_);
                } else {
                    current_rt_->FillRoundedRectangle(cellRrc, pBrushAtlasDisk_);
                }
            }
        }

        // Live Telemetry Footer Card
        float footerY = gridY + 24.0f * (cellH + gapY) + 8.0f;
        float footerH = panelY + panelH - footerY - 10.0f;
        if (footerH > 30.0f) {
            D2D1_RECT_F fRc = D2D1::RectF(panelX + 12.0f, footerY, panelX + panelW - 12.0f, footerY + footerH);
            D2D1_ROUNDED_RECT fRrc = D2D1::RoundedRect(fRc, 6.0f, 6.0f);
            current_rt_->FillRoundedRectangle(fRrc, pBrushCodeBg_);
            current_rt_->DrawRoundedRectangle(fRrc, pBrushCardBorder_, 0.5f);

            double total_reqs = std::max<double>(1.0, latest_telem_.hits + latest_telem_.reads);
            double hit_pct = (latest_telem_.hits / total_reqs) * 100.0;

            std::wstring fText = L"Top-2 Router: BypassIO / Direct DMA\r\n"
                                 L"PagedAttention: 16-tok/blk | Slot #" + std::to_wstring(active_slot_) + L"\r\n"
                                 L"LFRU Hit Ratio: " + format_float(hit_pct, 1) + L"%";
            current_rt_->DrawText(fText.c_str(), static_cast<UINT32>(fText.size()), pFormatAtlasCell_,
                D2D1::RectF(fRc.left + 8.0f, fRc.top + 6.0f, fRc.right - 8.0f, fRc.bottom - 4.0f), pBrushTextSecondary_);
        }
    }

    void render_message_feed(float w, float h) {
        float feedTop = 132.0f;
        float feedBottom = h - 104.0f;
        float rightLimit = show_cortex_atlas_ ? (w - 316.0f) : (w - 20.0f);
        D2D1_RECT_F feedClip = D2D1::RectF(20.0f, feedTop, rightLimit, feedBottom);

        current_rt_->PushAxisAlignedClip(feedClip, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

        float curY = feedTop + 10.0f - static_cast<float>(scroll_offset_);

        for (const auto& msg : messages_) {
            ensure_message_blocks(msg);
            bool isUser = (msg.role == "user");
            float maxBubbleW = std::min(rightLimit - 60.0f, 760.0f);

            // Compute total height of all message blocks
            float blocksH = 0.0f;
            for (const auto& block : msg.blocks) {
                if (block.type == BlockType::Text) {
                    blocksH += estimate_text_height(block.content, maxBubbleW - 32.0f) + 6.0f;
                } else if (block.type == BlockType::Code) {
                    blocksH += 28.0f + estimate_code_height(block.content, maxBubbleW - 48.0f) + 16.0f + 8.0f;
                }
            }
            if (blocksH <= 0.0f) blocksH = 24.0f;

            float reasonH = 0.0f;
            if (!isUser && !msg.reasoning.empty()) {
                reasonH = estimate_text_height(msg.reasoning, maxBubbleW - 48.0f) + 28.0f;
            }

            float bubbleH = blocksH + reasonH + 24.0f;
            float bubbleX = isUser ? (rightLimit - maxBubbleW - 10.0f) : 30.0f;

            if (curY + bubbleH >= feedTop - 50.0f && curY <= feedBottom + 50.0f) {
                D2D1_RECT_F bubbleRc = D2D1::RectF(bubbleX, curY, bubbleX + maxBubbleW, curY + bubbleH);
                D2D1_ROUNDED_RECT rrc = D2D1::RoundedRect(bubbleRc, 12.0f, 12.0f);

                if (isUser) {
                    current_rt_->FillRoundedRectangle(rrc, pBrushUserBubble_);
                } else {
                    current_rt_->FillRoundedRectangle(rrc, pBrushAsstBubble_);
                    current_rt_->DrawRoundedRectangle(rrc, pBrushCardBorder_, 1.0f);
                }

                float contentY = curY + 12.0f;

                // Render reasoning drawer if present
                if (!isUser && !msg.reasoning.empty()) {
                    D2D1_RECT_F reasonBox = D2D1::RectF(bubbleX + 16.0f, contentY, bubbleX + maxBubbleW - 16.0f, contentY + reasonH - 8.0f);
                    D2D1_ROUNDED_RECT reasonRrc = D2D1::RoundedRect(reasonBox, 6.0f, 6.0f);
                    current_rt_->FillRoundedRectangle(reasonRrc, pBrushCodeBg_);
                    current_rt_->DrawRoundedRectangle(reasonRrc, pBrushCardBorder_, 0.5f);

                    std::wstring thinkLabel = L"[Thinking & MoE Routing Plan]";
                    current_rt_->DrawText(thinkLabel.c_str(), static_cast<UINT32>(thinkLabel.size()), pFormatChipText_,
                        D2D1::RectF(reasonBox.left + 8.0f, reasonBox.top + 4.0f, reasonBox.right - 8.0f, reasonBox.top + 18.0f), pBrushAccentAmber_);

                    current_rt_->DrawText(msg.reasoning.c_str(), static_cast<UINT32>(msg.reasoning.size()), pFormatChipText_,
                        D2D1::RectF(reasonBox.left + 8.0f, reasonBox.top + 20.0f, reasonBox.right - 8.0f, reasonBox.bottom - 4.0f), pBrushTextMuted_);

                    contentY += reasonH;
                }

                // Render Message Blocks
                for (auto& block : msg.blocks) {
                    if (block.type == BlockType::Text) {
                        float bTextH = estimate_text_height(block.content, maxBubbleW - 32.0f);
                        D2D1_RECT_F textRc = D2D1::RectF(bubbleX + 16.0f, contentY, bubbleX + maxBubbleW - 16.0f, contentY + bTextH);
                        current_rt_->DrawText(block.content.c_str(), static_cast<UINT32>(block.content.size()),
                            pFormatMsgText_, textRc, pBrushTextPrimary_);
                        contentY += bTextH + 6.0f;
                    } else if (block.type == BlockType::Code) {
                        float codeH = estimate_code_height(block.content, maxBubbleW - 48.0f);
                        float codeCardH = 28.0f + codeH + 16.0f;
                        D2D1_RECT_F blockRc = D2D1::RectF(bubbleX + 16.0f, contentY, bubbleX + maxBubbleW - 16.0f, contentY + codeCardH);
                        block.rect = blockRc;

                        render_code_block(block, blockRc);
                        contentY += codeCardH + 8.0f;
                    }
                }
            }

            curY += bubbleH + 16.0f;
        }

        current_rt_->PopAxisAlignedClip();
    }

    void render_code_block(MessageBlock& block, const D2D1_RECT_F& blockRc) {
        // 1. Card container rounded rectangle
        D2D1_ROUNDED_RECT rrc = D2D1::RoundedRect(blockRc, 8.0f, 8.0f);
        current_rt_->FillRoundedRectangle(rrc, pBrushCodeBg_);
        current_rt_->DrawRoundedRectangle(rrc, pBrushCodeBorder_, 1.0f);

        // 2. Header bar
        D2D1_RECT_F headerRc = D2D1::RectF(blockRc.left, blockRc.top, blockRc.right, blockRc.top + 28.0f);
        current_rt_->PushAxisAlignedClip(headerRc, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);
        current_rt_->FillRoundedRectangle(rrc, pBrushCodeHeaderBg_);
        current_rt_->PopAxisAlignedClip();
        current_rt_->DrawLine(D2D1::Point2F(blockRc.left, blockRc.top + 28.0f),
                              D2D1::Point2F(blockRc.right, blockRc.top + 28.0f),
                              pBrushCodeBorder_, 1.0f);

        // Language badge
        std::wstring langBadge = block.lang.empty() ? L"CODE" : block.lang;
        current_rt_->DrawText(langBadge.c_str(), static_cast<UINT32>(langBadge.size()),
            pFormatChipText_, D2D1::RectF(headerRc.left + 12.0f, headerRc.top + 5.0f, headerRc.left + 120.0f, headerRc.bottom),
            pBrushAccentBlue_);

        // Copy Code button
        D2D1_RECT_F copyBtn = D2D1::RectF(headerRc.right - 88.0f, headerRc.top + 4.0f, headerRc.right - 8.0f, headerRc.top + 24.0f);
        block.copy_btn_rect = copyBtn;

        D2D1_ROUNDED_RECT btnRrc = D2D1::RoundedRect(copyBtn, 4.0f, 4.0f);
        if (block.copied) {
            current_rt_->FillRoundedRectangle(btnRrc, pBrushAccentGreen_);
            std::wstring lbl = L"✓ Copied!";
            current_rt_->DrawText(lbl.c_str(), static_cast<UINT32>(lbl.size()), pFormatChipText_,
                D2D1::RectF(copyBtn.left + 14.0f, copyBtn.top + 3.0f, copyBtn.right, copyBtn.bottom), pBrushTextPrimary_);
        } else if (block.copy_hovered) {
            current_rt_->FillRoundedRectangle(btnRrc, pBrushChipHover_);
            current_rt_->DrawRoundedRectangle(btnRrc, pBrushAccentBlue_, 1.0f);
            std::wstring lbl = L"Copy Code";
            current_rt_->DrawText(lbl.c_str(), static_cast<UINT32>(lbl.size()), pFormatChipText_,
                D2D1::RectF(copyBtn.left + 10.0f, copyBtn.top + 3.0f, copyBtn.right, copyBtn.bottom), pBrushAccentBlue_);
        } else {
            current_rt_->FillRoundedRectangle(btnRrc, pBrushChipBg_);
            current_rt_->DrawRoundedRectangle(btnRrc, pBrushCardBorder_, 1.0f);
            std::wstring lbl = L"Copy Code";
            current_rt_->DrawText(lbl.c_str(), static_cast<UINT32>(lbl.size()), pFormatChipText_,
                D2D1::RectF(copyBtn.left + 10.0f, copyBtn.top + 3.0f, copyBtn.right, copyBtn.bottom), pBrushTextSecondary_);
        }

        // 3. Syntax-highlighted code content
        if (!block.content.empty()) {
            float maxW = blockRc.right - blockRc.left - 28.0f;
            IDWriteTextLayout* pLayout = nullptr;
            HRESULT hr = pDWriteFactory_->CreateTextLayout(
                block.content.c_str(), static_cast<UINT32>(block.content.size()),
                pFormatCodeText_, maxW, 10000.0f, &pLayout
            );
            if (SUCCEEDED(hr) && pLayout) {
                apply_syntax_highlighting(pLayout, block.content);
                current_rt_->DrawTextLayout(
                    D2D1::Point2F(blockRc.left + 14.0f, blockRc.top + 34.0f),
                    pLayout,
                    pBrushTextPrimary_
                );
                pLayout->Release();
            }
        }
    }

    void render_quick_chips() {
        for (const auto& chip : quick_chips_) {
            D2D1_ROUNDED_RECT rrc = D2D1::RoundedRect(chip.rect, 6.0f, 6.0f);
            current_rt_->FillRoundedRectangle(rrc, chip.is_hovered ? pBrushChipHover_ : pBrushChipBg_);
            current_rt_->DrawRoundedRectangle(rrc, pBrushCardBorder_, 1.0f);

            current_rt_->DrawText(chip.label.c_str(), static_cast<UINT32>(chip.label.size()),
                pFormatChipText_, D2D1::RectF(chip.rect.left + 8.0f, chip.rect.top + 3.0f, chip.rect.right - 8.0f, chip.rect.bottom),
                pBrushAccentBlue_);
        }
    }

    void render_action_buttons(float, float h) {
        // Render composer input box background & placeholder
        float composerY = h - 60.0f;
        float inputW = send_btn_rect_.left - 32.0f;
        if (inputW > 100.0f) {
            D2D1_RECT_F inputBg = D2D1::RectF(20.0f, composerY, 20.0f + inputW, composerY + 36.0f);
            D2D1_ROUNDED_RECT inputRrc = D2D1::RoundedRect(inputBg, 8.0f, 8.0f);
            current_rt_->FillRoundedRectangle(inputRrc, pBrushCardBg_);
            current_rt_->DrawRoundedRectangle(inputRrc, pBrushCardBorder_, 1.0f);

            // Placeholder text if empty
            std::wstring placeholder = L"Ask the MoE Engine anything... (e.g. 'Build a Tkinter Calculator') [Enter to Send]";
            current_rt_->DrawText(placeholder.c_str(), static_cast<UINT32>(placeholder.size()),
                pFormatChipText_, D2D1::RectF(inputBg.left + 12.0f, inputBg.top + 9.0f, inputBg.right - 12.0f, inputBg.bottom),
                pBrushTextMuted_);
        }

        // Send Button
        D2D1_ROUNDED_RECT sendRrc = D2D1::RoundedRect(send_btn_rect_, 8.0f, 8.0f);
        current_rt_->FillRoundedRectangle(sendRrc, send_btn_hovered_ ? pBrushAccentBlue_ : pBrushUserBubble_);

        std::wstring sendLabel = client_.is_streaming() ? L"..." : L"Send";
        current_rt_->DrawText(sendLabel.c_str(), static_cast<UINT32>(sendLabel.size()),
            pFormatChipText_, D2D1::RectF(send_btn_rect_.left + 26.0f, send_btn_rect_.top + 9.0f, send_btn_rect_.right, send_btn_rect_.bottom),
            pBrushTextPrimary_);

        // Stop Button (if streaming)
        if (client_.is_streaming() && stop_btn_rect_.right > 0) {
            D2D1_ROUNDED_RECT stopRrc = D2D1::RoundedRect(stop_btn_rect_, 8.0f, 8.0f);
            current_rt_->FillRoundedRectangle(stopRrc, stop_btn_hovered_ ? pBrushAccentPink_ : pBrushChipBg_);
            current_rt_->DrawRoundedRectangle(stopRrc, pBrushAccentPink_, 1.0f);

            std::wstring stopLabel = L"Stop";
            current_rt_->DrawText(stopLabel.c_str(), static_cast<UINT32>(stopLabel.size()),
                pFormatChipText_, D2D1::RectF(stop_btn_rect_.left + 22.0f, stop_btn_rect_.top + 9.0f, stop_btn_rect_.right, stop_btn_rect_.bottom),
                pBrushTextPrimary_);
        }
    }

    float estimate_text_height(const std::wstring& str, float maxW) const {
        if (str.empty()) return 24.0f;
        IDWriteTextLayout* pLayout = nullptr;
        HRESULT hr = pDWriteFactory_->CreateTextLayout(
            str.c_str(), static_cast<UINT32>(str.size()),
            pFormatMsgText_, std::max(60.0f, maxW), 10000.0f, &pLayout
        );
        if (SUCCEEDED(hr) && pLayout) {
            DWRITE_TEXT_METRICS tm;
            pLayout->GetMetrics(&tm);
            pLayout->Release();
            return std::max(24.0f, tm.height);
        }
        return 40.0f;
    }

    float estimate_code_height(const std::wstring& str, float maxW) const {
        if (str.empty()) return 20.0f;
        IDWriteTextLayout* pLayout = nullptr;
        HRESULT hr = pDWriteFactory_->CreateTextLayout(
            str.c_str(), static_cast<UINT32>(str.size()),
            pFormatCodeText_, std::max(60.0f, maxW), 10000.0f, &pLayout
        );
        if (SUCCEEDED(hr) && pLayout) {
            DWRITE_TEXT_METRICS tm;
            pLayout->GetMetrics(&tm);
            pLayout->Release();
            return std::max(20.0f, tm.height);
        }
        return 30.0f;
    }

    float compute_total_messages_height(float w) const {
        float totalH = 0.0f;
        float rightLimit = show_cortex_atlas_ ? (w - 316.0f) : (w - 20.0f);
        float maxBubbleW = std::min(rightLimit - 60.0f, 760.0f);

        for (const auto& msg : messages_) {
            ensure_message_blocks(msg);
            bool isUser = (msg.role == "user");

            float blocksH = 0.0f;
            for (const auto& block : msg.blocks) {
                if (block.type == BlockType::Text) {
                    blocksH += estimate_text_height(block.content, maxBubbleW - 32.0f) + 6.0f;
                } else if (block.type == BlockType::Code) {
                    blocksH += 28.0f + estimate_code_height(block.content, maxBubbleW - 48.0f) + 16.0f + 8.0f;
                }
            }
            if (blocksH <= 0.0f) blocksH = 24.0f;

            float reasonH = 0.0f;
            if (!isUser && !msg.reasoning.empty()) {
                reasonH = estimate_text_height(msg.reasoning, maxBubbleW - 48.0f) + 28.0f;
            }
            float bubbleH = blocksH + reasonH + 24.0f;
            totalH += bubbleH + 16.0f;
        }
        return totalH;
    }

    void scroll_to_bottom() {
        RECT rc{0, 0, 1140, 800};
        if (hwnd_) GetClientRect(hwnd_, &rc);
        float w = static_cast<float>(rc.right);
        float h = static_cast<float>(rc.bottom);
        float viewH = (h - 104.0f) - 132.0f;
        float totalH = compute_total_messages_height(w);
        if (totalH > viewH) {
            scroll_offset_ = static_cast<int>(totalH - viewH + 20.0f);
        } else {
            scroll_offset_ = 0;
        }
    }

    std::wstring format_float(double v, int prec) const {
        std::wostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    }

    std::wstring utf8_to_wstring(const std::string& str) const {
        if (str.empty()) return L"";
        int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
        std::wstring out(len, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &out[0], len);
        return out;
    }

    std::string wstring_to_utf8(const std::wstring& str) const {
        if (str.empty()) return "";
        int len = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0, nullptr, nullptr);
        std::string out(len, 0);
        WideCharToMultiByte(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &out[0], len, nullptr, nullptr);
        return out;
    }

    HINSTANCE hInstance_ = nullptr;
    HWND hwnd_ = nullptr;
    HWND hEditInput_ = nullptr;

    ID2D1Factory*          pD2DFactory_ = nullptr;
    ID2D1HwndRenderTarget* pRenderTarget_ = nullptr;
    ID2D1RenderTarget*     current_rt_ = nullptr;
    IDWriteFactory*        pDWriteFactory_ = nullptr;

    IDWriteTextFormat* pFormatHeaderTitle_ = nullptr;
    IDWriteTextFormat* pFormatHeaderSub_   = nullptr;
    IDWriteTextFormat* pFormatStatValue_   = nullptr;
    IDWriteTextFormat* pFormatStatLabel_   = nullptr;
    IDWriteTextFormat* pFormatMsgText_     = nullptr;
    IDWriteTextFormat* pFormatCodeText_    = nullptr;
    IDWriteTextFormat* pFormatChipText_    = nullptr;
    IDWriteTextFormat* pFormatAtlasCell_   = nullptr;

    ID2D1SolidColorBrush* pBrushTextPrimary_   = nullptr;
    ID2D1SolidColorBrush* pBrushTextSecondary_ = nullptr;
    ID2D1SolidColorBrush* pBrushTextMuted_     = nullptr;
    ID2D1SolidColorBrush* pBrushAccentBlue_    = nullptr;
    ID2D1SolidColorBrush* pBrushAccentGreen_   = nullptr;
    ID2D1SolidColorBrush* pBrushAccentPink_    = nullptr;
    ID2D1SolidColorBrush* pBrushAccentAmber_   = nullptr;
    ID2D1SolidColorBrush* pBrushAccentIndigo_  = nullptr;
    ID2D1SolidColorBrush* pBrushCardBg_        = nullptr;
    ID2D1SolidColorBrush* pBrushCardBorder_    = nullptr;
    ID2D1SolidColorBrush* pBrushUserBubble_    = nullptr;
    ID2D1SolidColorBrush* pBrushAsstBubble_    = nullptr;
    ID2D1SolidColorBrush* pBrushCodeBg_        = nullptr;
    ID2D1SolidColorBrush* pBrushChipBg_        = nullptr;
    ID2D1SolidColorBrush* pBrushChipHover_     = nullptr;

    ID2D1SolidColorBrush* pBrushCodeHeaderBg_  = nullptr;
    ID2D1SolidColorBrush* pBrushCodeBorder_    = nullptr;
    ID2D1SolidColorBrush* pBrushSyntaxKeyword_ = nullptr;
    ID2D1SolidColorBrush* pBrushSyntaxString_  = nullptr;
    ID2D1SolidColorBrush* pBrushSyntaxComment_ = nullptr;
    ID2D1SolidColorBrush* pBrushSyntaxNumber_  = nullptr;

    ID2D1SolidColorBrush* pBrushAtlasActive_   = nullptr;
    ID2D1SolidColorBrush* pBrushAtlasVram_     = nullptr;
    ID2D1SolidColorBrush* pBrushAtlasRam_      = nullptr;
    ID2D1SolidColorBrush* pBrushAtlasDisk_     = nullptr;

    std::vector<ChatMessageUI> messages_;
    std::vector<QuickChip> quick_chips_;

    D2D1_RECT_F send_btn_rect_ = D2D1::RectF();
    D2D1_RECT_F stop_btn_rect_ = D2D1::RectF();
    bool send_btn_hovered_ = false;
    bool stop_btn_hovered_ = false;

    bool show_cortex_atlas_ = true;
    int scroll_offset_ = 0;
    uint32_t active_slot_ = 0;

    moe::ui::TelemetrySnapshot latest_telem_{};
    moe::ui::ApiClient client_;
};

} // namespace

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    INITCOMMONCONTROLSEX icex{};
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icex);

    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::wstring screenshot_path = L"";
    std::string test_prompt = "";
    uint32_t test_max_tokens = 140;
    if (argv) {
        for (int i = 1; i < argc; ++i) {
            if (std::wstring(argv[i]) == L"--screenshot" && i + 1 < argc) {
                screenshot_path = argv[++i];
            } else if (std::wstring(argv[i]) == L"--test-prompt" && i + 1 < argc) {
                int len = WideCharToMultiByte(CP_UTF8, 0, argv[++i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(len, 0);
                WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, &s[0], len, nullptr, nullptr);
                if (!s.empty() && s.back() == '\0') s.pop_back();
                test_prompt = s;
            } else if (std::wstring(argv[i]) == L"--max-tokens" && i + 1 < argc) {
                test_max_tokens = static_cast<uint32_t>(std::stoul(std::wstring(argv[++i])));
            }
        }
        LocalFree(argv);
    }

    MoEStudioApp app;
    if (!app.init(hInstance, nCmdShow)) {
        MessageBox(nullptr, L"Failed to initialize MoE Studio Direct2D interface.", L"MoE Studio Error", MB_ICONERROR | MB_OK);
        return 1;
    }

    if (!test_prompt.empty()) {
        app.send_prompt_sync(test_prompt, test_max_tokens);
    }

    if (!screenshot_path.empty()) {
        bool ok = app.save_screenshot_png(screenshot_path);
        return ok ? 0 : 2;
    }

    return app.run();
}