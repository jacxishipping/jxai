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

struct ChatMessageUI {
    std::string role; // "user" or "assistant"
    std::wstring content;
    std::wstring reasoning;
    moe::ui::TelemetrySnapshot telem;
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
        if (pDWriteFactory_) pDWriteFactory_->Release();
        if (pD2DFactory_) pD2DFactory_->Release();
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
        int width = 1120;
        int height = 780;
        int screenW = GetSystemMetrics(SM_CXSCREEN);
        int screenH = GetSystemMetrics(SM_CYSCREEN);
        int posX = std::max(20, (screenW - width) / 2);
        int posY = std::max(20, (screenH - height) / 2);

        hwnd_ = CreateWindowEx(
            0,
            wc.lpszClassName,
            L"MoE Studio - Next-Gen Mixture-of-Experts Control Suite (C++ Native)",
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
            posX, posY, width, height,
            nullptr, nullptr, hInstance, this
        );

        if (!hwnd_) return false;

        // Enable Windows 11 Immersive Dark Titlebar
        BOOL darkMode = TRUE;
        DwmSetWindowAttribute(hwnd_, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &darkMode, sizeof(darkMode));

        // Create child single-line or multi-line edit control for input
        create_input_controls();

        // Populate initial welcome messages
        ChatMessageUI welcome{};
        welcome.role = "assistant";
        welcome.content = L"Welcome to MoE Studio C++ Native Suite!\r\n\r\n"
                          L"Direct2D hardware-accelerated interface connected to the local Hybrid-Tiered MoE Streaming Inference Engine.\r\n"
                          L"Sparse MoE weights are streamed asynchronously via Win32 IOCP / PILOT prefetch, while dense backbone attention matrices remain resident in zero-copy RAM/VRAM.\r\n\r\n"
                          L"Type a prompt below or click any quick action chip to start controlling the AI!";
        welcome.reasoning = L"MoE Engine Core Initialized | Top-2 router routing | Win32 IOCP Prefetch Pipeline active";
        welcome.telem.tok_per_sec = 7.1;
        welcome.telem.ttft_ms = 145.2;
        welcome.telem.latency_ms = 140.5;
        welcome.telem.hits = 142;
        welcome.telem.reads = 12;
        welcome.telem.vram_experts = 12;
        welcome.telem.disk_experts = 12;
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

    bool save_screenshot_png(const std::wstring& out_path, float w = 1120.0f, float h = 780.0f) {
        auto health = client_.fetch_health_sync();
        latest_telem_.server_online = health.server_online;
        if (!health.model_name.empty()) latest_telem_.model_name = health.model_name;
        if (!health.arch.empty()) latest_telem_.arch = health.arch;
        if (!health.simd.empty()) latest_telem_.simd = health.simd;
        if (!health.io_backend.empty()) latest_telem_.io_backend = health.io_backend;
        if (latest_telem_.vram_experts == 0) latest_telem_.vram_experts = health.vram_experts;
        if (latest_telem_.disk_experts == 0) latest_telem_.disk_experts = health.disk_experts;
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
                return 0;
            }

            case WM_COMMAND: {
                // If Enter pressed in Edit control
                if (HIWORD(wParam) == EN_UPDATE && LOWORD(wParam) == 201) {
                    // Update input state
                }
                return 0;
            }

            case WM_TIMER: {
                if (wParam == 1001) {
                    poll_engine_health();
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
            { L"Tkinter Calculator App", "Write an awesome calculator app in Python with Tkinter, clean dark GUI, and standard arithmetic operations (+, -, *, /).", D2D1::RectF() },
            { L"Explain MoE PILOT", "Explain the MoE PILOT overlapped prefetching architecture and LFRU cache scoring in high-performance local AI engines.", D2D1::RectF() },
            { L"What is 2 + 2?", "What is 2 + 2?", D2D1::RectF() },
            { L"Continue Generation", "continue", D2D1::RectF() },
            { L"Clear Chat", "__clear__", D2D1::RectF() }
        };
    }

    void handle_chip_click(const QuickChip& chip) {
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
            DWRITE_FONT_STRETCH_NORMAL, 11.5f, L"en-US", &pFormatCodeText_
        );
        pDWriteFactory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, 10.0f, L"en-US", &pFormatChipText_
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

        // 3. Render Message Feed
        render_message_feed(w, h);

        // 4. Render Quick Chips
        render_quick_chips();

        // 5. Render Buttons (Send / Stop)
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

    void render_message_feed(float w, float h) {
        float feedTop = 132.0f;
        float feedBottom = h - 104.0f;
        D2D1_RECT_F feedClip = D2D1::RectF(20.0f, feedTop, w - 20.0f, feedBottom);

        current_rt_->PushAxisAlignedClip(feedClip, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

        float curY = feedTop + 10.0f - static_cast<float>(scroll_offset_);

        for (const auto& msg : messages_) {
            bool isUser = (msg.role == "user");
            float bubbleW = std::min(w - 120.0f, 820.0f);

            // Compute text height
            float textH = estimate_text_height(msg.content, bubbleW - 32.0f);
            float reasonH = 0.0f;
            if (!isUser && !msg.reasoning.empty()) {
                reasonH = estimate_text_height(msg.reasoning, bubbleW - 48.0f) + 28.0f;
            }
            float bubbleH = textH + reasonH + 28.0f;

            float bubbleX = isUser ? (w - 40.0f - bubbleW) : 30.0f;

            if (curY + bubbleH >= feedTop - 50.0f && curY <= feedBottom + 50.0f) {
                D2D1_RECT_F bubbleRc = D2D1::RectF(bubbleX, curY, bubbleX + bubbleW, curY + bubbleH);
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
                    D2D1_RECT_F reasonBox = D2D1::RectF(bubbleX + 16.0f, curY + 10.0f, bubbleX + bubbleW - 16.0f, curY + 10.0f + reasonH - 8.0f);
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

                // Render content text
                D2D1_RECT_F textRc = D2D1::RectF(bubbleX + 16.0f, contentY, bubbleX + bubbleW - 16.0f, curY + bubbleH - 12.0f);
                current_rt_->DrawText(msg.content.c_str(), static_cast<UINT32>(msg.content.size()),
                    pFormatMsgText_, textRc, pBrushTextPrimary_);
            }

            curY += bubbleH + 16.0f;
        }

        current_rt_->PopAxisAlignedClip();
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

    float estimate_text_height(const std::wstring& str, float maxW) {
        if (str.empty()) return 24.0f;
        IDWriteTextLayout* pLayout = nullptr;
        HRESULT hr = pDWriteFactory_->CreateTextLayout(
            str.c_str(), static_cast<UINT32>(str.size()),
            pFormatMsgText_, maxW, 10000.0f, &pLayout
        );
        if (SUCCEEDED(hr) && pLayout) {
            DWRITE_TEXT_METRICS tm;
            pLayout->GetMetrics(&tm);
            pLayout->Release();
            return std::max(24.0f, tm.height);
        }
        return 40.0f;
    }

    float compute_total_messages_height(float w) {
        float totalH = 0.0f;
        for (const auto& msg : messages_) {
            bool isUser = (msg.role == "user");
            float bubbleW = std::min(w - 120.0f, 820.0f);
            float textH = estimate_text_height(msg.content, bubbleW - 32.0f);
            float reasonH = 0.0f;
            if (!isUser && !msg.reasoning.empty()) {
                reasonH = estimate_text_height(msg.reasoning, bubbleW - 48.0f) + 28.0f;
            }
            float bubbleH = textH + reasonH + 28.0f;
            totalH += bubbleH + 16.0f;
        }
        return totalH;
    }

    void scroll_to_bottom() {
        RECT rc{0, 0, 1120, 780};
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

    std::wstring format_float(double v, int prec) {
        std::wostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    }

    std::wstring utf8_to_wstring(const std::string& str) {
        if (str.empty()) return L"";
        int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
        std::wstring out(len, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &out[0], len);
        return out;
    }

    std::string wstring_to_utf8(const std::wstring& str) {
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

    std::vector<ChatMessageUI> messages_;
    std::vector<QuickChip> quick_chips_;

    D2D1_RECT_F send_btn_rect_ = D2D1::RectF();
    D2D1_RECT_F stop_btn_rect_ = D2D1::RectF();
    bool send_btn_hovered_ = false;
    bool stop_btn_hovered_ = false;

    int scroll_offset_ = 0;
    uint32_t active_slot_ = 0;

    moe::ui::TelemetrySnapshot latest_telem_{};
    moe::ui::ApiClient client_;
};

} // namespace

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // Enable High-DPI awareness
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
