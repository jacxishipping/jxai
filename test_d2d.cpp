#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <iostream>

int main() {
    ID2D1Factory* pFactory = nullptr;
    HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);
    std::cout << "D2D1CreateFactory hr: 0x" << std::hex << hr << std::dec << std::endl;
    if (FAILED(hr)) return 1;

    HWND hwnd = CreateWindowEx(0, L"STATIC", L"Test", WS_OVERLAPPEDWINDOW, 0, 0, 400, 300, nullptr, nullptr, nullptr, nullptr);
    D2D1_SIZE_U size = D2D1::SizeU(400, 300);

    ID2D1HwndRenderTarget* pRT = nullptr;
    hr = pFactory->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties(hwnd, size),
        &pRT
    );
    std::cout << "CreateHwndRenderTarget DEFAULT hr: 0x" << std::hex << hr << std::dec << std::endl;

    if (FAILED(hr)) {
        hr = pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_SOFTWARE),
            D2D1::HwndRenderTargetProperties(hwnd, size),
            &pRT
        );
        std::cout << "CreateHwndRenderTarget SOFTWARE hr: 0x" << std::hex << hr << std::dec << std::endl;
    }

    if (pRT) {
        pRT->BeginDraw();
        pRT->Clear(D2D1::ColorF(0.2f, 0.4f, 0.8f));
        hr = pRT->EndDraw();
        std::cout << "Draw test hr: 0x" << std::hex << hr << std::dec << std::endl;
        pRT->Release();
    }
    pFactory->Release();
    DestroyWindow(hwnd);
    return 0;
}
