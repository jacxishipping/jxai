#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wincodec.h>
#include <iostream>

int main() {
    CoInitialize(nullptr);

    IWICImagingFactory* pWICFactory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pWICFactory));
    if (FAILED(hr)) return 1;

    ID2D1Factory* pD2DFactory = nullptr;
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pD2DFactory);
    if (FAILED(hr)) return 2;

    IDWriteFactory* pDWriteFactory = nullptr;
    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&pDWriteFactory));
    if (FAILED(hr)) return 3;

    int w = 800;
    int h = 600;

    IWICBitmap* pWicBitmap = nullptr;
    hr = pWICFactory->CreateBitmap(w, h, GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnDemand, &pWicBitmap);
    if (FAILED(hr)) return 4;

    D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
        0.0f, 0.0f,
        D2D1_RENDER_TARGET_USAGE_NONE,
        D2D1_FEATURE_LEVEL_DEFAULT
    );

    ID2D1RenderTarget* pRT = nullptr;
    hr = pD2DFactory->CreateWicBitmapRenderTarget(pWicBitmap, rtProps, &pRT);
    if (FAILED(hr)) {
        std::cout << "CreateWicBitmapRenderTarget failed: 0x" << std::hex << hr << std::dec << std::endl;
        return 5;
    }

    ID2D1SolidColorBrush* pBgBrush = nullptr;
    pRT->CreateSolidColorBrush(D2D1::ColorF(0.04f, 0.06f, 0.10f), &pBgBrush);

    ID2D1SolidColorBrush* pBlueBrush = nullptr;
    pRT->CreateSolidColorBrush(D2D1::ColorF(0.22f, 0.74f, 0.97f), &pBlueBrush);

    IDWriteTextFormat* pTextFormat = nullptr;
    pDWriteFactory->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 24.0f, L"en-US", &pTextFormat);

    pRT->BeginDraw();
    pRT->Clear(D2D1::ColorF(0.04f, 0.06f, 0.10f));

    D2D1_RECT_F cardRc = D2D1::RectF(50, 50, 750, 250);
    pRT->FillRectangle(cardRc, pBgBrush);
    pRT->DrawRectangle(cardRc, pBlueBrush, 2.0f);

    std::wstring text = L"MoE Studio Direct2D Hardware Acceleration - Verified OK!";
    pRT->DrawText(text.c_str(), static_cast<UINT32>(text.size()), pTextFormat, D2D1::RectF(70, 70, 730, 200), pBlueBrush);

    pRT->EndDraw();

    // Save to PNG
    IWICStream* pStream = nullptr;
    pWICFactory->CreateStream(&pStream);
    pStream->InitializeFromFilename(L"test_render.png", GENERIC_WRITE);

    IWICBitmapEncoder* pEncoder = nullptr;
    pWICFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder);
    pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);

    IWICBitmapFrameEncode* pFrame = nullptr;
    pEncoder->CreateNewFrame(&pFrame, nullptr);
    pFrame->Initialize(nullptr);
    pFrame->SetSize(w, h);

    WICPixelFormatGUID format = GUID_WICPixelFormat32bppPBGRA;
    pFrame->SetPixelFormat(&format);
    pFrame->WriteSource(pWicBitmap, nullptr);
    pFrame->Commit();
    pEncoder->Commit();

    std::cout << "Successfully generated test_render.png!" << std::endl;

    pFrame->Release();
    pEncoder->Release();
    pStream->Release();
    pTextFormat->Release();
    pBlueBrush->Release();
    pBgBrush->Release();
    pRT->Release();
    pWicBitmap->Release();
    pDWriteFactory->Release();
    pD2DFactory->Release();
    pWICFactory->Release();
    CoUninitialize();

    return 0;
}

