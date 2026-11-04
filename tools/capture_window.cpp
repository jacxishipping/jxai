#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>

bool SaveHBitmapToBmp(HBITMAP hBitmap, const std::wstring& filePath) {
    HDC hdc = GetDC(nullptr);
    BITMAP bmp;
    GetObject(hBitmap, sizeof(BITMAP), &bmp);

    BITMAPINFOHEADER bi{};
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = bmp.bmWidth;
    bi.biHeight = bmp.bmHeight;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    DWORD dwSize = ((bmp.bmWidth * 32 + 31) / 32) * 4 * bmp.bmHeight;
    std::vector<BYTE> pixels(dwSize);

    GetDIBits(hdc, hBitmap, 0, bmp.bmHeight, pixels.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    ReleaseDC(nullptr, hdc);

    BITMAPFILEHEADER bfh{};
    bfh.bfType = 0x4D42; // "BM"
    bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bfh.bfSize = bfh.bfOffBits + dwSize;

    std::ofstream ofs(filePath.c_str(), std::ios::binary);
    if (!ofs) return false;

    ofs.write(reinterpret_cast<const char*>(&bfh), sizeof(bfh));
    ofs.write(reinterpret_cast<const char*>(&bi), sizeof(bi));
    ofs.write(reinterpret_cast<const char*>(pixels.data()), dwSize);
    return true;
}

int main(int argc, char* argv[]) {
    HWND hwnd = FindWindowW(L"MoEStudioWindowClass", nullptr);
    if (!hwnd) {
        std::cerr << "Window not found!" << std::endl;
        return 1;
    }

    RECT rc;
    GetWindowRect(hwnd, &rc);
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) return 2;

    HDC hdcScreen = GetDC(nullptr);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    HBITMAP hbm = CreateCompatibleBitmap(hdcScreen, w, h);
    SelectObject(hdcMem, hbm);

    // BitBlt directly from screen DC which includes DWM composited Direct2D
    BitBlt(hdcMem, 0, 0, w, h, hdcScreen, rc.left, rc.top, SRCCOPY);

    std::wstring out_path = (argc > 1) ? L"C:\\Users\\RDP\\.gemini\\antigravity\\brain\\9c15bc63-3d5e-4ae4-bc9a-8203acf60398\\moe_studio_window.bmp" : L"moe_studio_window.bmp";
    bool ok = SaveHBitmapToBmp(hbm, out_path);

    DeleteObject(hbm);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);

    if (ok) {
        std::wcout << L"Successfully captured window to: " << out_path << L" (" << w << L"x" << h << L")" << std::endl;
        return 0;
    }
    return 3;
}
