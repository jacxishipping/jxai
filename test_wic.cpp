#include <windows.h>
#include <wincodec.h>
#include <iostream>

int main() {
    CoInitialize(nullptr);
    IWICImagingFactory* pFactory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));
    std::cout << "WIC Factory hr: 0x" << std::hex << hr << std::dec << std::endl;
    if (pFactory) pFactory->Release();
    CoUninitialize();
    return 0;
}

