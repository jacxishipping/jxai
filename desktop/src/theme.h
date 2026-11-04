#pragma once
#include <d2d1.h>
#include <dwrite.h>

namespace moe::ui {

// Color definitions (RGBA 0.0f - 1.0f)
namespace colors {
    inline const D2D1_COLOR_F BgDark         = D2D1::ColorF(0x07 / 255.0f, 0x0A / 255.0f, 0x12 / 255.0f, 1.0f);
    inline const D2D1_COLOR_F HeaderBg       = D2D1::ColorF(0x0A / 255.0f, 0x0F / 255.0f, 0x1D / 255.0f, 1.0f);
    inline const D2D1_COLOR_F CardBg         = D2D1::ColorF(0x0E / 255.0f, 0x14 / 255.0f, 0x24 / 255.0f, 1.0f);
    inline const D2D1_COLOR_F CardBorder     = D2D1::ColorF(0x1E / 255.0f, 0x29 / 255.0f, 0x3B / 255.0f, 1.0f);
    inline const D2D1_COLOR_F CardBorderGlow = D2D1::ColorF(0x3B / 255.0f, 0x82 / 255.0f, 0xF6 / 255.0f, 0.4f);
    inline const D2D1_COLOR_F UserBubble     = D2D1::ColorF(0x25 / 255.0f, 0x63 / 255.0f, 0xEB / 255.0f, 0.95f);
    inline const D2D1_COLOR_F AsstBubble     = D2D1::ColorF(0x0F / 255.0f, 0x17 / 255.0f, 0x2A / 255.0f, 0.95f);
    inline const D2D1_COLOR_F CodeBg         = D2D1::ColorF(0x06 / 255.0f, 0x0A / 255.0f, 0x14 / 255.0f, 1.0f);
    inline const D2D1_COLOR_F InputBg        = D2D1::ColorF(0x0D / 255.0f, 0x13 / 255.0f, 0x22 / 255.0f, 1.0f);
    inline const D2D1_COLOR_F InputBorder    = D2D1::ColorF(0x33 / 255.0f, 0x41 / 255.0f, 0x55 / 255.0f, 1.0f);
    
    inline const D2D1_COLOR_F AccentBlue     = D2D1::ColorF(0x38 / 255.0f, 0xBD / 255.0f, 0xF8 / 255.0f, 1.0f); // #38bdf8
    inline const D2D1_COLOR_F AccentGreen    = D2D1::ColorF(0x10 / 255.0f, 0xB9 / 255.0f, 0x81 / 255.0f, 1.0f); // #10b981
    inline const D2D1_COLOR_F AccentPink     = D2D1::ColorF(0xF4 / 255.0f, 0x3F / 255.0f, 0x5E / 255.0f, 1.0f); // #f43f5e
    inline const D2D1_COLOR_F AccentAmber    = D2D1::ColorF(0xF5 / 255.0f, 0x9E / 255.0f, 0x0B / 255.0f, 1.0f); // #f59e0b
    inline const D2D1_COLOR_F AccentIndigo   = D2D1::ColorF(0x81 / 255.0f, 0x8C / 255.0f, 0xF8 / 255.0f, 1.0f); // #818cf8

    inline const D2D1_COLOR_F TextPrimary    = D2D1::ColorF(0xF8 / 255.0f, 0xFA / 255.0f, 0xFC / 255.0f, 1.0f); // #f8fafc
    inline const D2D1_COLOR_F TextSecondary  = D2D1::ColorF(0x94 / 255.0f, 0xA3 / 255.0f, 0xB8 / 255.0f, 1.0f); // #94a3b8
    inline const D2D1_COLOR_F TextMuted      = D2D1::ColorF(0x64 / 255.0f, 0x74 / 255.0f, 0x8B / 255.0f, 1.0f); // #64748b
    inline const D2D1_COLOR_F ChipBg         = D2D1::ColorF(0x1E / 255.0f, 0x29 / 255.0f, 0x3B / 255.0f, 0.7f);
    inline const D2D1_COLOR_F ChipHover      = D2D1::ColorF(0x33 / 255.0f, 0x41 / 255.0f, 0x55 / 255.0f, 0.9f);
}

} // namespace moe::ui

