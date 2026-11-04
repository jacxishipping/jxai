#pragma once

#include "transformer.h"
#include <string>
#include <vector>

namespace moe {

struct DenoiseStepPreview {
    uint32_t step = 0;
    uint32_t total_steps = 0;
    float    sigma = 0.0f;
    std::string svg_data_uri;
};

struct ImageGenerationResult {
    std::string prompt;
    uint32_t width = 512;
    uint32_t height = 512;
    uint32_t steps = 12;
    float    guidance_scale = 7.5f;
    uint64_t seed = 42;
    double   elapsed_ms = 0.0;
    std::vector<uint32_t> active_experts;
    std::vector<DenoiseStepPreview> previews;
    std::string final_svg_data_uri;
};

class NeuralLatentSynthesizer {
public:
    static ImageGenerationResult generate(
        MoETransformerEngine& engine,
        const std::string& prompt,
        uint32_t width,
        uint32_t height,
        uint32_t steps,
        float guidance_scale,
        uint64_t seed
    );
};

} // namespace moe

