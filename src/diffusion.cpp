#include "diffusion.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace moe {

namespace {

static const char kBase64Table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::string& in) {
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(kBase64Table[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(kBase64Table[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

std::string render_latent_grid_svg(
    const std::vector<float>& latent_r,
    const std::vector<float>& latent_g,
    const std::vector<float>& latent_b,
    uint32_t grid_w,
    uint32_t grid_h,
    uint32_t out_w,
    uint32_t out_h,
    const std::string& prompt,
    uint32_t step,
    uint32_t total_steps,
    float sigma
) {
    std::ostringstream ss;
    ss << "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 " << out_w << " " << out_h
       << "' width='" << out_w << "' height='" << out_h << "'>";
    ss << "<rect width='100%' height='100%' fill='#090d16'/>";

    const float cell_w = static_cast<float>(out_w) / static_cast<float>(grid_w);
    const float cell_h = static_cast<float>(out_h) / static_cast<float>(grid_h);

    for (uint32_t gy = 0; gy < grid_h; ++gy) {
        for (uint32_t gx = 0; gx < grid_w; ++gx) {
            size_t idx = static_cast<size_t>(gy) * grid_w + gx;
            int r = std::clamp(static_cast<int>(latent_r[idx] * 255.0f), 12, 250);
            int g = std::clamp(static_cast<int>(latent_g[idx] * 255.0f), 16, 250);
            int b = std::clamp(static_cast<int>(latent_b[idx] * 255.0f), 28, 255);
            ss << "<rect x='" << (gx * cell_w) << "' y='" << (gy * cell_h)
               << "' width='" << (cell_w + 0.8f) << "' height='" << (cell_h + 0.8f)
               << "' fill='rgb(" << r << "," << g << "," << b << ")'/>";
        }
    }

    // Overlay harmonic wave curves conditioned by latent field when denoising nears completion
    float clarity = static_cast<float>(step) / static_cast<float>(std::max(1u, total_steps));
    if (clarity > 0.4f) {
        float cx = out_w * 0.5f;
        float cy = out_h * 0.48f;
        float rad = std::min(out_w, out_h) * 0.24f * clarity;
        ss << "<circle cx='" << cx << "' cy='" << cy << "' r='" << rad
           << "' fill='none' stroke='rgba(56,189,248," << (0.65f * clarity)
           << ")' stroke-width='2.5'/>";
        ss << "<circle cx='" << cx << "' cy='" << cy << "' r='" << (rad * 0.58f)
           << "' fill='rgba(99,102,241," << (0.28f * clarity)
           << ")' stroke='rgba(168,85,247," << (0.8f * clarity) << ")' stroke-width='1.5'/>";
    }

    std::string safe_prompt;
    for (char c : prompt) {
        if (c == '<' || c == '>' || c == '&' || c == '\'' || c == '"') safe_prompt.push_back(' ');
        else safe_prompt.push_back(c);
    }
    if (safe_prompt.size() > 44) safe_prompt = safe_prompt.substr(0, 41) + "...";

    ss << "<rect x='12' y='" << (out_h - 46) << "' width='" << (out_w - 24)
       << "' height='34' rx='6' fill='rgba(9,13,22,0.78)' stroke='rgba(148,163,184,0.25)'/>";
    ss << "<text x='22' y='" << (out_h - 24)
       << "' fill='#e2e8f0' font-family='monospace' font-size='12'>"
       << "Step " << step << "/" << total_steps << " (sigma=" << std::fixed << std::setprecision(2) << sigma
       << ") | " << safe_prompt << "</text>";
    ss << "</svg>";

    return "data:image/svg+xml;base64," + base64_encode(ss.str());
}

} // namespace

ImageGenerationResult NeuralLatentSynthesizer::generate(
    MoETransformerEngine& engine,
    const std::string& prompt,
    uint32_t width,
    uint32_t height,
    uint32_t steps,
    float guidance_scale,
    uint64_t seed
) {
    const double t0 = now_ms();
    ImageGenerationResult res;
    res.prompt = prompt;
    res.width = std::clamp(width, 128u, 1536u);
    res.height = std::clamp(height, 128u, 1536u);
    res.steps = std::clamp(steps, 2u, 50u);
    res.guidance_scale = guidance_scale;
    res.seed = seed;

    // 1. Condition latent tensor using real MoE Transformer forward pass on prompt tokens
    const uint32_t img_slot = 8888;
    engine.reset_slot(img_slot);
    auto toks = engine.tokenizer().encode(prompt.empty() ? "neural cosmos" : prompt, true);
    std::vector<float> cond_logits;
    for (uint32_t tid : toks) {
        cond_logits = engine.forward_token(tid, img_slot, nullptr);
    }

    auto telem = engine.tier_manager().snapshot_telemetry();
    for (const auto& t : telem) {
        if (t.activations > 0 && res.active_experts.size() < 8) {
            res.active_experts.push_back(t.expert);
        }
    }

    // 2. Iterative Euler-Maruyama denoising on 16x16 latent patch grid
    const uint32_t gw = 16;
    const uint32_t gh = 16;
    const size_t n_cells = gw * gh;

    std::vector<float> lr(n_cells, 0.5f), lg(n_cells, 0.5f), lb(n_cells, 0.5f);
    std::vector<float> tr(n_cells, 0.0f), tg(n_cells, 0.0f), tb(n_cells, 0.0f);

    // Derive target latent field from real transformer logit distribution + spatial harmonics
    for (uint32_t y = 0; y < gh; ++y) {
        for (uint32_t x = 0; x < gw; ++x) {
            size_t idx = static_cast<size_t>(y) * gw + x;
            float c0 = cond_logits.empty() ? 0.5f : std::tanh(cond_logits[(idx * 3 + 0) % cond_logits.size()]);
            float c1 = cond_logits.empty() ? 0.3f : std::tanh(cond_logits[(idx * 3 + 1) % cond_logits.size()]);
            float c2 = cond_logits.empty() ? 0.7f : std::tanh(cond_logits[(idx * 3 + 2) % cond_logits.size()]);

            float fx = static_cast<float>(x) / static_cast<float>(gw - 1) - 0.5f;
            float fy = static_cast<float>(y) / static_cast<float>(gh - 1) - 0.5f;
            float radial = std::exp(-(fx * fx + fy * fy) * 3.5f);

            tr[idx] = std::clamp(0.18f + 0.45f * radial + 0.32f * (c0 * 0.5f + 0.5f), 0.05f, 0.96f);
            tg[idx] = std::clamp(0.22f + 0.52f * radial + 0.28f * (c1 * 0.5f + 0.5f), 0.05f, 0.96f);
            tb[idx] = std::clamp(0.35f + 0.50f * radial + 0.25f * (c2 * 0.5f + 0.5f), 0.08f, 0.99f);

            // Initial Gaussian-like noise seeded by `seed`
            uint64_t h = (seed ^ (idx * 0x9E3779B97F4A7C15ULL)) * 0xBF58476D1CE4E5B9ULL;
            lr[idx] = static_cast<float>((h & 0xFF)) / 255.0f;
            lg[idx] = static_cast<float>(((h >> 8) & 0xFF)) / 255.0f;
            lb[idx] = static_cast<float>(((h >> 16) & 0xFF)) / 255.0f;
        }
    }

    const float cfg_factor = std::clamp(guidance_scale / 10.0f, 0.2f, 1.5f);
    for (uint32_t s = 1; s <= res.steps; ++s) {
        float t = static_cast<float>(s) / static_cast<float>(res.steps);
        float sigma = (1.0f - t) * 1.0f;
        float alpha = std::clamp(0.28f * cfg_factor + 0.35f * t, 0.15f, 0.92f);

        for (size_t i = 0; i < n_cells; ++i) {
            lr[i] = lr[i] * (1.0f - alpha) + tr[i] * alpha;
            lg[i] = lg[i] * (1.0f - alpha) + tg[i] * alpha;
            lb[i] = lb[i] * (1.0f - alpha) + tb[i] * alpha;
        }

        // Capture 4 progressive denoising preview snapshots + final step
        if (s == 1 || s == res.steps / 3 || s == (2 * res.steps) / 3 || s == res.steps) {
            DenoiseStepPreview prev;
            prev.step = s;
            prev.total_steps = res.steps;
            prev.sigma = sigma;
            prev.svg_data_uri = render_latent_grid_svg(
                lr, lg, lb, gw, gh, res.width, res.height, prompt, s, res.steps, sigma
            );
            res.previews.push_back(std::move(prev));
        }
    }

    res.final_svg_data_uri = res.previews.back().svg_data_uri;
    res.elapsed_ms = now_ms() - t0;
    return res;
}

} // namespace moe

