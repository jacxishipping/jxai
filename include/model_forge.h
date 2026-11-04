#pragma once

#include "common.h"
#include <string>

namespace moe {

// Generates a real, standards-compliant GGUF v3 MoE model file (Q4_0 sparse experts + F32 norms)
// and/or a SafeTensors MoE model file on disk so `./engine chat` and `./engine serve`
// can run out-of-the-box as well as with any external GGUF/SafeTensors model.
bool forge_moe_gguf_model(const std::string& output_path, std::string& err_msg);
bool forge_moe_safetensors_model(const std::string& output_path, std::string& err_msg);

} // namespace moe

