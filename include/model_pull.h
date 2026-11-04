#pragma once
#include <string>

namespace moe {

bool pull_model_from_hub(
    const std::string& input,
    const std::string& out_dir,
    std::string& out_file,
    std::string& err_msg
);

} // namespace moe
