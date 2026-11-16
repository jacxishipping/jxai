#pragma once
#include <functional>
#include <string>

namespace moe {

using PullProgressCallback = std::function<void(size_t downloaded, size_t total, double mb_per_sec, double eta_sec)>;

bool pull_model_from_hub(
    const std::string& input,
    const std::string& out_dir,
    std::string& out_file,
    std::string& err_msg,
    PullProgressCallback on_progress = nullptr
);

} // namespace moe

