#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace moe {

using PullProgressCallback = std::function<void(size_t downloaded, size_t total, double mb_per_sec, double eta_sec)>;

// One file listed by the Hugging Face Hub API (tree/main endpoint).
struct HfFileInfo {
    std::string filename;      // repo-relative path, e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    uint64_t size = 0;         // bytes as reported by the Hub (0 = unknown)
    bool is_shard = false;     // true for "model-00001-of-00002.gguf" style names
    uint32_t shard_index = 0;  // 1-based part number
    uint32_t shard_total = 0;  // total number of parts in the group
    std::string shard_group;   // base name of the shard group, e.g. "qwen2.5-7b-instruct-q4_k_m.gguf"
};

// Filename picking rules (see pick_gguf_file() in model_pull.cpp):
//   1. Single-file .gguf files are preferred over sharded ones; a sharded
//      group is only picked when the repo has no single-file .gguf at all.
//   2. Among candidates of the same kind, lower rank wins, then smaller size.
//      Rank order: Q4_K_M < Q4_0 < Q5_K_M < Q4_K_S < Q5_K_S < Q6_K < Q8_0
//      < Q4_K/Q5_K < Q3_K_M/Q5_0 < Q2_K < FP16/F16.
//   3. An explicit file (repo:file or a direct URL) always wins over picking.
int gguf_pick_rank(const std::string& filename);

// True when "filename" matches "<base>-NNNNN-of-MMMMM.gguf". On success
// fills group ("<base>.gguf"), index and total.
bool parse_sharded_gguf_name(const std::string& filename, std::string& group,
                             uint32_t& index, uint32_t& total);

// Query the Hugging Face Hub API for <repo> (validated "owner/name").
// Fills gguf_files with every *.gguf file (sorted: best pick first) and
// st_files with every *.safetensors file. Sets gated=true when the repo
// requires access approval (downloads will 401 without an HF_TOKEN).
// Downloads nothing. Returns false with err_msg on network/API errors;
// a repo with no .gguf files is NOT an error (check the vectors).
bool list_hf_model_files(const std::string& repo,
                         std::vector<HfFileInfo>& gguf_files,
                         std::vector<HfFileInfo>& st_files,
                         bool& gated,
                         std::string& err_msg);

// Download a model: a direct https URL, "owner/name:file.gguf", or a bare
// "owner/name" repo id (resolved via the HF API, see picking rules above).
// Safe: inputs are validated and the download is executed without a shell
// (fork/exec curl on POSIX, WinINet on Windows). Supports resume (-C -),
// verifies Content-Length against downloaded bytes, and validates the
// file header (GGUF magic or SafeTensors header length).
// For sharded GGUF repos ALL parts are downloaded and out_file is set to
// the first part; the caller must NOT treat a single shard as a model.
// For .safetensors downloads from the Hub, sibling config.json and
// tokenizer.json are fetched alongside on a best-effort basis.
bool pull_model_from_hub(
    const std::string& input,
    const std::string& out_dir,
    std::string& out_file,
    std::string& err_msg,
    PullProgressCallback on_progress = nullptr
);

} // namespace moe
