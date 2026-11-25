# JXAI Support Matrix

What the native C++ engine (`moe_engine`) is verified to run, what it
refuses, and what it only warns about. This is a **correctness** document:
a row marked "verified" means a real model of that kind was loaded and
produced coherent output on this engine; a row marked "unverified" means
the code path exists but no real model was exercised, and the engine says
so loudly at load time instead of silently producing wrong output.

Last updated: 2026-11-25 (investigation — retired fixed short-read /
cache_slot seams; confirmed CI absent from tree).
Engine state at that date: `moe_engine test` 11/11 (incl. Test 1d K-quant
remainder checks and Tests 1e/1f malformed-input guards).

> **Toy models are not evidence.** The forged toy model
> (`forge_moe_gguf_model`, qwen2moe, vocab 512, dim 128, 4 layers) is used
> only for offline self-tests. It proves the code paths run; it proves
> nothing about real-model accuracy. Every "verified" row below names the
> real model or fixture it was checked against.

---

## 1. Architectures

The loader reads `general.architecture` from GGUF metadata (or the
SafeTensors config) and applies a per-architecture RoPE rule
(`arch_uses_neox_rope()` in `src/transformer.cpp`). RoPE is **per-arch**,
not global: Qwen/Gemma/Phi-3 need Neox, Llama/Mistral need interleaved.
Getting this wrong produces fluent-looking garbage, which is why the
engine refuses unknown architectures instead of guessing.

| Architecture | Status | Evidence |
|---|---|---|
| `qwen2` | ✅ Verified | Qwen2.5-0.5B-Instruct Q4_0 — "The answer to 2+2 is 4." |
| `qwen2moe` | ⚠️ Partial — toy + ST packing only | Forged toy (self-test); `qwen2moe-tiny/` SafeTensors packing byte-exact. No real-model chat verified. |
| `llama` | ✅ Verified | Llama-3.2-1B-Instruct Q4_K_M — "The capital of France is Paris." |
| `qwen3` | ✅ Verified | Qwen3-0.6B Q4_K_M / Q2_K — coherent thinking; needs `max_tokens > 128` to finish thinking and answer. |
| `gemma3` | ✅ Verified | Gemma-3-1b-it Q4_K_M — "2 + 2 = 4". Nested `text_config` handled. |
| `phi3` | ✅ Verified | Phi-3-mini-4k-instruct Q2_K (GGUF + 2-shard SafeTensors) — Neox RoPE, verified by clean A/B vs interleaved. |
| `mistral` | ✅ Verified | Mistral-7B-Instruct-v0.3 Q2_K — GGUF reports arch `llama`; `[INST]` bracket special tokens handled. |
| `qwen3moe`, `mixtral`, `gemma`, `gemma2`, `gemma4`, `phi`, `phi4`, `deepseek`, `deepseek2`, `olmoe`, `glm`, `gptneox`, `gpt2`, `gptj`, `smollm`, `stablelm`, `starcoder2`, `exaone`, `command-r`, `falcon` | ⚠️ Unverified — loads with a loud stderr warning | Code path exists (per-arch RoPE table entry or generic default). **True `deepseek`/`deepseek2` arch is unverified**: DeepSeek-LLM-7B's GGUF reports arch `llama`, so it runs as llama and generates coherently, but that says nothing about a real deepseek-arch file. |
| Any other architecture string | ❌ Refused at load | `bind_tensors()` errors: unknown architecture. |
| Missing `general.architecture` | ❌ Refused at load | Previously silently defaulted to `qwen2moe`. |
| `mamba`, `bert`, `nomic-bert` | ❌ Refused at load | Non-transformer architectures; the engine has no SSM/encoder path. |

Hyperparameters are range-checked before any allocation (dim 1–65536,
n_layers 1–512, n_heads 1–256, n_kv_heads ≤ n_heads, n_experts 1–1024,
n_experts_used ≤ n_experts, max_seq_len 1–1,048,576, finite
`rms_norm_eps` / `rope_freq_base`). `q_dim` must equal
`n_heads * head_dim` and `kv_dim` must equal `n_kv_heads * head_dim`;
`dim == n_heads * head_dim` is **not** required (Gemma2-9B: dim 3584,
q_dim 4096). A layer whose attention weights (wq/wk/wv/wo) are missing
fails the load — previously this was a null-pointer segfault in the
matmul kernels, which do not null-check. A layer with no working FFN
fails the load. A MoE layer with no router (`ffn_gate_inp.weight`)
fails the load — previously every token silently routed to expert 0.

## 2. Quantization types (GGUF)

| Type | Status | Notes |
|---|---|---|
| F32, F16, BF16 | ✅ Supported | BF16 via dequantize path. |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | ✅ Supported | Bit-exact vs ggml (1b/1c tests). |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | ✅ Supported | Bit-exact vs ggml; **struct field order fixed vs ggml** (Q2_K/Q3_K had `d`/`dmin` first; ggml has them last). Trailing partial blocks (k not divisible by 256) are handled — Test 1d (k=896/100). |
| IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ1_M | ✅ Supported (dequantize) | Ported bit-exact vs ggml grids. Quantizers are simplified (ggml's need an importance matrix). |
| IQ2_M | ❌ Unsupported | **Does not exist in ggml** (GGUF id 36 is IQ4_NL_4_4). Refused with a warning, no approximation invented. |
| I8, I16, I32, I64, F64 | ❌ Unsupported | Refused at load (unknown-type error in the GGUF parser). |
| Unknown type id (e.g. 999) | ❌ Refused at parse | Parser validates `type_raw` against the known GGML enum. |

Partial-block edge cases (k = 1, 31, 33, 63, 65, 257) are covered by
Test 1e against the FP32 reference dot product for Q4_K, plus zero-vector,
constant, large (1e4) and tiny (1e-6) magnitude checks. **Fixed after
Test 1e was written:** `vec_dot_q4_0_f32` / `vec_dot_q8_0_f32` (AVX2 +
scalar) and `dequantize_row_q4_0` / `dequantize_row_q8_0` used to drop the
trailing partial block (k < 32 → dot exactly 0); they now consume only the
valid elements of the final partial block, the same remainder handling the
`vec_dot_deq_f32` template implements for the K-quant family. Verified by
Test 1e and by a targeted 18/18 harness (AVX2 + scalar). IQ
dequantizers/quantizers still drop remainders on direct calls for the same
reason.

## 3. Model file formats

| Format | Status | Validation |
|---|---|---|
| GGUF (v2/v3) | ✅ Supported | Magic, version (2 or 3 only), KV/tensor counts capped and cross-checked against file size, per-tensor `n_dims` 1–4, dims > 0 with checked `ne` product, known type id, checked byte size, absolute offset within file, duplicate tensor name → error. `general.alignment` must be 1–1 MiB. Array metadata values are bounds-checked before allocation. |
| SafeTensors (single file, sharded dirs) | ✅ Supported | Header length capped (256 MiB) and checked against file size; string-aware brace matching (braces inside tensor names work); dtype must be present and known (unknown → hard error, no longer "warn and treat as F32"); shape must be present, 1–4 dims, all > 0, checked numel; `data_offsets` must satisfy `start ≤ end ≤ data section` and the span must **exactly** equal `dtype_size * numel`; truncated header, empty header, missing fields, trailing data → error. |
| Sharded GGUF (`--merge`) | ✅ Supported | `moe_engine convert --merge` verified byte-exact on synthetic shards and a real 2-shard Qwen2.5-7B Q4_K_M. |
| MoE SafeTensors packing | ✅ Supported | Verified byte-exact on tiny-random Qwen2Moe + Mixtral (both layouts); fused QKV split byte-exact on real Phi-3 weights; multi-file shards load via directory or shard path. |
| `convert --quantize` | ✅ Supported | Targets Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q4_1/Q5_0/Q5_1/Q8_1/Q2_K/Q3_K/Q8_K; refuses F32/IQ/mixed with a clear error. |

Malformed-file behavior is covered by Test 1f (bad magic, version 99,
n_tensors = 1,000,000, truncation, unknown type 999, zero dim,
duplicate name; 11 SafeTensors fixtures incl. unterminated header and
trailing garbage). Every case must fail with a useful error, never a
crash and never plausible-but-wrong output.

## 4. Tokenizer and chat templates

| Area | Behavior |
|---|---|
| `tokenizer.json` (BPE) | ✅ Supported. Vocab ids are strictly parsed (no sign, no `+`, no overflow) and capped at 1,048,576 — a larger id fails the load instead of allocating a huge vocab. Added-token ids above the cap are skipped. |
| `\u` escapes | ✅ Surrogate pairs combine (`\uD83D\uDE00` → U+1F600); lone surrogates become U+FFFD. |
| UTF-8 decoding | ✅ Strict. Continuation bytes, overlong forms, surrogates, values > U+10FFFF, and truncated sequences all yield U+FFFD advancing exactly 1 byte (Test 1e). `codepoint_to_utf8` maps surrogates and > U+10FFFF to U+FFFD. |
| BOS/EOS from metadata | ✅ Applied only when in range; out-of-range ids warn and are ignored (previously OOB vocab access was possible). Empty vocab after all fallbacks fails the load. |
| Scores / merges absent | ⚠️ Heuristic. Scores are fabricated from token length when no scores/merges exist. Documented as a heuristic, not a real BPE ranking. |
| Unigram / SentencePiece models | ⚠️ Fidelity limitation. Merging is BPE-style; Unigram probability-based merging is not implemented. |
| Byte-fallback vocab | ⚠️ Toy-only. The synthetic byte vocab is labeled toy output in the code and must not be presented as a real tokenizer. |
| Chat templates (Jinja) | ✅ Supported constructs: `{{ }}` expressions, `{% if %}` / `{% elif %}` / `{% else %}` / `{% endif %}`, `{% for %}` / `{% endfor %}`, `{% set %}` (incl. namespaces), `{# comments #}`, `{%- -%}` whitespace control, `tools` variable. |
| Chat templates — unsupported | ❌ Rejected at load. `{% block %}`, `{% macro %}`, `{% raw %}`, `{% filter %}`, `{% include %}`, `{% extends %}`, `{% import %}`, `{% do %}`, `{% call %}`, `{% with %}`, unclosed `{{`/`{%`/`{#`, unbalanced `{% for %}`/`{% if %}`, stray enders, `{% elif %}`/`{% else %}` outside `{% if %}`, empty statements. Previously these rendered literally into the prompt. On rejection the template is cleared and the per-arch fallback template is used, with a stderr warning. (Test 1e.) |
| Stop tokens / special tokens | ✅ `eos_id_` (and `bos_id_`) are honored by `generate_stream`; stop sequences are checked per generated token. Special tokens from `tokenizer.json` `added_tokens` are loaded when their ids are in range. |

## 5. Sampling and generation

Validated in the sampler/transformer layer (`validate_sampler_config`,
`sample_next_token`, `generate_stream`, `verify_speculative_batch`):

| Parameter | Rule |
|---|---|
| `temperature` | Must be finite and ≥ 0. NaN/negative/Inf → rejected by `generate_stream`/`verify_speculative_batch` (no tokens); `sample_next_token` defensively falls back to greedy (0.0). |
| `top_p` | Must be in [0, 1]. Out of range → 1.0 (nucleus off). |
| `top_k` | uint32; 1 = greedy. |
| `repetition_penalty` | Must be finite and > 0. 0/NaN/Inf → 1.0 (no division by zero). |
| `max_new_tokens` | ≤ 16,384. Larger → rejected (no tokens). |
| Draft tokens (speculative) | ≤ 64. More, or an out-of-vocab draft token, → empty verification result (draft not accepted). |
| Empty logits | Generation stops (no token-0 loop). All-NaN logits → token 0. |
| Cancellation | `on_token` returning false stops the run; the decoder-tail flush is suppressed after a cancel (Test 1f asserts exactly one callback invocation). |
| Seed | `SamplerConfig.seed` (default 42); per-thread RNG, seeded lazily. |

Server-side validation also exists in `src/server.cpp`
(`validate_sampler_params`) — the engine checks above are
defense-in-depth so a bad value can never reach the sampler through
any caller. `cache_slot` is validated by the server as an integer in
`[0, 63]` (400 otherwise); the engine independently rejects any slot
outside user `0..63` plus the internal scratch ids.

## 6. Concurrency model

Explicitly: `engine_mtx_` is a plain `std::mutex` (not recursive) that
serializes every `forward_token`. `get_or_create_slot` must only be
called with the lock held. `generate_stream` /
`verify_speculative_batch` hold it only around slot-state accesses, so
two generations on the **same** slot must not run concurrently — the
caller / HTTP server serializes per slot. Different slots are fully
isolated: separate KV caches, separate token histories, no shared
mutable state. Slot ids: user slots **0–63**; internal scratch slots
9998 (embeddings), 9999 (SystemOne), 8888 (diffusion) are allowed by
the engine but are not user-reachable through the server. Slot
isolation, reset independence, and two concurrent generations on
different slots are covered by Test 1f. There is no bounded queue:
concurrent requests serialize on the mutex; a request that cannot get
a slot does not silently share another slot's state.

## 7. KV cache

- Capacity is computed from hyperparameters and available RAM **before**
  allocation; unreasonable dimensions (kv_dim 0, n_layers 0, sequence
  length that would overflow the allocation) throw instead of
  allocating a wrapped-size cache.
- If available RAM cannot be determined (`avail_ram == 0`), the engine
  assumes 2 GiB with a warning instead of skipping the RAM guard (the
  old code allocated the full 32,768-token cache in that case).
- Capacity is re-guarded after the layer loop with the **final** kv_dim
  (per-layer overrides can raise it).
- Compaction keeps sink tokens (4) + the most recent `keep_recent`
  tokens in **both** the KV cache and `token_history` (Test 1f asserts
  history `{1,2,3,4,16,17,18,19,20}` after compacting 20 tokens with
  keep 5). `keep_recent` is clamped before use — the old code could
  underflow `seq_pos - keep_recent` to a huge uint32 and memmove out
  of bounds.
- A failed compaction is re-checked: if `seq_pos >= capacity` after
  compacting, `forward_token` returns empty logits rather than writing
  out of bounds.
- Multiple slots: each slot has its own paged KV cache; eviction and
  pinning operate per slot and cannot touch another slot's buffers.

## 8. Known limitations (do not present as supported)

1. **BF16 norms/biases are ignored.** The forward pass only applies
   F32 norms/biases. All-BF16 SafeTensors models (e.g. `qwen2moe-tiny`,
   `mixtral-tiny`) load with a stderr warning and run with identity
   norms. This is a documented limitation, not a silent success.
2. **No batched prefill.** Requests serialize on `engine_mtx_`;
   throughput-bound matmul dominates (96% of prefill, 82% of decode).
   Batched prefill was assessed and deliberately not landed (ceiling
   ≈1.35×, needs a layer-loop rewrite with silent-corruption risk).
3. **Tool calling is single-turn.** `tools` are accepted and parsed
   (`<tool_call>` / `[TOOL_CALLS]` / bare JSON), but multi-turn `tool`
   role / `tool_calls` history messages are not parsed from requests.
   `parse_chat_messages` also drops any message whose `content` is empty
   or null, so an assistant turn that only carries `tool_calls` never
   reaches the template.
4. **Qwen3 needs a large token budget.** Thinking consumes the budget;
   `max_tokens ≤ 64` yields an empty answer. Use ≥ 128.
5. **No trending-audio / no real video generation.** Reels built from
   stills are a product-level workaround, not an engine capability.
   (Not an engine limitation per se — recorded here so it is not
   claimed as one.)
6. **Vulkan is detection-only.** Device query exists; no compute kernels
   are implemented. Inference always runs on CPU.
7. **Web UI API key is memory-only** (no `localStorage` persistence since
   the security workstream). The key is re-entered each session and cleared
   on reload; in-page XSS could still read it from memory while loaded.
8. **IQ quantizers are simplified** (no importance matrix) and IQ
   dequantizers drop remainders on direct calls (see §2).

### Fixed (kept for audit trail)

- **Expert short-read detection** — `read_expert_sync` now sets
  `StreamedExpertWeights.ok`; `forward_token` refuses to admit a failed
  read into the expert cache and falls back to mmap (or skips).
- **Server `cache_slot` range** — server and engine both enforce user
  slots `[0, 63]` (server returns 400; engine rejects out-of-range).

## 9. Tested models / fixtures (local only, gitignored)

`~/workspace/jxai/models/real/` — Qwen2.5-0.5B Q4_0, Llama-3.2-1B
Q4_K_M, Qwen2.5-0.5B Q4_K_M, Qwen3-0.6B Q4_K_M + Q2_K, Gemma-3-1b-it
Q4_K_M, Phi-3-mini Q2_K GGUF + `phi3-mini-4k/` SafeTensors directory,
Mistral-7B Q2_K, DeepSeek-LLM-7B Q2_K (arch `llama`), Qwen2.5-7B
Q4_K_M shards + merged file, `qwen2moe-tiny/`, `mixtral-tiny/`
(all-BF16 SafeTensors, packing fixtures).

Offline self-test fixtures: Tests 1e/1f in `src/main.cpp`
(`moe_engine test`) — quant edge cases, UTF-8 boundaries,
tokenizer.json fixtures, chat-template validator cases, sampler
guards, malformed GGUF/SafeTensors fixtures, slot/sampler/generation
guards, KV compaction, slot isolation, concurrent slots. No downloads,
no network.
