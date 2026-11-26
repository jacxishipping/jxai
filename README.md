# JXAI — Native C++ Inference Engine

JXAI is a local inference product built around a custom native C++ engine
(`moe_engine`): GGUF/SafeTensors model loading, a C++20 SIMD transformer
(AVX2 with runtime dispatch, optional AVX-512), an OpenAI-compatible HTTP
server with SSE streaming, and a React web UI served by the engine itself.
There is also a Tauri desktop shell that spawns the same engine.

**The native engine is the default backend.** The normal workflow
(`./start_jxai.sh`) builds or locates `moe_engine`, starts
`moe_engine serve` on `127.0.0.1:8001`, waits for a successful `/health`
check (60 s), and serves the project's own web UI from the same origin.
It never starts Ollama, never depends on an Ollama installation, and never
silently falls back to one — if the engine fails to start, the launcher
prints the failure and the log tail and exits. The Ollama proxy
(`jxai_gateway.py`) is a separate, optional integration; see
[Optional Ollama proxy](#optional-ollama-proxy-not-the-native-engine).

What is verified, what is not, and what the engine refuses lives in
[`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md). Read it before
assuming a model, architecture, or quant type will work.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| C++20 compiler | GCC 11+ / Clang 14+ on Linux; MSVC on Windows. The Linux/macOS build is a **portable baseline with runtime AVX2 dispatch** — it runs on CPUs without AVX2. The MSVC build still requires AVX2 (see [Acceleration](#6-acceleration-and-io)). |
| CMake ≥ 3.20 | `cmake -S . -B build` |
| Python 3 | Used by `start_jxai.sh` to read `moe_config.json`. No Python runtime is needed for inference. |
| Node.js 22 + npm | Only to build the web UI (`cd web && npm ci && npm run build`). The launcher builds it automatically if the bundle is missing. |
| A real GGUF model | The engine refuses to start without one — it will not silently forge a toy model as a backend. See [Model setup](#4-model-setup). |
| Linux (primary) / Windows | Linux is the verified platform. Windows launchers exist (`run_studio.bat`, `run_open_webui.bat`, `install.bat`) but **were rewritten and have not been executed on a Windows host**; treat Windows as best-effort until someone runs them there. |

No Ollama installation is needed for the default path. No GPU is needed;
inference runs on CPU (see [Acceleration](#6-acceleration-and-io)).

---

## 2. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Produces build/moe_engine
```

Windows (MSVC):

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
# Produces build\Release\moe_engine.exe
```

Build the web UI (the launcher does this for you if `web/dist` is missing):

```bash
cd web && npm ci && npm run build
```

---

## 3. Run — the default path

```bash
./start_jxai.sh
```

What it does, in order:

1. Resolves a model: `--model <path>` → `MOE_MODEL` → `moe_config.json`
   `"model_path"` → `models/real/qwen2.5-0.5b-instruct-q4_0.gguf`.
   If no model file exists it **refuses to start** and tells you how to get
   one (see [Model setup](#4-model-setup)).
2. Builds `build/moe_engine` if the binary is missing (unless `--no-build`).
3. Builds the web UI bundle if `web/dist/index.html` is missing.
4. Starts `moe_engine serve --model <model> --host 127.0.0.1 --port 8001
   --web-dir web/dist`, logging to `build/jxai-engine.log`.
5. Waits up to 60 s for `/health` to report `{"status":"ok"}`. If the engine
   exits early or never becomes healthy, it prints the last 30 lines of the
   log and exits — no fallback to anything else.
6. If a healthy engine is already on the port it reuses it instead of
   starting a second one; if something else holds the port it refuses.
7. Ctrl-C stops the engine cleanly (up to 5 s grace, then forced) and
   releases the port. No orphaned server is left behind.

When it is up:

```
Backend:  native C++ engine (moe_engine serve)
Web UI:   http://127.0.0.1:8001/
Engine:   http://127.0.0.1:8001/health
Chat API: http://127.0.0.1:8001/v1/chat/completions
```

Useful options: `./start_jxai.sh --model <path> --port <n> --host <addr>
--no-build`. `--public` was removed: this launcher never starts Open WebUI
or a public tunnel.

### Manual serve (no launcher)

```bash
./build/moe_engine serve --model models/real/qwen2.5-0.5b-instruct-q4_0.gguf \
  --host 127.0.0.1 --port 8001 --web-dir web/dist
```

Note: `serve` without `--port` defaults to **8000**; the launcher, the
desktop shell, and `moe_config.json` all use **8001**. Pass `--port 8001`
explicitly to match them.

### Interactive CLI chat

```bash
./build/moe_engine chat --model models/real/qwen2.5-0.5b-instruct-q4_0.gguf --max-tokens 128
```

### Linux terminal UI (`jxai_term`)

Full-screen TUI client for Linux/macOS. Talks to `moe_engine serve` over
HTTP/SSE (multi-turn history, live tok/s, slash commands). No ncurses.

```bash
./start_jxai_term.sh
# or, with an engine already running on :8001:
cmake --build build -j --target jxai_term && ./build/jxai_term
```

Keys: Enter send · Ctrl+N newline · Esc abort stream · PgUp/PgDn scroll.
Commands: `/help` `/clear` `/temp` `/tokens` `/slot` `/system` `/retry` `/quit`.

### Desktop (Tauri)

The desktop shell (`desktop/`) spawns the same native engine
(`moe_engine serve --port 8001 --web-dir web/dist`) and opens the built-in
UI. It does not use Ollama.

---

## 4. Model setup

### Download a model (Hugging Face)

```bash
./build/moe_engine pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
# or a specific file:
./build/moe_engine pull Qwen/Qwen2.5-0.5B-Instruct-GGUF Qwen2.5-0.5B-Instruct-Q4_0.gguf
./build/moe_engine pull --list   # list downloadable files for a repo
```

- Downloads are **HTTPS only**; `http://` URLs are rejected.
- `HF_TOKEN` (for gated/private repos) is attached **only to
  huggingface.co hosts** — never to arbitrary user-supplied URLs or
  redirect destinations.
- Downloaded files are validated (GGUF/SafeTensors structure) before the
  engine will load them; malformed files fail with an error instead of
  producing plausible-but-wrong output. See
  [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md) §3 for what is
  checked.
- Models land under `models/real/` (gitignored). Point the launcher at one
  with `--model`, `MOE_MODEL`, or `moe_config.json` `"model_path"`.

### Use a local file

Place any GGUF (v2/v3) or SafeTensors file (single file or sharded
directory) anywhere and pass it with `--model`. Sharded GGUF can be merged
first:

```bash
./build/moe_engine convert --merge <shard1.gguf> <shard2.gguf> -o merged.gguf
./build/moe_engine convert --quantize model.gguf --to Q8_0 -o model-q8_0.gguf
./build/moe_engine info model.gguf   # architecture, quant types, warnings
```

### Which models work?

See [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md) — architectures,
quantization types, tokenizer/template behaviour, and the real model each
row was verified against. Quick summary of what has actually been run
end-to-end (greedy, temperature 0):

| Model | Quant | 2+2 | Capital of France |
| :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B-Instruct | Q4_0 | "The answer to 2+2 is 4." ✅ | "The capital of France is Paris." ✅ |
| Llama-3.2-1B-Instruct | Q4_K_M | "The answer to 2+2 is 4." ✅ | "The capital of France is Paris." ✅ |
| Qwen3-0.6B | Q4_K_M | ✅ with a large budget — thinks first, needs &gt;128 tokens; at 254 tokens: "the answer is simply 4" | ✅ same caveat |
| Gemma-3-1b-it | Q4_K_M | "2 + 2 = 4" ✅ | "The capital of France is **Paris**." ✅ |
| Phi-3-mini-4k-instruct | Q2_K | Coherent but verbose; 32 tokens cut off before stating "4" ⚠️ | "The capital of France is Paris, and it serves as the country's most populous city…" ✅ |
| Mistral-7B-Instruct-v0.3 | Q2_K | ✅ | ✅ |

**Not verified:** true `deepseek`/`deepseek2` architecture (DeepSeek-LLM-7B's
GGUF reports arch `llama`, so it runs as llama — that says nothing about a
real deepseek-arch file). Unknown architectures are refused at load instead
of guessed.

---

## 5. API

Base URL is the engine itself (`http://127.0.0.1:8001` via the launcher).
All responses are JSON; errors are `{"error": "...", "message": "..."}`.

| Endpoint | Method | Auth | Notes |
| :--- | :--- | :--- | :--- |
| `/health` | GET | Open | Liveness probe: status, model, hardware, tier stats |
| `/v1/chat/completions` | POST | Key* | OpenAI-compatible; `stream: true` gives SSE with dual-channel reasoning deltas; `stream: false` returns one JSON object |
| `/v1/completions`, `/api/chat`, `/api/generate` | POST | Key* | Compatibility aliases for chat |
| `/v1/models` | GET | Open | Lists models; **exposes model filesystem paths** (deliberate, for model selection in the UI) |
| `/v1/models/load` | POST | Key* | Load a model by path |
| `/v1/models/pull` | POST | Key* | Pull a model from Hugging Face |
| `/v1/embeddings` | POST | Key* | Embeddings via the loaded model |
| `/v1/systemone` | POST | Key* | Zero-token categorical evaluation via LM-head logits (see below) |
| `/v1/images/generations` | POST | Key* | **Latent SVG visualisation, not image generation** (see below) |
| `/v1/config` | GET / POST | Key* (POST) | GET redacts the stored API key |
| `/v1/slots`, `/v1/slots/clear` | GET / POST | Key* (POST) | Paged KV-cache slot management |
| `/experts` | GET | Open | Live expert telemetry: tier, activations, LFRU score, pinning |
| `/profile` | GET | Open | Latency / throughput snapshot |
| `/` | GET | Open | Serves the compiled web UI (`web/dist`) |

\* Key = required when `MOE_API_KEY` is configured on the server, as
`Authorization: Bearer <key>` or `x-api-key: <key>`. Without a key the
response is `401 {"error":"unauthorized"}`. `GET /health` stays open.

```bash
# Chat (streaming)
curl -N -H "Authorization: Bearer $MOE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"stream":true,"temperature":0}' \
  http://127.0.0.1:8001/v1/chat/completions

# Chat (single JSON response)
curl -H "Authorization: Bearer $MOE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"stream":false,"temperature":0}' \
  http://127.0.0.1:8001/v1/chat/completions

# Tool calling (single-turn only — see Known limitations)
curl -H "Authorization: Bearer $MOE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"stream":false,"temperature":0,"max_tokens":128}' \
  http://127.0.0.1:8001/v1/chat/completions
```

Sampling parameters are validated server-side: `temperature` must be
finite and ≥ 0, `top_p` in [0, 1], `max_new_tokens` ≤ 16,384. Invalid
values are rejected, not silently accepted. (Qwen3 needs `max_tokens`
≥ 128 to finish its thinking block and answer.)

**SystemOne** (`/v1/systemone`) scores candidate options directly from
the LM-head logits — 0 tokens generated. Entropy and confidence shown in
the UI are computed in the browser from those probabilities; they are
derived, not calibrated. On the built-in 30-item sentiment benchmark the
small reference model scores **10/30 (33.3% = baseline)**, so treat
SystemOne output as a ranking signal, not a verified classifier.

**Latent visualisation** (`/v1/images/generations`) renders a 16×16 latent
SVG grid conditioned by a transformer forward pass. It is **not image
generation** and the UI labels it as a visualisation. Do not present its
output as a generated image.

**Expert telemetry** (`/experts`) is live: tier, activations, LFRU score,
and pinning come from the running engine. The "topic" label shown per
expert in the UI is **illustrative** — a fixed label assigned by
layer/expert index, not derived from the model.

---

## 6. Acceleration and I/O

Honest numbers only — see [Benchmarks](#8-benchmarks) for the measured
results and the command that produced them.

- **CPU dispatch:** portable baseline build with runtime AVX2 dispatch on
  Linux/macOS. The binary is safe on CPUs without AVX2. The MSVC build
  still requires AVX2 (Haswell+); on a non-AVX2 machine it may fault
  (SIGILL) — `moe_engine doctor` reports this.
- **AVX-512:** AVX-512 dot kernels exist for Q4_0/Q4_K/Q8_0, but they are
  **off by default**. On the VM where they were measured they were ~5×
  *slower* than AVX2 on prefill (287.6 vs 57.0 ms/forward), and they have
  no dedicated Test 1 assertions. Opt in with `MOE_AVX512=1` only if you
  measure a win on your hardware.
- **Vulkan:** device detection only. No Vulkan compute kernels are
  implemented; inference runs on CPU. Do not expect GPU acceleration.
- **I/O (Linux):** synchronous `pread()` + `mmap` with background prefetch
  of the next layer's experts while the current layer computes. This is
  **not** async I/O, io_uring, or DMA — older docs that said otherwise
  were wrong. The `bypass_io` / `direct_dma` keys in `moe_config.json`
  are legacy and do not change Linux I/O behaviour.
- **Prefetch:** bounded background prefetch; cache-hit, read, wait, and
  latency figures in `/profile` and the UI are measured counters, not
  estimates.

---

## 7. Web UI

Served by the engine itself at `http://127.0.0.1:8001/` — same origin, so
the UI calls relative `/v1/*` and `/health` URLs. No separate frontend
server is needed.

- **Chat** — streaming chat with dual-channel reasoning, per-token
  telemetry, and model/slot info. Handles disconnects, cancellation,
  empty responses, malformed chunks, and server errors without hanging.
- **3D Cortex Atlas** — live expert routing map from `/experts`.
- **Profiler** — measured latency/throughput from `/profile`.
- **SystemOne** — zero-token categorical evaluation (see §5).
- **Latent Visualisation** — SVG-grid visualisation (see §5), not image
  generation.
- **Settings** — API key, system prompt, sampler settings, language, and
  a connection card showing backend (native engine), loaded model, and
  max context.

**API key handling:** the key is **memory-only**. It is kept in the page's
JavaScript memory, never written to `localStorage`, `sessionStorage`, a
cookie, or disk, and it is cleared when the page reloads — re-enter it
each session. Tradeoff: an in-page script (XSS) could still read it from
memory while the page is loaded. Keep the engine loopback-only (the
default) and do not load untrusted pages in the same browser profile.

---

## 8. Benchmarks

Reproducible with the built-in command (no model downloads — use a local
model):

```bash
OMP_NUM_THREADS=1 ./build/moe_engine bench --model models/real/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "What is 2+2?" --tokens 16
```

Measured on this Linux VM (AMD EPYC, pinned single thread,
`OMP_NUM_THREADS=1`), model Qwen2.5-0.5B-Instruct Q4_0:

| Metric | Measured |
| :--- | :--- |
| Prefill | ~49–57 ms/forward |
| Decode | ~13 tok/s |

Do not compare these numbers to other hardware without naming the
hardware and settings. Do not quote toy-model throughput as real-model
performance: the self-test's forged 1.3 MB toy model is a correctness
fixture, not a benchmark. Older README figures (e.g. "2247 tok/s",
"~5 tok/s") were either the toy self-test or stale and have been removed.

---

## 9. Security

- **Loopback by default.** `moe_engine serve` binds `127.0.0.1` unless
  `--host` says otherwise. Binding a non-loopback address with no key
  configured is **refused** unless you pass `--allow-insecure-remote` —
  that flag means anyone who can reach the port can load models, download
  files, and overwrite the config.
- **API key.** Set `MOE_API_KEY` in the environment, or `"api_key"` in
  `moe_config.json` (env wins). Never commit a real key. When a key is
  configured, every mutating, model-loading, model-pulling, and inference
  endpoint requires it (see §5). `GET /health` stays open as the liveness
  probe; `GET /v1/config` redacts the stored key. Read-only GET endpoints
  (model list, profiles, telemetry, static UI) do not require a key —
  treat the server as a trusted-local service, not a public one.
- **CORS.** Same-origin by default: no `Access-Control-Allow-Origin`
  header is emitted at all. Allow additional origins explicitly with
  `MOE_CORS_ORIGINS` (comma-separated exact origins) or `"cors_origins"`
  in `moe_config.json`.
- **Model downloads.** HTTPS only; `HF_TOKEN` is attached only to
  huggingface.co hosts. Downloaded files are structure-validated before
  loading.
- **Web UI key.** Memory-only (see §7). The key never touches browser
  storage or the frontend bundle.
- **No secrets** in source, logs, frontend bundles, test fixtures, or
  docs. `MOE_API_KEY` is passed by environment only.

**Known remaining exposures (do not ignore):**
- `HF_TOKEN` is passed to `curl` on the command line during pulls, so it
  is visible in the process list (`ps`) while a pull runs.
- `/v1/models` exposes model filesystem paths (deliberate, for model
  selection).
- Piping CLI pull output to a closed stdout pipe can still SIGPIPE.
- The Windows WinINet model-pull path has not been compile-tested.
- `cache_slot` must be an integer in `[0, 63]` (server 400; engine also
  rejects out-of-range slots).

---

## 10. Tests

```bash
# Native engine: 11-section self-test (quant/dot, RoPE, GGUF parser,
# toy end-to-end, malformed-input guards, security section 11).
# Synthetic vectors + the forged toy model only — it proves the code
# paths run; it proves nothing about real-model accuracy.
./build/moe_engine test

# Web UI: 10 tests (offline unit tests + live API-contract tests).
# Live tests need a real engine + model; without them they skip with a
# clear message instead of failing.
cd web && npm test
MOE_ENGINE_BIN=../build/moe_engine \
  MOE_MODEL=../models/real/qwen2.5-0.5b-instruct-q4_0.gguf \
  npm test

# Hardware / SIMD diagnostics
./build/moe_engine doctor
./build/moe_engine doctor --quick
```

CI: `.github/workflows/ci.yml` runs `./build/moe_engine test` (11/11,
no model download) and `cd web && npm test` (live API cases skip without
a model) on push/PR to `main`.

---

## 11. Known limitations

Full detail in [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md) §8.
The short list:

1. **No batched prefill.** Requests serialize on a mutex; matmul
   dominates (96% of prefill, 82% of decode).
2. **Tool calling is single-turn.** Multi-turn `tool` role / `tool_calls`
   history messages are not parsed; empty-content assistant turns are
   dropped before templating.
3. **Non-F32 (F16/BF16) norms/biases are ignored.** All-half SafeTensors
   models load with a warning and run with identity norms.
4. **Qwen3 needs a large token budget** (`max_tokens` ≥ 128) to finish
   thinking and answer.
5. **SystemOne at baseline** on the built-in sentiment set (10/30).
6. **IQ quantizers are simplified** (no importance matrix); IQ
   dequantizers drop remainders on direct calls.
7. **Vulkan is device detection only**; no GPU compute kernels.
8. **Windows launchers unexecuted** on a Windows host; Windows WinINet
   pull path not compile-tested. Desktop UI config still defaults to
   `models/test_qwen.gguf` (launchers refuse it).
9. **True deepseek/deepseek2 arch unverified** (see §4).
10. **No default model in a clean clone** — live API tests / launcher
    need a downloaded GGUF or `MOE_MODEL`.

---

## Optional Ollama proxy — NOT the native engine

`jxai_gateway.py` is a **separate, optional** OpenAI-compatible proxy to
Ollama, for users who explicitly want to run inference through Ollama +
Open WebUI. It is **not** part of the default path, not started by
`start_jxai.sh`, and not the native engine — its `/health` reports
`"ollama_connected"` and every chat request is forwarded to the Ollama
daemon on `127.0.0.1:11434`. It requires Ollama to be installed and
running. Do not confuse it with `moe_engine serve`.

```bash
JXAI_ALLOW_OLLAMA_PROXY=1 python3 jxai_gateway.py --port 8002
# Opt-in env is required. Default port 8001 is the engine's — pick another
# if moe_engine is already running. CORS mirrors MOE_CORS_ORIGINS (empty =
# same-origin only; never Access-Control-Allow-Origin: *).
```

The old `run_open_webui.bat` starts the native engine on :8001 and Open
WebUI on :8080 as an optional UI choice; the built-in UI served by the
engine itself remains the default.

---

## License

MIT — see [LICENSE](LICENSE).
