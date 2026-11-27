JXAI for Windows (x64)
======================

This zip is a prebuilt package — you do NOT need CMake, MSVC, or a compiler.

Contents
--------
  moe_engine.exe   Native inference engine + OpenAI-compatible HTTP server
  moe_studio.exe   Direct2D desktop UI (optional)
  start_jxai.bat   Start the engine + open the web UI
  run_studio.bat   Start the engine + MoE Studio
  web\dist\        Built-in chat UI (served by the engine)
  models\real\     Put downloaded GGUF models here
  vcomp140.dll     OpenMP runtime (when bundled) for multi-core matmul

Requirements
------------
  - Windows 10/11 x64
  - CPU with AVX2 (Intel Haswell / AMD Excavator or newer)
  - A real GGUF model (not included — download separately)

IMPORTANT — GPU PCs
-------------------
Inference runs on the **CPU** (AVX2 + OpenMP). Vulkan is detection-only;
there are no CUDA / DirectML / Vulkan compute kernels yet. Your GPU will
stay idle — that is expected. Speed comes from:

  1. Using all CPU cores (OpenMP — default = all logical cores)
  2. Choosing a smaller / more quantized GGUF (e.g. Q4_K_M 1B–3B)
  3. Keeping context short for chat

Quick start
-----------
1. Unzip anywhere (e.g. Desktop\jxai-windows-x64).

2. Download a small model into models\real\, for example in PowerShell
   from this folder:

     .\moe_engine.exe pull Qwen/Qwen2.5-0.5B-Instruct-GGUF --out models\real

   Or copy any compatible .gguf into models\real\ and set MOE_MODEL.

3. Double-click start_jxai.bat
   - Web UI:  http://127.0.0.1:8001/
   - Health:  http://127.0.0.1:8001/health
   - Chat:    http://127.0.0.1:8001/v1/chat/completions

4. Optional: double-click run_studio.bat for the native Direct2D app.

Speed tips
----------
  - Leave OMP_NUM_THREADS unset (engine uses all cores), or set it
    explicitly, e.g.  set OMP_NUM_THREADS=16
  - Prefer Q4_K_M / Q4_0 over Q8 / F16 on CPU
  - Prefer ~0.5B–3B instruct models for interactive chat
  - First load of a large GGUF prefaults pages (can take a few seconds);
    later tokens should be smoother

Notes
-----
  - Models are large; keep them under models\real\ next to the .exe files.
  - First pull needs network access (Hugging Face).
  - Antivirus may scan new .exe files the first time — that is normal.
  - Source / issues: https://github.com/jacxishipping/jxai
