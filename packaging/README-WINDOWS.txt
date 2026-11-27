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

Requirements
------------
  - Windows 10/11 x64
  - CPU with AVX2 (Intel Haswell / AMD Excavator or newer)
  - A real GGUF model (not included — download separately)

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

Notes
-----
  - Models are large; keep them under models\real\ next to the .exe files.
  - First pull needs network access (Hugging Face).
  - Antivirus may scan new .exe files the first time — that is normal.
  - Source / issues: https://github.com/jacxishipping/jxai
