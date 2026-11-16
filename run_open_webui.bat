@echo off
title JXAI - MoE Stream Engine + Open WebUI Stack
echo ===================================================================
echo   JXAI HYBRID STACK: MoE STREAM ENGINE (:8000) + OPEN WEBUI (:8080)
echo ===================================================================
echo.

cd /d "%~dp0"

:: 1. Ensure MoE Stream Engine is running on port 8000
curl -s http://127.0.0.1:8000/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [1/2] Starting MoE Stream Engine on http://127.0.0.1:8000 ...
    if exist "models\qwen2.5-1.5b-instruct-q4_0.gguf" (
        start "MoE Stream Engine" /min cmd /c ".\build\moe_engine.exe serve --model models/qwen2.5-1.5b-instruct-q4_0.gguf --port 8000 --web-dir web/dist"
    ) else (
        start "MoE Stream Engine" /min cmd /c ".\build\moe_engine.exe serve --model models/qwen_run.gguf --port 8000 --web-dir web/dist"
    )
    timeout /t 2 /nobreak >nul
) else (
    echo [1/2] MoE Stream Engine already active on http://127.0.0.1:8000
)

:: 2. Configure Open WebUI environment variables to point at MoE Stream Engine
set WEBUI_NAME=JXAI MoE Suite
set WEBUI_AUTH=False
set ENABLE_OLLAMA_API=True
set OLLAMA_BASE_URL=http://127.0.0.1:8000
set OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
set OPENAI_API_KEY=sk-moe-local
set PORT=8080

echo [2/2] Launching Open WebUI on http://127.0.0.1:8080 ...
where open-webui >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    open-webui serve --port 8080
) else (
    echo Open WebUI is not installed yet. Installing via pip...
    python -m pip install open-webui
    open-webui serve --port 8080
)
