@echo off
title JXAI - Native Engine + Open WebUI (optional UI)
echo ===================================================================
echo   JXAI: NATIVE C++ ENGINE (:8001) + OPEN WEBUI (:8080, optional)
echo   The engine is the backend. Open WebUI is just one UI choice;
echo   the built-in UI is served by the engine itself on :8001.
echo ===================================================================
echo.

cd /d "%~dp0"

:: 1. Pick a real GGUF model. Never start the engine without one: it would
::    auto-forge a synthetic toy model, which is not a real backend.
set "MODEL=models\real\qwen2.5-0.5b-instruct-q4_0.gguf"
if not exist "%MODEL%" (
    if exist "models\test_qwen.gguf" (
        set "MODEL=models\test_qwen.gguf"
    ) else (
        echo [ERROR] No GGUF model found.
        echo Download one first, e.g.:
        echo   .\build\moe_engine.exe pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
        echo then re-run this script.
        pause
        exit /b 1
    )
)

:: 2. Ensure the native engine is running on port 8001
curl -s http://127.0.0.1:8001/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [1/2] Starting native C++ engine on http://127.0.0.1:8001 ...
    echo       Model: %MODEL%
    start "JXAI Native Engine" /min cmd /c ".\build\moe_engine.exe serve --model %MODEL% --host 127.0.0.1 --port 8001 --web-dir web\dist ^> build\jxai-engine.log 2^>^&1"
    set /a WAITED=0
    :wait_engine
    curl -s http://127.0.0.1:8001/health >nul 2>&1
    if %ERRORLEVEL% EQU 0 goto engine_up
    set /a WAITED+=1
    if %WAITED% GEQ 60 (
        echo [ERROR] Native engine did not become healthy within 60s.
        echo See build\jxai-engine.log for the failure. No fallback to
        echo Ollama is attempted -- fix the engine failure above.
        pause
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
    goto wait_engine
    :engine_up
    echo [1/2] Native engine healthy on http://127.0.0.1:8001
) else (
    echo [1/2] Native engine already healthy on http://127.0.0.1:8001
)

:: 3. Point Open WebUI at the native engine's OpenAI-compatible API.
::    Auth: the engine requires MOE_API_KEY on API calls when it is set;
::    pass the same key through. Never hardcode a key here.
set WEBUI_NAME=JXAI (native engine)
set WEBUI_AUTH=True
set OPENAI_API_BASE_URL=http://127.0.0.1:8001/v1
set OPENAI_API_KEY=%MOE_API_KEY%
set PORT=8080

if "%MOE_API_KEY%"=="" (
    echo [NOTE] MOE_API_KEY is not set: engine auth is OFF (loopback only).
    echo        Set MOE_API_KEY before starting the engine to require a key.
) else (
    echo [NOTE] MOE_API_KEY is set: Open WebUI will send it to the engine.
)

echo [2/2] Launching Open WebUI on http://127.0.0.1:8080 ...
echo       (Built-in UI is also available at http://127.0.0.1:8001/ )
where open-webui >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    open-webui serve --port 8080
) else (
    echo Open WebUI is not installed yet. Installing via pip...
    python -m pip install open-webui
    open-webui serve --port 8080
)
