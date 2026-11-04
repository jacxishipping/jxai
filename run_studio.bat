@echo off
setlocal
cd /d "%~dp0"
title MoE Studio Launcher

echo ===================================================================
echo   NEXT-GEN MoE STREAMING INFERENCE ENGINE ^& STUDIO SUITE
echo ===================================================================
echo.

:: Check if server is responding on port 8000
curl -s http://127.0.0.1:8000/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [LAUNCH] Starting MoE Engine Server in background...
    if exist "models\test_qwen.gguf" (
        start "MoE Engine Server" /min "build\moe_engine.exe" serve --model "models\test_qwen.gguf" --port 8000 --web-dir "web\dist"
    ) else if exist "models\qwen2moe_stream_q4.gguf" (
        start "MoE Engine Server" /min "build\moe_engine.exe" serve --model "models\qwen2moe_stream_q4.gguf" --port 8000 --web-dir "web\dist"
    ) else (
        start "MoE Engine Server" /min "build\moe_engine.exe" serve --port 8000 --web-dir "web\dist"
    )
    timeout /t 2 /nobreak >nul
)

echo [LAUNCH] Launching MoE Studio C++ Direct2D Native Desktop App...
start "" "build\moe_studio.exe"
echo [OK] MoE Studio is running.
