@echo off
setlocal
cd /d "%~dp0"
title JXAI - Native Engine + MoE Studio Launcher

echo ===================================================================
echo   JXAI: NATIVE C++ ENGINE (:8001) ^& MOE STUDIO
echo ===================================================================
echo.

:: Pick a real GGUF model. Never start the engine without one, and never
:: fall back to models\test_qwen.gguf (that file is the forged toy from
:: `moe_engine setup` / self-tests — not a real backend).
if defined MOE_MODEL (
    set "MODEL=%MOE_MODEL%"
) else (
    set "MODEL=models\real\qwen2.5-0.5b-instruct-q4_0.gguf"
)
if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    echo Download one first, e.g.:
    echo   .\build\moe_engine.exe pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
    echo then re-run with the file under models\real\, or set MOE_MODEL.
    echo Refusing to start with a synthetic toy model.
    pause
    exit /b 1
)

:: Native engine /health must include "architecture" (and must not be the
:: optional Ollama proxy, which reports ollama_connected).
set "HEALTH_OK=0"
for /f "delims=" %%H in ('curl -s http://127.0.0.1:8001/health 2^>nul') do set "HEALTH_BODY=%%H"
echo %HEALTH_BODY% | findstr /C:"\"status\":\"ok\"" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo %HEALTH_BODY% | findstr /C:"\"architecture\"" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo %HEALTH_BODY% | findstr /C:"\"ollama_connected\"" >nul 2>&1
        if %ERRORLEVEL% NEQ 0 set "HEALTH_OK=1"
    )
)
if "%HEALTH_OK%"=="0" (
    echo [LAUNCH] Starting native C++ engine on http://127.0.0.1:8001 ...
    echo          Model: %MODEL%
    start "JXAI Native Engine" /min "build\moe_engine.exe" serve --model "%MODEL%" --host 127.0.0.1 --port 8001 --web-dir "web\dist"
    set /a WAITED=0
    :wait_engine
    set "HEALTH_OK=0"
    for /f "delims=" %%H in ('curl -s http://127.0.0.1:8001/health 2^>nul') do set "HEALTH_BODY=%%H"
    echo %HEALTH_BODY% | findstr /C:"\"status\":\"ok\"" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo %HEALTH_BODY% | findstr /C:"\"architecture\"" >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            echo %HEALTH_BODY% | findstr /C:"\"ollama_connected\"" >nul 2>&1
            if %ERRORLEVEL% NEQ 0 set "HEALTH_OK=1"
        )
    )
    if "%HEALTH_OK%"=="1" goto engine_up
    set /a WAITED+=1
    if %WAITED% GEQ 60 (
        echo [ERROR] Native engine did not become healthy within 60s.
        echo No fallback is attempted -- check the engine output and retry.
        pause
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
    goto wait_engine
    :engine_up
    echo [OK] Native engine healthy on http://127.0.0.1:8001
) else (
    echo [OK] Native engine already healthy on http://127.0.0.1:8001
)

echo [LAUNCH] Launching MoE Studio C++ Direct2D Native Desktop App...
start "" "build\moe_studio.exe"
echo [OK] MoE Studio is running. Built-in web UI: http://127.0.0.1:8001/
