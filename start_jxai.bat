@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title JXAI - Native Engine

:: Resolve moe_engine.exe: packaged zip (same folder) or local build tree.
set "ENGINE="
if exist "%~dp0moe_engine.exe" set "ENGINE=%~dp0moe_engine.exe"
if not defined ENGINE if exist "%~dp0build\Release\moe_engine.exe" set "ENGINE=%~dp0build\Release\moe_engine.exe"
if not defined ENGINE if exist "%~dp0build\moe_engine.exe" set "ENGINE=%~dp0build\moe_engine.exe"
if not defined ENGINE (
    echo [ERROR] moe_engine.exe not found next to this script or under build\.
    echo Download a Windows release zip, or build with MSVC:
    echo   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    echo   cmake --build build --config Release --parallel
    pause
    exit /b 1
)

:: Web UI dir: packaged web\dist, else repo web\dist.
set "WEBDIR="
if exist "%~dp0web\dist\index.html" set "WEBDIR=%~dp0web\dist"
if not defined WEBDIR if exist "%~dp0web\calculator\index.html" set "WEBDIR=%~dp0web\calculator"
if not defined WEBDIR (
    echo [ERROR] No web UI found. Expected web\dist\index.html in this folder.
    pause
    exit /b 1
)

if defined MOE_MODEL (
    set "MODEL=%MOE_MODEL%"
) else (
    set "MODEL=models\real\qwen2.5-0.5b-instruct-q4_0.gguf"
)
if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    echo.
    echo Download one into models\real\ first, e.g.:
    echo   "%ENGINE%" pull Qwen/Qwen2.5-0.5B-Instruct-GGUF --out models\real
    echo then re-run start_jxai.bat, or set MOE_MODEL to your .gguf path.
    pause
    exit /b 1
)

set "ENGINE_HOST=127.0.0.1"
set "ENGINE_PORT=8001"

echo ===================================================================
echo   JXAI native engine
echo   Model:  %MODEL%
echo   UI:     http://%ENGINE_HOST%:%ENGINE_PORT%/
echo ===================================================================
echo.

"%ENGINE%" serve --model "%MODEL%" --host "%ENGINE_HOST%" --port "%ENGINE_PORT%" --web-dir "%WEBDIR%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [ERROR] Engine exited with code %RC%.
    pause
)
exit /b %RC%
