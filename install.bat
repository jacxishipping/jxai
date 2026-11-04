@echo off
setlocal enabledelayedexpansion

echo ===================================================================
echo   jxAI / MoE Stream Engine - Windows 1-Click Installation Setup
echo ===================================================================
echo.

:: 1. Validate CMake
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake is not found in PATH.
    echo Please install CMake or add it to your PATH and re-run install.bat.
    pause
    exit /b 1
)

:: 2. Validate Compiler (g++ / clang++ / cl)
where g++ >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] C++ compiler (g++) is not found in PATH.
    echo Please ensure MinGW-w64 or Clang is installed and in PATH.
    pause
    exit /b 1
)

echo [1/4] Configuring Build Directory with CMake...
if not exist "build" (
    cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
    if %ERRORLEVEL% neq 0 (
        cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
    )
)

echo [2/4] Building MoE Stream Engine & Native Direct2D Studio...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed! Check compiler logs above.
    pause
    exit /b 1
)

echo [3/4] Initializing Hardware Qualifications and Profile...
.\build\moe_engine.exe setup
if %ERRORLEVEL% neq 0 (
    echo [WARNING] moe_engine setup returned non-zero. Continuing...
)

echo [4/4] Creating Desktop Shortcut...
set "CURRENT_DIR=%~dp0"
set "STUDIO_EXE=%CURRENT_DIR%build\moe_studio.exe"
set "ICON_EXE=%STUDIO_EXE%"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\MoE Studio.lnk"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%STUDIO_EXE%'; $s.WorkingDirectory = '%CURRENT_DIR%'; $s.IconLocation = '%STUDIO_EXE%,0'; $s.Description = 'MoE Studio - Hybrid-Tiered Mixture-of-Experts Control Suite'; $s.Save()"

if exist "%SHORTCUT_PATH%" (
    echo [SUCCESS] Created shortcut on your Desktop: "MoE Studio"
)

echo.
echo ===================================================================
echo   INSTALLATION COMPLETED SUCCESSFULLY!
echo ===================================================================
echo.
echo You can now run MoE Studio from:
echo   1. Desktop Shortcut: "MoE Studio"
echo   2. Command Line:     .\build\moe_studio.exe
echo   3. Background API:   .\build\moe_engine.exe serve --model models/test_qwen.gguf --port 8000
echo.
echo Launching MoE Studio First-Run Setup Wizard...
start "" "%STUDIO_EXE%" --wizard 0

