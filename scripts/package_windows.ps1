# Package a portable jxai-windows-x64.zip from a local MSVC build tree.
# Usage (from repo root, after Release build + web build):
#   pwsh -File scripts/package_windows.ps1
#   pwsh -File scripts/package_windows.ps1 -OutDir dist -BuildDir build

param(
    [string]$BuildDir = "build",
    [string]$OutDir = "dist",
    [string]$ZipName = "jxai-windows-x64.zip"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Find-Exe([string]$Name) {
    $candidates = @(
        (Join-Path $Root "$BuildDir\Release\$Name"),
        (Join-Path $Root "$BuildDir\$Name"),
        (Join-Path $Root $Name)
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    return $null
}

$engine = Find-Exe "moe_engine.exe"
if (-not $engine) { throw "moe_engine.exe not found under $BuildDir (build Release first)" }
$studio = Find-Exe "moe_studio.exe"

$webDist = Join-Path $Root "web\dist\index.html"
if (-not (Test-Path $webDist)) {
    throw 'web\dist\index.html missing - run: cd web; npm ci; npm run build'
}

$stage = Join-Path $Root "$OutDir\jxai-windows-x64"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage "web\dist") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage "models\real") | Out-Null

Copy-Item $engine (Join-Path $stage "moe_engine.exe")
if ($studio) {
    Copy-Item $studio (Join-Path $stage "moe_studio.exe")
} else {
    Write-Warning "moe_studio.exe not found - packaging engine-only zip"
}

# OpenMP runtime (vcomp*.dll) — required when moe_engine is built with /openmp.
# Prefer the VS toolchain copy next to cl.exe; fall back to System32.
$vcompCandidates = @()
try {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) {
        $vsRoot = Split-Path (Split-Path (Split-Path $cl.Source))
        $vcompCandidates += Get-ChildItem -Path $vsRoot -Recurse -Filter "vcomp140.dll" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }
} catch {}
$vcompCandidates += @(
    "$env:SystemRoot\System32\vcomp140.dll",
    "$env:SystemRoot\SysWOW64\vcomp140.dll"
)
$vcomp = $vcompCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if ($vcomp) {
    Copy-Item $vcomp (Join-Path $stage "vcomp140.dll")
    Write-Host "Bundled OpenMP runtime: $vcomp"
} else {
    Write-Warning "vcomp140.dll not found - OpenMP may require the VC++ redistributable"
}

Copy-Item -Recurse (Join-Path $Root "web\dist\*") (Join-Path $stage "web\dist\")
Copy-Item (Join-Path $Root "start_jxai.bat") $stage
Copy-Item (Join-Path $Root "run_studio.bat") $stage
Copy-Item (Join-Path $Root "packaging\README-WINDOWS.txt") (Join-Path $stage "README-WINDOWS.txt")

# Empty placeholder so the folder exists in the zip.
Set-Content -Path (Join-Path $stage "models\real\.gitkeep") -Value ""

New-Item -ItemType Directory -Force -Path (Join-Path $Root $OutDir) | Out-Null
$zipPath = Join-Path $Root "$OutDir\$ZipName"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path $stage -DestinationPath $zipPath -Force

Write-Host "Packaged: $zipPath"
Get-Item $zipPath | Format-List FullName, Length
