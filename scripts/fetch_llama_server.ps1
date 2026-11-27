# Fetch a prebuilt llama.cpp llama-server.exe into tools\llama for hybrid GPU.
# PCs without a GPU do not need this — native moe_engine CPU is the default.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Out = Join-Path $Root "tools\llama"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$Tag = if ($env:LLAMA_CPP_TAG) { $env:LLAMA_CPP_TAG } else { "b5758" }
$Asset = "llama-$Tag-bin-win-cuda-cu12.4-x64.zip"
# Prefer CUDA build when available; fall back to CPU/Vulkan-capable win zip if set.
if ($env:LLAMA_CPP_ASSET) { $Asset = $env:LLAMA_CPP_ASSET }
$Url = "https://github.com/ggerganov/llama.cpp/releases/download/$Tag/$Asset"
$Tmp = Join-Path $env:TEMP ("llama-fetch-" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
  Write-Host "Downloading $Url"
  $Zip = Join-Path $Tmp $Asset
  Invoke-WebRequest -Uri $Url -OutFile $Zip
  Expand-Archive -Path $Zip -DestinationPath (Join-Path $Tmp "extract") -Force
  $Found = Get-ChildItem -Path (Join-Path $Tmp "extract") -Recurse -Filter "llama-server.exe" |
    Select-Object -First 1
  if (-not $Found) {
    throw "llama-server.exe not found in archive $Asset"
  }
  Copy-Item -Force $Found.FullName (Join-Path $Out "llama-server.exe")
  $BinDir = $Found.DirectoryName
  Get-ChildItem -Path $BinDir -File | ForEach-Object {
    if ($_.Extension -in @(".dll", ".exe") -and $_.Name -ne "llama-server.exe") {
      Copy-Item -Force $_.FullName (Join-Path $Out $_.Name)
    }
  }
  Write-Host "Installed: $(Join-Path $Out 'llama-server.exe')"
  Write-Host "Use: moe_engine.exe serve --model <gguf> --backend auto"
  Write-Host "Or set LLAMA_SERVER to that path"
} finally {
  Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
