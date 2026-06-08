<#
run_all.ps1 - One-step launcher for this project (PowerShell)

Features:
 - Uses the project-local virtual environment `voice_env` (always calls its python.exe)
 - Optional: install dependencies (-InstallDeps)
 - Sets QT_QPA_PLATFORM_PLUGIN_PATH to avoid PyQt5 plugin errors
 - Launch GUI (default) or skip GUI (-NoGui)
 - Optional: playback tests (-TestPlay) that save logs

Usage examples:
    # Start GUI
    .\run_all.ps1

    # Install deps only
    .\run_all.ps1 -InstallDeps -NoGui

    # Skip GUI, run playback tests
    .\run_all.ps1 -NoGui -TestPlay
#>
param(
    [switch]$InstallDeps,
    [switch]$NoGui,
    [switch]$TestPlay
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "[run_all] Working directory: $ScriptDir"

$venvPath = Join-Path $ScriptDir 'voice_env'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path $venvPath)) {
    Write-Host "ERROR: 找不到虚拟环境 voice_env，请先创建或确认路径。" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: 找不到 voice_env 的 python 可执行文件： $pythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Green

# 可选：安装依赖
if ($InstallDeps) {
    Write-Host "[run_all] Installing/updating dependencies from requirements.txt..." -ForegroundColor Cyan
    & $pythonExe -m pip install --upgrade pip setuptools wheel
    if (Test-Path (Join-Path $ScriptDir 'requirements.txt')) {
        & $pythonExe -m pip install -r (Join-Path $ScriptDir 'requirements.txt')
    } else {
        Write-Host "Warning: requirements.txt not found, skipping dependency install." -ForegroundColor Yellow
    }
}

# 设置 Qt 插件路径，优先使用 venv 中 PyQt5 的插件目录
$qtPluginPath = Join-Path $venvPath 'Lib\site-packages\PyQt5\Qt5\plugins\platforms'
if (Test-Path $qtPluginPath) {
    Write-Host "[run_all] Setting QT_QPA_PLATFORM_PLUGIN_PATH -> $qtPluginPath" -ForegroundColor Green
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = $qtPluginPath
} else {
    Write-Host "[run_all] Warning: Qt platforms path not found: $qtPluginPath" -ForegroundColor Yellow
}

# 启动 GUI（默认）
if (-not $NoGui) {
    Write-Host "[run_all] Launching GUI (using voice_env python) ..." -ForegroundColor Cyan
    try {
        & $pythonExe (Join-Path $ScriptDir 'voice_gui.py')
        $rc = $LASTEXITCODE
        Write-Host "[run_all] GUI exited with return code: $rc"
    } catch {
        Write-Host "[run_all] Failed to launch GUI: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[run_all] -NoGui specified: skipping GUI launch." -ForegroundColor Cyan
}

# 播放测试（可选）
if ($TestPlay) {
    Write-Host "[run_all] Running playback tests: s6.wav and converted_test_16k.wav (logs saved)" -ForegroundColor Cyan
    $log1 = Join-Path $ScriptDir 'play_s6.log'
    & $pythonExe (Join-Path $ScriptDir 'playback_worker.py') 's6.wav' 0 > $log1 2>&1
    Write-Host "[run_all] s6.wav playback log: $log1 (ExitCode:$LASTEXITCODE)"

    $log2 = Join-Path $ScriptDir 'play_converted.log'
    & $pythonExe (Join-Path $ScriptDir 'playback_worker.py') 'converted_test_16k.wav' 0 > $log2 2>&1
    Write-Host "[run_all] converted_test_16k.wav playback log: $log2 (ExitCode:$LASTEXITCODE)"
}

Write-Host "[run_all] Done." -ForegroundColor Green
