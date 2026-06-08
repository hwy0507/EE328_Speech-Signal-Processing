# GUI Startup Script for Voice Changer
# Windows PowerShell / PowerShell Core

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Voice Gender Changer - GUI Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check virtual environment
$venvPath = Join-Path $ScriptDir "voice_env_clean"
if (-not (Test-Path $venvPath)) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv voice_env_clean" -ForegroundColor Yellow
    Write-Host "Then run: .\voice_env_clean\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK: Virtual environment detected" -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\voice_env_clean\Scripts\Activate.ps1"

# Set Qt plugin path
$QtPluginPath = Join-Path $venvPath "Lib\site-packages\PyQt5\Qt5\plugins\platforms"
if (-not (Test-Path $QtPluginPath)) {
    Write-Host "ERROR: Qt plugin path not found!" -ForegroundColor Red
    Write-Host "Try: pip install --upgrade PyQt5" -ForegroundColor Yellow
    exit 1
}

$env:QT_QPA_PLATFORM_PLUGIN_PATH = $QtPluginPath
Write-Host "OK: Qt plugin path configured" -ForegroundColor Green

# Check GUI script
if (-not (Test-Path "voice_gui.py")) {
    Write-Host "ERROR: voice_gui.py not found!" -ForegroundColor Red
    exit 1
}

Write-Host "OK: GUI script detected" -ForegroundColor Green
Write-Host ""
Write-Host "Launching GUI window..." -ForegroundColor Cyan
Write-Host "- Drag WAV file to the center area" -ForegroundColor Gray
Write-Host "- Click 'Convert' button to start" -ForegroundColor Gray
Write-Host "- Use Play Original / Play Converted for A/B listening" -ForegroundColor Gray
Write-Host "- Save F0 and Mel comparison charts from GUI" -ForegroundColor Gray
Write-Host ""

# Launch GUI
& (Join-Path $venvPath "Scripts\\python.exe") voice_gui.py
$guiExit = $LASTEXITCODE

Write-Host ""
if ($guiExit -eq 0) {
    Write-Host "GUI closed normally (exit code 0)." -ForegroundColor Green
} else {
    Write-Host "GUI exited abnormally (exit code $guiExit)." -ForegroundColor Red
}
