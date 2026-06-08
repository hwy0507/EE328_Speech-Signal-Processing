@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM run_all.bat - one-click launcher (batch)
REM Usage: run_all.bat [install] [nogui] [test]
REM   install - install dependencies from requirements.txt
REM   nogui   - do not launch GUI
REM   test    - run playback tests after GUI (or without GUI if nogui)

cd /d "%~dp0"
set PYTHON=%CD%\voice_env\Scripts\python.exe
if not exist "%PYTHON%" (
    echo ERROR: voice_env Python not found: %PYTHON%
    pause
    exit /b 1
)

echo Using Python: %PYTHON%

set INSTALL_DEPS=0
set NO_GUI=0
set DO_TEST=0

for %%A in (%*) do (
    if /I "%%~A"=="install" set INSTALL_DEPS=1
    if /I "%%~A"=="nogui" set NO_GUI=1
    if /I "%%~A"=="test" set DO_TEST=1
)

if "%INSTALL_DEPS%"=="1" (
    echo Installing/updating dependencies...
    "%PYTHON%" -m pip install --upgrade pip setuptools wheel
    if exist requirements.txt (
        "%PYTHON%" -m pip install -r requirements.txt
    ) else (
        echo Warning: requirements.txt not found, skipping.
    )
)

REM Set QT plugin path for PyQt5 if present
set QTPLUG=%CD%\voice_env\Lib\site-packages\PyQt5\Qt5\plugins\platforms
if exist "%QTPLUG%" (
    set "QT_QPA_PLATFORM_PLUGIN_PATH=%QTPLUG%"
    echo Set QT_QPA_PLATFORM_PLUGIN_PATH=%QT_QPA_PLATFORM_PLUGIN_PATH%
) else (
    echo Warning: Qt platforms directory not found: %QTPLUG%
)

if "%NO_GUI%"=="0" (
    echo Launching GUI...
    "%PYTHON%" voice_gui.py
    if errorlevel 1 (
        echo GUI exited with error code %ERRORLEVEL%
    ) else (
        echo GUI exited normally.
    )
) else (
    echo Skipping GUI (nogui specified).
)

if "%DO_TEST%"=="1" (
    echo Running playback tests...
    "%PYTHON%" playback_worker.py s6.wav 0 > play_s6.log 2>&1
    echo s6.wav log saved to play_s6.log (ExitCode=%ERRORLEVEL%)
    "%PYTHON%" playback_worker.py converted_test_16k.wav 0 > play_converted.log 2>&1
    echo converted_test_16k.wav log saved to play_converted.log (ExitCode=%ERRORLEVEL%)
)

echo Done.
pause
