@echo off
REM Training Job Submission Script
REM Created: 2025-12-10

REM MSVC environment variables for Triton / torch.compile
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;%PATH%"
set "INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
set "LIB=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe"

echo ========================================
echo Starting Multi-Config Training Pipeline
echo ========================================
echo.

REM Set the base command
set PYTHON_CMD=python main.py --config

REM ====================================
REM Training Jobs
REM ====================================

echo [Job 1] Training with tcn-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/tcn-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 1 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 1 completed.
echo.

echo [Job 2] Training with scnn-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/scnn-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 2 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 2 completed.
echo.

REM ====================================
REM End of training jobs
REM ====================================

echo ========================================
echo All training jobs completed!
echo ========================================
pause
