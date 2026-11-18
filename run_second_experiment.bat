@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "PY=%~dp0..\xpool\.venv\Scripts\python.exe"
set "VIDS=%~dp0..\xpool\data\MSRVTT\vids"
set "CLIP=%~dp0..\xpool\openai\clip-vit-base-patch32"
set "PYTHONUTF8=1"
set "TRANSFORMERS_OFFLINE=1"
set "CODE=2"
if not exist "%PY%" (
  echo Python environment not found: %PY%
  goto done
)
if not exist "%VIDS%" (
  echo MSR-VTT videos not found: %VIDS%
  goto done
)
for %%F in (config.json pytorch_model.bin vocab.json merges.txt) do (
  if not exist "%CLIP%\%%F" (
    echo CLIP asset not found: %CLIP%\%%F
    goto done
  )
)
set "TS="
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_fff')"') do set "TS=%%i"
if not defined TS goto done
if not exist "outputs" mkdir "outputs"
set "ENVLOG=outputs\gradcache32_smoke_%TS%_environment.txt"
echo Starting GradCache-32 smoke %DATE% %TIME% > "%ENVLOG%"
git rev-parse HEAD >> "%ENVLOG%" 2>&1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "%ENVLOG%" 2>&1
call :smoke 2
set "CODE2=%ERRORLEVEL%"
call :smoke 4
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" set "CODE=%CODE2%"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "%ENVLOG%" 2>&1
echo Completed %DATE% %TIME% exit %CODE% >> "%ENVLOG%"
echo Environment log: %ENVLOG%
goto done

:smoke
set "NAME=gradcache32_smoke_c%1_%TS%"
set "LOG=outputs\%NAME%_console.log"
echo [smoke] Effective batch 32, chunk %1, max 10 optimizer updates
"%PY%" -u run_training_session.py --hours 0.17 --max-steps 10 --log-step 1 --exp-name %NAME% --videos-dir "%VIDS%" --clip-model-path "%CLIP%" --batch-size 32 --grad-cache-chunk-size %1 --check-recompute > "%LOG%" 2>&1
set "SMOKE_CODE=%ERRORLEVEL%"
echo Chunk %1 exit code: %SMOKE_CODE%
echo Results: outputs\%NAME%\training_profile.json
echo Console log: %LOG%
exit /b %SMOKE_CODE%

:done
if /I not "%~1"=="--no-pause" pause
exit /b %CODE%
