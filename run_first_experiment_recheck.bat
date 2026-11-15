@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "PY=%~dp0..\xpool\.venv\Scripts\python.exe"
set "SRC=%~dp0..\xpool"
set "CK1=%SRC%\outputs\msrvtt9k_train_vram_20261107_054714\checkpoint-epoch1.pth"
set "CK3=%SRC%\outputs\msrvtt9k_split_training\checkpoint-epoch3.pth"
set "PYTHONIOENCODING=utf-8"
set "TRANSFORMERS_OFFLINE=1"
set "LOG=%~dp0outputs\first_experiment\recheck"
if not exist "%LOG%" mkdir "%LOG%"
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss')"') do set "TS=%%i"
echo started %DATE% %TIME% utc_tag %TS% > "%LOG%\run_status.txt"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv > "%LOG%\gpu_before.csv" 2>&1

echo [1/4] gradcheck with TF32 off ... (about 1 min)
"%PY%" -u run_with_tf32_off.py gradcheck --source-root "%SRC%" --name gradcheck_tf32off_%TS% > "%LOG%\run_gradcheck_tf32off.log" 2>&1
echo gradcheck_tf32off exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo [2/4] drift every 10 steps ... (about 2 min)
"%PY%" -u first_experiment.py drift --source-root "%SRC%" --every 10 --steps 100 --name drift_every10_%TS% > "%LOG%\run_drift_every10.log" 2>&1
echo drift_every10 exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo [3/4] drift from epoch-1 checkpoint ... (about 5 min)
"%PY%" -u first_experiment.py drift --source-root "%SRC%" --checkpoint "%CK1%" --name drift_epoch1_%TS% > "%LOG%\run_drift_epoch1.log" 2>&1
echo drift_epoch1 exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo [4/4] drift from epoch-3 checkpoint ... (about 5 min)
"%PY%" -u first_experiment.py drift --source-root "%SRC%" --checkpoint "%CK3%" --name drift_epoch3_%TS% > "%LOG%\run_drift_epoch3.log" 2>&1
echo drift_epoch3 exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo all done %DATE% %TIME% >> "%LOG%\run_status.txt"
echo.
echo Done. You can close this window.
pause
