@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "PY=%~dp0..\xpool\.venv\Scripts\python.exe"
set "SRC=%~dp0..\xpool"
set "PYTHONIOENCODING=utf-8"
set "TRANSFORMERS_OFFLINE=1"
set "LOG=%~dp0outputs\first_experiment"
if not exist "%LOG%" mkdir "%LOG%"
echo started %DATE% %TIME% > "%LOG%\run_status.txt"
nvidia-smi > "%LOG%\nvidia_smi_before.txt" 2>&1

echo [1/3] steptime ... (about 2 min)
"%PY%" -u first_experiment.py steptime --source-root "%SRC%" > "%LOG%\run_steptime.log" 2>&1
echo steptime exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo [2/3] gradcheck ... (about 2 min)
"%PY%" -u first_experiment.py gradcheck --source-root "%SRC%" > "%LOG%\run_gradcheck.log" 2>&1
echo gradcheck exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo [3/3] drift ... (about 5 min)
"%PY%" -u first_experiment.py drift --source-root "%SRC%" > "%LOG%\run_drift.log" 2>&1
echo drift exit %ERRORLEVEL% %TIME% >> "%LOG%\run_status.txt"

echo all done %DATE% %TIME% >> "%LOG%\run_status.txt"
echo.
echo Done. You can close this window.
pause
