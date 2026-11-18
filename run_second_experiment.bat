@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "PY=%~dp0..\xpool\.venv\Scripts\python.exe"
set "VIDS=%~dp0..\xpool\data\MSRVTT\vids"
set "PYTHONIOENCODING=utf-8"
set "TRANSFORMERS_OFFLINE=1"
if not exist "%PY%" (
  echo Python environment not found: %PY%
  exit /b 2
)
if not exist "%VIDS%" (
  echo MSR-VTT videos not found: %VIDS%
  exit /b 2
)
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss')"') do set "TS=%%i"
set "NAME=gradcache32_smoke_%TS%"
set "LOG=outputs\%NAME%_console.log"
if not exist "outputs" mkdir "outputs"
set "ENVLOG=outputs\%NAME%_environment.txt"
echo Starting GradCache-32 smoke %DATE% %TIME% > "%ENVLOG%"
git rev-parse HEAD >> "%ENVLOG%" 2>&1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "%ENVLOG%" 2>&1
echo [smoke] Effective batch 32, chunk 2, max 10 optimizer updates
echo Log: %LOG%
"%PY%" -u train.py --exp_name=%NAME% --videos_dir="%VIDS%" --dataset_name=MSRVTT --msrvtt_train_file=9k --huggingface --batch_size=32 --grad_cache_chunk_size=2 --eval_batch_size=4 --num_workers=2 --noclip_lr=3e-5 --transformer_dropout=0.3 --no_tensorboard --evals_per_epoch=0 --max_train_steps=10 --profile_training --grad_cache_check_recompute --log_step=1 > "%LOG%" 2>&1
set "CODE=%ERRORLEVEL%"
echo Exit code: %CODE%
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "%ENVLOG%" 2>&1
echo Completed %DATE% %TIME% exit %CODE% >> "%ENVLOG%"
echo Results: outputs\%NAME%\training_profile.json
echo Console log: %LOG%
echo Environment log: %ENVLOG%
exit /b %CODE%
