@echo off
chcp 65001 >nul
cd /d "d:\aigc-project\inference_server"

echo ==========================================
echo   Installing StableMaterials PBR dependencies...
echo ==========================================
echo.

REM Activate conda env
set PATH=D:\anaconda3\envs\GPUpytorch-env;D:\anaconda3\envs\GPUpytorch-env\Scripts;D:\anaconda3\envs\GPUpytorch-env\Library\bin;%PATH%

echo [1/1] Updating diffusers (for LCMScheduler support)...
D:\anaconda3\envs\GPUpytorch-env\python.exe -m pip install "diffusers>=0.30.0" --quiet
echo Done.

echo.
echo NOTE: StableMaterials uses trust_remote_code=True, so the model
echo code is downloaded from HuggingFace automatically on first use.
echo No additional pip packages required beyond diffusers/transformers.
echo.

echo Verifying imports...
D:\anaconda3\envs\GPUpytorch-env\python.exe -c "from diffusers import DiffusionPipeline, LCMScheduler; print('OK: diffusers with LCMScheduler')"

echo.
echo ==========================================
echo   PBR dependencies installed!
echo ==========================================
echo.
echo   Next step: double-click start.bat to launch the server.
echo   First PBR generation will auto-download StableMaterials model (~2-3GB).
echo ==========================================
pause
