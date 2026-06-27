@echo off
chcp 65001 >nul
cd /d "d:\aigc-project\inference_server"

echo ==========================================
echo   Installing TripoSR dependencies...
echo ==========================================
echo.

REM Activate conda env
set PATH=D:\anaconda3\envs\GPUpytorch-env;D:\anaconda3\envs\GPUpytorch-env\Scripts;D:\anaconda3\envs\GPUpytorch-env\Library\bin;%PATH%

echo [1/3] Installing trimesh...
D:\anaconda3\envs\GPUpytorch-env\python.exe -m pip install trimesh>=4.0.5 --quiet
echo Done.

echo [2/3] Installing rembg (background removal)...
D:\anaconda3\envs\GPUpytorch-env\python.exe -m pip install rembg>=2.0.0 --quiet
echo Done.

echo [3/3] Installing omegaconf + einops + xatlas...
D:\anaconda3\envs\GPUpytorch-env\python.exe -m pip install omegaconf>=2.3.0 einops>=0.7.0 xatlas>=0.0.9 --quiet
echo Done.

echo.
echo NOTE: torchmcubes (CUDA marching cubes) is NOT installed.
echo Instead, we use a CPU-based skimage alternative.
echo This avoids needing Visual Studio C++ compiler.
echo.

echo Verifying install...
D:\anaconda3\envs\GPUpytorch-env\python.exe -c "import trimesh; import rembg; import omegaconf; import einops; import xatlas; print('Base packages: OK')"

REM Verify our torchmcubes compatibility module
D:\anaconda3\envs\GPUpytorch-env\python.exe -c "import sys; sys.path.insert(0, '.'); from torchmcubes import marching_cubes; print('torchmcubes compat: OK')"

echo.
echo ==========================================
echo   TripoSR dependencies installed!
echo ==========================================
echo.
echo   Next step: double-click start.bat to launch the server.
echo   First 3D generation will auto-download TripoSR model (~1.68GB).
echo ==========================================
pause
