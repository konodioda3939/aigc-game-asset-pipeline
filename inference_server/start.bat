@echo off
title AIGC Inference Server

echo ================================================
echo   AIGC Inference Server - Starting...
echo ================================================
echo.
echo   Model loading takes 5-30 seconds.
echo   Ready when you see "Uvicorn running on http://127.0.0.1:8000"
echo.
echo   Close this window to STOP the server.
echo ================================================
echo.

cd /d "d:\aigc-project\inference_server"
D:\anaconda3\envs\GPUpytorch-env\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
