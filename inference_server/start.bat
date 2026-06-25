@echo off
chcp 65001 >nul
title AIGC LoRA 推理服务

echo ================================================
echo   AIGC LoRA 推理服务 — 启动中...
echo ================================================
echo.
echo   模型加载需要 5-30 秒，请留意下方进度提示
echo   看到 "Uvicorn running on http://127.0.0.1:8000" 即启动成功
echo.
echo   关闭此窗口即可停止服务
echo ================================================
echo.

cd /d "d:\aigc-project\inference_server"
D:\anaconda3\envs\GPUpytorch-env\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
