@echo off
title AIGC Game Art Workflows

echo ================================================
echo   AIGC Game Art Workflows - 4 pipelines
echo ================================================
echo.
echo   Model loading takes 5-30 seconds.
echo   Ready when you see "Uvicorn running on http://127.0.0.1:8000"
echo.
echo   After startup:
echo     API docs  : http://127.0.0.1:8000/docs
echo     Web UI    : http://127.0.0.1:8000/workflow-ui/
echo     Unity     : Tools > AI Asset Generator > Workflow tab
echo.
echo   Workflows:
echo     1. Character concept  (text -> 4-angle turnaround)
echo     2. Asset icon         (text/sketch -> refined icon)
echo     3. Scene concept      (text + mood -> concept art)
echo     4. UI elements        (text/reference -> UI set)
echo.
echo   Close this window to STOP the server.
echo ================================================
echo.

cd /d "d:\aigc-project\inference_server"
D:\anaconda3\envs\GPUpytorch-env\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
