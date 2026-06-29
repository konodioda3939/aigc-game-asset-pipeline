@echo off
chcp 65001 >nul
title AIGC Game Art Workflows — AI 游戏美术工作流

echo ================================================
echo   🎮 AIGC Game Art Workflows — 4 条标准化管线
echo ================================================
echo.
echo   模型加载需要 5-30 秒，请留意下方进度提示。
echo   看到 "Uvicorn running on http://127.0.0.1:8000" 即就绪。
echo.
echo   启动后可以访问：
echo     📖 API 文档:    http://127.0.0.1:8000/docs
echo     🎨 工作流 Web UI: http://127.0.0.1:8000/workflow-ui/
echo     🔌 Unity 插件:   Tools ^> AI Asset Generator ^> 切换到「工作流」
echo.
echo   4 条工作流：
echo     🎭 角色概念图 — 文字 → 4 角度角色设计
echo     ⚔️ 道具图标   — 文字/草图 → 精修游戏图标
echo     🏞️ 场景氛围图 — 文字 + 氛围 → 概念图
echo     🎨 UI 元素    — 文字/参考图 → 成套 UI 素材
echo.
echo   关闭此窗口即可停止服务
echo ================================================
echo.

cd /d "d:\aigc-project\inference_server"
D:\anaconda3\envs\GPUpytorch-env\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
