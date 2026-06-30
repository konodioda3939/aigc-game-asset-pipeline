@echo off
title ComfyUI

cd /d "d:\aigc-project\ComfyUI"

set TQDM_DISABLE=1

echo Starting ComfyUI...
echo GPU: RTX 4060 Laptop (8GB VRAM)
echo Port: 8188
echo Open http://127.0.0.1:8188 in your browser
echo.

D:\anaconda3\envs\GPUpytorch-env\python.exe main.py --fp16-unet --fp16-vae --use-pytorch-cross-attention --reserve-vram 1.0 --disable-async-offload --listen 127.0.0.1 --port 8188

pause
