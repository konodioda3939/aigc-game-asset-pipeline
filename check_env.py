import sys
import subprocess

print("Python version:", sys.version)
print("Python path:", sys.executable)

# Check CUDA via nvidia-smi
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10)
    print("GPU:", result.stdout.strip())
except:
    print("GPU: nvidia-smi not found")

# Check torch
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
except ImportError:
    print("PyTorch: NOT INSTALLED")

# Check diffusers
try:
    import diffusers
    print("diffusers:", diffusers.__version__)
except ImportError:
    print("diffusers: NOT INSTALLED")

# Check xformers
try:
    import xformers
    print("xformers:", xformers.__version__)
except ImportError:
    print("xformers: NOT INSTALLED")

# Check cudatoolkit
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
    print("nvcc:", result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "not found")
except:
    print("nvcc: not found (this is fine, conda handles CUDA differently)")
