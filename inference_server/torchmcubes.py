"""
桥接文件：当 TripoSR 执行 "from torchmcubes import marching_cubes" 时，
自动路由到 CPU 版实现（skimage），无需编译 CUDA 扩展。
"""
from torchmcubes_compat import marching_cubes

__all__ = ["marching_cubes"]
