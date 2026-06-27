"""
torchmcubes 兼容模块

用 skimage.measure.marching_cubes（CPU）替代 torchmcubes（需 CUDA 编译）。
接口与 torchmcubes.marching_cubes 保持一致，TripoSR 可直接使用。

    用法：将此文件所在目录加入 sys.path，import torchmcubes 即可。
"""

import numpy as np
import torch
from skimage.measure import marching_cubes as _skimage_mc


def marching_cubes(volume: torch.Tensor, threshold: float = 0.0):
    """
    与 torchmcubes.marching_cubes 相同的接口。

    参数：
        volume: (D, H, W) 形状的密度场（torch.Tensor 或 numpy array）
        threshold: 等值面阈值
    返回：
        (vertices, faces): vertices 是 (N, 3) 的坐标，faces 是 (M, 3) 的面索引
    """
    # 转为 numpy（CPU）
    if isinstance(volume, torch.Tensor):
        volume_np = volume.detach().cpu().numpy()
    else:
        volume_np = volume

    # skimage marching_cubes
    verts, faces, _, _ = _skimage_mc(volume_np.astype(np.float32), level=threshold)

    # 转回 torch tensor
    verts = torch.from_numpy(verts.copy()).float()
    faces = torch.from_numpy(faces.copy().astype(np.int64)).long()

    return verts, faces


# 确保此模块可以被 "from torchmcubes import marching_cubes" 导入
__all__ = ["marching_cubes"]
