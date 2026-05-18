from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np



def pca_axes_fix_dir(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA: 第一・第二主成分ベクトルを返す（右手系をなるべく保つ）
    向きは「第一主成分ベクトルの x 成分が負」になるように反転する。
    戻り値: (mean(3,), pc1(3,), pc2(3,))
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts shape must be (N,3), got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for PCA")

    mean = pts.mean(axis=0)
    X = pts - mean

    # SVDで主成分（安定で簡単）
    # X = U S Vt, 主成分軸は Vt の行（= V の列）
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0].astype(np.float32)
    pc2 = vt[1].astype(np.float32)

    # pc1 の矢印向きを「x座標負」に固定
    if pc1[0] > 0:
        pc1 = -pc1

    # pc2 も向きが不定なので、pc1×pc2 の z が正寄りになるよう調整（任意だが安定化）
    if np.cross(pc1, pc2)[2] < 0:
        pc2 = -pc2

    # 念のため正規化
    pc1 /= (np.linalg.norm(pc1) + 1e-12)
    pc2 /= (np.linalg.norm(pc2) + 1e-12)

    return mean.astype(np.float32), pc1.astype(np.float32), pc2.astype(np.float32)