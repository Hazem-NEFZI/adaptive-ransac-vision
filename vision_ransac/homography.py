"""Numerically stable normalized DLT for 2D homographies."""

import numpy as np


def _normalize(points):
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1).mean()
    if dist < 1e-10:
        raise ValueError("Degenerate point sample")
    scale = np.sqrt(2) / dist
    T = np.array(
        [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]],
        dtype=float,
    )
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    return (T @ homogeneous.T).T[:, :2], T


def project(H, points):
    points = np.asarray(points, dtype=float)
    q = (np.asarray(H) @ np.column_stack([points, np.ones(len(points))]).T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def normalized_dlt(src, dst):
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("src/dst must be matching Nx2 arrays, N>=4")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Correspondences must be finite")
    x, Ts = _normalize(src)
    y, Td = _normalize(dst)
    A = []
    for (sx, sy), (dx, dy) in zip(x, y):
        A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
    A = np.asarray(A)
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Degenerate correspondence sample")
    # full_matrices=True gives the nullspace even for the minimal 8x9 case.
    _, _, Vt = np.linalg.svd(A, full_matrices=(A.shape[0] < A.shape[1]))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    norm = np.linalg.norm(H)
    if norm < 1e-12 or not np.all(np.isfinite(H)):
        raise ValueError("Invalid homography")
    H = H / norm
    return H / H[2, 2] if abs(H[2, 2]) > 1e-10 else H


def symmetric_error(H, src, dst):
    """Mean forward and backward Euclidean reprojection error, in pixels."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    a = np.linalg.norm(project(H, src) - dst, axis=1)
    b = np.linalg.norm(project(H_inv, dst) - src, axis=1)
    result = (a + b) / 2
    return np.where(np.isfinite(result), result, np.inf)
