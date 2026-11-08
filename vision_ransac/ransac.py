from dataclasses import dataclass
import math
import numpy as np
from .homography import normalized_dlt, symmetric_error


@dataclass
class RansacResult:
    H: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray
    iterations: int
    estimated_required_iterations: int


def adaptive_ransac(
    src, dst, threshold=2.5, confidence=0.995, max_iterations=5000, seed=7
):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("At least four paired Nx2 correspondences are required")
    if threshold <= 0 or not (0 < confidence < 1) or max_iterations < 1:
        raise ValueError("Invalid threshold, confidence, or maximum iterations")
    if (
        not np.isfinite(src).all()
        or not np.isfinite(dst).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Inputs and threshold must be finite")
    if not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer")
    rng = np.random.default_rng(seed)
    best_H, best_mask, best_cost = None, None, np.inf
    goal = max_iterations
    i = 0
    while i < min(goal, max_iterations):
        i += 1
        subset = rng.choice(len(src), size=4, replace=False)
        try:
            H = normalized_dlt(src[subset], dst[subset])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = symmetric_error(H, src, dst)
        mask = err <= threshold
        count = int(mask.sum())
        cost = float(err[mask].sum()) if count else np.inf
        if (
            best_mask is None
            or count > best_mask.sum()
            or (count == best_mask.sum() and cost < best_cost)
        ):
            best_H, best_mask, best_cost = H, mask, cost
            w = count / len(src)
            p_good = w**4
            if p_good >= 1 - 1e-12:
                goal = 1
            elif p_good > 0:
                new_goal = math.ceil(math.log(1 - confidence) / math.log(1 - p_good))
                goal = min(goal, max(1, new_goal))
    if best_mask is None or best_mask.sum() < 4:
        raise RuntimeError("RANSAC could not find a valid consensus")
    # Refit using the consensus set, then reevaluate and optionally refine.
    for _ in range(4):
        try:
            H = normalized_dlt(src[best_mask], dst[best_mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        errors = symmetric_error(H, src, dst)
        new_mask = errors <= threshold
        if new_mask.sum() < 4:
            break
        old_errors = symmetric_error(best_H, src, dst)
        old_quality = (
            int((old_errors <= threshold).sum()),
            -float(np.minimum(old_errors, threshold).sum()),
        )
        new_quality = (int(new_mask.sum()), -float(np.minimum(errors, threshold).sum()))
        if new_quality < old_quality:
            break
        best_H = H
        if np.array_equal(new_mask, best_mask):
            break
        best_mask = new_mask
    errors = symmetric_error(best_H, src, dst)
    final_mask = errors <= threshold
    return RansacResult(best_H, final_mask, errors, i, goal)
