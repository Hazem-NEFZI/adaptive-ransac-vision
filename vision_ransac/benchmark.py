"""Compare custom adaptive RANSAC with OpenCV RANSAC over multiple seeds."""

import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from .demo import generate
from .homography import project
from .ransac import adaptive_ransac


def run(output="outputs/benchmark", seeds=5):
    rows = []
    for fraction in (0.1, 0.3, 0.5, 0.65):
        for seed in range(seeds):
            nout = int(150 * fraction)
            x, y, truth, H = generate(seed=seed, n_inliers=150 - nout, n_outliers=nout)
            for method in ("adaptive", "opencv"):
                start = time.perf_counter()
                try:
                    if method == "adaptive":
                        fit = adaptive_ransac(x, y, threshold=3, seed=seed)
                        estimate = fit.H
                        iterations = fit.iterations
                    else:
                        cv2.setRNGSeed(seed)
                        estimate, _ = cv2.findHomography(
                            x, y, cv2.RANSAC, 3, maxIters=5000, confidence=0.995
                        )
                        iterations = None
                    if estimate is None:
                        raise ValueError("No model")
                    error = float(
                        np.median(
                            np.linalg.norm(
                                project(estimate, x[truth]) - project(H, x[truth]),
                                axis=1,
                            )
                        )
                    )
                    if not np.isfinite(error):
                        raise ValueError("Nonfinite projection")
                    row = {
                        "success_under_2px": error < 2,
                        "median_clean_error_px": error,
                        "iterations": iterations,
                    }
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    row = {"success_under_2px": False, "error": str(e)}
                rows.append(
                    {
                        "method": method,
                        "outlier_fraction": fraction,
                        "seed": seed,
                        "seconds": time.perf_counter() - start,
                        **row,
                    }
                )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "data": "synthetic homographies",
                "note": "OpenCV uses forward reprojection; custom method uses symmetric distance. Thresholds are not mathematically identical.",
                "runs": rows,
            },
            indent=2,
        )
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/benchmark")
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    run(a.output, a.seeds)


if __name__ == "__main__":
    main()
