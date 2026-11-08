import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .homography import project, normalized_dlt
from .ransac import adaptive_ransac


def generate(seed=7, n_inliers=85, n_outliers=65):
    rng = np.random.default_rng(seed)
    H = np.array([[1.07, 0.05, 26], [-0.035, 0.96, 18], [0.00018, -0.00024, 1]])
    src = rng.uniform(0, 400, (n_inliers + n_outliers, 2))
    dst = project(H, src)
    dst[:n_inliers] += rng.normal(0, 0.65, (n_inliers, 2))
    dst[n_inliers:] = rng.uniform(0, 430, (n_outliers, 2))
    perm = rng.permutation(len(src))
    return src[perm], dst[perm], (perm < n_inliers), H


def main(output="outputs"):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    src, dst, truth, Htrue = generate()
    fit = adaptive_ransac(src, dst, threshold=3, seed=4)
    try:
        naive = normalized_dlt(src, dst)
        naive_e = np.linalg.norm(project(naive, src[truth]) - dst[truth], axis=1).mean()
    except Exception:
        naive_e = float("nan")
    inlier_err = np.linalg.norm(project(fit.H, src[truth]) - dst[truth], axis=1)
    stats = {
        "data": "synthetic",
        "seed": 7,
        "estimated_inliers": int(fit.inliers.sum()),
        "ground_truth_inliers": int(truth.sum()),
        "iterations": fit.iterations,
        "inlier_precision": float(
            np.sum(fit.inliers & truth) / max(fit.inliers.sum(), 1)
        ),
        "inlier_recall": float(np.sum(fit.inliers & truth) / truth.sum()),
        "ground_truth_mean_forward_error_px": float(inlier_err.mean()),
        "naive_DLT_mean_forward_error_px": float(naive_e),
    }
    (path / "metrics.json").write_text(json.dumps(stats, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax[0].scatter(
        src[:, 0],
        src[:, 1],
        c=np.where(fit.inliers, "#12acb3", "#eb6278"),
        s=18,
        alpha=0.8,
    )
    ax[0].set_title("Adaptive RANSAC: accepted vs rejected")
    ax[0].set_xlabel("Source x (px)")
    ax[0].set_ylabel("Source y (px)")
    ax[1].scatter(dst[:, 0], dst[:, 1], c="#b0b7c5", s=14, label="Observed", alpha=0.45)
    prediction = project(fit.H, src[truth])
    ax[1].scatter(
        prediction[:, 0],
        prediction[:, 1],
        c="#0eafb8",
        s=11,
        label="Estimated true matches",
    )
    ax[1].set_title("Target points and estimated projection")
    ax[1].set_xlabel("Target x (px)")
    ax[1].set_ylabel("Target y (px)")
    ax[1].legend(fontsize=8)
    fig.suptitle("ROBUST PROJECTIVE GEOMETRY | SYNTHETIC BENCHMARK", weight="bold")
    fig.savefig(path / "demo.png", dpi=165, facecolor="white")
    plt.close(fig)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    main(args.output)
