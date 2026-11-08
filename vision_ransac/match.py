"""ORB matching plus custom RANSAC; works on user-provided image pairs."""

import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from .ransac import adaptive_ransac


def match_images(first, second, ratio=0.75):
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be in (0,1)")
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (first, second)]
    if any(x is None for x in images):
        raise ValueError("Cannot decode one or both images")
    if any(x.size > 24_000_000 for x in images):
        raise ValueError("Resize images below 24 megapixels")
    orb = cv2.ORB_create(nfeatures=2500)
    k1, d1 = orb.detectAndCompute(images[0], None)
    k2, d2 = orb.detectAndCompute(images[1], None)
    if d1 is None or d2 is None:
        raise ValueError("No descriptors; use textured images")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [
        p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance
    ]
    # Keep one source correspondence per target descriptor.
    unique = {}
    for m in sorted(good, key=lambda m: m.distance):
        unique.setdefault(m.trainIdx, m)
    good = list(unique.values())
    if len(good) < 8:
        raise ValueError("Fewer than eight reliable matches")
    x = np.array([k1[m.queryIdx].pt for m in good])
    y = np.array([k2[m.trainIdx].pt for m in good])
    start = time.perf_counter()
    fit = adaptive_ransac(x, y, threshold=3)
    elapsed = time.perf_counter() - start
    vis = cv2.drawMatches(
        images[0],
        k1,
        images[1],
        k2,
        good,
        None,
        matchesMask=fit.inliers.astype(int).tolist(),
        flags=2,
    )
    report = {
        "matches": len(good),
        "inliers": int(fit.inliers.sum()),
        "iterations": fit.iterations,
        "seconds": elapsed,
        "H_source_to_target": fit.H.tolist(),
        "median_inlier_symmetric_error_px": float(np.median(fit.errors[fit.inliers])),
    }
    return vis, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--output", default="outputs/matching")
    a = p.parse_args()
    vis, report = match_images(a.source, a.target)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "matches.png"), vis)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
