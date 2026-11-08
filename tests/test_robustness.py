import numpy as np
import pytest
from vision_ransac.homography import normalized_dlt, project
from vision_ransac.ransac import adaptive_ransac
from vision_ransac.demo import generate


@pytest.mark.parametrize("seed", range(5))
def test_noisy_outlier_recovery(seed):
    x, y, truth, H = generate(seed=seed, n_inliers=90, n_outliers=60)
    fit = adaptive_ransac(x, y, threshold=3, seed=seed)
    assert fit.inliers[truth].mean() > 0.90
    assert fit.inliers[~truth].mean() < 0.10
    assert (
        np.median(
            np.linalg.norm(project(fit.H, x[truth]) - project(H, x[truth]), axis=1)
        )
        < 1
    )


def test_collinear_rejected():
    x = np.column_stack([np.arange(6), np.arange(6)])
    with pytest.raises(ValueError):
        normalized_dlt(x, x)
    with pytest.raises(RuntimeError):
        adaptive_ransac(x, x, max_iterations=20)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(bad):
    x, y, _, _ = generate()
    x[0, 0] = bad
    with pytest.raises(ValueError):
        adaptive_ransac(x, y)


def test_deterministic_and_budget():
    x, y, _, _ = generate()
    a = adaptive_ransac(x, y, max_iterations=300, seed=7)
    b = adaptive_ransac(x, y, max_iterations=300, seed=7)
    assert a.iterations <= 300
    assert np.array_equal(a.inliers, b.inliers)
    assert np.allclose(a.H, b.H)
