import numpy as np
import pytest
from vision_ransac.demo import generate
from vision_ransac.homography import normalized_dlt, project
from vision_ransac.ransac import adaptive_ransac


def test_dlt_exact():
    H = np.array([[1.2, 0.02, 8], [-0.03, 0.97, -3], [0.0001, 0.0002, 1]])
    x = np.array([[0, 0], [2, 0], [2, 4], [0, 4], [5, 7], [8, 1]], float)
    out = normalized_dlt(x, project(H, x))
    assert np.max(np.abs(project(out, x) - project(H, x))) < 1e-6


def test_outliers_rejected():
    x, y, truth, _ = generate()
    result = adaptive_ransac(x, y, seed=4, threshold=3)
    assert np.mean(result.inliers[truth]) > 0.87
    assert np.mean(result.inliers[~truth]) < 0.15
    assert result.iterations <= 5000


def test_bad_input():
    with pytest.raises(ValueError):
        adaptive_ransac(np.zeros((3, 2)), np.zeros((3, 2)))
