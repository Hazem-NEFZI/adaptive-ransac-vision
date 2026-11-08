import cv2
import numpy as np
from vision_ransac.match import match_images


def test_orb_to_homography(tmp_path):
    rng = np.random.default_rng(8)
    im = np.zeros((480, 640), np.uint8)
    for _ in range(200):
        xy = tuple(map(int, rng.integers([10, 10], [630, 470])))
        cv2.circle(im, xy, int(rng.integers(3, 9)), int(rng.integers(70, 255)), -1)
    H = np.array([[1.0, 0.02, 15], [-0.01, 1.0, 12], [0, 0, 1.0]])
    warped = cv2.warpPerspective(im, H, (640, 480))
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    cv2.imwrite(str(a), im)
    cv2.imwrite(str(b), warped)
    visual, report = match_images(a, b)
    assert report["inliers"] > 30
    assert report["median_inlier_symmetric_error_px"] < 2
    assert visual.shape[1] == 1280
