# Design: robust geometric estimation

## Numerical method

Coordinates are centered and scaled before DLT. The minimal 8×9 design matrix needs its full nullspace; overdetermined fits use an economy SVD to avoid allocating a large square left-singular-vector matrix. Nonfinite inputs and rank-deficient samples fail explicitly.

Each hypothesis uses four correspondences. The confidence-driven budget uses the estimated inlier ratio raised to the fourth power. This is a probabilistic stopping estimate, not a guarantee, especially for correlated or degenerate matches. Consensus uses the mean of forward and backward Euclidean reprojection errors. Refitting is accepted only when consensus quality does not degrade.

## Real image matching

ORB descriptors → ratio-filtered Hamming matches → unique target assignments → custom RANSAC → visualized inliers. This assumes a planar scene or a camera rotation compatible with a homography. Parallax, repetitive textures, and too few distinct features can invalidate the model. A good fit is not proof of physical scene correctness.

## Benchmark fairness

The comparative runner includes OpenCV RANSAC, records each seed, runtime and clean-correspondence error, and retains failures. The algorithms use different error definitions, so identical numeric thresholds do not imply identical acceptance regions. Five seeds across four outlier levels are a small stress test, not an exhaustive leaderboard.
