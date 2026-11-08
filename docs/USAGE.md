# Usage

```bash
python -m vision_ransac.match first.jpg second.jpg --output outputs/pair
python -m vision_ransac.benchmark --seeds 5 --output outputs/benchmark
```

The transform maps source pixels to target pixels. Match outputs include the 3×3 homography, accepted count, median symmetric error and computation time. The visualization shows accepted matches. Unreadable images or insufficient descriptors produce explicit errors.

API: `adaptive_ransac(src, dst, threshold=3, confidence=.995, max_iterations=5000, seed=7)` with finite N×2 paired coordinates. The input threshold is in pixel coordinates, not meters.
