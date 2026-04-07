# Ghost Stream — Temporal Consistency Analysis

Ghost Stream processes each frame independently (no temporal modeling). This analysis measures whether frame-by-frame processing introduces flickering artifacts.

## Results

| Clip | GT Frame Diff | Lanczos Frame Diff | Ghost Stream Frame Diff | GS/Lanczos Ratio | Verdict |
|------|--------------|-------------------|------------------------|-------------------|---------|
| Big Buck Bunny | 0.010404 | 0.010363 | 0.010336 | **0.997x** | GOOD |
| Elephants Dream | 0.043253 | 0.033357 | 0.033974 | **1.018x** | GOOD |
| Sintel | 0.062538 | 0.061574 | 0.061530 | **0.999x** | GOOD |
| Tears of Steel | 0.007731 | 0.007528 | 0.007485 | **0.994x** | GOOD |

## Interpretation

**Ghost Stream is as temporally stable as Lanczos.** All GS/Lanczos ratios are within 2% of 1.0.

- **Ratio < 1.0** means Ghost Stream is MORE stable than Lanczos (BBB, Sintel, ToS)
- **Ratio ~1.0** means identical stability
- **Ratio > 1.1** would indicate mild flickering
- **Ratio > 1.5** would indicate problematic flickering

The model achieves this stability because:
1. The SPAN architecture is deterministic (no dropout, no stochastic layers)
2. Similar input patches produce similar output patches
3. AV1 compression already removes high-frequency temporal noise

## Methodology

- Frame-to-frame mean absolute difference measured on Y-channel
- 150 frames per clip analyzed
- Compared: ground truth (720p original), Lanczos upscale, Ghost Stream upscale
- All from the same AV1-compressed 360p input
