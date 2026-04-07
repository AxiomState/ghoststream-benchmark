# Ghost Stream — BD-Rate Analysis

Model beats Lanczos at **every CRF point** from near-lossless (CRF 23) to heavily compressed (CRF 51).

## Results

| CRF | Quality Level | Lanczos VMAF | Ghost Stream VMAF | Delta |
|-----|--------------|-------------|-------------------|-------|
| 23 | Near-lossless | 90.37 | 90.99 | **+0.62** |
| 28 | High quality | 88.53 | 89.14 | **+0.60** |
| 35 | Standard | 84.76 | 85.35 | **+0.59** |
| 42 | Low quality | 79.61 | 80.13 | **+0.52** |
| 51 | Very low | 78.71 | 79.03 | **+0.32** |

## Analysis

- Delta is consistent across quality levels (+0.32 to +0.62)
- Largest improvement at high quality (CRF 23-28) where fine detail matters most
- Smallest improvement at very low quality (CRF 51) where AV1 destroys too much information
- **No CRF point where Lanczos beats the model**

## Methodology

- 4 Blender Open Movie clips (BBB, Elephants Dream, Sintel, Tears of Steel)
- SVT-AV1 encoder, preset 6, each CRF point
- Pipeline: 720p → Lanczos ↓2x → 360p → AV1 encode → decode → upscale → VMAF vs 720p original

*Note: CRF 51 benchmark ran on only 3 clips (Elephants Dream encoding failed at extreme compression).*
