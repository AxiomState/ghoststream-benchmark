# Ghost Stream — Standard Sequence Benchmark

**Date:** 2026-04-06
**Pipeline:** 720p → Lanczos ↓2x → 360p → SVT-AV1 CRF 35 → decode → model ↑2x → 720p → VMAF
**Encoder:** SVT-AV1 b486d83 (release), preset 6
**Clips:** Blender Open Movie sequences (freely redistributable)

## Results

| Sequence | Type | Lanczos VMAF | Ghost Stream VMAF | Delta | VMAF NEG Gap |
|----------|------|-------------|-------------------|-------|-------------|
| Big Buck Bunny | Animation | 89.23 | 90.99 | **+1.76** | 2.80 vs 2.33 |
| Elephants Dream | CGI/Surreal | 72.31 | 81.33 | **+9.02** | 3.26 vs 2.48 |
| Sintel | CGI/Dark | 90.43 | 93.14 | **+2.71** | 3.01 vs 2.65 |
| Tears of Steel | Live Action + VFX | 87.05 | 92.40 | **+5.35** | 2.48 vs 2.48 |
| **Average** | | **84.76** | **89.47** | **+4.71** | **2.89 vs 2.49** |

## VMAF NEG Honesty Check

VMAF NEG penalizes enhancement artifacts that inflate standard VMAF. Ghost Stream's average NEG gap (2.89) is 0.40 wider than Lanczos (2.49), indicating mild perceptual enhancement beyond pure reconstruction. This is disclosed for transparency.

## Reproducibility

```bash
git clone https://github.com/AxiomState/ghoststream-benchmark
cd ghoststream-benchmark
pip install -r requirements.txt
python benchmark/benchmark.py --all --download --verify
```

Cross-machine variance of ±0.5 VMAF is expected due to SVT-AV1 non-determinism across CPU architectures.
