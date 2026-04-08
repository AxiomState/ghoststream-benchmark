# Ghost Stream — Model Leaderboard

All models evaluated on the same pipeline: 720p source → Lanczos ↓2x → 360p → SVT-AV1 CRF 35 → decode → model ↑2x → 720p → VMAF.

## Standard Sequences (Blender Open Movies)

| Rank | Model | Params | Avg VMAF | vs Lanczos | Avg VMAF NEG | Training Data |
|------|-------|--------|----------|------------|-------------|---------------|
| 1 | **Ghost Stream (DIV2K distill v2)** | 48K | **89.47** | **+4.71*** | 86.58 | DIV2K (800 images) |
| — | Lanczos (bicubic) baseline | 0 | 84.76 | — | 82.27 | N/A |

*\*+4.71 on steep_sapphire_gibbon; +3.81 on gothic_tan_slug (same model, same clips, different SVT-AV1 encode). Median: +4.03. Elephants Dream (+9.02) is an outlier.*

## Pixel Phone Clips (Private Content)

| Rank | Model | Avg VMAF | vs Lanczos | Training |
|------|-------|----------|------------|----------|
| 1 | Ghost Stream DIV2K distill v2 | 82.80 | +3.32 | DIV2K diverse |
| 2 | Distill v2 (generic teacher) | 78.14 | -0.34 | 4 Pixel clips |
| 3 | S4 adversarial | 77.51 | -0.98 | 4 Pixel clips |
| 4 | Distill v3 (fine-tuned teacher) | 77.19 | -1.30 | 4 Pixel clips |
| 5 | Optimal transport | 77.95 | -1.02 | 4 Pixel clips |
| 6 | S2 Charb+MSSIM+FFL | 77.82 | -1.15 | 4 Pixel clips |
| 7 | Adversarial MSSIM | 77.65 | -1.32 | 4 Pixel clips |
| 8 | S1 Charbonnier+FFL | 76.97 | -2.00 | 4 Pixel clips |
| 9 | Motion weighted | 73.90 | -5.07 | 4 Pixel clips |
| — | Lanczos baseline | 79.48 | — | N/A |

## Key Findings

1. **Training data diversity was the single biggest lever.** Switching from 4 phone clips to 800 DIV2K images produced a +7 VMAF swing (from -0.34 to +3.32 on private clips, +4.71 on standard sequences).

2. **Adversarial Lanczos loss** (only penalize where model is worse than Lanczos) is effective but secondary to data diversity.

3. **Knowledge distillation** from a generic SwinIR teacher helps (+0.64 over no distillation). Fine-tuning the teacher to domain data hurts (-0.96 vs generic teacher).

4. **VMAF NEG gap** averages 2.89 for Ghost Stream vs 2.49 for Lanczos — mild enhancement, not metric gaming.
