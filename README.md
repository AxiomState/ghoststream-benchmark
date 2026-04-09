# Ghost Stream: Neural Video Super-Resolution in the Browser

**Patent Pending** | US Provisional Application Filed April 2026

**+3.53 VMAF over Lanczos on real camera footage. +3.81 to +4.71 on standard sequences.**
47,980 parameters (94 KB float16 weights; ~200 KB packaged as PyTorch checkpoint). Browser deployment via WebGPU.

### Standard Test Sequences (Blender Open Movies)

| Sequence | Type | Lanczos VMAF | Ghost Stream VMAF | Delta |
|----------|------|-------------|-------------------|-------|
| Big Buck Bunny | Animation | 89.23 | 90.99 | +1.76 |
| Elephants Dream* | CGI/Surreal | 72.31 | 81.33 | +9.02 |
| Sintel | CGI/Dark | 90.43 | 93.14 | +2.71 |
| Tears of Steel | Live Action + VFX | 87.05 | 92.40 | +5.35 |
| **Mean** | | **84.76** | **89.47** | **+4.71** |
| **Median** | | | | **+4.03** |

*\*Elephants Dream shows an unusually large improvement (+9.02), likely due to its surreal CGI content with large uniform regions that benefit disproportionately from learned upscaling. Excluding this outlier, the remaining 3 clips average +3.27.*

*Cross-machine note: A second benchmark run on different hardware produced +3.81 average on the same 4 clips (SVT-AV1 non-determinism across CPU architectures). Both measurements are valid. We report per-clip deltas for transparency.*

### Real Camera Footage (Pixel Phone)

| Clip | Lanczos VMAF | Ghost Stream VMAF | Delta | NEG Gap Diff |
|------|-------------|-------------------|-------|-------------|
| 20210119 | 87.38 | 91.10 | +3.73 | -0.06 |
| 20210728 | 76.67 | 80.35 | +3.68 | +0.12 |
| 20220618 | 78.63 | 82.94 | +4.31 | +0.26 |
| 20230109 | 76.26 | 78.67 | +2.41 | +0.03 |
| **Average** | **79.73** | **83.27** | **+3.53** | **+0.09** |

All metrics include [VMAF NEG](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) for metric gaming verification. PSNR also improves (+0.78 dB on camera footage), confirming genuine reconstruction rather than hallucinated detail.

Pipeline: 720p source → Lanczos ↓2x → 360p → SVT-AV1 CRF 35 → decode → model ↑2x → 720p → VMAF vs original.

## Quick Start

```bash
git clone https://github.com/AxiomState/ghoststream-benchmark
cd ghoststream-benchmark
pip install -r requirements.txt

# Download Blender Open Movie clips and run full benchmark
python benchmark/benchmark.py --all --download

# Verify results match published numbers (±0.5 VMAF tolerance)
python benchmark/benchmark.py --all --download --verify

# Test on your own video
python benchmark/benchmark.py --input your_video.mp4
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- ffmpeg with libvmaf support (in PATH)
- [SvtAv1EncApp](https://gitlab.com/AOMediaCodec/SVT-AV1) (in PATH)

## How It Works

Ghost Stream upscales AV1-compressed 360p video to 720p using a tiny neural network (47,980 parameters) that outperforms classical Lanczos interpolation on perceptual quality (VMAF).

**Architecture:** [SPAN](https://github.com/hongyuanyu/SPAN) (Swift Parameter-free Attention Network) with sigmoid attention. 24 channels, 4 residual blocks, PixelShuffle 2x upsampling. Ghost Stream processes the Y (luminance) channel only; chroma is preserved from the Lanczos upscale. This eliminates color artifacts by design.

**Training approach:**
- Adversarial baseline training: gradient updates restricted to pixels where the model underperforms Lanczos
- Knowledge distillation from a generic [SwinIR-light](https://github.com/JingyunLiang/SwinIR) teacher (910K params)
- Trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) with realistic AV1 degradation (not standard bicubic downscale)

**Key insight:** Training data diversity was the single largest improvement. Switching from 4 domain-specific clips to 800 diverse DIV2K images produced a ~4 VMAF point swing across all metrics.

## Benchmark Methodology

Every number in this repo is reproducible. The benchmark pipeline:

1. **Source:** 720p ground truth (Blender Open Movie clips — freely downloadable)
2. **Downscale:** Lanczos 720p → 360p
3. **Encode:** SVT-AV1 CRF 35, preset 6 (simulates real-world AV1 streaming)
4. **Decode:** AV1 → raw frames (this is the model's input)
5. **Upscale:** Lanczos bicubic ↑2x (baseline) OR Ghost Stream model ↑2x
6. **Measure:** VMAF, VMAF NEG, PSNR-Y, CAMBI at 720p vs original

### Cross-Machine Reproducibility

SVT-AV1 produces non-deterministic output across different CPU architectures (AVX-512 vs AVX2, thread count). Absolute VMAF scores may vary ±1.0 across machines. **Cross-machine delta variance measured at ±0.9 VMAF (same model scored +4.71 on one machine, +3.81 on another). Same-machine reproducibility across different training seeds: ±0.07. We ship pre-encoded AV1 bitstreams so third parties can bypass encoder variance entirely.**

The `--verify` flag accounts for this with ±1.0 tolerance (cross-machine delta variance measured at ±0.9).

### VMAF NEG & Anti-Gaming Analysis

We report [VMAF NEG](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) alongside standard VMAF for every result. VMAF NEG penalizes enhancement artifacts that inflate standard VMAF scores.

**Real camera footage:** NEG gap difference of **+0.09** vs Lanczos — negligible metric inflation. This is the most representative test for real-world content.

**Standard sequences (CGI):** NEG gap difference of **+0.78** vs Lanczos — mild perceptual enhancement. This is consistent with the adversarial training objective, which explicitly targets perceptual improvement on difficult regions.

**Anti-hallucination evidence:** PSNR improves by +0.78 dB on real camera footage alongside the VMAF improvement. Hallucinated detail would show VMAF↑ with PSNR↓ (better perceptual scores but worse pixel accuracy). We see both improving together, confirming genuine reconstruction.

For context, our VIF-trained model (discarded early in development) showed a NEG gap of +5.55 — that's deliberate metric gaming. Ghost Stream's +0.09 on real content is not in the same category.

## Additional Benchmarks

### BD-Rate (5 CRF Points)

Model beats Lanczos at every quality level tested:

| CRF | Quality Level | Delta VMAF | Note |
|-----|--------------|------------|------|
| 23 | Near-lossless | +4.01 | |
| 28 | High quality | +3.94 | |
| 35 | Standard | +3.81 | Same clips as standard sequence table |
| 42 | Low quality | +3.39 | |
| 51 | Very low | +2.83 | Extreme compression |

*BD-rate measured on gothic_tan_slug (RTX PRO 6000). CRF 35 delta (+3.81) differs from standard sequence table (+4.71) due to cross-machine SVT-AV1 non-determinism.*

### Temporal Consistency

Despite frame-independent processing, Ghost Stream is as temporally stable as Lanczos. Frame-to-frame variance ratios (GS/Lanczos) range from 0.999x to 1.118x — all within 12% of baseline.

See [`results/`](results/) for full data including [`leaderboard.md`](results/leaderboard.md) and [`bd_rate/`](results/bd_rate/).

### Ablation Study

| Configuration | vs Lanczos | Improvement | Note |
|--------------|------------|-------------|------|
| Charbonnier only | -2.00 | Baseline | |
| + Adversarial Lanczos loss | -0.98 | +1.02 | |
| + Knowledge distillation | -0.34 | +0.64 | Generic SwinIR teacher |
| + Diverse data (DIV2K) | +3.53* | +3.87 | 800 images vs 4 clips |

*Camera footage on gothic_tan_slug. Standard sequences show +3.81 to +4.71 depending on machine (SVT-AV1 non-determinism). Ablation rows 1-3 measured on EU-CZ-1 hardware; diverse data row on US-NC-1.*

## Model Details

| Property | Value |
|----------|-------|
| Architecture | SPANPlus (24ch, 4 blocks, sigmoid attention) |
| Parameters | 47,980 |
| Float16 weights (WebGPU) | 94 KB |
| Float32 checkpoint (PyTorch) | ~200 KB |
| Input | Y-channel, any resolution (fully convolutional) |
| Output | 2x input resolution, Y-channel |
| Scale factor | 2x |
| Training data | DIV2K (800 images) with AV1 CRF 35 degradation |
| Training steps | 600,000 |

### Cost

| Item | Cost |
|------|------|
| **Single training run** (600K steps, RTX 4070 Ti, 77 min) | **$0.50** |
| Single training run (RTX 3090, 90 min) | $0.55 |
| Diagnostic gate (20K steps) | $0.03 |
| **Total R&D** (10+ experiments, benchmarking, debugging, idle pods) | **~$200** |

The production cost to train one model is $0.50. The $200 total includes all failed experiments, infrastructure debugging, and $120 in idle pod time before auto-shutdown was implemented.

## Known Limitations

- **Small test set:** Results validated on 8 clips (4 CGI, 4 real camera). Broader validation across sports, anime, low-light, and talking-head content is planned.
- **Elephants Dream outlier:** +9.02 VMAF on this clip pulls the standard sequence mean up significantly. Median (+4.03) is more representative.
- **CGI NEG gap:** +0.78 wider than Lanczos on CGI content, indicating mild perceptual enhancement. Only +0.09 on real camera footage. PSNR improves +0.78 dB alongside VMAF, confirming genuine reconstruction — not hallucinated detail.
- **SVT-AV1 non-determinism:** Same model, same clips produce +3.81 to +4.71 delta across different machines. We ship pre-encoded reference files for reproducibility and report per-clip results.
- **Ablation across environments:** First 3 ablation rows (Charbonnier through distillation) were measured on EU-CZ-1 hardware; diverse data row on US-NC-1. Directionally accurate but not precisely comparable.
- **No comparison models:** Not benchmarked against EDSR, FSRCNN, or other SR models at similar parameter counts.
- **No subjective evaluation:** All quality claims are based on VMAF/PSNR metrics. No Mean Opinion Score (MOS) human evaluation has been conducted.
- **Frame-independent:** No temporal modeling. Temporal consistency is measured (within 12% of Lanczos) but not optimized.
- **WebGPU deployment untested on mobile:** Battery, thermal, and FPS performance on phones/tablets has not been measured.
- **Training not reproducible:** Only inference benchmark and weights are public. Training code, loss implementation, and experiment configs are proprietary.

## What's NOT in This Repo

- Training code and loss function implementations
- Adversarial Lanczos loss details
- Knowledge distillation pipeline
- Training data preprocessing scripts

The model weights are published under MIT license. The architecture (SPAN) is public.

## Browser Deployment

Ghost Stream is designed for client-side super-resolution in web browsers via WebGPU:

- **Model size:** 94 KB float16 weights (smaller than a typical favicon)
- **Runtime:** No server-side inference — runs entirely on the viewer's device
- **Compatibility:** WebGPU (Chrome 113+, Safari 18+, Edge 113+)
- **Fallback:** ONNX Runtime Web (WASM) for browsers without WebGPU

See [`demo/`](demo/) for a working browser demo.

## Citation

```bibtex
@misc{ghoststream2026,
  title={Ghost Stream: Lightweight Neural Video Super-Resolution for AV1 Content},
  author={Jordan Olivas},
  year={2026},
  howpublished={\url{https://github.com/AxiomState/ghoststream-benchmark}}
}
```

## License

MIT. See [LICENSE](LICENSE). A project by [Olivas Venture Capital LLC](https://harvv.com).
