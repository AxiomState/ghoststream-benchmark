# Ghost Stream: Neural Video Super-Resolution in the Browser

**+4.71 VMAF over Lanczos** on standard test sequences.
48K parameters. 200KB model. Designed for browser deployment via WebGPU.

| Sequence | Type | Lanczos VMAF | Ghost Stream VMAF | Delta |
|----------|------|-------------|-------------------|-------|
| Big Buck Bunny | Animation | 89.23 | 90.99 | +1.76 |
| Elephants Dream | CGI/Surreal | 72.31 | 81.33 | +9.02 |
| Sintel | CGI/Dark | 90.43 | 93.14 | +2.71 |
| Tears of Steel | Live Action + VFX | 87.05 | 92.40 | +5.35 |
| **Average** | | **84.76** | **89.47** | **+4.71** |

All metrics include [VMAF NEG](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) to verify no metric gaming.
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

Ghost Stream upscales AV1-compressed 360p video to 720p using a tiny neural network (48K parameters) that outperforms classical Lanczos interpolation on perceptual quality (VMAF).

**Architecture:** [SPAN](https://github.com/hongyuanyu/SPAN) (Swift Parameter-free Attention Network) with sigmoid attention. 24 channels, 4 residual blocks, PixelShuffle 2x upsampling. Y-channel only — operates on luma, chroma handled by standard resize.

**Training approach:**
- Adversarial baseline training: gradient updates restricted to pixels where the model underperforms Lanczos
- Knowledge distillation from a generic [SwinIR-light](https://github.com/JingyunLiang/SwinIR) teacher (910K params)
- Trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) with realistic AV1 degradation (not standard bicubic downscale)

**Key insight:** Training data diversity was the single largest improvement. Switching from 4 domain-specific clips to 800 diverse DIV2K images produced a +7 VMAF swing across all metrics.

## Benchmark Methodology

Every number in this repo is reproducible. The benchmark pipeline:

1. **Source:** 720p ground truth (Blender Open Movie clips — freely downloadable)
2. **Downscale:** Lanczos 720p → 360p
3. **Encode:** SVT-AV1 CRF 35, preset 6 (simulates real-world AV1 streaming)
4. **Decode:** AV1 → raw frames (this is the model's input)
5. **Upscale:** Lanczos bicubic ↑2x (baseline) OR Ghost Stream model ↑2x
6. **Measure:** VMAF, VMAF NEG, PSNR-Y, CAMBI at 720p vs original

### Cross-Machine Reproducibility

SVT-AV1 produces non-deterministic output across different CPU architectures (AVX-512 vs AVX2, thread count). Absolute VMAF scores may vary ±1.0 across machines. **The delta (Ghost Stream minus Lanczos) is consistent to ±0.3 because both use the same encoded input.**

The `--verify` flag accounts for this with ±0.5 tolerance.

### VMAF NEG Honesty Check

We report [VMAF NEG](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) alongside standard VMAF for every result. VMAF NEG penalizes enhancement artifacts that inflate standard VMAF scores.

Ghost Stream's NEG gap (VMAF minus VMAF NEG) averages **2.89**, compared to Lanczos's **2.49**. This 0.40 difference indicates the model applies mild perceptual enhancement beyond pure reconstruction. We consider this acceptable but disclose it for full transparency.

## Detailed Results

See [`results/leaderboard.md`](results/leaderboard.md) for a full comparison of all model configurations tested during development.

### Ablation Study

| Configuration | Avg VMAF | vs Lanczos | Note |
|--------------|----------|------------|------|
| Charbonnier only | ~77.0 | -2.00 | Baseline loss |
| + Adversarial Lanczos loss | ~77.5 | -0.98 | Only penalize where worse than Lanczos |
| + Knowledge distillation | ~78.1 | -0.34 | Generic SwinIR teacher |
| + Diverse training data (DIV2K) | **~89.5** | **+4.71** | 800 images vs 4 clips |

*Ablation measured on private Pixel phone clips (first 3 rows) and standard sequences (final row). Deltas are consistent across test sets.*

## Model Details

| Property | Value |
|----------|-------|
| Architecture | SPANPlus (24ch, 4 blocks, sigmoid attention) |
| Parameters | 47,980 |
| Weight file size | ~200KB |
| Input | Y-channel, any resolution (fully convolutional) |
| Output | 2x input resolution, Y-channel |
| Scale factor | 2x |
| Training data | DIV2K (800 images) with AV1 CRF 35 degradation |
| Training steps | 600,000 |
| Training cost | ~$0.50 (RunPod 4070 Ti, 77 minutes) |

## What's NOT in This Repo

The following are proprietary and not published:

- Training code and loss function implementations
- Adversarial Lanczos loss details
- Knowledge distillation pipeline
- Training data preprocessing scripts
- Experiment configuration files

The model weights are published under MIT license. The architecture (SPAN) is public. How we trained the model is our competitive advantage.

## Browser Deployment

Ghost Stream is designed for client-side super-resolution in web browsers via WebGPU:

- **Model size:** ~200KB (smaller than a typical hero image)
- **Runtime:** No server-side inference — runs entirely on the viewer's device
- **Compatibility:** WebGPU (Chrome 113+, Safari 18+, Edge 113+)
- **Fallback:** ONNX Runtime Web (WASM) for browsers without WebGPU

A browser demo is planned for [`demo/`](demo/).

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

MIT. See [LICENSE](LICENSE).
