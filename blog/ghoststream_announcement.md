# I Built a Neural Video Upscaler That Beats Lanczos in a Browser — For $200 in GPU Costs

Video delivery is expensive. A 10-minute 1080p video at 8 Mbps is 600MB. Serve it to a million viewers and you're looking at 600TB of bandwidth. The industry's answer is ABR (adaptive bitrate) — store 4-6 copies of every video at different resolutions and let the client pick. That's 4-6x the storage cost, and the lowest quality tier still looks bad because bicubic upscaling can't recover detail that was thrown away by compression.

What if you could serve one tiny 360p file and have the viewer's browser reconstruct the 720p version — better than Lanczos upscaling, using a 200KB neural network running entirely client-side? That's Ghost Stream.

## The Model

Ghost Stream uses a [SPAN](https://github.com/hongyuanyu/SPAN) architecture — 48K parameters, 200KB weight file. It processes only the Y (luma) channel, leaving chroma to standard resize. The entire model fits in a single WebGPU compute shader.

The training recipe has three components:

1. **Adversarial baseline loss.** Instead of penalizing all reconstruction errors equally, we only compute gradients on pixels where the model is worse than Lanczos upscaling. This focuses the model's limited 48K-parameter capacity on the regions where classical methods actually fail — compression artifacts, fine textures, sharp edges.

2. **Knowledge distillation.** A generic [SwinIR-light](https://github.com/JingyunLiang/SwinIR) teacher model (910K params) pre-computes "ideal" outputs. The student learns from both the ground truth and the teacher's reconstruction. Counterintuitively, a generic teacher trained on DIV2K worked better than one fine-tuned to our specific content.

3. **Realistic degradation.** Training pairs are generated through our actual deployment pipeline: HR image → Lanczos downscale → SVT-AV1 CRF 35 encode → decode. This means the model sees real AV1 compression artifacts during training, not clean bicubic-downscaled images.

## The Results

Tested on [Blender Open Movie](https://www.blender.org/about/projects/) sequences — freely downloadable, widely recognized in video research:

| Sequence | Type | Lanczos VMAF | Ghost Stream VMAF | Delta |
|----------|------|-------------|-------------------|-------|
| Big Buck Bunny | Animation | 89.23 | 90.99 | +1.76 |
| Elephants Dream | CGI/Surreal | 72.31 | 81.33 | +9.02 |
| Sintel | CGI/Dark | 90.43 | 93.14 | +2.71 |
| Tears of Steel | Live Action + VFX | 87.05 | 92.40 | +5.35 |
| **Average** | | **84.76** | **89.47** | **+4.71** |

Every number is reproducible. [Clone the repo](https://github.com/AxiomState/ghoststream-benchmark), run one command, verify yourself:

```bash
python benchmark/benchmark.py --all --download --verify
```

We also report VMAF NEG (Netflix's anti-gaming metric) alongside standard VMAF. Ghost Stream's NEG gap is 2.89 vs Lanczos's 2.49 — a 0.40 difference indicating mild perceptual enhancement, not metric exploitation.

## The Journey

This project took a week of iteration, mostly fighting infrastructure rather than model design.

**Round 1: Wrong data, wrong metrics.** Started with 4 phone clips, a DataLoader that ran at 8 it/s, and a benchmark that compared at the wrong resolution. VMAF showed numbers that were off by 20 points. Spent two days fixing measurement before any real training could start.

**Round 2: Loss function experiments.** Tried Charbonnier, MS-SSIM, DLM, VIF, optimal transport, focal frequency loss. VIF caused the model to hallucinate textures (400x the expected variance). DLM diverged due to division by near-zero. The winning formula was adversarial Lanczos loss — dead simple, conceptually clean.

**Round 3: Distillation bug.** Implemented knowledge distillation from a SwinIR teacher. The model trained for 600K steps and scored -2.27 VMAF vs Lanczos — worse than the non-distilled version. Turns out the distillation loss was using ground truth as the target instead of the teacher's output. The crop positions applied to GT and LR weren't being applied to teacher outputs. A one-line fix (`teacher_b = teacher_outputs[idx,:,y*2:(y+PS)*2,x*2:(x+PS)*2]`) improved the result from -2.27 to -0.34.

**Round 4: The obvious thing.** After all the loss engineering, the single biggest improvement came from using standard SR training data (DIV2K, 800 images) instead of 4 phone clips. The model jumped from -0.34 to +3.32 VMAF vs Lanczos on private clips, and +4.71 on standard sequences. I should have done this first.

**Total GPU cost: ~$200** — including $110 wasted on idle pods that didn't auto-shutdown (a bug in the infrastructure code where `RUNPOD_POD_ID` was always empty).

## What's Next

- **WebGPU deployment:** Ship a working browser demo with side-by-side comparison
- **Temporal consistency:** Currently frame-by-frame SR with no temporal modeling
- **Scale up:** Test 150K parameter models for higher quality ceiling
- **Full benchmark suite:** Xiph/Derf test sequences, BD-rate curves, multi-CRF evaluation

## Try It Yourself

- [Benchmark repo](https://github.com/AxiomState/ghoststream-benchmark) — model weights, inference code, one-command benchmark
- [Technical paper](https://github.com/AxiomState/ghoststream-benchmark/blob/main/paper/ghoststream.pdf) — full methodology and results

The model is 200KB. The benchmark runs in 10 minutes. Every claim is verifiable. That's the whole point.
