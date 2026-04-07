# Ghost Stream WebGPU Demo

Real-time neural video super-resolution running in your browser.

## Quick Start

```bash
cd demo/
python -m http.server 8080
# Open http://localhost:8080/player.html in Chrome 113+
```

Or just open `player.html` directly (file:// may have CORS issues with weight loading — use a local server).

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU required)
- Any GPU (discrete or integrated)

## Files

| File | Size | Purpose |
|------|------|---------|
| `player.html` | ~6KB | Demo page with side-by-side comparison |
| `player.js` | ~8KB | Video decode → WebGPU inference → canvas render |
| `shader.wgsl` | ~3KB | SPAN forward pass as WebGPU compute shader |
| `manifest.json` | ~1KB | Weight layer names, shapes, offsets |
| `weights.bin` | ~96KB | Float16 model weights |
| `demo_360p.mp4` | ~1MB | Pre-encoded demo clip (Blender Open Movie) |

## Architecture

The SPAN model is implemented as a sequence of WebGPU compute shader dispatches:

```
Input (Y-channel, 360p)
  ↓ Conv2d(1→24, 3×3)        — head
  ↓ Block × 4:
  │   Conv2d(24→24, 3×3)     — conv1
  │   Sigmoid Attention        — sigmoid(x) * x
  │   Conv2d(24→24, 3×3)     — conv2
  │   + Residual skip
  ↓ Conv2d(24→24, 3×3)       — project
  ↓ Conv2d(24→4, 3×3)        — tail
  ↓ PixelShuffle(2)           — rearrange channels to spatial
Output (Y-channel, 720p)
```

Total: 47,980 parameters, ~96KB as float16.

## Performance

Target: 30fps on discrete GPU, 15fps on integrated.
Actual performance depends on GPU and resolution.

## Weight Format

`weights.bin` contains all model parameters as contiguous float16 values.
`manifest.json` maps layer names to byte offsets and shapes.

To regenerate from a PyTorch checkpoint:
```python
python export_weights.py path/to/checkpoint.pth demo/
```
