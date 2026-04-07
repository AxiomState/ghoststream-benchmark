"""
Ghost Stream — SPANPlus Architecture (Inference Only)

SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution
Based on: https://github.com/hongyuanyu/SPAN (ECCV 2024 AIS Workshop)
Modified for single-channel (Y) processing with sigmoid attention.

Architecture: 24 channels, 4 blocks, PixelShuffle 2x upsampling
Parameters: 47,980
Input: Y-channel (N, 1, H, W) float32 [0, 1]
Output: Y-channel (N, 1, 2H, 2W) float32 [0, 1]
Scale: 2x

This file is self-contained — no external dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn


class SigmoidAttention(nn.Module):
    """Parameter-free attention: sigmoid gating."""
    def forward(self, x):
        return torch.sigmoid(x) * x


class ResidualBlock(nn.Module):
    """Conv → SigmoidAttention → Conv + skip connection."""
    def __init__(self, channels=24):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attn = SigmoidAttention()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.attn(self.conv1(x))) + x


class GhostStreamSPAN(nn.Module):
    """
    Ghost Stream SPANPlus — 48K parameter super-resolution model.

    Architecture:
        head (1→24 channels) → 4 × ResidualBlock → project (24→24) →
        tail (24→4) → PixelShuffle(2) → output (1 channel, 2x resolution)

    Parameters: 47,980
        head:     24×1×3×3 + 24 = 240
        blocks:   4 × (24×24×3×3×2 + 24×2) = 41,664
        project:  24×24×3×3 + 24 = 5,208
        tail:     4×24×3×3 + 4 = 868
        Total:    47,980
    """
    def __init__(self, channels=24, num_blocks=4, scale=2):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.project = nn.Conv2d(channels, channels, 3, padding=1)
        self.tail = nn.Conv2d(channels, scale * scale, 3, padding=1)
        self.upsample = nn.PixelShuffle(scale)

    def forward(self, x):
        """
        Args:
            x: (N, 1, H, W) Y-channel input, float32, range [0, 1]
        Returns:
            (N, 1, 2H, 2W) Y-channel output, float32
        """
        return self.upsample(self.tail(self.project(self.blocks(self.head(x)))))


def load_model(checkpoint_path, device="cpu"):
    """
    Load Ghost Stream model from checkpoint.

    Args:
        checkpoint_path: Path to .pth file (200KB)
        device: "cpu" or "cuda"

    Returns:
        Loaded model in eval mode
    """
    model = GhostStreamSPAN()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle both standalone (22 keys) and NeoSR (204 keys) formats
    if len(state_dict) == 22:
        # Standalone format: h, b.0.c1, b.0.c2, ..., p, t
        mapped = {}
        key_map = {
            'h.weight': 'head.weight', 'h.bias': 'head.bias',
            'p.weight': 'project.weight', 'p.bias': 'project.bias',
            't.weight': 'tail.weight', 't.bias': 'tail.bias',
        }
        for i in range(4):
            key_map[f'b.{i}.c1.weight'] = f'blocks.{i}.conv1.weight'
            key_map[f'b.{i}.c1.bias'] = f'blocks.{i}.conv1.bias'
            key_map[f'b.{i}.c2.weight'] = f'blocks.{i}.conv2.weight'
            key_map[f'b.{i}.c2.bias'] = f'blocks.{i}.conv2.bias'

        for old_key, new_key in key_map.items():
            if old_key in state_dict:
                mapped[new_key] = state_dict[old_key]
        model.load_state_dict(mapped)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Ghost Stream model loaded: {param_count:,} parameters")

    return model


if __name__ == "__main__":
    # Quick test
    model = GhostStreamSPAN()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    x = torch.randn(1, 1, 360, 640)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (1, 1, 720, 1280), f"Expected (1,1,720,1280), got {y.shape}"
    print("Architecture test PASSED")
