#!/usr/bin/env python3
"""Download standard test clips for Ghost Stream benchmark.

Uses Blender Open Movie clips — freely redistributable, widely recognized.
All clips are trimmed to 10 seconds and losslessly encoded to 720p MP4.
"""

import os
import subprocess
import sys
import tempfile

CLIPS = {
    "big_buck_bunny": {
        "url": "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.avi",
        "ss": "60", "t": "10",
        "desc": "Animation — clean edges, smooth gradients, bright colors",
    },
    "sintel": {
        "url": "http://peach.themazzone.com/durian/movies/sintel-1024-surround.mp4",
        "ss": "180", "t": "10",
        "desc": "CGI — dark scenes, fine detail, dramatic lighting",
    },
    "tears_of_steel": {
        "url": "http://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/tears_of_steel_720p.mov",
        "ss": "60", "t": "10",
        "desc": "Live action + VFX — real actors, CGI compositing",
    },
    "elephants_dream": {
        "url": "https://archive.org/download/ElephantsDream/ed_hd.avi",
        "ss": "120", "t": "10",
        "desc": "CGI/Surreal — organic textures, complex geometry",
    },
}

def download_clip(name, info, output_dir):
    """Download and process a single clip."""
    output_path = os.path.join(output_dir, f"{name}_720p.mp4")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000:
        print(f"  {name}: already exists ({os.path.getsize(output_path) // 1024}KB)")
        return output_path

    print(f"  {name}: downloading from {info['url'][:60]}...")

    with tempfile.NamedTemporaryFile(suffix=".tmp") as tmp:
        # Download
        dl_cmd = ["wget", "-q", "--timeout=120", "-O", tmp.name, info["url"]]
        result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or os.path.getsize(tmp.name) < 1000:
            print(f"  {name}: download FAILED")
            return None

        # Trim + scale to 720p + lossless encode
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", tmp.name,
            "-ss", info["ss"], "-t", info["t"],
            "-vf", "scale=1280:720:flags=lanczos",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-an", output_path,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  {name}: ffmpeg processing FAILED")
            return None

    size = os.path.getsize(output_path)
    print(f"  {name}: OK ({size // 1024}KB) — {info['desc']}")
    return output_path


def download_all(output_dir="clips"):
    """Download all standard test clips."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Blender Open Movie test clips to {output_dir}/\n")

    results = {}
    for name, info in CLIPS.items():
        path = download_clip(name, info, output_dir)
        if path:
            results[name] = path

    print(f"\nDownloaded {len(results)}/{len(CLIPS)} clips")
    if len(results) < 3:
        print("WARNING: Fewer than 3 clips available. Results may not be representative.")

    return results


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "clips"
    download_all(output_dir)
