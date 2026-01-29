#!/usr/bin/env python3
"""
Download sample traffic videos for testing
"""

import urllib.request
import os
from pathlib import Path


def download_sample_videos():
    """
    Download free sample traffic videos from Pexels
    """

    print("\n" + "="*70)
    print("📥 DOWNLOADING SAMPLE TRAFFIC VIDEOS")
    print("="*70 + "\n")

    # Create test_videos folder
    output_dir = Path("test_videos")
    output_dir.mkdir(exist_ok=True)

    # Sample video URLs (these are free stock videos)
    videos = {
        "traffic_highway.mp4": "https://videos.pexels.com/video-files/2103099/2103099-hd_1920_1080_30fps.mp4",
        "urban_traffic.mp4": "https://videos.pexels.com/video-files/854675/854675-hd_1920_1080_30fps.mp4",
        "motorcycle_traffic.mp4": "https://videos.pexels.com/video-files/3015486/3015486-hd_1920_1080_25fps.mp4",
    }

    print("ℹ️  Note: This will download ~100MB of video files\n")

    for filename, url in videos.items():
        output_path = output_dir / filename

        if output_path.exists():
            print(f"✅ {filename} already exists, skipping...")
            continue

        try:
            print(f"⬇️  Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ Downloaded {filename} ({file_size:.1f} MB)")

        except Exception as e:
            print(f"   ❌ Failed to download {filename}: {e}")

    print(f"\n✅ Sample videos saved to: {output_dir.absolute()}")
    print("\nTest your model with:")
    print(f"  python test_model.py --model runs/detect/runs/vehicle_speed/busay_v1/weights/best.pt --source test_videos/traffic_highway.mp4\n")


if __name__ == "__main__":
    download_sample_videos()
