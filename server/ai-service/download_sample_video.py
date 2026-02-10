#!/usr/bin/env python3
"""
Download sample traffic video for testing
"""
import requests
import sys

# Sample traffic videos (public domain)
SAMPLE_VIDEOS = {
    '1': {
        'name': 'Traffic Scene 1',
        'url': 'https://www.pexels.com/download/video/2103099/?fps=25.0&h=1080&w=1920',
        'filename': 'traffic_sample.mp4'
    }
}

def download_video(url, filename):
    """Download video from URL"""
    print(f"Downloading {filename}...")
    print(f"URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        print(f"\n✅ Downloaded: {filename}")
        print(f"Size: {downloaded / 1024 / 1024:.1f} MB")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

if __name__ == "__main__":
    print("Sample Traffic Video Downloader")
    print("=" * 50)

    # You can manually specify a video URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
        filename = sys.argv[2] if len(sys.argv) > 2 else 'traffic_video.mp4'
        download_video(url, filename)
    else:
        print("\nUsage:")
        print("  python download_sample_video.py <video_url> [output_filename]")
        print("\nExample:")
        print("  python download_sample_video.py https://example.com/video.mp4 test.mp4")
