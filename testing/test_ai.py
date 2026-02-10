#!/usr/bin/env python3
"""
Test script for Road Sentinel AI Service
Downloads a sample traffic image and tests the detection endpoints
"""

import requests
import time
import sys
from pathlib import Path

# AI Service URL
AI_SERVICE_URL = "http://localhost:8000"

# Sample traffic image URL (from Unsplash)
SAMPLE_IMAGE_URL = "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop"

def test_health_check():
    """Test the health endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{AI_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the AI service is running on http://localhost:8000")
        return False

def download_sample_image():
    """Download a sample traffic image"""
    print("\n📥 Downloading sample traffic image...")
    try:
        response = requests.get(SAMPLE_IMAGE_URL, timeout=10)
        if response.status_code == 200:
            image_path = Path("test_image.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Sample image downloaded: {image_path}")
            return image_path
        else:
            print(f"❌ Failed to download image: status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None

def test_detection(image_path, endpoint="/api/detect"):
    """Test detection endpoint"""
    endpoint_name = endpoint.split('/')[-1]
    print(f"\n🚗 Testing {endpoint_name} endpoint...")

    try:
        with open(image_path, "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            data = {
                "camera_id": "TEST-CAM-001",
                "confidence_threshold": "0.5"
            }

            start_time = time.time()
            response = requests.post(
                f"{AI_SERVICE_URL}{endpoint}",
                files=files,
                data=data,
                timeout=30
            )
            processing_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Detection successful!")
                print(f"   Processing time: {processing_time:.2f}ms")
                print(f"   Server reported: {result.get('processing_time_ms', 0):.2f}ms")

                # Show detections
                if 'detections' in result and result['detections']:
                    print(f"\n   🚙 Vehicles detected: {len(result['detections'])}")
                    for i, det in enumerate(result['detections'][:5], 1):  # Show first 5
                        print(f"      {i}. {det['class']} (confidence: {det['confidence']:.2f})")
                        print(f"         bbox: x={det['bbox']['x']}, y={det['bbox']['y']}, "
                              f"w={det['bbox']['width']}, h={det['bbox']['height']}")
                else:
                    print("   No vehicles detected")

                # Show incidents
                if 'incidents' in result and result['incidents']:
                    print(f"\n   ⚠️  Incidents detected: {len(result['incidents'])}")
                    for i, inc in enumerate(result['incidents'], 1):
                        print(f"      {i}. {inc['type']} - {inc['severity']} severity")
                        print(f"         {inc['description']}")
                else:
                    print("   No incidents detected")

                return True
            else:
                print(f"❌ Detection failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("\n📊 Testing stats endpoint...")
    try:
        response = requests.get(f"{AI_SERVICE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats retrieved successfully!")
            print(f"   Traffic model: {stats['traffic_model']['path']}")
            print(f"   Incident model: {stats['incident_model']['path']}")
            print(f"   Device: {stats['traffic_model']['device']}")
            print(f"   Confidence threshold: {stats['configuration']['confidence_threshold']}")
            return True
        else:
            print(f"❌ Stats failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚦 Road Sentinel AI Service Test Suite")
    print("=" * 60)

    # Test 1: Health check
    if not test_health_check():
        print("\n❌ AI service is not running. Please start it first:")
        print("   cd server/ai-service")
        print("   python -m app.main")
        sys.exit(1)

    # Test 2: Stats
    test_stats()

    # Test 3: Download sample image
    image_path = download_sample_image()
    if not image_path:
        print("\n⚠️  Could not download sample image. Tests incomplete.")
        sys.exit(1)

    # Test 4: Combined detection
    test_detection(image_path, "/api/detect")

    # Test 5: Traffic only
    test_detection(image_path, "/api/detect/traffic")

    # Test 6: Incidents only
    test_detection(image_path, "/api/detect/incidents")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n💡 Tip: You can also test manually with curl:")
    print(f"   curl -X POST {AI_SERVICE_URL}/api/detect \\")
    print(f"     -F 'image=@test_image.jpg' \\")
    print(f"     -F 'camera_id=TEST-001'")

    # Clean up
    if image_path and image_path.exists():
        image_path.unlink()
        print(f"\n🧹 Cleaned up test image")

if __name__ == "__main__":
    main()
