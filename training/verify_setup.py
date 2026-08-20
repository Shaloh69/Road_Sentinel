#!/usr/bin/env python3
"""
Verify Road Sentinel Setup
Checks if all dependencies are installed correctly
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - MISSING ({e})")
        return False

def check_ultralytics():
    """Check ultralytics and YOLO specifically"""
    try:
        from ultralytics import YOLO
        print(f"✅ {'ultralytics':20s} - OK (YOLO import successful)")

        # Check version
        import ultralytics
        version = ultralytics.__version__
        print(f"   Version: {version}")
        return True
    except ImportError as e:
        print(f"❌ {'ultralytics':20s} - FAILED")
        print(f"   Error: {e}")
        print(f"   Fix: pip install ultralytics")
        return False

def check_torch():
    """Check PyTorch and GPU availability"""
    try:
        import torch
        print(f"✅ {'torch':20s} - OK")
        print(f"   Version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"   ⚠️  No GPU detected - will use CPU (slower)")
        return True
    except ImportError as e:
        print(f"❌ {'torch':20s} - MISSING")
        print(f"   Fix: pip install torch torchvision")
        return False

def main():
    print("="*70)
    print("🔍 ROAD SENTINEL SETUP VERIFICATION")
    print("="*70 + "\n")

    print("Checking Python packages...\n")

    all_ok = True

    # Core dependencies
    all_ok &= check_ultralytics()
    all_ok &= check_torch()
    all_ok &= check_package("opencv-python", "cv2")
    all_ok &= check_package("numpy")

    # Optional dependencies
    print("\nOptional packages:")
    check_package("pandas")
    check_package("matplotlib")
    check_package("seaborn")
    check_package("pillow", "PIL")

    print("\n" + "="*70)

    if all_ok:
        print("✅ ALL CORE DEPENDENCIES INSTALLED!")
        print("="*70)
        print("\n🚀 You're ready to start training!\n")
        print("Quick start:")
        print("  python run_merge_busay.py")
        print("  python train.py --dataset vehicle --model-size n --epochs 100")
        print("\nOr see docs/TRAINING_GUIDE.md for more options.")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING")
        print("="*70)
        print("\n📦 Install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nThen run this script again to verify.")
        sys.exit(1)

    print()

if __name__ == "__main__":
    main()
