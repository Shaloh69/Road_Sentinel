#!/usr/bin/env python3
"""
Complete Training Pipeline for Vehicle Speed Detection System
Uses YOLOv8 with COCO dataset for vehicle detection (bicycle, car, motorcycle)
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
from datetime import datetime


def check_gpu():
    """Check if GPU is available and display information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        return 0
    else:
        print("⚠️  No GPU found. Training on CPU (will be slower)")
        return 'cpu'


def train_vehicle_detector(
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    image_size=640,
    project_name='vehicle_speed',
    experiment_name='coco_v1'
):
    """
    Train YOLOv8 for vehicle detection

    Args:
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
        project_name: Project directory name
        experiment_name: Experiment name
    """

    # Check GPU availability
    device = check_gpu()

    # Create output directory
    output_dir = f'runs/{project_name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("🚗 VEHICLE SPEED DETECTION - TRAINING PIPELINE")
    print("="*70 + "\n")

    # Model selection
    model_path = f'yolov8{model_size}.pt'
    print(f"📦 Loading pretrained model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"⚠️  Could not load {model_path}, downloading...")
        model = YOLO(model_path)

    print("\n📊 Training Configuration:")
    print(f"   - Model: YOLOv8{model_size}")
    print(f"   - Dataset: COCO 2017 (auto-download)")
    print(f"   - Classes: bicycle, car, motorcycle, bus, truck")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Image size: {image_size}x{image_size}")
    print(f"   - Device: {'GPU' if device == 0 else 'CPU'}")

    # Training parameters
    training_args = {
        'data': 'coco.yaml',           # COCO dataset config (auto-downloads)
        'epochs': epochs,              # Number of epochs
        'imgsz': image_size,           # Image size
        'batch': batch_size,           # Batch size
        'device': device,              # GPU/CPU
        'workers': 8,                  # Data loading workers
        'project': f'runs/{project_name}',
        'name': experiment_name,

        # Optimization
        'optimizer': 'SGD',            # SGD or Adam
        'lr0': 0.01,                   # Initial learning rate
        'lrf': 0.01,                   # Final learning rate
        'momentum': 0.937,             # SGD momentum
        'weight_decay': 0.0005,        # Weight decay

        # Augmentation
        'hsv_h': 0.015,                # Hue augmentation
        'hsv_s': 0.7,                  # Saturation
        'hsv_v': 0.4,                  # Value
        'degrees': 0.0,                # Rotation
        'translate': 0.1,              # Translation
        'scale': 0.5,                  # Scale
        'flipud': 0.0,                 # Flip up-down
        'fliplr': 0.5,                 # Flip left-right
        'mosaic': 1.0,                 # Mosaic augmentation

        # Training settings
        'patience': 20,                # Early stopping patience
        'save': True,                  # Save checkpoints
        'save_period': 10,             # Save every N epochs
        'plots': True,                 # Generate plots
        'val': True,                   # Validate during training

        # Focus on vehicle classes only (optional)
        # Uncomment to train only on specific classes
        # 'classes': [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
    }

    print("\n🚀 Starting training...")
    print("   This may take several hours depending on your hardware.")
    print("   COCO dataset will be auto-downloaded if not present (~20GB)\n")

    start_time = datetime.now()

    try:
        # Train the model
        results = model.train(**training_args)

        elapsed = datetime.now() - start_time

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n📁 Results saved to: runs/{project_name}/{experiment_name}/")
        print(f"🏆 Best model: runs/{project_name}/{experiment_name}/weights/best.pt")
        print(f"📈 Last model: runs/{project_name}/{experiment_name}/weights/last.pt")
        print(f"⏱️  Training time: {elapsed.total_seconds()/3600:.2f} hours")

        # Validate the model
        print("\n🔍 Validating best model...")
        metrics = model.val()

        print("\n📊 Performance Metrics:")
        print(f"   - mAP50: {metrics.box.map50:.3f}")
        print(f"   - mAP50-95: {metrics.box.map:.3f}")
        print(f"   - Precision: {metrics.box.mp:.3f}")
        print(f"   - Recall: {metrics.box.mr:.3f}")

        print("\n" + "="*70)
        print("🎉 Training pipeline completed successfully!")
        print("="*70 + "\n")

        return model, results, metrics

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Last checkpoint saved")
        return None, None, None

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_trained_model(model_path, test_source, save_results=True, conf_threshold=0.25):
    """
    Test the trained model on images or videos

    Args:
        model_path: Path to trained model weights
        test_source: Path to image, video, or folder
        save_results: Whether to save results
        conf_threshold: Confidence threshold for detections
    """

    print("\n" + "="*70)
    print("🧪 TESTING TRAINED MODEL")
    print("="*70 + "\n")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"🎯 Testing on: {test_source}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   Save results: {save_results}\n")

    try:
        results = model.predict(
            source=test_source,
            save=save_results,
            conf=conf_threshold,
            iou=0.45,
            show=False,  # Set to True to display results
            classes=[1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
            verbose=True
        )

        print(f"\n✅ Test complete! Results saved to: runs/detect/predict/")
        return results

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None


def main():
    """Main execution function"""

    import argparse

    parser = argparse.ArgumentParser(
        description='Train YOLOv8 for Vehicle Speed Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (YOLOv8n, 100 epochs)
  python train_vehicle_detector.py

  # Train with larger model and more epochs
  python train_vehicle_detector.py --model s --epochs 150 --batch 32

  # Train on CPU with smaller batch
  python train_vehicle_detector.py --batch 4

  # Test a trained model
  python train_vehicle_detector.py --test --model-path runs/vehicle_speed/coco_v1/weights/best.pt --source test_video.mp4
        """
    )

    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16, reduce if out of memory)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--project', type=str, default='vehicle_speed',
                       help='Project name (default: vehicle_speed)')
    parser.add_argument('--name', type=str, default='coco_v1',
                       help='Experiment name (default: coco_v1)')

    # Testing arguments
    parser.add_argument('--test', action='store_true',
                       help='Test mode instead of training')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model for testing')
    parser.add_argument('--source', type=str,
                       help='Test source (image, video, or folder)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for testing (default: 0.25)')

    args = parser.parse_args()

    if args.test:
        # Testing mode
        if not args.model_path or not args.source:
            print("❌ Error: --model-path and --source are required for testing")
            parser.print_help()
            return

        test_trained_model(args.model_path, args.source, conf_threshold=args.conf)

    else:
        # Training mode
        model, results, metrics = train_vehicle_detector(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            project_name=args.project,
            experiment_name=args.name
        )

        if model is not None:
            print("\n💡 Next steps:")
            print(f"   1. Test your model: python {__file__} --test --model-path runs/{args.project}/{args.name}/weights/best.pt --source your_video.mp4")
            print(f"   2. Use in speed detection: Update the model path in your speed detector script")
            print(f"   3. Review training plots in: runs/{args.project}/{args.name}/")


if __name__ == "__main__":
    main()
