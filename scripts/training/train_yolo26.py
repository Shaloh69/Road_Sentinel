#!/usr/bin/env python3
"""
YOLO26 Training Script for Road Sentinel

Trains two models:
1. Vehicle Detection (busay_vehicle_detection)
2. Accident Detection (busay_accident_detection)

Usage:
    python train_yolo26.py --dataset vehicle --epochs 100
    python train_yolo26.py --dataset accident --epochs 100
    python train_yolo26.py --dataset both --epochs 100
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics --upgrade")
    exit(1)


# Dataset paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "processed"

DATASETS = {
    "vehicle": {
        "name": "busay_vehicle_detection",
        "yaml": DATASETS_DIR / "busay_vehicle_detection" / "data.yaml",
        "description": "Vehicle detection (car, motorcycle, bus, truck, bicycle)"
    },
    "accident": {
        "name": "busay_accident_detection",
        "yaml": DATASETS_DIR / "busay_accident_detection" / "data.yaml",
        "description": "Accident/incident detection"
    }
}

# Training output directory
RUNS_DIR = PROJECT_ROOT / "models" / "runs"


def train_model(
    dataset_key: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",  # GPU device or "cpu"
    resume: bool = False,
    pretrained: str = None
):
    """
    Train YOLO26 model on specified dataset

    Args:
        dataset_key: 'vehicle' or 'accident'
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: GPU device ID or 'cpu'
        resume: Resume from last checkpoint
        pretrained: Path to pretrained weights (optional)
    """

    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return None

    dataset = DATASETS[dataset_key]
    data_yaml = dataset["yaml"]

    if not data_yaml.exists():
        print(f"Dataset not found: {data_yaml}")
        print("Please ensure the dataset exists at the specified path.")
        return None

    print("=" * 70)
    print(f"YOLO26 Training - {dataset['description']}")
    print("=" * 70)
    print(f"Dataset: {dataset['name']}")
    print(f"Data YAML: {data_yaml}")
    print(f"Model: yolo26{model_size}.pt")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print("=" * 70)

    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_key}_yolo26{model_size}_{timestamp}"

    # Load YOLO26 model
    if pretrained and Path(pretrained).exists():
        print(f"Loading pretrained weights: {pretrained}")
        model = YOLO(pretrained)
    else:
        model_name = f"yolo26{model_size}.pt"
        print(f"Loading base model: {model_name}")
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"YOLO26 not available yet, falling back to YOLO11: {e}")
            model_name = f"yolo11{model_size}.pt"
            print(f"Loading fallback model: {model_name}")
            model = YOLO(model_name)

    # Training parameters
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(RUNS_DIR / dataset_key),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "patience": 50,  # Early stopping patience
        "save": True,
        "save_period": 10,  # Save checkpoint every N epochs
        "cache": True,  # Cache images for faster training
        "workers": 8,
        "cos_lr": True,  # Cosine learning rate scheduler
        "close_mosaic": 10,  # Disable mosaic augmentation for last N epochs
        "amp": True,  # Automatic mixed precision
    }

    if resume:
        train_args["resume"] = True

    print("\nStarting training...")
    print(f"Output will be saved to: {RUNS_DIR / dataset_key / run_name}")
    print()

    # Train the model
    results = model.train(**train_args)

    # Print results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    best_weights = RUNS_DIR / dataset_key / run_name / "weights" / "best.pt"
    print(f"Best weights saved to: {best_weights}")

    return best_weights


def validate_model(weights_path: str, data_yaml: str, device: str = "0"):
    """Validate a trained model"""
    model = YOLO(weights_path)
    results = model.val(data=data_yaml, device=device)
    return results


def export_model(weights_path: str, format: str = "onnx"):
    """Export model to different formats"""
    model = YOLO(weights_path)
    model.export(format=format)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 models for Road Sentinel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train vehicle detection model
  python train_yolo26.py --dataset vehicle --epochs 100

  # Train accident detection model
  python train_yolo26.py --dataset accident --epochs 100

  # Train both models
  python train_yolo26.py --dataset both --epochs 100

  # Train with larger model
  python train_yolo26.py --dataset vehicle --model-size s --epochs 150

  # Resume training
  python train_yolo26.py --dataset vehicle --resume
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["vehicle", "accident", "both"],
        required=True,
        help="Dataset to train on"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge). Default: n"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training. Default: 640"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Default: 16"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID or 'cpu'. Default: 0"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained weights"
    )

    args = parser.parse_args()

    # Check ultralytics version
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")

    if args.dataset == "both":
        # Train both models
        print("\n" + "=" * 70)
        print("Training BOTH models")
        print("=" * 70 + "\n")

        # Vehicle detection first
        vehicle_weights = train_model(
            "vehicle",
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained
        )

        # Then accident detection
        accident_weights = train_model(
            "accident",
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained
        )

        print("\n" + "=" * 70)
        print("All training complete!")
        print("=" * 70)
        print(f"Vehicle model: {vehicle_weights}")
        print(f"Accident model: {accident_weights}")

    else:
        # Train single model
        train_model(
            args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained
        )


if __name__ == "__main__":
    main()
