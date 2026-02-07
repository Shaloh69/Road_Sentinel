#!/usr/bin/env python3
"""
YOLO26 Training Script for Road Sentinel
Optimized for NVIDIA RTX 3060 Ti (8GB VRAM)

Trains two models:
1. Vehicle Detection (busay_vehicle_detection)
2. Accident Detection (busay_accident_detection)

Usage:
    python train_yolo26.py --dataset vehicle --epochs 100
    python train_yolo26.py --dataset accident --epochs 100
    python train_yolo26.py --dataset both --epochs 100

GPU Memory Guide (RTX 3060 Ti - 8GB VRAM):
    Model Nano  (n): batch 16, ~3-4 GB VRAM
    Model Small (s): batch 8,  ~5-6 GB VRAM
    Model Medium(m): batch 4,  ~7-8 GB VRAM  (tight, may need batch 2)
    Model Large (l): NOT recommended for 8GB (use --batch 2 at your own risk)
    Model XLarge(x): NOT recommended for 8GB
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

try:
    import torch
except ImportError:
    print("PyTorch not installed. Install with CUDA support first:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

try:
    from ultralytics import YOLO
    import ultralytics
except ImportError:
    print("Ultralytics not installed. Install with:")
    print("  pip install ultralytics>=8.3.0")
    sys.exit(1)


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

# RTX 3060 Ti recommended batch sizes per model size (8GB VRAM)
GPU_8GB_BATCH_SIZES = {
    "n": 16,  # Nano: plenty of headroom
    "s": 8,   # Small: comfortable fit
    "m": 4,   # Medium: tight fit
    "l": 2,   # Large: may OOM, not recommended
    "x": 1,   # XLarge: will likely OOM
}


def check_gpu():
    """Check GPU availability and display info"""
    print("=" * 70)
    print("GPU / CUDA Information")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Training will run on CPU (very slow).")
        print("Make sure you installed PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return "cpu"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_version = torch.version.cuda
    print(f"GPU:          {gpu_name}")
    print(f"VRAM:         {gpu_memory:.1f} GB")
    print(f"CUDA Version: {cuda_version}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Ultralytics:  {ultralytics.__version__}")
    print("=" * 70)
    return "0"


def get_recommended_batch(model_size: str, vram_gb: float = 8.0) -> int:
    """Get recommended batch size based on model size and VRAM"""
    if vram_gb >= 24:  # RTX 4090, A5000, etc.
        return {"n": 64, "s": 32, "m": 16, "l": 8, "x": 4}[model_size]
    elif vram_gb >= 12:  # RTX 4070 Ti, 3080, etc.
        return {"n": 32, "s": 16, "m": 8, "l": 4, "x": 2}[model_size]
    elif vram_gb >= 8:  # RTX 3060 Ti, 3070, etc.
        return GPU_8GB_BATCH_SIZES[model_size]
    else:  # 6GB or less
        return {"n": 8, "s": 4, "m": 2, "l": 1, "x": 1}[model_size]


def train_model(
    dataset_key: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    device: str = "0",
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
        batch: Batch size (-1 for auto based on GPU memory)
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
        print()
        print("You need to prepare your dataset first. Run:")
        print(f"  cd {PROJECT_ROOT / 'scripts' / 'training'}")
        print("  python run_merge_busay.py")
        print()
        print("Or download datasets from Roboflow:")
        print("  python download_roboflow_datasets.py")
        return None

    # Fix data.yaml path to be absolute (YOLO resolves relative to CWD, not yaml location)
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    yaml_dir = data_yaml.resolve().parent
    data_path = Path(data_config.get('path', '.'))
    if not data_path.is_absolute():
        data_config['path'] = str(yaml_dir / data_path)
    # Write a resolved copy next to the original
    resolved_yaml = yaml_dir / 'data_resolved.yaml'
    with open(resolved_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    data_yaml = resolved_yaml

    # Auto-detect batch size for 8GB GPUs
    if batch == -1:
        if device != "cpu" and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch = get_recommended_batch(model_size, vram)
        else:
            batch = 4  # Conservative CPU default

    # Warn if model is too large for 8GB
    if model_size in ("l", "x") and device != "cpu" and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram <= 8:
            print(f"WARNING: YOLO26-{model_size.upper()} may not fit in {vram:.0f}GB VRAM.")
            print(f"Recommended: use model size 'n' or 's' for your GPU.")
            print(f"Continuing with batch size {batch}... (reduce with --batch if OOM)")
            print()

    print("=" * 70)
    print(f"YOLO26 Training - {dataset['description']}")
    print("=" * 70)
    print(f"  Dataset:    {dataset['name']}")
    print(f"  Data YAML:  {data_yaml}")
    print(f"  Model:      yolo26{model_size}.pt")
    print(f"  Epochs:     {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Device:     {device}")
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
        model = YOLO(model_name)

    # Training parameters optimized for RTX 3060 Ti
    train_args = {
        "data": str(data_yaml.resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(RUNS_DIR / dataset_key),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "patience": 50,           # Early stopping patience
        "save": True,
        "save_period": 10,        # Save checkpoint every 10 epochs
        "cache": "ram",           # Cache images in RAM for faster training
        "workers": 8,             # Data loading workers
        "cos_lr": True,           # Cosine learning rate scheduler
        "close_mosaic": 10,       # Disable mosaic for last 10 epochs
        "amp": True,              # Mixed precision - CRITICAL for 8GB VRAM (saves ~40%)
        "plots": True,            # Generate training plots
        "val": True,              # Validate during training
    }

    if resume:
        train_args["resume"] = True

    print(f"\nStarting training...")
    print(f"Output: {RUNS_DIR / dataset_key / run_name}")
    print()

    start_time = datetime.now()

    try:
        results = model.train(**train_args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print()
            print("=" * 70)
            print("OUT OF MEMORY ERROR")
            print("=" * 70)
            print(f"Your GPU ran out of memory with batch size {batch}.")
            print(f"Try again with a smaller batch size:")
            print(f"  python train_yolo26.py --dataset {dataset_key} --model-size {model_size} --batch {max(1, batch // 2)}")
            if model_size not in ("n",):
                print(f"  Or use a smaller model:")
                print(f"  python train_yolo26.py --dataset {dataset_key} --model-size n --batch 16")
            return None
        raise

    elapsed = datetime.now() - start_time

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)

    best_weights = RUNS_DIR / dataset_key / run_name / "weights" / "best.pt"
    last_weights = RUNS_DIR / dataset_key / run_name / "weights" / "last.pt"
    print(f"  Best weights: {best_weights}")
    print(f"  Last weights: {last_weights}")
    print(f"  Training time: {elapsed.total_seconds()/3600:.2f} hours")
    print(f"  Training plots: {RUNS_DIR / dataset_key / run_name}")

    # Run validation on best weights
    print("\nValidating best model...")
    val_results = model.val(data=str(data_yaml), device=device)
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall:    {val_results.box.mr:.4f}")

    return best_weights


def validate_model(weights_path: str, data_yaml: str, device: str = "0"):
    """Validate a trained model"""
    model = YOLO(weights_path)
    results = model.val(data=data_yaml, device=device)
    return results


def export_model(weights_path: str, fmt: str = "onnx"):
    """Export model to different formats (onnx, torchscript, engine, etc.)"""
    model = YOLO(weights_path)
    model.export(format=fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 models for Road Sentinel (optimized for RTX 3060 Ti)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RTX 3060 Ti (8GB VRAM) Recommended Settings:
  Nano  (n): --batch 16  (~3-4 GB, fastest training)
  Small (s): --batch 8   (~5-6 GB, good accuracy)
  Medium(m): --batch 4   (~7-8 GB, best accuracy for 8GB)

Examples:
  # Train vehicle detection (recommended for 3060 Ti)
  python train_yolo26.py --dataset vehicle --model-size s --epochs 100

  # Train accident detection
  python train_yolo26.py --dataset accident --model-size n --epochs 100

  # Train both models sequentially
  python train_yolo26.py --dataset both --model-size s --epochs 100

  # Resume interrupted training
  python train_yolo26.py --dataset vehicle --resume

  # Use auto batch size detection
  python train_yolo26.py --dataset vehicle --model-size s
        """
    )

    parser.add_argument(
        "--dataset", type=str, choices=["vehicle", "accident", "both"],
        required=True, help="Dataset to train on"
    )
    parser.add_argument(
        "--model-size", type=str, default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (default: s for RTX 3060 Ti)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=-1,
        help="Batch size (-1 = auto based on GPU memory, default: -1)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="GPU device ID or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained weights"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        choices=["onnx", "torchscript", "engine"],
        help="Export trained model to format after training"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = check_gpu()
    else:
        check_gpu()

    if args.dataset == "both":
        print()
        print("=" * 70)
        print("Training BOTH models (vehicle + accident)")
        print("=" * 70)
        print()

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

        print()
        print("=" * 70)
        print("All training complete!")
        print("=" * 70)
        print(f"  Vehicle model:  {vehicle_weights}")
        print(f"  Accident model: {accident_weights}")

        if args.export and vehicle_weights:
            export_model(str(vehicle_weights), args.export)
        if args.export and accident_weights:
            export_model(str(accident_weights), args.export)

    else:
        weights = train_model(
            args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained
        )

        if args.export and weights:
            export_model(str(weights), args.export)

    print()
    print("Next steps:")
    print("  1. Check training plots in models/runs/<dataset>/")
    print("  2. Copy best.pt to models/production/")
    print("  3. Update server/ai-service/.env with the model path")
    print("  4. Test with: python test_model.py --weights <path_to_best.pt>")


if __name__ == "__main__":
    main()
