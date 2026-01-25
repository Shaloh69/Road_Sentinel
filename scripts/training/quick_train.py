#!/usr/bin/env python3
"""
Quick Start Training Script - One-Line YOLO Training
Auto-downloads COCO dataset and starts training immediately
"""

from ultralytics import YOLO

print("="*70)
print("🚀 QUICK START: Vehicle Detection Training")
print("="*70)
print("\nThis will:")
print("  1. Auto-download YOLOv8 pretrained weights")
print("  2. Auto-download COCO dataset (~20GB)")
print("  3. Start training on vehicle classes")
print("  4. Save best model to runs/vehicle_speed/quick_v1/weights/best.pt")
print("\n" + "="*70 + "\n")

# Load pretrained model (will auto-download if needed)
print("📦 Loading YOLOv8 nano model...")
model = YOLO('yolov8n.pt')

# Train on COCO dataset (will auto-download if needed)
print("🚀 Starting training...\n")

results = model.train(
    data='coco.yaml',          # Built-in COCO config - auto-downloads!
    epochs=100,                # Training epochs
    imgsz=640,                 # Image size
    batch=16,                  # Batch size (reduce to 8 or 4 if out of memory)
    device=0,                  # Use GPU 0 (or 'cpu' if no GPU)
    workers=8,                 # Number of workers
    project='runs/vehicle_speed',
    name='quick_v1',

    # Performance settings
    patience=20,               # Early stopping
    save=True,                 # Save checkpoints
    plots=True,                # Generate plots

    # Focus on vehicle classes only
    classes=[1, 2, 3, 5, 7],   # bicycle, car, motorcycle, bus, truck
)

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Best model saved to: runs/vehicle_speed/quick_v1/weights/best.pt")
print(f"📊 View training plots in: runs/vehicle_speed/quick_v1/")
print("\n🧪 Test your model:")
print("   yolo predict model=runs/vehicle_speed/quick_v1/weights/best.pt source=your_video.mp4")
print("="*70 + "\n")
