#!/usr/bin/env python3
"""
Download and Use Roboflow Datasets for YOLO
Already in YOLO format - no conversion needed!
"""

from roboflow import Roboflow
import os

def download_roboflow_dataset(api_key, workspace, project, version=1):
    """
    Download dataset from Roboflow Universe

    Args:
        api_key: Your Roboflow API key (free at roboflow.com)
        workspace: Workspace name
        project: Project name
        version: Dataset version (default: 1)
    """

    print("="*70)
    print("📥 DOWNLOADING ROBOFLOW DATASET")
    print("="*70)
    print()

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Get project
    project_obj = rf.workspace(workspace).project(project)

    # Download in YOLOv8 format
    print(f"📦 Downloading {project} from {workspace}...")
    print("   Format: YOLOv8")
    print()

    dataset = project_obj.version(version).download("yolov8")

    print()
    print("✅ Download complete!")
    print(f"📁 Location: {dataset.location}")
    print(f"📊 Ready to train with: {dataset.location}/data.yaml")
    print()

    return dataset.location


# Example: Popular traffic/vehicle datasets on Roboflow

def download_traffic_surveillance():
    """
    Example: Download traffic surveillance dataset
    """
    # Get your API key from: https://app.roboflow.com/settings/api
    API_KEY = "YOUR_API_KEY_HERE"

    # Example public datasets (you can search for more)

    # 1. Traffic Surveillance
    dataset_path = download_roboflow_dataset(
        api_key=API_KEY,
        workspace="traffic-monitoring",
        project="vehicle-detection-overhead",
        version=1
    )

    return dataset_path


def download_night_vision():
    """
    Example: Download night vision vehicle dataset
    """
    API_KEY = "YOUR_API_KEY_HERE"

    # 2. Night Vision Vehicles
    dataset_path = download_roboflow_dataset(
        api_key=API_KEY,
        workspace="night-vision",
        project="night-vehicle-detection",
        version=1
    )

    return dataset_path


def download_accident_detection():
    """
    Example: Download accident/crash detection dataset
    """
    API_KEY = "YOUR_API_KEY_HERE"

    # 3. Accident Detection
    dataset_path = download_roboflow_dataset(
        api_key=API_KEY,
        workspace="traffic-safety",
        project="accident-detection",
        version=1
    )

    return dataset_path


def main():
    """
    Guide to using Roboflow datasets
    """

    print("="*70)
    print("🚀 ROBOFLOW DATASET DOWNLOAD GUIDE")
    print("="*70)
    print()

    print("📋 STEP-BY-STEP GUIDE:")
    print()

    print("1️⃣  GET API KEY (FREE)")
    print("   → Go to: https://roboflow.com")
    print("   → Sign up (free account)")
    print("   → Go to: Settings → API")
    print("   → Copy your API key")
    print()

    print("2️⃣  SEARCH FOR DATASETS")
    print("   → Go to: https://universe.roboflow.com")
    print("   → Search for:")
    print("      • 'traffic surveillance'")
    print("      • 'vehicle overhead'")
    print("      • 'night vehicle detection'")
    print("      • 'accident detection'")
    print("      • 'crash detection'")
    print()

    print("3️⃣  DOWNLOAD DATASET")
    print("   → Click on dataset you want")
    print("   → Click 'Download'")
    print("   → Choose 'YOLOv8' format")
    print("   → Copy the download code")
    print()

    print("4️⃣  TRAIN IMMEDIATELY")
    print("   → No conversion needed!")
    print("   → Dataset already in YOLO format")
    print("   → Just run training command")
    print()

    print("="*70)
    print("💡 RECOMMENDED DATASETS FOR BUSAY:")
    print("="*70)
    print()

    print("For Vehicle Detection + Speed:")
    print("  • Search: 'traffic surveillance overhead'")
    print("  • Look for: 5,000+ images")
    print("  • Angle: Overhead/angled views")
    print()

    print("For Night Vision:")
    print("  • Search: 'night vehicle detection'")
    print("  • Look for: Infrared or low-light images")
    print("  • 2,000+ images recommended")
    print()

    print("For Crash Detection:")
    print("  • Search: 'accident detection' or 'crash detection'")
    print("  • Look for: Accident annotations")
    print("  • 1,000+ images minimum")
    print()

    print("="*70)
    print("📥 EXAMPLE DOWNLOAD CODE:")
    print("="*70)
    print()
    print("from roboflow import Roboflow")
    print("rf = Roboflow(api_key='YOUR_API_KEY')")
    print("project = rf.workspace('workspace-name').project('project-name')")
    print("dataset = project.version(1).download('yolov8')")
    print()
    print("# Train immediately!")
    print("from ultralytics import YOLO")
    print("model = YOLO('yolov8n.pt')")
    print("model.train(data=f'{dataset.location}/data.yaml', epochs=100)")
    print()

    print("="*70)
    print("⏱️  TRAINING TIME:")
    print("="*70)
    print()
    print("With Roboflow datasets (already formatted):")
    print("  • No conversion time needed! ✅")
    print("  • 5,000 images: ~3-4 hours training")
    print("  • 10,000 images: ~6-7 hours training")
    print()

    print("vs AI City Challenge:")
    print("  • Conversion: ~30 min ⚠️")
    print("  • Training: ~6-8 hours")
    print("  • Total: ~7-9 hours")
    print()

    print("💡 Roboflow saves you time and hassle!")
    print()


if __name__ == "__main__":
    main()
