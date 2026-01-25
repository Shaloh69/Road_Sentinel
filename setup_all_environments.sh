#!/bin/bash
#
# Setup All Virtual Environments for Road Sentinel
# Creates and configures separate environments for each script folder
#

set -e  # Exit on error

echo "========================================================================"
echo "🚀 Road Sentinel - Environment Setup"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Function to setup an environment
setup_environment() {
    local folder=$1
    local env_name=$2
    local description=$3

    echo "========================================================================"
    echo -e "${BLUE}Setting up: $description${NC}"
    echo "========================================================================"

    cd "$PROJECT_ROOT/scripts/$folder"

    # Check if environment already exists
    if [ -d "$env_name" ]; then
        echo -e "${YELLOW}⚠️  Environment '$env_name' already exists${NC}"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Removing old environment..."
            rm -rf "$env_name"
        else
            echo "⏭️  Skipping $description"
            echo ""
            return
        fi
    fi

    # Create virtual environment
    echo "📦 Creating virtual environment: $env_name"
    python3 -m venv "$env_name"

    # Activate environment
    echo "🔌 Activating environment..."
    source "$env_name/bin/activate"

    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip --quiet

    # Install dependencies
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt --quiet

    # Verify installation
    echo "✅ Verifying installation..."
    case $folder in
        "extract_frames")
            python -c "import cv2, numpy; print('  OpenCV:', cv2.__version__)"
            ;;
        "download")
            python -c "from ultralytics import YOLO; print('  Ultralytics: OK')"
            ;;
        "training")
            python -c "from ultralytics import YOLO; import torch, pandas; print('  All packages: OK')"
            python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
            ;;
    esac

    # Deactivate
    deactivate

    echo -e "${GREEN}✅ $description - Complete!${NC}"
    echo ""
}

# Setup each environment
echo "This script will create 3 virtual environments:"
echo "  1. Frame Extraction (venv_frames) - ~150MB"
echo "  2. Speed Detection (venv) - ~2-3GB"
echo "  3. Model Training (venv_training) - ~3GB"
echo ""
echo "Total disk space needed: ~5-6GB"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""

# Setup environments
setup_environment "extract_frames" "venv_frames" "Frame Extraction Environment"
setup_environment "download" "venv" "Speed Detection Environment"
setup_environment "training" "venv_training" "Model Training Environment"

# Final summary
echo "========================================================================"
echo -e "${GREEN}🎉 All Environments Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "📋 Environment Summary:"
echo ""
echo "1️⃣  Frame Extraction:"
echo "   Location: scripts/extract_frames/venv_frames"
echo "   Activate: cd scripts/extract_frames && source venv_frames/bin/activate"
echo ""
echo "2️⃣  Speed Detection:"
echo "   Location: scripts/download/venv"
echo "   Activate: cd scripts/download && source venv/bin/activate"
echo ""
echo "3️⃣  Model Training:"
echo "   Location: scripts/training/venv_training"
echo "   Activate: cd scripts/training && source venv_training/bin/activate"
echo ""
echo "========================================================================"
echo ""
echo "📚 Next Steps:"
echo "   1. See ENVIRONMENT_SETUP.md for usage guide"
echo "   2. Activate the environment you need"
echo "   3. Run the corresponding scripts"
echo ""
echo "💡 Quick Start:"
echo "   cd scripts/download"
echo "   source venv/bin/activate"
echo "   python auto_download_coco.py"
echo ""
echo "========================================================================"
