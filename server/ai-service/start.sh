#!/bin/bash
# Start Road Sentinel AI Service

echo "🚀 Starting Road Sentinel AI Service..."
echo "================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo "   Edit .env if you need to change settings"
    echo ""
fi

# Load env vars for use in this script
set -a
source .env
set +a

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
python3 -c "import fastapi, ultralytics, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed. Run:"
    echo "   pip install -r requirements.txt        # GPU"
    echo "   pip install -r requirements-cpu.txt    # CPU only"
    exit 1
fi
echo "✅ Dependencies OK"
echo ""

# Create models directory if it doesn't exist
mkdir -p models
echo "📁 Models directory: ./models/"
echo ""

# Resolve settings with defaults
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# GPU: keep workers=1 so only one process owns CUDA context
WORKERS="${WORKERS:-1}"

echo "🤖 Starting AI service on http://${HOST}:${PORT}  (workers=${WORKERS})"
echo "   Device: ${DEVICE:-cuda}"
echo "   Press Ctrl+C to stop"
echo "================================================"
echo ""

exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"
