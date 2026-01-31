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

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
python3 -c "import fastapi, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed. Please run:"
    echo "   pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies OK"
echo ""

# Create models directory if it doesn't exist
mkdir -p models
echo "📁 Models directory: ./models/"
echo ""

# Start the service
echo "🤖 Starting AI service on http://0.0.0.0:8000"
echo "   Press Ctrl+C to stop"
echo "================================================"
echo ""

python3 -m app.main
