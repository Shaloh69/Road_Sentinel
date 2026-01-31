from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
import logging
from typing import Optional
import time

from app.models.traffic_detector import TrafficDetector
from app.models.incident_detector import IncidentDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Road Sentinel AI Service",
    description="YOLOv8-based traffic and incident detection service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors (lazy loading)
traffic_detector: Optional[TrafficDetector] = None
incident_detector: Optional[IncidentDetector] = None


def get_traffic_detector() -> TrafficDetector:
    """Get or create traffic detector instance"""
    global traffic_detector
    if traffic_detector is None:
        logger.info("Initializing traffic detector...")
        traffic_detector = TrafficDetector(
            model_path=os.getenv('TRAFFIC_MODEL_PATH', './models/traffic.pt'),
            device=os.getenv('DEVICE', 'cuda'),
            confidence=float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))
        )
    return traffic_detector


def get_incident_detector() -> IncidentDetector:
    """Get or create incident detector instance"""
    global incident_detector
    if incident_detector is None:
        logger.info("Initializing incident detector...")
        incident_detector = IncidentDetector(
            model_path=os.getenv('INCIDENT_MODEL_PATH', './models/incident.pt'),
            device=os.getenv('DEVICE', 'cuda'),
            confidence=float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))
        )
    return incident_detector


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Road Sentinel AI Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.post("/api/detect")
async def detect_all(
    image: UploadFile = File(...),
    camera_id: str = Form(...),
    confidence_threshold: Optional[float] = Form(None)
):
    """
    Detect both traffic and incidents in an image
    """
    try:
        start_time = time.time()

        # Read image
        image_bytes = await image.read()

        # Get detectors
        traffic_det = get_traffic_detector()
        incident_det = get_incident_detector()

        # Override confidence if provided
        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))

        # Run detections
        traffic_results = traffic_det.detect(image_bytes, conf)
        incident_results = incident_det.detect(image_bytes, conf)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Detection completed for {camera_id} in {processing_time:.2f}ms")

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "detections": traffic_results,
            "incidents": incident_results,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/traffic")
async def detect_traffic(
    image: UploadFile = File(...),
    camera_id: str = Form(...),
    confidence_threshold: Optional[float] = Form(None)
):
    """
    Detect traffic (vehicles) only
    """
    try:
        start_time = time.time()

        # Read image
        image_bytes = await image.read()

        # Get detector
        detector = get_traffic_detector()

        # Override confidence if provided
        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))

        # Run detection
        results = detector.detect(image_bytes, conf)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Traffic detection for {camera_id}: {len(results)} vehicles detected")

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "detections": results,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Traffic detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/incidents")
async def detect_incidents(
    image: UploadFile = File(...),
    camera_id: str = Form(...),
    confidence_threshold: Optional[float] = Form(None)
):
    """
    Detect incidents only
    """
    try:
        start_time = time.time()

        # Read image
        image_bytes = await image.read()

        # Get detector
        detector = get_incident_detector()

        # Override confidence if provided
        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))

        # Run detection
        results = detector.detect(image_bytes, conf)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Incident detection for {camera_id}: {len(results)} incidents detected")

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "incidents": results,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Incident detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get AI service statistics"""
    return {
        "traffic_model": {
            "loaded": traffic_detector is not None,
            "path": os.getenv('TRAFFIC_MODEL_PATH', './models/traffic.pt'),
            "device": os.getenv('DEVICE', 'cuda')
        },
        "incident_model": {
            "loaded": incident_detector is not None,
            "path": os.getenv('INCIDENT_MODEL_PATH', './models/incident.pt'),
            "device": os.getenv('DEVICE', 'cuda')
        },
        "configuration": {
            "confidence_threshold": float(os.getenv('CONFIDENCE_THRESHOLD', '0.75')),
            "iou_threshold": float(os.getenv('IOU_THRESHOLD', '0.45'))
        }
    }


if __name__ == "__main__":
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    workers = int(os.getenv('WORKERS', '1'))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=True
    )
