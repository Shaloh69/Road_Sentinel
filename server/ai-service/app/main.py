from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
from pathlib import Path
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

# Root of the ai-service package (server/ai-service/) — used to resolve
# relative model paths regardless of the process's current working directory
# (e.g. `python -m app.main` from a different cwd, or a systemd WorkingDirectory
# that isn't server/ai-service/). Absolute paths in .env are passed through as-is.
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_path(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (AI_SERVICE_ROOT / path).resolve()
    return str(path)

# ── Local PC storage ───────────────────────────────────────────────────────────
# Files are saved here and served publicly via Cloudflare tunnel.
# Set STORAGE_BASE_URL to your Cloudflare tunnel URL, e.g.:
#   STORAGE_BASE_URL=https://your-pc.trycloudflare.com
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
(MEDIA_DIR / "incidents").mkdir(exist_ok=True)
(MEDIA_DIR / "recordings").mkdir(exist_ok=True)

STORAGE_BASE_URL = os.getenv("STORAGE_BASE_URL", "").rstrip("/")

# ── Initialize FastAPI app ─────────────────────────────────────────────────────
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

# Serve stored media files at /media/...
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# Initialize detectors (lazy loading)
traffic_detector: Optional[TrafficDetector] = None
incident_detector: Optional[IncidentDetector] = None


def get_traffic_detector() -> TrafficDetector:
    """Get or create traffic detector instance"""
    global traffic_detector
    if traffic_detector is None:
        model_path = _resolve_model_path(os.getenv('TRAFFIC_MODEL_PATH', './models/traffic.pt'))
        logger.info(f"Initializing traffic detector — model_path={model_path}")
        traffic_detector = TrafficDetector(
            model_path=model_path,
            device=os.getenv('DEVICE', 'cuda'),
            confidence=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        )
        logger.info(
            f"Traffic detector ready — custom_model={traffic_detector.is_custom_model} "
            f"(False means it silently fell back to stock yolov8n.pt — check model_path above)"
        )
    return traffic_detector


def get_incident_detector() -> IncidentDetector:
    """Get or create incident detector instance"""
    global incident_detector
    if incident_detector is None:
        model_path = _resolve_model_path(os.getenv('INCIDENT_MODEL_PATH', './models/incident.pt'))
        logger.info(f"Initializing incident detector — model_path={model_path}")
        incident_detector = IncidentDetector(
            model_path=model_path,
            device=os.getenv('DEVICE', 'cuda'),
            confidence=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        )
        if incident_detector.model is None:
            logger.warning(
                "No incident model loaded — running heuristic fallback "
                "(brightness-variance placeholder, not a real detector). "
                "Train training/train.py --dataset accident to replace this."
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
    confidence_threshold: Optional[float] = Form(None),
    pixels_per_meter: float = Form(0.0),
    speed_limit: float = Form(0.0),
    homography_points: Optional[str] = Form(None),
):
    """
    Detect both traffic and incidents in an image.
    When homography_points is provided (JSON string: {"image_points":[[x,y]x4],
    "real_points":[[x,y]x4]}), perspective-corrected speed is used — accurate
    regardless of where in frame the vehicle is tracked. Otherwise, when
    pixels_per_meter > 0, the simpler raw-pixel-distance estimate is used.
    When speed_limit > 0, speeding incidents are generated automatically.
    """
    try:
        start_time = time.time()

        image_bytes = await image.read()

        traffic_det  = get_traffic_detector()
        incident_det = get_incident_detector()

        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

        homography = None
        if homography_points:
            try:
                homography = json.loads(homography_points)
            except (TypeError, ValueError) as exc:
                logger.warning(f"Ignoring malformed homography_points for {camera_id}: {exc}")

        # Traffic detections — with optional speed estimation
        traffic_results = traffic_det.detect(
            image_bytes, conf,
            camera_id=camera_id,
            pixels_per_meter=pixels_per_meter,
            homography_points=homography,
        )

        # Incident detections from incident model
        incident_results = incident_det.detect(image_bytes, conf)

        # Auto-generate speeding incidents from speed estimates
        if speed_limit > 0:
            for det in traffic_results:
                spd = det.get("speed")
                if spd and spd > speed_limit:
                    over  = spd - speed_limit
                    sev   = "critical" if over > 30 else ("high" if over > 15 else "medium")
                    cls   = det.get("class", "Vehicle").capitalize()
                    incident_results.append({
                        "type":        "speeding",
                        "severity":    sev,
                        "confidence":  det.get("confidence", 0.8),
                        "description": f"{cls} at {spd:.0f} km/h (limit {speed_limit:.0f} km/h)",
                        "speed":       spd,
                        "is_heuristic": False,  # derived from real tracked speed, not a guess
                    })

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Detection for {camera_id} in {processing_time:.2f}ms — "
                    f"{len(traffic_results)} vehicles, {len(incident_results)} incidents")

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
        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

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
        conf = confidence_threshold if confidence_threshold else float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

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


@app.post("/api/storage/upload")
async def storage_upload(
    file: UploadFile = File(...),
    path: str = Form(...),
):
    """
    Save a file to local PC storage and return its public URL.
    path  — relative path inside the media dir, e.g. 'incidents/cam_a/2026-01-01_abc.jpg'
    Returns { url, path }
    """
    # Sanitise: prevent directory traversal
    safe_path = Path(path).as_posix().lstrip("/")
    if ".." in safe_path:
        raise HTTPException(status_code=400, detail="Invalid path")

    dest = MEDIA_DIR / safe_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    dest.write_bytes(content)

    # Build public URL from Cloudflare tunnel base or fall back to localhost
    base = STORAGE_BASE_URL or f"http://localhost:{os.getenv('PORT', '8000')}"
    public_url = f"{base}/media/{safe_path}"

    logger.info(f"Storage upload: {safe_path}  ({len(content):,} bytes) → {public_url}")
    return JSONResponse({"success": True, "url": public_url, "path": safe_path})


@app.delete("/api/storage/delete")
async def storage_delete(path: str = Query(...)):
    """
    Delete a file from local PC storage.
    path — relative path inside the media dir
    """
    safe_path = Path(path).as_posix().lstrip("/")
    if ".." in safe_path:
        raise HTTPException(status_code=400, detail="Invalid path")

    dest = MEDIA_DIR / safe_path
    if dest.exists() and dest.is_file():
        dest.unlink()
        logger.info(f"Storage delete: {safe_path}")
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "error": "File not found"}, status_code=404)


@app.get("/api/storage/list")
async def storage_list(prefix: str = ""):
    """List stored files under an optional prefix."""
    base = MEDIA_DIR / prefix if prefix else MEDIA_DIR
    if not base.exists():
        return JSONResponse({"files": []})
    files = [
        str(p.relative_to(MEDIA_DIR)).replace("\\", "/")
        for p in base.rglob("*")
        if p.is_file()
    ]
    return JSONResponse({"files": sorted(files)})


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
            "confidence_threshold": float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
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
