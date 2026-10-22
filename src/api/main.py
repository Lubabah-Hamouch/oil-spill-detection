import base64
import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import ErrorResponse, HealthResponse, PredictResponse, RootResponse
from src.inference import predict as inference

logger = logging.getLogger(__name__)

API_NAME = "Oil Spill Detection API"
API_VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at startup so the first request isn't slow.
    inference.get_model()
    inference.get_segmentation_models()
    yield


app = FastAPI(title=API_NAME, version=API_VERSION, lifespan=lifespan)

# TODO: restrict origins before any non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Safety net: anything that slips past the endpoint's own handling still returns JSON, never a trace.
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


def _read_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force full decode so truncated/corrupt files fail here
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")


@app.get("/", response_model=RootResponse)
def root():
    return RootResponse(name=API_NAME, version=API_VERSION, docs="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models_loaded=inference.models_loaded())


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(file: UploadFile = File(..., description="Image file (PNG, JPEG, ...).")):
    # Sync endpoint: FastAPI runs it in a worker thread, so inference doesn't block the event loop.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    image = _read_image(data)

    try:
        result = inference.predict(image, mode="dual")
        spill_detected = result["binary_label"] == 1

        overlay_b64 = None
        area_percentage = None
        seg_status = "No spill detected"

        if spill_detected:
            models = inference.get_segmentation_models()
            if not models:
                seg_status = "Model not available"
            else:
                overlay, seg_status, mask = inference.segment_image_with_mask(image, next(iter(models)))
                if seg_status.startswith("Error"):
                    raise RuntimeError(f"Segmentation failed: {seg_status}")
                if mask is not None:
                    area_percentage = round(float(mask.mean()) * 100, 2)
                    # cv2 encodes BGR; overlay is RGB
                    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise RuntimeError("Failed to encode overlay as PNG.")
                    overlay_b64 = base64.b64encode(png.tobytes()).decode("ascii")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs for details.")

    return PredictResponse(
        probability=result["probability"],
        binary_label=result["binary_label"],
        confidence_label=result["confidence_label"],
        spill_detected=spill_detected,
        segmentation_status=seg_status,
        overlay_png_base64=overlay_b64,
        area_percentage=area_percentage,
    )
