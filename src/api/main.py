import base64
import io
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
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
# Uploads whose longer side exceeds this are downscaled (aspect ratio kept) right after decoding.
# Both models run at 224x224, so this only bounds the full-resolution buffers (decode, overlay, PNG).
MAX_IMAGE_SIDE = 2000
# Comma-separated list of allowed CORS origins; "*" (any origin) when unset.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
# "1" frees the segmentation model after every request that used it (lower idle memory, slower spill
# requests). Off by default: peak memory is the same either way, so keeping it loaded is free.
UNLOAD_SEGMENTATION = os.environ.get("SEGMENTATION_UNLOAD_AFTER_USE", "0") == "1"

# Single shared CPU: extra intra-/inter-op thread pools only add memory, not speed.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# One inference at a time. FastAPI runs sync endpoints on a thread pool, so without this, concurrent
# uploads would each hold their own full-resolution image buffers and activations at once.
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only the small classifier is loaded at startup; the segmentation model loads on the first spill image.
    inference.get_model()
    yield


app = FastAPI(title=API_NAME, version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
        target = _capped_size(image.size)
        if target != image.size:
            # JPEG only: decode at a reduced 1/2, 1/4 or 1/8 scale that is still >= target, so a huge
            # photo is never held at full size. No-op for other formats.
            image.draft(None, target)
        image.load()  # force full decode so truncated/corrupt files fail here
        # convert() always copies, even RGB -> RGB; skip it for the common case.
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        return image
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")


def _capped_size(size: tuple[int, int]) -> tuple[int, int]:
    """(width, height) scaled so the longer side is at most MAX_IMAGE_SIDE, aspect ratio preserved."""
    width, height = size
    longest = max(width, height)
    if longest <= MAX_IMAGE_SIDE:
        return size
    scale = MAX_IMAGE_SIDE / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


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

    with _inference_lock:
        # Decoding is inside the lock too: the decoded image is the largest per-request buffer.
        image = _read_image(data)

        try:
            result = inference.predict(image, mode="dual")
            spill_detected = result["binary_label"] == 1

            overlay_b64 = None
            area_percentage = None
            seg_status = "No spill detected"

            if spill_detected:
                if not inference.segmentation_available():
                    seg_status = "Model not available"
                else:
                    try:
                        overlay, seg_status, mask = inference.segment_image_with_mask(
                            image, inference.SEGMENTATION_MODEL_NAME
                        )
                    finally:
                        if UNLOAD_SEGMENTATION:
                            inference.unload_segmentation_models()
                    if seg_status.startswith("Error"):
                        raise RuntimeError(f"Segmentation failed: {seg_status}")
                    if mask is not None:
                        area_percentage = round(float(mask.mean()) * 100, 2)
                        # cv2 encodes BGR; overlay is RGB (converted in place, it isn't used afterwards)
                        ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR, dst=overlay))
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
