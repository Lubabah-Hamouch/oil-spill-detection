import os
import time
import logging
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import sys
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classification import build_classification_model
from src.models.segmentation import build_segmentation_model

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["No Spill", "Spill"]

def get_model_directory():
    # An explicit MODEL_DIR environment variable wins
    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        return env_dir
    # Otherwise attempt to locate models relative to this script
    script_dir = Path(__file__).parent.parent.parent.absolute()
    possible_paths = [
        script_dir / "models",
        Path.cwd() / "models",
    ]
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return str(path)
    return str(Path.cwd() / "models")

MODEL_DIR = get_model_directory()

def _load_checkpoint_model(build, model_path):
    """Build a model and load its checkpoint without ever holding two copies of the weights.

    The model is built on the meta device (no memory for a throwaway random init) and the checkpoint
    is memory-mapped, so the parameters are assigned straight from the mapped file rather than
    copied out of a temporary state_dict.
    """
    with torch.device("meta"):
        model = build()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback for old models
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True, assign=True)
    # strict=True does not cover non-persistent buffers; anything left on meta was never loaded.
    missing = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(f"Checkpoint {model_path} did not provide: {missing}")
    return model.to(device).eval()

def load_production_model():
    model_path = os.path.join(MODEL_DIR, "classification_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at {model_path}")

    return _load_checkpoint_model(lambda: build_classification_model(pretrained=False), model_path)

# Thread-safe lazy model loading
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_production_model()
    return _model

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image_classification(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return _transform(image).unsqueeze(0).to(device)

# Inference Configuration
# Probability thresholds (P = predicted spill probability).
#   THRESHOLD_SPILL               - decision cutoff: P >= this means binary_label=1, a segmentation
#                                   mask is produced, and the result is labelled a spill.
#   THRESHOLD_LOW                 - lower bound of "LOW_CONFIDENCE_SPILL". Kept equal to
#                                   THRESHOLD_SPILL so a spill label always comes with binary_label=1.
#   THRESHOLD_HIGH_CONFIDENCE     - from here up: "HIGH_CONFIDENCE_SPILL".
THRESHOLD_SPILL = 0.25
THRESHOLD_LOW = THRESHOLD_SPILL
THRESHOLD_HIGH_CONFIDENCE = 0.5
# Pixel probability above which the segmentation model marks a pixel as spill.
SEGMENTATION_PIXEL_THRESHOLD = 0.5
# Overlay mask colour (RGB): amber #F59E0B, matching the frontend's spill colour.
OVERLAY_COLOR_RGB = (245, 158, 11)
OVERLAY_ALPHA = 0.4
# The overlay is original + OVERLAY_ALPHA * colour on spill pixels (saturating). Precomputed as an
# integer offset so it can be added in place; matches cv2.addWeighted's rounding as long as no
# channel lands on exactly .5 (true for this colour: 98, 63.2, 4.4).
OVERLAY_OFFSET_RGB = tuple(round(OVERLAY_ALPHA * c) for c in OVERLAY_COLOR_RGB)
PREDICTION_MODE = "dual"  # Options: "binary", "dual"

def predict(image, mode=PREDICTION_MODE):
    """
    Refined, hardened inference function.
    Handles multiple output types and ensures stable probability computation.
    """
    model = get_model()
    model.eval()  # Safety: ensure evaluation mode
    
    image_tensor = preprocess_image_classification(image)
    
    with torch.no_grad():  # Safety: disable gradient computation
        outputs = model(image_tensor)
        
        # 1. CORRECT PROBABILITY COMPUTATION (Detect shape)
        if outputs.shape[1] == 2:
            # CrossEntropy setup [1, 2]
            prob_spill = torch.softmax(outputs, dim=1)[0, 1].item()
        else:
            # BCE setup [1, 1] or [1]
            prob_spill = torch.sigmoid(outputs).view(-1)[0].item()
            
    # 2. PROBABILITY SANITY CLAMP
    prob_spill = float(prob_spill)
    prob_spill = max(0.0, min(1.0, prob_spill))
    
    # 3. SINGLE THRESHOLD LOGIC (binary)
    binary_label = 1 if prob_spill >= THRESHOLD_SPILL else 0
    
    # 4. IMPROVED CONFIDENCE LABEL NAMING
    if prob_spill >= THRESHOLD_HIGH_CONFIDENCE:
        confidence_label = "HIGH_CONFIDENCE_SPILL"
    elif prob_spill >= THRESHOLD_LOW:
        confidence_label = "LOW_CONFIDENCE_SPILL"
    else:
        confidence_label = "NO_SPILL"
        
    if mode == "dual":
        return {
            "probability": round(prob_spill, 4),
            "binary_label": binary_label,
            "confidence_label": confidence_label
        }
    else:
        # Backward compatibility for binary mode
        label = CLASS_NAMES[binary_label]
        return label, round(prob_spill * 100, 2)

SEGMENTATION_MODEL_NAME = "UNet_ResNet34"
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "segmentation_best.pth")

def load_segmentation_models():
    """Load segmentation checkpoints. Raises if the checkpoint exists but cannot be loaded."""
    models = {}

    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        logger.warning("Segmentation checkpoint not found at %s; segmentation disabled.", SEGMENTATION_MODEL_PATH)
        return models

    # No ImageNet encoder weights: the trained checkpoint replaces every encoder weight anyway.
    models[SEGMENTATION_MODEL_NAME] = _load_checkpoint_model(
        lambda: build_segmentation_model(encoder_weights=None), SEGMENTATION_MODEL_PATH
    )
    return models

# Thread-safe lazy loading (loaded on first call, not at import or startup: only spill images need it)
_segmentation_models = None
_segmentation_lock = threading.Lock()

def get_segmentation_models():
    global _segmentation_models
    if _segmentation_models is None:
        with _segmentation_lock:
            if _segmentation_models is None:
                _segmentation_models = load_segmentation_models()
    return _segmentation_models

def unload_segmentation_models():
    """Drop the segmentation model; the next get_segmentation_models() call reloads it.

    The caller must make sure no other thread is still running segmentation.
    """
    global _segmentation_models
    with _segmentation_lock:
        _segmentation_models = None

def segmentation_available():
    """True if a segmentation checkpoint exists, without loading it."""
    return os.path.exists(SEGMENTATION_MODEL_PATH)

def models_loaded():
    """True once the classifier is loaded. The segmentation model loads on demand (first spill image)."""
    return _model is not None

def preprocess_image_segmentation(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

def segment_image_with_mask(image, model_name, spill_detected=True):
    """Segment a spill and return (RGB overlay uint8, status string, mask or None).

    mask is an (H, W) uint8 array at the input resolution (1 = spill), or None unless
    status == "Success".

    The input is treated as RGB throughout (PIL / Gradio convention), so the overlay is
    returned as RGB with no channel reordering. If spill_detected is False the original
    image is returned unchanged with a "No spill detected" status.
    """
    # Prepare original image as RGB uint8. np.array always copies, so the overlay can later be
    # drawn into `original` in place without touching the caller's image.
    if isinstance(image, Image.Image):
        original = np.array(image if image.mode == "RGB" else image.convert("RGB"))
    else:
        original = np.array(image)

    # Ensure (H, W, 3) uint8
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
    original = original.astype(np.uint8, copy=False)

    if not spill_detected:
        return original, "No spill detected", None

    try:
        segmentation_models = get_segmentation_models()
        if model_name not in segmentation_models:
            return original, "Model not available", None

        model = segmentation_models[model_name]
        original_size = original.shape[:2]

        # Resize input BEFORE model
        resized = cv2.resize(original, (224, 224))

        image_tensor = preprocess_image_segmentation(resized)
        image_tensor = image_tensor.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs)
            pred_mask = (probs > SEGMENTATION_PIXEL_THRESHOLD).int().squeeze().cpu().numpy()
        logger.debug("Segmentation inference time: %.4fs", time.perf_counter() - start_time)

        pred_mask = cv2.resize(pred_mask.astype(np.uint8),
                              (original_size[1], original_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Overlay drawn in place: original is RGB, so the offset is RGB too and the result stays RGB.
        # Same pixels as cv2.addWeighted(original, 1.0, colour_mask, OVERLAY_ALPHA, 0), without
        # allocating two more full-resolution images.
        cv2.add(original, OVERLAY_OFFSET_RGB + (0,), dst=original, mask=pred_mask)

        return original, "Success", pred_mask

    except Exception as e:
        logger.exception("Segmentation failed")
        return original, f"Error: {e}", None


def segment_image(image, model_name, spill_detected=True):
    """Segment a spill and return (RGB overlay uint8, status string)."""
    overlay, status, _ = segment_image_with_mask(image, model_name, spill_detected)
    return overlay, status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Predict oil spill in image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--mode", type=str, default="dual", choices=["binary", "dual"], help="Inference mode.")
    parser.add_argument("--save-overlay", type=str, default=None, help="If a spill is predicted, save the segmentation overlay (RGB) to this path.")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
    else:
        try:
            img = Image.open(image_path)
            result = predict(img, mode=args.mode)
            print("\nPrediction Result:")
            print(json.dumps(result, indent=4))

            if args.save_overlay:
                spill = result["binary_label"] == 1 if args.mode == "dual" else result[0] == CLASS_NAMES[1]
                models = get_segmentation_models()
                model_name = next(iter(models), None)
                overlay, status = segment_image(img, model_name, spill_detected=spill)
                print(f"Segmentation status: {status}")
                if status == "Success":
                    cv2.imwrite(args.save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # cv2 writes BGR
                    print(f"Overlay saved to {args.save_overlay}")
        except Exception as e:
            print(f"Error processing image: {e}")
