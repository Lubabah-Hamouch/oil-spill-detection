import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["No Spill", "Spill"]

def get_model_directory():
    # Attempt to locate models relative to this script
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

def load_production_model():
    model_path = os.path.join(MODEL_DIR, "classification_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at {model_path}")
        
    model = build_classification_model(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        # Fallback for old models
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model

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
DEFAULT_THRESHOLD = 0.25
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
    binary_label = 1 if prob_spill >= DEFAULT_THRESHOLD else 0
    
    # 4. IMPROVED CONFIDENCE LABEL NAMING
    if prob_spill >= 0.5:
        confidence_label = "HIGH_CONFIDENCE_SPILL"
    elif prob_spill >= 0.2:
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

def load_segmentation_models():
    models = {}
    
    # New single robust segmentation model
    model_path = os.path.join(MODEL_DIR, "segmentation_best.pth")
    if os.path.exists(model_path):
        try:
            model = build_segmentation_model().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
            model.eval()
            models["UNet_ResNet34"] = model
        except Exception as e:
            print(f"Warning: Failed to load segmentation model: {e}")
            
    return models

segmentation_models = load_segmentation_models()

def preprocess_image_segmentation(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

def segment_image(image, model_name):
    print("SEGMENT START")
    
    # Prepare original image as RGB uint8
    if isinstance(image, Image.Image):
        original = np.array(image)
    else:
        original = image.copy()
    
    # Ensure (H, W, 3) uint8
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
    original = original.astype(np.uint8)
    
    try:
        if model_name not in segmentation_models:
            print("SEGMENT DONE")
            return original, "Model not available"
        
        model = segmentation_models[model_name]
        original_size = original.shape[:2]
        
        # Resize input BEFORE model
        resized = cv2.resize(original, (224, 224))
        
        image_tensor = preprocess_image_segmentation(resized)
        image_tensor = image_tensor.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs)
            pred_mask = (probs > 0.5).int().squeeze().cpu().numpy()
        inference_time = time.time() - start_time
        print(f"Segmentation inference time: {inference_time:.4f}s")
        
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        mask_rgba = np.zeros_like(original, dtype=np.uint8)
        mask_rgba[pred_mask == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(original, 1.0, mask_rgba, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        print("SEGMENT DONE")
        return overlay.astype(np.uint8), "Success"
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        print("SEGMENT DONE")
        return original, f"Error: {e}"
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Predict oil spill in image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--mode", type=str, default="dual", choices=["binary", "dual"], help="Inference mode.")
    
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
        except Exception as e:
            print(f"Error processing image: {e}")
