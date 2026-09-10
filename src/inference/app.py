import os
import sys
from pathlib import Path
import gradio as gr
import numpy as np
import cv2
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import predict, segment_image, segmentation_models

def create_overlay(image, mask, alpha=0.4):
    """Create a red overlay on the original image where the mask is 1."""
    img_np = np.array(image).copy()
    
    # Ensure image is RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    mask_rgba = np.zeros_like(img_np, dtype=np.uint8)
    mask_rgba[mask == 1] = [255, 0, 0] # Red
    
    # Blending
    overlay = cv2.addWeighted(img_np, 1.0, mask_rgba, alpha, 0)
    return overlay

def process_pipeline(image, view_mode):
    if image is None:
        return None, "### ⚠️ No image uploaded\nPlease upload an image to begin.", gr.update(visible=False)
    
    try:
        # 1. Classification
        results = predict(image, mode="dual")
        prob = results["probability"]
        label = results["binary_label"]
        conf_label = results["confidence_label"]
        
        # Mapping & Indicators
        mapping = {
            "HIGH_CONFIDENCE_SPILL": ("🔴", "Strong visual evidence of oil spill", "Recommended: Immediate attention required"),
            "LOW_CONFIDENCE_SPILL": ("🟠", "Possible spill — requires verification", "Recommended: Manual verification"),
            "NO_SPILL": ("🟢", "No spill detected", "No action needed")
        }
        indicator, explanation, recommendation = mapping.get(conf_label, ("", "", ""))
        
        # UI Strings - differentiate based on confidence level
        if conf_label == "HIGH_CONFIDENCE_SPILL":
            status_text = "⚠️ Oil Spill Detected"
        elif conf_label == "LOW_CONFIDENCE_SPILL":
            status_text = "⚠️ Possible Oil Spill (Review Needed)"
        else:
            status_text = "✅ No Oil Spill Detected"
        
        # Format results for Markdown display
        results_html = f"""
### 🧠 Classification Result

**Status:** {status_text}

**Probability:** {prob:.4f}

**Confidence:** {indicator} {conf_label}

**Details:** {explanation}

**Action:** {recommendation}
"""
        
        # 2. Segmentation - ONLY run on HIGH confidence detections
        if conf_label == "HIGH_CONFIDENCE_SPILL" and segmentation_models:
            model_name = list(segmentation_models.keys())[0]
            output_img, status = segment_image(image, model_name)
            return output_img, results_html, gr.update(visible=True)
        
        # Default return for no spill or failed segmentation
        return None, results_html, gr.update(visible=False)
            
    except Exception as e:
        error_html = f"### ❌ Error processing image\n\nRuntime error: {str(e)}"
        return None, error_html, gr.update(visible=False)

def create_app():
    with gr.Blocks(title="Oil Spill Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛢️ Oil Spill Detection & Analysis")
        gr.Markdown("Decision-support system for marine environment monitoring.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Image Input")
                input_img = gr.Image(type="pil", label="Marine Satellite Patch")
                
                with gr.Row():
                    view_mode = gr.Radio(
                        choices=["Overlay View", "Mask Only"],
                        value="Overlay View",
                        label="Segmentation Visualization"
                    )
                
                # Dynamic Examples Search
                example_list = []
                test_dir = PROJECT_ROOT / "data" / "splits" / "classification" / "test"
                if test_dir.exists():
                    spill_samples = list((test_dir / "1").glob("*.png"))[:1]
                    no_spill_samples = list((test_dir / "0").glob("*.png"))[:1]
                    example_list = [str(p) for p in spill_samples + no_spill_samples]
                
                if example_list:
                    gr.Examples(examples=example_list, inputs=input_img, label="Example Test Patches")
                
                btn = gr.Button("Run Detection Pipeline", variant="primary")
            
            with gr.Column(scale=1):
                out_results = gr.Markdown("### 📊 Analysis Output\nPlease run the analysis to see results.")
                
                with gr.Column(visible=False) as seg_section:
                    gr.Markdown("### 🎯 Segmentation Analysis")
                    out_mask = gr.Image(label="Spill Localization", interactive=False)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("© 2024 Oil Spill Detection Project | Final Decision Support System v1.1")

        # Logic
        btn.click(
            fn=process_pipeline,
            inputs=[input_img, view_mode],
            outputs=[out_mask, out_results, seg_section]
        )
        
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7861, share=False)