from typing import Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    probability: float = Field(..., description="Predicted spill probability (0-1).")
    binary_label: int = Field(..., description="1 = spill, 0 = no spill.")
    confidence_label: str = Field(..., description="NO_SPILL, LOW_CONFIDENCE_SPILL or HIGH_CONFIDENCE_SPILL.")
    spill_detected: bool = Field(..., description="True when binary_label == 1.")
    segmentation_status: str = Field(..., description="'Success', 'No spill detected' or 'Model not available'.")
    overlay_png_base64: Optional[str] = Field(
        None, description="Base64-encoded PNG of the input with the spill mask in red. Null if segmentation did not run."
    )
    area_percentage: Optional[float] = Field(
        None, description="Percent of image pixels flagged as spill. Null if segmentation did not run."
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class ErrorResponse(BaseModel):
    detail: str
