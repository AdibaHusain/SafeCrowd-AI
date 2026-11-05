from typing import List, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class InferenceResult(BaseModel):
    """
    Common response format for ML models.

    CSRNet currently uses:
        - model_name
        - estimated_count
        - density_map_available
        - density_map_base64
        - inference_time_ms

    YOLO/P2PNet can use the same schema later.
    """

    model_name: str
    estimated_count: float

    density_map_available: bool = False
    density_map_base64: Optional[str] = None

    boxes: Optional[List[BoundingBox]] = None

    inference_time_ms: float