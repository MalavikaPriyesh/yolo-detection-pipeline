from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class BoundingBox(BaseModel):
    """Exact pixel coordinates of detected object."""
    x_min: float = Field(description="Left edge pixel coordinate")
    y_min: float = Field(description="Top edge pixel coordinate")
    x_max: float = Field(description="Right edge pixel coordinate")
    y_max: float = Field(description="Bottom edge pixel coordinate")
    width: float = Field(description="Width in pixels")
    height: float = Field(description="Height in pixels")

class DetectedObject(BaseModel):
    """A single object detected in the frame."""
    class_id: int = Field(description="YOLOv8 class ID (0-79)")
    class_name: str = Field(description="Object class name (e.g., 'person', 'car', 'dog')")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence (0-1)")
    bounding_box: BoundingBox = Field(description="Pixel coordinates of detection")
    area_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of frame occupied by object")

class FrameDetectionResult(BaseModel):
    """Complete detection results for a single frame."""
    frame_id: int = Field(description="Sequential frame number")
    timestamp: str = Field(description="ISO format timestamp")
    frame_width: int = Field(description="Video frame width in pixels")
    frame_height: int = Field(description="Video frame height in pixels")
    total_detections: int = Field(ge=0, description="Total objects detected in frame")
    detected_objects: List[DetectedObject] = Field(default_factory=list, description="Array of detected objects")
    detection_quality: Literal["low", "medium", "high"] = Field(description="Overall detection confidence assessment")

class VideoProcessingConfig(BaseModel):
    """Configuration for detection pipeline."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence to keep detections")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="Non-max suppression IOU threshold")
    model_name: Literal["yolov8n", "yolov8s", "yolov8m", "yolov8l"] = Field(default="yolov8m", description="YOLO model size")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu", description="Processing device")