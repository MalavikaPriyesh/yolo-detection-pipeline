import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List
from ultralytics import YOLO
from schemas import FrameDetectionResult, DetectedObject, BoundingBox, VideoProcessingConfig

class YOLODetector:
    """Production-grade YOLO inference engine."""
    
    # Standard COCO class names (YOLOv8 uses 80 classes)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.model = YOLO(f"{config.model_name}.pt")
        
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FrameDetectionResult]:
        """
        Run YOLO inference on a single frame.
        Returns annotated frame and structured detection data.
        """
        height, width = frame.shape[:2]
        
        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )
        
        detected_objects = []
        annotated_frame = frame.copy()
        
        # Extract detections
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"Class_{class_id}"
                
                # Calculate metrics
                box_width = x2 - x1
                box_height = y2 - y1
                area_percentage = (box_width * box_height) / (width * height) * 100
                
                # Create structured detection
                detection = DetectedObject(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x_min=x1,
                        y_min=y1,
                        x_max=x2,
                        y_max=y2,
                        width=box_width,
                        height=box_height
                    ),
                    area_percentage=area_percentage
                )
                detected_objects.append(detection)
                
                # Draw on frame
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Determine overall quality
        if len(detected_objects) == 0:
            quality = "low"
        elif len(detected_objects) > 5:
            avg_conf = np.mean([d.confidence for d in detected_objects])
            quality = "high" if avg_conf > 0.75 else "medium"
        else:
            quality = "medium"
        
        # Create result schema
        result = FrameDetectionResult(
            frame_id=0,
            timestamp=datetime.now().isoformat(),
            frame_width=width,
            frame_height=height,
            total_detections=len(detected_objects),
            detected_objects=detected_objects,
            detection_quality=quality
        )
        
        return annotated_frame, result
    
    def process_video(self, video_path: str, max_frames: int = None):
        """Process video file and yield annotated frames with detections."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            annotated_frame, detection_result = self.detect_frame(frame)
            detection_result.frame_id = frame_count
            
            yield annotated_frame, detection_result
            frame_count += 1
        
        cap.release()