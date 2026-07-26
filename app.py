import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import json
from detector import YOLODetector
from schemas import VideoProcessingConfig

st.set_page_config(page_title="YOLO Object Detection Pipeline", layout="wide")

st.title("Real-Time Object Detection Pipeline")
st.markdown("Production-grade YOLOv8 inference with structured detection outputs for robotics and autonomous systems.")

# Sidebar configuration
with st.sidebar:
    st.header("Detection Configuration")
    
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou = st.slider("IOU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    model_size = st.selectbox("YOLO Model Size", ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                              help="Smaller = faster, Larger = more accurate")
    
    config = VideoProcessingConfig(
        confidence_threshold=confidence,
        iou_threshold=iou,
        model_name=model_size,
        device="cpu"
    )

# Initialize detector
if "detector" not in st.session_state:
    with st.spinner("Loading YOLOv8 model..."):
        st.session_state.detector = YOLODetector(config)
else:
    st.session_state.detector = YOLODetector(config)

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Source")
    input_type = st.radio("Select input type:", ["Webcam (Screenshot)", "Upload Image", "Upload Video"])

with col2:
    st.subheader("Detection Results")

# Process based on input type
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner("Running detection..."):
            annotated_frame, detection_result = st.session_state.detector.detect_frame(image_cv)
        
        col1.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Annotated Detection")
        
        col2.metric("Total Detections", detection_result.total_detections)
        col2.metric("Detection Quality", detection_result.detection_quality)
        
        # Display detections table
        st.subheader("Detected Objects")
        for i, obj in enumerate(detection_result.detected_objects):
            st.write(f"**Object {i+1}:** {obj.class_name} (Confidence: {obj.confidence:.2%})")
            st.write(f"  Bounding Box: ({obj.bounding_box.x_min}, {obj.bounding_box.y_min}) to ({obj.bounding_box.x_max}, {obj.bounding_box.y_max})")
            st.write(f"  Area: {obj.area_percentage:.2f}% of frame")
        
        # Export JSON
        if st.button("Export Detection JSON"):
            json_output = detection_result.model_dump_json(indent=2)
            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="detection_result.json",
                mime="application/json"
            )

elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        max_frames = st.slider("Max frames to process", 1, 300, 30)
        
        if st.button("Process Video"):
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            total_frames = 0
            total_detections = 0
            avg_confidence = []
            
            for frame_idx, (annotated_frame, detection_result) in enumerate(
                st.session_state.detector.process_video(tmp_path, max_frames)
            ):
                if frame_idx >= max_frames:
                    break
                
                total_frames += 1
                total_detections += detection_result.total_detections
                
                if detection_result.detected_objects:
                    confidences = [obj.confidence for obj in detection_result.detected_objects]
                    avg_confidence.extend(confidences)
                
                frame_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_idx}: {detection_result.total_detections} objects detected"
                )
                
                progress_bar.progress((frame_idx + 1) / max_frames)
            
            # Final statistics
            with stats_placeholder.container():
                st.subheader("Video Processing Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Frames Processed", total_frames)
                col2.metric("Total Detections", total_detections)
                col3.metric("Avg Confidence", f"{np.mean(avg_confidence):.2%}" if avg_confidence else "N/A")

else:  # Webcam
    st.info("Webcam feature requires direct camera access. For this demo, upload an image or video instead.")