

\# YOLO Object Detection Pipeline



A computer vision application for detecting objects in images and videos using YOLOv8. The project provides a Streamlit interface for running inference, visualizing bounding boxes and confidence scores, and exporting validated detection results as JSON.



The pipeline is designed as a portfolio demonstration of practical deep learning, computer vision inference, structured outputs, and robotics-oriented system design.



\---



\## Overview



Robotics and autonomous systems need more than an annotated image. They need structured information about detected objects, including:



\- Object class, such as `person`, `car`, or `dog`

\- Detection confidence score

\- Bounding box coordinates

\- Frame dimensions and timestamp

\- Object area relative to the image frame



This project converts raw YOLOv8 predictions into structured Pydantic models. The resulting JSON can be consumed by robotics middleware, backend APIs, databases, monitoring systems, or downstream planning algorithms.



\---



\## Features



\- Object detection using pretrained YOLOv8 models

\- Image upload and video upload support

\- Bounding box visualization with class labels and confidence scores

\- Configurable confidence threshold

\- Configurable IoU threshold for non-maximum suppression

\- Selectable YOLO model sizes: Nano, Small, Medium, and Large

\- Structured JSON output using Pydantic schemas

\- JSON export for downstream systems

\- Video frame-by-frame inference and detection statistics

\- Local inference with no external API key required



\---



\## Architecture



```text

Input Image or Video

&#x20;       |

&#x20;       v

+-----------------------+

| Streamlit Application |

+-----------------------+

&#x20;       |

&#x20;       v

+-----------------------+

| OpenCV Frame Handling |

+-----------------------+

&#x20;       |

&#x20;       v

+-----------------------+

| YOLOv8 Inference      |

| - Class ID            |

| - Confidence          |

| - Bounding Box        |

+-----------------------+

&#x20;       |

&#x20;       v

+-----------------------+

| Pydantic Validation   |

| - Type checking       |

| - Range validation    |

| - Structured schema   |

+-----------------------+

&#x20;       |

&#x20;       v

+-----------------------+

| JSON Output / Export  |

+-----------------------+

```



\---



\## Technology Stack



| Component | Technology | Purpose |

|---|---|---|

| Deep Learning Model | YOLOv8 / Ultralytics | Performs object detection on images and video frames |

| Computer Vision | OpenCV | Reads video frames, processes image arrays, and draws bounding boxes |

| Data Validation | Pydantic v2 | Defines and validates structured detection output contracts |

| Frontend | Streamlit | Provides an interactive interface for image/video upload and result visualization |

| Image Processing | Pillow, NumPy | Handles image conversion and numerical operations |

| Language | Python | Main application language |



\---



\## How YOLO Works



YOLO stands for "You Only Look Once." Unlike older object detection systems that generate multiple image regions before classifying them, YOLO performs object localization and classification in a single neural network forward pass.



For each detected object, YOLO returns:



\- `class\_id`: Numeric class identifier

\- `class\_name`: Object label such as `person`, `car`, or `dog`

\- `confidence`: Probability-like score representing model confidence

\- `bounding\_box`: Pixel coordinates of the object location



The pretrained model uses the COCO dataset class set, which includes 80 common object categories such as people, cars, bicycles, trucks, animals, chairs, bottles, laptops, and traffic lights.



\---



\## Structured Detection Output



Raw model outputs are converted into validated Pydantic schemas.



\### Bounding Box Schema



```python

class BoundingBox(BaseModel):

&#x20;   x\_min: float

&#x20;   y\_min: float

&#x20;   x\_max: float

&#x20;   y\_max: float

&#x20;   width: float

&#x20;   height: float

```



\### Detected Object Schema



```python

class DetectedObject(BaseModel):

&#x20;   class\_id: int

&#x20;   class\_name: str

&#x20;   confidence: float = Field(ge=0.0, le=1.0)

&#x20;   bounding\_box: BoundingBox

&#x20;   area\_percentage: float = Field(ge=0.0, le=100.0)

```



\### Frame Result Schema



```python

class FrameDetectionResult(BaseModel):

&#x20;   frame\_id: int

&#x20;   timestamp: str

&#x20;   frame\_width: int

&#x20;   frame\_height: int

&#x20;   total\_detections: int

&#x20;   detected\_objects: List\[DetectedObject]

&#x20;   detection\_quality: Literal\["low", "medium", "high"]

```



\---



\## Example JSON Output



```json

{

&#x20; "frame\_id": 0,

&#x20; "timestamp": "2026-03-20T10:15:22.123456",

&#x20; "frame\_width": 1280,

&#x20; "frame\_height": 720,

&#x20; "total\_detections": 2,

&#x20; "detection\_quality": "high",

&#x20; "detected\_objects": \[

&#x20;   {

&#x20;     "class\_id": 0,

&#x20;     "class\_name": "person",

&#x20;     "confidence": 0.93,

&#x20;     "bounding\_box": {

&#x20;       "x\_min": 120,

&#x20;       "y\_min": 85,

&#x20;       "x\_max": 410,

&#x20;       "y\_max": 700,

&#x20;       "width": 290,

&#x20;       "height": 615

&#x20;     },

&#x20;     "area\_percentage": 19.34

&#x20;   },

&#x20;   {

&#x20;     "class\_id": 2,

&#x20;     "class\_name": "car",

&#x20;     "confidence": 0.88,

&#x20;     "bounding\_box": {

&#x20;       "x\_min": 560,

&#x20;       "y\_min": 260,

&#x20;       "x\_max": 1100,

&#x20;       "y\_max": 640,

&#x20;       "width": 540,

&#x20;       "height": 380

&#x20;     },

&#x20;     "area\_percentage": 22.27

&#x20;   }

&#x20; ]

}

```



\---



\## Project Structure



```text

yolo-detection-pipeline/

├── app.py               # Streamlit user interface

├── detector.py          # YOLOv8 inference and video processing logic

├── schemas.py           # Pydantic validation schemas

├── requirements.txt     # Project dependencies

├── .gitignore           # Files excluded from Git tracking

└── README.md            # Project documentation

```



\---



\## Installation



\### Prerequisites



\- Python 3.10 or higher

\- Git

\- Webcam is optional

\- GPU is optional; the project can run on CPU



\### Clone the Repository



```bash

git clone https://github.com/YOUR\_GITHUB\_USERNAME/yolo-detection-pipeline.git

cd yolo-detection-pipeline

```



\### Create a Virtual Environment



Windows:



```bash

py -m venv venv

venv\\Scripts\\activate

```



Linux or macOS:



```bash

python3 -m venv venv

source venv/bin/activate

```



\### Install Dependencies



```bash

python -m pip install -r requirements.txt

```



\### Run the Application



```bash

python -m streamlit run app.py

```



Open the application in a browser:



```text

http://localhost:8501

```



The YOLO model weights will be downloaded automatically the first time a selected model is loaded.



\---



\## Usage



1\. Start the Streamlit application.

2\. Select a YOLO model size from the sidebar.

3\. Set the confidence threshold and IoU threshold.

4\. Choose either image upload or video upload.

5\. Upload a file.

6\. Run detection.

7\. Review bounding boxes, labels, confidence scores, and structured metadata.

8\. Export the detection result as JSON if needed.



\---



\## Model Size Trade-Offs



| Model | Speed | Accuracy | Recommended Use Case |

|---|---|---|---|

| `yolov8n` | Fastest | Lowest | Edge devices, Raspberry Pi, low-latency prototypes |

| `yolov8s` | Fast | Moderate | General real-time detection |

| `yolov8m` | Balanced | High | Desktop applications and robotics prototypes |

| `yolov8l` | Slowest | Highest | Accuracy-focused offline analysis or GPU deployment |



The model selection exposes a common machine learning engineering trade-off: larger models generally improve detection accuracy but require more memory and inference time.



\---



\## Configuration Parameters



\### Confidence Threshold



The confidence threshold controls which predictions are retained.



\- Lower threshold, such as `0.30`: detects more objects but may introduce false positives.

\- Higher threshold, such as `0.80`: produces fewer but more reliable detections.

\- Default threshold: `0.50`.



\### IoU Threshold



IoU means Intersection over Union. It is used during non-maximum suppression to remove duplicate overlapping bounding boxes.



\- Lower IoU threshold: removes more overlapping detections.

\- Higher IoU threshold: allows more overlapping detections.

\- Default threshold: `0.45`.



\---



\## Robotics and ML Applications



This detection pipeline can be extended for several real-world use cases:



\### Warehouse Robotics



A warehouse robot can detect people, packages, forklifts, and obstacles. Bounding box centers can be used by a motion planner to determine where the robot should move or stop.



\### Autonomous Vehicles



A vehicle perception system can detect cars, pedestrians, bicycles, buses, trucks, and traffic lights. Detection outputs can be passed to a tracking and path-planning component.



\### Manufacturing Inspection



The pretrained YOLO model can be fine-tuned on custom datasets to detect defective products, missing components, damaged packaging, or safety violations.



\### Security Monitoring



The pipeline can detect people and vehicles in camera feeds, count objects over time, and send structured alerts when rules are triggered.



\### Drone Navigation



A drone can use object detection as one input for obstacle awareness, target following, and scene understanding.



\---



\## Production Improvements



This project is an MVP. A production version could include:



\- FastAPI backend for asynchronous inference endpoints

\- Docker containerization

\- GPU deployment using CUDA

\- NVIDIA Jetson deployment for edge robotics

\- ROS or ROS2 integration using detection messages and ROS topics

\- Object tracking with ByteTrack or DeepSORT

\- Detection logging to PostgreSQL or MongoDB

\- Monitoring with Prometheus and Grafana

\- Model quantization for faster edge-device inference

\- Fine-tuning YOLO on a custom domain dataset

\- Unit tests for schema validation and coordinate transformations

\- Batch processing and job queue support



\---



\## Limitations



\- The default pretrained YOLO model only recognizes COCO dataset classes.

\- It cannot reliably detect specialized objects such as manufacturing defects without fine-tuning.

\- CPU inference may be slower for high-resolution video.

\- Object detection does not provide object identity tracking across frames; a tracking model would be needed for that.

\- Confidence scores represent model confidence, not a guaranteed probability of correctness.



\---



\## License



This project is intended for educational and portfolio purposes. Refer to the Ultralytics YOLO license for conditions related to YOLO model usage.

```

