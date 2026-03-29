# Construction Safety Monitor

AI-powered computer vision system that monitors construction sites in real time, detecting workers, identifying PPE compliance (hard hats, high-visibility vests, and masks), and flagging safety violations.

Built with **YOLOv8** for object detection and a custom **rule engine** for compliance checking.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [How YOLOv8 Works](#how-yolov8-works)
- [Dataset](#dataset)
- [Safety Rules](#safety-rules)
- [Model Performance](#model-performance)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## Project Structure

```
construction-safety-monitor/
├── data/
│   ├── download_dataset.py        # Alternative Roboflow download script
│   ├── prepare_dataset.py         # Validate, remap classes, analyze
│   ├── sample_images/             # Sample images for demo
│   ├── README.md                  # Full dataset documentation
│   └── dataset/
│       ├── css-data/              # Kaggle dataset (train/valid/test)
│       │   ├── data.yaml          # YOLOv8 dataset config
│       │   ├── train/images+labels/
│       │   ├── valid/images+labels/
│       │   └── test/images+labels/
│       ├── results_yolov8n_100e/  # Pre-trained model + Kaggle notebook
│       └── source_files/          # Extra test images and videos
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: class distribution, bbox analysis
│   ├── 02_training.ipynb          # Colab-compatible training notebook
│   └── 03_evaluation.ipynb        # mAP analysis, confusion matrix
├── src/
│   ├── config.py                  # Central configuration and class mappings
│   ├── detector.py                # YOLOv8 wrapper
│   ├── safety_rules.py            # Compliance rule engine with zone support
│   ├── violation_tracker.py       # Temporal tracking for video streams
│   ├── annotator.py               # Frame annotation and report generation
│   └── utils.py                   # Spatial reasoning: worker-PPE pairing
├── app/
│   └── streamlit_app.py           # Interactive web demo
├── tests/
│   ├── test_safety_rules.py       # Unit tests for compliance logic
│   └── test_detector.py           # Detection module tests
├── train.py                       # CLI training script
├── inference.py                   # CLI inference (image/video/webcam)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/construction-safety-monitor.git
cd construction-safety-monitor
pip install -r requirements.txt
```

### 2. Get the Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) and extract into `data/dataset/`. The download already includes pre-trained weights.

### 3. Run Inference (Using Pre-trained Weights)

The Kaggle download includes a YOLOv8n model already trained for 100 epochs. You can use it immediately:

```bash
# Single image
python inference.py --source path/to/image.jpg

# Video file
python inference.py --source data/dataset/source_files/source_files/hardhat.mp4

# Webcam
python inference.py --source 0

# Directory of images
python inference.py --source data/dataset/css-data/test/images/ --output outputs/
```

### 4. Train Your Own Model (Optional)

**Option A: Google Colab (Recommended)**

Open `notebooks/02_training.ipynb` in [Google Colab](https://colab.research.google.com/) with GPU runtime enabled.

**Option B: Local**

```bash
python train.py --model yolov8n.pt --data data/dataset/css-data/data.yaml --epochs 100
```

### 5. Launch Demo App

```bash
streamlit run app/streamlit_app.py
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## How It Works

The system operates as a 5-stage pipeline:

```
Image/Frame
    │
    ▼
┌─────────────────┐
│  1. YOLOv8       │  Detect all objects (workers, PPE, NO-PPE, etc.)
│     Detection    │  Returns bounding boxes + class + confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Worker-PPE   │  Match PPE detections to specific workers using
│     Pairing      │  spatial reasoning (head region → helmet, torso → vest)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Compliance   │  Check each worker against safety rules
│     Check        │  (configurable per-zone PPE requirements)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Violation    │  For video: require violations to persist across
│     Tracking     │  multiple frames before alerting (reduces false alarms)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Annotation   │  Draw green/red boxes, PPE labels, violation alerts,
│     & Reporting  │  summary statistics, and text reports
└─────────────────┘
```

### Stage 1: Detection

YOLOv8 processes the entire image in a single forward pass and outputs bounding boxes for all 10 classes. See [How YOLOv8 Works](#how-yolov8-works) for details.

### Stage 2: Worker-PPE Pairing

After detection, we have independent boxes for `Person`, `Hardhat`, `NO-Hardhat`, `Safety Vest`, etc. The challenge is determining *which PPE belongs to which worker*.

We solve this with **spatial reasoning**:
- **Head region** = top 30% of the worker's bounding box. We match `Hardhat` and `NO-Hardhat` detections here.
- **Torso region** = middle 40% of the worker's bounding box. We match `Safety Vest` and `NO-Safety Vest` here.
- **Face region** = upper 25%, narrowed horizontally. We match `Mask` and `NO-Mask` here.

Matching uses IoU (Intersection over Union) and center-point containment checks.

### Stage 3: Compliance Check

The rule engine evaluates each worker against the required PPE list:
- If the worker has the PPE → compliant
- If a `NO-*` class was detected → violation with confidence from the detection
- If neither was detected → violation flagged with zero confidence

### Stage 4: Temporal Tracking (Video Only)

In video streams, single-frame false positives are common. The `ViolationTracker` requires a violation to appear in **N consecutive frames** before reporting it. This dramatically reduces false alarms.

### Stage 5: Annotation & Reporting

The annotator draws:
- **Green boxes** around compliant workers
- **Red boxes** around non-compliant workers
- PPE labels with confidence scores
- A summary panel showing total workers, compliant count, and violation count
- A full text report suitable for logging

---

## How YOLOv8 Works

YOLOv8 (You Only Look Once, version 8) is a real-time object detection model. Here's how it processes an image:

### Architecture Overview

```
Input Image (640×640)
       │
       ▼
┌──────────────┐
│  Backbone     │  CSPDarknet53 — extracts visual features at multiple
│  (Feature     │  scales. Uses cross-stage partial connections to reduce
│   Extraction) │  computation while preserving gradient flow.
└──────┬───────┘
       │  Feature maps at 3 scales (small, medium, large objects)
       ▼
┌──────────────┐
│  Neck         │  Feature Pyramid Network (FPN) + Path Aggregation
│  (Multi-scale │  Network (PAN). Fuses features from different scales
│   Fusion)     │  so small objects get high-resolution detail and large
│               │  objects get semantic context.
└──────┬───────┘
       │  Multi-scale feature maps
       ▼
┌──────────────┐
│  Head         │  Decoupled head — predicts three things independently:
│  (Detection)  │   • Object class probabilities (which of the 10 classes)
│               │   • Bounding box coordinates (x, y, width, height)
│               │   • Objectness score (is there an object here?)
└──────┬───────┘
       │
       ▼
  Non-Maximum Suppression (NMS)
  Removes duplicate detections for the same object
       │
       ▼
  Final Detections: [(bbox, class, confidence), ...]
```

### Why "You Only Look Once"?

Traditional detectors (like R-CNN) process an image in two stages: first propose regions that might contain objects, then classify each region. This is slow.

YOLO processes the entire image in a **single pass** through the network. It divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell simultaneously. This makes it fast enough for real-time video processing (60+ FPS on a GPU).

### Transfer Learning

We don't train from scratch. We start with YOLOv8 weights pre-trained on the COCO dataset (80 common object classes like people, cars, chairs). These weights already know how to detect edges, textures, body shapes, and objects. We then **fine-tune** on our construction safety dataset, teaching the model our specific 10 classes. This requires far less data and training time than starting from random weights.

### Model Sizes

| Model | Parameters | Speed (GPU) | mAP (COCO) | Use Case |
|-------|-----------|-------------|------------|----------|
| YOLOv8n (nano) | 3.2M | Fastest | 37.3 | Edge devices, mobile |
| YOLOv8s (small) | 11.2M | Fast | 44.9 | Good accuracy/speed balance |
| YOLOv8m (medium) | 25.9M | Moderate | 50.2 | Higher accuracy needs |

We use **YOLOv8n** for this project because it's fast enough for real-time monitoring and fits within the constraints of Kaggle/Colab free-tier GPUs.

---

## Dataset

**Source:** [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (Kaggle)

Originally from [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety), with images collected from YouTube surveillance videos and multiple Roboflow PPE detection projects.

### Why This Dataset?

The key feature is **dual-class labelling**: it annotates both `Hardhat` and `NO-Hardhat`, `Safety Vest` and `NO-Safety Vest`. This lets the model *actively detect the absence of PPE*, which is more reliable than inferring a violation from a missing detection.

### Statistics

| Split | Images |
|-------|--------|
| Train | 2,605 |
| Valid | 114 |
| Test | 82 |
| **Total** | **2,801** |

### 10 Classes

| ID | Class | Role |
|----|-------|------|
| 0 | Hardhat | PPE present |
| 1 | Mask | PPE present |
| 2 | NO-Hardhat | Violation indicator |
| 3 | NO-Mask | Violation indicator |
| 4 | NO-Safety Vest | Violation indicator |
| 5 | Person | Worker detection |
| 6 | Safety Cone | Scene context |
| 7 | Safety Vest | PPE present |
| 8 | machinery | Scene context |
| 9 | vehicle | Scene context |

See [data/README.md](data/README.md) for full dataset documentation including augmentation details and licensing.

---

## Safety Rules

### Defined Rules

| Rule ID | Rule | Required PPE | Severity | When Triggered |
|---------|------|-------------|----------|----------------|
| R1 | Hard Hat Required | Hardhat | **Critical** | `NO-Hardhat` detected or no helmet found in head region |
| R2 | High-Visibility Vest Required | Safety Vest | **High** | `NO-Safety Vest` detected or no vest found in torso region |
| R3 | Face Mask Required (optional) | Mask | Medium | Only enforced when explicitly enabled |

### What Counts as a Violation

| Scenario | Detection Output | Compliance Result |
|----------|-----------------|-------------------|
| Worker wearing hardhat + vest | `Person` + `Hardhat` + `Safety Vest` | SAFE |
| Worker without hardhat | `Person` + `NO-Hardhat` + `Safety Vest` | VIOLATION (R1) |
| Worker without vest | `Person` + `Hardhat` + `NO-Safety Vest` | VIOLATION (R2) |
| Worker without any PPE | `Person` + `NO-Hardhat` + `NO-Safety Vest` | VIOLATION (R1 + R2) |
| Distant worker, PPE unclear | `Person` only (no PPE detected) | VIOLATION (flagged with low confidence) |

### Zone-Based Rules (Advanced Feature)

Different areas of a site can have different requirements:

```python
from src.safety_rules import SafetyRuleEngine, Zone, Severity

zones = [
    Zone(name="scaffolding",
         polygon=[(0,0), (300,0), (300,400), (0,400)],
         required_ppe=["Hardhat", "Safety Vest", "Mask"],
         severity=Severity.CRITICAL),
    Zone(name="parking",
         polygon=[(500,0), (640,0), (640,400), (500,400)],
         required_ppe=["Safety Vest"],
         severity=Severity.LOW),
]
engine = SafetyRuleEngine(zones=zones)
```

---

## Model Performance

**Model:** YOLOv8n (nano) trained for 100 epochs on the CSS dataset.

### Overall Metrics (Epoch 99)

| Metric | Value |
|--------|-------|
| mAP@50 | **0.809** |
| mAP@50-95 | **0.507** |
| Precision | **0.900** |
| Recall | **0.731** |

### Training Progression

The model converges around epoch 80-90, with early stopping patience of 10 epochs:

- Training box loss: 1.37 → 0.75 (45% reduction)
- Training cls loss: 3.06 → 0.56 (82% reduction)
- Validation mAP@50: 0.25 → 0.81 (224% improvement)

### Why These Numbers Matter

- **Precision 0.90** = When the model flags a violation, it's correct 90% of the time. Low false alarm rate.
- **Recall 0.73** = The model catches 73% of actual violations. Some violations are missed, especially for distant or occluded workers.
- **mAP@50 0.81** = Strong overall detection quality across all classes.

For a safety system, **high precision is more important than high recall** because frequent false alarms cause alert fatigue and operators stop paying attention. The 90% precision is encouraging.

---

## Known Limitations

1. **Distant workers** - Detection accuracy drops for small bounding boxes (workers far from camera)
2. **Heavy occlusion** - When workers overlap, PPE pairing may assign equipment to the wrong worker
3. **Night / low-light** - Training data is primarily daytime; low-light performance is untested
4. **Partial PPE** - The model detects presence/absence but cannot tell if PPE is worn *correctly* (e.g., helmet not fastened)
5. **Small validation set** - Only 114 validation images; metrics could shift with a larger eval set
6. **YOLOv8n size** - The nano model trades accuracy for speed; a larger model (YOLOv8s/m) would improve detection at the cost of inference time

## Future Improvements

- **Object tracking** (DeepSORT / ByteTrack) for consistent worker IDs across video frames
- **Edge deployment** with TensorRT / ONNX for embedded cameras
- **Additional PPE** - goggles, gloves, harness detection
- **Pose estimation** for unsafe posture detection (working at height without fall protection)
- **Alert system** with email/SMS notifications for persistent violations
- **Historical dashboard** with violation analytics over time
- **Larger model** (YOLOv8s) for improved accuracy in safety-critical applications

---

## License

MIT
