# Dataset Documentation

## Source

**Primary Dataset:** [Construction Site Safety Image Dataset Roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) from Kaggle.

Originally sourced from [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) (v28, exported February 2023). Images were collected from:
- YouTube construction site surveillance videos
- Roboflow PPE detection projects
- Indoor scene datasets (as null/background images)

## Why This Dataset?

This dataset is ideal for construction safety monitoring because:

1. **Dual-class labelling** - It labels both the *presence* and *absence* of PPE (e.g., `Hardhat` vs `NO-Hardhat`). Most datasets only label when PPE is present, making it impossible to distinguish "no PPE detected" from "worker has no PPE". This design directly enables violation detection.

2. **Real-world diversity** - Images span indoor/outdoor sites, different lighting conditions (daylight, overcast, artificial), varying camera distances, and multiple workers per scene.

3. **Pre-split and YOLO-formatted** - Comes with train/valid/test splits and annotations already in YOLOv8 format, reducing preprocessing overhead.

4. **10 useful classes** - Beyond just helmets, it covers vests, masks, cones, machinery, and vehicles, enabling richer safety analysis.

## Dataset Statistics

| Split | Images | Labels |
|-------|--------|--------|
| Train | 2,605 | 2,605 |
| Valid | 114 | 114 |
| Test | 82 | 82 |
| **Total** | **2,801** | **2,801** |

Note: 24 images have empty annotation files (no objects in scene). These serve as negative/background examples.

## Class Definitions (10 Classes)

| ID | Class | Description | Role in Safety Check |
|----|-------|-------------|----------------------|
| 0 | `Hardhat` | Hard hat / safety helmet (worn) | PPE present |
| 1 | `Mask` | Face mask (worn) | PPE present |
| 2 | `NO-Hardhat` | Worker head visible without hard hat | **Violation indicator** |
| 3 | `NO-Mask` | Face visible without mask | **Violation indicator** |
| 4 | `NO-Safety Vest` | Torso visible without high-vis vest | **Violation indicator** |
| 5 | `Person` | Person / worker in the scene | Worker detection |
| 6 | `Safety Cone` | Traffic / safety cone | Scene context |
| 7 | `Safety Vest` | High-visibility vest (worn) | PPE present |
| 8 | `machinery` | Construction machinery (cranes, etc.) | Scene context |
| 9 | `vehicle` | Vehicle on site | Scene context |

### How the Dual-Class Scheme Works

The key insight is that the dataset annotates **both outcomes** for each PPE type:

```
Worker with helmet    --> labelled as: Person + Hardhat
Worker without helmet --> labelled as: Person + NO-Hardhat
```

This means our model learns to *actively detect the absence of PPE*, rather than just inferring it from the lack of a positive detection. This is more reliable because "I didn't detect a helmet" could mean the helmet is occluded or the model missed it, but "I detected a NO-Hardhat" is a positive signal of a violation.

## Annotation Format

YOLO format (one `.txt` file per image):

```
class_id  center_x  center_y  width  height
```

All coordinates are normalized to [0, 1] relative to image dimensions.

**Example** (one line per detected object):
```
0  0.077  0.394  0.091  0.059     # Hardhat
5  0.163  0.420  0.327  0.178     # Person
4  0.250  0.550  0.120  0.200     # NO-Safety Vest
```

## Pre-processing Applied (by Roboflow)

- Auto-orientation of pixel data (EXIF-orientation stripping)
- Resize to 640x640 (stretch)

## Augmentation Applied (by Roboflow)

The training set was augmented with 5 versions of each source image:
- 50% probability of horizontal flip
- Random crop (0-20%)
- Random rotation (-12 to +12 degrees)
- Random shear (-2 to +2 degrees)
- Random brightness (-25% to +25%)
- Random exposure (-20% to +20%)
- Random Gaussian blur (0 to 0.5 pixels)

## Pre-trained Model

The Kaggle dataset includes results from a YOLOv8n model trained for 100 epochs:

| Metric | Value |
|--------|-------|
| mAP@50 | 0.809 |
| mAP@50-95 | 0.507 |
| Precision | 0.900 |
| Recall | 0.731 |

Weights are located at: `data/dataset/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt`

## How to Reproduce

```bash
# 1. Download from Kaggle
# Visit: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
# Download and extract into data/dataset/

# 2. Verify structure
ls data/dataset/css-data/train/images/ | wc -l   # Should show 2605
ls data/dataset/css-data/valid/images/ | wc -l   # Should show 114
ls data/dataset/css-data/test/images/ | wc -l    # Should show 82

# 3. Train (or use pretrained weights from the download)
python train.py --data data/dataset/css-data/data.yaml
```

## Licensing

The Roboflow Construction Site Safety dataset is available under **CC BY 4.0** license.
