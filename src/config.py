"""Central configuration for the Construction Safety Monitor."""

from pathlib import Path
from dataclasses import dataclass, field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class ModelConfig:
    """Model-related configuration."""
    base_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    img_size: int = 640

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 16
    patience: int = 10
    learning_rate: float = 0.01
    lr_final: float = 0.01
    img_size: int = 640


# 10-class mapping from the Kaggle Construction Site Safety dataset
CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# Classes that indicate PPE is present (worn correctly)
PPE_PRESENT = {"Hardhat", "Mask", "Safety Vest"}

# Classes that indicate PPE is missing
PPE_MISSING = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# Mapping: missing-PPE class -> the PPE that should be worn
MISSING_TO_PPE = {
    "NO-Hardhat": "Hardhat",
    "NO-Mask": "Mask",
    "NO-Safety Vest": "Safety Vest",
}


# Color scheme (BGR for OpenCV)
COLORS = {
    "Hardhat": (0, 255, 0),          # Green
    "Mask": (0, 255, 0),             # Green
    "NO-Hardhat": (0, 0, 255),       # Red
    "NO-Mask": (0, 0, 255),          # Red
    "NO-Safety Vest": (0, 0, 255),   # Red
    "Person": (255, 165, 0),         # Orange
    "Safety Cone": (0, 255, 255),    # Yellow
    "Safety Vest": (0, 255, 0),      # Green
    "machinery": (128, 128, 128),    # Gray
    "vehicle": (128, 128, 128),      # Gray
    "compliant": (0, 200, 0),        # Green
    "violation": (0, 0, 255),        # Red
}

# Paths
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset" / "css-data"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = DATA_DIR / "sample_images"

# Pre-trained weights from Kaggle notebook (YOLOv8n, 100 epochs)
PRETRAINED_WEIGHTS = (
    DATA_DIR / "dataset" / "results_yolov8n_100e"
    / "kaggle" / "working" / "runs" / "detect" / "train" / "weights" / "best.pt"
)
