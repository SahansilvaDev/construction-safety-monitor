"""YOLOv8 wrapper for construction safety detection."""

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

from src.config import CLASS_NAMES, ModelConfig


@dataclass
class Detection:
    """A single detection result from YOLOv8."""
    bbox: tuple        # (x1, y1, x2, y2) pixel coordinates
    class_id: int      # 0-9 matching CLASS_NAMES
    class_name: str    # e.g. "Hardhat", "NO-Hardhat"
    confidence: float  # 0.0 to 1.0


    @property
    def center(self) -> tuple:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class SafetyDetector:
    """Wraps YOLOv8 for construction site safety detection.

    The model detects 10 classes:
        0: Hardhat          - hard hat worn correctly
        1: Mask             - face mask worn correctly
        2: NO-Hardhat       - head visible WITHOUT hard hat  <- violation
        3: NO-Mask          - face visible WITHOUT mask      <- violation
        4: NO-Safety Vest   - torso visible WITHOUT vest     <- violation
        5: Person           - worker detected
        6: Safety Cone      - traffic cone
        7: Safety Vest      - hi-vis vest worn correctly
        8: machinery        - construction equipment
        9: vehicle          - vehicle on site
    """

    def __init__(self, model_path: str, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = YOLO(model_path)
        self.class_names = CLASS_NAMES


    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (from cv2.imread).

        Returns:
            List of Detection objects, one per detected object.
        """
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            verbose=False,
        )
        return self._parse_results(results[0])

    def _parse_results(self, result) -> list[Detection]:
        """Convert ultralytics Result into our Detection objects."""
        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            detections.append(Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=cls_id,
                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                confidence=conf,
            ))

        return detections
