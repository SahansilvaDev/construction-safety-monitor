"""Spatial reasoning to pair PPE detections with workers."""

from dataclasses import dataclass
from src.detector import Detection


@dataclass
class WorkerPPEPair:
    """A worker paired with their associated PPE detections.

    After YOLOv8 runs, we get separate boxes for Person, Hardhat,
    NO-Hardhat, Safety Vest, etc. This class holds the result of
    figuring out which PPE belongs to which worker.
    """
    worker: Detection
    hardhat: Detection | None = None
    vest: Detection | None = None
    mask: Detection | None = None
    no_hardhat: Detection | None = None
    no_vest: Detection | None = None
    no_mask: Detection | None = None


    @property
    def has_hardhat(self) -> bool:
        return self.hardhat is not None

    @property
    def has_vest(self) -> bool:
        return self.vest is not None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute how much two bounding boxes overlap.

    Returns a value from 0.0 (no overlap) to 1.0 (identical boxes).
    Used to match PPE detections to worker body regions.

    Args:
        box1, box2: (x1, y1, x2, y2) pixel coordinates.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def point_in_box(point: tuple, box: tuple) -> bool:
    """Check if a point (x, y) falls inside a bounding box."""
    px, py = point
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]

def get_head_region(worker_bbox: tuple, ratio: float = 0.30) -> tuple:
    """Get the top 30% of a worker bbox - this is the head region.

    A Hardhat / NO-Hardhat detection whose center is in this region
    belongs to this worker.
    """
    x1, y1, x2, y2 = worker_bbox
    head_y2 = y1 + (y2 - y1) * ratio
    return (x1, y1, x2, head_y2)


def get_torso_region(worker_bbox: tuple, head_ratio: float = 0.30,
                     torso_ratio: float = 0.40) -> tuple:
    """Get the middle 40% of a worker bbox - this is the torso region.

    A Safety Vest / NO-Safety Vest detection here belongs to this worker.
    """
    x1, y1, x2, y2 = worker_bbox
    h = y2 - y1
    torso_y1 = y1 + h * head_ratio
    torso_y2 = torso_y1 + h * torso_ratio
    return (x1, torso_y1, x2, torso_y2)

def pair_workers_with_ppe(
    detections: list[Detection],
    head_ratio: float = 0.30,
    torso_ratio: float = 0.40,
    min_overlap: float = 0.1,
) -> list[WorkerPPEPair]:
    """Associate PPE detections with the correct worker.

    For each Person detection, we look for PPE detections
    that spatially overlap with the worker's head and torso regions.
    """
    workers   = [d for d in detections if d.class_name == "Person"]
    hardhats  = [d for d in detections if d.class_name == "Hardhat"]
    no_hats   = [d for d in detections if d.class_name == "NO-Hardhat"]
    vests     = [d for d in detections if d.class_name == "Safety Vest"]
    no_vests  = [d for d in detections if d.class_name == "NO-Safety Vest"]
    masks     = [d for d in detections if d.class_name == "Mask"]
    no_masks  = [d for d in detections if d.class_name == "NO-Mask"]

    pairs = []
    for worker in workers:
        head   = get_head_region(worker.bbox, head_ratio)
        torso  = get_torso_region(worker.bbox, head_ratio, torso_ratio)

        pair = WorkerPPEPair(worker=worker)
        pair.hardhat    = _best_match(hardhats, head,  min_overlap)
        pair.no_hardhat = _best_match(no_hats,  head,  min_overlap)
        pair.vest       = _best_match(vests,    torso, min_overlap)
        pair.no_vest    = _best_match(no_vests, torso, min_overlap)
        pair.mask       = _best_match(masks,    head,  min_overlap)
        pair.no_mask    = _best_match(no_masks, head,  min_overlap)
        pairs.append(pair)

    return pairs


def _best_match(
    candidates: list[Detection],
    region: tuple,
    min_overlap: float,
) -> Detection | None:
    """Find the detection that best overlaps with a body region."""
    best, best_iou = None, min_overlap

    for det in candidates:
        iou = compute_iou(det.bbox, region)
        if iou > best_iou:
            best_iou = iou
            best = det
        # Also match if the center point falls inside the region
        if best is None and point_in_box(det.center, region):
            best = det

    return best
