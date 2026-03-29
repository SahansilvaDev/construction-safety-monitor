"""Tests for the Detection dataclass."""

from src.detector import Detection


def test_detection_center():
    det = Detection(bbox=(100, 100, 200, 300),
                    class_id=5, class_name="Person", confidence=0.9)
    assert det.center == (150.0, 200.0)


def test_detection_width_height():
    det = Detection(bbox=(0, 0, 100, 200),
                    class_id=5, class_name="Person", confidence=0.9)
    assert det.width == 100.0
    assert det.height == 200.0
