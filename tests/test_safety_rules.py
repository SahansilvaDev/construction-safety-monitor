"""Unit tests for the safety rules engine."""

import pytest
from src.detector import Detection
from src.utils import (
    WorkerPPEPair,
    compute_iou,
    get_head_region,
    get_torso_region,
    point_in_box,
)
from src.safety_rules import SafetyRuleEngine, Zone, Severity


def make_detection(cls_name, bbox=(100, 100, 200, 400), conf=0.9):
    """Create a Detection object for testing."""
    cls_map = {
        "Hardhat": 0, "Mask": 1, "NO-Hardhat": 2, "NO-Mask": 3,
        "NO-Safety Vest": 4, "Person": 5, "Safety Cone": 6,
        "Safety Vest": 7, "machinery": 8, "vehicle": 9,
    }
    return Detection(
        bbox=bbox,
        class_id=cls_map[cls_name],
        class_name=cls_name,
        confidence=conf,
    )


class TestComplianceCheck:

    def test_fully_compliant_worker(self):
        """Worker with hardhat and vest should be safe."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            hardhat=make_detection("Hardhat"),
            vest=make_detection("Safety Vest"),
        )
        result = SafetyRuleEngine().check_compliance(pair)

        assert result.is_compliant
        assert len(result.violations) == 0
        assert "Hardhat" in result.detected_ppe
        assert "Safety Vest" in result.detected_ppe

    def test_missing_hardhat_flagged(self):
        """NO-Hardhat detection should produce a CRITICAL violation."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            no_hardhat=make_detection("NO-Hardhat"),
            vest=make_detection("Safety Vest"),
        )
        result = SafetyRuleEngine().check_compliance(pair)

        assert not result.is_compliant
        assert result.violations[0].missing_ppe == "Hardhat"
        assert result.violations[0].severity == Severity.CRITICAL

    def test_missing_vest_flagged(self):
        """NO-Safety Vest detection should produce a HIGH violation."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            hardhat=make_detection("Hardhat"),
            no_vest=make_detection("NO-Safety Vest"),
        )
        result = SafetyRuleEngine().check_compliance(pair)

        assert not result.is_compliant
        assert result.violations[0].missing_ppe == "Safety Vest"
        assert result.violations[0].severity == Severity.HIGH

    def test_missing_both_gives_two_violations(self):
        """Worker with no PPE at all should get two violations."""
        pair = WorkerPPEPair(worker=make_detection("Person"))
        result = SafetyRuleEngine().check_compliance(pair)

        assert not result.is_compliant
        assert len(result.violations) == 2

    def test_mask_not_checked_by_default(self):
        """Mask rule is optional - not enforced unless explicitly required."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            hardhat=make_detection("Hardhat"),
            vest=make_detection("Safety Vest"),
            no_mask=make_detection("NO-Mask"),
        )
        result = SafetyRuleEngine().check_compliance(pair)

        # Default rules don't include Mask
        assert result.is_compliant

    def test_mask_checked_when_required(self):
        """Mask violation raised when Mask is in required_ppe."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            hardhat=make_detection("Hardhat"),
            vest=make_detection("Safety Vest"),
            no_mask=make_detection("NO-Mask"),
        )
        engine = SafetyRuleEngine(
            required_ppe=["Hardhat", "Safety Vest", "Mask"]
        )
        result = engine.check_compliance(pair)

        assert not result.is_compliant
        assert result.violations[0].missing_ppe == "Mask"

    def test_confidence_scores_recorded(self):
        """Confidence from PPE detections should be stored in result."""
        pair = WorkerPPEPair(
            worker=make_detection("Person"),
            hardhat=make_detection("Hardhat", conf=0.85),
            vest=make_detection("Safety Vest", conf=0.72),
        )
        result = SafetyRuleEngine().check_compliance(pair)

        assert result.confidence_scores["Hardhat"] == pytest.approx(0.85)
        assert result.confidence_scores["Safety Vest"] == pytest.approx(0.72)


class TestZoneRules:

    def test_zone_violation_includes_zone_name(self):
        """Violation inside a zone should record the zone name."""
        zone = Zone(
            name="scaffolding",
            polygon=[(0, 0), (500, 0), (500, 500), (0, 500)],
            required_ppe=["Hardhat", "Safety Vest"],
        )
        pair = WorkerPPEPair(
            worker=make_detection("Person", bbox=(100, 100, 200, 400))
        )
        result = SafetyRuleEngine(zones=[zone]).check_compliance(pair)

        assert not result.is_compliant
        assert result.violations[0].zone == "scaffolding"

    def test_worker_outside_zone_uses_global_rules(self):
        """Worker outside all zones should use the global required_ppe."""
        zone = Zone(
            name="far_zone",
            polygon=[(900, 900), (1000, 900), (1000, 1000), (900, 1000)],
        )
        pair = WorkerPPEPair(
            worker=make_detection("Person", bbox=(100, 100, 200, 400)),
            hardhat=make_detection("Hardhat"),
            vest=make_detection("Safety Vest"),
        )
        result = SafetyRuleEngine(zones=[zone]).check_compliance(pair)

        assert result.is_compliant


class TestSummary:

    def test_summary_counts(self):
        engine = SafetyRuleEngine()
        pairs = [
            WorkerPPEPair(
                worker=make_detection("Person", bbox=(100, 100, 200, 400)),
                hardhat=make_detection("Hardhat"),
                vest=make_detection("Safety Vest"),
            ),
            WorkerPPEPair(
                worker=make_detection("Person", bbox=(300, 100, 400, 400)),
            ),
        ]
        summary = engine.get_summary(engine.check_all(pairs))

        assert summary["total_workers"] == 2
        assert summary["compliant_workers"] == 1
        assert summary["non_compliant_workers"] == 1
        assert summary["total_violations"] == 2


class TestSpatialUtils:

    def test_iou_identical_boxes(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0 < iou < 1

    def test_point_inside_box(self):
        assert point_in_box((50, 50), (0, 0, 100, 100))

    def test_point_outside_box(self):
        assert not point_in_box((200, 50), (0, 0, 100, 100))

    def test_head_region_is_top_30_percent(self):
        head = get_head_region((0, 0, 100, 400), ratio=0.30)
        assert head == (0, 0, 100, 120)

    def test_torso_region_is_middle_40_percent(self):
        torso = get_torso_region((0, 0, 100, 400), 0.30, 0.40)
        assert torso == (0, 120, 100, 280)
