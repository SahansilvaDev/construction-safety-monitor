"""Safety rules engine - decides if a worker is compliant or not."""

from dataclasses import dataclass, field
from enum import Enum

from src.detector import Detection
from src.utils import WorkerPPEPair


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Zone:
    """A safety zone with specific PPE requirements.

    Different areas of a site can have stricter rules.
    e.g. scaffolding requires hardhat + vest + mask,
    but a parking area only requires a vest.
    """
    name: str
    polygon: list[tuple]   # (x, y) pixel coordinates defining the area
    required_ppe: list[str] = field(default_factory=lambda: ["Hardhat", "Safety Vest"])
    severity: Severity = Severity.HIGH


@dataclass
class Violation:
    """One detected safety violation."""
    worker: Detection
    rule: str
    missing_ppe: str
    severity: Severity
    confidence: float
    zone: str | None = None
    message: str = ""


@dataclass
class ComplianceResult:
    """Full compliance result for one worker."""
    worker: Detection
    is_compliant: bool
    violations: list[Violation] = field(default_factory=list)
    detected_ppe: list[str] = field(default_factory=list)
    confidence_scores: dict = field(default_factory=dict)

class SafetyRuleEngine:
    """Checks workers against safety rules and produces violations."""

    def __init__(
        self,
        required_ppe: list[str] = None,
        zones: list[Zone] = None,
    ):
        # Default: every worker must have hardhat AND vest
        self.required_ppe = required_ppe or ["Hardhat", "Safety Vest"]
        self.zones = zones or []

    def check_compliance(self, pair: WorkerPPEPair) -> ComplianceResult:
        """Check one worker-PPE pair against the rules."""
        violations = []
        detected_ppe = []
        confidence_scores = {}

        # Use zone rules if worker is inside a zone, else global rules
        active_zone = self._get_worker_zone(pair.worker)
        required = active_zone.required_ppe if active_zone else self.required_ppe
        zone_name = active_zone.name if active_zone else None

        # --- Check Hardhat ---
        if "Hardhat" in required:
            if pair.has_hardhat:
                detected_ppe.append("Hardhat")
                confidence_scores["Hardhat"] = pair.hardhat.confidence
            elif pair.no_hardhat:
                # Model actively detected NO hardhat
                violations.append(Violation(
                    worker=pair.worker,
                    rule="R1 - Hard Hat Required",
                    missing_ppe="Hardhat",
                    severity=Severity.CRITICAL,
                    confidence=pair.no_hardhat.confidence,
                    zone=zone_name,
                    message=f"Worker missing hard hat (conf: {pair.no_hardhat.confidence:.2f})",
                ))
            else:
                # No hardhat detected at all - flag with low confidence
                violations.append(Violation(
                    worker=pair.worker,
                    rule="R1 - Hard Hat Required",
                    missing_ppe="Hardhat",
                    severity=Severity.CRITICAL,
                    confidence=0.0,
                    zone=zone_name,
                    message="Worker missing hard hat (no PPE detected in head region)",
                ))

        # --- Check Safety Vest ---
        if "Safety Vest" in required:
            if pair.has_vest:
                detected_ppe.append("Safety Vest")
                confidence_scores["Safety Vest"] = pair.vest.confidence
            elif pair.no_vest:
                violations.append(Violation(
                    worker=pair.worker,
                    rule="R2 - Safety Vest Required",
                    missing_ppe="Safety Vest",
                    severity=Severity.HIGH,
                    confidence=pair.no_vest.confidence,
                    zone=zone_name,
                    message=f"Worker missing safety vest (conf: {pair.no_vest.confidence:.2f})",
                ))
            else:
                violations.append(Violation(
                    worker=pair.worker,
                    rule="R2 - Safety Vest Required",
                    missing_ppe="Safety Vest",
                    severity=Severity.HIGH,
                    confidence=0.0,
                    zone=zone_name,
                    message="Worker missing safety vest (no PPE detected in torso region)",
                ))

        # --- Check Mask (only if explicitly required) ---
        if "Mask" in required:
            if pair.has_mask:
                detected_ppe.append("Mask")
                confidence_scores["Mask"] = pair.mask.confidence
            elif pair.no_mask:
                violations.append(Violation(
                    worker=pair.worker,
                    rule="R3 - Face Mask Required",
                    missing_ppe="Mask",
                    severity=Severity.MEDIUM,
                    confidence=pair.no_mask.confidence,
                    zone=zone_name,
                    message=f"Worker missing face mask (conf: {pair.no_mask.confidence:.2f})",
                ))

        return ComplianceResult(
            worker=pair.worker,
            is_compliant=len(violations) == 0,
            violations=violations,
            detected_ppe=detected_ppe,
            confidence_scores=confidence_scores,
        )

    def check_all(self, pairs: list[WorkerPPEPair]) -> list[ComplianceResult]:
        """Check all workers at once."""
        return [self.check_compliance(pair) for pair in pairs]

    def _get_worker_zone(self, worker: Detection) -> Zone | None:
        """Check if worker is inside any defined zone."""
        for zone in self.zones:
            if self._point_in_polygon(worker.center, zone.polygon):
                return zone
        return None

    @staticmethod
    def _point_in_polygon(point: tuple, polygon: list[tuple]) -> bool:
        """Ray casting algorithm - checks if a point is inside a polygon."""
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def get_summary(self, results: list[ComplianceResult]) -> dict:
        """Summary stats for a full frame or session."""
        total = len(results)
        compliant = sum(1 for r in results if r.is_compliant)
        all_violations = [v for r in results for v in r.violations]

        return {
            "total_workers": total,
            "compliant_workers": compliant,
            "non_compliant_workers": total - compliant,
            "compliance_rate": compliant / total if total > 0 else 1.0,
            "total_violations": len(all_violations),
        }
