"""Frame annotation - draws detections and violations on images."""

import cv2
import numpy as np

from src.config import COLORS
from src.safety_rules import ComplianceResult, Zone


class FrameAnnotator:
    """Draws bounding boxes, labels, and violation overlays on frames."""

    def __init__(self, font_scale: float = 0.5, thickness: int = 2):
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX


    def annotate_frame(
        self,
        frame: np.ndarray,
        results: list[ComplianceResult],
        zones: list[Zone] = None,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Annotate a frame with compliance results.

        Args:
            frame: BGR image from OpenCV.
            results: Compliance result for each detected worker.
            zones: Optional zone polygons to draw.
            show_confidence: Whether to show confidence scores.

        Returns:
            Annotated copy of the frame.
        """
        annotated = frame.copy()

        if zones:
            self._draw_zones(annotated, zones)

        for result in results:
            self._draw_worker(annotated, result, show_confidence)

        self._draw_summary(annotated, results)

        return annotated


    def _draw_worker(
        self,
        frame: np.ndarray,
        result: ComplianceResult,
        show_confidence: bool,
    ):
        """Draw a worker's bbox with green (safe) or red (violation)."""
        x1, y1, x2, y2 = [int(v) for v in result.worker.bbox]

        color = COLORS["compliant"] if result.is_compliant else COLORS["violation"]
        status = "SAFE" if result.is_compliant else "VIOLATION"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Draw status label with filled background
        label = status
        if show_confidence:
            label += f" ({result.worker.confidence:.2f})"

        label_size = cv2.getTextSize(
            label, self.font, self.font_scale, self.thickness
        )[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color, -1,  # -1 = filled
        )
        cv2.putText(
            frame, label, (x1, y1 - 5),
            self.font, self.font_scale, (255, 255, 255), self.thickness,
        )

        # Draw detected PPE below the box (green)
        y_offset = y2 + 15
        for ppe in result.detected_ppe:
            conf = result.confidence_scores.get(ppe, 0)
            text = f"+ {ppe} ({conf:.2f})" if show_confidence else f"+ {ppe}"
            cv2.putText(
                frame, text, (x1, y_offset),
                self.font, self.font_scale * 0.8, COLORS["compliant"], 1,
            )
            y_offset += 15

        # Draw violations below detected PPE (red)
        for v in result.violations:
            text = f"X {v.missing_ppe}"
            if show_confidence and v.confidence > 0:
                text += f" ({v.confidence:.2f})"
            cv2.putText(
                frame, text, (x1, y_offset),
                self.font, self.font_scale * 0.8, COLORS["violation"], 1,
            )
            y_offset += 15


    def _draw_summary(
        self,
        frame: np.ndarray,
        results: list[ComplianceResult],
    ):
        """Draw a stats panel in the top-right corner."""
        total = len(results)
        compliant = sum(1 for r in results if r.is_compliant)
        violations = total - compliant

        h, w = frame.shape[:2]
        pw, ph = 220, 80
        x1, y1 = w - pw - 10, 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + pw, y1 + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Workers:    {total}",
                    (x1 + 10, y1 + 20), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Compliant:  {compliant}",
                    (x1 + 10, y1 + 40), self.font, 0.5, COLORS["compliant"], 1)
        cv2.putText(frame, f"Violations: {violations}",
                    (x1 + 10, y1 + 60), self.font, 0.5, COLORS["violation"], 1)

    def _draw_zones(self, frame: np.ndarray, zones: list[Zone]):
        """Draw zone boundaries as semi-transparent polygons."""
        overlay = frame.copy()
        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, COLORS["zone"], 2)
            cv2.fillPoly(overlay, [pts], COLORS["zone"])
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(frame, zone.name, tuple(centroid),
                        self.font, self.font_scale, COLORS["zone"], self.thickness)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


    def create_violation_report(
        self,
        frame: np.ndarray,
        results: list[ComplianceResult],
    ) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=" * 50,
            "CONSTRUCTION SAFETY VIOLATION REPORT",
            "=" * 50,
            "",
        ]

        total = len(results)
        compliant = sum(1 for r in results if r.is_compliant)
        lines.append(f"Workers Detected : {total}")
        lines.append(f"Compliant        : {compliant}")
        lines.append(f"Non-Compliant    : {total - compliant}")
        if total > 0:
            lines.append(f"Compliance Rate  : {compliant/total*100:.1f}%")
        lines.append("")

        n = 0
        for result in results:
            if result.is_compliant:
                continue
            n += 1
            cx, cy = result.worker.center
            lines.append(f"Violation #{n} — Worker at ({cx:.0f}, {cy:.0f})")
            for v in result.violations:
                lines.append(f"  Rule     : {v.rule}")
                lines.append(f"  Missing  : {v.missing_ppe}")
                lines.append(f"  Severity : {v.severity.value}")
                if v.confidence > 0:
                    lines.append(f"  Conf     : {v.confidence:.2f}")
                if v.zone:
                    lines.append(f"  Zone     : {v.zone}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)
