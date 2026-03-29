"""Temporal violation tracking to reduce false positives in video."""

import time
from dataclasses import dataclass, field

from src.safety_rules import Violation

@dataclass
class TrackedViolation:
    """A violation being tracked across multiple frames."""
    violation: Violation
    first_seen: float    # timestamp
    last_seen: float     # timestamp
    frame_count: int = 1
    reported: bool = False


class ViolationTracker:
    """Only report violations that persist across multiple frames.

    In a video stream, a single-frame false positive is common.
    This class requires a violation to appear in at least
    `persistence_frames` consecutive frames before reporting it.

    Example:
        tracker = ViolationTracker(persistence_frames=10)
        confirmed = tracker.update(violations_this_frame)
        # confirmed only contains violations seen 10+ frames in a row
    """

    def __init__(
        self,
        persistence_frames: int = 10,
        timeout_seconds: float = 2.0,
    ):
        """
        Args:
            persistence_frames: How many consecutive frames a violation
                                 must appear before being reported.
            timeout_seconds: If a violation disappears for this long,
                             remove it from tracking.
        """
        self.persistence_frames = persistence_frames
        self.timeout_seconds = timeout_seconds
        self.tracked: dict[str, TrackedViolation] = {}
        self.violation_log: list[dict] = []

    def update(self, violations: list[Violation]) -> list[Violation]:
        """Update tracker with violations from the current frame.

        Args:
            violations: All violations detected in this frame.

        Returns:
            Only the confirmed violations (persistent enough to report).
        """
        now = time.time()

        # Update or create tracked entries
        current_keys = set()
        for v in violations:
            key = self._make_key(v)
            current_keys.add(key)

            if key in self.tracked:
                self.tracked[key].last_seen = now
                self.tracked[key].frame_count += 1
                self.tracked[key].violation = v  # keep latest detection
            else:
                self.tracked[key] = TrackedViolation(
                    violation=v,
                    first_seen=now,
                    last_seen=now,
                )

        # Remove violations that haven't been seen recently
        stale = [k for k, t in self.tracked.items()
                 if now - t.last_seen > self.timeout_seconds]
        for k in stale:
            del self.tracked[k]

        # Return violations that have persisted long enough
        confirmed = []
        for tracked in self.tracked.values():
            if tracked.frame_count >= self.persistence_frames:
                confirmed.append(tracked.violation)
                # Log it once when first confirmed
                if not tracked.reported:
                    tracked.reported = True
                    self._log(tracked)

        return confirmed


    def _make_key(self, violation: Violation) -> str:
        """Unique key for a violation based on position + type.

        We quantize position into grid cells to handle slight
        bounding box jitter between frames.
        """
        cx, cy = violation.worker.center
        grid_x = int(cx / 50)
        grid_y = int(cy / 50)
        return f"{violation.missing_ppe}_{grid_x}_{grid_y}"

    def _log(self, tracked: TrackedViolation):
        """Add a confirmed violation to the session log."""
        self.violation_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rule": tracked.violation.rule,
            "missing_ppe": tracked.violation.missing_ppe,
            "severity": tracked.violation.severity.value,
            "confidence": tracked.violation.confidence,
            "position": tracked.violation.worker.center,
            "zone": tracked.violation.zone,
        })

    def get_summary(self) -> dict:
        """Session-level violation summary."""
        return {
            "active_tracked": len(self.tracked),
            "total_logged": len(self.violation_log),
            "log": self.violation_log,
        }

    def reset(self):
        """Clear all tracking state."""
        self.tracked.clear()
        self.violation_log.clear()
