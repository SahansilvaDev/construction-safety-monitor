"""Run safety monitoring on images, videos, or webcam.

Usage:
    uv run python inference.py --source image.jpg
    uv run python inference.py --source video.mp4
    uv run python inference.py --source 0          # webcam
    uv run python inference.py --source data/dataset/css-data/test/images/
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import ModelConfig, MODEL_DIR, OUTPUT_DIR, PRETRAINED_WEIGHTS
from src.detector import SafetyDetector
from src.safety_rules import SafetyRuleEngine
from src.violation_tracker import ViolationTracker
from src.annotator import FrameAnnotator
from src.utils import pair_workers_with_ppe


def process_frame(frame, detector, rule_engine, annotator, tracker=None):
    """Run the full pipeline on one frame.

    Pipeline:
        detect → pair workers with PPE → check rules → track → annotate
    """
    detections = detector.detect(frame)
    pairs = pair_workers_with_ppe(detections)
    results = rule_engine.check_all(pairs)

    if tracker:
        all_violations = [v for r in results for v in r.violations]
        tracker.update(all_violations)

    annotated = annotator.annotate_frame(frame, results)
    return annotated, results


def run_image(source, detector, rule_engine, annotator, output_dir):
    """Process a single image or a folder of images."""
    source_path = Path(source)
    images = (
        list(source_path.glob("*.jpg")) +
        list(source_path.glob("*.jpeg")) +
        list(source_path.glob("*.png"))
        if source_path.is_dir()
        else [source_path]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Could not read: {img_path}")
            continue

        annotated, results = process_frame(frame, detector, rule_engine, annotator)

        # Print report to terminal
        report = annotator.create_violation_report(frame, results)
        print(f"\n{img_path.name}")
        print(report)

        # Save annotated image
        out_path = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")


def run_video(source, detector, rule_engine, annotator, output_dir):
    """Process a video file or live webcam stream."""
    is_webcam = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)

    if not cap.isOpened():
        print(f"Could not open: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Set up video writer for non-webcam sources
    writer = None
    if not is_webcam:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"annotated_{Path(source).name}"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h),
        )

    tracker = ViolationTracker(persistence_frames=10)
    frame_count = 0
    start = time.time()
    print("Running... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, results = process_frame(
            frame, detector, rule_engine, annotator, tracker
        )

        # Show FPS
        frame_count += 1
        fps_actual = frame_count / (time.time() - start)
        cv2.putText(annotated, f"FPS: {fps_actual:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("Construction Safety Monitor", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {out_path}")
    cv2.destroyAllWindows()

    summary = tracker.get_summary()
    print(f"\nFrames processed : {frame_count}")
    print(f"Average FPS      : {frame_count / (time.time() - start):.1f}")
    print(f"Violations logged: {summary['total_logged']}")


def find_model(model_arg):
    """Find model weights - check user path, then models/, then Kaggle weights."""
    p = Path(model_arg)
    if p.exists():
        return str(p)
    if (MODEL_DIR / "best.pt").exists():
        return str(MODEL_DIR / "best.pt")
    if PRETRAINED_WEIGHTS.exists():
        print(f"Using Kaggle pretrained weights: {PRETRAINED_WEIGHTS}")
        return str(PRETRAINED_WEIGHTS)
    return model_arg


def main():
    parser = argparse.ArgumentParser(
        description="Construction Safety Monitor"
    )
    parser.add_argument("--source", required=True,
        help="Image, video, folder path, or 0 for webcam")
    parser.add_argument("--model",
        default=str(MODEL_DIR / "best.pt"),
        help="Path to model weights")
    parser.add_argument("--output", type=Path,
        default=OUTPUT_DIR)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    model_path = find_model(args.model)
    detector = SafetyDetector(model_path, ModelConfig(
        confidence_threshold=args.confidence
    ))
    rule_engine = SafetyRuleEngine()
    annotator = FrameAnnotator()

    source = args.source
    if source.isdigit() or Path(source).suffix in (".mp4", ".avi", ".mov"):
        run_video(source, detector, rule_engine, annotator, args.output)
    else:
        run_image(source, detector, rule_engine, annotator, args.output)


if __name__ == "__main__":
    main()
