"""Streamlit web demo for Construction Safety Monitor.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import ModelConfig, MODEL_DIR, SAMPLE_DIR, PRETRAINED_WEIGHTS
from src.detector import SafetyDetector
from src.safety_rules import SafetyRuleEngine
from src.annotator import FrameAnnotator
from src.utils import pair_workers_with_ppe


st.set_page_config(
    page_title="Construction Safety Monitor",
    page_icon="🏗️",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str, confidence: float):
    """Load model once and cache it - avoids reloading on every interaction."""
    return SafetyDetector(model_path, ModelConfig(
        confidence_threshold=confidence
    ))


def find_model() -> str:
    """Find the best available model weights."""
    if (MODEL_DIR / "best.pt").exists():
        return str(MODEL_DIR / "best.pt")
    if PRETRAINED_WEIGHTS.exists():
        return str(PRETRAINED_WEIGHTS)
    return str(MODEL_DIR / "best.pt")


def main():
    st.title("Construction Safety Monitor")
    st.markdown(
        "AI-powered PPE compliance detection. "
        "Upload a construction site image to analyse it."
    )

    # --- Sidebar ---
    st.sidebar.header("Settings")

    model_path = st.sidebar.text_input("Model Path", value=find_model())

    confidence = st.sidebar.slider(
        "Confidence Threshold", 0.1, 0.95, 0.5, 0.05,
        help="Only show detections the model is this confident about"
    )

    required_ppe = st.sidebar.multiselect(
        "Required PPE",
        options=["Hardhat", "Safety Vest", "Mask"],
        default=["Hardhat", "Safety Vest"],
        help="Which PPE items are enforced on this site"
    )

    show_confidence = st.sidebar.checkbox(
        "Show Confidence Scores", value=True
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Input")

    uploaded = st.sidebar.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    # Sample images from data/sample_images/
    samples = list(SAMPLE_DIR.glob("*")) if SAMPLE_DIR.exists() else []
    selected = "None"
    if samples:
        selected = st.sidebar.selectbox(
            "Or use a sample image",
            ["None"] + [p.name for p in samples],
        )


    # Load model
    try:
        detector = load_model(model_path, confidence)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info(
            "Make sure the Kaggle dataset is in data/dataset/ — "
            "the pretrained weights come with it."
        )
        return

    rule_engine = SafetyRuleEngine(required_ppe=required_ppe)
    annotator = FrameAnnotator()

    # Get image
    image = None
    if uploaded:
        image = Image.open(uploaded)
    elif selected != "None":
        image = Image.open(SAMPLE_DIR / selected)

    if image is None:
        st.info("Upload an image or pick a sample from the sidebar.")
        _show_rules_table()
        return


    # Convert PIL → OpenCV BGR
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Analysing..."):
        detections  = detector.detect(frame)
        pairs       = pair_workers_with_ppe(detections)
        results     = rule_engine.check_all(pairs)
        annotated   = annotator.annotate_frame(
            frame, results, show_confidence=show_confidence
        )

    # Side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Analysis")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    # Metrics row
    summary = rule_engine.get_summary(results)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workers",    summary["total_workers"])
    c2.metric("Compliant",  summary["compliant_workers"])
    c3.metric("Violations", summary["non_compliant_workers"])
    c4.metric("Compliance Rate",
              f"{summary['compliance_rate'] * 100:.0f}%")


    # Violation details
    if summary["non_compliant_workers"] > 0:
        st.subheader("Violations")
        for result in results:
            if result.is_compliant:
                continue
            cx, cy = result.worker.center
            with st.expander(
                f"Worker at ({cx:.0f}, {cy:.0f})", expanded=True
            ):
                for v in result.violations:
                    icon = {
                        "critical": "🔴",
                        "high":     "🟠",
                        "medium":   "🟡",
                        "low":      "🟢",
                    }.get(v.severity.value, "⚪")
                    st.markdown(
                        f"{icon} **{v.rule}** — "
                        f"Missing: `{v.missing_ppe}` | "
                        f"Severity: **{v.severity.value}** | "
                        f"Confidence: {v.confidence:.2f}"
                    )

    # Full text report
    with st.expander("Full Violation Report"):
        st.code(annotator.create_violation_report(frame, results))


def _show_rules_table():
    """Show safety rules when no image is loaded."""
    st.markdown("---")
    st.subheader("Enforced Safety Rules")
    st.markdown("""
| Rule | Description | Severity |
|------|-------------|----------|
| R1 - Hard Hat Required | All workers must wear a hard hat | 🔴 Critical |
| R2 - Safety Vest Required | All workers must wear a high-vis vest | 🟠 High |
| R3 - Mask Required | Face mask (optional, enable in sidebar) | 🟡 Medium |
    """)


if __name__ == "__main__":
    main()
