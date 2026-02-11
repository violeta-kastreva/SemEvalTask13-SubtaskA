"""
Pipeline for AI vs Human code detection.
Orchestrates language detection and AI detection models.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from .adapters import (
    LangModel,
    DetectorModel,
    SimpleLangModel,
    TrainedDetectorModel,
    DummyDetectorModel,
    LangResult,
    DetectResult,
)


@dataclass
class Output:
    """Output from the detection pipeline."""
    lang_guess: str
    lang_used: str
    label: str
    proba_ai: float
    features: Dict[str, float]
    lang_meta: Dict[str, Any]
    detect_meta: Dict[str, Any]


def load_models() -> Tuple[LangModel, DetectorModel]:
    """
    Load the language and detector models.
    
    Tries to load the trained model first, falls back to heuristic model.
    
    Returns:
        Tuple of (LangModel, DetectorModel)
    """
    # Language model
    lang_model = SimpleLangModel()
    
    # Detector model - try trained model first
    det_model = None
    
    # Check for environment variable override
    module_name = os.getenv("APP_MODELS_MODULE", "").strip()
    if module_name:
        try:
            m = __import__(module_name, fromlist=["load_lang_model", "load_detector_model"])
            if hasattr(m, "load_lang_model"):
                lang_model = m.load_lang_model()
            if hasattr(m, "load_detector_model"):
                det_model = m.load_detector_model()
        except ImportError:
            pass
    
    # Try to load trained model
    if det_model is None:
        possible_paths = [
            Path(__file__).parent.parent.parent / "models" / "ai_detector.joblib",
            Path("models/ai_detector.joblib"),
            Path("../models/ai_detector.joblib"),
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    det_model = TrainedDetectorModel(str(path))
                    break
                except Exception as e:
                    print(f"Warning: Could not load model from {path}: {e}")
    
    # Fallback to dummy model
    if det_model is None:
        print("Warning: Using heuristic detector (no trained model found)")
        det_model = DummyDetectorModel()
    
    return lang_model, det_model


def run(
    code: str,
    lang_override: Optional[str],
    lang_model: LangModel,
    det_model: DetectorModel
) -> Output:
    """
    Run the detection pipeline on a code snippet.
    
    Args:
        code: The code snippet to analyze
        lang_override: Optional language override (e.g., "python", "java")
        lang_model: Language detection model
        det_model: AI detection model
        
    Returns:
        Output object with prediction results
    """
    # Detect language
    lr: LangResult = lang_model.predict(code)
    lang_guess = (lr.lang or "unknown").lower()
    
    # Apply override if specified
    lang_used = lang_guess
    if lang_override and lang_override.lower() not in ("auto", ""):
        lang_used = lang_override.lower()
    
    # Run detection
    dr: DetectResult = det_model.predict(code, lang=lang_used)
    
    return Output(
        lang_guess=lang_guess,
        lang_used=lang_used,
        label=dr.label,
        proba_ai=float(dr.proba_ai),
        features=dr.features or {},
        lang_meta=lr.meta or {},
        detect_meta=dr.meta or {},
    )
