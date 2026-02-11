"""
Model adapters for AI vs Human code detection.
Provides interfaces for language detection and AI detection models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import joblib

from .features import extract_features, features_to_array, detect_language, FEATURE_NAMES


@dataclass
class LangResult:
    lang: str
    meta: Dict[str, Any]


@dataclass
class DetectResult:
    proba_ai: float
    label: str
    features: Dict[str, float]
    meta: Dict[str, Any]


class LangModel:
    """Base class for language detection models."""
    
    def predict(self, code: str) -> LangResult:
        raise NotImplementedError


class DetectorModel:
    """Base class for AI detection models."""
    
    def predict(self, code: str, lang: Optional[str] = None) -> DetectResult:
        raise NotImplementedError


class SimpleLangModel(LangModel):
    """Simple rule-based language detector."""
    
    def predict(self, code: str) -> LangResult:
        lang = detect_language(code)
        return LangResult(lang=lang, meta={"method": "rule-based"})


class TrainedDetectorModel(DetectorModel):
    """
    AI detector using the trained LogReg model with 10 features.
    
    Features used:
    - verb_ratio_comments: Verbs in comments (higher in human code)
    - text_like_ratio: Natural language content (higher in human code)
    - comments_code_like_ratio_to_total: Comments that look like code
    - comments_text_like_ratio_to_total: Comments that are explanatory
    - comments_code_like_ratio_comments: Proportion of code-like comments
    - comments_text_like_ratio_comments: Proportion of text comments
    - comment_ratio: Overall comment density
    - bucket_large/medium/small: Code size indicators
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Load the trained model bundle."""
        if model_path is None:
            # Try common paths
            possible_paths = [
                Path(__file__).parent.parent.parent / "models" / "ai_detector.joblib",
                Path("models/ai_detector.joblib"),
                Path("../models/ai_detector.joblib"),
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                "Could not find ai_detector.joblib. "
                "Run the model training cell in models_with_features.ipynb first."
            )
        
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.imputer = bundle['imputer']
        self.feature_names = bundle['features']
        self.test_f1 = bundle.get('test_f1_macro', 0.0)
    
    def predict(self, code: str, lang: Optional[str] = None) -> DetectResult:
        """
        Predict whether code is AI-generated or human-written.
        
        Returns:
            DetectResult with probability, label, features, and metadata
        """
        # Extract features
        features = extract_features(code)
        
        # Convert to array in correct order
        X = features_to_array(features)
        
        # Preprocess
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        
        # Predict
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Class 1 = AI-generated (assuming label encoding)
        proba_ai = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        # Label with threshold
        label = "AI-generated" if proba_ai >= 0.5 else "Human-written"
        
        # Get feature importances (coefficients for LogReg)
        coefficients = {}
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_[0]
            for i, name in enumerate(self.feature_names):
                coefficients[name] = float(coefs[i])
        
        meta = {
            "lang_used": lang or detect_language(code),
            "model_type": "LogisticRegression",
            "model_f1": self.test_f1,
            "feature_coefficients": coefficients,
            "scaled_features": {
                name: float(X_scaled[0][i]) 
                for i, name in enumerate(self.feature_names)
            },
        }
        
        return DetectResult(
            proba_ai=proba_ai,
            label=label,
            features=features,
            meta=meta,
        )


# Backward compatibility aliases
DummyLangModel = SimpleLangModel


class DummyDetectorModel(DetectorModel):
    """Fallback detector using simple heuristics (no model file needed)."""
    
    def predict(self, code: str, lang: Optional[str] = None) -> DetectResult:
        features = extract_features(code)
        
        # Simple heuristic based on key features
        # Higher comment ratios and text content suggest human code
        human_score = (
            features['comment_ratio'] * 0.3 +
            features['verb_ratio_comments'] * 0.2 +
            features['comments_text_like_ratio_to_total'] * 0.3 +
            features['text_like_ratio'] * 0.2
        )
        
        proba_ai = max(0.01, min(0.99, 0.65 - human_score))
        label = "AI-generated" if proba_ai >= 0.5 else "Human-written"
        
        return DetectResult(
            proba_ai=proba_ai,
            label=label,
            features=features,
            meta={"lang_used": lang, "method": "heuristic"},
        )
