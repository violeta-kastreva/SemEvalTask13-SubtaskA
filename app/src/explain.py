"""
Explanation generation for AI detection results.
Provides human-readable reasoning based on feature values.
"""

from typing import Any, Dict, List, Tuple


# Feature explanations and thresholds
# Derived from model coefficients:
#   comments_text_like_ratio_to_total: +2.38 (strongest AI signal)
#   text_like_ratio:                   +1.18 (strong AI signal)
#   comment_ratio:                     +0.49 (moderate AI signal)
#   verb_ratio_comments:               +0.07 (weak AI signal)
#   comments_code_like_ratio_to_total: -0.05 (very weak human signal)
FEATURE_EXPLANATIONS = {
    'comment_ratio': {
        'name': 'Comment Density',
        'high_threshold': 0.15,
        'low_threshold': 0.05,
        'high_human': False,  # High comment density pushes toward AI (coef +0.49)
        'explanation_high': 'Code has substantial comments ({:.1%}), common in AI-generated code which tends to be heavily commented',
        'explanation_low': 'Code has few comments ({:.1%}), more typical of human-written code',
        'explanation_mid': 'Comment density is moderate ({:.1%})',
    },
    'verb_ratio_comments': {
        'name': 'Verb Usage in Comments',
        'high_threshold': 0.15,
        'low_threshold': 0.05,
        'high_human': False,  # Slightly pushes toward AI (coef +0.07, weak)
        'explanation_high': 'Comments contain natural language with verbs ({:.1%}), slightly indicative of AI-generated documentation style',
        'explanation_low': 'Comments have low verb diversity ({:.1%}), slightly more typical of terse human comments',
        'explanation_mid': 'Verb usage in comments is average ({:.1%})',
    },
    'comments_text_like_ratio_to_total': {
        'name': 'Explanatory Comments',
        'high_threshold': 0.08,
        'low_threshold': 0.02,
        'high_human': False,  # Strongest AI signal (coef +2.38)
        'explanation_high': 'Code has explanatory text comments ({:.1%} of lines), a strong indicator of AI-generated code',
        'explanation_low': 'Few explanatory comments ({:.1%}), more typical of human-written code',
        'explanation_mid': 'Explanatory comment ratio is moderate ({:.1%})',
    },
    'comments_code_like_ratio_to_total': {
        'name': 'Code-like Comments',
        'high_threshold': 0.05,
        'low_threshold': 0.01,
        'high_human': True,  # Slightly pushes toward human (coef -0.05, very weak)
        'explanation_high': 'Contains code-like comments ({:.1%}), slightly more common in human-written code',
        'explanation_low': 'Few code-like comments ({:.1%})',
        'explanation_mid': 'Code-like comment ratio is typical ({:.1%})',
    },
    'text_like_ratio': {
        'name': 'Natural Language Content',
        'high_threshold': 0.20,
        'low_threshold': 0.05,
        'high_human': False,  # Strong AI signal (coef +1.18)
        'explanation_high': 'Contains significant natural language ({:.1%}), common in AI-generated code with verbose documentation',
        'explanation_low': 'Mostly pure code with little text ({:.1%}), more typical of human-written code',
        'explanation_mid': 'Natural language ratio is moderate ({:.1%})',
    },
}


def get_feature_explanation(name: str, value: float) -> Tuple[str, str]:
    """
    Get an explanation for a single feature value.
    
    Returns:
        Tuple of (explanation_text, signal) where signal is 'human', 'ai', or 'neutral'
    """
    config = FEATURE_EXPLANATIONS.get(name)
    if not config:
        return f"{name}: {value:.3f}", 'neutral'
    
    if value >= config['high_threshold']:
        explanation = config['explanation_high'].format(value)
        signal = 'human' if config['high_human'] else 'ai'
    elif value <= config['low_threshold']:
        explanation = config['explanation_low'].format(value)
        signal = 'ai' if config['high_human'] else 'human'
    else:
        explanation = config['explanation_mid'].format(value)
        signal = 'neutral'
    
    return explanation, signal


def build_reasoning(features: Dict[str, float], meta: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Build human-readable reasoning from features and model metadata.
    
    Args:
        features: Dictionary of feature values
        meta: Model metadata including coefficients
        
    Returns:
        Tuple of (explanation_lines, feature_rows)
    """
    lines: List[str] = []
    rows: List[Tuple[str, float]] = []
    
    if not features:
        return (["No feature evidence available."], [])
    
    # Collect all features for the table
    for name, value in sorted(features.items()):
        rows.append((name, float(value)))
    
    # Get key feature explanations
    human_signals = []
    ai_signals = []
    
    # Most important features for explanation
    key_features = [
        'comment_ratio',
        'verb_ratio_comments', 
        'comments_text_like_ratio_to_total',
        'text_like_ratio',
    ]
    
    for feat in key_features:
        if feat in features:
            explanation, signal = get_feature_explanation(feat, features[feat])
            if signal == 'human':
                human_signals.append(explanation)
            elif signal == 'ai':
                ai_signals.append(explanation)
    
    # Build explanation lines
    if human_signals:
        lines.append("**Human-like indicators:**")
        for sig in human_signals[:3]:
            lines.append(f"  - {sig}")
    
    if ai_signals:
        lines.append("**AI-like indicators:**")
        for sig in ai_signals[:3]:
            lines.append(f"  - {sig}")
    
    # Code size context
    if features.get('bucket_small', 0):
        lines.append("Code is small (<=50 lines)")
    elif features.get('bucket_medium', 0):
        lines.append("Code is medium-sized (51-200 lines)")
    elif features.get('bucket_large', 0):
        lines.append("Code is large (>200 lines)")
    
    # Add coefficient-based insights if available
    if 'feature_coefficients' in meta:
        coefs = meta['feature_coefficients']
        scaled = meta.get('scaled_features', {})
        
        # Find most influential features for this prediction
        contributions = []
        for name, coef in coefs.items():
            if name in scaled:
                contribution = coef * scaled[name]
                contributions.append((name, contribution, coef))
        
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if contributions:
            lines.append("")
            lines.append("**Most influential features:**")
            for name, contrib, coef in contributions[:3]:
                direction = "toward AI" if contrib > 0 else "toward Human"
                lines.append(f"  - {name}: {direction} (contribution: {contrib:.3f})")
    
    if not lines:
        lines.append("Analysis based on code structure and comment patterns")
    
    return (lines, rows)


def get_confidence_description(proba_ai: float) -> str:
    """Get a human-readable confidence description."""
    if proba_ai >= 0.8:
        return "High confidence AI-generated"
    elif proba_ai >= 0.6:
        return "Likely AI-generated"
    elif proba_ai >= 0.4:
        return "Uncertain (borderline)"
    elif proba_ai >= 0.2:
        return "Likely human-written"
    else:
        return "High confidence human-written"
