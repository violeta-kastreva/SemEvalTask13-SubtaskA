"""
Feature extraction for AI vs Human code detection.
Extracts the 10 features used by the trained LogReg model.
"""

import re
import os
import joblib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np

# Lazy-loaded globals
_nlp = None
_parsers = None
_text_vs_code_clf = None

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS AND PATTERNS
# ═══════════════════════════════════════════════════════════════════════

COMMENT_RE = re.compile(r"(#|//|/\*|\*/)")

# Feature names expected by the model
FEATURE_NAMES = [
    'verb_ratio_comments',
    'text_like_ratio',
    'comments_code_like_ratio_to_total',
    'comments_text_like_ratio_to_total',
    'comments_code_like_ratio_comments',
    'comments_text_like_ratio_comments',
    'comment_ratio',
    'bucket_large',
    'bucket_medium',
    'bucket_small',
]


# ═══════════════════════════════════════════════════════════════════════
# LAZY LOADERS
# ═══════════════════════════════════════════════════════════════════════

def _load_spacy(model: str = "en_core_web_sm"):
    """Load spaCy model for NLP analysis."""
    global _nlp
    if _nlp is not None:
        return _nlp
    
    try:
        import spacy
        _nlp = spacy.load(model)
    except Exception:
        # spacy not available or model not installed - use fallback
        _nlp = "unavailable"
    
    return _nlp


def _load_parsers():
    """Load tree-sitter parsers for multiple languages."""
    global _parsers
    if _parsers is not None:
        return _parsers
    
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_java
        import tree_sitter_c_sharp
        import tree_sitter_php
        import tree_sitter_cpp
        import tree_sitter_go
        
        LANGUAGE_MODULES = {
            "python": tree_sitter_python,
            "javascript": tree_sitter_javascript,
            "java": tree_sitter_java,
            "c_sharp": tree_sitter_c_sharp,
            "php": tree_sitter_php,
            "cpp": tree_sitter_cpp,
            "go": tree_sitter_go,
        }
        
        _parsers = {}
        for name, module in LANGUAGE_MODULES.items():
            parser = Parser()
            if name == "php":
                parser.language = Language(module.language_php())
            else:
                parser.language = Language(module.language())
            _parsers[name] = parser
        
    except ImportError:
        # Fallback: no tree-sitter available
        _parsers = {}
    
    return _parsers


def _load_text_vs_code_classifier():
    """Load the text vs code line classifier."""
    global _text_vs_code_clf
    if _text_vs_code_clf is not None:
        return _text_vs_code_clf
    
    # Try to find the model file
    possible_paths = [
        Path(__file__).parent.parent.parent / "models" / "textvscode_classifier.joblib",
        Path("models/textvscode_classifier.joblib"),
        Path("../models/textvscode_classifier.joblib"),
    ]
    
    for path in possible_paths:
        if path.exists():
            _text_vs_code_clf = joblib.load(path)
            return _text_vs_code_clf
    
    # Fallback: return None and use simple heuristics
    return None


# ═══════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_language(code: str) -> str:
    """Simple rule-based language detection."""
    s = code.lower()
    
    # Python indicators
    if "def " in s or "import " in s or "class " in s and ":" in s:
        if "self" in s or "print(" in s or "__init__" in s:
            return "python"
    
    # JavaScript/TypeScript
    if "const " in s or "let " in s or "=>" in s or "console.log" in s:
        return "javascript"
    
    # Java
    if "public class" in s or "public static void main" in s:
        return "java"
    
    # C/C++
    if "#include" in s:
        if "std::" in s or "cout" in s or "iostream" in s:
            return "cpp"
        return "cpp"
    
    # Go
    if "package main" in s or "func main" in s or "fmt.Println" in s:
        return "go"
    
    # PHP
    if "<?php" in s or "$_" in s:
        return "php"
    
    # C#
    if "using System" in s or "namespace " in s:
        return "c_sharp"
    
    return "python"  # default


# ═══════════════════════════════════════════════════════════════════════
# AST HELPERS
# ═══════════════════════════════════════════════════════════════════════

def walk(root):
    """Walk the AST tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def parse_code(code_str: str) -> Tuple[str, Any]:
    """Parse code with tree-sitter."""
    parsers = _load_parsers()
    if not parsers:
        return detect_language(code_str), None
    
    lang = detect_language(code_str)
    
    # Map detected language to parser key
    lang_map = {
        "c": "cpp",
        "c++": "cpp",
        "c#": "c_sharp",
    }
    parser_key = lang_map.get(lang, lang)
    
    parser = parsers.get(parser_key)
    if parser:
        code_bytes = code_str.encode("utf8")
        tree = parser.parse(code_bytes)
        return lang, tree
    
    return lang, None


# ═══════════════════════════════════════════════════════════════════════
# COMMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def extract_comment_text(comment_line: str) -> str:
    """Clean comment markers from a comment line."""
    comment_line = comment_line.strip()
    
    # Single-line comments
    if comment_line.startswith("#"):
        return comment_line[1:].strip()
    if comment_line.startswith("//"):
        return comment_line[2:].strip()
    
    # Multi-line comments
    if comment_line.startswith("/*") and comment_line.endswith("*/"):
        return comment_line[2:-2].strip()
    if comment_line.startswith("/*"):
        return comment_line[2:].strip()
    if comment_line.endswith("*/"):
        return comment_line[:-2].strip()
    if comment_line.startswith("*"):
        return comment_line[1:].strip()
    
    return comment_line


def compute_verb_ratio(texts: List[str]) -> float:
    """Compute the ratio of verbs in the given texts using spaCy."""
    if not texts:
        return 0.0
    
    nlp = _load_spacy()
    
    # Fallback if spacy not available
    if nlp is None or nlp == "unavailable":
        # Simple heuristic: count common verb patterns
        combined = " ".join(texts).lower()
        words = combined.split()
        if not words:
            return 0.0
        
        # Common verb indicators
        verb_patterns = ['is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'do', 'does', 'did',
                        'will', 'would', 'could', 'should', 'may', 'might',
                        'must', 'shall', 'can']
        verb_count = sum(1 for w in words if w in verb_patterns or w.endswith('ing') or w.endswith('ed'))
        return min(1.0, verb_count / len(words))
    
    combined = "\n".join(texts)
    doc = nlp(combined)
    
    word_tokens = [t for t in doc if t.is_alpha]
    if not word_tokens:
        return 0.0
    
    verb_tokens = [t for t in word_tokens if t.pos_ in {"VERB", "AUX"}]
    return len(verb_tokens) / len(word_tokens)


def get_comment_ratio(code: str) -> float:
    """Compute the ratio of comment lines to total lines."""
    lines = code.splitlines()
    loc = len(lines)
    if loc == 0:
        return 0.0
    
    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            comment_lines += 1
        elif "/*" in stripped or "*/" in stripped:
            comment_lines += 1
    
    return comment_lines / loc


# ═══════════════════════════════════════════════════════════════════════
# TEXT VS CODE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def classify_lines_as_code(lines: List[str]) -> List[bool]:
    """
    Classify each line as code (True) or text (False).
    Uses the trained classifier if available, otherwise simple heuristics.
    """
    if not lines:
        return []
    
    clf = _load_text_vs_code_classifier()
    
    if clf is not None:
        try:
            predictions = clf.predict(lines)
            return [bool(p) for p in predictions]
        except Exception:
            pass
    
    # Fallback: simple heuristics
    results = []
    for line in lines:
        s = line.strip()
        # Code indicators
        is_code = (
            "=" in s or
            "(" in s or
            "{" in s or
            ";" in s or
            s.startswith("if ") or
            s.startswith("for ") or
            s.startswith("def ") or
            s.startswith("class ") or
            s.startswith("return ") or
            s.startswith("import ") or
            s.startswith("#include") or
            s.startswith("public ") or
            s.startswith("private ")
        )
        results.append(is_code)
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# FILE SIZE BUCKET
# ═══════════════════════════════════════════════════════════════════════

def file_size_bucket(total_lines: int) -> str:
    """Classify code into size buckets (matches training data thresholds)."""
    if total_lines < 20:
        return "small"
    elif total_lines <= 70:
        return "medium"
    else:
        return "large"


def bucket_to_onehot(bucket: str) -> Dict[str, int]:
    """Convert bucket to one-hot encoding."""
    return {
        "bucket_large": 1 if bucket == "large" else 0,
        "bucket_medium": 1 if bucket == "medium" else 0,
        "bucket_small": 1 if bucket == "small" else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_features(code: str) -> Dict[str, float]:
    """
    Extract all 10 features required by the AI detection model.
    
    Returns a dictionary with the following keys:
    - verb_ratio_comments: Ratio of verbs in comment text
    - text_like_ratio: Ratio of text-like lines to total lines
    - comments_code_like_ratio_to_total: Code-like comment lines / total lines
    - comments_text_like_ratio_to_total: Text-like comment lines / total lines
    - comments_code_like_ratio_comments: Code-like comment lines / total comments
    - comments_text_like_ratio_comments: Text-like comment lines / total comments
    - comment_ratio: Comment lines / total lines
    - bucket_large: 1 if code is large (>200 lines)
    - bucket_medium: 1 if code is medium (51-200 lines)
    - bucket_small: 1 if code is small (<=50 lines)
    """
    if not code or not code.strip():
        return {name: 0.0 for name in FEATURE_NAMES}
    
    # Basic stats
    lines = [line for line in code.splitlines() if line.strip()]
    total_lines = max(1, len(lines))
    
    # File size bucket
    bucket = file_size_bucket(total_lines)
    bucket_onehot = bucket_to_onehot(bucket)
    
    # Comment ratio
    comment_ratio = get_comment_ratio(code)
    
    # Parse code to extract comments
    lang, tree = parse_code(code)
    
    comments = []
    if tree is not None:
        for node in walk(tree.root_node):
            if "comment" in node.type and node.text:
                comment_text = node.text.decode("utf8")
                comments.append(comment_text)
    else:
        # Fallback: regex-based comment extraction
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                comments.append(stripped)
    
    # Extract clean comment content
    comments_content = []
    for comment in comments:
        extracted = extract_comment_text(comment)
        if extracted:
            comments_content.append(extracted)
    
    # Verb ratio in comments
    verb_ratio_comments = compute_verb_ratio(comments_content) if comments_content else 0.0
    
    # Text-like ratio for all lines
    line_classifications = classify_lines_as_code(lines)
    num_code_lines = sum(line_classifications)
    num_text_lines = len(lines) - num_code_lines
    text_like_ratio = num_text_lines / total_lines if total_lines > 0 else 0.0
    
    # Comment text/code classification
    if comments_content:
        comment_classifications = classify_lines_as_code(comments_content)
        num_code_comments = sum(comment_classifications)
        num_text_comments = len(comments_content) - num_code_comments
        
        comments_code_like_ratio_to_total = num_code_comments / total_lines
        comments_text_like_ratio_to_total = num_text_comments / total_lines
        comments_code_like_ratio_comments = num_code_comments / len(comments_content)
        comments_text_like_ratio_comments = num_text_comments / len(comments_content)
    else:
        comments_code_like_ratio_to_total = 0.0
        comments_text_like_ratio_to_total = 0.0
        comments_code_like_ratio_comments = 0.0
        comments_text_like_ratio_comments = 0.0
    
    return {
        'verb_ratio_comments': verb_ratio_comments,
        'text_like_ratio': text_like_ratio,
        'comments_code_like_ratio_to_total': comments_code_like_ratio_to_total,
        'comments_text_like_ratio_to_total': comments_text_like_ratio_to_total,
        'comments_code_like_ratio_comments': comments_code_like_ratio_comments,
        'comments_text_like_ratio_comments': comments_text_like_ratio_comments,
        'comment_ratio': comment_ratio,
        **bucket_onehot,
    }


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in correct order."""
    return np.array([[features[name] for name in FEATURE_NAMES]])
