"""
AI vs Human Code Detector - Streamlit Application

A tool to analyze code snippets and predict whether they were written by
an AI model or a human programmer.
"""

import time
import streamlit as st
import pandas as pd

from src.pipeline import load_models, run
from src.explain import build_reasoning, get_confidence_description

# Page configuration
st.set_page_config(
    page_title="AI vs Human Code Detector",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def get_models():
    """Load and cache the detection models."""
    return load_models()


def should_run_now(code: str, auto: bool, throttle_ms: int, clicked: bool) -> bool:
    """Determine if analysis should run based on auto-run settings."""
    if clicked:
        return True
    if not auto:
        return False
    if not code.strip():
        return False

    now = time.time()
    last_t = st.session_state.get("last_run_t", 0.0)
    last_hash = st.session_state.get("last_code_hash", "")

    h = str(hash(code))
    if h == last_hash:
        return False

    if (now - last_t) * 1000.0 < throttle_ms:
        return False

    st.session_state["last_run_t"] = now
    st.session_state["last_code_hash"] = h
    return True


# Main title
st.title("AI vs Human Code Detector")
st.markdown("""
Analyze code snippets to determine if they were likely written by an AI or a human.
The model uses 10 features based on comment patterns, code structure, and linguistic analysis.
""")

# Load models
try:
    lang_model, det_model = get_models()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {e}")
    model_loaded = False

# Layout
left, right = st.columns([2, 1], gap="large")

with left:
    code = st.text_area(
        "Paste your code snippet here:",
        height=440,
        key="code",
        placeholder="# Enter code to analyze...\ndef hello_world():\n    print('Hello, World!')"
    )

with right:
    st.subheader("Settings")
    
    lang_override = st.selectbox(
        "Language override",
        ["Auto", "Python", "JavaScript", "Java", "C/C++", "Go", "PHP", "C#"],
        index=0,
        help="Override automatic language detection"
    )
    
    auto = st.checkbox("Auto-run on paste", value=True)
    throttle_ms = st.slider("Throttle (ms)", 100, 2000, 500, 50)
    
    st.markdown("---")
    clicked = st.button("Analyze", type="primary", use_container_width=True)

# Check if we should run
do_run = should_run_now(code, auto, throttle_ms, clicked)

if not code.strip():
    st.info("Paste a code snippet to analyze. The detector works best with code that has comments.")
    st.stop()

if not model_loaded:
    st.error("Cannot run analysis - models not loaded.")
    st.stop()

if do_run:
    with st.spinner("Analyzing code..."):
        out = run(code, lang_override, lang_model, det_model)

    # Results section
    st.markdown("---")
    st.header("Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Color-coded prediction
        color = "🔴" if out.label == "AI-generated" else "🟢"
        st.metric("Prediction", f"{color} {out.label}")
    
    with col2:
        st.metric("P(AI-generated)", f"{out.proba_ai:.1%}")
    
    with col3:
        st.metric("Confidence", get_confidence_description(out.proba_ai))
    
    with col4:
        st.metric("Detected Language", out.lang_guess.title())

    # Probability bar
    st.markdown("### Probability Distribution")
    prob_col1, prob_col2 = st.columns([out.proba_ai, 1 - out.proba_ai] if out.proba_ai > 0.01 else [1, 99])
    with prob_col1:
        st.progress(out.proba_ai, text=f"AI: {out.proba_ai:.1%}")
    
    st.markdown(f"**Human probability:** {1 - out.proba_ai:.1%}")

    # Explanation section
    st.markdown("### Analysis Explanation")
    lines, rows = build_reasoning(out.features, out.detect_meta)
    
    for ln in lines:
        if ln.startswith("**"):
            st.markdown(ln)
        else:
            st.write(ln)

    # Feature details in expandable sections
    with st.expander("Feature Values", expanded=False):
        # Create a nice table
        feature_data = []
        for name, value in sorted(out.features.items()):
            feature_data.append({
                "Feature": name,
                "Value": f"{value:.4f}" if isinstance(value, float) else str(value),
            })
        
        df_features = pd.DataFrame(feature_data)
        st.dataframe(df_features, width=None, hide_index=True)

    with st.expander("Model Details", expanded=False):
        st.json({
            "language_detected": out.lang_guess,
            "language_used": out.lang_used,
            "model_metadata": out.detect_meta,
        })

else:
    st.info("Ready to analyze. Type more code or click 'Analyze'.")

# Footer
st.markdown("---")
st.markdown("""
<small>
This detector uses a Logistic Regression model trained on code features including:
comment density, verb usage in comments, text-like content ratios, and code size.
Model F1-score: ~0.62 on test data. Results should be interpreted as probabilistic estimates.
</small>
""", unsafe_allow_html=True)
