"""
Digital Twin Dashboard - Main Application
Streamlit-based interface for wound analysis and visualization
"""

import os
# Fix OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import io
import json
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import dashboard_config, UNIFIED_DIR, OUTPUT_DIR, DEVICE
from pipeline.inference import InferencePipeline, PipelineResult
from pipeline.digital_twin import DigitalTwin
from models.simulation.trajectory import TrajectoryGenerator


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=dashboard_config.page_title,
    page_icon=dashboard_config.page_icon,
    layout=dashboard_config.layout,
    initial_sidebar_state=dashboard_config.initial_sidebar_state
)

# Custom CSS - Modern Elegant Design
st.markdown("""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --critical: #a855f7;
        --bg-dark: #0f172a;
        --bg-card: rgba(255, 255, 255, 0.05);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    .stApp > header {
        background: transparent;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }
    
    p, span, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    /* ===== RISK LEVEL BADGES ===== */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%);
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        animation: pulse 2s infinite;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: white;
        font-family: 'Outfit', sans-serif !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(5px);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        animation: glow 3s ease-in-out infinite;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.8);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* ===== IMAGES ===== */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
    }
    
    /* ===== SELECTBOX & INPUTS ===== */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #f8fafc;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    }
    
    /* ===== CHARTS ===== */
    [data-testid="stPlotlyChart"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.5) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        color: #e2e8f0;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div > div {
        border-top-color: #6366f1 !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.8);
    }
    
    /* ===== CAPTION ===== */
    .stCaption {
        color: #64748b !important;
    }
    
    /* ===== DATA FRAME ===== */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    /* ===== CODE BLOCKS ===== */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "result" not in st.session_state:
    st.session_state.result = None
if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = None
if "trajectory_generator" not in st.session_state:
    st.session_state.trajectory_generator = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_pipeline():
    """Load and cache the inference pipeline"""
    pipeline = InferencePipeline()
    pipeline.load()
    return pipeline


@st.cache_resource
def load_trajectory_generator():
    """Load trajectory generator"""
    gen = TrajectoryGenerator()
    gen.load()
    return gen


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        "low": "#4CAF50",
        "moderate": "#FF9800",
        "high": "#f44336",
        "critical": "#9C27B0"
    }
    return colors.get(risk_level, "#666")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/hospital.png", width=60)
    st.title("🏥 Digital Twin")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🔬 Analyze Wound", "📊 Dashboard", "🎬 Simulate Healing", "📜 History"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        patient_id = st.text_input("Patient ID", value="default_patient")
        wound_id = st.text_input("Wound ID", value="wound_1")
        
        skip_detection = st.checkbox("Skip Detection (use full image)", value=False)
        
        st.markdown("---")
        
        if st.button("🔄 Reset Pipeline"):
            st.session_state.pipeline = None
            st.session_state.result = None
            st.rerun()
    
    st.markdown("---")
    st.caption(f"Device: {DEVICE}")
    st.caption("v1.0.0 | Made with ❤️")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Page: Analyze Wound
if page == "🔬 Analyze Wound":
    st.markdown('<h1 class="main-header">🔬 Wound Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a wound image for AI-powered analysis</p>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Wound Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the wound for analysis"
    )
    
    # Sample images - initialize session state
    if "sample_images" not in st.session_state:
        st.session_state.sample_images = []
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None
    
    # Load Sample button
    col_upload, col_sample = st.columns([3, 1])
    with col_sample:
        if st.button("📁 Load Sample"):
            sample_dir = UNIFIED_DIR / "train"
            if sample_dir.exists():
                # Get sample images from different subdirectories for variety
                all_samples = list(sample_dir.rglob("*.jpg"))[:50]
                if all_samples:
                    # Pick 5 random samples
                    import random
                    st.session_state.sample_images = random.sample(all_samples, min(5, len(all_samples)))
                    st.session_state.selected_sample = None
                else:
                    st.error("No sample images found in train directory")
            else:
                st.error(f"Sample directory not found: {sample_dir}")
    
    # Display sample images if loaded
    if st.session_state.sample_images:
        st.subheader("📷 Sample Images")
        st.caption("Click on a sample to use it for analysis")
        
        sample_cols = st.columns(len(st.session_state.sample_images))
        for i, (col, sample_path) in enumerate(zip(sample_cols, st.session_state.sample_images)):
            with col:
                # Load and display thumbnail
                sample_img = cv2.imread(str(sample_path))
                if sample_img is not None:
                    sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
                    st.image(sample_img_rgb, use_container_width=True)
                    if st.button(f"Use #{i+1}", key=f"sample_{i}"):
                        st.session_state.selected_sample = sample_path
                        st.rerun()
    
    # Determine which image to use (uploaded or sample)
    image = None
    image_source = None
    
    if uploaded_file is not None:
        # Use uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_source = "uploaded"
    elif st.session_state.selected_sample is not None:
        # Use selected sample
        image = cv2.imread(str(st.session_state.selected_sample))
        image_source = f"sample: {st.session_state.selected_sample.name}"
    
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(numpy_to_pil(image), use_container_width=True)
            if image_source:
                st.caption(f"Source: {image_source}")
        
        # Analyze button
        if st.button("🚀 Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Loading AI models..."):
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = load_pipeline()
            
            with st.spinner("Analyzing wound..."):
                progress = st.progress(0)
                
                # Run analysis
                result = st.session_state.pipeline.analyze(image, skip_detection=skip_detection)
                st.session_state.result = result
                
                progress.progress(100)
            
            # Update digital twin - but skip background/non-wound images
            is_wound = True
            if result.classification and result.classification.wound_type in ["background", "normal"]:
                is_wound = False
                st.warning("⚠️ Image classified as non-wound (background/normal skin). Not added to digital twin.")
            
            if is_wound:
                if st.session_state.digital_twin is None:
                    st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)
                st.session_state.digital_twin.update(result)
                st.success("✅ Analysis complete! Added to digital twin.")
            else:
                st.success("✅ Analysis complete!")
            st.rerun()
        
        # Show results if available
        if st.session_state.result is not None:
            result = st.session_state.result
            
            with col2:
                st.subheader("🎯 Detection & ROI")
                if result.roi_image is not None:
                    st.image(numpy_to_pil(result.roi_image), use_container_width=True)
            
            st.markdown("---")
            
            # Metrics row
            st.subheader("📊 Key Metrics")
            
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                if result.classification:
                    st.metric(
                        "Wound Type",
                        result.classification.wound_type.title(),
                        f"{result.classification.wound_type_confidence:.0%} conf"
                    )
            
            with m2:
                if result.classification:
                    st.metric(
                        "Severity",
                        result.classification.severity.title()
                    )
            
            with m3:
                if result.risk:
                    st.metric(
                        "Risk Score",
                        f"{result.risk.risk_score:.2f}",
                        delta=None
                    )
            
            with m4:
                if result.risk:
                    color = get_risk_color(result.risk.risk_level)
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <small>Risk Level</small><br>
                        <strong>{result.risk.risk_level.upper()}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎨 Tissue Segmentation")
                if result.segmentation and result.segmentation.overlay is not None:
                    st.image(numpy_to_pil(result.segmentation.overlay), use_container_width=True)
                    
                    # Tissue composition chart
                    if result.segmentation.class_percentages:
                        fig = px.pie(
                            names=list(result.segmentation.class_percentages.keys()),
                            values=list(result.segmentation.class_percentages.values()),
                            title="Tissue Composition",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📐 Depth Estimation")
                if result.depth is not None:
                    # Use wound-focused overlay for better visualization
                    if result.depth.depth_wound_overlay is not None:
                        st.image(numpy_to_pil(result.depth.depth_wound_overlay), use_container_width=True)
                        st.caption(" 🔵 Blue = Deeper cavity | 🔴 Red = Shallower")
                    else:
                        st.image(numpy_to_pil(result.depth.depth_colored), use_container_width=True)
                    
                    # Depth metrics
                    if result.volume:
                        st.markdown("**Volume Metrics:**")
                        st.write(f"- Total Volume: {result.volume.total_volume:.2f}")
                        st.write(f"- Surface Area: {result.volume.surface_area:.0f} px")
                        st.write(f"- Mean Depth: {result.volume.mean_depth:.4f}")
                    
                    # Qualitative depth assessment
                    if result.depth and hasattr(result.depth, 'depth_category'):
                        st.markdown("**Depth Assessment:**")
                        cat = result.depth.depth_category
                        cavity = "✅ Yes" if result.depth.has_central_cavity else "❌ No"
                        st.write(f"- Category: **{cat.upper()}**")
                        st.write(f"- Central cavity: {cavity}")
                        if result.depth.depth_description:
                            st.caption(result.depth.depth_description)
            
            st.markdown("---")
            
            # Grad-CAM Explainability Section
            st.subheader("🔍 Grad-CAM Explainability")
            st.caption("Visual explanation of what the AI focused on for classification")
            
            gradcam_col1, gradcam_col2 = st.columns(2)
            
            with gradcam_col1:
                st.markdown("**Wound Type Focus**")
                try:
                    if st.session_state.pipeline._classifier is not None and result.roi_image is not None:
                        _, gradcam_overlay_type = st.session_state.pipeline._classifier.get_gradcam(
                            result.roi_image, task="wound_type"
                        )
                        st.image(numpy_to_pil(gradcam_overlay_type), use_container_width=True)
                        if result.classification:
                            st.caption(f"Predicted: {result.classification.wound_type.title()} ({result.classification.wound_type_confidence:.0%})")
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM for wound type: {str(e)}")
            
            with gradcam_col2:
                st.markdown("**Severity Focus**")
                try:
                    if st.session_state.pipeline._classifier is not None and result.roi_image is not None:
                        _, gradcam_overlay_severity = st.session_state.pipeline._classifier.get_gradcam(
                            result.roi_image, task="severity"
                        )
                        st.image(numpy_to_pil(gradcam_overlay_severity), use_container_width=True)
                        if result.classification:
                            st.caption(f"Predicted: {result.classification.severity.title()} ({result.classification.severity_confidence:.0%})")
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM for severity: {str(e)}")
            
            st.markdown("---")
            
            # 3D Wound Visualization Section
            st.subheader("🔬 3D Wound Visualization")
            st.caption("Interactive 3D surface reconstruction from depth estimation")
            
            if result.depth is not None and result.depth.depth_map is not None:
                try:
                    from models.visualization.wound_3d import create_3d_visualization
                    
                    # Get the wound image and depth map
                    wound_img = result.roi_image if result.roi_image is not None else result.original_image
                    depth_map = result.depth.depth_map
                    
                    # Get wound mask if available
                    wound_mask = None
                    if result.segmentation and result.segmentation.mask is not None:
                        # Wound mask = all non-background pixels
                        wound_mask = result.segmentation.mask > 0
                    
                    # Settings for 3D view
                    col_settings, col_empty = st.columns([1, 2])
                    with col_settings:
                        depth_scale = st.slider("Depth Exaggeration", 1.0, 10.0, 4.0, 0.5,
                                               help="Increase to make depth differences more visible")
                    
                    # Generate 3D visualizations
                    with st.spinner("Generating 3D visualization..."):
                        textured_fig, depth_fig, stats_3d = create_3d_visualization(
                            depth_map=depth_map,
                            rgb_image=cv2.cvtColor(wound_img, cv2.COLOR_BGR2RGB) if len(wound_img.shape) == 3 else wound_img,
                            wound_mask=wound_mask,
                            depth_scale=depth_scale,
                            resolution=80
                        )
                    
                    # Display 3D views
                    view_3d_col1, view_3d_col2 = st.columns(2)
                    
                    with view_3d_col1:
                        st.markdown("**📸 Textured Surface**")
                        st.plotly_chart(textured_fig, use_container_width=True)
                    
                    with view_3d_col2:
                        st.markdown("**🌈 Depth-Colored Surface**")
                        st.plotly_chart(depth_fig, use_container_width=True)
                    
                    # 3D Stats
                    with st.expander("📊 3D Mesh Statistics"):
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            st.metric("Max Depth", f"{stats_3d['max_depth']:.2f}")
                        with stat_cols[1]:
                            st.metric("Mean Depth", f"{stats_3d['mean_depth']:.2f}")
                        with stat_cols[2]:
                            st.metric("Wound Area", f"{stats_3d['wound_area_ratio']*100:.1f}%")
                        with stat_cols[3]:
                            st.metric("Mesh Vertices", f"{stats_3d['mesh_vertices']:,}")
                    
                    st.info("💡 **Tip:** Click and drag to rotate the 3D model. Scroll to zoom.")
                    
                except Exception as e:
                    st.warning(f"Could not generate 3D visualization: {str(e)}")
            else:
                st.info("3D visualization requires depth estimation. Depth data not available.")
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if result.risk and result.risk.recommendations:
                for rec in result.risk.recommendations:
                    st.info(rec)
            
            # Timing
            with st.expander("⏱️ Processing Times"):
                for stage, time_val in result.inference_times.items():
                    st.write(f"- {stage}: {time_val:.3f}s")
                st.write(f"**Total: {result.total_time:.2f}s**")


# Page: Dashboard
elif page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 Digital Twin Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)
    
    twin = st.session_state.digital_twin
    
    if len(twin.states) == 0:
        st.warning("No wound data available. Please analyze a wound first.")
    else:
        # Current state
        current = twin.get_current_state()
        
        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Measurements", len(twin.states))
        with col2:
            st.metric("Wound Type", current.wound_type.title())
        with col3:
            st.metric("Current Risk", f"{current.risk_score:.2f}")
        with col4:
            color = get_risk_color(current.risk_level)
            st.markdown(f"""
            <div style="background-color: {color}; padding: 0.5rem; border-radius: 5px; text-align: center; color: white;">
                {current.risk_level.upper()}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Trends
        if len(twin.states) >= 2:
            trends = twin.compute_trends()
            prediction = twin.predict_healing_time()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Trend Analysis")
                
                # Timeline chart
                timestamps = [s.timestamp for s in twin.states]
                risk_scores = [s.risk_score for s in twin.states]
                areas = [s.wound_area for s in twin.states]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=timestamps, y=risk_scores, name="Risk Score", mode="lines+markers"))
                fig.update_layout(title="Risk Score Over Time", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Healing Prediction")
                
                if prediction["status"] == "predicted":
                    st.metric(
                        "Est. Days to Heal",
                        f"{prediction['estimated_days_to_heal']:.0f}",
                        delta=f"Trend: {trends['overall_trend']}"
                    )
                    
                    st.progress(prediction["confidence"])
                    st.caption(f"Confidence: {prediction['confidence']:.0%}")
                else:
                    st.info("More data needed for prediction")
        
        # Report
        st.markdown("---")
        st.subheader("📋 Full Report")
        
        with st.expander("View Report"):
            st.text(twin.generate_report())


# Page: Simulate Healing
elif page == "🎬 Simulate Healing":
    st.markdown('<h1 class="main-header">🎬 Healing Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-generated healing trajectory visualization</p>', unsafe_allow_html=True)
    
    if st.session_state.result is None:
        st.warning("⚠️ Please analyze a wound first before simulating healing.")
    else:
        result = st.session_state.result
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Simulation Settings")
            
            current_severity = st.selectbox(
                "Current Severity",
                options=[4, 3, 2, 1, 0],
                format_func=lambda x: ["Healed", "Mild", "Moderate", "Severe", "Critical"][x],
                index=2
            )
            
            treatment = st.selectbox(
                "Treatment Scenario",
                options=["optimal", "standard", "suboptimal"]
            )
            
            num_frames = st.slider("Simulation Steps", 3, 10, 5)
            
            if st.button("🎬 Generate Trajectory", type="primary"):
                with st.spinner("Generating healing simulation..."):
                    if st.session_state.trajectory_generator is None:
                        st.session_state.trajectory_generator = TrajectoryGenerator()
                    
                    gen = st.session_state.trajectory_generator
                    
                    # Generate trajectory
                    frames = gen.generate_trajectory(
                        start_image=result.roi_image if result.roi_image is not None else result.original_image,
                        start_severity=current_severity,
                        treatment_scenario=treatment,
                        num_days=30,
                        frames_per_day=0.3
                    )
                    
                    st.session_state.trajectory_frames = frames
                
                st.success(f"✅ Generated {len(frames)} frames!")
        
        with col2:
            st.subheader("📽️ Trajectory Preview")
            
            if "trajectory_frames" in st.session_state and st.session_state.trajectory_frames:
                frames = st.session_state.trajectory_frames
                
                # Slider to navigate frames
                frame_idx = st.slider("Timeline", 0, len(frames) - 1, 0)
                frame = frames[frame_idx]
                
                # Display frame
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.image(numpy_to_pil(frame.image), use_container_width=True)
                
                with col_b:
                    st.write(f"**Day {frame.day}**")
                    st.write(f"Severity: {frame.severity_name.title()}")
                    
                    # Tissue chart
                    if frame.tissue_change:
                        fig = px.bar(
                            x=list(frame.tissue_change.keys()),
                            y=[v * 100 for v in frame.tissue_change.values()],
                            title="Predicted Tissue %"
                        )
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'Generate Trajectory' to simulate healing")


# Page: History
elif page == "📜 History":
    st.markdown('<h1 class="main-header">📜 Analysis History</h1>', unsafe_allow_html=True)
    
    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)
    
    twin = st.session_state.digital_twin
    
    if len(twin.states) == 0:
        st.info("No history available yet. Analyze wounds to build history.")
    else:
        # History table
        history_data = []
        for i, state in enumerate(twin.states):
            history_data.append({
                "#": i + 1,
                "Timestamp": state.timestamp,
                "Type": state.wound_type,
                "Severity": state.severity,
                "Risk Score": f"{state.risk_score:.2f}",
                "Risk Level": state.risk_level,
                "Wound Area": f"{state.wound_area:.0f}"
            })
        
        st.dataframe(history_data, use_container_width=True)
        
        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Timeline JSON"):
                path = twin.export_timeline()
                st.success(f"Exported to {path}")
        
        with col2:
            if st.button("📋 Copy Report"):
                report = twin.generate_report()
                st.code(report)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        Digital Twin System for Chronic Wound Analysis | 
        Powered by PyTorch & Streamlit | 
        For Research Purposes Only
    </small>
</div>
""", unsafe_allow_html=True)
