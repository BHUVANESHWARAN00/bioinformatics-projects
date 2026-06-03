"""
Cell Detection and Counting from Microscopy Images
A complete Streamlit web application for automated cell analysis.
"""

import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; environment variables still work without it
    load_dotenv = lambda: None

# Securely retrieve the Gemini API key from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Model cascade: tried in order until one succeeds
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",   # most generous free-tier quota
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

import io
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Glyph.*missing from current font.*",
    category=UserWarning,
)

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from segmentation import preprocess as preprocess_image, segment as segment_image
from analysis import generate_heatmap as generate_heatmap_analysis, count_cells as count_cells_analysis
from utils import ensure_rgb as ensure_rgb_image, save_figure_bytes as save_figure_bytes_util
from ai_module import get_ai_interpretation as get_ai_interpretation_ai

# ─── Safe third-party imports ────────────────────────────────────────────────
logger = logging.getLogger("cellquantx")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
    else:
        genai = None
        gemini_available = False
    gemini_import_available = True
except ImportError:
    genai = None
    gemini_available = False
    gemini_import_available = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage, PageBreak, KeepTogether
    )
    reportlab_available = True
except ImportError:
    reportlab_available = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CellQuantX",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_EXPERT_BADGE = '<span style="background-color:#E53E3E;color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:bold;margin-bottom:0.5rem;display:inline-block;">EXPERT FEATURE</span>'

st.markdown("""
<style>
.stButton>button {
    padding: 0.25rem 0.45rem !important;
    min-width: 2.3rem !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

if "show_help" not in st.session_state:
    st.session_state.show_help = False

HELP_CONTENT = """
<div style='background:linear-gradient(160deg,#0d1a2e,#0a1222);border:1px solid rgba(99,179,237,0.28);border-radius:16px;padding:1.5rem;margin-bottom:1rem;color:#e2e8f0;font-family:Inter,sans-serif;'>
  <h2 style='color:#90cdf4;margin-top:0;'>🔎 CellQuantX Help & About</h2>
  <p style='color:#a0aec0;margin-bottom:1rem;'>Use the bottom sidebar help button to toggle this guidance on and off.</p>
  
  <h3 style='color:#63b3ed;margin-bottom:0.3rem;'>🌟 New Features & Segmentation</h3>
  <ul style='padding-left:1.2rem;color:#cbd5e0;'>
    <li><b>Enhanced Labeling Mode</b>: Advanced pipeline using Gaussian blur and adaptive thresholding to detect faint or small cells.</li>
    <li><b>Marker-based Watershed</b>: Superior algorithm for resolving densely packed, overlapping, or clumped cells.</li>
    <li><b>Connected Component Labeling (CCL)</b>: Alternative detection method to map pixel-connected blobs instead of tracing contours.</li>
    <li><b>Method Comparison</b>: Compare up to three segmentation algorithms side-by-side with automated critical analysis.</li>
  </ul>

  <h3 style='color:#63b3ed;margin-bottom:0.3rem;'>🚀 Quick Workflow</h3>
  <ul style='padding-left:1.2rem;color:#cbd5e0;'>
    <li><b>Upload</b> an image or batch of microscopy files.</li>
    <li><b>Choose preprocessing</b> (e.g., Top-Hat background removal, CLAHE) and the optimal segmentation method.</li>
    <li><b>Enable Expert Features</b> only when you need deeper analytics.</li>
    <li><b>Export</b> your results to CSV, PNG, and a comprehensive PDF report.</li>
  </ul>

  <h3 style='color:#63b3ed;margin-bottom:0.3rem;'>🔬 Expert Features</h3>
  <ul style='padding-left:1.2rem;color:#cbd5e0;'>
    <li><b>Density Heatmap</b> overlays spatial positioning patterns on the ROI.</li>
    <li><b>Phenotypic Clustering</b> groups cells by morphology-only labels (Compact-Round, Elongated, Irregular).</li>
    <li><b>Intensity Quantification</b> measures per-cell brightness across the selected expression channel.</li>
    <li><b>AI Interpretation</b> uses Gemini to analyze cell populations and suggest experimental insights.</li>
  </ul>

  <h3 style='color:#63b3ed;margin-bottom:0.3rem;'>💾 Export Notes</h3>
  <ul style='padding-left:1.2rem;color:#cbd5e0;'>
    <li><b>CSV</b> contains unified per-cell metrics and enabled expert columns.</li>
    <li><b>Custom Filenames</b> are prompted directly via the browser for all exports.</li>
    <li><b>PDF Report</b> embeds visible graphs, batch summaries, and all active expert analytics in high resolution.</li>
  </ul>
</div>
"""

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark theme base */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #0a1522 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2444 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #90cdf4, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-header p {
    color: #90cdf4;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    opacity: 0.85;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(26,58,92,0.6) 0%, rgba(13,36,68,0.8) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,179,237,0.5);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #90cdf4;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}
.section-header h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #90cdf4;
    margin: 0;
}

/* AI interpretation box */
.ai-box {
    background: linear-gradient(135deg, rgba(16,40,70,0.7) 0%, rgba(10,26,50,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-left: 4px solid #63b3ed;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #cbd5e0;
}
.ai-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a3a5c, #0d2444);
    border: 1px solid rgba(99,179,237,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.72rem;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

/* Accuracy card */
.accuracy-section {
    background: linear-gradient(135deg, rgba(26,58,92,0.4) 0%, rgba(13,36,68,0.6) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Image display */
.stImage img {
    border-radius: 10px;
    border: 1px solid rgba(99,179,237,0.2);
}

/* Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f) !important;
    color: white !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #2d6a9f, #4299e1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,179,237,0.3) !important;
}

/* Selectbox / slider labels */
.stSlider label, .stSelectbox label, .stNumberInput label,
.stCheckbox label, .stTextInput label {
    color: #90cdf4 !important;
    font-weight: 500 !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    overflow: hidden;
}

/* Divider */
hr {
    border-color: rgba(99,179,237,0.15) !important;
}

/* Plot background */
.stPlotlyChart, .stPyplot {
    background: rgba(13,26,50,0.6);
    border-radius: 10px;
    border: 1px solid rgba(99,179,237,0.15);
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
_EXPERT_BADGE = (
    "<div style='display:inline-flex;align-items:center;gap:0.4rem;"
    "background:linear-gradient(135deg,#1a3a5c,#0d2444);"
    "border:1px solid rgba(99,179,237,0.4);border-radius:20px;"
    "padding:3px 12px;font-size:0.72rem;color:#63b3ed;"
    "text-transform:uppercase;letter-spacing:0.1em;font-weight:600;"
    "margin-bottom:0.6rem;'>🧪 Expert Feature</div>"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Guarantee a 3-channel uint8 RGB image."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
    return img.astype(np.uint8)


@st.cache_data(show_spinner=False)
def preprocess(
    img_bytes: bytes,
    shape: tuple,
    apply_tophat: bool,
    apply_scharr: bool,
    flat_field_correct: bool = False,
    channel: str = "Grayscale",
    blur_radius: int = 3,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    disable_heavy_preprocessing: bool = False,
):
    """Blur + morph + edge detection + histogram equalisation + optional flat-field correction."""
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)

    # ── Color Channel Selector ────────────────────────────────────────────────
    if img.ndim == 3:
        if channel == "Auto-Detect (Brightest)":
            # auto detect brightest channel (max mean intensity)
            means = [img[:,:,i].mean() for i in range(3)]
            best_c = np.argmax(means)
            gray = img[:, :, best_c]
        elif channel == "Red (R)":
            gray = img[:, :, 0]
        elif channel == "Green (G)":
            gray = img[:, :, 1]
        elif channel == "Blue (B)":
            gray = img[:, :, 2]
        else:  # Grayscale (default)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # ── Flat-Field Correction (Vignetting removal) ─────────────────────────────
    if flat_field_correct:
        gray_f = gray.astype(np.float32)
        sigma = max(gray_f.shape) * 0.05
        ksize = int(sigma * 6) | 1  # must be odd
        flat = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
        flat = np.where(flat < 1, 1, flat)
        corrected = gray_f / flat * 128.0
        gray = np.clip(corrected, 0, 255).astype(np.uint8)

    if disable_heavy_preprocessing:
        # Return raw grayscale so we don't accidentally blur out faint/tiny features 
        return gray, gray

    if apply_tophat:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    if apply_scharr:
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        gray = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    blur_ksize = max(1, min(31, blur_radius * 2 + 1))
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(blur)
    return gray, enhanced


def _remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    """Remove long horizontal and vertical grid artifacts from binary masks."""
    h, w = binary.shape[:2]
    horiz_len = max(40, w // 20)
    vert_len = max(40, h // 20)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    mask_lines = cv2.bitwise_or(horiz_lines, vert_lines)
    return cv2.bitwise_and(binary, cv2.bitwise_not(mask_lines))


def _mask_quality_score(mask: np.ndarray) -> float:
    """Heuristic score: prefer realistic foreground density and object counts."""
    fg_ratio = cv2.countNonZero(mask) / float(mask.size)
    if fg_ratio <= 0.0:
        return -1e9

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return -1e6

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.sum((areas >= 8) & (areas <= 20000))
    tiny = np.sum(areas < 8)

    # Penalize implausible all-black/all-white masks while favoring useful objects.
    ratio_penalty = abs(fg_ratio - 0.15) * 120.0
    tiny_penalty = tiny * 0.2
    return float(valid) - ratio_penalty - tiny_penalty


def _auto_select_polarity(binary: np.ndarray) -> np.ndarray:
    """Pick normal vs inverted mask automatically when user didn't force invert."""
    inv = cv2.bitwise_not(binary)
    s_bin = _mask_quality_score(binary)
    s_inv = _mask_quality_score(inv)
    return inv if s_inv > (s_bin * 1.25) else binary


def _remove_tiny_speckles(binary: np.ndarray, min_px: int = 4) -> np.ndarray:
    """Drop tiny isolated components that are almost always noise."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary
    keep = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == i] = 255
    return keep


@st.cache_data(show_spinner=False)
def segment(
    processed_bytes: bytes,
    shape: tuple,
    method: str,
    invert: bool,
    watershed_thresh: float = 0.5,
    fill_holes: bool = False,
    manual_threshold: int = 0,
    adaptive_block: int = 11,
    adaptive_c: int = 2,
    morph_adjust: int = 0,
    remove_grid: bool = False,
    split_touching: bool = False,
) -> np.ndarray:
    """Segmentation with morphological cleanup. Cached on bytes."""
    processed = np.frombuffer(processed_bytes, dtype=np.uint8).reshape(shape)
    # Stabilize local thresholding for noisy microscope frames.
    processed = cv2.medianBlur(processed, 3)
    # Shared contrast boost improves sensitivity across all methods.
    _clahe_core = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    proc_for_thresh = _clahe_core.apply(processed)

    # 1. Base Binarisation
    if method == "Enhanced Labeling Mode":
        # Apply local contrast enhancement to FORCE faint cells to appear
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        boosted = clahe.apply(processed)
        
        # Use Morphological Top-Hat to completely flatten the background
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        tophat = cv2.morphologyEx(boosted, cv2.MORPH_TOPHAT, kernel_tophat)
        
        blur = cv2.GaussianBlur(tophat, (3, 3), 0)
        # Use highly sensitive Triangle threshold on the flat background
        th_tri, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        # Threshold at 50% of triangle threshold for ultra-sensitivity
        _, binary = cv2.threshold(blur, max(5.0, th_tri * 0.5), 255, cv2.THRESH_BINARY)
    elif manual_threshold > 0:
        _, binary = cv2.threshold(proc_for_thresh, manual_threshold, 255, cv2.THRESH_BINARY)
    elif method == "Adaptive Threshold":
        adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        adaptive_block = max(3, adaptive_block)
        # Fuse Gaussian + Mean adaptive maps to recover faint rims/cells.
        binary_g = cv2.adaptiveThreshold(
            proc_for_thresh, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            adaptive_block, adaptive_c
        )
        binary_m = cv2.adaptiveThreshold(
            proc_for_thresh, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            adaptive_block, adaptive_c
        )
        binary = cv2.bitwise_or(binary_g, binary_m)
    else:  # Global Threshold and Watershed start with hybrid Otsu-Triangle
        th_otsu, _ = cv2.threshold(proc_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_tri, _ = cv2.threshold(proc_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        # Weight triangle threshold heavily (80%) for max sensitivity to faint cells
        hybrid_th = ((th_otsu * 0.2) + (th_tri * 0.8)) * 0.9
        hybrid_th = np.clip(hybrid_th, 1, 254)
        _, binary = cv2.threshold(proc_for_thresh, hybrid_th, 255, cv2.THRESH_BINARY)

    if invert:
        binary = cv2.bitwise_not(binary)
    else:
        # Improve robustness across bright/dark staining without changing UI.
        binary = _auto_select_polarity(binary)

    if remove_grid:
        binary = _remove_grid_lines(binary)

    # 2. Fill Holes ASAP (CRITICAL for hollow fluorescent cells)
    if fill_holes:
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        h_p, w_p = padded.shape
        mask = np.zeros((h_p + 2, w_p + 2), np.uint8)
        cv2.floodFill(padded, mask, (0, 0), 255)
        padded_inv = cv2.bitwise_not(padded)
        binary = binary | padded_inv[1:-1, 1:-1]

    # 3. Morphological Cleanup and optional erosion/dilation
    kernel = np.ones((3, 3), np.uint8)
    if morph_adjust < 0:
        binary = cv2.erode(binary, kernel, iterations=min(5, -morph_adjust))
    elif morph_adjust > 0:
        binary = cv2.dilate(binary, kernel, iterations=min(5, morph_adjust))

    if method != "Enhanced Labeling Mode":
        # Open removes salt noise; close recovers weak fragmented rims.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        # Keep enhanced mode permissive but still remove 1-3 px speckles.
        binary = _remove_tiny_speckles(binary, min_px=3)

    # 4. Advanced Split (Watershed)
    if method == "Watershed" or split_touching:
        # Skip opening entirely to keep 100% of faint/thin cells
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = binary.copy()
        
        # Background area
        sure_bg = cv2.dilate(opening, kernel_e, iterations=2)

        # Distance transform for seed/peak detection
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # Adaptive Gaussian blurring of distance map
        dist_s = cv2.GaussianBlur(dist, (5, 5), 0.5)
        
        # Dynamic seed detection
        # nms_k controls the minimum separation between seeds
        nms_k = max(3, int(6 * watershed_thresh)) | 1
        local_max = cv2.dilate(dist_s, np.ones((nms_k, nms_k), np.uint8))

        # Refined noise floor: absolute fraction of max distance instead of percentile
        # Percentile filtering arbitrarily drops small cells if many normal cells exist!
        valid_dist = dist_s[dist_s > 0]
        if len(valid_dist) > 0:
            noise_floor = dist_s.max() * (watershed_thresh * 0.06)
        else:
            noise_floor = 0

        # Markers for Watershed
        sure_fg = np.zeros_like(dist_s, dtype=np.uint8)
        sure_fg[(dist_s >= (local_max - 1e-6)) & (dist_s > noise_floor)] = 255
        
        # Dilate dots slightly to form robust seeds (do NOT use MORPH_OPEN here)
        sure_fg = cv2.dilate(sure_fg, np.ones((2, 2), np.uint8), iterations=1)
        
        # fallback if no seeds found
        if cv2.countNonZero(sure_fg) < 3:
            sure_fg = cv2.erode(opening, kernel_e, iterations=2)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        img_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(img_color, markers)
        binary = np.zeros_like(processed, dtype=np.uint8)
        binary[markers > 1] = 255
        
        # Dilate the watershed boundaries to ensure rigorous 8-connected separation.
        # Without this, cv2.findContours WILL merge cells if they touch diagonally across a 1-pixel boundary.
        boundaries = np.zeros_like(processed, dtype=np.uint8)
        boundaries[markers == -1] = 255
        boundaries = cv2.dilate(boundaries, np.ones((3, 3), np.uint8))
        binary[boundaries == 255] = 0

    # Final denoise pass for every method.
    binary = _remove_tiny_speckles(binary, min_px=4)
    return binary


def generate_heatmap(shape: tuple, df: pd.DataFrame) -> np.ndarray:
    """Generate a KDE-style spatial density heatmap from cell centroids."""
    h, w = shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    if df.empty or "Centroid X" not in df.columns:
        return cv2.cvtColor(np.zeros((h, w, 3), dtype=np.uint8), cv2.COLOR_BGR2RGB)
    xs = df["Centroid X"].to_numpy(dtype=np.int32, copy=False)
    ys = df["Centroid Y"].to_numpy(dtype=np.int32, copy=False)
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if np.any(in_bounds):
        np.add.at(heatmap, (ys[in_bounds], xs[in_bounds]), 1)
    sigma = max(h, w) * 0.04
    ksize = int(sigma * 6) | 1
    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), sigma)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def count_cells(
    binary: np.ndarray,
    original_rgb: np.ndarray,
    min_area: int,
    max_area: int = 0,           # 0 = no upper limit
    box_color: tuple = (0, 255, 80),
    pixel_size: float = 0.5,
    scale_bar_um: float = 100.0,
    intensity_img: np.ndarray = None, # Greayscale image to measure brightness
    min_circularity: float = 0.0,
    method: str = "",
):
    """Detect contours or connected components and annotate image."""
    output = original_rgb.copy()
    data = []
    valid_id = 1

    if method == "Enhanced Labeling Mode":
        # Multi-scale contour detection for Enhanced Labeling Mode
        # This preserves the sensitivity while allowing metric calculation
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Use a slightly lower threshold for area in enhanced mode
            if area < (min_area * 0.5):
                continue
            if max_area > 0 and area > max_area:
                continue

            # 1. Shape Metrics
            perimeter = cv2.arcLength(cnt, True)
            circularity = (min((4 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0)
            x, y, w, h_box = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h_box, 4) if h_box > 0 else 0.0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = round(area / hull_area, 4) if hull_area > 0 else 0.0

            # 2. Centroid
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00']) if M['m00'] > 0 else x + w//2
            cy = int(M['m01']/M['m00']) if M['m00'] > 0 else y + h_box//2

            # 3. Intensity Quantification
            mean_intensity = 0.0
            if intensity_img is not None:
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = round(cv2.mean(intensity_img, mask=mask)[0], 2)

            if circularity < min_circularity:
                continue

            # 4. Phenotype
            if circularity > 0.75 and solidity > 0.85:
                phenotype = "Compact-Round"
            elif aspect_ratio > 1.8:
                phenotype = "Elongated"
            else:
                phenotype = "Irregular"

            cv2.rectangle(output, (x, y), (x + w, y + h_box), box_color, 2)
            cv2.putText(output, str(valid_id), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            data.append({
                "Cell ID": valid_id,
                "Area (px²)": round(area, 2),
                "Centroid X": cx,
                "Centroid Y": cy,
                "Perimeter": round(perimeter, 2),
                "Circularity": round(circularity, 4),
                "Aspect Ratio": aspect_ratio,
                "Solidity": solidity,
                "Mean Intensity": mean_intensity,
                "Phenotype": phenotype
            })
            valid_id += 1
    else:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            
            # 1. Shape Metrics
            perimeter = cv2.arcLength(cnt, True)
            circularity = (min((4 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0)
            x, y, w, h_box = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h_box, 4) if h_box > 0 else 0.0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = round(area / hull_area, 4) if hull_area > 0 else 0.0
            
            # 2. Centroid for Heatmap
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00']) if M['m00'] > 0 else x + w//2
            cy = int(M['m01']/M['m00']) if M['m00'] > 0 else y + h_box//2
            
            # 3. Intensity Quantification
            mean_intensity = 0.0
            if intensity_img is not None:
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = round(cv2.mean(intensity_img, mask=mask)[0], 2)

            if circularity < min_circularity:
                continue

            # 4. Phenotypic Clustering (Rule-based)
            if circularity > 0.75 and solidity > 0.85:
                phenotype = "Compact-Round"
            elif aspect_ratio > 1.8:
                phenotype = "Elongated"
            else:
                phenotype = "Irregular"

            # Drawing
            cv2.rectangle(output, (x, y), (x + w, y + h_box), box_color, 2)
            cv2.putText(output, str(valid_id), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            data.append({
                "Cell ID": valid_id,
                "Area (px²)": round(area, 2),
                "Centroid X": cx,
                "Centroid Y": cy,
                "Perimeter": round(perimeter, 2),
                "Circularity": round(circularity, 4),
                "Aspect Ratio": aspect_ratio,
                "Solidity": solidity,
                "Mean Intensity": mean_intensity,
                "Phenotype": phenotype
            })
            valid_id += 1

    # Optional scientific scale bar overlay (disabled in current UI by default).
    if pixel_size > 0 and scale_bar_um > 0:
        img_h, img_w = output.shape[:2]
        bar_px = int(scale_bar_um / pixel_size)          # pixels that represent scale_bar_um µm
        bar_px = max(bar_px, 10)                          # sanity guard
        margin = max(16, img_w // 40)
        bar_y   = img_h - margin
        bar_x2  = img_w - margin
        bar_x1  = bar_x2 - bar_px

        if bar_x1 > 0:  # only draw if it fits
            # Shadow for contrast on any background
            cv2.line(output, (bar_x1 - 1, bar_y + 1), (bar_x2 + 1, bar_y + 1), (0, 0, 0), 5)
            cv2.line(output, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), 3)
            # Tick marks at both ends
            tick_h = max(6, img_h // 60)
            for xp in (bar_x1, bar_x2):
                cv2.line(output, (xp, bar_y - tick_h), (xp, bar_y + tick_h), (255, 255, 255), 2)
            # Label
            label = f"{int(scale_bar_um)} µm"
            font_scale = max(0.35, img_w / 2000)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            lx = bar_x1 + (bar_px - lw) // 2
            ly = bar_y - tick_h - 4
            # Text shadow
            cv2.putText(output, label, (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return output, pd.DataFrame(data)


def offline_interpretation(n: int, avg_area: float, avg_circ: float, mode: str,
                           fmt: str = "Paragraph") -> str:
    """Generate a rich local biological interpretation from statistics."""
    density = "high" if n > 100 else "moderate" if n > 30 else "low"
    shape = "highly circular" if avg_circ > 0.8 else "moderately circular" if avg_circ > 0.5 else "irregular"
    size_desc = "large" if avg_area > 500 else "medium-sized" if avg_area > 100 else "small"
    health = "normal physiological conditions" if avg_circ > 0.6 else "possible cellular stress or abnormal morphology"

    if mode == "Figure Legend Writer":
        return (
            f"Figure 1. Automated cell analysis of a microscopy field. "
            f"Segmentation identified {n} cells with a mean area of {avg_area:.1f} µm² and mean circularity of {avg_circ:.3f}. "
            f"The image was processed using the selected method, and no scale-bar overlay is shown on the exported annotation."
        )
    elif mode == "Experimental Troubleshooting":
        if fmt == "Bullet Points":
            return (
                "• Check sample preparation: hypo-osmotic buffer can cause swelling and inflated area readings.\n"
                "• Verify fixation quality: under-fixation often lowers circularity and yields irregular contours.\n"
                "• Confirm illumination uniformity: uneven lighting changes contrast and can bias cell detection."
            )
        return (
            f"Offline suggestion: Review sample preparation and imaging conditions. The current dataset shows {density} cell density and mean circularity of {avg_circ:.3f}, "
            f"which may indicate either normal morphology or mild preparation artifacts depending on expected cell type. "
            f"Focus on consistent fixation and even illumination to improve segmentation reliability."
        )

    segmentation_quality = "well-segmented" if avg_circ > 0.5 else "challenging to segment accurately"
    body = (
        f"Automated analysis identified {n} cells in the field with {density} density and a mean area of {avg_area:.1f} µm². "
        f"Mean circularity of {avg_circ:.3f} suggests {shape} morphology, which is {health}. "
        f"This indicates the sample is {segmentation_quality} and may require further review if the morphology appears abnormal."
    )
    if fmt == "Bullet Points":
        return (
            f"• Cell count: {n} ({density} density)\n"
            f"• Mean area: {avg_area:.1f} µm² — {size_desc} morphology\n"
            f"• Mean circularity: {avg_circ:.3f} — {shape}\n"
            f"• Interpretation: {health}\n"
            f"• Recommendation: Review imaging conditions if morphology appears abnormal"
        )
    return body


def get_ai_interpretation(
    n: int, avg_area: float, avg_circ: float,
    cell_type: str, mode: str, pixel_size: float, method: str, df: pd.DataFrame,
    # NEW optional context
    cell_density: float = 0.0,
    confluence_pct: float = 0.0,
    avg_aspect: float = 0.0,
    avg_solidity: float = 0.0,
    ai_detail: str = "Standard",
    ai_format: str = "Paragraph",
) -> str:
    """Try each Gemini model in GEMINI_MODELS cascade; fall back to offline."""

    # ── Rich context block passed to every prompt ──────────────────────────────
    base_stats = (
        f"Cell type: {cell_type}. "
        f"Total cells detected: {n}. "
        f"Mean area: {avg_area:.2f} µm². "
        f"Mean circularity: {avg_circ:.4f}. "
        f"Mean aspect ratio: {avg_aspect:.4f} "
        f"({'elongated/migrating' if avg_aspect > 1.8 else 'near-round'} morphology). "
        f"Mean solidity: {avg_solidity:.4f} "
        f"({'compact' if avg_solidity > 0.85 else 'irregular/fragmented'} shape). "
        f"Cell density: {cell_density:.1f} cells/mm². "
        f"Confluence / coverage: {confluence_pct:.1f}%. "
        f"Segmentation method: {method}. "
        f"Calibration: {pixel_size} µm/pixel."
    )

    # ── Detail & format instructions appended to every prompt ──────────────────
    _detail_map = {
        "Brief":    "Reply in 1-2 concise sentences only, using clear scientific language.",
        "Standard": "Reply in 3-4 informative sentences in polished paragraph form.",
        "Detailed": "Reply in 5-7 sentences with specific biological reasoning, practical interpretation, and supporting rationale.",
    }
    _fmt_map = {
        "Paragraph":    "Format your response as a single flowing paragraph.",
        "Bullet Points": "Format your response as a numbered bullet-point list, one insight per bullet.",
        "Structured":   "Format your response with short bold sub-headings followed by one sentence each.",
    }
    format_instr = (
        f"{_detail_map.get(ai_detail, '')} "
        f"{_fmt_map.get(ai_format, '')}"
    ).strip()

    # ── Mode-specific prompts ────────────────────────────────────────────────
    if mode == "Biological Hypothesis":
        prompt = (
            f"You are a senior cell biology researcher. Data: {base_stats} "
            f"Based on the density ({cell_density:.1f} cells/mm²), confluence ({confluence_pct:.1f}%), "
            f"morphology (aspect ratio {avg_aspect:.2f}, solidity {avg_solidity:.2f}), and circularity, "
            f"propose a formal, testable biological hypothesis about the current state of this {cell_type} culture "
            f"(e.g., cell cycle phase, migration state, differentiation, apoptosis). "
            f"Cite specific metric values as evidence. {format_instr}"
        )
    elif mode == "Experimental Troubleshooting":
        prompt = (
            f"You are a cell biology wet-lab expert. Data: {base_stats} "
            f"Critically evaluate these metrics for signs of experimental error. "
            f"Consider: (a) unusually low/high circularity suggesting fixation or osmotic issues, "
            f"(b) unexpectedly high/low confluence hinting at seeding errors, "
            f"(c) irregular solidity ({avg_solidity:.2f}) suggesting clumping or debris. "
            f"Suggest 2-3 specific corrective actions. {format_instr}"
        )
    elif mode == "Figure Legend Writer":
        prompt = (
            f"Write a formal, publication-ready 'Figure Legend' for a peer-reviewed microscopy paper. "
            f"Data: {base_stats} "
            f"The legend must include: the staining/imaging modality implied by the cell type, "
            f"the automated segmentation method used ({method}), the calibration ({pixel_size} µm/px), "
            f"the quantitative results (N, mean area, circularity, density, confluence), "
            f"and an explicit note that no scale-bar overlay is drawn on the exported annotation. "
            f"Write it as a single formal academic paragraph in the style of Nature Cell Biology. {format_instr}"
        )
    elif mode == "Smart Outlier Detection":
        if not df.empty:
            df_disp = df.copy()
            df_disp["Area (µm²)"] = (df_disp["Area (px²)"] * pixel_size ** 2).round(2)
            _out_mask = (
                (df_disp["Circularity"] < 0.25)
                | (df_disp["Solidity"] < 0.6)
                | (df_disp["Area (px²)"] > df_disp["Area (px²)"].quantile(0.95))
                | (df_disp["Aspect Ratio"] > 3.0)
            ) if "Solidity" in df_disp.columns else (
                (df_disp["Circularity"] < 0.25)
                | (df_disp["Area (px²)"] > df_disp["Area (px²)"].quantile(0.95))
            )
            extreme = df_disp[_out_mask].head(8)
            _cols = [c for c in ["Cell ID", "Area (µm²)", "Circularity", "Aspect Ratio", "Solidity"] if c in extreme.columns]
            outlier_str = extreme[_cols].to_string(index=False) if not extreme.empty else "No extreme outliers found."
        else:
            outlier_str = "No cells detected."
        prompt = (
            f"You are an expert microscopist and data analyst. Overall stats: {base_stats} "
            f"The following cells are flagged as potential artifacts (extreme area, low circularity, "
            f"low solidity, or high aspect ratio):\n{outlier_str}\n"
            f"For each flagged Cell ID, state whether it is likely: (A) a real cell, "
            f"(B) an imaging artifact (dust, bubble, debris), or (C) a cell clump. "
            f"Then advise on which slider to adjust to remove false positives. {format_instr}"
        )
    elif mode == "Clinical Grading":
        prompt = (
            f"You are a clinical pathologist reviewing automated cell morphometry data. "
            f"Data: {base_stats} "
            f"Using established morphological criteria, provide a preliminary qualitative grading of this "
            f"{cell_type} sample: assess nuclear shape irregularity (circularity {avg_circ:.3f}), "
            f"pleomorphism (aspect ratio {avg_aspect:.2f}), cell cohesion (solidity {avg_solidity:.2f}), "
            f"and density ({cell_density:.1f} cells/mm²). "
            f"Conclude with: 'Grade I / II / III — likely benign / borderline / malignant' with a brief rationale. {format_instr}"
        )
    else:  # Basic Summary
        prompt = (
            f"You are a cell biology expert. Data: {base_stats} "
            f"Give a complete biological interpretation covering: population density assessment, "
            f"morphological health (using circularity and solidity), "
            f"growth phase estimate (using confluence {confluence_pct:.1f}%), "
            f"and any noteworthy observations from aspect ratio ({avg_aspect:.2f}) or solidity ({avg_solidity:.2f}). {format_instr}"
        )

    if not gemini_available:
        return "AI feature not available. Please configure API key."

    ai_errors = []
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            ai_errors.append(f"{model_name}: {e}")
            logger.warning("Gemini model call failed for %s: %s", model_name, e)
            continue  # try next model

    # All models failed — visible fallback
    if ai_errors:
        st.warning(
            "AI model call failed; showing offline interpretation instead. "
            f"Last error: {ai_errors[-1]}"
        )
    return offline_interpretation(n, avg_area, avg_circ, mode, ai_format)


def generate_pdf(
    total_cells: int,
    mean_area_um2: float,
    mean_circ: float,
    ai_text: str,
    df: pd.DataFrame,
    uploaded_img: np.ndarray = None,
    result_img: np.ndarray = None,
    hist_bytes: bytes = None,
    sens_bytes: bytes = None,
    scat_bytes: bytes = None,
    heat_bytes: bytes = None,
    pheno_bytes: bytes = None,
    intensity_bytes: bytes = None,
    # New metrics
    cell_density: float = 0.0,
    confluence_pct: float = 0.0,
    mean_aspect: float = 0.0,
    mean_solidity: float = 0.0,
    mean_intensity: float = 0.0,
    image_name: str = "",
    cell_type: str = "Unknown",
    method: str = "Global Threshold",
    pixel_size: float = 0.5,
    batch_df: pd.DataFrame = None,
) -> bytes:
    """Generate a PDF report and return it as bytes (no disk I/O)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=20,
        fontName="Helvetica-Bold",
        textColor=rl_colors.HexColor("#1a3a5c"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica",
        textColor=rl_colors.HexColor("#4a6fa1"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=rl_colors.HexColor("#1a3a5c"),
        spaceAfter=4,
        spaceBefore=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        fontName="Helvetica",
        textColor=rl_colors.HexColor("#2d3748"),
        spaceAfter=4,
        leading=13,
        alignment=TA_JUSTIFY,
    )

    story = []

    # Title
    story.append(Paragraph("🔬 CellQuantX Report", title_style))
    story.append(Paragraph("Cell Detection & Counting from Microscopy Images", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#63b3ed"), spaceAfter=12))

    # Summary metrics
    story.append(Paragraph("Summary Metrics", section_style))
    summary_data = [
        ["Metric", "Value"],
        ["Image", image_name or "—"],
        ["Cell Type", cell_type],
        ["Segmentation Method", method],
        ["Calibration (µm/px)", f"{pixel_size:.3f}"],
        ["Total Cells Detected", str(total_cells)],
        ["Cell Density (cells/mm²)", f"{cell_density:.2f}"],
        ["Confluence / Coverage", f"{confluence_pct:.1f}%"],
        ["Mean Cell Area (µm²)", f"{mean_area_um2:.2f}"],
        ["Mean Circularity", f"{mean_circ:.4f}"],
        ["Mean Aspect Ratio", f"{mean_aspect:.4f}"],
        ["Mean Solidity", f"{mean_solidity:.4f}"],
        ["Mean Intensity (ADU)", f"{mean_intensity:.2f}"],
    ]
    t = Table(summary_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), rl_colors.HexColor("#edf2f7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#edf2f7"), rl_colors.HexColor("#f7fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#a0aec0")),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.1 * inch))

    # AI interpretation
    story.append(Paragraph("AI Biological Interpretation", section_style))
    story.append(Paragraph(ai_text, body_style))
    story.append(Spacer(1, 0.1 * inch))

    # Images section
    if uploaded_img is not None and result_img is not None:
        story.append(Paragraph("Image Analysis Results", section_style))
        
        def get_rl_image(img_arr, max_width, max_height):
            img_pil = Image.fromarray(img_arr)
            img_buf = io.BytesIO()
            img_pil.save(img_buf, format="PNG")
            img_buf.seek(0)
            img_rl = RLImage(img_buf)
            aspect = img_pil.height / float(img_pil.width)
            img_rl.drawWidth = max_width
            img_rl.drawHeight = max_width * aspect
            if img_rl.drawHeight > max_height:
                img_rl.drawHeight = max_height
                img_rl.drawWidth = max_height / aspect
            return img_rl

        try:
            up_img = get_rl_image(uploaded_img, 3.2 * inch, 3.5 * inch)
            res_img = get_rl_image(result_img, 3.2 * inch, 3.5 * inch)
            img_table = Table([[up_img, res_img]], colWidths=[3.25*inch, 3.25*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(img_table)
            
            cap_table = Table([["Region of Interest", "Detected Cells Overlay"]], colWidths=[3.25*inch, 3.25*inch])
            cap_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Oblique'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('TEXTCOLOR', (0,0), (-1,-1), rl_colors.HexColor("#4a6fa1")),
            ]))
            story.append(cap_table)
            story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            story.append(Paragraph(f"Error rendering images: {e}", body_style))

    if hist_bytes:
        try:
            hist_rl = RLImage(io.BytesIO(hist_bytes))
            hist_rl.drawWidth = 6.5 * inch
            hist_rl.drawHeight = (6.5 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Histogram (Area Distribution)", section_style),
                hist_rl,
                Spacer(1, 0.1 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering histogram: {e}", body_style))

    if sens_bytes:
        try:
            sens_rl = RLImage(io.BytesIO(sens_bytes))
            sens_rl.drawWidth = 7.0 * inch
            sens_rl.drawHeight = (7.0 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Sensitivity Analysis", section_style),
                sens_rl,
                Spacer(1, 0.2 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering sensitivity graph: {e}", body_style))

    if scat_bytes:
        try:
            scat_rl = RLImage(io.BytesIO(scat_bytes))
            scat_rl.drawWidth = 7.0 * inch
            scat_rl.drawHeight = (7.0 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Morphology: Circularity vs Area", section_style),
                scat_rl,
                Spacer(1, 0.2 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering morphology graph: {e}", body_style))

    if heat_bytes:
        try:
            heat_rl = RLImage(io.BytesIO(heat_bytes))
            heat_rl.drawWidth = 7.0 * inch
            heat_rl.drawHeight = (7.0 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Density Heatmap Overlay", section_style),
                heat_rl,
                Spacer(1, 0.2 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering heatmap overlay: {e}", body_style))

    if pheno_bytes:
        try:
            pheno_rl = RLImage(io.BytesIO(pheno_bytes))
            pheno_rl.drawWidth = 7.0 * inch
            pheno_rl.drawHeight = (7.0 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Phenotype Summary", section_style),
                pheno_rl,
                Spacer(1, 0.2 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering phenotype chart: {e}", body_style))

    if intensity_bytes:
        try:
            intensity_rl = RLImage(io.BytesIO(intensity_bytes))
            intensity_rl.drawWidth = 7.0 * inch
            intensity_rl.drawHeight = (7.0 * inch) / 2.5
            story.append(KeepTogether([
                Paragraph("Intensity Distribution", section_style),
                intensity_rl,
                Spacer(1, 0.2 * inch)
            ]))
        except Exception as e:
            story.append(Paragraph(f"Error rendering intensity chart: {e}", body_style))

    # Batch Summary (if provided)
    if batch_df is not None and not batch_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Batch Analysis Summary", section_style))
        batch_header_style = ParagraphStyle(
            "BatchHeader",
            parent=styles["Normal"],
            fontSize=8,
            fontName="Helvetica-Bold",
            textColor=rl_colors.white,
            alignment=TA_CENTER,
            leading=10,
        )
        batch_cell_style = ParagraphStyle(
            "BatchCell",
            parent=body_style,
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
        )
        batch_table_data = [[Paragraph(str(c), batch_header_style) for c in batch_df.columns]]
        for _, row in batch_df.iterrows():
            batch_table_data.append([Paragraph(str(v), batch_cell_style) for v in row.values])
        bcol_w = max(0.8 * inch, 6.5 * inch / len(batch_df.columns))
        bt = Table(batch_table_data, colWidths=[bcol_w] * len(batch_df.columns), repeatRows=1)
        bt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#edf2f7"), rl_colors.HexColor("#ffffff")]),
            ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#a0aec0")),
        ]))
        story.append(bt)
        story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())
    # Data table (first 50 rows for PDF brevity)
    story.append(Paragraph("Cell Data Table (first 50 cells)", section_style))
    rows_to_show = min(len(df), 50)
    if rows_to_show > 0:
        table_header_style = ParagraphStyle(
            "DataHeader",
            parent=styles["Normal"],
            fontSize=8,
            fontName="Helvetica-Bold",
            textColor=rl_colors.white,
            alignment=TA_CENTER,
            leading=10,
        )
        table_cell_style = ParagraphStyle(
            "DataCell",
            parent=body_style,
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
        )
        table_data = [[Paragraph(str(c), table_header_style) for c in df.columns]]
        for _, row in df.head(rows_to_show).iterrows():
            table_data.append([Paragraph(str(v), table_cell_style) for v in row.values])
        col_count = len(df.columns)
        col_width = max(0.7 * inch, 6.5 * inch / col_count)
        dt = Table(table_data, colWidths=[col_width] * col_count, repeatRows=1)
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2d6a9f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#edf2f7"), rl_colors.HexColor("#ffffff")]),
            ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#a0aec0")),
        ]))
        story.append(dt)
    else:
        story.append(Paragraph("No cell data available.", body_style))

    doc.build(story)
    return buffer.getvalue()


def save_figure_bytes(fig, facecolor="#0a0e1a"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=facecolor)
    buf.seek(0)
    return buf.getvalue()


# ─── Critical Analysis Texts ────────────────────────────────────────────────
CRITICAL_ANALYSIS = {
    ("Global Threshold", "Watershed"): (
        "Global Threshold applies a single Otsu-computed cutoff across the entire image. "
        "It is fast and deterministic but merges touching cells into one contour. "
        "Watershed uses a distance-transform seeding approach to separate overlapping cells, "
        "yielding a higher cell count on dense cultures at the cost of extra compute time. "
        "If the two counts agree closely, the image has well-separated cells and Global Threshold suffices."
    ),
    ("Adaptive Threshold", "Watershed"): (
        "Adaptive Threshold computes a local threshold per 11×11 pixel neighbourhood, "
        "handling illumination gradients that fool global methods. "
        "However it can fragment large cells or import background texture as false positives. "
        "Watershed's marker-based splitting is more robust for dense cultures but assumes "
        "roughly convex, well-separated cell bodies — irregular or elongated cells may be over-split."
    ),
    ("Watershed", "Global Threshold"): (
        "Watershed is the reference heavy-hitter for dense, overlapping cultures. "
        "Global Threshold offers a rapid sanity-check: if its count roughly matches Watershed, "
        "cells are sparse enough that the simpler method is adequate and will run faster. "
        "A large discrepancy (>20%) indicates significant cell clumping that only Watershed resolves."
    ),
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    if "quick_preset_profile" not in st.session_state:
        st.session_state.quick_preset_profile = "Manual"
    if "last_applied_quick_preset" not in st.session_state:
        st.session_state.last_applied_quick_preset = "Manual"

    _preset_values = {
        "Brightfield Sparse": {
            "method": "Global Threshold", "invert": False, "min_area": 20, "manual_threshold": 0,
            "separate_touching": False, "remove_grid": False, "morph_adjust": 0,
            "enable_heavy_preprocessing": False, "fill_holes": True, "flat_field": False, "apply_tophat": False,
        },
        "Dense Clumps": {
            "method": "Watershed", "invert": False, "min_area": 10, "manual_threshold": 0,
            "separate_touching": True, "remove_grid": False, "morph_adjust": 1,
            "enable_heavy_preprocessing": False, "fill_holes": True, "flat_field": False, "apply_tophat": False,
        },
        "Uneven Illumination": {
            "method": "Adaptive Threshold", "invert": False, "min_area": 12, "manual_threshold": 0,
            "separate_touching": True, "remove_grid": False, "morph_adjust": 0,
            "enable_heavy_preprocessing": True, "fill_holes": True, "flat_field": True, "apply_tophat": True,
        },
        "Faint Cells": {
            "method": "Enhanced Labeling Mode", "invert": False, "min_area": 5, "manual_threshold": 0,
            "separate_touching": True, "remove_grid": False, "morph_adjust": 0,
            "enable_heavy_preprocessing": False, "fill_holes": True, "flat_field": False, "apply_tophat": False,
        },
        "Sedimentation / Clumped Cells": {
            "method": "Watershed", "invert": False, "min_area": 8, "manual_threshold": 0,
            "separate_touching": True, "remove_grid": False, "morph_adjust": 0,
            "enable_heavy_preprocessing": True, "fill_holes": True, "flat_field": False, "apply_tophat": True,
            "watershed_thresh": 0.4,
        },
    }
    _recommended_defaults = {
        "method": "Watershed", "invert": False, "min_area": 10, "manual_threshold": 0,
        "separate_touching": True, "remove_grid": False, "morph_adjust": 0,
        "enable_heavy_preprocessing": False, "fill_holes": True, "flat_field": False, "apply_tophat": False,
        "watershed_thresh": 0.5,
    }

    def _apply_preset_to_state(cfg: dict):
        for _k, _v in cfg.items():
            st.session_state[_k] = _v

    def _on_preset_change():
        _sel = st.session_state.get("quick_preset_profile", "Manual")
        if _sel == "Manual":
            st.session_state.last_applied_quick_preset = "Manual"
            return
        _apply_preset_to_state(_preset_values[_sel])
        st.session_state.last_applied_quick_preset = _sel

    def _on_reset_defaults():
        _apply_preset_to_state(_recommended_defaults)
        st.session_state.quick_preset_profile = "Manual"
        st.session_state.last_applied_quick_preset = "Manual"

    st.markdown("## 🔬 Controls")
    st.markdown("---")

    with st.expander("Segmentation Settings", expanded=True):
        method = st.selectbox(
            "Segmentation Method",
            ["Global Threshold", "Watershed", "Adaptive Threshold", "Enhanced Labeling Mode"],
            help="Algorithm used to separate cells from background.",
            key="method",
        )
        
        if method == "Watershed":
            tip = "Best for dense cell cultures and separating clustered/touching cells."
        elif method == "Global Threshold":
            tip = "Best for bright-field or fluorescence images with extremely even background illumination."
        elif method == "Adaptive Threshold":
            tip = "Best for phase-contrast or DIC images with heavy uneven shadows, flares, or gradients."
        elif method == "Enhanced Labeling Mode":
            tip = "Best for highly faint, tiny, or subtle cells that usually vanish during standard filtering."
            
        st.markdown(f"<div style='font-size:0.75rem; color:#a0aec0; margin-top:-0.6rem; margin-bottom:1rem; padding-left:0.2rem;'>💡 <i>{tip}</i></div>", unsafe_allow_html=True)
        color_choice = st.selectbox(
            "Bounding Box Color",
            ["Cyan", "Green", "Red", "Magenta", "Yellow"],
            help="Change the annotation color if it is hard to see against your cell stain (e.g., Red for GFP images)."
        )
        color_map = {
            "Green": (0, 255, 80),
            "Red": (255, 50, 50),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
            "Yellow": (255, 255, 0)
        }
        box_color = color_map[color_choice]

        invert = st.checkbox(
            "Invert Binary Mask", value=False,
            help="Flip black and white in the binary mask (useful for dark cells on bright background).",
            key="invert",
        )
        min_area = st.slider(
            "Min Cell Area (px²)", 5, 2000, 10, step=5,
            help="Lower values (5–10) improve detection of smaller cells",
            key="min_area",
        )
        if method == "Enhanced Labeling Mode":
            min_area = 5

        _max_area_enable = st.checkbox(
            "Enable Max Cell Area Filter", value=False,
            help="Exclude objects larger than this (clumps, debris)."
        )
        max_area = 0
        if _max_area_enable:
            max_area = st.slider(
                "Max Cell Area (px²)", min_area + 10, 10000,
                min(min_area * 20, 5000), step=10,
                help="Cells larger than this are ignored (e.g. cell clumps or tissue debris)."
            )

        _preset_col, _reset_col = st.columns([2, 1])
        with _preset_col:
            preset_profile = st.selectbox(
                "Preset",
                ["Manual", "Brightfield Sparse", "Dense Clumps", "Uneven Illumination", "Faint Cells", "Sedimentation / Clumped Cells"],
                help="Quick one-click setup for common microscopy scenarios.",
                key="quick_preset_profile",
                on_change=_on_preset_change,
            )
        with _reset_col:
            st.markdown("<div style='height:1.55rem;'></div>", unsafe_allow_html=True)
            reset_recommended = st.button(
                "Reset",
                help="Apply stable recommended defaults.",
                on_click=_on_reset_defaults,
            )

        enable_comparison = st.checkbox(
            "Enable Method Comparison",
            value=False,
            help="Turn on optional segmentation comparisons only when you want to inspect alternate methods.",
            key="enable_comparison",
        )
        comparison_methods = []
        if enable_comparison:
            if 'selected_comparisons' not in st.session_state:
                st.session_state.selected_comparisons = []
            available_methods = [m for m in ["Global Threshold", "Adaptive Threshold", "Watershed"] if m != method]
            st.session_state.selected_comparisons = [m for m in st.session_state.selected_comparisons if m in available_methods]
            comparison_methods = st.multiselect(
                "Methods to Compare",
                available_methods,
                default=st.session_state.selected_comparisons,
                help="Select which segmentation methods to compare against the chosen method."
            )
            st.session_state.selected_comparisons = comparison_methods
            if not comparison_methods:
                st.info("Select one or more comparison methods to enable method comparison.")

        watershed_thresh = 0.5
        if method == "Watershed":
            watershed_thresh = st.slider(
                "Watershed Peak Strictness", 0.1, 0.9, 0.5, step=0.05,
                help="Higher values force overlapping cells to separate but may over-segment large cells.",
                key="watershed_thresh",
            )

    with st.expander("Microscopy Parameters", expanded=False):
        min_diameter_um = st.slider(
            "Min Cell Diameter (µm)", 0.0, 200.0, 0.0, step=0.5,
            help="Exclude objects smaller than this physical diameter. 0 = disabled."
        )
        max_diameter_um = st.slider(
            "Max Cell Diameter (µm)", 0.0, 200.0, 0.0, step=0.5,
            help="Exclude objects larger than this physical diameter. 0 = disabled."
        )
        min_circularity = st.slider(
            "Min Circularity", 0.0, 1.0, 0.0, step=0.01,
            help="Remove elongated or fragmented objects from the count."
        )

        separate_touching = st.checkbox(
            "Separate touching cells / clumps", value=True,
            help="Apply additional splitting to resolve adjacent cells and clumps. Disable when objects are already isolated.",
            key="separate_touching",
        )
        remove_grid = st.checkbox(
            "Exclude counting chamber grid lines", value=False,
            help="Remove long horizontal or vertical grid artifacts before counting.",
            key="remove_grid",
        )

        manual_threshold = st.slider(
            "Manual Threshold (0 = auto)", 0, 255, 0, step=1,
            help="Use a fixed threshold level instead of automatic Otsu/adaptive thresholding.",
            key="manual_threshold",
        )
        adaptive_block = st.slider(
            "Adaptive Block Size", 3, 201, 11, step=2,
            help="Adaptive threshold neighbourhood size. Larger values produce smoother segmentation."
        )
        adaptive_c = st.slider(
            "Adaptive C Constant", -20, 20, 2, step=1,
            help="Constant subtracted from adaptive thresholding mean/gaussian. Higher values reduce sensitivity."
        )
        morph_adjust = st.slider(
            "Binary Morphology (Erode/Dilate)", -5, 5, 0, step=1,
            help="Negative values erode the mask to remove noise, positive values dilate to close small holes.",
            key="morph_adjust",
        )

    with st.expander("Advanced Preprocessing", expanded=False):
        enhanced_mode_forces_light = method == "Enhanced Labeling Mode"
        enable_heavy_preprocessing = st.checkbox(
            "Enable Heavy Preprocessing",
            value=False,
            disabled=enhanced_mode_forces_light,
            help="Turn ON to activate full advanced preprocessing controls.",
            key="enable_heavy_preprocessing",
        )
        if enhanced_mode_forces_light:
            st.info("Enhanced Labeling Mode keeps heavy preprocessing disabled to preserve faint/tiny cells.")
        disable_heavy_preprocessing = (not enable_heavy_preprocessing) or enhanced_mode_forces_light
        adv_disabled = disable_heavy_preprocessing
            
        channel = st.selectbox(
            "🎨 Color Channel",
            ["Auto-Detect (Brightest)", "Grayscale", "Blue (B)", "Green (G)", "Red (R)"],
            help="Auto-Detect automatically picks the most intense channel for fluorescent images like DAPI.",
            disabled=adv_disabled,
        )
        blur_radius = st.slider(
            "Pre-Blur Radius (px)", 1, 15, 3, step=1,
            help="Smooth image noise before thresholding while preserving cell boundaries.",
            disabled=adv_disabled,
        )
        clahe_clip = st.slider(
            "CLAHE Clip Limit", 1.0, 7.0, 2.5, step=0.5,
            help="Local contrast enhancement strength used before binarisation.",
            disabled=adv_disabled,
        )
        clahe_grid = st.slider(
            "CLAHE Tile Grid Size", 4, 32, 4, step=4,
            help="Tile grid size for local contrast enhancement. Smaller values increase fine contrast.",
            disabled=adv_disabled,
        )
        tophat_kernel_size = st.slider(
            "Top-Hat Kernel Size", 3, 61, 31, step=2,
            help="Kernel size used for top-hat background removal.",
            disabled=adv_disabled,
        )
        core_clahe_clip = st.slider(
            "Segmentation Core CLAHE Clip", 1.0, 5.0, 2.2, step=0.1,
            help="Contrast boost used inside segmentation.",
        )
        enhanced_clahe_clip = st.slider(
            "Enhanced Mode CLAHE Clip", 1.0, 7.0, 3.0, step=0.1,
            help="Contrast boost used in Enhanced Labeling Mode.",
        )
        enhanced_tophat_kernel_size = st.slider(
            "Enhanced Mode Top-Hat Kernel", 3, 61, 21, step=2,
            help="Kernel size for Enhanced Labeling Mode background flattening.",
        )
        apply_tophat = st.checkbox("Apply Top-Hat Filter", value=False, help="Background subtraction to flatten uneven lighting/halos.", disabled=adv_disabled, key="apply_tophat")
        apply_scharr = st.checkbox("Apply Scharr Edge Detection", value=False, help="Highlights cell boundaries (useful for Phase Contrast).", disabled=adv_disabled)
        fill_holes = st.checkbox("Fill Cell Holes", value=True, help="Automatically fill holes inside detected cells after thresholding.", key="fill_holes")
        flat_field  = st.checkbox(
            "Flat-Field Correction", value=False,
            help="Removes vignetting (dark edges) by dividing the image by a heavily blurred version of itself. "
                 "Essential for wide-field microscopy where the center is brighter than the edges.",
            disabled=adv_disabled,
            key="flat_field",
        )

    with st.expander("Display & Reporting", expanded=False):
        resize_before_processing = st.checkbox(
            "Resize image before processing (faster)",
            value=True,
            help="Resizes ROI to 512x512 before preprocessing/segmentation to speed up heavy methods.",
        )
        show_density   = st.checkbox("Cell Density (cells/mm²)", value=True,
            help="Cells per square millimetre calculated from pixel size and image area.")
        show_confluence = st.checkbox("Confluence / Coverage %", value=True,
            help="Percentage of image area covered by detected cells — key for cell culture monitoring.")
        show_aspect    = st.checkbox("Aspect Ratio", value=True,
            help="Width / Height of bounding box. Values near 1 = round; >2 = elongated / migrating.")
        show_solidity  = st.checkbox("Solidity", value=True,
            help="Contour area / Convex-hull area. Values near 1 = compact; low = irregular or fragmented.")

        # Scale bar overlay is intentionally disabled to keep output images clean.
        scale_bar_um = 0

        enable_intensity = st.checkbox("Toggle Intensity Quantification", value=False, help="Measure average brightness per cell (Quantify Expression).")
        enable_heatmap = st.checkbox("Toggle Spatial Density Heatmap", value=False, help="Visualize cell distribution patterns.")
        enable_phenotype = st.checkbox("Toggle Phenotypic Clustering", value=False, help="Categorize cells into morphology groups (Compact-Round, Elongated, Irregular).")

        show_hist = st.checkbox("Show Area Distribution Histogram", value=True)
        show_sens = st.checkbox("Show Threshold Sensitivity Graph", value=True)
        show_scat = st.checkbox("Show Circularity vs Area Scatter", value=True)
        show_pheno_chart = st.checkbox("Show Phenotype Breakdown Chart", value=True)
        show_batch_summary = st.checkbox(
            "Show Batch Summary (multi-image only)",
            value=False,
            help="Enable only when presenting multi-image analysis."
        )

    st.markdown("---")
    st.markdown("### 📐 Calibration")

    # ── Objective Preset Selector ────────────────────────────────────────────
    _OBJECTIVE_PRESETS = {
        "Custom (manual entry)": None,
        "——— Widefield (0.65 NA) ———": None,
        "4×  — 2.44 µm/px": 2.44,
        "10× — 0.65 µm/px": 0.65,
        "20× — 0.32 µm/px": 0.32,
        "40× — 0.16 µm/px": 0.16,
        "60× — 0.11 µm/px": 0.11,
        "100× — 0.065 µm/px": 0.065,
        "——— Confocal (1.4 NA) ———": None,
        "20× Confocal — 0.30 µm/px": 0.30,
        "40× Confocal — 0.10 µm/px": 0.10,
        "63× Confocal — 0.07 µm/px": 0.07,
        "——— Phase Contrast ———": None,
        "10× Phase — 1.00 µm/px":  1.00,
        "20× Phase — 0.50 µm/px":  0.50,
        "40× Phase — 0.25 µm/px":  0.25,
        "——— Whole-Slide (WSI) ———": None,
        "20× WSI — 0.50 µm/px":  0.50,
        "40× WSI — 0.25 µm/px":  0.25,
    }
    _preset_choice = st.selectbox(
        "🔬 Objective / Preset",
        list(_OBJECTIVE_PRESETS.keys()),
        help="Select your microscope objective to auto-fill µm/pixel. Choose 'Custom' to enter manually."
    )
    _preset_val = _OBJECTIVE_PRESETS.get(_preset_choice)
    _default_px = _preset_val if _preset_val is not None else 0.50

    pixel_size = st.number_input(
        "µm per Pixel",
        min_value=0.001,
        max_value=20.0,
        value=float(_default_px),
        step=0.001,
        format="%.3f",
        disabled=(_preset_val is not None),
        help="Overridden by the preset above. Switch to 'Custom' to type your own value."
    )
    # Show info line for selected preset
    if _preset_val is not None:
        st.markdown(
            f"<small style='color:#63b3ed;'>ℹ️ Preset: {_preset_choice} → <b>{pixel_size} µm/px</b> — "
            f"1 px ≈ {pixel_size:.3f} µm &nbsp;|&nbsp; "
            f"100 µm ≈ {100/pixel_size:.0f} px</small>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<small style='color:#4a6fa1;'>1 px = {pixel_size:.3f} µm &nbsp;|&nbsp; "
            f"100 µm ≈ {100/pixel_size:.0f} px</small>",
            unsafe_allow_html=True
        )

    # Convert diameter thresholds to pixel area thresholds for contour filtering.
    min_diameter_area = 0
    max_diameter_area = 0
    if min_diameter_um > 0 and pixel_size > 0:
        min_diameter_area = int(np.pi * ((min_diameter_um / 2) / pixel_size) ** 2)
    if max_diameter_um > 0 and pixel_size > 0:
        max_diameter_area = int(np.pi * ((max_diameter_um / 2) / pixel_size) ** 2)
    if min_diameter_area > 0 or max_diameter_area > 0:
        st.markdown(
            f"<small style='color:#63b3ed;'>Equivalent size thresholds: "
            f"min {min_diameter_area or '—'} px², max {max_diameter_area or '—'} px².</small>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🤖 AI Interpretation")
    cell_type = st.text_input("Cell Type", value="Unknown",
        help="e.g., HeLa, Lymphocytes, RBC, Yeast, CHO, MCF-7")
    ai_mode = st.selectbox(
        "AI Task",
        [
            "Basic Summary",
            "Biological Hypothesis",
            "Experimental Troubleshooting",
            "Figure Legend Writer",
        ]
    )
    ai_detail = st.select_slider(
        "Detail Level",
        options=["Brief", "Standard", "Detailed"],
        value="Standard",
        help="Brief = 1-2 sentences. Standard = 3-4. Detailed = 5-7 with full reasoning."
    )
    ai_format = st.selectbox(
        "Output Format",
        ["Paragraph", "Bullet Points", "Structured"],
        help="Paragraph: flowing text. Bullets: numbered list. Structured: bold sub-headings."
    )
    st.markdown(
        "<small style='color:#4a6fa1;'>✅ Gemini Auto (flash-lite → flash → latest)</small>",
        unsafe_allow_html=True,
    )


    st.markdown("---")
    st.markdown(
        "<small style='color:#4a6fa1;'>CellQuantX v1.0 · Built with Streamlit + OpenCV</small>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    if st.button("❔", key="help_toggle", help="Show or hide help", ):
        st.session_state.show_help = not st.session_state.show_help
    if st.session_state.show_help:
        st.markdown("<hr style='border-color:rgba(99,179,237,0.2);margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown(HELP_CONTENT, unsafe_allow_html=True)

# ─── Main header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>🔬 CellQuantX</h1>
  <p>CellQuantX Detection & Counting from Microscopy Images</p>
</div>
""", unsafe_allow_html=True)

# ─── Image upload ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h2>📁 Upload Image</h2></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["png", "jpg", "jpeg", "tif", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("👆 Upload one or more microscopy images to begin analysis.")
    st.stop()

file_names = [f.name for f in uploaded_files]
col_img1, col_img2 = st.columns([3, 1])
with col_img1:
    selected_name = st.selectbox("Select image to analyse", file_names)
with col_img2:
    rotation = st.selectbox("Rotate Image", ["0°", "90°", "180°", "270°"])

selected_file = uploaded_files[file_names.index(selected_name)]

# Read image
selected_file.seek(0)
pil_img = Image.open(selected_file)
img_array = np.array(pil_img)

# Apply rotation (np.rot90 rotates counter-clockwise by default, hence negative k for clockwise)
if rotation != "0°":
    k_map = {"90°": -1, "180°": -2, "270°": -3}
    img_array = np.rot90(img_array, k=k_map[rotation])

img = ensure_rgb_image(img_array)

h, w = img.shape[:2]

# ─── ROI Selection ────────────────────────────────────────────────────────────

enable_crop = st.checkbox("✂️ Enable Image Cropping (ROI)")

if enable_crop:
    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
    with roi_col1:
        x1 = st.slider("X1 (left)", 0, w - 1, 0)
    with roi_col2:
        y1 = st.slider("Y1 (top)", 0, h - 1, 0)
    with roi_col3:
        x2 = st.slider("X2 (right)", 1, w, w)
    with roi_col4:
        y2 = st.slider("Y2 (bottom)", 1, h, h)

    # Enforce valid bounds
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        st.error("❌ ROI is empty — please adjust the X/Y sliders.")
        st.stop()
else:
    roi = img.copy()
if resize_before_processing:
    roi = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_AREA)

# ─── Preprocessing ────────────────────────────────────────────────────────────

with st.spinner("Preprocessing…"):
    gray, enhanced = preprocess_image(
        roi.tobytes(), roi.shape, apply_tophat, apply_scharr, flat_field, channel,
        blur_radius=blur_radius,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        disable_heavy_preprocessing=disable_heavy_preprocessing,
        tophat_kernel_size=tophat_kernel_size,
    )

# ─── Segmentation ─────────────────────────────────────────────────────────────

with st.spinner("Segmenting…"):
    binary = segment_image(
        enhanced.tobytes(), enhanced.shape, method, invert, watershed_thresh, fill_holes,
        manual_threshold=manual_threshold,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        morph_adjust=morph_adjust,
        remove_grid=remove_grid,
        split_touching=separate_touching,
        core_clahe_clip=core_clahe_clip,
        enhanced_clahe_clip=enhanced_clahe_clip,
        enhanced_tophat_kernel_size=enhanced_tophat_kernel_size,
    )

    if method == "Enhanced Labeling Mode":
        # Optional: Getting existing method count for comparison
        binary_existing = segment_image(
            enhanced.tobytes(), enhanced.shape, "Adaptive Threshold", invert, watershed_thresh, fill_holes,
            manual_threshold=manual_threshold, adaptive_block=adaptive_block, adaptive_c=adaptive_c,
            morph_adjust=morph_adjust, remove_grid=remove_grid, split_touching=separate_touching,
            core_clahe_clip=core_clahe_clip,
            enhanced_clahe_clip=enhanced_clahe_clip,
            enhanced_tophat_kernel_size=enhanced_tophat_kernel_size,
        )
        _, df_existing = count_cells_analysis(
            binary_existing, roi, 10, max_area if max_area > 0 else 0, box_color, pixel_size, scale_bar_um,
            intensity_img=None, min_circularity=min_circularity, method="Adaptive Threshold"
        )
        existing_count = len(df_existing)

# ─── Safe overlay ─────────────────────────────────────────────────────────────

mask_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [255, 50, 50]
overlay = cv2.addWeighted(roi.astype(np.uint8), 0.7, mask_rgb.astype(np.uint8), 0.3, 0)

# ─── Cell counting ────────────────────────────────────────────────────────────

min_area_effective = min_area
if min_diameter_area > 0:
    min_area_effective = max(min_area_effective, min_diameter_area)

max_area_effective = max_area
if max_diameter_area > 0:
    max_area_effective = max_area_effective if max_area_effective > 0 else max_diameter_area
    if max_area > 0:
        max_area_effective = min(max_area_effective, max_diameter_area)

detected_img, df = count_cells_analysis(
    binary, roi, min_area_effective, max_area_effective,
    box_color, pixel_size, scale_bar_um,
    intensity_img=enhanced if enable_intensity else None,
    min_circularity=min_circularity,
    method=method,
)

# ─── Add Area (µm²) column to df ─────────────────────────────────────────────
if not df.empty:
    df["Area (µm²)"] = (df["Area (px²)"] * pixel_size ** 2).round(4)

# ─── Compute derived metrics needed throughout the page ───────────────────────
_roi_h, _roi_w = roi.shape[:2]
_area_mm2 = _roi_w * _roi_h * (pixel_size ** 2) / 1_000_000
cell_density   = round(len(df) / _area_mm2, 2) if _area_mm2 > 0 else 0.0
confluence_pct = round(np.sum(binary > 0) / binary.size * 100, 1)

# ─── Detection Quality Heuristics ────────────────────────────────────────────
fg_ratio = (np.sum(binary > 0) / binary.size) if binary.size > 0 else 0.0
num_labels_q, _, stats_q, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
cc_count_q = int(np.sum(stats_q[1:, cv2.CC_STAT_AREA] >= min_area_effective)) if num_labels_q > 1 else 0
primary_count = len(df)
if max(primary_count, cc_count_q) > 0:
    count_stability = 1.0 - abs(primary_count - cc_count_q) / max(primary_count, cc_count_q)
else:
    count_stability = 0.0

_confidence_score = (
    min(1.0, count_stability * 0.7 + max(0.0, 1.0 - abs(fg_ratio - 0.12) / 0.12) * 0.3)
)
if _confidence_score >= 0.75:
    detection_confidence = "High"
elif _confidence_score >= 0.45:
    detection_confidence = "Medium"
else:
    detection_confidence = "Low"

if not df.empty:
    avg_area_um2 = df["Area (px²)"].mean() * pixel_size ** 2
    avg_circ = df["Circularity"].mean()
    avg_aspect = df["Aspect Ratio"].mean()
    avg_solidity = df["Solidity"].mean()
    avg_intensity = df["Mean Intensity"].mean() if "Mean Intensity" in df.columns else 0.0
else:
    avg_area_um2 = 0.0
    avg_circ = 0.0
    avg_aspect = 0.0
    avg_solidity = 0.0
    avg_intensity = 0.0

# ─── PDF Export Stub ────────────────────────────────────────────────────────
if enable_heatmap and not df.empty:
    heatmap_img = generate_heatmap_analysis(binary.shape, df)
    heatmap_overlay = cv2.addWeighted(roi, 0.55, heatmap_img, 0.45, 0)
    heat_buf = io.BytesIO()
    _heat_pil = Image.fromarray(heatmap_overlay)
    _heat_pil.save(heat_buf, format="PNG")
    st.session_state["heat_bytes"] = heat_buf.getvalue()
else:
    st.session_state["heat_bytes"] = None

if enable_phenotype and not df.empty:
    st.session_state["pheno_bytes"] = None
else:
    st.session_state["pheno_bytes"] = None

# ─── Display: preprocessing stages ───────────────────────────────────────────

st.markdown('<div class="section-header"><h2>🔍 Preprocessing Stages</h2></div>', unsafe_allow_html=True)

# Use 4 columns for more compact, medium-sized images
pre_col1, pre_col2, pre_col3, pre_col4 = st.columns(4)
with pre_col1:
    st.image(roi, caption="1. ROI", use_container_width=True)
with pre_col2:
    st.image(gray, caption="2. Grayscale", use_container_width=True, clamp=True)
with pre_col3:
    st.image(enhanced, caption="3. Enhanced", use_container_width=True, clamp=True)
with pre_col4:
    st.image(binary, caption="4. Binary Mask", use_container_width=True, clamp=True)

st.markdown('<div class="section-header"><h2>🧩 Segmentation Output</h2></div>', unsafe_allow_html=True)

if method == "Enhanced Labeling Mode":
    st.info(f"✨ Enhanced Labeling Mode uses pixel-based detection for higher sensitivity.\n\n**Counts Comparison**: Pixel-based mode found **{len(df)}** cells vs. existing contour method's **{existing_count}** cells.")

if fg_ratio < 0.01:
    st.warning("Detection quality warning: foreground is very low. Try lowering min area, enabling invert, or switching to Adaptive Threshold.")
elif fg_ratio > 0.55:
    st.warning("Detection quality warning: foreground is very high. Try increasing min area, disabling invert, or switching to Global Threshold.")

_conf_color = {"High": "#48bb78", "Medium": "#f6ad55", "Low": "#f56565"}[detection_confidence]
st.markdown(
    f"<small style='color:{_conf_color};'>Detection confidence: <b>{detection_confidence}</b> "
    f"(mask quality + count stability check)</small>",
    unsafe_allow_html=True
)

seg_col1, seg_col2, seg_col3 = st.columns([0.25, 0.5, 0.25])
with seg_col2:
    st.image(
        detected_img,
        caption=f"{method} segmentation result — {len(df)} cells detected",
        use_container_width=True,
    )

# ─── Segmentation Method Analysis ────────────────────────────────────────────

METHOD_INFO = {
    "Global Threshold": {
        "aka": "Otsu's Binarisation",
        "mechanism": (
            "Computes a single intensity threshold that minimises intra-class variance across "
            "the entire image histogram (Otsu, 1979). Every pixel above the threshold is "
            "labelled foreground (cell); every pixel below is background."
        ),
        "morph_role": (
            "A 3×3 morphological opening (erosion → dilation) is applied after binarisation "
            "to eliminate isolated noise pixels and thin salt artifacts without shrinking the "
            "main cell bodies."
        ),
        "strengths": [
            "✅ Extremely fast — O(n) histogram computation",
            "✅ Fully automatic threshold selection",
            "✅ Works well when cell/background intensities are bimodal",
        ],
        "weaknesses": [
            "❌ Fails under non-uniform illumination (vignetting, hotspots)",
            "❌ Touching cells merge into one contour",
            "❌ A single threshold cannot handle multi-modal histograms",
        ],
        "best_for": "Bright-field or fluorescence images with even illumination and clearly separated cells.",
    },
    "Adaptive Threshold": {
        "aka": "Gaussian Adaptive Binarisation",
        "mechanism": (
            "Divides the image into overlapping local windows (11×11 px) and computes a "
            "threshold per region as the Gaussian-weighted mean minus a constant C=2. "
            "This locally adapts to illumination gradients across the field."
        ),
        "morph_role": (
            "Morphological opening suppresses the fine grid-like noise that adaptive "
            "thresholding introduces at window boundaries, consolidating small pixel islands "
            "into coherent cell regions."
        ),
        "strengths": [
            "✅ Robust to gradual illumination changes",
            "✅ Preserves faint cells near the image edges",
            "✅ No global parameter tuning required",
        ],
        "weaknesses": [
            "❌ Slower than global threshold",
            "❌ Window size must be tuned for cell scale",
            "❌ Can over-segment textured backgrounds",
        ],
        "best_for": "Phase-contrast or DIC images with uneven background illumination.",
    },
    "Watershed": {
        "aka": "Distance-Transform Watershed",
        "mechanism": (
            "After Otsu binarisation, the Euclidean distance transform amplifies the "
            "interior of each cell. Peaks in that distance map become watershed seeds; "
            "the algorithm floods outward from each seed, building a dam wherever two "
            "catchment basins meet — effectively separating touching cells."
        ),
        "morph_role": (
            "Opening before the distance transform removes thin bridges between adjacent "
            "cells so their distance peaks remain isolated, preventing seed merging."
        ),
        "strengths": [
            "✅ Separates touching or clustered cells that fool simple thresholding",
            "✅ Shape-preserving — respects cell boundaries accurately",
            "✅ Handles moderate cell overlap",
        ],
        "weaknesses": [
            "❌ Computationally heavier",
            "❌ Over-segments cells with irregular shapes or holes",
            "❌ Sensitive to image noise at distance-transform peaks",
        ],
        "best_for": "Dense cell cultures where cells touch, or when individual cell boundaries matter.",
    },
    "Enhanced Labeling Mode": {
        "aka": "Pixel-Connected Adaptive",
        "mechanism": (
            "Uses a wide-block adaptive threshold coupled with 8-way Connected Component Labeling "
            "instead of traditional contour detection, tracking every single lit pixel blob without area filtering."
        ),
        "morph_role": (
            "A strict 5x5 elliptical opening scrubs out single-pixel noise before labeling."
        ),
        "strengths": [
            "✅ Highly sensitive to faint or tiny cells",
            "✅ Immune to irregular shapes breaking contours",
        ],
        "weaknesses": [
            "❌ Can mistake large artifacts for cells due to lack of area filtering",
            "❌ Very slow on extreme high-res images",
        ],
        "best_for": "Faint, tiny, or subtle cells scattered across noisy backgrounds.",
    },
}

info = METHOD_INFO.get(method, {})
with st.expander(f"🔬 Segmentation Method Analysis — {method} ({info.get('aka','')})", expanded=False):
    col_exp1, col_exp2 = st.columns([3, 2])
    with col_exp1:
        st.markdown("**How it works**")
        st.write(info.get("mechanism", ""))
        st.markdown("**Morphological Post-processing**")
        st.write(info.get("morph_role", ""))
        st.markdown("**Best suited for**")
        st.info(info.get("best_for", ""))
    with col_exp2:
        st.markdown("**Strengths**")
        for s in info.get("strengths", []):
            st.markdown(f"- {s}")
        st.markdown("**Weaknesses**")
        for w in info.get("weaknesses", []):
            st.markdown(f"- {w}")

    # --- Connected Component Labeling (alternative to contour detection) ---
    st.markdown("---")
    st.markdown("**Connected Component Labeling (CCL) — alternative detection method**")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ccl_cells = [
        stats[i] for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    ccl_count = len(ccl_cells)
    _n = len(df)  # total_cells is defined later; use len(df) here
    _agreement_rate = max(0.0, 100 - abs(_n - ccl_count) / max(1, max(_n, ccl_count)) * 100)
    st.write(
        f"Contour detection found **{_n}** cells. "
        f"CCL (8-connectivity) found **{ccl_count}** cell regions above the {min_area} px² threshold. "
        f"Differences arise because contours trace boundaries while CCL counts pixel-connected blobs directly. "
        f"Agreement rate: **{_agreement_rate:.1f}%**"
    )

    # ── Parameter Advice when agreement rate is low (offline, no API call) ───
    if _agreement_rate < 80.0:
        _gap = abs(_n - ccl_count)
        if _n > ccl_count:
            _tip = "Contours found more cells than CCL."
        else:
            _tip = "CCL found more cells than contours."
        st.warning(f"⚠️ **Low Agreement Rate ({_agreement_rate:.1f}%)** — Parameter Tip: {_tip}")

# ─── Statistics ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h2>📊 Statistics</h2></div>', unsafe_allow_html=True)

total_cells = len(df)
roi_h, roi_w = roi.shape[:2]

if not df.empty:
    avg_area_px   = df["Area (px²)"].mean()
    avg_area_um2  = avg_area_px * (pixel_size ** 2)
    avg_circ      = df["Circularity"].mean()
    avg_aspect    = df["Aspect Ratio"].mean()    if "Aspect Ratio"   in df.columns else 0.0
    avg_solidity  = df["Solidity"].mean()        if "Solidity"       in df.columns else 0.0
    avg_intensity = df["Mean Intensity"].mean()  if "Mean Intensity" in df.columns else 0.0
else:
    avg_area_px = avg_area_um2 = avg_circ = avg_aspect = avg_solidity = avg_intensity = 0.0

# ── Row 1: core metrics ──────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="value">{total_cells}</div><div class="label">Total Cells</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="value">{avg_area_um2:.1f}</div><div class="label">Avg Area (µm²)</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="value">{avg_circ:.3f}</div><div class="label">Avg Circularity</div></div>', unsafe_allow_html=True)
with m4:
    if enable_intensity:
        st.markdown(f'<div class="metric-card"><div class="value">{avg_intensity:.1f}</div><div class="label">Avg Intensity</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card"><div class="value">{total_cells}</div><div class="label">Total Population</div></div>', unsafe_allow_html=True)

# — Row 2: optional extra metrics ────────────────────────────────────────────
_extra_cols = []
_extra_cols.append((detection_confidence, "Detection Confidence"))
if show_density:    _extra_cols.append((f"{cell_density:.1f}",  "Cell Density (cells/mm²)"))
if show_confluence: _extra_cols.append((f"{confluence_pct:.1f}%", "Confluence / Coverage"))
if show_aspect:     _extra_cols.append((f"{avg_aspect:.3f}",    "Avg Aspect Ratio"))
if show_solidity:   _extra_cols.append((f"{avg_solidity:.3f}",  "Avg Solidity"))

if _extra_cols:
    _ec = st.columns(min(len(_extra_cols), 4))
    for _col, (_val, _lbl) in zip(_ec, _extra_cols):
        with _col:
            st.markdown(f'<div class="metric-card"><div class="value">{_val}</div><div class="label">{_lbl}</div></div>', unsafe_allow_html=True)

# ─── Data Table ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h2>📋 Cell Data Table</h2></div>', unsafe_allow_html=True)
if not df.empty:
    _cols = ["Cell ID", "Area (px²)", "Area (µm²)", "Circularity", "Aspect Ratio", "Solidity"]
    if enable_intensity and "Mean Intensity" in df.columns:
        _cols.append("Mean Intensity")
    if enable_phenotype and "Phenotype" in df.columns:
        _cols.append("Phenotype")
    st.dataframe(df[[c for c in _cols if c in df.columns]], use_container_width=True, height=250)
else:
    st.info("No data available.")

# ─── Graphs Section ──────────────────────────────────────────────────────────

if not df.empty:
    if show_hist:
        st.markdown('<div class="section-header"><h2>📈 Area Distribution</h2></div>', unsafe_allow_html=True)
        _areas_um2 = df["Area (px²)"] * pixel_size ** 2
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#0d1526")
        n, bins, patches = ax.hist(_areas_um2, bins=28, edgecolor="white", linewidth=0.6)
        _norm = plt.Normalize(bins.min(), bins.max())
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(plt.cm.viridis(_norm(left)))
            patch.set_alpha(0.92)
        ax.axvline(_areas_um2.mean(), color="#f6e05e", lw=1.8, ls="--",
                   label=f"Mean {_areas_um2.mean():.1f} µm²")
        ax.axvline(_areas_um2.median(), color="#63b3ed", lw=1.4, ls=":",
                   label=f"Median {_areas_um2.median():.1f} µm²")
        ax.set_xlabel("Cell Area (µm²)", color="#90cdf4", fontsize=10)
        ax.set_ylabel("Cell Count", color="#90cdf4", fontsize=10)
        ax.tick_params(colors="#90cdf4", labelsize=9)
        ax.spines[:].set_color("#1a3a5c")
        ax.grid(color="#1a3751", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9, facecolor="#0d1526", labelcolor="white", framealpha=0.75)
        ax.set_title("Cell Area Distribution", color="#bee3f8", fontsize=11, pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.session_state["hist_bytes"] = save_figure_bytes_util(fig)
        plt.close(fig)
    else:
        st.session_state["hist_bytes"] = None

    if show_scat:
        st.markdown('<div class="section-header"><h2>🔵 Morphology: Circularity vs Area</h2></div>', unsafe_allow_html=True)
        _areas_um2_s = df["Area (px²)"] * pixel_size ** 2
        _pheno_color_map_s = {"Compact-Round": "#48bb78", "Elongated": "#4299e1", "Irregular": "#f56565"}
        _scatter_colors = df["Phenotype"].map(_pheno_color_map_s).fillna("#90cdf4") if "Phenotype" in df.columns else "#90cdf4"
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#0d1526")
        ax.scatter(_areas_um2_s, df["Circularity"], c=_scatter_colors,
                   s=45, alpha=0.8, edgecolors="w", linewidth=0.3)
        ax.axhline(0.75, color="#f6e05e", lw=1.2, ls=":", alpha=0.8, label="Circularity threshold")
        ax.set_xlabel("Cell Area (µm²)", color="#90cdf4", fontsize=10)
        ax.set_ylabel("Circularity", color="#90cdf4", fontsize=10)
        ax.tick_params(colors="#90cdf4", labelsize=9)
        ax.spines[:].set_color("#1a3a5c")
        ax.set_ylim(0, 1.05)
        ax.grid(color="#1a3751", linestyle="--", linewidth=0.5, alpha=0.45)
        if "Phenotype" in df.columns:
            for _lbl, _clr in _pheno_color_map_s.items():
                if _lbl in df["Phenotype"].values:
                    ax.scatter([], [], c=_clr, label=_lbl, s=40, alpha=0.9, edgecolors="none")
        ax.legend(fontsize=8, facecolor="#0d1526", labelcolor="white", framealpha=0.8)
        ax.set_title("Circularity vs Area — coloured by Phenotype", color="#bee3f8", fontsize=11, pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.session_state["scat_bytes"] = save_figure_bytes_util(fig)
        plt.close(fig)
    else:
        st.session_state["scat_bytes"] = None

    if show_sens:
        st.markdown('<div class="section-header"><h2>📉 Threshold Sensitivity Analysis</h2></div>', unsafe_allow_html=True)
        contours_all, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_areas = np.array([cv2.contourArea(cnt) for cnt in contours_all])
        _ax_max = max(float(all_areas.max()) if len(all_areas) > 0 else 500.0, 50)
        ts = np.linspace(0, _ax_max * 1.1, 60)
        cs = [np.sum(all_areas >= t) for t in ts]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#0d1526")
        ax.fill_between(ts, cs, alpha=0.22, color="#63b3ed")
        ax.plot(ts, cs, color="#63b3ed", lw=2, marker="o", markersize=3, markerfacecolor="#90cdf4", markeredgecolor="none")
        ax.axvline(min_area, color="#f56565", lw=1.8, ls="--",
                   label=f"Current Min Area = {min_area} px² → {total_cells} cells")
        ax.set_xlabel("Min Cell Area Threshold (px²)", color="#90cdf4", fontsize=10)
        ax.set_ylabel("Cells Detected", color="#90cdf4", fontsize=10)
        ax.tick_params(colors="#90cdf4", labelsize=9)
        ax.spines[:].set_color("#1a3a5c")
        ax.grid(color="#1a3751", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend(fontsize=9, facecolor="#0d1526", labelcolor="white", framealpha=0.8)
        ax.set_title("How cell count changes with Min Area cutoff", color="#bee3f8", fontsize=11, pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.session_state["sens_bytes"] = save_figure_bytes_util(fig)
        plt.close(fig)
    else:
        st.session_state["sens_bytes"] = None

# ─── Method Analysis (Modular) ──────────────────────────────────────────────

# ─── Method Analysis ────────────────────────────────────────────────────────

if comparison_methods:
    st.markdown('<div class="section-header"><h2>⚖️ Method Comparison</h2></div>', unsafe_allow_html=True)
    with st.expander(f"⚖️ Compare: {method} vs {', '.join(comparison_methods)}", expanded=False):
        comparison_results = []
        for comp_method in comparison_methods:
            with st.spinner(f"Running {comp_method}…"):
                binary_comp = segment_image(
                    enhanced.tobytes(), enhanced.shape, comp_method, invert, watershed_thresh, fill_holes,
                    manual_threshold=manual_threshold,
                    adaptive_block=adaptive_block,
                    adaptive_c=adaptive_c,
                    morph_adjust=morph_adjust,
                    remove_grid=remove_grid,
                    split_touching=separate_touching,
                    core_clahe_clip=core_clahe_clip,
                    enhanced_clahe_clip=enhanced_clahe_clip,
                    enhanced_tophat_kernel_size=enhanced_tophat_kernel_size,
                )
            detected_comp, df_comp = count_cells_analysis(
                binary_comp, roi, min_area_effective, max_area_effective,
                box_color, pixel_size, scale_bar_um,
                intensity_img=enhanced if enable_intensity else None,
                min_circularity=min_circularity,
            )
            comparison_results.append((comp_method, binary_comp, detected_comp, df_comp))

        st.markdown("**Segmented Outputs**")
        img_cols = st.columns(len(comparison_methods) + 1)
        with img_cols[0]:
            st.image(detected_img, caption=f"🔵 {method} — {total_cells} cells", use_container_width=True)
        for idx, (comp_method, _, detected_comp, df_comp) in enumerate(comparison_results, start=1):
            with img_cols[idx]:
                st.image(detected_comp, caption=f"🟠 {comp_method} — {len(df_comp)} cells", use_container_width=True)

        st.markdown("**Binary Masks**")
        mask_cols = st.columns(len(comparison_methods) + 1)
        with mask_cols[0]:
            st.image(binary, caption=f"{method} — Binary Mask", use_container_width=True, clamp=True)
        for idx, (comp_method, binary_comp, _, df_comp) in enumerate(comparison_results, start=1):
            with mask_cols[idx]:
                st.image(binary_comp, caption=f"{comp_method} — Binary Mask", use_container_width=True, clamp=True)

        st.markdown("**Quantitative Comparison**")
        avg_area_base = df["Area (px²)"].mean() if not df.empty else 0.0
        avg_circ_base = df["Circularity"].mean() if not df.empty else 0.0

        cmp_data = {
            "Metric": ["Cell Count", "Avg Area (px²)", "Avg Circularity", "Processing Speed"],
            method: [
                total_cells,
                f"{avg_area_base:.1f}",
                f"{avg_circ_base:.4f}",
                "Faster" if method != "Watershed" else "Slower",
            ],
        }
        for comp_method, _, _, df_comp in comparison_results:
            avg_area_comp = df_comp["Area (px²)"].mean() if not df_comp.empty else 0.0
            avg_circ_comp = df_comp["Circularity"].mean() if not df_comp.empty else 0.0
            cmp_data[comp_method] = [
                len(df_comp),
                f"{avg_area_comp:.1f}",
                f"{avg_circ_comp:.4f}",
                "Faster" if comp_method != "Watershed" else "Slower",
            ]

        cmp_df = pd.DataFrame(cmp_data)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.markdown("**Critical Analysis**")
        for comp_method, _, _, df_comp in comparison_results:
            key = (method, comp_method) if (method, comp_method) in CRITICAL_ANALYSIS else (comp_method, method)
            analysis_text = CRITICAL_ANALYSIS.get(
                key,
                f"Both {method} and {comp_method} perform binary segmentation on the enhanced "
                f"grayscale image but differ in how they compute the decision boundary. "
                f"The method yielding a cell count closer to the ground truth (manual count) "
                f"should be preferred for this image type."
            )
            st.markdown(f"""
            <div class="ai-box">
              <div class="ai-badge">⚖️ Critical Analysis · {method} vs {comp_method}</div><br/>
              {analysis_text}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Select at least one method to compare in the sidebar.")

# ─── AI Interpretation ────────────────────────────────────────────────────────

st.markdown(f'<div class="section-header"><h2>🧬 AI Interpretation: {ai_mode}</h2></div>', unsafe_allow_html=True)

# Show current AI settings as info pills
ai_status = (
    "Enabled" if gemini_available else
    "Disabled: API key missing" if GEMINI_API_KEY is None else
    "Disabled: google.generativeai not installed"
)

status_color = "#48bb78" if gemini_available else "#f6ad55"

st.markdown(
    f"<div style='display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem;'>"
    f"<span style='background:rgba(99,179,237,0.12);border:1px solid rgba(99,179,237,0.3);"
    f"border-radius:20px;padding:2px 10px;font-size:0.75rem;color:#90cdf4;'>📊 {ai_format}</span>"
    f"<span style='background:rgba(99,179,237,0.12);border:1px solid rgba(99,179,237,0.3);"
    f"border-radius:20px;padding:2px 10px;font-size:0.75rem;color:#90cdf4;'>📌 {ai_detail}</span>"
    f"<span style='background:rgba(99,179,237,0.12);border:1px solid rgba(99,179,237,0.3);"
    f"border-radius:20px;padding:2px 10px;font-size:0.75rem;color:#90cdf4;'>🧫 {cell_type}</span>"
    f"<span style='background:rgba(72,187,120,0.12);border:1px solid rgba(72,187,120,0.3);"
    f"border-radius:20px;padding:2px 10px;font-size:0.75rem;color:{status_color};'>🟢 AI Status: {ai_status}</span>"
    f"</div>",
    unsafe_allow_html=True
)

if st.button(f"🤖 Generate {ai_mode}", type="primary", use_container_width=True):
    with st.spinner(f"Generating {ai_mode.lower()} with Gemini…"):
        ai_text = get_ai_interpretation_ai(
            total_cells, avg_area_um2, avg_circ, cell_type, ai_mode,
            pixel_size, method, df,
            cell_density=cell_density,
            confluence_pct=confluence_pct,
            avg_aspect=avg_aspect,
            avg_solidity=avg_solidity,
            ai_detail=ai_detail,
            ai_format=ai_format,
        )
    st.markdown(f"""
    <div class="ai-box">
      <div class="ai-badge">🧬 Gemini · {ai_mode} · {cell_type} · {ai_format} · {ai_detail}</div><br/>
      {ai_text}
    </div>
    """, unsafe_allow_html=True)
    st.session_state["ai_text"] = ai_text
else:
    if "ai_text" in st.session_state:
        st.markdown(f"""
        <div class="ai-box">
          <div class="ai-badge">🧬 Gemini · {ai_mode} · {cell_type} · {ai_format} · {ai_detail}</div><br/>
          {st.session_state['ai_text']}
        </div>
        """, unsafe_allow_html=True)

ai_text_for_export = st.session_state.get("ai_text", offline_interpretation(total_cells, avg_area_um2, avg_circ, "Basic Summary", ai_format))

def js_download_button(label: str, data: bytes, ext: str, mime: str, default_name: str="CellQuantX_Data"):
    import base64
    import streamlit.components.v1 as components
    b64 = base64.b64encode(data).decode()
    html = f"""
    <div style="margin: 0; padding: 0; display: flex; box-sizing: border-box;">
        <button onclick="downloadPrompt()" style="
            width: 100%;
            background-color: #ffffff;
            color: #31333F;
            border: 1px solid rgba(49, 51, 63, 0.2);
            padding: 8px 14px;
            font-size: 16px;
            border-radius: 0.5rem;
            cursor: pointer;
            font-family: 'Source Sans Pro', sans-serif;
        " onmouseover="this.style.borderColor='#FF4B4B'; this.style.color='#FF4B4B';" 
           onmouseout="this.style.borderColor='rgba(49, 51, 63, 0.2)'; this.style.color='#31333F';">
            {label}
        </button>
    </div>
    <script>
    function downloadPrompt() {{
        var user_name = prompt("Enter file name for your {ext.upper()} download:", "{default_name}");
        if (user_name) {{
            if (!user_name.toLowerCase().endsWith(".{ext}")) {{
                user_name = user_name + ".{ext}";
            }}
            var link = document.createElement("a");
            link.download = user_name;
            link.href = "data:{mime};base64,{b64}";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
    }}
    </script>
    """
    components.html(html, height=55)

# ─── Batch Summary Table ──────────────────────────────────────────────────────
batch_df = None
if len(uploaded_files) > 1 and show_batch_summary:
    st.markdown('<div class="section-header"><h2>🗂️ Batch Summary</h2></div>', unsafe_allow_html=True)

    _batch_rows = []
    for _f in uploaded_files:
        try:
            _f.seek(0)
            _pil = Image.open(_f)
            _arr = ensure_rgb_image(np.array(_pil))
            if resize_before_processing:
                _arr = cv2.resize(_arr, (512, 512), interpolation=cv2.INTER_AREA)
            _g, _enh = preprocess_image(
                _arr.tobytes(), _arr.shape, apply_tophat, apply_scharr, flat_field, channel,
                blur_radius=blur_radius,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                tophat_kernel_size=tophat_kernel_size,
            )
            _bin = segment_image(
                _enh.tobytes(), _enh.shape, method, invert, watershed_thresh, fill_holes,
                manual_threshold=manual_threshold,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                morph_adjust=morph_adjust,
                remove_grid=remove_grid,
                split_touching=separate_touching,
                core_clahe_clip=core_clahe_clip,
                enhanced_clahe_clip=enhanced_clahe_clip,
                enhanced_tophat_kernel_size=enhanced_tophat_kernel_size,
            )
            _min_area_eff = min_area
            if min_diameter_area > 0:
                _min_area_eff = max(_min_area_eff, min_diameter_area)
            _max_area_eff = max_area
            if max_diameter_area > 0:
                _max_area_eff = _max_area_eff if _max_area_eff > 0 else max_diameter_area
                if max_area > 0:
                    _max_area_eff = min(_max_area_eff, max_diameter_area)
            _, _dft = count_cells_analysis(
                _bin, _arr, _min_area_eff, _max_area_eff,
                box_color, pixel_size, scale_bar_um,
                intensity_img=_enh,
                min_circularity=min_circularity,
            )
            _n  = len(_dft)
            _h, _w = _arr.shape[:2]
            _area_mm2 = _w * _h * (pixel_size ** 2) / 1_000_000
            _density  = round(_n / _area_mm2, 2) if _area_mm2 > 0 else 0.0
            _conf     = round(np.sum(_bin > 0) / _bin.size * 100, 1)
            _avg_a    = round(_dft["Area (px²)"].mean() * pixel_size ** 2, 2) if _n > 0 else 0.0
            _avg_c    = round(_dft["Circularity"].mean(), 4) if _n > 0 else 0.0
            _avg_ar   = round(_dft["Aspect Ratio"].mean(), 4) if _n > 0 else 0.0
            _avg_sol  = round(_dft["Solidity"].mean(), 4) if _n > 0 else 0.0
            _batch_rows.append({
                "Image": _f.name,
                "Cells": _n,
                "Density (cells/mm²)": _density,
                "Confluence (%)": _conf,
                "Avg Area (µm²)": _avg_a,
                "Avg Circularity": _avg_c,
                "Avg Aspect Ratio": _avg_ar,
                "Avg Solidity": _avg_sol,
            })
        except Exception as e:
            st.warning(f"Batch processing error for {_f.name}: {e}")
            logger.warning("Batch processing failed for %s: %s", _f.name, e)
            _batch_rows.append({"Image": _f.name, "Cells": "Error",
                                "Density (cells/mm²)": "—", "Confluence (%)": "—",
                                "Avg Area (µm²)": "—", "Avg Circularity": "—",
                                "Avg Aspect Ratio": "—", "Avg Solidity": "—"})

    batch_df = pd.DataFrame(_batch_rows)
    st.dataframe(batch_df, use_container_width=True, hide_index=True)

# ─── Expert Features ─────────────────────────────────────────────────────────

if enable_heatmap and not df.empty:
    st.markdown('<div class="section-header"><h2>🔥 Spatial Density Heatmap</h2></div>', unsafe_allow_html=True)
    st.markdown(_EXPERT_BADGE, unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#718096;'>Kernel-density heatmap of cell centroid positions. "
        "Red = high cell concentration, Blue = sparse. Use this to identify clustering zones, "
        "migration fronts, or uneven seeding across the field of view.</small>",
        unsafe_allow_html=True
    )

    heatmap_img = generate_heatmap_analysis(binary.shape, df)
    heatmap_overlay = cv2.addWeighted(roi, 0.55, heatmap_img, 0.45, 0)

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.image(roi, caption="📷 Original ROI", use_container_width=True)
    with h_col2:
        st.image(heatmap_overlay, caption="🔥 Density Heatmap Overlay (Red = Dense)", use_container_width=True)

    # ── Quadrant Analysis ────────────────────────────────────────────────────
    _qh, _qw = roi.shape[:2]
    _cx_mid, _cy_mid = _qw // 2, _qh // 2
    _q_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    _q_counts = [
        len(df[(df["Centroid X"] < _cx_mid) & (df["Centroid Y"] < _cy_mid)]),
        len(df[(df["Centroid X"] >= _cx_mid) & (df["Centroid Y"] < _cy_mid)]),
        len(df[(df["Centroid X"] < _cx_mid) & (df["Centroid Y"] >= _cy_mid)]),
        len(df[(df["Centroid X"] >= _cx_mid) & (df["Centroid Y"] >= _cy_mid)]),
    ]
    _q_total = sum(_q_counts) or 1
    st.markdown("**📐 Quadrant Cell Distribution**")
    q_cols = st.columns(4)
    _q_colors = ["#63b3ed", "#48bb78", "#f6e05e", "#f56565"]
    for _qci, (_ql, _qn) in enumerate(zip(_q_labels, _q_counts)):
        with q_cols[_qci]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='color:{_q_colors[_qci]};'>{_qn}</div>"
                f"<div class='label'>{_ql}<br>{_qn/_q_total*100:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )

if enable_phenotype and not df.empty:
    st.markdown('<div class="section-header"><h2>🧬 Phenotypic Clustering</h2></div>', unsafe_allow_html=True)
    st.markdown(_EXPERT_BADGE, unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#718096;'>Rule-based morphological classification. "
        "🟢 <b>Compact-Round</b> — high circularity (&gt;0.75) + solidity (&gt;0.85). "
        "🔵 <b>Elongated</b> — aspect ratio &gt;1.8. "
        "🔴 <b>Irregular</b> — all others (including debris, clumps, fragmented).</small>",
        unsafe_allow_html=True
    )

    _PHENO_COLORS_MAP = {
        "Compact-Round":    "#48bb78",
        "Elongated": "#4299e1",
        "Irregular": "#f56565",
    }
    pheno_counts = df["Phenotype"].value_counts().reset_index()
    pheno_counts.columns = ["Phenotype", "Count"]
    pheno_counts["Color"] = pheno_counts["Phenotype"].map(_PHENO_COLORS_MAP)
    _pheno_palette = pheno_counts["Color"].tolist()

    # ── Summary pills ────────────────────────────────────────────────────────
    _pill_html = "<div style='display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:1rem;'>"
    for _, _pr in pheno_counts.iterrows():
        _pct = _pr["Count"] / len(df) * 100
        _pill_html += (
            f"<div style='background:rgba(99,179,237,0.08);border:1px solid {_pr['Color']}33;"
            f"border-left:4px solid {_pr['Color']};border-radius:8px;padding:0.5rem 1rem;'>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{_pr['Color']};'>{_pr['Count']}</div>"
            f"<div style='font-size:0.72rem;color:#90cdf4;text-transform:uppercase;letter-spacing:0.07em;'>{_pr['Phenotype']}</div>"
            f"<div style='font-size:0.8rem;color:#718096;'>{_pct:.1f}% of population</div>"
            f"</div>"
        )
    _pill_html += "</div>"
    st.markdown(_pill_html, unsafe_allow_html=True)

    # ── Charts: Pie + Bar ────────────────────────────────────────────────────
    pc1, pc2 = st.columns(2)
    with pc1:
        if show_pheno_chart:
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            fig_pie.patch.set_facecolor("#0a1020")
            ax_pie.set_facecolor("#0d1a2e")
            wedges, texts, autotexts = ax_pie.pie(
                pheno_counts["Count"], labels=pheno_counts["Phenotype"],
                autopct='%1.1f%%', colors=_pheno_palette,
                textprops={'color': "white", 'fontsize': 9},
                wedgeprops={'linewidth': 1.5, 'edgecolor': '#0a1020'},
                startangle=90,
            )
            for at in autotexts:
                at.set_fontsize(8)
            ax_pie.set_title("Phenotype Distribution", color="#90cdf4", fontsize=10, pad=10)
            st.pyplot(fig_pie)
            plt.close(fig_pie)

    with pc2:
        if show_pheno_chart:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            fig_bar.patch.set_facecolor("#0a1020")
            ax_bar.set_facecolor("#0d1a2e")
            bars = ax_bar.barh(pheno_counts["Phenotype"], pheno_counts["Count"],
                               color=_pheno_palette, edgecolor="#0a1020", height=0.5)
            for bar, cnt in zip(bars, pheno_counts["Count"]):
                ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                            str(cnt), va='center', color='white', fontsize=9)
            ax_bar.set_xlabel("Cell Count", color="#90cdf4", fontsize=9)
            ax_bar.tick_params(colors="white", labelsize=8)
            ax_bar.spines[:].set_color("#1a3a5c")
            ax_bar.set_title("Phenotype Counts", color="#90cdf4", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

    if enable_phenotype and not df.empty:
        fig_pheno, (ax_pie_export, ax_bar_export) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0a1020")
        ax_pie_export.set_facecolor("#0d1a2e")
        ax_bar_export.set_facecolor("#0d1a2e")
        wedges, texts, autotexts = ax_pie_export.pie(
            pheno_counts["Count"], labels=pheno_counts["Phenotype"],
            autopct='%1.1f%%', colors=_pheno_palette,
            textprops={'color': "white", 'fontsize': 8},
            wedgeprops={'linewidth': 1.5, 'edgecolor': '#0a1020'},
            startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax_pie_export.set_title("Phenotype Distribution", color="#90cdf4", fontsize=10, pad=8)
        bars = ax_bar_export.barh(pheno_counts["Phenotype"], pheno_counts["Count"],
                                  color=_pheno_palette, edgecolor="#0a1020", height=0.55)
        for bar, cnt in zip(bars, pheno_counts["Count"]):
            ax_bar_export.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                               str(cnt), va='center', color='white', fontsize=8)
        ax_bar_export.set_xlabel("Cell Count", color="#90cdf4", fontsize=9)
        ax_bar_export.tick_params(colors="white", labelsize=8)
        ax_bar_export.spines[:].set_color("#1a3a5c")
        ax_bar_export.set_title("Phenotype Counts", color="#90cdf4", fontsize=10, pad=8)
        plt.tight_layout()
        st.session_state["pheno_bytes"] = save_figure_bytes_util(fig_pheno)
        plt.close(fig_pheno)
    else:
        st.session_state["pheno_bytes"] = None

    # ── Per-phenotype statistics table ───────────────────────────────────────
    st.markdown("**📊 Per-Phenotype Statistics**")
    _pheno_stat_rows = []
    for _ph in pheno_counts["Phenotype"]:
        _sub = df[df["Phenotype"] == _ph]
        _pheno_stat_rows.append({
            "Phenotype":       _ph,
            "Count":           len(_sub),
            "% of Total":      f"{len(_sub)/len(df)*100:.1f}%",
            "Avg Area (µm²)":  f"{(_sub['Area (px²)'].mean() * pixel_size**2):.1f}" if len(_sub) > 0 else "—",
            "Avg Circularity": f"{_sub['Circularity'].mean():.3f}" if len(_sub) > 0 else "—",
            "Avg Aspect Ratio":f"{_sub['Aspect Ratio'].mean():.3f}" if len(_sub) > 0 else "—",
            "Avg Solidity":    f"{_sub['Solidity'].mean():.3f}" if len(_sub) > 0 else "—",
            "Avg Intensity":   f"{_sub['Mean Intensity'].mean():.1f}" if (len(_sub) > 0 and enable_intensity) else "—",
        })
    st.dataframe(pd.DataFrame(_pheno_stat_rows), use_container_width=True, hide_index=True)

    # ── Color-annotated overlay by phenotype ────────────────────────────────
    _pheno_overlay = roi.copy()
    _PHENO_BGR = {
        "Compact-Round":     (72, 187, 120),   # green
        "Elongated": (66, 153, 225),   # blue
        "Irregular":  (245, 101, 101),  # red
    }
    contours_all_p, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_p in contours_all_p:
        area_p = cv2.contourArea(cnt_p)
        if area_p < min_area:
            continue
        M_p = cv2.moments(cnt_p)
        if M_p["m00"] == 0:
            continue
        cx_p = int(M_p["m10"] / M_p["m00"])
        cy_p = int(M_p["m01"] / M_p["m00"])
        perim_p = cv2.arcLength(cnt_p, True)
        circ_p  = min((4 * np.pi * area_p) / (perim_p ** 2), 1.0) if perim_p > 0 else 0.0
        x_p, y_p, w_p, h_p = cv2.boundingRect(cnt_p)
        ar_p = w_p / h_p if h_p > 0 else 1.0
        hull_p = cv2.convexHull(cnt_p)
        hull_area_p = cv2.contourArea(hull_p)
        sol_p = area_p / hull_area_p if hull_area_p > 0 else 1.0
        if circ_p > 0.75 and sol_p > 0.85:
            ph_key = "Compact-Round"
        elif ar_p > 1.8:
            ph_key = "Elongated"
        else:
            ph_key = "Irregular"
        _color_bgr = _PHENO_BGR[ph_key]
        cv2.drawContours(_pheno_overlay, [cnt_p], -1, _color_bgr, 2)

    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.image(_pheno_overlay,
                 caption="🎨 Color-annotated by morphology group — 🟢 Compact-Round  🔵 Elongated  🔴 Irregular",
                 width=460)

# ─── Expert Feature 3: Intensity Quantification ──────────────────────────────

if enable_intensity and not df.empty and "Mean Intensity" in df.columns:
    st.markdown('<div class="section-header"><h2>💡 Intensity Quantification</h2></div>', unsafe_allow_html=True)
    st.markdown(_EXPERT_BADGE, unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#718096;'>Per-cell fluorescence intensity analysis. "
        "Quantify expression levels, identify subpopulations, and correlate morphology with brightness.</small>",
        unsafe_allow_html=True
    )

    _int_vals = df["Mean Intensity"]
    _int_min, _int_max, _int_mean, _int_med = (
        _int_vals.min(), _int_vals.max(), _int_vals.mean(), _int_vals.median()
    )
    # Tier thresholds (33rd / 66th percentile)
    _t33, _t66 = _int_vals.quantile(0.33), _int_vals.quantile(0.66)
    _n_low  = int((_int_vals < _t33).sum())
    _n_mid  = int(((_int_vals >= _t33) & (_int_vals < _t66)).sum())
    _n_high = int((_int_vals >= _t66).sum())

    # ── Intensity summary cards ──────────────────────────────────────────────
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#f6e05e;'>{_int_mean:.1f}</div>"
                    f"<div class='label'>Mean Intensity (ADU)</div></div>", unsafe_allow_html=True)
    with i2:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#68d391;'>{_n_high}</div>"
                    f"<div class='label'>High Intensity Cells</div></div>", unsafe_allow_html=True)
    with i3:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#63b3ed;'>{_n_mid}</div>"
                    f"<div class='label'>Mid Intensity Cells</div></div>", unsafe_allow_html=True)
    with i4:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#fc8181;'>{_n_low}</div>"
                    f"<div class='label'>Low Intensity Cells</div></div>", unsafe_allow_html=True)

    # ── Intensity histogram with tier shading ────────────────────────────────
    fig_int, ax_int = plt.subplots(figsize=(10, 3.5))
    fig_int.patch.set_facecolor("#0a1020")
    ax_int.set_facecolor("#0d1a2e")
    ax_int.hist(_int_vals, bins=30, color="#f6e05e", alpha=0.85, edgecolor="#0a1020")
    ax_int.axvline(_t33, color="#fc8181", ls="--", lw=1.2, label=f"33rd pct ({_t33:.0f})")
    ax_int.axvline(_t66, color="#68d391", ls="--", lw=1.2, label=f"66th pct ({_t66:.0f})")
    ax_int.axvline(_int_mean, color="#90cdf4", ls="-",  lw=1.5, label=f"Mean ({_int_mean:.0f})")
    ax_int.set_xlabel("Mean Intensity (ADU)", color="#90cdf4")
    ax_int.set_ylabel("Cell Count", color="#90cdf4")
    ax_int.tick_params(colors="white")
    ax_int.spines[:].set_color("#1a3a5c")
    ax_int.legend(fontsize=8, facecolor="#0d1a2e", labelcolor="white")
    ax_int.set_title("Per-Cell Intensity Distribution", color="#90cdf4", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_int)
    buf_int = io.BytesIO()
    fig_int.savefig(buf_int, format="png", bbox_inches="tight", facecolor="#0a1020")
    st.session_state["intensity_bytes"] = buf_int.getvalue()
    plt.close(fig_int)

    # ── Per-phenotype intensity if phenotype feature also enabled ────────────
    if enable_phenotype and "Phenotype" in df.columns:
        st.markdown("**📊 Intensity by Phenotype**")
        _int_pheno = (
            df.groupby("Phenotype")["Mean Intensity"]
            .agg(["mean", "median", "min", "max", "std"])
            .round(2)
            .reset_index()
        )
        _int_pheno.columns = ["Phenotype", "Mean", "Median", "Min", "Max", "Std Dev"]
        st.dataframe(_int_pheno, use_container_width=True, hide_index=True)

# ─── EXPORT RESULTS ────────────────────────────────────────────────────────

# ─── Export: CSV, Annotated Image & PDF ─────────────────────────────────────

st.markdown('<div class="section-header"><h2>💾 Export Results</h2></div>', unsafe_allow_html=True)

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    if not df.empty:
        df_export = df.copy()
        df_export["Area (µm²)"] = (df_export["Area (px²)"] * pixel_size ** 2).round(4)
        export_cols = ["Cell ID", "Area (px²)", "Area (µm²)", "Circularity", "Aspect Ratio", "Solidity"]
        if enable_intensity and "Mean Intensity" in df_export.columns:
            export_cols.append("Mean Intensity")
        if enable_phenotype and "Phenotype" in df_export.columns:
            export_cols.append("Phenotype")
        csv_bytes = df_export[export_cols].to_csv(index=False).encode("utf-8")
        js_download_button("📄 Download CSV", csv_bytes, "csv", "text/csv")
    else:
        st.info("No data to export.")

with exp_col2:
    # Download annotated image as PNG
    _img_pil = Image.fromarray(detected_img)
    _img_buf = io.BytesIO()
    _img_pil.save(_img_buf, format="PNG")
    js_download_button("🖼️ Download Annotated Image", _img_buf.getvalue(),
                       "png", "image/png", "CellQuantX_Annotated")

with exp_col3:
    if reportlab_available:
        if not df.empty:
            df_pdf = df.copy()
            df_pdf["Area (µm²)"] = (df_pdf["Area (px²)"] * pixel_size ** 2).round(4)
            try:
                pdf_bytes = generate_pdf(
                    total_cells, avg_area_um2, avg_circ,
                    ai_text_for_export, df_pdf,
                    roi, detected_img,
                    st.session_state.get("hist_bytes"),
                    st.session_state.get("sens_bytes"),
                    st.session_state.get("scat_bytes"),
                    st.session_state.get("heat_bytes"),
                    st.session_state.get("pheno_bytes"),
                    st.session_state.get("intensity_bytes"),
                    cell_density=cell_density,
                    confluence_pct=confluence_pct,
                    mean_aspect=avg_aspect,
                    mean_solidity=avg_solidity,
                    mean_intensity=avg_intensity,
                    image_name=selected_name,
                    cell_type=cell_type,
                    method=method,
                    pixel_size=pixel_size,
                    batch_df=batch_df if len(uploaded_files) > 1 else None,
                )
                js_download_button("📑 Download PDF Report", pdf_bytes, "pdf", "application/pdf", "CellQuantX_Report")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        else:
            st.info("No data to export.")
    else:
        st.info("📦 Install **reportlab** (`pip install reportlab`) to enable PDF export.")

