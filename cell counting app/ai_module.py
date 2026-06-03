import logging
import os

import pandas as pd
import streamlit as st

logger = logging.getLogger("cellquantx")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-flash-latest"]

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
    else:
        gemini_available = False
except ImportError:
    genai = None
    gemini_available = False


def offline_interpretation(n: int, avg_area: float, avg_circ: float, mode: str, fmt: str = "Paragraph") -> str:
    density = "high" if n > 100 else "moderate" if n > 30 else "low"
    shape = "highly circular" if avg_circ > 0.8 else "moderately circular" if avg_circ > 0.5 else "irregular"
    if fmt == "Bullet Points":
        return f"• Cell count: {n} ({density})\n• Mean area: {avg_area:.1f} µm²\n• Mean circularity: {avg_circ:.3f} ({shape})"
    return f"Detected {n} cells with mean area {avg_area:.1f} µm² and mean circularity {avg_circ:.3f} ({shape}, {density} density)."


def get_ai_interpretation(
    n: int, avg_area: float, avg_circ: float, cell_type: str, mode: str, pixel_size: float, method: str, df: pd.DataFrame,
    cell_density: float = 0.0, confluence_pct: float = 0.0, avg_aspect: float = 0.0, avg_solidity: float = 0.0,
    ai_detail: str = "Standard", ai_format: str = "Paragraph",
) -> str:
    base_stats = (
        f"Cell type: {cell_type}. Total cells: {n}. Mean area: {avg_area:.2f} µm². Mean circularity: {avg_circ:.4f}. "
        f"Mean aspect ratio: {avg_aspect:.4f}. Mean solidity: {avg_solidity:.4f}. "
        f"Cell density: {cell_density:.1f} cells/mm². Confluence: {confluence_pct:.1f}%. "
        f"Method: {method}. Calibration: {pixel_size} µm/pixel."
    )
    prompt = f"You are a cell biology expert. Data: {base_stats}. Provide a {ai_detail.lower()} {ai_format.lower()} interpretation."

    if not gemini_available:
        return "AI feature not available. Please configure API key."

    errors = []
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            errors.append(str(e))
            logger.warning("Gemini failed on %s: %s", model_name, e)
    if errors:
        st.warning(f"AI model call failed; using offline interpretation. Last error: {errors[-1]}")
    return offline_interpretation(n, avg_area, avg_circ, mode, ai_format)
