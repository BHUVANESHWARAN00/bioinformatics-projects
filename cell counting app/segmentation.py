import cv2
import numpy as np
import streamlit as st


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
    tophat_kernel_size: int = 31,
):
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
    if img.ndim == 3:
        if channel == "Auto-Detect (Brightest)":
            means = [img[:, :, i].mean() for i in range(3)]
            gray = img[:, :, int(np.argmax(means))]
        elif channel == "Red (R)":
            gray = img[:, :, 0]
        elif channel == "Green (G)":
            gray = img[:, :, 1]
        elif channel == "Blue (B)":
            gray = img[:, :, 2]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    if flat_field_correct:
        gray_f = gray.astype(np.float32)
        sigma = max(gray_f.shape) * 0.05
        ksize = int(sigma * 6) | 1
        flat = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
        flat = np.where(flat < 1, 1, flat)
        corrected = gray_f / flat * 128.0
        gray = np.clip(corrected, 0, 255).astype(np.uint8)

    if disable_heavy_preprocessing:
        return gray, gray

    if apply_tophat:
        tk = max(3, int(tophat_kernel_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tk, tk))
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


def _mask_quality_score(mask: np.ndarray) -> float:
    fg_ratio = cv2.countNonZero(mask) / float(mask.size)
    if fg_ratio <= 0.0:
        return -1e9
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return -1e6
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.sum((areas >= 8) & (areas <= 20000))
    tiny = np.sum(areas < 8)
    ratio_penalty = abs(fg_ratio - 0.15) * 120.0
    tiny_penalty = tiny * 0.2
    return float(valid) - ratio_penalty - tiny_penalty


def _auto_select_polarity(binary: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(binary)
    return inv if _mask_quality_score(inv) > (_mask_quality_score(binary) * 1.25) else binary


def _remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape[:2]
    horiz_len = max(40, w // 20)
    vert_len = max(40, h // 20)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    mask_lines = cv2.bitwise_or(horiz_lines, vert_lines)
    return cv2.bitwise_and(binary, cv2.bitwise_not(mask_lines))


def _remove_tiny_speckles(binary: np.ndarray, min_px: int = 4) -> np.ndarray:
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
    core_clahe_clip: float = 2.2,
    core_clahe_grid: int = 8,
    enhanced_clahe_clip: float = 3.0,
    enhanced_tophat_kernel_size: int = 21,
):
    processed = np.frombuffer(processed_bytes, dtype=np.uint8).reshape(shape)
    processed = cv2.medianBlur(processed, 3)
    core_grid = max(2, int(core_clahe_grid))
    _clahe_core = cv2.createCLAHE(clipLimit=core_clahe_clip, tileGridSize=(core_grid, core_grid))
    proc_for_thresh = _clahe_core.apply(processed)

    if method == "Enhanced Labeling Mode":
        e_grid = max(2, int(core_clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=enhanced_clahe_clip, tileGridSize=(e_grid, e_grid))
        boosted = clahe.apply(processed)
        # Use a more robust Top-Hat with a slightly larger kernel for better background flattening
        et = max(5, int(enhanced_tophat_kernel_size) | 1)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (et, et))
        tophat = cv2.morphologyEx(boosted, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Multi-scale cleanup for Enhanced mode
        blur = cv2.bilateralFilter(tophat, 5, 45, 45)
        th_tri, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        # Extremely sensitive threshold for faint cells
        _, binary = cv2.threshold(blur, max(3.0, th_tri * 0.45), 255, cv2.THRESH_BINARY)
        
        # Small morphological closing to bridge gaps in faint cell rims
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    elif manual_threshold > 0:
        _, binary = cv2.threshold(proc_for_thresh, manual_threshold, 255, cv2.THRESH_BINARY)
    elif method == "Adaptive Threshold":
        adaptive_block = max(3, adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1)
        binary_g = cv2.adaptiveThreshold(proc_for_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block, adaptive_c)
        binary_m = cv2.adaptiveThreshold(proc_for_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptive_block, adaptive_c)
        binary = cv2.bitwise_or(binary_g, binary_m)
    else:
        th_otsu, _ = cv2.threshold(proc_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_tri, _ = cv2.threshold(proc_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        hybrid_th = np.clip(((th_otsu * 0.2) + (th_tri * 0.8)) * 0.9, 1, 254)
        _, binary = cv2.threshold(proc_for_thresh, hybrid_th, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary) if invert else _auto_select_polarity(binary)
    if remove_grid:
        binary = _remove_grid_lines(binary)

    if fill_holes:
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        h_p, w_p = padded.shape
        mask = np.zeros((h_p + 2, w_p + 2), np.uint8)
        cv2.floodFill(padded, mask, (0, 0), 255)
        binary = binary | cv2.bitwise_not(padded)[1:-1, 1:-1]

    kernel = np.ones((3, 3), np.uint8)
    if morph_adjust < 0:
        binary = cv2.erode(binary, kernel, iterations=min(5, -morph_adjust))
    elif morph_adjust > 0:
        binary = cv2.dilate(binary, kernel, iterations=min(5, morph_adjust))

    if method != "Enhanced Labeling Mode":
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        # Permissive cleanup for enhanced mode
        binary = _remove_tiny_speckles(binary, min_px=3)

    if method == "Watershed" or split_touching:
        # Improved Watershed with refined seeding logic
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = binary.copy()
        
        # Robust background
        sure_bg = cv2.dilate(opening, kernel_e, iterations=3)
        
        # Highly accurate distance transform
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Multi-stage blurring of distance map to suppress noise-induced over-segmentation
        dist_s = cv2.GaussianBlur(dist, (5, 5), 0.8)
        
        # Adaptive peak detection: distance peaks must be local maxima
        nms_k = max(3, int(7 * watershed_thresh)) | 1
        local_max = cv2.dilate(dist_s, np.ones((nms_k, nms_k), np.uint8))
        
        # Improved noise floor: dynamic cutoff based on local distance distribution
        if np.any(dist_s > 0):
            # Use 5% of max distance plus a small flat epsilon for stability
            noise_floor = (dist_s.max() * (watershed_thresh * 0.08)) + 0.5
        else:
            noise_floor = 0
            
        sure_fg = np.zeros_like(dist_s, dtype=np.uint8)
        # Seeds are pixels where distance == local_max AND above noise floor
        sure_fg[(dist_s >= (local_max - 1e-3)) & (dist_s > noise_floor)] = 255
        
        # Denoise seeds to prevent fragmentation of a single cell center
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        sure_fg = cv2.dilate(sure_fg, np.ones((2, 2), np.uint8), iterations=1)
        
        if cv2.countNonZero(sure_fg) < 2:
            # Fallback for extremely sparse frames
            sure_fg = cv2.erode(opening, kernel_e, iterations=1)
            
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(img_color, markers)
        
        binary = np.zeros_like(processed, dtype=np.uint8)
        binary[markers > 1] = 255
        
        # Enforce separation boundaries
        boundaries = np.zeros_like(processed, dtype=np.uint8)
        boundaries[markers == -1] = 255
        boundaries = cv2.dilate(boundaries, np.ones((3, 3), np.uint8))
        binary[boundaries == 255] = 0

    return _remove_tiny_speckles(binary, min_px=4)
