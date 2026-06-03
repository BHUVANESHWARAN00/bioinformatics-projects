import io

import cv2
import numpy as np


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
    return img.astype(np.uint8)


def save_figure_bytes(fig, facecolor="#0a0e1a"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=facecolor)
    buf.seek(0)
    return buf.getvalue()
