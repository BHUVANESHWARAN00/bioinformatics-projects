import cv2
import numpy as np
import pandas as pd


def generate_heatmap(shape: tuple, df: pd.DataFrame) -> np.ndarray:
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
    return cv2.cvtColor(cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def count_cells(
    binary: np.ndarray,
    original_rgb: np.ndarray,
    min_area: int,
    max_area: int = 0,
    box_color: tuple = (0, 255, 80),
    pixel_size: float = 0.5,
    scale_bar_um: float = 100.0,
    intensity_img: np.ndarray = None,
    min_circularity: float = 0.0,
    method: str = "",
):
    output = original_rgb.copy()
    data = []
    valid_id = 1

    if method == "Enhanced Labeling Mode":
        num_labels, labels = cv2.connectedComponents(binary)
        for label in range(1, num_labels):
            mask = (labels == label).astype("uint8") * 255
            area = cv2.countNonZero(mask)
            if area < min_area or (max_area > 0 and area > max_area):
                continue
            x, y, w, h_box = cv2.boundingRect(mask)
            M = cv2.moments(mask)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h_box // 2
            mean_intensity = round(cv2.mean(intensity_img, mask=mask)[0], 2) if intensity_img is not None else 0.0
            cv2.rectangle(output, (x, y), (x + w, y + h_box), box_color, 2)
            cv2.putText(output, str(valid_id), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            data.append({
                "Cell ID": valid_id, "Area (px²)": round(area, 2), "Centroid X": cx, "Centroid Y": cy,
                "Perimeter": 0.0, "Circularity": 1.0, "Aspect Ratio": round(w / h_box, 4) if h_box > 0 else 1.0,
                "Solidity": 1.0, "Mean Intensity": mean_intensity, "Phenotype": "Compact-Round"
            })
            valid_id += 1
    else:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or (max_area > 0 and area > max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = (min((4 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0)
            if circularity < min_circularity:
                continue
            x, y, w, h_box = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h_box, 4) if h_box > 0 else 0.0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = round(area / hull_area, 4) if hull_area > 0 else 0.0
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h_box // 2
            mean_intensity = 0.0
            if intensity_img is not None:
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = round(cv2.mean(intensity_img, mask=mask)[0], 2)
            phenotype = "Compact-Round" if (circularity > 0.75 and solidity > 0.85) else ("Elongated" if aspect_ratio > 1.8 else "Irregular")
            cv2.rectangle(output, (x, y), (x + w, y + h_box), box_color, 2)
            cv2.putText(output, str(valid_id), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            data.append({
                "Cell ID": valid_id, "Area (px²)": round(area, 2), "Centroid X": cx, "Centroid Y": cy,
                "Perimeter": round(perimeter, 2), "Circularity": round(circularity, 4), "Aspect Ratio": aspect_ratio,
                "Solidity": solidity, "Mean Intensity": mean_intensity, "Phenotype": phenotype
            })
            valid_id += 1

    return output, pd.DataFrame(data)
