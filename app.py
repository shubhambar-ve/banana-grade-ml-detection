"""Local Gradio app for non-destructive banana grade prediction.

The app loads an already-trained Random Forest package, extracts image features
with OpenCV/skimage, accepts the hardware values manually, and predicts Grade
A/B/C. STM32 serial input is intentionally left as a future integration point.
"""


from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import cv2
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import threading
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

try:
    import serial
except ImportError:
    serial = None

SERIAL_PORT = os.getenv("STM32_SERIAL_PORT", "COM13")
ser = None
serial_error = "pyserial is not installed" if serial is None else "not connected"
serial_lock = threading.Lock()
serial_line_buffer = ""


def connect_stm32():
    """Open the configured STM32 port if the serial library and port are available."""
    global ser, serial_error
    if serial is None:
        serial_error = "Install dependencies with: pip install -r requirements.txt"
        return False
    if ser is not None and ser.is_open:
        return True
    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.1)
        serial_error = ""
        return True
    except (OSError, serial.SerialException) as exc:
        ser = None
        serial_error = str(exc)
        return False


connect_stm32()



try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # Older scikit-image releases used British spelling.
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "banana_grade_model_literature_guided_rf_v2.pkl"

HARDWARE_FEATURES = [
    "weight_g",
    "surface_temp_C",
    "ambient_temp_C",
    "relative_humidity_pct",
    "pressure_hPa",
    "gas_resistance_kohm",
]

IMAGE_FEATURES = [
    "peel_green_pct",
    "peel_yellow_pct",
    "peel_brown_spot_pct",
    "peel_other_pct",
    "peel_hue_mean_deg",
    "peel_hue_std_deg",
    "peel_saturation_mean",
    "peel_brightness_mean",
    "peel_texture_contrast",
    "peel_texture_homogeneity",
    "banana_area_fraction",
    "background_uniformity_score",
    "blur_score",
    "exposure_score",
    "image_quality_score",
]

EXPECTED_FEATURE_COLS = HARDWARE_FEATURES + IMAGE_FEATURES
CLASS_LABELS = ["A", "B", "C"]

GRADE_MEANINGS = {
    "A": "good-quality / ripe / market-acceptable",
    "B": "intermediate / borderline / partially ripe",
    "C": "unripe, overripe, or poor quality",
}

VALIDATION_SUMMARY = {
    "Dataset": "banana_literature_guided_nondestructive_ml_dataset_v2.csv",
    "Total rows": 900,
    "Classes": "A, B, C",
    "Validation method": "5-Fold Stratified Group Cross-Validation",
    "Group column": "validation_group_id",
    "Mean CV accuracy": 0.9344,
    "Standard deviation": 0.0482,
    "Minimum fold accuracy": 0.8722,
    "Maximum fold accuracy": 0.9833,
}

CONFUSION_MATRIX = np.array(
    [
        [272, 28, 0],
        [25, 274, 1],
        [0, 5, 295],
    ]
)

FEATURE_IMPORTANCE = {
    "peel_yellow_pct": 0.336356,
    "peel_green_pct": 0.192228,
    "weight_g": 0.110344,
    "peel_brown_spot_pct": 0.079100,
    "peel_hue_mean_deg": 0.063942,
    "peel_texture_contrast": 0.043908,
    "peel_hue_std_deg": 0.038779,
    "surface_temp_C": 0.032923,
    "peel_texture_homogeneity": 0.024317,
    "ambient_temp_C": 0.022873,
    "peel_other_pct": 0.012438,
    "gas_resistance_kohm": 0.008875,
    "peel_brightness_mean": 0.006786,
    "peel_saturation_mean": 0.004634,
    "pressure_hPa": 0.003614,
    "relative_humidity_pct": 0.003469,
    "background_uniformity_score": 0.003346,
    "image_quality_score": 0.003267,
    "banana_area_fraction": 0.003105,
    "blur_score": 0.002995,
    "exposure_score": 0.002701,
}


def load_model_package() -> Tuple[object, List[str], str]:
    """Load the trained model package and extract the stored feature order."""
    try:
        model_package = joblib.load(MODEL_PATH)
        model = model_package["model"]
        feature_cols = list(model_package["feature_cols"])
        return model, feature_cols, ""
    except Exception as exc:
        return None, EXPECTED_FEATURE_COLS, f"Model load failed: {exc}"


MODEL, FEATURE_COLS, MODEL_LOAD_ERROR = load_model_package()


def read_hardware_values_from_stm32():

    line = ser.readline().decode().strip()

    print("STM32:", line)

    match = re.search(
        r'Weight:\s*([\d.]+)\s*g\s*\|\s*Temp:\s*([\d.]+)',
        line
    )

    if match:

        weight = float(match.group(1))
        temp = float(match.group(2))

        return {
            "weight_g": weight,
            "surface_temp_C": temp,
            "ambient_temp_C": 26.0,
            "relative_humidity_pct": 55.0,
            "pressure_hPa": 1013.25,
            "gas_resistance_kohm": 80.0,
        }

    return {
        "weight_g": 120.0,
        "surface_temp_C": 27.0,
        "ambient_temp_C": 26.0,
        "relative_humidity_pct": 55.0,
        "pressure_hPa": 1013.25,
        "gas_resistance_kohm": 80.0,
    }
    """Future hook for STM32 serial integration.

    For this version, the app uses manual Gradio number inputs. Later, this
    function can read and validate values from a serial port before prediction.
    """
    raise NotImplementedError("STM32 serial input is planned for a later version.")


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize Gradio image input to a three-channel uint8 RGB array."""
    if image is None:
        raise ValueError("Please upload one banana image before prediction.")

    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if rgb.ndim != 3:
        raise ValueError("The uploaded file was not recognized as an image.")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.shape[2] != 3:
        raise ValueError("The uploaded image must have RGB channels.")

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if np.nanmax(rgb) <= 1.0:
            rgb *= 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def resize_for_processing(rgb: np.ndarray, max_dim: int = 1000) -> np.ndarray:
    """Limit very large images so the local OpenCV pipeline stays responsive."""
    height, width = rgb.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_dim:
        return rgb

    scale = max_dim / largest_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def get_border_pixels(image: np.ndarray, border_width: int) -> np.ndarray:
    """Collect border pixels used to estimate the mostly uniform background."""
    if image.ndim == 2:
        image = image[:, :, None]
    channels = image.shape[2]
    top = image[:border_width, :, :]
    bottom = image[-border_width:, :, :]
    left = image[:, :border_width, :]
    right = image[:, -border_width:, :]
    return np.concatenate(
        [
            top.reshape(-1, channels),
            bottom.reshape(-1, channels),
            left.reshape(-1, channels),
            right.reshape(-1, channels),
        ],
        axis=0,
    )


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes after thresholding and morphology cleanup."""
    flood = mask.copy()
    h, w = mask.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component as the banana mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise ValueError("Segmentation failed: no banana-sized foreground region was found.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = (labels == largest_label).astype(np.uint8) * 255
    if largest.sum() == 0:
        raise ValueError("Segmentation failed: banana mask is empty.")
    return largest


def segment_banana(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Segment banana by Lab distance from the border-estimated background."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    border_width = max(5, min(height, width) // 20)

    border_lab = get_border_pixels(lab, border_width).astype(np.float32)
    background_lab = np.median(border_lab, axis=0)
    background_std = float(np.mean(np.std(border_lab, axis=0)))
    background_uniformity_score = float(np.clip(1.0 - (background_std / 45.0), 0.0, 1.0))

    # Pixels far from the background color are foreground candidates.
    distance = np.linalg.norm(lab.astype(np.float32) - background_lab, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    distance_u8 = cv2.GaussianBlur(distance_u8, (5, 5), 0)
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The image border should mostly be background. Invert if thresholding flipped it.
    border_mask = get_border_pixels(mask[:, :, None], border_width).reshape(-1)
    if np.mean(border_mask > 0) > 0.50:
        mask = cv2.bitwise_not(mask)

    kernel_size = max(3, (min(height, width) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = fill_mask_holes(mask)
    mask = keep_largest_component(mask)
    return mask.astype(bool), background_uniformity_score


def compute_quality_scores(rgb: np.ndarray, banana_mask: np.ndarray, background_uniformity_score: float) -> Dict[str, float]:
    """Compute simple image quality indicators used by the trained feature set."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip(blur_raw / 500.0, 0.0, 1.0))

    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range_score = float(np.clip((p95 - p05) / 140.0, 0.0, 1.0))
    clipped_fraction = float(np.mean((gray <= 3) | (gray >= 252)))
    clipping_score = float(np.clip(1.0 - (clipped_fraction / 0.35), 0.0, 1.0))
    midtone_score = float(np.clip(1.0 - abs(float(np.mean(gray)) - 150.0) / 150.0, 0.0, 1.0))
    exposure_score = float(np.clip(0.50 * clipping_score + 0.30 * dynamic_range_score + 0.20 * midtone_score, 0.0, 1.0))

    banana_area_fraction = float(np.mean(banana_mask))
    image_quality_score = float(
        np.clip(
            0.40 * background_uniformity_score + 0.30 * blur_score + 0.30 * exposure_score,
            0.0,
            1.0,
        )
    )

    return {
        "banana_area_fraction": banana_area_fraction,
        "background_uniformity_score": background_uniformity_score,
        "blur_score": blur_score,
        "exposure_score": exposure_score,
        "image_quality_score": image_quality_score,
    }


def compute_texture_features(rgb: np.ndarray, banana_mask: np.ndarray) -> Tuple[float, float]:
    """Compute GLCM contrast and homogeneity from the isolated banana crop."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(banana_mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    crop_gray = gray[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = banana_mask[y1 : y2 + 1, x1 : x2 + 1]

    if crop_gray.size == 0 or not np.any(crop_mask):
        return 0.0, 0.0

    filled = crop_gray.copy()
    filled[~crop_mask] = int(np.median(crop_gray[crop_mask]))
    if max(filled.shape) > 256:
        scale = 256 / max(filled.shape)
        filled = cv2.resize(
            filled,
            (max(1, int(filled.shape[1] * scale)), max(1, int(filled.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if min(filled.shape) < 3:
        return 0.0, 0.0

    levels = 32
    quantized = np.clip(filled.astype(np.float32) / (256 / levels), 0, levels - 1).astype(np.uint8)
    try:
        glcm = graycomatrix(
            quantized,
            distances=[1, 2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    except Exception:
        return 0.0, 0.0
    return contrast, homogeneity


def extract_image_features(rgb_input: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, int]:
    """Run segmentation and extract the 15 image features expected by the model."""
    rgb = resize_for_processing(ensure_rgb_uint8(rgb_input))
    banana_mask, background_uniformity_score = segment_banana(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_deg = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1].astype(np.float32)
    brightness = hsv[:, :, 2].astype(np.float32)

    banana_pixels = banana_mask
    pixel_count = int(np.sum(banana_pixels))
    if pixel_count == 0:
        raise ValueError("Segmentation failed: banana mask is empty.")

    # Non-overlapping peel color masks: brown first, then green, then yellow.
    brown = banana_pixels & (
        (brightness < 85)
        | ((hue_deg >= 8) & (hue_deg <= 45) & (saturation >= 35) & (brightness < 170))
    )
    green = banana_pixels & ~brown & (hue_deg >= 70) & (hue_deg <= 160) & (saturation >= 30) & (brightness >= 45)
    yellow = banana_pixels & ~brown & ~green & (hue_deg >= 35) & (hue_deg < 70) & (saturation >= 25) & (brightness >= 70)
    other = banana_pixels & ~brown & ~green & ~yellow

    texture_contrast, texture_homogeneity = compute_texture_features(rgb, banana_mask)
    quality = compute_quality_scores(rgb, banana_mask, background_uniformity_score)

    features = {
        "peel_green_pct": 100.0 * float(np.sum(green)) / pixel_count,
        "peel_yellow_pct": 100.0 * float(np.sum(yellow)) / pixel_count,
        "peel_brown_spot_pct": 100.0 * float(np.sum(brown)) / pixel_count,
        "peel_other_pct": 100.0 * float(np.sum(other)) / pixel_count,
        "peel_hue_mean_deg": float(np.mean(hue_deg[banana_pixels])),
        "peel_hue_std_deg": float(np.std(hue_deg[banana_pixels])),
        "peel_saturation_mean": float(np.mean(saturation[banana_pixels])),
        "peel_brightness_mean": float(np.mean(brightness[banana_pixels])),
        "peel_texture_contrast": texture_contrast,
        "peel_texture_homogeneity": texture_homogeneity,
        **quality,
    }

    mask_valid = int(
        features["image_quality_score"] >= 0.50
        and features["background_uniformity_score"] >= 0.30
        and features["banana_area_fraction"] >= 0.03
        and features["banana_area_fraction"] <= 0.65
    )

    overlay = make_mask_overlay(rgb, banana_mask)
    binary_mask = (banana_mask.astype(np.uint8) * 255)
    binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    isolated = np.full_like(rgb, 255)
    isolated[banana_mask] = rgb[banana_mask]
    return features, overlay, binary_mask_rgb, isolated, mask_valid


def make_mask_overlay(rgb: np.ndarray, banana_mask: np.ndarray) -> np.ndarray:
    """Show the mask as a translucent green overlay on top of the uploaded image."""
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[:, :, 1] = 255
    blended = cv2.addWeighted(rgb, 0.65, color, 0.35, 0)
    overlay[banana_mask] = blended[banana_mask]
    return overlay


def build_feature_row(hardware_values: Dict[str, float], image_features: Dict[str, float]) -> pd.DataFrame:
    """Combine hardware and image values in the exact order stored by the model."""
    all_values = {**hardware_values, **image_features}
    missing = [col for col in FEATURE_COLS if col not in all_values]
    if missing:
        raise ValueError(f"The model expects missing feature(s): {', '.join(missing)}")

    row = {col: float(all_values[col]) for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_grade(input_vector: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Run the loaded Random Forest model and format class probabilities."""
    if MODEL_LOAD_ERROR:
        raise ValueError(MODEL_LOAD_ERROR)
    if MODEL is None:
        raise ValueError("Model is not available. Place the .pkl file in the models folder.")

    prediction = str(MODEL.predict(input_vector)[0])

    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(input_vector)[0]
        model_classes = [str(cls) for cls in getattr(MODEL, "classes_", CLASS_LABELS)]
        probability_lookup = {label: float(prob) for label, prob in zip(model_classes, probabilities)}
    else:
        probability_lookup = {label: 0.0 for label in CLASS_LABELS}
        probability_lookup[prediction] = 1.0

    probability_df = pd.DataFrame(
        {
            "Grade": CLASS_LABELS,
            "Probability": [probability_lookup.get(label, 0.0) for label in CLASS_LABELS],
            "Probability (%)": [round(probability_lookup.get(label, 0.0) * 100.0, 2) for label in CLASS_LABELS],
            "Meaning": [GRADE_MEANINGS[label] for label in CLASS_LABELS],
        }
    )
    return prediction, probability_df


def format_key_value_table(values: Dict[str, float]) -> pd.DataFrame:
    """Display feature dictionaries as a professor-friendly two-column table."""
    return pd.DataFrame(
        {
            "Feature": list(values.keys()),
            "Value": [round(float(value), 6) for value in values.values()],
        }
    )


def make_validation_summary_markdown() -> str:
    """Render the supplied model validation evidence in the app."""
    lines = ["### Model Validation Summary"]
    for key, value in VALIDATION_SUMMARY.items():
        lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


def make_probability_plot(probability_df: pd.DataFrame):
    """Create a compact bar plot for Grade A/B/C probabilities."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["#2f9e44", "#f08c00", "#c92a2a"]
    ax.bar(probability_df["Grade"], probability_df["Probability"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Predicted class")
    ax.set_title("Class probabilities")
    ax.grid(axis="y", alpha=0.25)
    for idx, probability in enumerate(probability_df["Probability"]):
        ax.text(idx, probability + 0.02, f"{probability * 100:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def make_empty_probability_plot(message: str):
    """Create a readable plot placeholder when prediction cannot run."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def make_confusion_matrix_plot():
    """Plot the fixed validation confusion matrix supplied for this model."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(CONFUSION_MATRIX, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Actual grade")
    ax.set_title("Validation confusion matrix")

    threshold = CONFUSION_MATRIX.max() / 2.0
    for row in range(CONFUSION_MATRIX.shape[0]):
        for col in range(CONFUSION_MATRIX.shape[1]):
            value = CONFUSION_MATRIX[row, col]
            text_color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_feature_importance_plot():
    """Plot all feature importance values from the training evidence."""
    importance = pd.Series(FEATURE_IMPORTANCE).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.barh(importance.index, importance.values, color="#3b82f6")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return fig


def build_quality_status(image_features: Dict[str, float], mask_valid: int) -> str:
    """Explain whether the segmentation and image are suitable for prediction."""
    status = "Valid for prediction" if mask_valid else "Needs review"
    return (
        f"### {status}\n"
        f"- `mask_valid`: {mask_valid}\n"
        f"- `image_quality_score`: {image_features['image_quality_score']:.3f}\n"
        f"- `background_uniformity_score`: {image_features['background_uniformity_score']:.3f}\n"
        f"- `banana_area_fraction`: {image_features['banana_area_fraction']:.3f}\n"
        "\n"
        "Mask validity requires image quality >= 0.50, background uniformity >= 0.30, "
        "and banana area fraction between 0.03 and 0.65."
    )


def build_prediction_markdown(prediction: str) -> str:
    """Format the final grade prediction."""
    meaning = GRADE_MEANINGS.get(prediction, "Unknown class")
    return f"## Predicted Grade: {prediction}\n**Interpretation:** {meaning}"


def build_decision_explanation(
    prediction: str,
    probability_df: pd.DataFrame,
    feature_values: Dict[str, float],
    mask_valid: int,
) -> str:
    """Generate a plain-language explanation using important model features."""
    top_features = sorted(FEATURE_IMPORTANCE.items(), key=lambda item: item[1], reverse=True)[:7]
    driver_lines = []
    for feature, importance in top_features:
        if feature in feature_values:
            driver_lines.append(f"- `{feature}` = {feature_values[feature]:.3f} (importance {importance:.3f})")

    max_probability = float(probability_df["Probability"].max()) if not probability_df.empty else 0.0
    if prediction == "A":
        grade_reason = (
            "Grade A usually appears when yellow peel coverage is strong, green coverage is lower, "
            "and brown spotting or rough texture is limited."
        )
    elif prediction == "B":
        grade_reason = (
            "Grade B is usually a borderline pattern with mixed yellow, green, brown, or texture signals."
        )
    else:
        grade_reason = (
            "Grade C is usually driven by strong green, brown-spot, texture, or poor-quality patterns."
        )

    quality_note = (
        "The image mask passed the app quality checks."
        if mask_valid
        else "The image mask did not pass every quality check, so review the segmentation outputs before trusting the grade."
    )

    return (
        f"### Decision Explanation\n"
        f"{grade_reason}\n\n"
        f"The model predicted **Grade {prediction}** with top probability **{max_probability * 100:.2f}%**. "
        f"{quality_note}\n\n"
        f"Important current feature values:\n" + "\n".join(driver_lines)
    )


def empty_outputs(message: str):
    """Return safe placeholder outputs when prediction cannot be completed."""
    empty_df = pd.DataFrame()
    return (
        f"## Prediction unavailable\n{message}",
        f"### Image quality unavailable\n{message}",
        pd.DataFrame(columns=["Grade", "Probability", "Probability (%)", "Meaning"]),
        make_empty_probability_plot(message),
        None,
        None,
        None,
        None,
        empty_df,
        empty_df,
        pd.DataFrame(columns=FEATURE_COLS),
        f"### Decision Explanation\n{message}",
        make_validation_summary_markdown(),
        make_confusion_matrix_plot(),
        make_feature_importance_plot(),
    )

def get_live_sensor_text():

    if ser is None:
        return "STM32 not connected"

    try:
        line = ser.readline().decode().strip()

        print(line)

        return line

    except Exception as e:
        return f"Error: {e}"
    


def update_sensor_values():
    return read_stm32_measurement()


def read_stm32_measurement():
    """Read buffered STM32 output and extract the final weight and temperature."""
    global serial_line_buffer
    if not connect_stm32():
        return gr.update(), gr.update(), f"STM32 disconnected on {SERIAL_PORT}: {serial_error}"

    latest = None
    try:
        with serial_lock:
            waiting = ser.in_waiting
            if waiting:
                serial_line_buffer += ser.read(waiting).decode(errors="ignore")
            lines = serial_line_buffer.splitlines(keepends=True)
            serial_line_buffer = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                serial_line_buffer = lines.pop()
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                print("STM32:", line)
                match = re.search(
                    r"(?:FINAL\s*->\s*)?Weight:\s*(-?[\d.]+)\s*g\s*\|\s*Temp:\s*(-?[\d.]+)",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    latest = (float(match.group(1)), float(match.group(2)))
    except (OSError, serial.SerialException) as exc:
        return gr.update(), gr.update(), f"STM32 read failed on {SERIAL_PORT}: {exc}"

    if latest is None:
        return gr.update(), gr.update(), f"Connected to {SERIAL_PORT}. Waiting for `FINAL -> Weight: … g | Temp: … C`."
    weight, temperature = latest
    return gr.update(value=weight), gr.update(value=temperature), (
        f"Read final STM32 measurement: **{weight:.2f} g**, **{temperature:.2f} °C**."
    )


def capture_from_esp32(camera_url):
    """Request a still image from the ESP32 and return it to the image input."""
    base_url = (camera_url or "").strip().rstrip("/")
    if not base_url:
        return None, "Enter the ESP32 camera address (for example http://192.168.1.50)."
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url

    flash_enabled = False
    try:
        flash_request = Request(
            f"{base_url}/control?var=led_intensity&val=180",
            headers={"Cache-Control": "no-cache"},
        )
        with urlopen(flash_request, timeout=5) as response:
            flash_enabled = response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        # Older/custom camera servers may not expose the LED intensity control.
        pass

    capture_url = f"{base_url}/capture?flash=1"
    try:
        request = Request(capture_url, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            image_bytes = response.read()
        if content_type != "image/jpeg" or not image_bytes:
            return None, "ESP32 did not return a JPEG. Check its firmware capture endpoint."
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None, "The ESP32 response could not be decoded as an image."
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        status = (
            "Photo captured with flash enabled. Review it below, then select Predict Grade."
            if flash_enabled
            else "Photo captured, but this camera server did not accept the flash control. Check LED_GPIO_NUM and flash support in its firmware."
        )
        return image, status
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, f"Could not reach the ESP32 at {base_url}. Confirm both devices are on the same Wi-Fi and the address is correct. ({exc})"
    finally:
        if flash_enabled:
            try:
                off_request = Request(
                    f"{base_url}/control?var=led_intensity&val=0",
                    headers={"Cache-Control": "no-cache"},
                )
                with urlopen(off_request, timeout=5):
                    pass
            except (HTTPError, URLError, TimeoutError, OSError):
                pass

def run_prediction(
    image,
    weight_g,
    surface_temp_c,
    ambient_temp_c,
    relative_humidity_pct,
    pressure_hpa,
    gas_resistance_kohm,
):
    

    """Main Gradio callback: image features + manual hardware values + model prediction."""
    try:
        hardware_values = {
    "weight_g": float(weight_g),
    "surface_temp_C": float(surface_temp_c),
    "ambient_temp_C": float(ambient_temp_c),
    "relative_humidity_pct": float(relative_humidity_pct),
    "pressure_hPa": float(pressure_hpa),
    "gas_resistance_kohm": float(gas_resistance_kohm),
}
    except (TypeError, ValueError):
        return empty_outputs("Please enter valid numeric values for all hardware/sensor inputs.")

    try:
        image_features, overlay, binary_mask, isolated, mask_valid = extract_image_features(image)
        input_vector = build_feature_row(hardware_values, image_features)
        prediction, probability_df = predict_grade(input_vector)

        all_feature_values = {**hardware_values, **image_features}
        prediction_md = build_prediction_markdown(prediction)
        quality_md = build_quality_status(image_features, mask_valid)
        explanation_md = build_decision_explanation(prediction, probability_df, all_feature_values, mask_valid)

        return (
            prediction_md,
            quality_md,
            probability_df,
            make_probability_plot(probability_df),
            resize_for_processing(ensure_rgb_uint8(image)),
            overlay,
            binary_mask,
            isolated,
            format_key_value_table(hardware_values),
            format_key_value_table(image_features),
            input_vector,
            explanation_md,
            make_validation_summary_markdown(),
            make_confusion_matrix_plot(),
            make_feature_importance_plot(),
        )
    except Exception as exc:
        return empty_outputs(f"Prediction failed: {exc}")


CUSTOM_CSS = """
.gradio-container {
    max-width: 1320px !important;
}
#result-grade {
    border-left: 5px solid #2f9e44;
    padding-left: 12px;
}
"""


with gr.Blocks(title="Banana Grade Prediction", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Banana Grade Prediction App\n"
        "Local Random Forest inference from one banana image plus manual non-destructive sensor values."
    )

    with gr.Row():
        model_state = (
            f"Model loaded from `{MODEL_PATH}` with `{len(FEATURE_COLS)}` feature columns."
            if not MODEL_LOAD_ERROR
            else f"Model status: {MODEL_LOAD_ERROR}"
        )
        gr.Markdown(model_state)

    gr.Markdown("## 1. User Inputs")
    
    with gr.Row():
        with gr.Column(scale=1):
            camera_url = gr.Textbox(
                label="ESP32-CAM address",
                value=os.getenv("ESP32_CAMERA_URL", "http://192.168.1.50"),
                info="Use the IP printed in the Arduino Serial Monitor.",
            )
            capture_status = gr.Markdown("Connect the ESP32-CAM to the same Wi-Fi as this app.")
            capture_button = gr.Button("Capture photo with flash", variant="secondary")
            image_input = gr.Image(
                label="Banana photo (camera capture or upload)",
                image_mode="RGB",
                sources=["upload"],
                type="numpy",
            )
            
            predict_button = gr.Button("Predict Grade", variant="primary")
        with gr.Column(scale=1):
            weight_g = gr.Number(value=120.0, label="weight_g", precision=2)
            surface_temp_c = gr.Number(value=27.0, label="surface_temp_C", precision=2)
            ambient_temp_c = gr.Number(value=26.0, label="ambient_temp_C", precision=2)
            relative_humidity_pct = gr.Number(value=55.0, label="relative_humidity_pct", precision=2)
            pressure_hpa = gr.Number(value=1013.25, label="pressure_hPa", precision=2)
            gas_resistance_kohm = gr.Number(value=80.0, label="gas_resistance_kohm", precision=2)
            stm32_status = gr.Markdown(
                f"STM32 status: {'connected on ' + SERIAL_PORT if ser is not None else 'not connected on ' + SERIAL_PORT + ' — ' + serial_error}"
            )
            read_stm32_button = gr.Button("Read latest STM32 measurement")
            timer = gr.Timer(1)

            timer.tick(
                fn=update_sensor_values,
                outputs=[weight_g, surface_temp_c, stm32_status]
            ) 

            read_stm32_button.click(
                fn=read_stm32_measurement,
                outputs=[weight_g, surface_temp_c, stm32_status],
            )

    gr.Markdown("## 2. Final Result")
    with gr.Row():
        with gr.Column(scale=1):
            predicted_grade = gr.Markdown("## Predicted Grade: waiting for input", elem_id="result-grade")
            quality_status = gr.Markdown("### Image quality status will appear after prediction.")
            probability_table = gr.Dataframe(
                headers=["Grade", "Probability", "Probability (%)", "Meaning"],
                label="Class probability table",
                interactive=False,
            )
            hardware_table = gr.Dataframe(label="Hardware input table", interactive=False)
        with gr.Column(scale=1):
            uploaded_image_output = gr.Image(label="Uploaded image", type="numpy")
            probability_plot = gr.Plot(label="Probability bar plot")

    gr.Markdown("## 3. Image Processing Backend")
    with gr.Row():
        mask_overlay_output = gr.Image(label="Banana mask overlay", type="numpy")
        binary_mask_output = gr.Image(label="Binary banana mask", type="numpy")
        isolated_banana_output = gr.Image(label="Isolated banana region", type="numpy")

    gr.Markdown("## 4. Extracted Image Features")
    image_feature_table = gr.Dataframe(label="Extracted image feature table", interactive=False)

    gr.Markdown("## 5. Final Model Input Vector")
    final_input_vector = gr.Dataframe(
        value=pd.DataFrame(columns=FEATURE_COLS),
        label="Final model input vector in model feature_cols order",
        interactive=False,
        wrap=True,
    )

    gr.Markdown("## 6. Model Explanation")
    decision_explanation = gr.Markdown("### Decision Explanation\nRun prediction to generate the explanation.")

    gr.Markdown("## 7. Training and Evaluation Evidence")
    with gr.Row():
        with gr.Column(scale=1):
            validation_summary = gr.Markdown(make_validation_summary_markdown())
            confusion_matrix_plot = gr.Plot(value=make_confusion_matrix_plot(), label="Confusion matrix plot")
        with gr.Column(scale=1):
            feature_importance_plot = gr.Plot(value=make_feature_importance_plot(), label="Feature importance plot")

    

    predict_button.click(
        fn=run_prediction,
        inputs=[
            image_input,
            weight_g,
            surface_temp_c,
            ambient_temp_c,
            relative_humidity_pct,
            pressure_hpa,
            gas_resistance_kohm,
        ],
        outputs=[
            predicted_grade,
            quality_status,
            probability_table,
            probability_plot,
            uploaded_image_output,
            mask_overlay_output,
            binary_mask_output,
            isolated_banana_output,
            hardware_table,
            image_feature_table,
            final_input_vector,
            decision_explanation,
            validation_summary,
            confusion_matrix_plot,
            feature_importance_plot,
        ],
    )

    capture_button.click(
        fn=capture_from_esp32,
        inputs=[camera_url],
        outputs=[image_input, capture_status],
    )

    

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
        quiet=True,
    )
    print("App started. Open http://127.0.0.1:7860 on this computer.", flush=True)
    demo.block_thread()
