"""
Data Preprocessing & Input Validation Module for MNIST Digit Recognition.

Student:  Nafize Ali
Branch:   B.Tech Artificial Intelligence & Data Science
Model:    Convolutional Neural Network (TensorFlow / Keras)
Dataset:  MNIST

Features:
- Loading and normalizing the official MNIST dataset.
- Input validation:
  - Blank image detection (insufficient ink)
  - Connected component analysis (scipy.ndimage) for multi-digit detection (e.g. '1234', '25', '89')
  - Stroke clustering to safely preserve multi-stroke single digits (e.g. '4', '5', '7')
  - Bounding box aspect ratio validation
- Real-world image preprocessing:
  - Grayscale conversion
  - Smart polarity/inversion detection (dark-on-light vs light-on-dark)
  - Aspect-ratio preserving crop & scale to 20x20
  - Centering inside 28x28 canvas (matching MNIST training distribution)
  - Normalization to [0.0, 1.0]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label, binary_closing, center_of_mass
import tensorflow as tf


@dataclass
class PreprocessingResult:
    """Structured result from digit validation and preprocessing."""
    is_valid: bool
    status: str  # 'VALID', 'BLANK', 'MULTI_DIGIT', 'NOISE', 'INVALID_FORMAT'
    message: str
    model_tensor: Optional[np.ndarray] = None    # Shape: (1, 28, 28, 1), float32 in [0.0, 1.0]
    preview_28x28: Optional[np.ndarray] = None   # Shape: (28, 28), float32 in [0.0, 1.0]
    debug_info: Dict[str, Any] = field(default_factory=dict)


def load_and_preprocess_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads the official MNIST dataset from Keras, normalizes pixel values to [0.0, 1.0],
    and reshapes images to (N, 28, 28, 1) for CNN input.

    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def validate_and_preprocess_digit(
    image_input: Union[Image.Image, np.ndarray, str],
    invert_mode: str = "auto",
    apply_centering: bool = True,
    min_ink_pixels: int = 20,
    aspect_ratio_limit: float = 1.35,
) -> PreprocessingResult:
    """
    Validates that the input image contains exactly ONE single handwritten digit,
    rejects blank or multi-digit inputs (e.g. '1234', '25', '89'), and preprocesses
    the single digit to the standard MNIST format (28x28, centered, normalized).

    Pipeline:
    1. Convert input to grayscale ('L').
    2. Polarity detection (inverts light background so ink is bright on dark background).
    3. Foreground thresholding to extract handwritten ink mask.
    4. Blank check: if foreground ink < min_ink_pixels -> BLANK.
    5. Connected-Component Analysis (CCA) using scipy.ndimage.
    6. Multi-digit check:
       - Detect horizontally separated character clusters.
       - Check bounding box aspect ratio (union width / union height >= 1.35 indicates multiple digits).
       - If multiple distinct digits detected -> MULTI_DIGIT.
    7. Single digit cropping, aspect-ratio scaling to 20x20, and centering in 28x28 canvas.
    8. Normalization to [0.0, 1.0] and reshaping to (1, 28, 28, 1).

    Args:
        image_input: PIL Image, numpy array (uint8 or float), or file path.
        invert_mode: 'auto' (detect background), 'invert' (force invert), or 'none' (leave as is).
        apply_centering: whether to apply MNIST-style aspect-ratio crop and centering.
        min_ink_pixels: minimum foreground pixels to reject blank/accidental taps.
        aspect_ratio_limit: width/height threshold beyond which input is classified as multi-digit.

    Returns:
        PreprocessingResult dataclass with validation status, messages, and model tensor.
    """
    # ── 1. Convert to PIL Grayscale ──────────────────────────────────────────
    try:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                # RGBA - handle alpha channel (e.g. from canvas)
                # If transparent background with white/black ink, composite onto white background
                alpha = image_input[:, :, 3]
                rgb = image_input[:, :, :3]
                if np.max(alpha) > 0:
                    bg = np.ones_like(rgb) * 255
                    alpha_factor = (alpha / 255.0)[:, :, np.newaxis]
                    composited = (rgb * alpha_factor + bg * (1 - alpha_factor)).astype(np.uint8)
                    pil_img = Image.fromarray(composited)
                else:
                    pil_img = Image.fromarray(rgb.astype(np.uint8))
            elif image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
                pil_img = Image.fromarray(image_input)
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return PreprocessingResult(
                is_valid=False,
                status="INVALID_FORMAT",
                message=f"Unsupported image type: {type(image_input)}",
            )
    except Exception as e:
        return PreprocessingResult(
            is_valid=False,
            status="INVALID_FORMAT",
            message=f"Could not load image: {str(e)}",
        )

    pil_gray = pil_img.convert("L")
    gray_arr = np.array(pil_gray, dtype=np.float32)
    h, w = gray_arr.shape

    if h < 5 or w < 5:
        return PreprocessingResult(
            is_valid=False,
            status="BLANK",
            message="Image dimensions are too small to contain a digit.",
        )

    # ── 2. Polarity / Inversion Detection ───────────────────────────────────
    # In MNIST: background is ~0 (black), foreground ink is ~255 (white).
    # Check borders and corners to evaluate background intensity.
    corner_size = max(2, min(h, w) // 10)
    corners = [
        gray_arr[0:corner_size, 0:corner_size],
        gray_arr[0:corner_size, w - corner_size : w],
        gray_arr[h - corner_size : h, 0:corner_size],
        gray_arr[h - corner_size : h, w - corner_size : w],
    ]
    corner_mean = float(np.mean([np.mean(c) for c in corners]))
    overall_mean = float(np.mean(gray_arr))

    if invert_mode == "auto":
        should_invert = (corner_mean > 127.0) or (overall_mean > 127.0)
    elif invert_mode == "invert":
        should_invert = True
    else:
        should_invert = False

    if should_invert:
        gray_arr = 255.0 - gray_arr

    # ── 3. Foreground Extraction & Blank Detection ──────────────────────────
    # Clean background floor noise
    max_val = float(np.max(gray_arr))
    if max_val < 30.0:
        return PreprocessingResult(
            is_valid=False,
            status="BLANK",
            message="No handwritten digit detected. Please draw or upload a clear digit (0–9).",
            debug_info={"max_pixel": max_val, "ink_count": 0},
        )

    # Threshold for foreground ink
    thresh = max(25.0, 0.22 * max_val)
    binary_mask = (gray_arr > thresh).astype(np.uint8)
    ink_count = int(np.sum(binary_mask))

    if ink_count < min_ink_pixels:
        return PreprocessingResult(
            is_valid=False,
            status="BLANK",
            message="No handwritten digit detected. Please draw or upload a clear digit (0–9).",
            debug_info={"max_pixel": max_val, "ink_count": ink_count},
        )

    # ── 4. Connected-Component Analysis (CCA) ───────────────────────────────
    # Perform mild binary closing to bridge tiny 1-2 pixel pen gaps within a single stroke
    structure_8 = np.ones((3, 3), dtype=int)
    closed_mask = binary_closing(binary_mask, structure=structure_8)

    labeled_arr, num_features = label(closed_mask, structure=structure_8)

    # Collect significant components
    components: List[Dict[str, Any]] = []
    min_comp_area = max(15, int(0.015 * ink_count))

    for comp_idx in range(1, num_features + 1):
        comp_mask = (labeled_arr == comp_idx)
        area = int(np.sum(comp_mask))
        if area < min_comp_area:
            continue

        rows = np.where(np.any(comp_mask, axis=1))[0]
        cols = np.where(np.any(comp_mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue

        ymin, ymax = int(rows[0]), int(rows[-1])
        xmin, xmax = int(cols[0]), int(cols[-1])
        comp_h = ymax - ymin + 1
        comp_w = xmax - xmin + 1

        # Ignore tiny dot specks
        if comp_h < 4 and comp_w < 4:
            continue

        components.append({
            "idx": comp_idx,
            "area": area,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "height": comp_h,
            "width": comp_w,
            "cx": (xmin + xmax) / 2.0,
            "cy": (ymin + ymax) / 2.0,
        })

    if len(components) == 0:
        return PreprocessingResult(
            is_valid=False,
            status="BLANK",
            message="No handwritten digit detected. Please draw or upload a clear digit (0–9).",
            debug_info={"ink_count": ink_count, "components": 0},
        )

    # ── 5. Multi-Digit Detection Heuristics ──────────────────────────────────
    # Sort components horizontally from left to right
    components.sort(key=lambda c: c["xmin"])

    # Overall bounding box across all significant components
    all_ymin = min(c["ymin"] for c in components)
    all_ymax = max(c["ymax"] for c in components)
    all_xmin = min(c["xmin"] for c in components)
    all_xmax = max(c["xmax"] for c in components)

    union_h = all_ymax - all_ymin + 1
    union_w = all_xmax - all_xmin + 1
    aspect_ratio = float(union_w) / float(union_h)

    # Group components into horizontal digit clusters:
    # Components belonging to the SAME digit (e.g. crossbar of 4, top bar of 5, crossbar of 7)
    # have overlapping or adjacent horizontal intervals and are vertically integrated.
    digit_clusters: List[List[Dict[str, Any]]] = []

    for comp in components:
        if not digit_clusters:
            digit_clusters.append([comp])
        else:
            last_cluster = digit_clusters[-1]
            last_xmin = min(c["xmin"] for c in last_cluster)
            last_xmax = max(c["xmax"] for c in last_cluster)
            last_w = last_xmax - last_xmin + 1

            # Check if this component horizontally overlaps or is tightly adjacent to the cluster
            # If gap between components is significant (> 18% of cluster width or absolute > 10px),
            # it represents a separate character.
            overlap_or_touch = (comp["xmin"] <= last_xmax + max(4, int(0.18 * last_w)))

            # If it overlaps horizontally, check if it's vertically stacked / integrated
            if overlap_or_touch:
                last_cluster.append(comp)
            else:
                digit_clusters.append([comp])

    num_clusters = len(digit_clusters)
    is_multi_digit = False
    multi_reason = ""

    # Rule A: Multiple horizontally separated digit clusters (e.g. "1 2 3 4", "2 5", "8 9")
    if num_clusters >= 2:
        # Check if secondary clusters have significant weight
        total_cluster_areas = [sum(c["area"] for c in cl) for cl in digit_clusters]
        main_cluster_area = max(total_cluster_areas)
        significant_clusters = [a for a in total_cluster_areas if a >= 0.15 * main_cluster_area or a >= 40]
        if len(significant_clusters) >= 2:
            is_multi_digit = True
            multi_reason = f"Detected {len(significant_clusters)} distinct character regions."

    # Rule B: High aspect ratio (width significantly exceeds height, e.g. "1234", "25", "89")
    # Single digits in MNIST almost never have width/height > 1.25.
    # Touching/connected multi-digits have aspect ratios typically > 1.35.
    if not is_multi_digit and aspect_ratio >= aspect_ratio_limit and union_w > 20:
        is_multi_digit = True
        multi_reason = f"Bounding box aspect ratio ({aspect_ratio:.2f}) indicates multiple horizontal digits."

    if is_multi_digit:
        return PreprocessingResult(
            is_valid=False,
            status="MULTI_DIGIT",
            message="Multiple digits detected. Please provide an image containing only ONE handwritten digit (0–9).",
            debug_info={
                "aspect_ratio": round(aspect_ratio, 2),
                "num_clusters": num_clusters,
                "union_w": union_w,
                "union_h": union_h,
                "reason": multi_reason,
            },
        )

    # ── 6. Single Digit Cropping & MNIST Normalization ───────────────────────
    # Extract tight bounding box around the single detected digit from high-res image
    digit_crop = gray_arr[all_ymin : all_ymax + 1, all_xmin : all_xmax + 1]

    # Add a small padding margin around the crop (5% on each side)
    pad_y = max(1, int(0.05 * union_h))
    pad_x = max(1, int(0.05 * union_w))
    padded_ymin = max(0, all_ymin - pad_y)
    padded_ymax = min(h - 1, all_ymax + pad_y)
    padded_xmin = max(0, all_xmin - pad_x)
    padded_xmax = min(w - 1, all_xmax + pad_x)

    digit_crop = gray_arr[padded_ymin : padded_ymax + 1, padded_xmin : padded_xmax + 1]
    dh, dw = digit_crop.shape

    if h == 28 and w == 28 and not apply_centering:
        processed_arr = gray_arr
    elif apply_centering:
        # Resize preserving aspect ratio so the longest dimension is 20 pixels
        if dh >= dw:
            factor = 20.0 / float(dh)
            new_h = 20
            new_w = max(1, int(round(dw * factor)))
        else:
            factor = 20.0 / float(dw)
            new_w = 20
            new_h = max(1, int(round(dh * factor)))

        digit_pil = Image.fromarray(digit_crop.astype(np.uint8)).resize(
            (new_w, new_h), resample=Image.Resampling.BILINEAR
        )
        digit_resized = np.array(digit_pil, dtype=np.float32)

        # Place inside 28x28 canvas centered
        canvas = np.zeros((28, 28), dtype=np.float32)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = digit_resized

        # Center of mass fine-adjustment (MNIST standard)
        cy, cx = center_of_mass(canvas)
        if not np.isnan(cy) and not np.isnan(cx):
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))
            # Limit shift to avoid clipping
            shift_y = max(-3, min(3, shift_y))
            shift_x = max(-3, min(3, shift_x))
            if shift_y != 0 or shift_x != 0:
                shifted_canvas = np.zeros((28, 28), dtype=np.float32)
                src_y1 = max(0, -shift_y)
                src_y2 = min(28, 28 - shift_y)
                src_x1 = max(0, -shift_x)
                src_x2 = min(28, 28 - shift_x)
                dst_y1 = max(0, shift_y)
                dst_y2 = min(28, 28 + shift_y)
                dst_x1 = max(0, shift_x)
                dst_x2 = min(28, 28 + shift_x)
                shifted_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = canvas[src_y1:src_y2, src_x1:src_x2]
                canvas = shifted_canvas
        processed_arr = canvas
    else:
        raw_pil = Image.fromarray(digit_crop.astype(np.uint8)).resize(
            (28, 28), resample=Image.Resampling.BILINEAR
        )
        processed_arr = np.array(raw_pil, dtype=np.float32)

    # ── 7. Normalization to [0.0, 1.0] ───────────────────────────────────────
    normalized_28 = np.clip(processed_arr / 255.0, 0.0, 1.0).astype(np.float32)
    model_tensor = np.expand_dims(normalized_28, axis=(0, -1))

    return PreprocessingResult(
        is_valid=True,
        status="VALID",
        message="Valid single handwritten digit.",
        model_tensor=model_tensor,
        preview_28x28=normalized_28,
        debug_info={
            "aspect_ratio": round(aspect_ratio, 2),
            "num_clusters": num_clusters,
            "components": len(components),
            "union_w": union_w,
            "union_h": union_h,
            "ink_pixels": ink_count,
        },
    )


def preprocess_user_image(
    image_input: Union[Image.Image, np.ndarray, str],
    invert_mode: str = "auto",
    apply_centering: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible preprocessing function.
    Validates and returns (model_tensor, preview_28x28).
    Raises ValueError if input does not contain a valid single digit.
    """
    result = validate_and_preprocess_digit(
        image_input=image_input,
        invert_mode=invert_mode,
        apply_centering=apply_centering,
    )
    if not result.is_valid:
        raise ValueError(result.message)
    return result.model_tensor, result.preview_28x28
