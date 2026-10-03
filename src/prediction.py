"""
Prediction Module for Digit Recognition.

Student:  Nafize Ali
Branch:   B.Tech Artificial Intelligence & Data Science
Model:    Convolutional Neural Network (TensorFlow / Keras)
Dataset:  MNIST

Provides robust, validated inference:
- Validates input: rejects blank/empty and multi-digit inputs (e.g. '1234', '25', '89').
- Preprocesses single digit through the standard MNIST pipeline (crop, aspect-ratio scale, center).
- Predicts digit class (0-9) and confidence.
- Flags low-confidence predictions (< 60.0% configurable threshold).
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import tensorflow as tf

from src.data_preprocessing import validate_and_preprocess_digit, PreprocessingResult

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "digit_cnn.keras"
CONFIDENCE_THRESHOLD = 60.0  # Percentage threshold for low-confidence safeguard


def load_trained_model(model_path: Optional[Union[str, Path]] = None) -> tf.keras.Model:
    """
    Loads the trained Keras CNN model from disk.
    Uses pathlib for cross-platform compatibility (Linux / Windows).
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please run 'python src/train.py' first."
        )
    return tf.keras.models.load_model(str(path))


def predict_digit(
    image_input: Union[Image.Image, np.ndarray, str],
    model: tf.keras.Model,
    invert_mode: str = "auto",
    apply_centering: bool = True,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Performs validated inference on an image.

    Validates that the image contains exactly ONE clear handwritten digit (0-9).
    If multiple digits ('1234', '25', etc.) or blank canvas are provided, inference
    is halted and a descriptive validation error is returned instead of an arbitrary guess.

    Args:
        image_input: PIL Image, 2D/3D numpy array, or file path.
        model: Loaded tf.keras.Model instance.
        invert_mode: 'auto', 'invert', or 'none'.
        apply_centering: whether to apply MNIST-style aspect-ratio crop and centering.
        confidence_threshold: minimum confidence percentage for reliable prediction.

    Returns:
        Dictionary containing:
        - 'is_valid': bool (True if valid single digit, False otherwise)
        - 'status': 'VALID', 'BLANK', 'MULTI_DIGIT', etc.
        - 'message': explanation or warning
        - 'predicted_digit': int (0-9) or None
        - 'confidence': float (0-100%) or None
        - 'is_low_confidence': bool
        - 'low_confidence_message': str or None
        - 'probabilities': list of 10 floats (one per digit 0-9) or None
        - 'processed_image': 28x28 numpy array float32 in [0, 1] or None
        - 'debug_info': dict with validation metrics
    """
    # 1. Validate and preprocess input
    prep_result: PreprocessingResult = validate_and_preprocess_digit(
        image_input=image_input,
        invert_mode=invert_mode,
        apply_centering=apply_centering,
    )

    if not prep_result.is_valid:
        return {
            "is_valid": False,
            "status": prep_result.status,
            "message": prep_result.message,
            "predicted_digit": None,
            "confidence": None,
            "is_low_confidence": False,
            "low_confidence_message": None,
            "probabilities": None,
            "processed_image": None,
            "debug_info": prep_result.debug_info,
        }

    # 2. Run CNN Model Inference
    model_tensor = prep_result.model_tensor
    probabilities = model.predict(model_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100.0

    # 3. Low-confidence safeguard
    is_low_confidence = confidence < confidence_threshold
    low_conf_msg = None
    if is_low_confidence:
        low_conf_msg = (
            f"Low confidence prediction ({confidence:.2f}% < {confidence_threshold:.0f}% threshold). "
            "Please provide a clearer image of a single handwritten digit."
        )

    return {
        "is_valid": True,
        "status": "VALID",
        "message": "Valid single digit recognized successfully.",
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 2),
        "is_low_confidence": is_low_confidence,
        "low_confidence_message": low_conf_msg,
        "probabilities": [float(p) for p in probabilities],
        "processed_image": prep_result.preview_28x28,
        "debug_info": prep_result.debug_info,
    }
