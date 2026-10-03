"""
Automated Prediction Test Suite for Digit Recognition.

Verifies:
- Model loading from disk.
- Preprocessing of raw test samples.
- Correct prediction against ground-truth MNIST test labels.
- Valid confidence percentage output [0, 100].
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_and_preprocess_mnist
from src.prediction import load_trained_model, predict_digit


def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED PREDICTION TEST SUITE")
    print("=" * 60)

    # 1. Load model
    print("[1] Loading saved model from model/digit_cnn.keras...")
    model = load_trained_model()
    print("    Model successfully loaded!")

    # 2. Load MNIST test images
    print("[2] Loading MNIST test dataset...")
    _, (x_test, y_test) = load_and_preprocess_mnist()

    # 3. Test on 10 deterministic test samples
    test_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    correct_count = 0

    print("\n[3] Running inference on 10 test samples:")
    print(f"{'Sample #':<10} | {'Actual':<8} | {'Predicted':<10} | {'Confidence':<12} | {'Status'}")
    print("-" * 55)

    for idx in test_indices:
        sample_img = x_test[idx].squeeze()  # shape (28, 28)
        actual_label = int(y_test[idx])

        result = predict_digit(
            image_input=sample_img,
            model=model,
            invert_mode="none",  # already in MNIST polarity
            apply_centering=False,
        )

        pred_label = result["predicted_digit"]
        confidence = result["confidence"]
        is_correct = pred_label == actual_label
        if is_correct:
            correct_count += 1

        status_str = "PASS" if is_correct else "FAIL"
        print(f"Index {idx:<4} | {actual_label:<8} | {pred_label:<10} | {confidence:>6.2f}%     | {status_str}")

        assert 0 <= pred_label <= 9, f"Invalid predicted digit: {pred_label}"
        assert 0.0 <= confidence <= 100.0, f"Invalid confidence: {confidence}"
        assert len(result["probabilities"]) == 10, "Probabilities list must have 10 values"
        assert result["processed_image"].shape == (28, 28), "Processed image must be 28x28"

    accuracy_sample = (correct_count / len(test_indices)) * 100
    print("-" * 55)
    print(f"Sample Accuracy: {correct_count}/{len(test_indices)} ({accuracy_sample:.1f}%)")

    assert correct_count >= 9, f"Expected at least 9/10 correct, got {correct_count}"
    print("\nALL PREDICTION TESTS PASSED SUCCESSFULLY! ZERO ERRORS.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
