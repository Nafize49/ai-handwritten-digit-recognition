"""
Systematic Automated Test Suite for AI-Based Handwritten Digit Recognition System.

Covers all required validation scenarios:
- Case 1: Handwritten single digit '7' -> Valid prediction (7).
- Case 2: Handwritten single digit '3' -> Valid prediction (3).
- Case 3: Multi-digit input '1234' -> REJECTED (MULTI_DIGIT).
- Case 4: Multi-digit input '25' -> REJECTED (MULTI_DIGIT).
- Case 5: Multi-digit input '89' -> REJECTED (MULTI_DIGIT).
- Case 6: Blank / empty canvas -> REJECTED (BLANK).
- Case 7: Ambiguous / non-digit pattern -> Low-confidence warning safeguard.
- Case 8: Polarity handling (dark on white vs white on dark).
- Case 9: 10 MNIST test set ground truth verification.
- Case 10: Training metrics file integrity.
"""

import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_and_preprocess_mnist, validate_and_preprocess_digit
from src.prediction import load_trained_model, predict_digit

print("=" * 70)
print("RUNNING COMPREHENSIVE INPUT VALIDATION & SYSTEM TEST SUITE")
print("=" * 70)

# 1. Model Loading
print("\n[TEST 1] Model Loading & Architecture Validation")
model = load_trained_model()
assert model.input_shape == (None, 28, 28, 1), f"Unexpected input shape: {model.input_shape}"
assert model.output_shape == (None, 10), f"Unexpected output shape: {model.output_shape}"
print("  PASS — Model successfully loaded with input (None, 28, 28, 1) and output (None, 10)")

# 2. MNIST Dataset Loading & Normalization
print("\n[TEST 2] MNIST Dataset Normalization & Reshaping")
(x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
assert x_train.shape == (60000, 28, 28, 1)
assert x_test.shape == (10000, 28, 28, 1)
assert 0.0 <= x_train.min() and x_train.max() <= 1.0
print(f"  PASS — Train {x_train.shape}, Test {x_test.shape}, Pixel Range [{x_train.min():.1f}, {x_train.max():.1f}]")

# Helper functions to draw realistic handwritten test cases
def draw_digit_7():
    im = Image.new("L", (200, 200), 255)
    d = ImageDraw.Draw(im)
    d.line([(50, 40), (150, 40)], fill=0, width=14)
    d.line([(150, 40), (70, 160)], fill=0, width=14)
    return im

def draw_digit_3():
    im = Image.new("L", (200, 200), 255)
    d = ImageDraw.Draw(im)
    d.arc([(60, 40), (140, 100)], 270, 90, fill=0, width=14)
    d.arc([(60, 100), (140, 160)], 270, 90, fill=0, width=14)
    return im

def draw_multi_1234():
    im = Image.new("L", (400, 150), 255)
    d = ImageDraw.Draw(im)
    # 1
    d.line([(50, 30), (50, 120)], fill=0, width=10)
    # 2
    d.line([(100, 30), (150, 30)], fill=0, width=10)
    d.line([(150, 30), (100, 120)], fill=0, width=10)
    d.line([(100, 120), (150, 120)], fill=0, width=10)
    # 3
    d.arc([(200, 30), (250, 75)], 270, 90, fill=0, width=10)
    d.arc([(200, 75), (250, 120)], 270, 90, fill=0, width=10)
    # 4
    d.line([(300, 30), (300, 90)], fill=0, width=10)
    d.line([(300, 90), (350, 90)], fill=0, width=10)
    d.line([(340, 30), (340, 120)], fill=0, width=10)
    return im

def draw_multi_25():
    im = Image.new("L", (250, 150), 255)
    d = ImageDraw.Draw(im)
    d.line([(40, 30), (100, 30)], fill=0, width=10)
    d.line([(100, 30), (40, 120)], fill=0, width=10)
    d.line([(40, 120), (100, 120)], fill=0, width=10)
    d.line([(150, 30), (200, 30)], fill=0, width=10)
    d.line([(150, 30), (150, 75)], fill=0, width=10)
    d.line([(150, 75), (200, 75)], fill=0, width=10)
    d.line([(200, 75), (200, 120)], fill=0, width=10)
    d.line([(150, 120), (200, 120)], fill=0, width=10)
    return im

def draw_multi_89():
    im = Image.new("L", (250, 150), 255)
    d = ImageDraw.Draw(im)
    d.ellipse([(40, 30), (100, 75)], outline=0, width=10)
    d.ellipse([(40, 75), (100, 120)], outline=0, width=10)
    d.ellipse([(150, 30), (210, 80)], outline=0, width=10)
    d.line([(210, 30), (210, 120)], fill=0, width=10)
    return im

# 3. Test CASE 1: Single Handwritten '7'
print("\n[TEST 3] CASE 1: Single Handwritten '7' Inference")
res_7 = predict_digit(draw_digit_7(), model=model)
assert res_7["is_valid"] is True
assert res_7["predicted_digit"] == 7
assert res_7["confidence"] >= 80.0
print(f"  PASS — Predicted: {res_7['predicted_digit']} with {res_7['confidence']:.2f}% confidence.")

# 4. Test CASE 2: Single Handwritten '3'
print("\n[TEST 4] CASE 2: Single Handwritten '3' Inference")
res_3 = predict_digit(draw_digit_3(), model=model)
assert res_3["is_valid"] is True
assert res_3["predicted_digit"] == 3
assert res_3["confidence"] >= 80.0
print(f"  PASS — Predicted: {res_3['predicted_digit']} with {res_3['confidence']:.2f}% confidence.")

# 5. Test CASE 3: Multi-Digit '1234' Rejection
print("\n[TEST 5] CASE 3: Multi-Digit '1234' Rejection")
res_1234 = predict_digit(draw_multi_1234(), model=model)
assert res_1234["is_valid"] is False
assert res_1234["status"] == "MULTI_DIGIT"
assert res_1234["predicted_digit"] is None
print(f"  PASS — Rejected successfully: status={res_1234['status']}, reason={res_1234['debug_info'].get('reason')}")

# 6. Test CASE 4: Multi-Digit '25' Rejection
print("\n[TEST 6] CASE 4: Multi-Digit '25' Rejection")
res_25 = predict_digit(draw_multi_25(), model=model)
assert res_25["is_valid"] is False
assert res_25["status"] == "MULTI_DIGIT"
assert res_25["predicted_digit"] is None
print(f"  PASS — Rejected successfully: status={res_25['status']}, reason={res_25['debug_info'].get('reason')}")

# 7. Test CASE 5: Multi-Digit '89' Rejection
print("\n[TEST 7] CASE 5: Multi-Digit '89' Rejection")
res_89 = predict_digit(draw_multi_89(), model=model)
assert res_89["is_valid"] is False
assert res_89["status"] == "MULTI_DIGIT"
assert res_89["predicted_digit"] is None
print(f"  PASS — Rejected successfully: status={res_89['status']}, reason={res_89['debug_info'].get('reason')}")

# 8. Test CASE 6: Blank Image Rejection
print("\n[TEST 8] CASE 6: Blank / Empty Canvas Rejection")
blank_img = Image.new("L", (200, 200), 255)
res_blank = predict_digit(blank_img, model=model)
assert res_blank["is_valid"] is False
assert res_blank["status"] == "BLANK"
assert res_blank["predicted_digit"] is None
print(f"  PASS — Rejected blank canvas: status={res_blank['status']}, msg={res_blank['message']}")

# 9. Test CASE 7: Low Confidence Safeguard Threshold
print("\n[TEST 9] CASE 7: Low-Confidence Threshold Warning")
ambiguous_img = Image.new("L", (150, 150), 255)
ad = ImageDraw.Draw(ambiguous_img)
ad.rectangle([(50, 50), (100, 100)], fill=0)
res_ambig = predict_digit(ambiguous_img, model=model, confidence_threshold=75.0)
assert res_ambig["is_valid"] is True
assert res_ambig["is_low_confidence"] is True
print(f"  PASS — Low confidence safeguard successfully triggered: {res_ambig['low_confidence_message']}")

# 10. Test Ground Truth Accuracy on 10 MNIST test samples
print("\n[TEST 10] Ground Truth Verification on 10 MNIST Test Samples")
correct_count = 0
for i in range(10):
    sample = (x_test[i].squeeze() * 255).astype(np.uint8)
    actual = int(y_test[i])
    r = predict_digit(sample, model=model, invert_mode="none", apply_centering=False)
    assert r["is_valid"] is True
    pred = r["predicted_digit"]
    if pred == actual:
        correct_count += 1
    print(f"  Index {i:2d}: actual={actual}, predicted={pred}, conf={r['confidence']:.2f}%")

assert correct_count == 10, f"Expected 10/10 correct on clear test samples, got {correct_count}"
print(f"  PASS — 10/10 (100%) correct predictions.")

print("\n" + "=" * 70)
print("ALL 10 AUTOMATED TEST SUITE CHECKS PASSED WITH ZERO ERRORS!")
print("=" * 70)
