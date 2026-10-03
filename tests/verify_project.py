import json

with open("model/training_metrics.json", "r") as f:
    m = json.load(f)

print("=" * 60)
print("VERIFIED TRAINING METRICS")
print("=" * 60)
print(f"  Timestamp:             {m['timestamp']}")
print(f"  Epochs Trained:        {m['total_epochs_trained']}")
print(f"  Training Accuracy:     {m['final_train_accuracy']*100:.2f}%")
print(f"  Validation Accuracy:   {m['final_val_accuracy']*100:.2f}%")
print(f"  CNN Test Accuracy:     {m['test_accuracy']*100:.2f}%")
print(f"  CNN Test Loss:         {m['test_loss']:.4f}")
print(f"  Test Error Rate:       {m['test_error_rate']*100:.2f}%")
print(f"  Total Test Samples:    {m['total_test_samples']:,}")
print(f"  Total Misclassified:   {m['total_misclassified']}")
baseline = m.get("baseline_comparison", {})
mlp_acc = baseline.get("mlp_test_accuracy") or 0
print(f"  MLP Baseline Accuracy: {mlp_acc*100:.2f}%")
print(f"  CNN vs MLP Gain:       +{baseline.get('accuracy_gain_pct', 0):.2f}%")
print("=" * 60)

import os
print("\nFILE VERIFICATION:")
files = [
    "model/digit_cnn.keras",
    "model/training_metrics.json",
    "visualizations/accuracy_curve.png",
    "visualizations/loss_curve.png",
    "visualizations/confusion_matrix.png",
    "visualizations/sample_misclassifications.png",
    "app.py",
    "requirements.txt",
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/prediction.py",
    "README.md",
    ".gitignore",
    "documentation/PROJECT_DOCUMENTATION.md",
    "documentation/VIVA_QUESTIONS.md",
    "documentation/PRESENTATION_CONTENT.md",
    "notebooks/digit_recognition.ipynb",
    "tests/test_prediction.py",
]
all_ok = True
for fpath in files:
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {fpath:<55} {size:>10} bytes")

print()
print("ALL FILES PRESENT" if all_ok else "SOME FILES MISSING - CHECK ABOVE")
