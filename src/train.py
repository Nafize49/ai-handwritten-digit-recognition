"""
Model Training Script for MNIST Handwritten Digit Recognition System.

Executes the end-to-end training pipeline:
1. Loads official MNIST dataset (60,000 train, 10,000 test).
2. Normalizes pixel values [0, 1] and reshapes for CNN input (N, 28, 28, 1).
3. Trains a baseline Multi-Layer Perceptron (MLP) for empirical comparison.
4. Builds and compiles the deep Convolutional Neural Network (CNN).
5. Fits the CNN with EarlyStopping and ModelCheckpoint.
6. Evaluates both models on the unseen 10,000 test set.
7. Saves the best model in standard Keras format (model/digit_cnn.keras).
8. Exports visualization charts and evaluation metrics JSON.
"""

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_preprocessing import load_and_preprocess_mnist
from src.evaluate import plot_training_history, run_full_evaluation
from src.model import build_cnn_model, build_mlp_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
VIS_DIR = PROJECT_ROOT / "visualizations"
MODEL_PATH = MODEL_DIR / "digit_cnn.keras"
METRICS_PATH = MODEL_DIR / "training_metrics.json"


def train_pipeline(
    epochs: int = 10,
    batch_size: int = 64,
    train_mlp_baseline: bool = True,
) -> None:
    """
    Executes the complete training, evaluation, and artifact generation pipeline.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AI-BASED HANDWRITTEN DIGIT RECOGNITION SYSTEM - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    print(f"  Training samples:   {x_train.shape[0]:,} images, shape: {x_train.shape[1:]}")
    print(f"  Testing samples:    {x_test.shape[0]:,} images, shape: {x_test.shape[1:]}")
    print(f"  Pixel range:        [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  Class labels:       {sorted(list(set(y_train)))}")

    # 2. Baseline MLP Comparison (Optional / Academic requirement)
    mlp_test_acc = None
    mlp_test_loss = None
    if train_mlp_baseline:
        print("\n[Step 2/6] Training baseline Multi-Layer Perceptron (MLP) for comparison...")
        mlp_model = build_mlp_model()
        mlp_start = time.time()
        mlp_model.fit(
            x_train,
            y_train,
            epochs=5,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )
        mlp_time = time.time() - mlp_start
        mlp_loss, mlp_acc = mlp_model.evaluate(x_test, y_test, verbose=0)
        mlp_test_acc = float(mlp_acc)
        mlp_test_loss = float(mlp_loss)
        print(f"  MLP Baseline Test Accuracy: {mlp_test_acc * 100:.2f}% (Trained in {mlp_time:.1f}s)")

    # 3. Build CNN Model
    print("\n[Step 3/6] Building Convolutional Neural Network (CNN)...")
    cnn_model = build_cnn_model()
    cnn_model.summary()

    # 4. Train CNN
    print(f"\n[Step 4/6] Training CNN ({epochs} epochs, batch size {batch_size})...")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    cnn_start = time.time()
    history = cnn_model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    cnn_train_time = time.time() - cnn_start
    print(f"  CNN Training finished in {cnn_train_time:.1f} seconds.")

    # Reload best saved model
    best_model = tf.keras.models.load_model(str(MODEL_PATH))

    # 5. Visualizations & History
    print("\n[Step 5/6] Exporting training and validation curves...")
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    plot_training_history(history_dict, VIS_DIR)
    print(f"  Saved: {VIS_DIR / 'accuracy_curve.png'}")
    print(f"  Saved: {VIS_DIR / 'loss_curve.png'}")

    # 6. Full Evaluation
    print("\n[Step 6/6] Running comprehensive evaluation on test set (10,000 samples)...")
    eval_metrics = run_full_evaluation(best_model, x_test, y_test, VIS_DIR)
    print(f"  Saved: {VIS_DIR / 'confusion_matrix.png'}")
    print(f"  Saved: {VIS_DIR / 'sample_misclassifications.png'}")

    # Assemble complete metrics dictionary
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    full_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(cnn_train_time, 2),
        "total_epochs_trained": len(history.history["loss"]),
        "final_train_accuracy": round(final_train_acc, 4),
        "final_val_accuracy": round(final_val_acc, 4),
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "test_loss": round(eval_metrics["test_loss"], 4),
        "test_accuracy": round(eval_metrics["test_accuracy"], 4),
        "test_error_rate": round(eval_metrics["test_error_rate"], 4),
        "total_test_samples": eval_metrics["total_test_samples"],
        "total_misclassified": eval_metrics["total_misclassified"],
        "history": history_dict,
        "classification_report": eval_metrics["classification_report"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "sample_misclassifications": eval_metrics["sample_misclassifications"],
        "baseline_comparison": {
            "mlp_test_accuracy": round(mlp_test_acc, 4) if mlp_test_acc else None,
            "mlp_test_loss": round(mlp_test_loss, 4) if mlp_test_loss else None,
            "cnn_test_accuracy": round(eval_metrics["test_accuracy"], 4),
            "cnn_test_loss": round(eval_metrics["test_loss"], 4),
            "accuracy_gain_pct": round((eval_metrics["test_accuracy"] - (mlp_test_acc or 0)) * 100, 2),
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_record, f, indent=2)
    print(f"  Saved full metrics: {METRICS_PATH}")

    # Summary Output
    print("\n" + "=" * 70)
    print("FINAL TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Training Accuracy:     {final_train_acc * 100:.2f}%")
    print(f"  Validation Accuracy:   {final_val_acc * 100:.2f}%")
    print(f"  CNN Test Accuracy:     {eval_metrics['test_accuracy'] * 100:.2f}%")
    if mlp_test_acc:
        print(f"  MLP Test Accuracy:     {mlp_test_acc * 100:.2f}%")
        print(f"  CNN Improvement:       +{(eval_metrics['test_accuracy'] - mlp_test_acc) * 100:.2f}%")
    print(f"  Total Misclassified:   {eval_metrics['total_misclassified']} / 10,000 samples")
    print(f"  Model File Saved At:   {MODEL_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_pipeline()
