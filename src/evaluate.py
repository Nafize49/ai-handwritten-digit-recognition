"""
Evaluation Module for Handwritten Digit Recognition System.

Calculates:
- Test loss and test accuracy
- 10x10 Confusion Matrix
- Per-class Precision, Recall, and F1-Score (Classification Report)
- Misclassification analysis (true vs predicted, confidence, sample images)
- Generation of publication-quality visualizations
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    classes: List[str] = [str(i) for i in range(10)],
) -> np.ndarray:
    """
    Computes and saves an annotated 10x10 confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        linecolor="#e0e0e0",
        ax=ax,
    )
    ax.set_title("Confusion Matrix (10,000 MNIST Test Samples)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Digit", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Actual Digit (Ground Truth)", fontsize=12, fontweight="semibold", labelpad=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return cm


def plot_training_history(history_dict: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Plots and saves separate loss and accuracy progression curves across epochs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_dict["loss"]) + 1)

    # 1. Accuracy Curve
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(epochs, history_dict["accuracy"], "o-", color="#1f77b4", linewidth=2, label="Training Accuracy")
    if "val_accuracy" in history_dict:
        ax.plot(epochs, history_dict["val_accuracy"], "s--", color="#ff7f0e", linewidth=2, label="Validation Accuracy")
    ax.set_title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    # 2. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(epochs, history_dict["loss"], "o-", color="#d62728", linewidth=2, label="Training Loss")
    if "val_loss" in history_dict:
        ax.plot(epochs, history_dict["val_loss"], "s--", color="#9467bd", linewidth=2, label="Validation Loss")
    ax.set_title("Model Loss Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()


def analyze_misclassifications(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    num_samples: int = 15,
) -> List[Dict[str, Any]]:
    """
    Finds actual misclassified samples from the test set and saves a visual gallery.
    """
    misclassified_indices = np.where(y_true != y_pred)[0]
    total_misclassified = len(misclassified_indices)

    misclassified_data: List[Dict[str, Any]] = []

    # Sort or take representative samples
    selected_indices = misclassified_indices[:num_samples]

    for idx in selected_indices:
        actual = int(y_true[idx])
        pred = int(y_pred[idx])
        conf = float(y_probs[idx][pred]) * 100.0
        actual_conf = float(y_probs[idx][actual]) * 100.0
        # Flatten image to 28x28 list for JSON serializability
        img_matrix = x_test[idx].squeeze().tolist()
        misclassified_data.append(
            {
                "test_index": int(idx),
                "actual": actual,
                "predicted": pred,
                "predicted_confidence": round(conf, 2),
                "actual_confidence": round(actual_conf, 2),
                "image_data": img_matrix,
            }
        )

    # Plot sample misclassifications gallery
    if len(selected_indices) > 0:
        rows = 3
        cols = min(5, (len(selected_indices) + 2) // 3)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.2), dpi=300)
        axes = np.array(axes).reshape(-1)

        for i, idx in enumerate(selected_indices[: rows * cols]):
            ax = axes[i]
            img = x_test[idx].squeeze()
            ax.imshow(img, cmap="gray_r")
            ax.set_title(
                f"True: {y_true[idx]} | Pred: {y_pred[idx]}\nConf: {y_probs[idx][y_pred[idx]] * 100:.1f}%",
                fontsize=9,
                color="red",
                fontweight="bold",
            )
            ax.axis("off")

        # Turn off any remaining unused axes
        for j in range(len(selected_indices), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Sample Misclassified MNIST Test Digits (Total: {total_misclassified} / 10,000)",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "sample_misclassifications.png", dpi=300)
        plt.close()

    return misclassified_data


def run_full_evaluation(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    visualizations_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluates the model on the full test set, generates all required charts,
    and returns a clean structured summary dictionary.
    """
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test loss and accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # 2. Predictions and probabilities
    y_probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 3. Confusion Matrix
    cm = plot_confusion_matrix(y_test, y_pred, visualizations_dir / "confusion_matrix.png")

    # 4. Classification Report
    target_names = [f"Digit {i}" for i in range(10)]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # 5. Misclassification analysis
    misclass_samples = analyze_misclassifications(x_test, y_test, y_pred, y_probs, visualizations_dir)

    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_error_rate": float(1.0 - test_acc),
        "total_test_samples": int(len(y_test)),
        "total_misclassified": int(np.sum(y_test != y_pred)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "sample_misclassifications": misclass_samples,
    }

    return metrics_summary
