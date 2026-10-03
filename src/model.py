"""
Model Architecture Definition Module.

Defines:
1. Deep Convolutional Neural Network (CNN) for handwritten digit recognition.
2. Baseline Multi-Layer Perceptron (MLP) for comparative empirical analysis.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs and compiles the primary Convolutional Neural Network (CNN).

    Architecture:
    - Input: 28x28x1 grayscale image
    - Conv Block 1:
        - Conv2D (32 filters, 3x3, ReLU, same padding)
        - BatchNormalization
        - Conv2D (32 filters, 3x3, ReLU)
        - MaxPooling2D (2x2)
        - Dropout (0.25)
    - Conv Block 2:
        - Conv2D (64 filters, 3x3, ReLU, same padding)
        - BatchNormalization
        - Conv2D (64 filters, 3x3, ReLU)
        - MaxPooling2D (2x2)
        - Dropout (0.25)
    - Classifier Head:
        - Flatten
        - Dense (128 units, ReLU)
        - BatchNormalization
        - Dropout (0.4)
        - Dense (10 units, Softmax)

    Returns:
        Compiled tf.keras.Model
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            # Block 1: Feature Extraction (Low-level features: edges, curves)
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_1"),
            layers.BatchNormalization(name="bn1_1"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1_2"),
            layers.BatchNormalization(name="bn1_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            layers.Dropout(0.25, name="dropout1"),
            # Block 2: Feature Extraction (High-level features: digit loops, crossings)
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_1"),
            layers.BatchNormalization(name="bn2_1"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2_2"),
            layers.BatchNormalization(name="bn2_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
            layers.Dropout(0.25, name="dropout2"),
            # Dense Classification Head
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense1"),
            layers.BatchNormalization(name="bn_dense"),
            layers.Dropout(0.40, name="dropout3"),
            layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
        ],
        name="Handwritten_Digit_CNN",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_mlp_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs and compiles a baseline Multi-Layer Perceptron (MLP)
    to compare spatial feature extraction (CNN) vs flat representations (ANN/MLP).

    Returns:
        Compiled tf.keras.Model
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="mlp_input"),
            layers.Flatten(name="mlp_flatten"),
            layers.Dense(128, activation="relu", name="mlp_dense1"),
            layers.Dropout(0.2, name="mlp_dropout1"),
            layers.Dense(64, activation="relu", name="mlp_dense2"),
            layers.Dropout(0.2, name="mlp_dropout2"),
            layers.Dense(num_classes, activation="softmax", name="mlp_output"),
        ],
        name="Baseline_MLP",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn = build_cnn_model()
    cnn.summary()
