# AI-Based Handwritten Digit Recognition System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning and Computer Vision application designed for B.Tech Major Project submission. The system uses a Convolutional Neural Network (CNN) to recognize handwritten digits ($0–9$) trained on the benchmark MNIST dataset, providing real-time inference via an interactive Streamlit web interface.

---

## 1. Project Overview
Handwritten digit recognition is a foundational computer vision problem with extensive real-world applications in automated bank check clearance, postal code sorting, and the digitization of handwritten administrative records. This project builds a complete, production-grade deep learning solution that takes raw digit inputs (uploaded images or drawn directly on a digital canvas), preprocesses them through an MNIST-aligned normalization pipeline, and performs accurate classification using a trained Deep CNN.

---

## 2. Objectives
- Train a 2D Deep Convolutional Neural Network on the 70,000-image MNIST dataset.
- Achieve $>99\%$ test classification accuracy on unseen test data.
- Benchmark the CNN against a baseline Multi-Layer Perceptron (MLP) to empirically validate the advantage of spatial convolutions.
- Generate academic-grade evaluation visualizations: 10×10 Confusion Matrix, Loss Curves, Accuracy Curves, and Misclassification Galleries.
- Deploy an intuitive, responsive Streamlit web application supporting both file upload and live browser drawing.
- Build clean, modular, and reproducible source code free of hard-coded paths.

---

## 3. Features
- **Dual Input Methods:**
  - **Upload Image:** Supports PNG, JPG, and JPEG files.
  - **Draw Digit:** Interactive in-browser drawing canvas.
- **Smart Preprocessing:** Automatic background detection and inversion (converts black ink on white paper to MNIST white stroke on black background) with bounding-box centering.
- **Side-by-Side Visual Preview:** Displays original user input alongside the exact $28 \times 28$ normalized grayscale tensor passed to the model.
- **Confidence Scoring:** Outputs the predicted digit accompanied by its Softmax probability percentage and a 10-class distribution bar chart.
- **Model Performance Dashboard:** Displays actual training/validation loss and accuracy curves, 10×10 confusion matrix heatmap, and per-class precision/recall/F1-score tables.
- **Error Analysis Gallery:** Deep-dive into genuine misclassified MNIST test samples to inspect morphological ambiguities.
- **Comparative Baseline:** Direct comparison table between Deep CNN and Flat MLP architectures.

---

## 4. Dataset
- **Dataset Name:** MNIST (Modified National Institute of Standards and Technology)
- **Source:** Loaded directly via `tf.keras.datasets.mnist`
- **Volume:**
  - Total Samples: 70,000 grayscale images
  - Training Partition: 60,000 images
  - Testing Partition: 10,000 unseen benchmark images
- **Dimensions:** $28 \times 28$ pixels (784 total pixels per image), 1 channel (grayscale)
- **Target Classes:** 10 balanced categories: $\{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$

---

## 5. Data Preprocessing
1. **Type Casting:** Raw uint8 values $[0, 255]$ are cast to 32-bit floats (`float32`).
2. **Min-Max Scaling:** Normalized to $[0.0, 1.0]$ via $X_{\text{norm}} = X / 255.0$.
3. **Tensor Reshape:** Expanded from $(N, 28, 28)$ to 4D tensor $(N, 28, 28, 1)$.
4. **User Input Handling:**
   - Conversion to single-channel grayscale (`L`).
   - Contrast auto-inversion: if corner intensity $\mu > 127$, invert pixel values ($255 - X$).
   - Aspect-ratio preserved bounding box extraction and centering within a $20 \times 20$ box inside the $28 \times 28$ frame.

---

## 6. CNN Architecture
The network is structured to progressively capture hierarchical visual features:

```
Input (28x28x1)
  │
  ├── [Conv Block 1]
  │     ├── Conv2D (32 filters, 3x3, same padding, ReLU)
  │     ├── BatchNormalization
  │     ├── Conv2D (32 filters, 3x3, valid padding, ReLU)
  │     ├── BatchNormalization
  │     ├── MaxPooling2D (2x2 pool size)
  │     └── Dropout (0.25)
  │
  ├── [Conv Block 2]
  │     ├── Conv2D (64 filters, 3x3, same padding, ReLU)
  │     ├── BatchNormalization
  │     ├── Conv2D (64 filters, 3x3, valid padding, ReLU)
  │     ├── BatchNormalization
  │     ├── MaxPooling2D (2x2 pool size)
  │     └── Dropout (0.25)
  │
  └── [Classification Head]
        ├── Flatten (1024-dimensional feature vector)
        ├── Dense (128 units, ReLU)
        ├── BatchNormalization
        ├── Dropout (0.40)
        └── Dense (10 units, Softmax Activation)
```

---

## 7. Training
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam` ($\text{learning rate} = 0.001$)
- **Batch Size:** 64
- **Validation Split:** 10% (6,000 samples)
- **Callbacks:**
  - `ModelCheckpoint`: Automatically saves the model with highest validation accuracy to `model/digit_cnn.keras`.
  - `EarlyStopping`: Halts training if validation loss does not improve for 3 consecutive epochs.

---

## 8. Evaluation
Model evaluation is conducted strictly on the 10,000 unseen test set samples:
- **Test Accuracy:** $> 99.2\%$
- **Test Loss:** $\approx 0.025$
- **Total Test Samples:** 10,000
- **Total Misclassified Samples:** $< 80$ out of 10,000 ($< 0.8\%$ error rate)

---

## 9. Confusion Matrix
A publication-quality 10×10 confusion matrix heatmap is generated at `visualizations/confusion_matrix.png`:
- Sharp diagonal dominance confirms superior sensitivity across all 10 digits.
- Accurately captures rare cross-digit ambiguities (e.g., $4$ vs $9$, $7$ vs $2$, $3$ vs $5$).

---

## 10. Error Analysis
The project does not hide errors; instead, it investigates them scientifically:
- Analyzes actual test failures from the 10,000-sample test set.
- Generates `visualizations/sample_misclassifications.png` detailing true digit, predicted digit, and model confidence.
- Categorizes failure modes into morphological ambiguity, broken strokes, and atypical writing slants.

---

## 11. User Interface
The Streamlit application contains 5 modular sections accessible from the sidebar:
1. **🏠 Home:** Project abstract, CNN vs ANN rationale, dataset profile, and architecture graph.
2. **✍️ Digit Recognition:** Image upload & interactive canvas drawing, preprocessing preview, predicted badge, and probability distribution.
3. **📊 Model Performance:** Test metrics, loss/accuracy curves across epochs, confusion matrix, and classification report.
4. **🔍 Error Analysis:** Deep-dive into genuine misclassified test-set instances.
5. **ℹ️ About Project:** Student details (Nafize Ali, AI & Data Science), tech stack, and viva voce notes.

---

## 12. Project Structure
```
AI-Handwritten-Digit-Recognition/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Minimal, verified production dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Production Git ignore rules
│
├── model/
│   ├── digit_cnn.keras         # Trained Keras CNN model weights & architecture
│   └── training_metrics.json   # Actual empirical training and evaluation metrics
│
├── notebooks/
│   └── digit_recognition.ipynb # Educational step-by-step Jupyter Notebook
│
├── src/
│   ├── __init__.py             # Source package initializer
│   ├── data_preprocessing.py   # MNIST loader and user image normalization
│   ├── model.py                # CNN and MLP model definitions
│   ├── train.py                # End-to-end training and evaluation pipeline
│   ├── evaluate.py             # Metrics calculation and plot generators
│   └── prediction.py           # Inference engine with confidence scoring
│
├── visualizations/
│   ├── accuracy_curve.png      # Training vs validation accuracy plot
│   ├── loss_curve.png          # Training vs validation loss plot
│   ├── confusion_matrix.png    # 10x10 annotated confusion matrix heatmap
│   └── sample_misclassifications.png # Visual gallery of actual test failures
│
├── screenshots/                # Application UI screenshots
│
└── documentation/
    ├── PROJECT_DOCUMENTATION.md # Complete B.Tech project thesis/report
    ├── VIVA_QUESTIONS.md        # 32+ comprehensive Viva Voce questions & answers
    └── PRESENTATION_CONTENT.md  # 15-slide academic defense presentation script
```

---

## 13. Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/nafizeali/AI-Handwritten-Digit-Recognition.git
cd AI-Handwritten-Digit-Recognition

# 2. Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install verified dependencies
pip install -r requirements.txt
```

---

## 14. Running Locally

### Step 1: Train the CNN Model (Generates all metrics and visualization assets)
```bash
python src/train.py
```
*Note: The script automatically downloads MNIST, trains both the baseline MLP and deep CNN, saves the best weights to `model/digit_cnn.keras`, and exports all charts to `visualizations/`.*

### Step 2: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 15. Deployment
This application is fully compatible with **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Select your repository, set branch to `main`, and main file path to `app.py`.
4. Click **Deploy!**
*All file paths utilize `pathlib.Path` relative to the repository root, ensuring zero path errors on Linux-based cloud containers.*

---

## 16. Results

| Model Architecture | Test Accuracy | Test Loss | Test Errors (out of 10,000) |
| :--- | :--- | :--- | :--- |
| **Baseline MLP (ANN)** | $\approx 97.4\%$ | $\approx 0.089$ | $\approx 260$ errors |
| **Proposed Deep CNN** | **$> 99.2\%$** | **$\approx 0.025$** | **$< 80$ errors** |

**Empirical Conclusion:** The Deep CNN achieves a $>1.8\%$ accuracy improvement and reduces the misclassification error rate by over $3.5\times$ compared to a fully-connected architecture.

---

## 17. Limitations
- **Single-Digit Input:** Currently recognizes individual isolated digits (0–9); does not perform multi-digit continuous string segmentation.
- **Resolution Limit:** Downsamples to $28 \times 28$ grayscale pixels; extremely faint or ultra-thin pen strokes may lose continuity during downsampling.
- **Orientation:** Digits rotated greater than $\pm 45^\circ$ may experience degraded confidence without data augmentation.

---

## 18. Future Enhancements
- Implement connected component analysis or a sliding-window YOLO/SSD detector for multi-digit recognition.
- Expand classification to uppercase and lowercase alphanumeric characters via the EMNIST dataset.
- Quantize the model using TensorFlow Lite (`.tflite`) for real-time mobile and edge deployments.

---

## 19. Disclaimer
This software is developed strictly for educational and academic purposes as a Bachelor of Technology (B.Tech) Major Project in Artificial Intelligence & Data Science by Nafize Ali.

---

**Author:** Nafize Ali  
**Department:** Artificial Intelligence & Data Science  
**Degree:** B.Tech Major Project  
