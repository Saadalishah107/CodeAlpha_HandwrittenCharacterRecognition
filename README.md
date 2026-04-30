# Handwritten Character Recognition
### CodeAlpha Machine Learning Internship — Task 1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Dataset](https://img.shields.io/badge/Dataset-EMNIST%20Letters-green)


---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten characters and alphabets from images. Built as Task 1 of the **CodeAlpha Machine Learning Internship**, it covers the complete ML pipeline — from raw data loading and exploratory analysis to model training, evaluation, and inference.

**Dataset:** EMNIST Letters — 88,800 grayscale images (28×28 px) across 26 classes (A–Z)  
**Best Metric:** Test accuracy logged live during training via EarlyStopping on validation loss

---

## 🗂️ Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
│
├── CodeAlpha_HandwrittenCharacterRecognition.ipynb   ← Main notebook (run this)
├── README.md                                          ← This file
├── best_model.keras                                   ← Saved best checkpoint (after training)
└── handwritten_char_recognition_savedmodel/           ← SavedModel export (after training)
```

---

## 🧠 Model Architecture

```
Input (28×28×1)
    │
    ├── Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── GlobalAveragePooling2D
    │
    ├── Dense(256) → BatchNorm → Dropout(0.50)
    │
    └── Output: Dense(26, softmax)
```

| Attribute | Value |
|---|---|
| Total Parameters | ~350K |
| Optimizer | Adam (lr=1e-3) |
| Loss Function | Categorical Cross-Entropy |
| Regularization | BatchNormalization + Dropout |
| Augmentation | Random Rotation, Zoom, Translation |

---

## 📊 Pipeline Summary

| Step | Details |
|---|---|
| **Data Loading** | EMNIST Letters via `tensorflow_datasets`; falls back to MNIST if unavailable |
| **EDA** | Class distribution, sample image grids, pixel intensity analysis |
| **Preprocessing** | Normalize → Reshape (N,28,28,1) → One-hot encode → 90/10 train/val split |
| **Augmentation** | ±8° rotation, ±10% zoom, ±8% translation (applied on-the-fly during training) |
| **Training** | Up to 40 epochs with EarlyStopping (patience=8) + ReduceLROnPlateau |
| **Evaluation** | Confusion matrix, classification report, per-class accuracy |
| **Inference** | `predict_character()` function with top-5 confidence visualization |
| **Export** | `.keras` format + TensorFlow SavedModel |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Saadalishah107/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn tensorflow-datasets
```

### 3. Run the Notebook

**Option A — Google Colab (Recommended):**  
Upload the `.ipynb` file to [colab.research.google.com](https://colab.research.google.com) and run all cells. GPU runtime is available for free under *Runtime → Change runtime type → T4 GPU*.

**Option B — Locally with Jupyter:**
```bash
pip install jupyter
jupyter notebook CodeAlpha_HandwrittenCharacterRecognition.ipynb
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building & training |
| `tensorflow-datasets` | EMNIST dataset loading |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting & visualization |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Metrics, train/val split |

---

## 🔍 Results

After training, the notebook produces:
- **Accuracy & Loss curves** across epochs
- **Confusion Matrix** heatmap for all 26 classes
- **Per-class accuracy** bar chart (color-coded: green ≥ 90%, orange ≥ 75%, red < 75%)
- **Correct vs. Misclassified** sample grids with confidence scores
- **Confidence distribution** comparison (correct vs. wrong predictions)

Misclassifications are expected mainly between visually similar characters (e.g., **I ↔ l**, **O ↔ 0**, **C ↔ G**).

---

## 🔭 Possible Extensions

- **EMNIST Balanced** — extend to 47 classes (digits + upper + lowercase)
- **CRNN (CNN + LSTM)** — for full word or sentence recognition
- **Gradio / Streamlit App** — live drawing canvas for real-time prediction
- **TFLite Export** — quantize and deploy on mobile devices

---

## 👤 Author

**Syed Muhammad Saad Ali Shah**  
BS Bioinformatics, Quaid-i-Azam University  
Machine Learning Intern @ CodeAlpha  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/syed-muhammad-saad-ali-shah-63a1331a9)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/)

---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company driving innovation through AI and intelligent systems. This project was completed as part of their Machine Learning Internship Program.

🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

*⭐ If you found this project useful, consider starring the repository!*
