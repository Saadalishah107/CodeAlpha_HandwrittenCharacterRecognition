# Optical Character Recognition for Handwritten Text via Deep Convolutional Networks

## 1. Project Overview
This repository contains a deep learning pipeline for Optical Character Recognition (OCR), specifically designed to classify handwritten alphabetic characters. By implementing a custom Convolutional Neural Network (CNN) architecture, this project addresses the inherent challenges of handwritten text analysis, including high variance in stroke thickness, orientation, and visually ambiguous handwriting styles.

## 2. Research Objectives
* **Develop a Generalizable Classifier:** Build a CNN model capable of accurately identifying 26 distinct classes of the English alphabet (A–Z) from raw pixel data.
* **Overcome Visual Ambiguity:** Implement advanced regularization and training techniques (such as label smoothing) to handle characters with high structural similarity (e.g., 'I' vs. 'L', 'C' vs. 'G').
* **Optimize Architectural Efficiency:** Design a network that maximizes spatial feature extraction using Global Average Pooling (GAP) to minimize computational overhead and avoid parameter bloat.

## 3. Dataset Characteristics
The model is trained and evaluated using the **EMNIST (Extended MNIST) Letters** dataset, which serves as a more challenging and practical extension of the traditional MNIST digits dataset.
* **Total Samples:** 103,600 merged training and testing images.
* **Format:** 28x28 pixel grayscale images, normalized to a [0, 1] scale.
* **Classes:** 26 balanced classes representing alphabetical letters.

## 4. Model Architecture (v2)
The network is structured to progressively extract hierarchical spatial features, moving from simple edges to complex character topologies. The final model contains **~1.2M trainable parameters**.
* **Feature Extraction (4 Convolutional Blocks):** 
  * **Block 1:** Conv2D (32) → BatchNorm → Conv2D (32) → MaxPool → Dropout (0.25)
  * **Block 2:** Conv2D (64) → BatchNorm → Conv2D (64) → MaxPool → Dropout (0.25)
  * **Block 3:** Conv2D (128) → BatchNorm → MaxPool → Dropout (0.25)
  * **Block 4:** Conv2D (256) → BatchNorm → Dropout (0.30)
* **Dimensionality Reduction:** **Global Average Pooling (GAP)** is implemented prior to the dense layers to reduce the parameter count and mitigate overfitting, serving as a more robust alternative to standard Flatten layers.
* **Classification Head:** A fully connected Dense layer (512 units) with BatchNorm and Dropout (0.50) maps the extracted features to the final 26-class softmax probability distribution.

## 5. Training Methodology
* **Data Augmentation:** Real-time spatial transformations including Random Rotation (±8°), Random Zoom (±10%), and Random Translation (±8%).
* **Loss Function:** Categorical Cross-Entropy integrated with **Label Smoothing (0.1)**. This softens the target distributions, preventing the model from becoming overly confident in ambiguous cases.
* **Optimization Strategy:** Adam optimizer (initial learning rate = 1e-3).
* **Callbacks:** 
  * `ReduceLROnPlateau` (factor=0.5, patience=6) dynamically decays the learning rate when validation accuracy stagnates.
  * `EarlyStopping` (monitor=val_accuracy, patience=12) halts training to capture the optimal weight configuration.
* **Batch Size:** 256
* **Epochs:** 60

## 6. Results and Key Takeaways
* **Performance:** The model achieved a **68.40% Test Accuracy**. While character-level classification is inherently bottlenecked by human annotation inconsistencies (e.g., a poorly drawn 'O' vs. 'Q'), the network demonstrates strong feature extraction capabilities.
* **Architectural Stability:** The combination of Batch Normalization and aggressive Dropout provided strong regularization, preventing the model from memorizing the training set.
* **Impact of Label Smoothing:** Setting label smoothing to 0.1 proved highly effective in mitigating extreme confidence penalties when evaluating fundamentally ambiguous character pairs (like 'I' and 'L').

## 7. Future Scope
* **Extended Dataset Training:** Expanding the pipeline to the EMNIST Balanced dataset (47 classes, including digits and case-sensitive letters).
* **Sequence-to-Sequence Modeling:** Evolving the architecture into a Convolutional Recurrent Neural Network (CRNN) by adding LSTM layers to contextually process full cursive words rather than isolated characters.
* **Interactive Deployment:** Wrapping the inference logic in a Gradio or Streamlit web application with a live drawing canvas.
* **Edge Optimization:** Quantizing the model weights for TensorFlow Lite (TFLite) to enable low-latency, on-device mobile inference.
