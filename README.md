# Brain Tumor Detection Using CNNs

A deep learning project that uses Convolutional Neural Networks (CNNs) to detect brain tumors from MRI images.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Getting Started](#getting-started)  
- [Requirements](#requirements)  
- [Usage](#usage)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)

---

## Project Overview

This project implements a CNN-based model to classify MRI brain scans as tumor or non-tumor. The workflow includes preprocessing, building and training the CNN model, evaluating its performance, and visualizing results.

---

## Dataset

- **Source:** [Add dataset source link here]  
- **Number of images:** [Add total images]  
- **Classes:** Tumor, Non-Tumor (or specify tumor types if multiple)  
- **Preprocessing:** Resizing, normalization, data augmentation (rotation, flip, etc.)

---

## Methodology

- **Preprocessing:** Images resized to uniform dimensions, normalized, and augmented.  
- **CNN Architecture:** Multiple convolutional and pooling layers, dropout for regularization, dense layers for classification.  
- **Training:** Optimizer: Adam, Loss: categorical crossentropy, Epochs: 25, Batch size: 32  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Getting Started

These instructions will help you run the project on your local machine.

1. **Clone the repository**:

```bash
git clone https://github.com/prek5h4/Brain-Tumor-Detection-Using-CNNs.git
cd Brain-Tumor-Detection-Using-CNNs
```
Prepare the dataset:

Download the MRI dataset and place it in the folder dataset/ (or update paths in the notebook).

Ensure the images are organized into class folders (e.g., Tumor/, No_Tumor/).

Open the notebook:

jupyter notebook brain-tumor.ipynb


Run cells sequentially:

Data preprocessing (resizing, normalization, augmentation)

Model building and training

Model evaluation and visualization of results

##Requirements

Python 3.8 or higher

##Libraries:

pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python seaborn


Jupyter Notebook or JupyterLab

Optional for GPU acceleration: CUDA-enabled GPU and compatible TensorFlow version.

Usage

Preprocess the images: resizing to 64x64 (or 128x128 depending on notebook) and normalization.

Build and train the CNN model:

# Example from notebook
model.fit(train_data, validation_data=val_data, epochs=25, batch_size=32)


Evaluate the model:

loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)


Visualize predictions:

# Display a few predictions with images
plot_predictions(model, test_data)
