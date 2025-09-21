# Brain Tumor Detection Using CNNs 🧠

This project utilizes a **Convolutional Neural Network (CNN)** to accurately classify brain MRI scans, identifying the presence or absence of a tumor. The model is built with TensorFlow and Keras and is designed to serve as a robust tool for medical image analysis.


---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Usage](#-usage)

---

## 🚀 Project Overview

This project implements a CNN model to classify brain MRI scans into two categories: **Tumor** and **Non-Tumor**. The complete workflow involves data preprocessing, model architecture design, training, performance evaluation, and visualization of the results to ensure high accuracy and reliability.

---

## 📊 Dataset

The model was trained on a dataset of brain MRI images.

* **Source**: [Kaggle Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Total Images**: 253
* **Classes**:
    * `yes` (Tumor Detected)
    * `no` (No Tumor Detected)
* **Preprocessing**: To ensure model efficiency and accuracy, the following preprocessing steps were applied:
    * **Resizing**: Images were resized to a uniform `128x128` pixels.
    * **Normalization**: Pixel values were scaled to a range of `[0, 1]`.
    * **Data Augmentation**: Techniques like rotation, horizontal/vertical flips, and zooming were used to expand the dataset and prevent overfitting.

---

##  Methodology

The core of this project is a CNN designed for image classification.

* **CNN Architecture**: The model consists of multiple convolutional layers with `ReLU` activation, followed by `MaxPooling2D` layers for feature extraction. `Dropout` layers are included for regularization, and the final classification is performed by dense layers with a `sigmoid` activation function.
* **Training Parameters**:
    * **Optimizer**: Adam
    * **Loss Function**: Binary Crossentropy
    * **Epochs**: 25
    * **Batch Size**: 32
* **Evaluation Metrics**: The model's performance was assessed using:
    * Accuracy
    * Precision, Recall & F1-Score
    * Confusion Matrix

---

##  Results

The trained model achieved a high level of performance on the test dataset.

* **Test Accuracy**: `97.5%`
* **Loss**: `0.08`


---

## 🏁 Getting Started

Follow these instructions to set up and run the project on your local machine.

### **1. Prerequisites**

Ensure you have **Python 3.8+** installed. You will also need a virtual environment tool like `venv` or `conda`.

### **2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/prek5h4/Brain-Tumor-Detection-Using-CNNs.git](https://github.com/prek5h4/Brain-Tumor-Detection-Using-CNNs.git)
    cd Brain-Tumor-Detection-Using-CNNs
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a `requirements.txt` file with the libraries listed below for easy installation.)*

### **Required Libraries**
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
opencv-python
seaborn
jupyter

---

## 💻 Usage

1.  **Prepare the Dataset**:
    * Download the dataset from the link provided above.
    * Place the images in a `dataset/` folder within the project directory.
    * Ensure the images are organized into subfolders corresponding to their classes (e.g., `dataset/yes/` and `dataset/no/`).

2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook brain-tumor.ipynb
    ```

3.  **Run the Notebook Cells**:
    Execute the cells sequentially to perform:
    * Data loading and preprocessing.
    * Model building and compilation.
    * Model training.
    * Performance evaluation and visualization of predictions.

---
