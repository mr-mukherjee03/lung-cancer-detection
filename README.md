# 3D Lung Cancer Detection using LUNA16 Dataset

This project implements a 3D Convolutional Neural Network (CNN) to detect and classify lung cancer nodules from 3D CT scans. The model is trained on the publicly available [LUNA16 (LUng Nodule Analysis 16)](https://luna16.grand-challenge.org/) dataset.

![Lung CT Scan](https://img.shields.io/badge/Domain-Medical%20Imaging-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

##  Table of Contents
* [Project Overview](#-project-overview)
* [Features](#-features)
* [Dataset](#-dataset)
* [Architecture](#-architecture)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#-usage)
  * [1. Data Preprocessing](#1-data-preprocessing)
  * [2. Training the Model](#2-training-the-model)
  * [3. Evaluation](#3-evaluation)
* [File Structure](#-file-structure)
* [Contributing](#-contributing)
* [License](#-license)

##  Project Overview
The goal of this project is to build an accurate deep learning model for early-stage lung cancer detection. By leveraging 3D CNNs, the model can analyze volumetric data from CT scans to learn spatial features indicative of malignant nodules, which is often more effective than analyzing 2D slices independently.

##  Features
* **End-to-End Pipeline:** From data preprocessing of `.mhd` files to model training and evaluation.
* **3D Data Processing:** Scripts to handle 3D medical imaging data, including normalization and segmentation.
* **3D CNN Model:** A custom 3D CNN architecture designed for nodule classification.
* **Modular and Extendable:** The code is structured to allow for easy experimentation with different models and preprocessing techniques.

##  Dataset
This project uses the **LUNA16 dataset**, which is a large-scale, publicly available dataset for lung nodule detection. It contains 888 low-dose lung CT scans with annotations for nodules larger than 3mm.

You will need to download the dataset from the [official LUNA16 challenge website](https://luna16.grand-challenge.org/data/) and place it in the appropriate directory.

##  Architecture
The core of the project is a 3D Convolutional Neural Network. The typical architecture consists of:
1.  **Input Layer:** Accepts 3D chunks of CT scans (e.g., 64x64x64 voxels).
2.  **Convolutional Layers:** 3D convolutional layers with ReLU activation to extract hierarchical features.
3.  **Pooling Layers:** 3D Max-Pooling layers to reduce spatial dimensions.
4.  **Fully Connected Layers:** A series of dense layers for classification.
5.  **Output Layer:** A Softmax layer to output the probability of a nodule being an actual nodule.

##  Getting Started

### Prerequisites
* Python 3.8+
* Git

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mr-mukherjee03/lung-cancer-detection.git](https://github.com/mr-mukherjee03/lung-cancer-detection.git)
    cd lung-cancer-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

##  Usage
Run the notebook run_everything.ipynb for training and visualization.
Run ```tensorboard --logdir=runs``` to view training metrics.
