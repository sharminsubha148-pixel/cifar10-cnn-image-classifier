Deep-CIFAR10: High-Performance Image Classification Using Custom Deep CNN Architecture
A production-ready, professionally structured Deep Learning pipeline engineered from scratch using TensorFlow and Keras. This project showcases an advanced Convolutional Neural Network (CNN) optimized for classifying multi-class data from the CIFAR-10 dataset—demonstrating absolute data preprocessing, modern layer regularization, validation callback checkpoints, and complete mathematical diagnostic reports.

Deep Learning Project Architecture & Workflow Details
The core notebook pipeline (100% project code.ipynb) executes a strict, automated Deep Learning workflow. Each structural layer is strategically designed to maintain absolute gradient flow, maximize deep spatial feature extraction, and stabilize training convergence:

1. Advanced Data Ingestion & Tensor Normalization
4D Tensor Structuring: Dynamically ingests, reshapes, and maps raw image matrices into strict 4D tensors (batch_size, height, width, channels) required for CNN tensor inputs.
Pixel Vector Scaling: Normalizes raw pixel values from 
[
0
,
255
]
 directly into an optimized 
[
0
,
1
]
 mathematical distribution. This prevents immediate gradient explosion and stabilizes backpropagation weights.
Categorical One-Hot Encoding: Encodes categorical class targets into logical binary matrices to properly optimize multi-class categorical cross-entropy objectives.
2. Live Data Augmentation (Preventing Neural Overfitting)
To maximize model robustness and ensure high performance against unseen real-world images, an automated image augmentation framework transformations inputs dynamically during runtime:

Random Horizontal & Vertical Flips
Adaptive Spatial Rotations
Dynamic Width/Height Shift and Scaling Matrices
3. Deep Custom CNN Architectural Topology
The neural architecture bypasses basic linear models to incorporate a deep, multi-layered computer vision backbone:

Convolutional Stages (Conv2D): Employs multi-layer stacked filters (kernels) to capture spatial hierarchies, scanning simple primitive boundaries in lower levels and complex target shapes in deep feature maps.
Non-Linear Transformations (ReLU): Activates hidden layers via Rectified Linear Units to implement complex mathematical non-linearity, allowing the model to adapt to complex mathematical target curves.
Spatial Compressing (MaxPooling2D): Subsamples structural maps to reduce active spatial dimensions. This cuts down hardware computation costs while preserving highly active edge activations.
Network Stability Layer (BatchNormalization): Standardizes network layer outputs across batches, reducing internal covariate shift and ensuring rapid learning rates.
Overfitting Regularization (Dropout): Abruptly silences targeted percentages of internal neurons during backward passes, forcing distributed learning paths and reducing feature co-dependency.
Dense Softmax Classification Head: Flattens spatial feature vectors into structured 1D linear tensors, routing predictions to fully connected dense layers ending with a Softmax mathematical function to yield exact class probabilities.
4. Mathematical Model Compilation
The neural network relies on standard mathematical hyperparameters:

Optimizer: Powered by the Adam Optimizer, utilizing adaptive first and second-moment learning rate tracking to traverse steep loss surfaces safely.
Loss Function: Configured with CategoricalCrossentropy to heavily penalize margins of error between soft-max outputs and one-hot vectors.
Metrics Engine: Continuously tracks performance using Accuracy calculations on isolated training and evaluation sets.
5. Advanced Callbacks & Training Controls
EarlyStopping: Continuously monitors real-time validation losses. If the validation loss fails to improve across a strict patience window, training aborts instantly to secure optimal weights.
ModelCheckpoint: Audits continuous accuracy weights, saving only the premium historical state configuration to disk (cifar10_weights.weights.h5), eliminating overtraining deterioration.
6. Pipeline Diagnostics & Evaluation Metrics
Loss/Accuracy Curves: Generates visual line graphs comparing training history against validation parameters to pinpoint any subtle underfitting or overfitting trends.
Confusion Matrix: Constructs an 
N
×
N
 mapping table tracking true classes against model classifications to visually expose subtle target vulnerabilities.
Classification Report: Outputs complete performance metrics detailing high-precision mathematical metrics including Precision, Recall, and F1-Score per target class.
Environment Setup & Dynamic Installation
To run this Deep Learning project smoothly on your local workstation without package version errors, initialize your terminal and run the standard dependency command below:

pip install -r requirements.txt



# 🤖 CIFAR-10 Image Classification Web App

This is a Deep Learning-based image classification web application. In this project, a custom neural network model (`cifar10_model.h5`) has been utilized to successfully classify images into 10 distinct categories from the **CIFAR-10** dataset. When a user uploads any image from their local computer, the model automatically predicts the object inside the image and displays its confidence score.

---

## Features
* **Interactive UI:** Built using the Streamlit framework to provide a highly attractive and user-friendly interface.
* **Real-time Prediction:** As soon as an image is uploaded, the backend performs image pre-processing and displays instant results.
* **Confidence Breakdown:** It shows not only the predicted class but also the percentage probability of the remaining 9 classes in a text format.
* **Robust Error Handling:** Includes built-in mechanisms to handle any potential errors during model loading or image processing.

---

## Project Setup & Folder Structure

To run the application 100% correctly, all the following files and folders must be kept together inside a single folder named **`web app`** on your Desktop.

### 1. Final Look Inside the Folder:
When you open your Desktop's `web app` folder, exactly these items must be present together (no file should be placed inside any other sub-folder):
* **`myenv`** (The main Python Virtual Environment folder)
* **`cifar10_model.h5`** (Your trained main model file)
* **`app.py`** (The Streamlit web application Python code file)
* **`100% project code.ipynb`** (The main Jupyter Notebook file containing model training and research)

---

## 🚀 How to Run the Project via CMD (Instructions for Judges)

To run this web application on your local computer without any errors, execute the following 4 commands sequentially in your **Command Prompt (cmd)**:

### Step 1: Navigate to the Project Directory
First, open CMD on your computer, type the following command to enter the specific project path, and press Enter:
```bash
cd "C:\Users\Suvha\Desktop\web app"
myenv\Scripts\activate
pip install streamlit numpy pillow tensorflow
streamlit run app.py
