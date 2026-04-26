# Deep Learning ANN Using TensorFlow / PyTorch with TorchVision
This folder contains a hands-on Deep Learning project focused on building Artificial Neural Networks (ANN) for image classification and binary classification tasks. The project demonstrates importing datasets from TorchVision, transforming images into tensors, training feedforward neural networks, and evaluating model performance.
It is designed to showcase core Deep Learning concepts such as tensors, layers, activation functions, loss functions, optimization, and backpropagation.

## Folder Name

08-Deep-Learning-ANN

### 1. Dataset
Contains downloaded datasets:
- MNIST handwritten digit dataset
- Synthetic classification datasets

### 2. Scripts
Contains Python project files:
- deep_learning_ann.py

### 3. Models
Contains trained ANN model checkpoints / saved weights.

### 4. Reports
Contains charts, training logs, prediction outputs, and learning summaries.

## Files Included

### 1. deep_learning_ann.py
Python script covering:
- Importing datasets from TorchVision
- Applying transforms using ToTensor()
- Downloading MNIST training and testing data
- Tensor shape inspection
- Label distribution analysis
- Image visualization using Matplotlib
- ANN model creation using neural network layers
- Hidden layer design
- ReLU activation function
- Sigmoid output activation
- Binary Cross Entropy loss
- SGD optimizer
- Forward propagation
- Backpropagation
- Model training loop
- Accuracy evaluation
- Prediction testing on new inputs

## Dataset Used

### MNIST Dataset
Handwritten digit recognition dataset containing:
- 60,000 training images
- 10,000 testing images
- Grayscale images (28x28)
- Labels from 0 to 9

### Synthetic Dataset
Generated using Scikit-learn:
- Nonlinear circular classification data
- Binary target labels

## Libraries Used
- torch
- torchvision
- tensorflow concepts
- scikit-learn
- matplotlib
- numpy

## Deep Learning Concepts Demonstrated

### Tensor Operations
- Convert images into tensors
- Numeric matrix processing
- Batch handling using DataLoader

### Neural Network Architecture
- Input Layer
- Hidden Layers
- Output Layer

### Activation Functions
- ReLU
- Sigmoid

### Training Process
- Forward Pass
- Loss Calculation
- Backward Pass
- Gradient Descent
- Weight Updates

### Evaluation
- Prediction thresholding
- Test accuracy measurement

## Skills Demonstrated
- Deep Learning Fundamentals
- ANN Development
- Image Classification
- Tensor Processing
- Neural Network Training
- Computer Vision Basics
- Model Evaluation
- Python AI Development

## Business Use Case
Useful for building intelligent systems such as:
- Digit Recognition
- OCR Systems
- Fraud Detection
- Customer Churn Prediction
- Medical Diagnosis Models
- Credit Risk Prediction
- Recommendation Engines

## Key Insights
- Tensor conversion is essential for neural networks.
- Hidden layers learn nonlinear patterns effectively.
- ReLU improves training efficiency.
- Backpropagation helps optimize weights iteratively.
- ANN models can solve both image and tabular problems.

## Future Improvements
- CNN for image classification
- TensorFlow Keras implementation
- GPU acceleration
- Dropout regularization
- Hyperparameter tuning
- Real-time deployment API

## Author
Biswaditya07 (Biswaditya Saha) 
MBA in Business Analytics
