# Deep Learning Model Comparison Using ANN, CNN, and RNN for Motor Control Prediction
This folder contains an advanced Deep Learning project focused on predicting motor control success or failure using three powerful neural network architectures: Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN).
The project compares model accuracy, learning capability, and suitability for industrial control system data using structured performance metrics.

## Folder Name
09-Deep-Learning-Motor Control

### 1. Dataset
Contains raw dataset file:
- motor_control_performance_dataset_with_target.csv

### 2. Scripts
Contains Python project files:
- motor_control_deep_learning_model.py

### 3. Models
Contains trained model files, weights, and saved checkpoints.

### 4. Reports
Contains confusion matrices, accuracy comparison charts, training logs, and performance insights.

## Files Included
### 1. motor_control_deep_learning_model.py
Python script covering:
- Loading dataset using Pandas
- One-hot encoding categorical columns
- Feature-target separation
- Train-test split
- Feature scaling using StandardScaler
- ANN model building using Dense layers
- CNN model building using Conv1D
- RNN model building using SimpleRNN
- Model training and validation
- Accuracy comparison across models
- Confusion Matrix
- Classification Report
- Sequential data preparation for RNN
- Industrial performance interpretation

### 2. motor_control_performance_dataset_with_target.csv
Industrial motor performance dataset containing operational parameters and final control outcome.
Typical columns include:
- Speed
- Torque
- Flux
- Voltage
- Current
- Temperature
- PID related variables
- Platform / Method
- Disturbance metrics
- Response indicators
- Target (Success = 1, Failure = 0)

## Libraries Used
- pandas
- numpy
- tensorflow
- keras
- scikit-learn
- matplotlib

## Data Preprocessing Techniques Used

### Encoding
- One-hot encoding for categorical variables

### Feature Scaling
- Standardization using StandardScaler

### Sequence Preparation
- Time-step sequence creation for RNN modeling

## Deep Learning Models Implemented

## 1. Artificial Neural Network (ANN)
Architecture:
- Dense Hidden Layer (32 neurons)
- Dense Hidden Layer (16 neurons)
- Sigmoid Output Layer
Best for:
- Tabular industrial data
- Fast baseline predictions
- Nonlinear pattern learning

## 2. Convolutional Neural Network (CNN)
Architecture:
- Conv1D layer
- MaxPooling1D
- Flatten
- Dense Layers
Best for:
- Local feature interactions
- Signal pattern recognition
- Neighboring variable relationships

## 3. Recurrent Neural Network (RNN)
Architecture:
- SimpleRNN layer
- Dense Hidden Layer
- Sigmoid Output
Best for:
- Sequential control behavior
- Time-dependent system responses
- Predicting instability trends

## Model Comparison Focus
Compared based on:
- Accuracy
- Validation Performance
- Pattern Recognition Ability
- Sequential Learning Strength
- Business Applicability

## Expected Insights
- ANN performs strongly on structured tabular data.
- CNN detects local feature combinations effectively.
- RNN performs well when time sequence matters.
- Different architectures suit different industrial problems.

## Skills Demonstrated
- Deep Learning Engineering
- Neural Network Architecture Design
- Model Benchmarking
- Industrial AI Analytics
- Predictive Maintenance Concepts
- TensorFlow Development
- Sequence Modeling
- Classification Optimization

## Business Use Case
Useful for manufacturing plants, automation systems, robotics, and predictive maintenance teams to detect whether motor systems will succeed or fail under operational conditions.

## Real Industry Benefits
- Early failure detection
- Reduced machine downtime
- Better maintenance scheduling
- Higher production efficiency
- Intelligent control system monitoring

## Future Improvements
- LSTM / GRU models
- Hyperparameter tuning
- Real-time IoT sensor integration
- Dashboard deployment
- Explainable AI for fault causes
- Edge AI deployment

## Author
Biswaditya07 (Biswaditya Saha)
MBA in Business Analytics
