# Naive Bayes Salary Prediction Using Python (Gaussian vs Multinomial)
This folder contains a Python Machine Learning project focused on predicting salary categories using Naive Bayes classification techniques. The project compares Gaussian Naive Bayes and Multinomial Naive Bayes models to determine the better-performing algorithm for income prediction.

## Folder Name
05-Naive-Bayes

### 1. Dataset
Contains raw dataset files:
- SalaryData_Train.csv
- SalaryData_Test.csv

### 2. Scripts
Contains Python project files:
- Naive_Bayes_Model_Gaussian_Vs_Multinomial.py

### 3. Reports
Contains model outputs, confusion matrices, charts, and performance comparison summaries.

## Files Included

### 1. Naive_Bayes_Model_Gaussian_Vs_Multinomial.py
Python script covering:
- Loading training and testing datasets using Pandas
- Encoding categorical variables using LabelEncoder
- Splitting features and target variable
- Gaussian Naive Bayes model building
- Multinomial Naive Bayes model building
- Accuracy score comparison
- Classification Reports
- Confusion Matrix visualization
- Model performance interpretation
- Income prediction insights

### 2. SalaryData_Train.csv / SalaryData_Test.csv
Salary classification dataset containing employee demographic and professional details.
Typical columns include:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country
- Salary Target Variable  
  - <=50K  
  - >50K

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data Preprocessing Techniques Used

### Encoding
- Label Encoding for categorical variables

### Data Preparation
- Separate training and testing datasets
- Feature-target split

## Modeling Techniques Used

### Gaussian Naive Bayes
Best suited for continuous numerical variables assuming normal distribution.

### Multinomial Naive Bayes
Best suited for count/frequency based or discrete features.

## Model Comparison Summary

### Gaussian Naive Bayes
- Better overall accuracy
- Higher precision
- Better recall
- Stronger F1-score

### Multinomial Naive Bayes
- Slightly lower performance
- Useful for count-based feature scenarios

## Evaluation Metrics Used
- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

## Skills Demonstrated
- HR Analytics
- Salary Prediction
- Classification Modeling
- Probabilistic Algorithms
- Model Comparison
- Data Preprocessing
- Performance Evaluation

## Business Use Case
Help HR teams and organizations estimate employee income brackets, support compensation analytics, workforce segmentation, and predictive hiring intelligence.

## Key Insights
- Gaussian Naive Bayes performed better due to presence of continuous numerical variables.
- Proper encoding significantly improves model readiness.
- Naive Bayes offers fast and scalable classification for business datasets.

## Future Improvements
- Hyperparameter tuning
- Feature selection
- Cross Validation
- Ensemble models for comparison
- Dashboard deployment using Power BI

## Author
Biswaditya07 (Biswaditya Saha)
MBA in Business Analytics
