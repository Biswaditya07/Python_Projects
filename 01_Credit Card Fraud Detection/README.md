# Credit Card Fraud Detection

# Dataset Overview
The Credit Card Fraud Detection dataset contains transactional records made by European cardholders over a span of two days. It comprises approximately 284,807 observations, each representing an individual transaction. The primary objective of this dataset is to identify fraudulent transactions among a large volume of legitimate ones, making it a widely used benchmark for classification and anomaly detection problems in machine learning.

# Structure and Variables
The dataset consists of 31 variables, including both independent and dependent features. The majority of the variables, labeled from V1 to V28, are anonymized numerical features obtained through Principal Component Analysis (PCA). This transformation ensures confidentiality of sensitive financial information while preserving the underlying patterns necessary for analysis. In addition to these, there are two original features: “Time,” which represents the number of seconds elapsed since the first transaction, and “Amount,” which indicates the monetary value of the transaction. The dependent variable, “Class,” is binary in nature, where a value of 0 denotes a legitimate transaction and 1 indicates a fraudulent one.

# Nature of the Dataset
One of the most defining characteristics of this dataset is its extreme class imbalance. The overwhelming majority of transactions are legitimate, accounting for approximately 99.83% of the data, while fraudulent transactions make up only about 0.17%. This imbalance presents a significant challenge for predictive modeling, as traditional accuracy metrics may not effectively capture a model’s ability to detect fraud.

# Data Quality and Preprocessing
The dataset is clean and does not contain any missing or null values, which eliminates the need for imputation techniques. Furthermore, since most of the variables have already been transformed using PCA, they are standardized and comparable in scale. However, the “Time” and “Amount” features are not normalized and may require scaling to ensure that they do not disproportionately influence the performance of machine learning algorithms.

# Business Context and Importance
In a real-world business setting, the dataset reflects the critical challenge faced by financial institutions in detecting fraudulent activities. Although fraudulent transactions are rare, their financial impact can be substantial. Therefore, the focus is not only on correctly identifying fraud but also on minimizing false negatives, which represent missed fraud cases. At the same time, reducing false positives is important to maintain a positive customer experience. This balance highlights the importance of developing robust and sensitive fraud detection systems.

# Credit Card Fraud Detection Using Python
This folder contains a Python hands-on project focused on analyzing credit card transaction data, preprocessing features, building classification models, and detecting fraudulent transactions using Machine Learning techniques.

## Folder Name
Credit-Card-Fraud-Detection

## Files Included

### 1. credit_card_fraud_detection.py
Python script covering:
- Loading transaction dataset using Pandas
- Checking missing values
- Feature scaling using StandardScaler
- Exploratory Data Analysis
- Correlation analysis
- Logistic Regression modeling
- Decision Tree Classification
- Confusion Matrix generation
- Accuracy, Precision, Recall, F1-score evaluation
- Fraud detection insights

### 2. creditcard.csv
Credit card transaction dataset containing:
- Transaction time details
- Transaction amount
- V1 to V28 anonymized PCA transformed features
- Fraud classification label

## Libraries Used
- pandas
- scikit-learn
- matplotlib
- seaborn

## Data Preprocessing Techniques Used

### Missing Value Handling
- Null value checking
- Data validation before modeling

### Feature Scaling
- Standardization using StandardScaler for Time and Amount

### Feature Engineering
- Created scaled versions of transaction features
- Removed original unscaled columns

## Modeling Techniques Used
- Logistic Regression
- Decision Tree Classifier

## Skills Demonstrated
- Fraud Detection Analytics
- Data Cleaning
- Feature Engineering
- Classification Modeling
- Model Comparison
- Performance Evaluation
- Business Insights Generation

## Business Use Case
Identify suspicious credit card transactions, reduce financial fraud losses, and improve real-time transaction monitoring systems.

## Author
Biswaditya07 (Biswaditya Saha)
MBA in Business Analytics
