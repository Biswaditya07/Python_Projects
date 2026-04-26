# Travel Package Accepted / Rejected Prediction Using Python
This folder contains a Python end-to-end Machine Learning project focused on predicting whether a customer will accept or reject a travel package offer. The project includes data preprocessing, exploratory analysis, multiple classification models, performance comparison, and actionable business insights for travel sales optimization.

## Folder Name
04-Travel-Package-Acceptance

### 1. Dataset
Contains raw dataset file:
- Travel.csv

### 2. Scripts
Contains Python project files:
- random_forest.py

### 3. Reports
Contains model outputs, confusion matrices, charts, accuracy reports, and business insights.

## Files Included

### 1. random_forest vs other_models.py
Python script covering:
- Loading CSV dataset using Pandas
- Missing value analysis
- Numerical missing value treatment using imputation
- Exploratory Data Analysis (EDA)
- Customer behavior visualization
- One-Hot Encoding for categorical variables
- Standardization of numerical variables
- Train-test split
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Model training and prediction
- Accuracy, Precision, Recall, F1-score, ROC-AUC evaluation
- Confusion Matrix comparison
- Hyperparameter tuning preparation

### 2. Travel.csv
Travel sales dataset containing customer demographics, income, occupation, engagement history, and travel package response.
Typical columns include:
- Customer ID
- Age
- Gender
- Marital Status
- Occupation
- Monthly Income
- Number of Trips
- Passport Status
- Product Pitch Satisfaction
- Type of Contact
- Number of Follow-ups
- City Tier
- Existing Product Details
- Preferred Property Star Rating
- Number of Children
- Target Variable: ProdTaken  
  - 0 = Rejected Offer  
  - 1 = Accepted Offer

## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

## Data Preprocessing Techniques Used

### Missing Value Handling
- Identified null values across columns
- Mean imputation after transformation pipeline

### Encoding
- One-Hot Encoding for categorical features

### Feature Scaling
- Standardization using StandardScaler

### Feature Engineering
- Structured transformation using ColumnTransformer pipeline

## Exploratory Data Analysis Performed

### Customer Acceptance Analysis
- Product acceptance distribution

### Demographic Insights
- Age vs acceptance
- Income vs acceptance
- Occupation vs acceptance

### Sales Channel Insights
- Type of contact vs product taken

## Modeling Techniques Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Model Evaluation Metrics
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## Skills Demonstrated
- Marketing Analytics
- Customer Segmentation
- Sales Prediction
- Data Cleaning
- Feature Engineering
- Classification Modeling
- Ensemble Learning
- Model Comparison
- Business Intelligence

## Business Use Case
Help travel companies identify customers most likely to purchase travel packages, improve campaign targeting, reduce sales effort wastage, increase conversion rates, and optimize revenue through data-driven marketing.

## Key Business Insights
- Customers with stronger income profiles may show higher conversion probability.
- Contact strategy influences acceptance rates.
- Predictive targeting can reduce cold-calling inefficiency.
- Ensemble models like Random Forest and Gradient Boosting can improve conversion prediction accuracy.

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- XGBoost / CatBoost integration
- Lead scoring dashboard in Power BI
- Real-time CRM prediction deployment

## Author
Biswaditya07 (Biswaditya Saha) 
MBA in Business Analytics
