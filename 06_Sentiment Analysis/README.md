# SQL + Python Sentiment Analysis Using Database Retrieval
This folder contains an end-to-end Python project focused on connecting SQL databases with Python, retrieving sentiment data from a database, performing exploratory data analysis, training multiple Machine Learning models, and predicting sentiment classes.

The project demonstrates how structured sentiment datasets stored in SQL systems can be integrated into Python analytics workflows for classification and business intelligence.

## Folder Name
06-SQL+Python-Sentiment-Analysis

### 1. Dataset
Contains source dataset file:
- testsentimentclass.csv

### 2. Scripts
Contains Python project files:
- Sql+python Sentiment Analysis.py

### 3. Database
Contains SQL / AWS RDS connection workflow documentation.

### 4. Reports
Contains charts, model outputs, accuracy summaries, and feature importance reports.

## Files Included

### 1. Sql+python Sentiment Analysis.py
Python script covering:
- Connecting Python with MySQL database using SQLAlchemy
- AWS RDS database integration
- Loading CSV data into DataFrame
- Retrieving structured sentiment data from SQL source
- Data exploration using Pandas
- Null value checks
- Descriptive statistics
- Correlation analysis
- Class distribution analysis
- Multiple Machine Learning model training
- Accuracy, Precision, Recall, F1-score comparison
- Feature importance using Random Forest
- Data visualization and reporting

### 2. testsentimentclass.csv
Structured sentiment dataset transformed from text-based content into numerical features.
Each row represents a report, review, or text instance converted into measurable indicators.
Typical columns include:
- achievement
- approve
- growth
- increased
- reduction
- below
- Multiple sentiment keyword frequency features
- actual (Target Variable)

### Target Variable
- -1 = Negative Sentiment
- 0 = Neutral Sentiment
- 1 = Positive Sentiment

## Database Integration Used

### SQL Connectivity
- MySQL database connection
- SQLAlchemy engine creation
- Python database querying
- AWS RDS cloud database integration

### Business Relevance
Shows how production databases can directly feed Machine Learning pipelines.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- sqlalchemy
- pymysql
- xgboost

## Data Preprocessing Techniques Used

### Data Validation
- Null value detection
- Column checks
- Shape and datatype review

### Target Transformation

- Converted sentiment labels for model compatibility

### Feature Preparation
- Structured numerical feature matrix creation

## Exploratory Data Analysis Performed

### Statistical Analysis
- Head / Tail inspection
- Summary statistics
- Correlation matrix

### Visualization
- Class distribution plots
- Heatmaps
- Boxplots
- Pairplots
- Histogram comparisons
- Feature importance charts

## Models Implemented
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting

## Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Comparative model performance table

## Skills Demonstrated
- SQL + Python Integration
- Cloud Database Analytics
- Sentiment Analysis
- NLP Feature Engineering
- Classification Modeling
- Dashboard Ready Reporting
- Business Intelligence
- Model Benchmarking

## Business Use Case
Help organizations analyze customer feedback, company reports, product reviews, employee surveys, or market sentiment stored in databases and automatically classify sentiment for faster decision-making.

## Key Insights
- Tree-based models performed strongly due to nonlinear feature relationships.
- Database-connected analytics pipelines reduce manual CSV dependency.
- High-dimensional sentiment data benefits from feature importance analysis.
- Multi-model comparison improves confidence in deployment choices.

## Future Improvements
- Real-time sentiment dashboard
- NLP with TF-IDF / Word2Vec / BERT
- Automated SQL refresh pipelines
- API deployment for live predictions
- Power BI integration

## Author
Biswaditya07 (Biswaditya Saha)  
MBA in Business Analytics
