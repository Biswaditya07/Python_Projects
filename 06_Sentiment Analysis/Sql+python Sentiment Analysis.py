!pip install mypysql
import pandas as pd
from sqlalchemy import create_engine

# RDS credentials
host = "analytics.cdeum2gagkj5.ap-south-1.rds.amazonaws.com"
user = "admin"
password = "Sxuklab##2025"
db = "analytics"

# Load CSV
df_sentiment = pd.read_csv("/content/testsentimentclass.csv")


# Connect to RDS
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

print("df_sentiment")

!pip install pymysql

df_sentiment.head(50)

df_sentiment.tail()

df_sentiment.info()

df_sentiment.describe()

df_sentiment.isnull().sum()

df_sentiment['actual'].unique()

# Demonstrating basic pandas commands on the uploaded dataset

# 1. Display first 5 rows
head_rows = df_sentiment.head()

# 2. Display last 5 rows
tail_rows = df_sentiment.tail()

# 3. Shape of dataset (rows, columns)
shape_info = df_sentiment.shape

# 4. Column names
columns_list = df_sentiment.columns.tolist()

# 5. Data types of each column
dtypes_info = df_sentiment.dtypes

# 6. Summary statistics for numerical columns
describe_info = df_sentiment.describe()

# 7. Check for null values
null_info = df_sentiment.isnull().sum()

# 8. Unique values in target column 'actual'
unique_target = df_sentiment['actual'].unique()

# 9. Value counts of target column 'actual'
value_counts_target = df_sentiment['actual'].value_counts()

# 10. Correlation matrix (first 10 cols for readability)
correlation_matrix = df_sentiment.iloc[:, :10].corr()

import pandas as pd
import numpy as np

# Collect results in a dictionary to display
pandas_summary = {
    "Head (first 5 rows)": head_rows,
    "Tail (last 5 rows)": tail_rows,
    "Shape (rows, cols)": shape_info,
    "Column Names": columns_list[:10],  # show first 10 for brevity
    "Data Types": dtypes_info.head(10),
    "Summary Statistics": describe_info,
    "Null Values Count": null_info.head(10),
    "Unique Labels in Target": unique_target,
    "Value Counts in Target": value_counts_target,
    "Correlation Matrix (first 10 columns)": correlation_matrix
}

pandas_summary

!pip install pymysql

df_sentiment.head()

from matplotlib import pyplot as plt
df_Sentiment['below'].plot(kind='hist', bins=20, title='below')
plt.gca().spines[['top', 'right',]].set_visible(False)

df_sentiment.describe()

!pip install prettyplotlib

# Commented out IPython magic to ensure Python compatibility.
#essentials
import csv
import pandas as pd
import numpy as np
import string
import sklearn

#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#from sknn.mlp import Classifier, Layer



#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()

import brewer2mpl
from pandas.plotting import scatter_matrix, lag_plot

# %matplotlib inline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb


#preprocessing
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, normalize,FunctionTransformer, scale,StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn import metrics
from patsy import dmatrices
from sklearn.feature_selection import RFE, RFECV

#fix arrangement of columns
#scale and normalize if needed (but since this is cash, no need)
#encode categorical features.. if needed
#list all column names
cols = list(df_sentiment.columns.values) #usable
cols

#data exploration
df_sentiment.groupby('actual').mean()
#on average, Fraud = 1 happen more on CASH_OUT and TRANSFER transactions

df_sentiment.corr(method = 'pearson')

df_sentiment.groupby('reduction')

print(df_sentiment.groupby('reduction'))

display(df_sentiment.groupby('reduction'))

# Example: assuming your CSV has 'text' column and 'label' column
X = df_sentiment['below'] # Replace 'below' with your actual feature column name
y = df_sentiment['actual'] # Replace 'actual' with your actual target column name

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Gradient Boosting": GradientBoostingClassifier()
}

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Features (all except 'actual')
X = df_sentiment.drop(columns=['actual'])
# Target
y = df_sentiment['actual']

# Map target variable values to start from 0 for XGBoost
y = y.map({-1: 0, 0: 1, 1: 2})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
    except Exception as e:
        results[name] = {"Error": str(e)}

import pandas as pd
results_df = pd.DataFrame(results).T
results_df

# @title Precision

from matplotlib import pyplot as plt
results_df['Precision'].plot(kind='line', figsize=(18, 8), title='Precision')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title Accuracy

from matplotlib import pyplot as plt
results_df['Accuracy'].plot(kind='line', figsize=(18, 6), title='Accuracy')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title F1 Score

from matplotlib import pyplot as plt
results_df['F1 Score'].plot(kind='hist', bins=20, title='F1 Score')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Precision

from matplotlib import pyplot as plt
results_df['Precision'].plot(kind='hist', bins=20, title='Precision')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Accuracy

from matplotlib import pyplot as plt
results_df['Accuracy'].plot(kind='hist', bins=20, title='Accuracy')
plt.gca().spines[['top', 'right',]].set_visible(False)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas

# --- 1. Correlation Heatmap (first 20 features for readability) ---
plt.figure(figsize=(12, 8))
corr_subset = df_sentiment.iloc[:, :20].corr()
sns.heatmap(corr_subset, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap (First 20 Features)", fontsize=14)
plt.show()

# --- 2. Class Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='actual', data=df_sentiment, palette="viridis")
plt.title("Class Distribution (Target Variable)", fontsize=14)
plt.xlabel("Class Label (-1 = Negative, 0 = Neutral, 1 = Positive)")
plt.ylabel("Count")
plt.show()

# --- 3. Boxplot for Selected Features vs Target ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='actual', y='achievement', data=df_sentiment, palette="Set2")
plt.title("Distribution of 'achievement' Across Classes", fontsize=14)
plt.show()

# --- 4. Pairplot (only a few columns to avoid clutter) ---
sample_cols = ['achievement', 'approve', 'growth', 'increased', 'actual']
sns.pairplot(df_sentiment[sample_cols], hue='actual', palette="husl")
plt.suptitle("Pairplot of Selected Features vs Target", y=1.02, fontsize=14)
plt.show()

# --- 5. Feature Importance using RandomForest ---
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(df_sentiment.drop(columns=['actual']), df_sentiment['actual'])
importances = pd.Series(rf.feature_importances_, index=df_sentiment.drop(columns=['actual']).columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(20, 8))
sns.barplot(x=top_features.values, y=top_features.index, palette="magma")
plt.title("Top 10 Important Features (Random Forest)", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

"""Short Story of the Dataset

Imagine you’re analyzing corporate performance reports or textual disclosures where every word or phrase that signals performance (like “achievement”, “growth”, “approve”, “reduction”) has been numerically encoded into a structured dataset.

Each row (190 total) is a company or report instance.

Each column (113 features) counts how often a particular word/phrase appears.

The last column, actual, is the class label:

-1 → negative signal (bad performance / sentiment)

0 → neutral

1 → positive signal (good performance / sentiment)

So this dataset is basically a sentiment classification dataset built from text → turned into numbers → ready for machine learning.

Analytical Perspective

From a data science view:

Nature of Data

It’s high-dimensional (113 features but only 190 rows).

Target variable is imbalanced (127 positive, 42 neutral, 21 negative).

This makes model choice and evaluation critical.

Potential Analyses

Exploratory Data Analysis (EDA):

Which words correlate most with positive/negative outcomes?

Are some terms always associated with 1 (positive)?

Feature Selection:

Use correlation / mutual information to reduce irrelevant features.

Classification Modeling:

Algorithms like Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and SVM can predict sentiment.

Deep learning (LSTMs, Transformers) could be applied if raw text is available.

Key Insights (so far)

Decision Trees performed best (~74% accuracy), meaning nonlinear interactions between words matter.

Naïve Bayes had low accuracy but high precision → it’s picking strong word signals but missing broader context.

Imbalanced labels mean models might be biased toward the positive class (1).

✨ In summary:
Your dataset is a sentiment classification problem derived from text features. It captures how often performance-related terms appear in company reports and maps them to outcomes (negative, neutral, positive). Analytically, it’s a challenging high-dimensional, low-sample-size dataset where ensemble methods and careful feature selection will be most promising.
"""
