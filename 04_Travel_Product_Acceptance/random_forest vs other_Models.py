import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

df = pd.read_csv('/content/Travel.csv')
df.head()

df.isnull().sum()

df['Gender'].value_counts()

df['MaritalStatus'].value_counts()

## Handling the nan values
features_with_na = [features for features in df.columns if df[features].isnull().sum()>=1]
for feature in features_with_na:

  print(feature, np.round(df[feature].isnull().mean()*100, 5), '% missing values')

df[features_with_na].select_dtypes(exclude='object').describe()

from sklearn.model_selection import train_test_split
X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']

# Visualize the distribution of the target variable 'ProdTaken'
plt.figure(figsize=(6, 4))
sns.countplot(x='ProdTaken', data=df)
plt.title('Distribution of ProdTaken')
plt.show()

# Visualize the relationship between 'Age' and 'ProdTaken'
plt.figure(figsize=(8, 5))
sns.boxplot(x='ProdTaken', y='Age', data=df)
plt.title('Relationship between Age and ProdTaken')
plt.show()

# Visualize the relationship between 'MonthlyIncome' and 'ProdTaken'
plt.figure(figsize=(8, 5))
sns.boxplot(x='ProdTaken', y='MonthlyIncome', data=df)
plt.title('Relationship between Monthly Income and ProdTaken')
plt.show()

# Visualize the relationship between 'TypeofContact' and 'ProdTaken'
plt.figure(figsize=(10, 6))
sns.countplot(x='TypeofContact', hue='ProdTaken', data=df)
plt.title('Relationship between Type of Contact and ProdTaken')
plt.show()

# Visualize the relationship between 'Occupation' and 'ProdTaken'
plt.figure(figsize=(12, 6))
sns.countplot(x='Occupation', hue='ProdTaken', data=df)
plt.title('Relationship between Occupation and ProdTaken')
plt.xticks(rotation=45)
plt.show()

y.value_counts()

import pandas as pd
df = pd.read_csv('/content/Travel.csv')
from sklearn.model_selection import train_test_split
X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

#Create Column Transformer with 3 types of transformers
cat_features = X.select_dtypes(include='object').columns; num_features = X.select_dtypes(exclude='object').columns

# we use the column Transformer for transform the columns with combining multiple techniques
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop = 'first') # only (n-1) columns are required
preprocessor = ColumnTransformer(
    [
("OneHotEncoder", oh_transformer, cat_features),
('StandardScaler', numeric_transformer, num_features) ]
    )

print(preprocessor)

# Applying the Transformation
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

pd.DataFrame(X_train).head()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, \
confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
roc_curve

from sklearn.impute import SimpleImputer

# Defining all the models
models = {
"Logistic Regression":LogisticRegression(),
"Decision Tree":DecisionTreeClassifier(),
"Random Forest":RandomForestClassifier(),
"Gradient Boost":GradientBoostingClassifier()
}

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Training all the models
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_imputed, y_train)
    # Make Predictions
    y_train_pred = model.predict(X_train_imputed)
    y_test_pred = model.predict(X_test_imputed)
    # Training set Performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred)
    model_train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
    model_train_recall = recall_score(y_train, y_train_pred)
    model_train_f1 = f1_score(y_train, y_train_pred)
    model_train_precision = precision_score(y_train, y_train_pred)
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)
    # Testing set Performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    model_test_recall= recall_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test, y_test_pred)
    model_test_precision= precision_score(y_test, y_test_pred)
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred)
    print(list(models.keys())[i])
    # showing the performances
    print('Model performance for Training Set')
    print("-Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
    print(f"-Confusion Matrix:\n {model_test_confusion_matrix}")
    print('----------------------------------')
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
    print(f"-Confusion Matrix:\n {model_train_confusion_matrix}")
    print('='*35)
    print('\n')

rf_params = {
"max_depth":[5,8,15,None, 10],
"max_features":[5, 7, "auto", 8],
"min_samples_split" : [2,8, 15, 20],
"n_estimators":[100, 200, 500, 1000]
}

rf_params

!wget -O /content/Travel.csv https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Travel.csv
