# ==========================================
# Loan Approval Prediction - Combined Models
# Models:
# KNN, Naive Bayes, Logistic Regression,
# Artificial Neural Network, Decision Tree
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# STEP 1: Load Dataset
# ==========================================
df = pd.read_csv("loan_approval.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================
# STEP 2: Missing Values Check
# ==========================================
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ==========================================
# STEP 3: Encode Categorical Columns
# ==========================================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# ==========================================
# STEP 4: Define X and y
# ==========================================
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# ==========================================
# STEP 5: Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 6: Feature Scaling
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Function for Evaluation
# ==========================================
results = {}

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results[name] = acc

    print(f"\n{name}")
    print("="*40)
    print("Accuracy:", round(acc*100,2), "%")
    print(classification_report(y_test, pred))

# ==========================================
# STEP 7: KNN Model
# ==========================================
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model("KNN", knn)

# ==========================================
# STEP 8: Naive Bayes
# ==========================================
nb = GaussianNB()
evaluate_model("Naive Bayes", nb)

# ==========================================
# STEP 9: Logistic Regression
# ==========================================
lr = LogisticRegression()
evaluate_model("Logistic Regression", lr)

# ==========================================
# STEP 10: Decision Tree
# ==========================================
dt = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree", dt)

# ==========================================
# STEP 11: Artificial Neural Network
# ==========================================
ann = Sequential()

ann.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
ann.add(Dense(units=8, activation='relu'))
ann.add(Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train,
        batch_size=10,
        epochs=50,
        verbose=0)

pred_ann = ann.predict(X_test)
pred_ann = (pred_ann > 0.5).astype(int)

acc_ann = accuracy_score(y_test, pred_ann)
results["ANN"] = acc_ann

print("\nArtificial Neural Network")
print("="*40)
print("Accuracy:", round(acc_ann*100,2), "%")
print(classification_report(y_test, pred_ann))

# ==========================================
# STEP 12: Final Comparison
# ==========================================
print("\nFinal Model Accuracy Comparison")
print("="*40)

for model, score in results.items():
    print(model, ":", round(score*100,2), "%")

# ==========================================
# STEP 13: Plot Comparison
# ==========================================
plt.figure(figsize=(10,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Loan Approval Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

# ==========================================
# Best Model
# ==========================================
best_model = max(results, key=results.get)

print("\nBest Performing Model:", best_model)
print("Accuracy:", round(results[best_model]*100,2), "%")
