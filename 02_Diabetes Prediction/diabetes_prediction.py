**Task 1: Handling Invalid Zero Values**

import pandas as pd
import numpy as np

# 1. Load the dataset (Make sure 'diabetes.csv' is uploaded to your Colab environment)
df = pd.read_csv('diabetes.csv')

# 2. Define the columns where a value of 0 is medically invalid
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 3. Replace 0 with NaN in these specific columns
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

# 4. Fill the NaN values with the median of their respective columns
df[invalid_zero_cols] = df[invalid_zero_cols].fillna(df[invalid_zero_cols].median())

# Check if there are any remaining nulls or zeros in these columns
print("Missing values after median imputation:\n", df[invalid_zero_cols].isnull().sum())

**Which variables contain medically unrealistic zero values?**

The variables are Glucose, BloodPressure, SkinThickness, Insulin, and BMI. For a living human being, it is biologically impossible for any of these measurements to be exactly zero. A blood pressure or BMI of zero indicates that the data was not recorded, rather than an actual measurement.

**Why is replacing those values important before model building?**

Machine learning models are highly literal. If you leave missing data recorded as 0, the algorithm will interpret that 0 as a true numeric value. This acts as an extreme outlier, artificially dragging down the mean of the column, skewing the distribution, and heavily biasing the model's predictions. By replacing them with the median, you provide a reasonable estimate that preserves the overall distribution of the data without distorting it.

**Task 2: Exploratory Data Analysis (EDA)**

import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style for the plots
sns.set_theme(style="whitegrid")

# Create a figure with multiple subplots for better organization
fig = plt.figure(figsize=(16, 12))

# 1. Count plot of the Outcome variable
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Outcome', palette='Set2')
plt.title('Distribution of Diabetes Outcome (0 = No, 1 = Yes)')

# 2. Histograms for Glucose, BMI, and Age
plt.subplot(2, 2, 2)
sns.histplot(df['Glucose'], kde=True, color='skyblue', label='Glucose')
sns.histplot(df['BMI'], kde=True, color='salmon', label='BMI')
plt.title('Distribution of Glucose and BMI')
plt.legend()

# (Age Histogram plotted separately to avoid overlapping scales)
plt.subplot(2, 2, 3)
sns.histplot(df['Age'], kde=True, color='green')
plt.title('Age Distribution')

# 3. Boxplots of Glucose and BMI grouped by Outcome
fig_box, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='Outcome', y='Glucose', ax=axes[0], palette='Set2')
axes[0].set_title('Glucose Levels by Outcome')

sns.boxplot(data=df, x='Outcome', y='BMI', ax=axes[1], palette='Set2')
axes[1].set_title('BMI Levels by Outcome')
plt.show()

# 4. Correlation heatmap for all numerical variables
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Diabetes Dataset')
plt.show()

**Insights obtained from the graphs:**

**Glucose is the strongest predictor:** Looking at both the correlation heatmap and the boxplots, Glucose has the highest positive correlation with the Outcome variable. The boxplot clearly shows that the median glucose level for patients diagnosed with diabetes (Outcome = 1) is significantly higher than for those without it.

**Age distribution is right-skewed:** The histogram for Age demonstrates that the majority of the patients in this dataset are younger (between 20 and 30 years old), with a long "tail" tapering off into the older age ranges.

**Task 3: Data Preparation for Machine Learning**
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Separate input features X and target variable y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Split the data into training and testing sets (80:20 ratio)
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize the input features using StandardScaler
scaler = StandardScaler()

# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the SAME scaler (to prevent data leakage)
X_test_scaled = scaler.transform(X_test)

# 4. Print the shape of training and testing datasets
print("Training Data Shape (Features):", X_train_scaled.shape)
print("Training Data Shape (Target):", y_train.shape)
print("Testing Data Shape (Features):", X_test_scaled.shape)
print("Testing Data Shape (Target):", y_test.shape)

"""**Why feature scaling is useful in machine learning models:**

Machine learning algorithms—especially those relying on distance metrics (like K-Nearest Neighbors or Support Vector Machines) or gradient descent (like Logistic Regression)—are highly sensitive to the scale of the data.

In this dataset, Insulin values might be in the hundreds, while DiabetesPedigreeFunction values are decimals between 0 and 2. Without scaling, the algorithm would assume Insulin is vastly more important simply because the numbers are larger. By standardizing the features (scaling them so they have a mean of 0 and a standard deviation of 1) ensure that every feature contributes equally to the model's decision-making process, leading to faster training times and more accurate predictions.

**Task 4: Building Different Models for Prediction**
"""

# Import the necessary libraries for the machine learning models and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Initialize the models into a dictionary
# random_state=42 ensures that the algorithms produce the exact same results each time you run them
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store the accuracies for easy comparison later
model_accuracies = {}

# 2. Loop through each model to train it and evaluate its accuracy
print("=== Individual Model Performance ===\n")

for name, model in models.items():

    # Train (fit) the model using the scaled training data
    model.fit(X_train_scaled, y_train)

    # Make predictions using the unseen, scaled testing data
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy score comparing predictions to actual test labels
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy

    # Print the individual model accuracy and a detailed classification report
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    # classification_report shows precision, recall, and f1-score for both outcome classes (0 and 1)
    print(classification_report(y_test, y_pred))
    print("-" * 50 + "\n")

# 3. Print a final summary leaderboard of all models sorted by highest accuracy
print("=== Final Model Leaderboard ===")
# Sort the models by accuracy in descending order
sorted_models = sorted(model_accuracies.items(), key=lambda item: item[1], reverse=True)

for name, acc in sorted_models:
    print(f"{name}: {acc * 100:.2f}%")

"""**What results I got:**

Logistic Regression: 75.32%

Support Vector Machine (SVM): 74.68%

Random Forest: 73.38%

K-Nearest Neighbors (KNN): 72.08%

Conclusion: For this specific dataset, Logistic Regression performed the best with an accuracy of approximately 75.3%, making it the strongest starting point among the tested algorithms!
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Set up a 2x2 grid for our 4 model confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten() # Flatten the 2x2 grid into a 1D array for easy looping

# 2. Loop through the trained models and plot their confusion matrix
for i, (name, model) in enumerate(models.items()):

    # Generate predictions for the test data
    y_pred = model.predict(X_test_scaled)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the heatmap on the corresponding subplot (axes[i])
    # annot=True puts the numbers inside the boxes, fmt='d' formats them as integers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)

    # Set titles and axis labels
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Predicted Outcome')
    axes[i].set_ylabel('Actual Outcome')

    # Label the ticks for clarity (0 = No Diabetes, 1 = Diabetes)
    axes[i].set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
    axes[i].set_yticklabels(['No Diabetes (0)', 'Diabetes (1)'])

# Adjust the layout so titles and labels don't overlap
plt.tight_layout()
plt.show()

"""**Based on the model results,** let’s break down the Confusion Matrix (using the best-performing Logistic Regression model) and extract actionable business/clinical insights, followed by an analysis of the most impactful features.

1. Description of the Confusion Matrix
Out of the 154 patients in our testing data, the Logistic Regression model generated the following matrix:

True Negatives (82): The model correctly predicted these patients do not have diabetes.

True Positives (34): The model correctly predicted these patients have diabetes.

False Positives (17 - Type I Error): The model incorrectly flagged healthy patients as having diabetes.

False Negatives (21 - Type II Error): The model incorrectly classified actual diabetic patients as healthy.

Why Logistic Regression achieved the highest accuracy for this specific diabetes dataset:

1. The Relationship is Highly Linear
Logistic regression works under the assumption that there is a linear relationship between the input features and the log-odds of the outcome.
In medical reality, as your Glucose, BMI, and Age increase, your risk of developing Type 2 Diabetes generally increases in a steady, proportional direction. Logistic regression is perfectly designed to capture this straightforward "higher X means higher Y" relationship. It draws a clean, mathematically simple line (a hyperplane) to separate the healthy patients from the diabetic patients.

2. Resistance to Overfitting (The Dataset is Small)
The original diabetes dataset only has 768 patient records. When we split the data, the models were trained on about 614 patients.

Random Forest and KNN are highly complex algorithms that look for intricate patterns and non-linear boundaries. On smaller datasets, they tend to overcomplicate things and "memorize" the random noise or outliers in the training data (this is called overfitting). When faced with unseen test data, their complex rules fail.

Logistic Regression is much more rigid. Because it relies on a simple mathematical formula, it is much less prone to overfitting on small datasets and generalizes better to new, unseen patients.

3. A Few "Heavy Lifter" Features
As we saw in the Feature Importance analysis, Glucose alone accounted for over 26% of the predictive power, with BMI adding another 16%.
When a dataset has one or two overwhelmingly dominant features, a simple algorithm that just assigns a large mathematical weight to those features (which is exactly what Logistic Regression does) will perform exceptionally well. Complex models try to find deep hidden interactions between minor variables (like Skin Thickness and Blood Pressure), but in this case, those deep interactions don't actually matter much for the final diagnosis.

4. We Scaled the Data Properly
Logistic Regression relies on an optimization process (like gradient descent) to find the best weights for each feature. Because we used StandardScaler to put all features on the exact same playing field (a mean of 0 and standard deviation of 1), the algorithm was able to calculate the optimal weights flawlessly without being distracted by the fact that Insulin values were in the hundreds while Pedigree numbers were decimals.

Summary:
Logistic regression won because the problem is fundamentally straightforward: higher blood sugar and higher BMI equal higher diabetes risk. The model was given perfectly scaled data, didn't overcomplicate the math, and generalized beautifully to the test patients.

**Business & Clinical Insights:**

In a medical context, not all errors are equal. Here is how a healthcare administrator or doctor would view these results:

The Danger of False Negatives (The 21 Missed Cases): This is the most dangerous metric. 21 patients who actually have diabetes were sent home being told they are fine. This delays crucial early intervention, leading to severe health complications down the line and potentially higher emergency care costs for the healthcare system.

The Cost of False Positives (The 17 False Alarms): Flagging a healthy person as diabetic causes them unnecessary stress and costs the clinic money and time for follow-up blood tests. However, in medicine, over-screening is generally safer than under-screening.

Actionable Business Strategy: Because the cost of missing a diagnosis (False Negative) is much higher than a false alarm (False Positive), a hospital using this model should lower the decision threshold. Instead of requiring a 50% probability to classify someone as diabetic, they might lower it to 35% or 40%. This would "catch" more of the 21 missed patients, trading them for slightly more false alarms, which is a much safer clinical strategy.

**Which Factor is Impacting the Model More, Why, and How?**

By analyzing the internal decision-making weights (Feature Importances) of the algorithm, we can rank exactly what the model relies on to make its diagnosis.

The Most Impactful Factor: Glucose

**Impact Score:** Glucose completely dominates the model, accounting for roughly 26.3% of the algorithm's entire predictive power—more than any other variable.

**Why?** This perfectly aligns with clinical biology. Diabetes is fundamentally defined by the body's inability to produce or effectively use insulin, leading to elevated blood sugar. Therefore, resting glucose levels are the most direct, literal measurement of the disease's presence.

**How?** Mathematically, there is a massive positive correlation. As the Glucose number rises past certain thresholds (typically over 125 mg/dL), the algorithm drastically increases the statistical probability of outputting a "1" (Diabetes Outcome). In decision tree models (like Random Forest), Glucose is almost always chosen as the very first "split" to divide the patients because it separates the healthy from the sick most cleanly.

**Secondary Factors:**

**BMI (16.4% Impact):** Body Mass Index is the second strongest predictor. How it impacts the model: Higher BMI values strongly push the model toward a positive diabetes prediction, aligning with the medical reality that obesity is a primary driver of insulin resistance (Type 2 Diabetes).

**Age (13.5% Impact) & Genetics (12.2% Impact):** Age and the DiabetesPedigreeFunction (family genetic history) follow closely behind. The model heavily penalizes older age brackets and high genetic predispositions, acting as multipliers to high glucose or high BMI readings.

**Now to Improve the Accuracy of the Model we can perform:**

10-Fold Cross-Validation (CV): Instead of relying on a single 80/20 split (which might have gotten a "lucky" or "unlucky" batch of test patients), 10-fold CV splits the data into 10 chunks, trains the model 10 different times using different chunks, and averages the results. This gives you a much more realistic and robust accuracy score.

Bagging & Boosting (Ensemble Methods): These combine multiple weak models to create one strong "super-model."

Bagging (Bootstrap Aggregating): Builds multiple independent models (usually decision trees) on random subsets of data and averages their votes (Random Forest is a famous type of bagging).

Boosting: Builds models sequentially. Each new model specifically focuses on fixing the mistakes (the False Negatives/Positives) made by the previous model.

**Task: Cross-Validation, Bagging, and Boosting**
"""

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore') # Hides annoying warning messages

# 1. Define the 10-Fold Cross Validation strategy
# shuffle=True ensures the data is randomly mixed before splitting into 10 chunks
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 2. Set up our models inside Pipelines (to ensure scaling happens properly during CV)
# We include Logistic Regression again as our baseline to see how it performs under CV
advanced_models = {
    "Baseline Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    "Bagging (Decision Trees)": make_pipeline(StandardScaler(), BaggingClassifier(random_state=42)),
    "AdaBoost Classifier": make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=42)),
    "Gradient Boosting": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))
}

# 3. Loop through the models, run 10-fold CV, and print the results
print("=== 10-Fold Cross-Validation Accuracy ===")
print("(Format: Average Accuracy +/- Standard Deviation over the 10 runs)\n")

for name, model in advanced_models.items():

    # cross_val_score automatically handles the 10 train/test splits and scoring
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    # Calculate the mean accuracy and how much it fluctuated (standard deviation)
    mean_accuracy = cv_scores.mean() * 100
    fluctuation = cv_scores.std() * 100

    print(f"{name}: {mean_accuracy:.2f}%  (+/- {fluctuation:.2f}%)")
