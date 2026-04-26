import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('creditcard.csv')

# b) Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

# Scaling 'Amount' and 'Time'
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

# Drop the original unscaled columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

** Data Exploration **
a) Compute summary statistics for Amount
b) Plot distribution of Class and interpret imbalance
c) What business insight can be drawn from class imbalance?

# Business Insight: Fraud represents a microscopic fraction of total transactions. If a bank blocks too many transactions (false positives), it will frustrate customers and lose revenue. Conversely, missing actual fraud (false negatives) costs the bank directly. This imbalance means traditional accuracy isn't a good metric—banks need to balance detecting fraud without inconveniencing normal customers.

import matplotlib.pyplot as plt
import seaborn as sns

# Let's reload original just to get the raw Amount summary easily
temp_df = pd.read_csv('creditcard.csv')


# a) Summary statistics for Amount
print("Summary Statistics for Amount:")
print(temp_df['Amount'].describe()[['mean', 'std', 'min', 'max']])

# b) Plot distribution of Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution \n (0: Normal || 1: Fraud)')
plt.show()

# Interpret imbalance numerically
fraud_percent = (df['Class'].value_counts()[1] / len(df)) * 100
print(f"Percentage of Fraudulent Transactions: {fraud_percent:.3f}%")

** Modeling and Evaluation**

a) Compute correlation between features and Class
b) Identify top 3 important variables influencing Class
c) Why is dimensionality reduction useful in this dataset?

Dimensionality reduction (like PCA, which created V1-V28) hides the original identities of features (which banks need to do for data privacy and security). It also reduces computational cost, mitigates the curse of dimensionality, and helps remove noise and collinearity among features.

d) Create a confusion matrix
e) Calculate Accuracy, Precision, Recall, F1-score
f) Why is accuracy misleading in imbalanced dataset?

If a dataset contains 99.8% normal transactions and 0.2% fraud, a model that simply guesses "Normal" every single time will be 99.8% accurate. However, it catches 0 fraud cases. Accuracy completely masks the model's inability to detect the minority class.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# a) Compute correlation between features and Class
correlation_matrix = df.corr()

# b) Identify top 3 important variables influencing Class
# We take the absolute correlation values with 'Class' and sort them
correlations_with_class = correlation_matrix['Class'].abs().sort_values(ascending=False)
print("\nTop 3 variables most correlated with Class:")
print(correlations_with_class[1:4]) # skipping index 0 as it is 'Class' itself

# Prepare data for modeling
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# d) Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# e) Calculate Accuracy, Precision, Recall, F1-score
print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

"""**Plotting Correlation Heatmap of all Variables:**"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/creditcard.csv')

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)

plt.title("Correlation Heatmap of Credit Card Dataset")
plt.show()

"""### Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

# Train a Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree_model.predict(X_test)

# Create a confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Calculate and print evaluation metrics for Decision Tree
print("\nDecision Tree Evaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_dt):.4f}")

"""**Conclusion:**

The results show that both Logistic Regression and Decision Tree models achieved extremely high accuracy, with Logistic Regression slightly outperforming at 99.92% compared to 99.91% for the Decision Tree. At first glance, this indicates that both models are highly effective in classifying transactions. However, given the highly imbalanced nature of the dataset, where fraudulent transactions constitute a very small proportion, such high accuracy can be misleading.

The marginal difference between the two models suggests that Logistic Regression provides a more stable and generalized performance, while the Decision Tree, although powerful in capturing non-linear relationships, may be more prone to overfitting. In the context of fraud detection, relying solely on accuracy is insufficient; evaluation metrics such as recall and F1-score are more critical, as they reflect the model’s ability to correctly identify fraudulent transactions. Therefore, even though both models perform similarly in terms of accuracy, the model with better recall should be preferred for real-world deployment.

**Business Insights:**

From a business perspective, the results highlight the importance of focusing on fraud detection effectiveness rather than overall accuracy. Since fraudulent transactions are extremely rare, a model can achieve high accuracy simply by predicting most transactions as legitimate. However, this would fail to detect actual fraud cases, leading to significant financial losses.

The key insight is that organizations should prioritize minimizing false negatives, meaning fraudulent transactions that go undetected. A model with slightly lower accuracy but higher recall would be more valuable, as it ensures that more fraudulent activities are captured. At the same time, there is a need to balance this with false positives, as incorrectly flagging legitimate transactions can negatively impact customer experience and trust.

Additionally, the similar performance of both models suggests that simpler models like Logistic Regression can be effectively used in fraud detection systems, offering advantages such as interpretability, faster computation, and ease of deployment. This is particularly useful for real-time fraud detection where quick decision-making is essential.

Overall, the analysis emphasizes that fraud detection systems should be designed with a strong focus on risk management, where detecting rare but high-impact fraudulent transactions is more valuable than achieving high overall accuracy.
"""
