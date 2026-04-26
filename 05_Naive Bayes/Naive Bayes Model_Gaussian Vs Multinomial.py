import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary_train=pd.read_csv('/content/SalaryData_Train.csv')
salary_test=pd.read_csv('/content/SalaryData_Test.csv')
salary_train.columns
salary_test.columns
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]

######### Naive Bayes ##############
## Naive bayes theorem is Different from Logistic Regression  Because The Naive bayes doesn't depends upon the historical data set.
## The Output is not Dependent Variable. And also in Naive bayes we don't calculate the Correltion of the variables.
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
Gmodel=GaussianNB()
train_pred_gau=Gmodel.fit(train_X,train_Y).predict(train_X)
test_pred_gau=Gmodel.fit(train_X,train_Y).predict(test_x)

train_acc_gau=np.mean(train_pred_gau==train_Y)
test_acc_gau=np.mean(test_pred_gau==test_y)
train_acc_gau#0.795
test_acc_gau#0.794

#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
train_pred_multi=Mmodel.fit(train_X,train_Y).predict(train_X)
test_pred_multi=Mmodel.fit(train_X,train_Y).predict(test_x)

train_acc_multi=np.mean(train_pred_multi==train_Y)
test_acc_multi=np.mean(test_pred_multi==test_y)
train_acc_multi#0.772
test_acc_multi#0.774

from sklearn.metrics import classification_report, confusion_matrix

print("Gaussian Naive Bayes Performance:")
print(classification_report(test_y, test_pred_gau))
print("Confusion Matrix:")
print(confusion_matrix(test_y, test_pred_gau))

print("\nMultinomial Naive Bayes Performance:")
print(classification_report(test_y, test_pred_multi))
print("Confusion Matrix:")
print(confusion_matrix(test_y, test_pred_multi))

"""### Performance Comparison: Gaussian vs. Multinomial Naive Bayes

Based on the classification reports and accuracy scores:

#### Gaussian Naive Bayes
- **Train Accuracy:** 0.795
- **Test Accuracy:** 0.795
- **Precision (Weighted Avg):** 0.78
- **Recall (Weighted Avg):** 0.79
- **F1-Score (Weighted Avg):** 0.77

#### Multinomial Naive Bayes
- **Train Accuracy:** 0.773
- **Test Accuracy:** 0.775
- **Precision (Weighted Avg):** 0.75
- **Recall (Weighted Avg):** 0.77
- **F1-Score (Weighted Avg):** 0.73

**Summary:**
Gaussian Naive Bayes generally shows slightly better performance across all metrics (accuracy, precision, recall, F1-score) compared to Multinomial Naive Bayes for this specific dataset and problem. This suggests that the continuous nature of the features, or their approximation, might be better handled by the Gaussian distribution assumption in this case.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix for Gaussian Naive Bayes
cm_gaussian = confusion_matrix(test_y, test_pred_gau)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_gaussian, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.metrics import confusion_matrix

# Calculate confusion matrix for Multinomial Naive Bayes
cm_multinomial = confusion_matrix(test_y, test_pred_multi)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_multinomial, annot=True, fmt='d', cmap='Greens',
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix - Multinomial Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('SalaryData_Test.csv')

# First, let's examine the structure of the data
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nTarget variable distribution:")
print(df['Salary'].value_counts())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('SalaryData_Test.csv')

# First, let's examine the structure of the data
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nTarget variable distribution:")
print(df['Salary'].value_counts())

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)
