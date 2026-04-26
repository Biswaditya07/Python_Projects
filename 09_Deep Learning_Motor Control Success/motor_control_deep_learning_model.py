# Deep Learning Model for Motor Control Success Prediction
This notebook builds a simple Deep Learning model to predict motor control Success (1) or Failure (0).

## Step 1: Import Libraries
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

"""## Step 2: Load Dataset"""

df = pd.read_csv('motor_control_performance_dataset_with_target.csv')
df.head()

"""## Step 3: Convert Categorical Column"""

df = pd.get_dummies(df, columns=['Platform/Method'], drop_first=True)
df.head()

"""## Step 4: Separate Features and Target"""

X = df.drop('Target (Success = 1, Failure = 0)', axis=1)
y = df['Target (Success = 1, Failure = 0)']

"""## Step 5: Train-Test Split"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""## Step 6: Feature Scaling"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""## Step 7: Build Neural Network"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

"""## Step 8: Compile Model"""

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

"""## Step 9: Train Model"""

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

"""## Step 10: Evaluate Model"""

loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', accuracy)

"""## Step 11: Confusion Matrix"""

y_pred = (model.predict(X_test) > 0.5).astype('int32')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""## Model Completed Successfully ✅"""

import tensorflow as tf

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann_history = ann_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

ann_loss, ann_acc = ann_model.evaluate(X_test, y_test)
print("ANN Accuracy:", ann_acc)

"""Think of ANN like:

A smart student who looks at all parameters together and learns patterns.

It learns combinations like:

High control error + high disturbance → Failure

Good PID + low overshoot → Success

🧠 2️⃣ CNN (Convolutional Neural Network)
✅ Why CNN Here?

Normally CNN is for images.

But CNN also works well for:

Pattern detection

Local feature relationships

Structured numerical signals

Your motor parameters behave like a signal pattern across features.

So we treat features as a 1D signal.

CNN expects 3D input:

(samples, features, channels)
"""

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                           input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)
print("CNN Accuracy:", cnn_acc)

"""CNN detects:

Small local patterns in features

Interactions between neighboring parameters

Example:

Speed + Torque + Flux combined pattern
instead of treating each independently.

🧠 3️⃣ RNN (Recurrent Neural Network)
✅ Why RNN Here?

RNN is used for:

Sequential data

Time-dependent data

Signal processing

Your dataset has:

Time (s)

Simulation step

Control response over time

So we can treat it as sequence data.



RNN expects:

(samples, time_steps, features)

We create small sequences (example: 5 time steps)
"""

def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(pd.DataFrame(X_train), y_train)

X_seq_test, y_seq_test = create_sequences(pd.DataFrame(X_test), y_test)

rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='relu',
                              input_shape=(X_seq.shape[1], X_seq.shape[2])),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rnn_history = rnn_model.fit(
    X_seq, y_seq,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

rnn_loss, rnn_acc = rnn_model.evaluate(X_seq_test, y_seq_test)
print("RNN Accuracy:", rnn_acc)

"""🎯 When to Use Which?
Use ANN if:

Data is tabular

No strong time dependency

Use CNN if:

Feature patterns matter

Local interactions important

Use RNN if:

Time behavior matters

Sequential control response important

💡 Business Perspective
Model	Business Benefit
ANN	Fast and simple failure prediction
CNN	Better pattern-based fault detection
RNN	Predict future instability before failure
"""
