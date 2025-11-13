
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import pandas as pd

# Column names based on UCI dataset
columns = ['ID', 'Diagnosis',
           'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
           'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
           'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
           'compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
           'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
           'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']

# Load data
df = pd.read_csv('wdbc.data', header=None, names=columns)

# Check first 5 rows
print(df.head())

# Check info
print(df.info())

# Check for missing values
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Drop 'ID' column (not needed)
df = df.drop('ID', axis=1)

# 2. Encode target column: 'M' -> 1, 'B' -> 0
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

# 3. Split features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# 4. Split into train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature scaling (optional but helps models like Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create and train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create and train the Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# Plot the tree
plt.figure(figsize=(15,10))
plot_tree(tree, feature_names=X.columns, class_names=['Benign','Malignant'], filled=True)
plt.show()