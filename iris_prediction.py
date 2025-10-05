import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === Step 1: Load Data ===
data = pd.read_csv('iris.csv')


# === Step 2: Split Data ===
train, test = train_test_split(
    data,
    test_size=0.4,
    stratify=data['species'],
    random_state=42
)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# === Step 3: Train Decision Tree ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# === Step 4: Evaluate Model ===
y_pred = clf.predict(X_test)


# Save model in the current directory
import joblib

model_path = "decision_tree_model.joblib"  # saved in current folder
joblib.dump(clf, model_path)


