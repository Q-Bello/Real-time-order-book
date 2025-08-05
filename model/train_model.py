# This script loads the dataset created by generate_dataset.py.
# It trains an XGBoost classifier and automatically configures it for binary or
# multi-class classification based on the labels found in the data.
# The trained model is saved to a .pkl file for later use.

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, 'data_stream', 'lob_features.csv')
model_path = os.path.join(script_dir, 'xgboost_model.pkl')

try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully from {csv_path}")
except FileNotFoundError:
    print(f"ERROR: {csv_path} not found. Please run `data_stream/generate_dataset.py` first.")
    exit(1)

X = df.drop("label", axis=1)
y = df["label"]

label_counts = y.value_counts()
print("\nClass distribution:\n", label_counts)
if len(label_counts) < 2:
    print("ERROR: Dataset contains only one class — cannot train model.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Training XGBoost Model ---")

num_classes = y.nunique()
print(f"Found {num_classes} unique classes for the model.")

if num_classes == 2:
    print("Configuring model for BINARY classification.")
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
else:
    print("Configuring model for MULTI-CLASS classification.")
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\nXGBoost Results:")

full_label_map = { 0: 'STABLE (0)', 1: 'DOWN (1)', 2: 'UP (2)' }
unique_labels_in_data = np.sort(y.unique())
target_names = [full_label_map[label] for label in unique_labels_in_data]
print(f"\nReport will be generated for these classes: {target_names}")

print(classification_report(y_test, xgb_preds, target_names=target_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, xgb_preds))

joblib.dump(xgb_model, model_path)
print(f"\nXGBoost model saved to {model_path}")