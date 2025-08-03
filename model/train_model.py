import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load pre-engineered features from CSV
df = pd.read_csv("/Users/quddusbello/PycharmProjects/real-time-order-book/data_stream/lob_features.csv")

# Split into features (X) and target label (y)
X = df.drop("label", axis=1)
y = df["label"]

# Check class balance and exit early if model can't train
label_counts = y.value_counts()
print("Class distribution:\n", label_counts)

if len(label_counts) < 2:
    print("🚫 ERROR: Dataset contains only one class — cannot train model.")
    exit(1)

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression baseline model
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("F1 Score:", f1_score(y_test, log_preds, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))

# XGBoost model with class balancing
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("F1 Score:", f1_score(y_test, xgb_preds, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
