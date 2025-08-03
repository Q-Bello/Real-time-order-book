import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load engineered features from CSV
df = pd.read_csv("/Users/quddusbello/PycharmProjects/real-time-order-book/data_stream/lob_features.csv")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Display label distribution to detect imbalance
print("Class distribution:\n", y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Evaluate Logistic Regression
log_preds = log_model.predict(X_test)
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("F1 Score:", f1_score(y_test, log_preds, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))

# Initialize and train XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost
xgb_preds = xgb_model.predict(X_test)
print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("F1 Score:", f1_score(y_test, xgb_preds, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
