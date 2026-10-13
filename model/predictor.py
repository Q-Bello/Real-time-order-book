# Loads the trained model and turns a live feature row into a prediction.
# Shared by predict_live.py and the dashboard so both use the training feature set.

import os

import joblib
import numpy as np

from microstructure.data_labeller import LABEL_NAMES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.pkl")


class Predictor:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]
        self.features = bundle["features"]

    def predict(self, row):
        """Returns (label_name, confidence) for one feature row."""
        x = np.array([[row[f] for f in self.features]])
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return LABEL_NAMES.get(int(self.classes[idx]), "N/A"), float(proba[idx])
