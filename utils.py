# utils.py
import pickle
import numpy as np
import pandas as pd

def load_model():
    """Load the trained XGBoost model and scaler"""
    with open("model/loan_default_model.pkl", "rb") as file:
        model, scaler = pickle.load(file)
    return model, scaler


def preprocess_input(input_dict, scaler):
    """Convert dictionary input into model-ready numpy array"""
    df = pd.DataFrame([input_dict])
    scaled = scaler.transform(df)
    return scaled


def predict_default(model, scaled_features):
    """Return prediction and probability"""
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]
    return pred, prob
