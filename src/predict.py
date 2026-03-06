import os
import joblib
import pandas as pd

from src.feature_engineering import create_features

THRESHOLD = 0.40


def load_models():

    logistic = joblib.load("models/logistic_model.pkl")
    rf = joblib.load("models/random_forest_model.pkl")
    xgb = joblib.load("models/xgb_model.pkl")

    return logistic, rf, xgb


def predict_churn(input_data: dict):

    df = pd.DataFrame([input_data])

    df = create_features(df)

    logistic, rf, xgb = load_models()

    log_prob = logistic.predict_proba(df)[:,1][0]
    rf_prob = rf.predict_proba(df)[:,1][0]
    xgb_prob = xgb.predict_proba(df)[:,1][0]

    ensemble_prob = (log_prob + rf_prob + xgb_prob) / 3

    prediction = 1 if ensemble_prob >= THRESHOLD else 0

    return {
        "logistic_prob": float(log_prob),
        "rf_prob": float(rf_prob),
        "xgb_prob": float(xgb_prob),
        "churn_probability": float(ensemble_prob),
        "churn_prediction": int(prediction)
    }