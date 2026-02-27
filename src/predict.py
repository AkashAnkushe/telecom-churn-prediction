import os
import joblib
import pandas as pd

from src.feature_engineering import create_features

# Final threshold decided during modeling
THRESHOLD = 0.40


def load_model():
    """
    Load trained churn model.
    """
    model_path = os.path.join("models", "churn_model.pkl")
    model = joblib.load(model_path)
    return model


def predict_churn(input_data: dict):
    """
    Predict churn for a single customer.
    input_data must be a dictionary of customer features.
    """

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    df = create_features(df)

    # Load trained model
    model = load_model()

    # Get probability
    prob = model.predict_proba(df)[:, 1][0]

    # Apply threshold
    prediction = 1 if prob >= THRESHOLD else 0

    return {
        "churn_probability": float(prob),
        "churn_prediction": int(prediction)
    }