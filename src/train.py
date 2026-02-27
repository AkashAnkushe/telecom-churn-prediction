import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from feature_engineering import create_features
from pipeline import build_pipeline


def main():

    # Load dataset
    data_path = os.path.join("data", "raw", "telecom_churn.csv")
    df = pd.read_csv(data_path)

    # Drop ID column
    df = df.drop(columns=["customer_id"])

    # Apply feature engineering
    df = create_features(df)

    # Separate features and target
    X = df.drop(columns=["is_churn"])
    y = df["is_churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build pipeline
    model = build_pipeline(X)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    
    from sklearn.metrics import classification_report
    import numpy as np

    y_probs = model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_probs)

    print("ROC-AUC:", roc_score)

    print("\nThreshold Analysis:\n")

    for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred = (y_probs >= t).astype(int)
        print(f"\nThreshold: {t}")
        print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "churn_model.pkl"))

    print("Model saved successfully!")


if __name__ == "__main__":
    main()