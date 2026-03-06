import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import create_features
from pipeline import build_pipeline

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():

    # Load dataset
    data_path = os.path.join("data", "raw", "telecom_churn.csv")
    df = pd.read_csv(data_path)

    # Drop ID column
    df = df.drop(columns=["customer_id"])

    # Feature engineering
    df = create_features(df)

    # Separate features and target
    X = df.drop(columns=["is_churn"])
    y = df["is_churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression with improvements
    logistic_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        C=0.5
    )

    # Build pipeline
    model = build_pipeline(X)

    # Train
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Default threshold
    threshold = 0.40
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_probs)

    print("\nMODEL PERFORMANCE")
    print("---------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_score)

    
    # Random Forest 

    rf_model = Pipeline(steps=[
        ("preprocessing", model.best_estimator_.named_steps["preprocessing"]),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        ))
    ])

    rf_model.fit(X_train, y_train)

    rf_probs = rf_model.predict_proba(X_test)[:,1]

    rf_auc = roc_auc_score(y_test, rf_probs)

    print("\nRandom Forest ROC-AUC:", rf_auc)
    
    # XGBoost Model

    xgb_model = Pipeline(steps=[
        ("preprocessing", model.best_estimator_.named_steps["preprocessing"]),
        ("classifier", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    xgb_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)[:,1]

    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print("\nXGBoost ROC-AUC:", xgb_auc)
    
    # Ensemble prediction (average of models)
    ensemble_probs = (y_probs + rf_probs + xgb_probs) / 3

    ensemble_auc = roc_auc_score(y_test, ensemble_probs)

    print("\nEnsemble ROC-AUC:", ensemble_auc)
    
    # Save all models

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")

    print("\nAll models saved successfully!")


if __name__ == "__main__":
    main()
    
