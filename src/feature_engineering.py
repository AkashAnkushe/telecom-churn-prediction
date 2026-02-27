import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for churn prediction.
    Returns transformed dataframe.
    """

    df = df.copy()

    # Tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 36, 60, 120],
        labels=["0-1yr", "1-3yr", "3-5yr", "5yr+"]
    )

    # Complaint intensity
    df["complaint_intensity"] = (
        df["num_complaints_3m"] * 2 +
        df["num_complaints_12m"]
    )

    # Payment risk score
    df["payment_risk_score"] = (
        df["late_payment_flag_3m"] * 2 +
        df["avg_payment_delay_days"]
    )

    # Engagement score
    df["engagement_score"] = (
        df["app_logins_30d"] +
        df["selfcare_transactions_30d"]
    )

    # Bill shock flag
    df["bill_shock_flag"] = (
        df["overage_charges"] > df["overage_charges"].median()
    ).astype(int)

    return df