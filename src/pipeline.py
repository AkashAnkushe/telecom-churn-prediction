import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def build_pipeline(X: pd.DataFrame):

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=2000
    )

    pipeline = ImbPipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", logistic)
    ])

    param_grid = {
        "classifier__C": [0.01, 0.1, 0.5, 1, 5],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    return model