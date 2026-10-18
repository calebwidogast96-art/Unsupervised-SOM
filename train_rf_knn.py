from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend
import warnings

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import pandas as pd
import sys
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    "data/networkTraffic.csv",
    na_values=["?"]
)

df = df.loc[df["attack_cat"].notna()].copy()

X = df.drop(columns=["attack_cat", "id"])
y = LabelEncoder().fit_transform(df["attack_cat"].astype(str))

# remove known redundant features
redundant = [
    ("sbytes", "sloss"),
    ("dbytes", "dloss"),
    ("is_ftp_login", "ct_ftp_cmd"),
]

drop_features = [
    second for first, second in redundant
    if first in X.columns and second in X.columns
]

X = X.drop(columns=drop_features)

categorical_features = ["proto", "state", "service"]
numerical_features = [
    c for c in X.columns
    if c not in categorical_features
]

# stratification preserves class proportions across the split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=1234,
    stratify=y
)

def signed_log(X): return np.sign(X) * np.log1p(np.abs(X))

# KNN preprocessing: impute, transform, and scale numeric features.
knn_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(signed_log)),
    ("scaler", RobustScaler()),
])

# RF preprocessing: imputation only for numeric features.
rf_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# shared categorical preprocessing.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )),
])

knn_preprocessor = ColumnTransformer([
    ("num", knn_numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

rf_preprocessor = ColumnTransformer([
    ("num", rf_numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

X_tune, _, y_tune, _ = train_test_split(
    X_train,
    y_train,
    train_size=0.2,
    random_state=1234,
    stratify=y_train
)

def train_knn():
    knn_model = Pipeline([
        ("preprocessor", knn_preprocessor),
        ("classifier", KNeighborsClassifier())
    ])

    param_grid = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
        "classifier__leaf_size": [20, 30, 40]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1234
    )

    grid_search = GridSearchCV(
        estimator=knn_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=4,
        pre_dispatch=4,
        verbose=2
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        grid_search.fit(X_tune, y_tune)

    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv("results/knn_results.csv", index=False)

    print("Best KNN parameters:", grid_search.best_params_)
    print("Best CV Macro F1:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, "models/best_knn_model.pkl")

def train_rf():
    rf_model = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=1234,
            n_jobs=1
        ))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [2, 5],
        "classifier__max_features": ["sqrt", "log2"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1234
    )

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=4,
        pre_dispatch=4,
        verbose=2,
        error_score="raise"
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        grid_search.fit(X_tune, y_tune)

    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv("results/rf_results.csv", index=False)

    print("Best RF parameters:", grid_search.best_params_)
    print("Best CV Macro F1:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, "models/best_rf_model.pkl")

def eval_models():
    try:
        y_pred_knn = LabelEncoder().fit_transform(joblib.load("models/best_knn_model.pkl").predict(X_test))
        y_pred_rf = LabelEncoder().fit_transform(joblib.load("models/best_rf_model.pkl").predict(X_test))
    except Exception:
        print("Error: best model files not found, please run model discovery steps!")
        return

    models = {"KNN": y_pred_knn, "RF": y_pred_rf}
    for name, predictions in models.items():
        print(f"\n--- {name} ---")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("Balanced accuracy:", balanced_accuracy_score(y_test, predictions))
        print("Macro F1:", f1_score(y_test, predictions, average="macro"))
        print(classification_report(y_test, predictions, zero_division=0))

        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions,
            xticks_rotation="vertical",
            cmap="Blues")

        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"models/{name}_conf_mat.png", dpi=300)
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: argument must be given [train|eval]")
        return

    if sys.argv[1] == "train":
        train_knn()
        train_rf()
    elif sys.argv[1] == "eval":
        eval_models()
    else:
        print("Usage: argument must be given [train|eval]")
        return

if __name__ == "__main__":
    main()