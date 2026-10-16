from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(
    "networkTraffic.csv",
    na_values=["?"]
)

# Target and descriptive features
X = df.drop(columns=["attack_cat", "id"])
y = df["attack_cat"]

# Identify categorical and numerical columns
categorical_features = ["proto", "state", "service"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the appropriate preprocessing to each feature type
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

tree_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

# Stratification preserves class proportions across the splits
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def train_knn():
    knn_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])

    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    return y_pred_knn

def train_rf():
    rf_model = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    return y_pred_rf

def eval_models():
    y_pred_knn = train_knn()
    y_pred_rf = train_rf()

    models = {"KNN": y_pred_knn, "Decision Tree": y_pred_rf}

    for name, predictions in models.items():
        print(f"\n--- {name} ---")
        print("Accuracy:",
            accuracy_score(y_test, predictions))
        print("Balanced accuracy:",
            balanced_accuracy_score(y_test, predictions))
        print("Macro F1:",
            f1_score(y_test, predictions, average="macro"))
        print(classification_report(
            y_test, predictions, zero_division=0))

        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions,
            xticks_rotation="vertical",
            cmap="Blues")

        plt.title(name)
        plt.tight_layout()
        plt.show()