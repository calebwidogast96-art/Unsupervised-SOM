from minisom import MiniSom

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
from collections import Counter

df = pd.read_csv(
    "data/networkTraffic.csv",
    na_values=["?"]
)

df = df.loc[df["attack_cat"].notna()].copy()

X = df.drop(columns=["attack_cat", "id"])
y = df["attack_cat"].astype(str)

categorical_features = ["proto", "state", "service"]
numerical_features = [
    c for c in X.columns
    if c not in categorical_features
]

# Remove known perfectly redundant features, if present.
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

# Stratification preserves class proportions across the split.
# Labels are used for splitting, not SOM training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(signed_log)),
    ("scaler", RobustScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

X_tune, _, y_tune, _ = train_test_split(
    X_train_processed,
    y_train,
    train_size=0.2,
    random_state=1234,
    stratify=y_train
)

def train_som():
    rng = np.random.default_rng(seed=1234)
    min_qe = 1_000_000
    min_te = 1_000_000
    best_som = None

    results = []
    for size in [10, 15, 20]:
        for lr in [0.3, 0.5]:
            for sigma in [1.0, 2.0, 3.0]:
                print(f"Size: {size} Learning Rate: {lr} Sigma: {sigma}")
                som = MiniSom(
                    x=size, y=size,
                    input_len=X_tune.shape[1],
                    sigma=sigma,
                    learning_rate=lr,
                    neighborhood_function="gaussian",
                    topology="rectangular",
                    random_seed=1234,)

                som.random_weights_init(X_tune)
                som.train_random(X_tune,
                    num_iteration=30_000,
                    verbose=False)

                # Evaluate on the same feature subset for comparison.
                qe = som.quantization_error(X_tune)

                # Topographic error is more expensive, so use a subset.
                te_idx = rng.choice(len(X_tune), size=min(2000, len(X_tune)), replace=False)
                te = som.topographic_error(X_tune[te_idx])

                results.append({
                    "size": size,
                    "learning_rate": lr,
                    "sigma": sigma,
                    "quantization_error": qe,
                    "topographic_error": te,})

                if qe < min_qe and te < min_te:
                    min_qe = qe
                    min_te = te
                    best_som = som

    results_df = pd.DataFrame(results).sort_values(["quantization_error", "topographic_error"])
    results_df.to_csv("results/som_results.csv", index=False)
    print(results_df.to_string(index=False))

    joblib.dump(best_som, "models/best_som_model.pkl")

    label_neurons()

def label_neurons():
    som = joblib.load("models/best_som_model.pkl")

    # Label the neurons using training labels
    neuron_labels = {}

    for i in range(som.get_weights().shape[0]):
        for j in range(som.get_weights().shape[1]):
            neuron_labels[(i, j)] = []

    for x, label in zip(X_train_processed, y_train):
        bmu = som.winner(x)
        neuron_labels[bmu].append(label)

    # Assign each neuron its majority class
    neuron_classes = {}

    for bmu, labels in neuron_labels.items():
        if labels:
            neuron_classes[bmu] = Counter(labels).most_common(1)[0][0]

    joblib.dump(neuron_classes, "models/neuron_classes.pkl")


def predict(x):
    som = joblib.load("models/best_som_model.pkl")
    neuron_classes = joblib.load("models/neuron_classes.pkl")
    bmu = som.winner(x)

    return neuron_classes.get(bmu)

def eval_models():
    try:
        y_pred_som = predict(X_test_processed)
    except Exception:
        print("Error: best model files not found, please run model discovery steps!")
        return

    models = {"SOM": y_pred_som}
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
        train_som()
    elif sys.argv[1] == "eval":
        eval_models()
    else:
        print("Usage: argument must be given [train|eval]")
        return

if __name__ == "__main__":
    main()