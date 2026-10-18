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
from itertools import product
import time
import warnings
from sklearn.cluster import KMeans

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
    min_score = 1_000_000
    best_som = None

    sizes = [10, 15, 20, 25, 30]
    learning_rates = [0.1, 0.3, 0.5, 0.7]
    sigmas = [1.0, 2.0, 3.0, 5.0]
    neighborhood_functions = [
        "gaussian",
        "mexican_hat",
        "bubble",
    ]
    topologies = ["rectangular", "hexagonal"]

    combinations = product(
        sizes,
        learning_rates,
        sigmas,
        neighborhood_functions,
        topologies,
    )

    results = []
    for size, lr, sigma, neighborhood, topology in combinations:
        start = time.perf_counter()
        som = MiniSom(
            x=size, y=size,
            input_len=X_tune.shape[1],
            sigma=sigma,
            learning_rate=lr,
            neighborhood_function=neighborhood,
            topology=topology,
            random_seed=1234,)

        som.random_weights_init(X_tune)
        som.train_random(X_tune,
            num_iteration=30_000,
            verbose=False)

        # evaluate on the same feature subset for comparison.
        qe = som.quantization_error(X_tune)
        iter_time = time.perf_counter() - start

        print(f"Size: {size} Lr: {lr} Sigma: {sigma}",
            f"NF: {neighborhood} Topology: {topology}",
            f"QE: {qe:.4f} Time: {iter_time:.4f}\n")

        results.append({
            "iteration_time": iter_time,
            "size": size,
            "learning_rate": lr,
            "sigma": sigma,
            "neighborhood_function":neighborhood,
            "topology":topology,
            "quantization_error": qe,})

        if qe < min_score:
            min_score = qe
            best_som = som

    results_df = pd.DataFrame(results).sort_values(["quantization_error"])
    # results_df.to_csv("results/som_results.csv", index=False)
    print(results_df.to_string(index=False))

    best_som.random_weights_init(X_train_processed)
    best_som.train_random(X_train_processed,
        num_iteration=20_000, verbose=True)

    joblib.dump(best_som, "models/best_som_model.pkl")

    # cluster_codebook_vectors()
    # label_clusters()

    label_neurons()

def label_neurons():
    som = joblib.load("models/best_som_model.pkl")

    neuron_labels = {}

    for i in range(som.get_weights().shape[0]):
        for j in range(som.get_weights().shape[1]):
            neuron_labels[(i, j)] = []

    for x, label in zip(X_train_processed, y_train):
        bmu = som.winner(x)
        neuron_labels[bmu].append(label)

    # assign each neuron its majority class
    neuron_classes = {}

    for bmu, labels in neuron_labels.items():
        if labels:
            neuron_classes[bmu] = Counter(labels).most_common(1)[0][0]

    joblib.dump(neuron_classes, "models/neuron_classes.pkl")



def cluster_codebook_vectors():
    som = joblib.load("models/best_som_model.pkl")
    weights = som.get_weights()
    codebook_vectors = weights.reshape(-1, weights.shape[-1])

    kmeans = KMeans(
        n_clusters=10,
        random_state=1234,
        n_init=10,
    )

    cluster_ids = kmeans.fit_predict(codebook_vectors)
    # restore the SOM's two-dimensional grid layout
    # cluster_map = cluster_ids.reshape(weights.shape[:2])

    joblib.dump(kmeans, "models/codebook_kmeans.pkl")

def label_clusters():
    som = joblib.load("models/best_som_model.pkl")
    kmeans = joblib.load("models/codebook_kmeans.pkl")

    weights = som.get_weights()
    codebook_vectors = weights.reshape(-1, weights.shape[-1])
    cluster_ids = kmeans.predict(codebook_vectors)
    cluster_map = cluster_ids.reshape(weights.shape[:2])

    # collect training labels assigned to each cluster
    cluster_labels = {
        cluster: []
        for cluster in range(kmeans.n_clusters)
    }

    for x, label in zip(X_train_processed, y_train):
        bmu = som.winner(x)
        cluster = cluster_map[bmu]
        cluster_labels[cluster].append(label)

    # label each cluster using its majority class
    cluster_classes = {}
    for cluster, labels in cluster_labels.items():
        if labels:
            cluster_classes[cluster] = Counter(labels).most_common(1)[0][0]

    joblib.dump(cluster_classes, "models/cluster_classes.pkl")

def predict_cb(X):
    som = joblib.load("models/best_som_model.pkl")
    kmeans = joblib.load("models/codebook_kmeans.pkl")
    cluster_classes = joblib.load("models/cluster_classes.pkl")

    cluster_map = kmeans.labels_.reshape(
        som.get_weights().shape[:2]
    )

    predictions = []

    for x in X:
        bmu = som.winner(x)
        cluster = cluster_map[bmu]

        if cluster not in cluster_classes:
            raise ValueError(
                f"Cluster {cluster} has no assigned attack label."
            )

        predictions.append(cluster_classes[cluster])

    return predictions

def predict(X):
    som = joblib.load("models/best_som_model.pkl")
    neuron_classes = joblib.load("models/neuron_classes.pkl")

    weights = som.get_weights()
    labelled_neurons = list(neuron_classes.keys())

    if not labelled_neurons:
        raise ValueError("No labelled SOM neurons found.")

    y_pred = []
    for x in X:
        bmu = som.winner(x)

        if bmu in neuron_classes:
            y_pred.append(neuron_classes[bmu])
            continue

        # find the labelled neuron whose weight is closest to x.
        nearest = min(
            labelled_neurons,
            key=lambda n: np.linalg.norm(x - weights[n[0], n[1]])
        )

        y_pred.append(neuron_classes[nearest])

    return y_pred

def eval_models():
    try:
        y_pred_som = predict(X_test_processed)
    except Exception as e:
        import traceback
        traceback.print_exc()
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