# Unsupervised-SOM
Machine Learning 441 Assignment 4: Unsupervised learning using Self-Organising Maps

# Network Traffic Classification

A machine learning project for classifying network traffic into attack categories using **K-Nearest Neighbours (KNN)**, **Random Forest (RF)**, and a **Self-Organising Map (SOM)**. The project includes hyperparameter tuning, model evaluation, and visualisation of classification performance and SOM quantisation error.

## Models

- **KNN:** Grid search with stratified 5-fold cross-validation, using macro F1-score for model selection.
- **Random Forest:** Grid search with class balancing and stratified 5-fold cross-validation.
- **SOM:** Hyperparameter search based on quantisation error, with neurons labelled using the majority attack category of their assigned training samples.

## Dataset

The dataset is stored in `data/networkTraffic.csv`. The target variable is `attack_cat`, and the `id` column is excluded from training. Missing values are imputed, categorical features are one-hot encoded, and redundant features are removed. Numerical features for KNN and SOM undergo signed logarithmic transformation and robust scaling.

The data is split into **80% training and 20% testing**, with stratification and a fixed random seed.

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib minisom joblib
```

## Usage

Train the KNN and Random Forest models:

```bash
python train_rf_knn.py train
```

Evaluate the trained KNN and Random Forest models:

```bash
python train_rf_knn.py eval
```

Train the SOM:

```bash
python train_som.py train
```

Evaluate the trained SOM:

```bash
python train_som.py eval
```

Generate the SOM quantisation-error plot:

```bash
python create_plots.py
```

## Outputs

- `models/` — Saved models, SOM neuron labels, and confusion matrix plots.
- `results/` — Hyperparameter search results and evaluation data.
- `report/` — Project report and figures.

Ensure the `models/`, `results/`, and `report/figs/` directories exist before running the scripts.