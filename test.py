import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from tsfresh import extract_features
from tsfresh import select_features
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


df = pl.read_parquet("data/train_dataset_feature.parquet").to_pandas()


X = df.drop(["series_id", "timestamp", "onset", "wakeup", "state"], axis=1)
y = df["state"]
onset = df["onset"]
wakeup = df["wakeup"]
features_filtered = select_features(X, y)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


# Define parameters
params = {
    "objective": "binary",  # or 'multiclass' if you have more than two classes
    "metric": "binary_logloss",  # or 'multi_logloss' for multiclass
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "num_threads": 8,  # Adjust based on your machine's capability
    "histogram_pool_size": 1024
    * 12,  # Adjust based on your machine's capability
}

try:
    # Train the model
    lgbm_model = lgb.train(
        params, train_data, valid_sets=[test_data], num_boost_round=1000
    )
except Exception as e:
    print(e)
    print("Error in training. Check your parameters.")
