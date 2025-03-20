import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATASET_CONFIG

def load_and_preprocess_data(file_path, target_index):
    df = pd.read_csv(file_path)

    target_column = df.columns[target_index]
    X = df.drop(columns=[target_column])  # Features (drop the target column)
    y = df[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def load_model(model_path):
    return load_model(model_path)

def get_sensitive_non_sensitive_columns(X_test, sensitive_indices, target_index):
    sensitive_columns = [X_test.columns[i] for i in sensitive_indices if i != target_index]
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]
    return sensitive_columns, non_sensitive_columns


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    sample_a = X_test.iloc[np.random.choice(len(X_test))]
    sample_b = sample_a.copy()

    # Apply perturbation on sensitive features (random flipping)
    for col in sensitive_columns:
        if col in X_test.columns:
            sample_b[col] = np.random.choice(X_test[col].unique())

    # Apply perturbation on non-sensitive features
    for col in non_sensitive_columns:
        if col in X_test.columns:
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)  
            sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)
    return sample_a, sample_b


def compute_difference(model, sample_a, sample_b):
    # Convert sample_a and sample_b to numpy arrays and reshape
    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    # Model predictions
    prediction_a = model.predict(sample_a.reshape(1, -1))  # Reshape to fit model input format
    prediction_b = model.predict(sample_b.reshape(1, -1))

    pred_a = prediction_a[0][0]  
    pred_b = prediction_b[0][0]

    return abs(pred_a - pred_b)