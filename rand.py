import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATASET_CONFIG
import main

def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05, discrimination_pairs=None):
    if discrimination_pairs is None:
        discrimination_pairs = []

    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    prediction_a = model.predict(sample_a.reshape(1, -1))
    prediction_b = model.predict(sample_b.reshape(1, -1))

    pred_a = prediction_a[0][0]
    pred_b = prediction_b[0][0]

    if abs(pred_a - pred_b) > threshold:
        discrimination_pairs.append((sample_a, sample_b))
        return 1  # Individual discriminatory instance
    else:
        return 0  # Not a discriminatory instance


def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    discrimination_count = 0

    for _ in range(num_samples):
        sample_a, sample_b = main.generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
        discrimination_count += evaluate_discrimination(model, sample_a, sample_b)

    total_generated = num_samples

    IDI_ratio = discrimination_count / total_generated
    return IDI_ratio

def run_fairness_testing(dataset_name):
    config = DATASET_CONFIG.get(dataset_name)
    if config is None:
        print(f"No configuration found for dataset {dataset_name}")
        return

    model_path = config["model_path"]
    file_path = config["csv_path"]
    sensitive_indices = config["sensitive_indices"]
    target_index = config["target_index"]
    available_size = config["available_size"]

    X_train, X_test, y_train, y_test = main.load_and_preprocess_data(file_path, target_index)
    sensitive_columns, non_sensitive_columns = main.get_sensitive_non_sensitive_columns(X_test, sensitive_indices, target_index)
    
    model = keras.models.load_model(model_path)
    
    idi_ratio = calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns)
    print(f"IDI Ratio: {idi_ratio}")
    
if __name__ == "__main__":
    run_fairness_testing("ADULT")