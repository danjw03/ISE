# config.py

DATASET_CONFIG = {
    "ADULT": {
        "model_path": "DNN/model_processed_adult.h5",
        "csv_path": "dataset/processed_adult.csv",
        "sensitive_indices": [1, 0, 3],  # Gender, race, age
        "target_index": 10,  # Class-label
        "available_size": 45222,
    },
    "COMPAS": {
        "model_path": "DNN/model_processed_compas.h5",
        "csv_path": "dataset/processed_compas.csv",
        "sensitive_indices": [0, 2],  # Sex, race
        "target_index": 10,  # Recidivism
        "available_size": 6172,
    },
    "LAW_SCHOOL": {
        "model_path": "DNN/model_processed_law_school.h5",
        "csv_path": "dataset/processed_law_school.csv",
        "sensitive_indices": [8, 10],  # Male, race
        "target_index": 11,  # Pass_bar
        "available_size": 20708,
    },
    "KDD": {
        "model_path": "DNN/model_processed_kdd.h5",
        "csv_path": "dataset/processed_kdd.csv",
        "sensitive_indices": [10, 12],  # Race, sex
        "target_index": 36,  # Income
        "available_size": 284556,
    },
    "DUTCH": {
        "model_path": "DNN/model_processed_dutch.h5",
        "csv_path": "dataset/processed_dutch.csv",
        "sensitive_indices": [0, 1],  # Sex, age
        "target_index": 11,  # Occupation
        "available_size": 60420,
    },
    "CREDIT": {
        "model_path": "DNN/model_processed_credit.h5",
        "csv_path": "dataset/processed_credit_with_numerical.csv",
        "sensitive_indices": [1, 2, 3],  # Sex, education, marriage
        "target_index": 23,  # Class
        "available_size": 30000,
    },
    "CRIME": {
        "model_path": "DNN/model_processed_communities_crime.h5",
        "csv_path": "dataset/processed_communities_crime.csv",
        "sensitive_indices": [16, 7],  # Black, FemalePctDiv
        "target_index": 16,  # Class
        "available_size": 2215,
    },
    "GERMAN": {
        "model_path": "DNN/model_processed_german.h5",
        "csv_path": "dataset/processed_german.csv",
        "sensitive_indices": [7, 10],  # PersonStatusSex, AgeInYears
        "target_index": 0,  # CREDITRATING
        "available_size": 1000,
    }
}
