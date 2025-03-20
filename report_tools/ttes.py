import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

datasets = {
    "Adult": {
        "random_search": [0.430, 0.433, 0.448, 0.442, 0.424, 0.451, 0.400, 0.413, 0.415, 0.433],
        "genetic_algorithm": [0.830, 0.890, 0.940, 0.880, 0.860, 0.980, 0.880, 0.790, 0.960, 0.900]
    },
    "Compas": {
        "random_search": [0.062, 0.064, 0.072, 0.057, 0.062, 0.050, 0.067, 0.077, 0.068, 0.072],
        "genetic_algorithm": [0.840, 0.760, 0.760, 0.830, 0.720, 0.790, 0.780, 0.810, 0.770, 0.890]
    },
    "Law School": {
        "random_search": [0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
        "genetic_algorithm": [0.000, 0.000, 0.070, 0.000, 0.000, 0.000, 0.000, 0.000, 0.470, 0.000]
    },
    "Kdd": {
        "random_search": [0.042, 0.034, 0.032, 0.028, 0.027, 0.034, 0.037, 0.032, 0.025, 0.035],
        "genetic_algorithm": [0.450, 0.460, 0.510, 0.360, 0.450, 0.530, 0.480, 0.540, 0.310, 0.480]
    },
    "Dutch": {
        "random_search": [0.024, 0.019, 0.019, 0.019, 0.018, 0.025, 0.015, 0.025, 0.016, 0.020],
        "genetic_algorithm": [0.780, 0.670, 0.830, 0.840, 0.650, 0.720, 0.640, 0.710, 0.680, 0.760]
    },
    "Credit": {
        "random_search": [0.305, 0.297, 0.327, 0.321, 0.302, 0.320, 0.285, 0.306, 0.317, 0.284],
        "genetic_algorithm": [0.710, 0.810, 0.860, 0.710, 0.820, 0.790, 0.760, 0.830, 0.800, 0.871]
    },
    "German": {
        "random_search": [0.090, 0.076, 0.085, 0.074, 0.087, 0.090, 0.069, 0.094, 0.078, 0.073],
        "genetic_algorithm": [0.870, 0.810, 0.860, 0.850, 0.890, 0.820, 0.800, 0.810, 0.790, 0.890]
    },
}

for dataset, values in datasets.items():
    random_search = np.array(values["random_search"], dtype=np.float64)
    genetic_algorithm = np.array(values["genetic_algorithm"], dtype=np.float64)

    differences = genetic_algorithm - random_search

    shapiro_p = shapiro(differences).pvalue
    if shapiro_p > 0.05:
        # Normal
        ttest_res = ttest_rel(genetic_algorithm, random_search)
        ttest_p = ttest_res.pvalue
        test_used = "Paired t-test"
        p_value = ttest_p
    else:
        # Not normal
        wilcoxon_res = wilcoxon(genetic_algorithm, random_search)
        wilcoxon_p = wilcoxon_res.pvalue
        test_used = "Wilcoxon signed-rank test"
        p_value = wilcoxon_p

    print(f"{dataset}: Shapiro p = {shapiro_p:.4f}, {test_used} p = {p_value:.4e}")