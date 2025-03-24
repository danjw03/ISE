import pandas as pd
import numpy as np
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
from config import DATASET_CONFIG
import main


def generate_individual(X_test, sensitive_columns, non_sensitive_columns):
    return main.generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)


def fitness_individual(model, individual):
    sample_a, sample_b = individual
    return main.compute_difference(model, sample_a, sample_b)


def tournament_selection(population, fitnesses, tournament_size=3):
    candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def crossover_individuals(ind1, ind2, sensitive_columns, non_sensitive_columns, X_test):
    sample_a1, _ = ind1
    sample_a2, _ = ind2
    child_a = sample_a1.copy()

    for col in non_sensitive_columns:
        if random.random() < 0.5:
            child_a[col] = sample_a2[col]

    # randomly choose a value from the column so that its realistic data
    child_b = child_a.copy()
    for col in sensitive_columns:
        child_b[col] = np.random.choice(X_test[col].unique())
    return (child_a, child_b)


def mutate_individual(
    individual, sensitive_columns, non_sensitive_columns, X_test, mutation_prob=0.1
):
    sample_a, sample_b = individual
    child_a = sample_a.copy()
    child_b = sample_b.copy()

    for col in non_sensitive_columns:
        if random.random() < mutation_prob:
            min_val = X_test[col].min()
            max_val = X_test[col].max()

            perturbation = np.random.uniform(
                -0.1 * (max_val - min_val), 0.1 * (max_val - min_val)
            )
            unique_values = np.sort(X_test[col].unique())

            child_a[col] = unique_values[
                np.abs(
                    unique_values
                    - (np.clip(child_a[col] + perturbation, min_val, max_val))
                ).argmin()
            ]
            child_b[col] = unique_values[
                np.abs(
                    unique_values
                    - (np.clip(child_b[col] + perturbation, min_val, max_val))
                ).argmin()
            ]

    for col in sensitive_columns:
        if random.random() < mutation_prob:
            child_b[col] = np.random.choice(X_test[col].unique())
    return (child_a, child_b)


def genetic_algorithm(
    model,
    X_test,
    sensitive_columns,
    non_sensitive_columns,
    pop_size=100,
    generations=50,
    tournament_size=3,
    crossover_prob=0.8,
    mutation_prob=0.1,
    threshold=0.05,
):
    population = [
        generate_individual(X_test, sensitive_columns, non_sensitive_columns)
        for _ in range(pop_size)
    ]
    best_individual = None
    best_fitness = -1

    for gen in range(generations):
        fitnesses = [fitness_individual(model, ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = ind
        print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")

        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            # Crossover
            if random.random() < crossover_prob:
                child = crossover_individuals(
                    parent1, parent2, sensitive_columns, non_sensitive_columns, X_test
                )
            else:
                child = parent1  # clone if no crossover

            # Mutation
            child = mutate_individual(
                child, sensitive_columns, non_sensitive_columns, X_test, mutation_prob
            )
            new_population.append(child)
        population = new_population

    ga_discriminatory = 0
    for ind in population:
        fit = fitness_individual(model, ind)
        if fit > threshold:
            ga_discriminatory += 1
    ga_idi_ratio = ga_discriminatory / pop_size
    return best_individual, best_fitness, ga_idi_ratio


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

    _, X_test, _, _ = main.load_and_preprocess_data(file_path, target_index)

    sensitive_columns, non_sensitive_columns = main.get_sensitive_non_sensitive_columns(
        X_test, sensitive_indices, target_index
    )
    print(f"Sensitive columns: {sensitive_columns}")
    print(f"Non-sensitive columns: {non_sensitive_columns}")

    print(f"Dataset: {dataset_name} | Available Size: {available_size}")

    model = keras.models.load_model(model_path)

    print("Running Genetic Algorithm for Fairness Testing...")
    best_ind, best_fit, ga_idi_ratio = genetic_algorithm(
        model,
        X_test,
        sensitive_columns,
        non_sensitive_columns,
        pop_size=200,
        generations=100,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.3,
        threshold=0.05,
    )

    print("\nBest Test Pair from GA:")
    sample_a, sample_b = best_ind
    print("Sample A:", sample_a.to_dict())
    print("Sample B:", sample_b.to_dict())
    print("Best Fitness (|p(a)-p(b)|):", best_fit)
    print("GA-based IDI Ratio:", round(ga_idi_ratio, 4))


if __name__ == "__main__":
    run_fairness_testing("CRIME")
