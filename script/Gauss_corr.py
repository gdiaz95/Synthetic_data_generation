# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using a custom Gaussian Copula method. It will loop 10 times,
with each iteration learning correlations from the output of the previous one,
while always evaluating against the original data.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
import sys

# This makes the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom functions from your correlator.py file
from src.correlator import transform_dataset_into_gaussian, generate_correlations, transform_dataset_from_gaussian

# Import only the evaluation functions from SDV
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

# --------------------------------------------------------------------------
# 2. FUNCTION DEFINITIONS
# --------------------------------------------------------------------------

def load_and_prepare_data(metadata_path):
    """
    Fetches the UCI Adult dataset and prepares the data. It loads metadata
    from a file if it exists, otherwise it detects and saves it.
    """
    print("Loading and preparing original dataset...")
    adult = fetch_ucirepo(id=2)
    adult_df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    adult_df['income'] = adult_df['income'].str.strip().str.replace('.', '', regex=False)
    
    if os.path.exists(metadata_path):
        print(f"Loading metadata from '{metadata_path}'...")
        metadata = Metadata.load_from_json(filepath=metadata_path)
    else:
        print("Detecting metadata from data...")
        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name='adult_data', data=adult_df)
        metadata.save_to_json(filepath=metadata_path)
        print(f"Metadata detected and saved to '{metadata_path}'.")

    print("Dataset loaded successfully.")
    return adult_df, metadata

# NEW FUNCTION: This encapsulates your custom generation logic.
def generate_synthetic_data_custom(training_data, original_data):
    """
    Generates synthetic data using the custom Gaussian Copula method.

    Args:
        training_data (pd.DataFrame): The data to learn the correlations from.
        original_data (pd.DataFrame): The original data to learn the marginals for the inverse transform.

    Returns:
        pd.DataFrame: The newly generated synthetic data.
    """
    print("-> Step 1/4: Transforming training data to Gaussian space to learn correlations...")
    data_train_z = transform_dataset_into_gaussian(training_data)

    print("-> Step 2/4: Generating new independent Gaussian samples...")
    n_samples = len(original_data)
    n_cols = len(original_data.columns)
    z_independent = pd.DataFrame(np.random.randn(n_samples, n_cols), columns=original_data.columns)

    print("-> Step 3/4: Applying learned correlations to new samples...")
    z_correlated = generate_correlations(data_train_z, z_independent)

    print("-> Step 4/4: Transforming correlated samples back to original data space...")
    # The inverse transform MUST use the original data to learn the correct marginal distributions
    synthetic_data = transform_dataset_from_gaussian(z_correlated, original_data)
    
    return synthetic_data

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path):
    # This function remains unchanged
    # (No changes needed for evaluation)
    print(f"--- Evaluating against ORIGINAL data and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    combined_report_data = {
        'diagnostic_report': {'properties': diagnostic_report.get_properties().to_dict('records')},
        'quality_report': {
            'overall_score': quality_report.get_score(),
            'properties': quality_report.get_properties().to_dict('records'),
            'details': {
                'Column Shapes': quality_report.get_details('Column Shapes').to_dict('records'),
                'Column Pair Trends': quality_report.get_details('Column Pair Trends').to_dict('records')
            }
        }
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(combined_report_data, f, indent=4)
    print(f"Combined report saved successfully.")

def generate_and_save_plots(original_real_data, synthetic_data, metadata, image_dir):
    # This function remains unchanged
    # (No changes needed for plotting)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Saving comparison plots to '{image_dir}' directory...")
    for column in original_real_data.columns:
        plt.figure(figsize=(12, 7))
        if pd.api.types.is_numeric_dtype(original_real_data[column]):
            sns.kdeplot(original_real_data[column], label='Original Real', fill=True, alpha=0.5, warn_singular=False)
            sns.kdeplot(synthetic_data[column], label='Synthetic', fill=True, alpha=0.5, warn_singular=False)
            plt.title(f'Numerical Distribution: {column}', fontsize=16)
        else:
            combined_df = pd.concat([
                original_real_data[[column]].assign(Source='Original Real'),
                synthetic_data[[column]].assign(Source='Synthetic')
            ]).reset_index(drop=True)
            sns.countplot(data=combined_df, x=column, hue='Source')
            plt.title(f'Categorical Distribution: {column}', fontsize=16)
            plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        file_path = os.path.join(image_dir, f'{column}.png')
        plt.savefig(file_path)
        plt.close()
    print(f"All plots for this iteration saved.")

# --------------------------------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------------------------------

def main():
    """Main function to orchestrate the iterative synthetic data pipeline."""
    # --- Configuration ---
    TOTAL_ITERATIONS = 10
    METADATA_PATH = './metadata.json'
    BASE_REPORT_DIR = './reports/Gaussian_correlated/'
    BASE_IMAGE_DIR = './images/Gaussian_correlated/'

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} STARTING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        image_dir = os.path.join(BASE_IMAGE_DIR, str(i))

        # CHANGED: Replaced the synthesizer logic with your custom function
        print(f"\nIteration {i}: Generating new synthetic sample using custom method...")
        synthetic_data = generate_synthetic_data_custom(
            training_data=current_training_data,
            original_data=original_adult_df
        )
        
        # Evaluation and plotting functions are called as before
        evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path)
        generate_and_save_plots(original_adult_df, synthetic_data, metadata, image_dir)

        # The newly generated data becomes the training data for the next loop
        current_training_data = synthetic_data
        
        print(f"{'='*20} FINISHED ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

if __name__ == "__main__":
    main()