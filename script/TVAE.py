# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using the TVAE model. This version includes the latest
metadata handling to prevent warnings and can resume from any iteration.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.utils import load_synthesizer
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

# --------------------------------------------------------------------------
# 2. FUNCTION DEFINITIONS
# --------------------------------------------------------------------------

# UPDATED: This function now saves/loads metadata to prevent warnings.
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
        # This is the corrected line
        metadata = Metadata.load_from_json(filepath=metadata_path)
    else:
        print("Detecting metadata from data...")
        metadata = Metadata()
        metadata.detect_table_from_dataframe(
            table_name='adult_data',
            data=adult_df
        )
        metadata.save_to_json(filepath=metadata_path)
        print(f"Metadata detected and saved to '{metadata_path}'.")

    print("Dataset loaded successfully.")
    return adult_df, metadata

# UPDATED: This function can now load a model from any iteration.
def load_or_train_synthesizer(training_data, metadata, model_path, iteration):
    """
    Loads a synthesizer from the specified path if it exists for the current
    iteration. Otherwise, it trains a new synthesizer and saves it.
    """
    if os.path.exists(model_path):
        print(f"Iteration {iteration}: Found existing model at '{model_path}'. Loading...")
        synthesizer = load_synthesizer(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Iteration {iteration}: No model found for this iteration. Training a new model...")
        synthesizer = TVAESynthesizer(metadata)
        synthesizer.fit(training_data)
        print(f"Training complete. Saving model to '{model_path}'...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print("Model saved successfully.")
    return synthesizer

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path):
    """Generates and saves diagnostic and quality reports."""
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
    """Generates and saves distribution comparison plots."""
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

# UPDATED: The main function now defines and passes the metadata path.
def main():
    """Main function to orchestrate the iterative synthetic data pipeline."""
    # --- Configuration ---
    TOTAL_ITERATIONS = 10
    METADATA_PATH = './../metadata.json' 
    BASE_MODEL_DIR = './../models/TVAE/'
    BASE_REPORT_DIR = './../reports/TVAE/'
    BASE_IMAGE_DIR = './../images/TVAE/'

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} STARTING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

        model_path = os.path.join(BASE_MODEL_DIR, str(i), 'synthesizer.pkl')
        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        image_dir = os.path.join(BASE_IMAGE_DIR, str(i))

        synthesizer = load_or_train_synthesizer(current_training_data, metadata, model_path, i)

        print(f"\nIteration {i}: Generating new synthetic sample...")
        synthetic_data = synthesizer.sample(num_rows=len(original_adult_df))
        
        evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path)
        generate_and_save_plots(original_adult_df, synthetic_data, metadata, image_dir)

        current_training_data = synthetic_data
        
        print(f"{'='*20} FINISHED ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

if __name__ == "__main__":
    main()