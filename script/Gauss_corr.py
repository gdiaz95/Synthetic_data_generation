# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using a custom Gaussian Copula method, logging results to W&B.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import json
import pandas as pd
import numpy as np
import sys
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import wandb  # <-- ADDED

# This makes the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom functions from your correlator.py file
from src.correlator import transform_dataset_into_gaussian, generate_correlations, transform_dataset_from_gaussian

# --------------------------------------------------------------------------
# 2. FUNCTION DEFINITIONS
# --------------------------------------------------------------------------

def load_and_prepare_data(metadata_path):
    """Loads the Adult dataset and prepares the data."""
    print("Loading and preparing original dataset...")
    adult = fetch_ucirepo(id=2)
    adult_df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    adult_df['income'] = adult_df['income'].str.strip().str.replace('.', '', regex=False)
    
    if os.path.exists(metadata_path):
        metadata = Metadata.load_from_json(filepath=metadata_path)
    else:
        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name='adult_data', data=adult_df)
        metadata.save_to_json(filepath=metadata_path)
    print("Dataset loaded successfully.")
    return adult_df, metadata

def generate_synthetic_data_custom(training_data, original_data):
    """Generates synthetic data using the custom Gaussian Copula method."""
    print("-> Step 1/4: Transforming training data to Gaussian space...")
    data_train_z = transform_dataset_into_gaussian(training_data)

    print("-> Step 2/4: Generating new independent Gaussian samples...")
    n_samples = len(original_data)
    n_cols = len(original_data.columns)
    z_independent = pd.DataFrame(np.random.randn(n_samples, n_cols), columns=original_data.columns)

    print("-> Step 3/4: Applying learned correlations to new samples...")
    z_correlated = generate_correlations(data_train_z, z_independent)

    print("-> Step 4/4: Transforming correlated samples back to original data space...")
    synthetic_data = transform_dataset_from_gaussian(z_correlated, original_data)
    
    return synthetic_data

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path):
    """Generates reports, saves them, and returns the scores as a dictionary."""
    print(f"--- Evaluating against ORIGINAL data and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    combined_report_data = {
        'diagnostic_report': {'properties': diagnostic_report.get_properties().to_dict('records')},
        'quality_report': {
            'overall_score': quality_report.get_score(),
            'properties': quality_report.get_properties().to_dict('records')
        }
    }
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(combined_report_data, f, indent=4)
    print(f"Combined report saved successfully.")
    return combined_report_data # <-- MODIFIED: Returns data for logging

# --------------------------------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------------------------------

def main():
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'Gaussian_correlated' # <-- ADDED
    TOTAL_ITERATIONS = 10
    # Using the robust path definition
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, 'metadata.json')
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999 # For overwriting scores

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} PROCESSING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

        report_path = os.path.join(BASE_REPORT_DIR, str(i), f'{i}.json')
        # Since there's no model dir, we'll store the run ID with the report
        run_id_path = os.path.join(os.path.dirname(report_path), 'wandb_run_id.txt')

        # 1. W&B Initialization
        run_id = None
        if os.path.exists(run_id_path):
            with open(run_id_path, 'r') as f:
                run_id = f.read().strip()

        run = wandb.init(
            project="synthetic-data-generation",
            group=MODEL_TYPE,
            name=f"{MODEL_TYPE}_iteration-{i}",
            config={"model_type": MODEL_TYPE, "iteration": i},
            id=run_id,
            resume="allow"
        )

        # 2. Save the new ID if it was created.
        if not run.resumed:
            # We need to make sure the directory exists before writing the file
            os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
            with open(run_id_path, 'w') as f:
                f.write(run.id)

        # 3. Generate and Log Data
        print(f"\nIteration {i}: Generating new synthetic sample using custom method...")
        synthetic_data = generate_synthetic_data_custom(
            training_data=current_training_data,
            original_data=current_training_data
        )
        
        report_data = evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path)
        
        diag_props = {prop['Property']: prop['Score'] for prop in report_data['diagnostic_report']['properties']}
        qual_props = {prop['Property']: prop['Score'] for prop in report_data['quality_report']['properties']}
        
        wandb.log({
            'Data Validity': diag_props.get('Data Validity'),
            'Data Structure': diag_props.get('Data Structure'),
            'Overall Quality Score': report_data['quality_report'].get('overall_score'),
            'Column Shapes Score': qual_props.get('Column Shapes'),
            'Column Pair Trends Score': qual_props.get('Column Pair Trends')
        }, step=FINAL_EVAL_STEP)
        
        print(f"Evaluation scores logged/overwritten to W&B.")

        # 4. Finish the run
        run.finish()

        # Update training data for the next iteration
        current_training_data = synthetic_data
        
        print(f"{'='*20} FINISHED ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

if __name__ == "__main__":
    main()