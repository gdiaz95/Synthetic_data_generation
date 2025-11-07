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
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import time
import wandb  
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 

# This makes the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom functions from your correlator.py file
from src.correlator import transform_dataset_into_gaussian, generate_correlations, transform_dataset_from_gaussian
from src.metrics import get_metrics, run_tstr_evaluation
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

def generate_synthetic_data_custom(original_data,n_samples):
    """Generates synthetic data using the custom Gaussian Copula method."""
    print("-> Step 1/4: Transforming training data to Gaussian space...")
    data_train_z = transform_dataset_into_gaussian(original_data)

    print("-> Step 2/4: Generating new independent Gaussian samples...")
    n_cols = len(original_data.columns)
    z_independent = pd.DataFrame(np.random.randn(n_samples, n_cols), columns=original_data.columns)

    print("-> Step 3/4: Applying learned correlations to new samples...")
    z_correlated = generate_correlations(data_train_z, z_independent)

    print("-> Step 4/4: Transforming correlated samples back to original data space...")
    synthetic_data = transform_dataset_from_gaussian(z_correlated, original_data)
    
    return synthetic_data

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path, metrics_qa,training_time,evaluation_time,tstr_results):
    """Generates reports, saves them, and returns the scores as a dictionary."""
    print(f"--- Evaluating against ORIGINAL data and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    combined_report_data = {
        'diagnostic_report': {'properties': diagnostic_report.get_properties().to_dict('records')},
        'quality_report': {
            'overall_score': quality_report.get_score(),
            'properties': quality_report.get_properties().to_dict('records')
        },
        'metrics_qa': { "overall_accuracy": metrics_qa.accuracy.overall,
            "univariate_accuracy": metrics_qa.accuracy.univariate,
            "bivariate_accuracy": metrics_qa.accuracy.bivariate,
            "discriminator_auc": metrics_qa.similarity.discriminator_auc_training_synthetic,
            "identical_matches": metrics_qa.distances.ims_training,
            "dcr_training": metrics_qa.distances.dcr_training,
            "dcr_holdout": metrics_qa.distances.dcr_holdout,
            "dcr_share": metrics_qa.distances.dcr_share
            
        },
        "times":{"training_time": training_time,
                 "evaluation_time": evaluation_time},
        
        "tstr_evaluation": tstr_results
        
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
    MODEL_TYPE = 'Gaussian_correlated' 
    
    tstr_models = {
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    tstr_metrics = {
        "Accuracy": accuracy_score
    }
    target_column = 'income'
    
    TOTAL_ITERATIONS = 10
    # Using the robust path definition
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, 'metadata.json')
    BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', MODEL_TYPE)
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999 # For overwriting scores

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} PROCESSING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        model_path = os.path.join(BASE_MODEL_DIR, str(i))
        run_id_path = os.path.join(model_path, 'wandb_run_id.txt')

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
                
        # 3. Split data into training and holdout sets
        print(f"\nSplitting data into training and holdout sets for iteration {i}...")
        if i == 1:
            train_data, holdout_data = train_test_split(current_training_data, test_size=0.2, random_state=42)
        else:
            train_data = current_training_data
        print(f"Training data shape: {train_data.shape}, Holdout data shape: {holdout_data.shape}")

        # 4. Generate and Log Data
        start_time = time.time()
        print(f"\nIteration {i}: Generating new synthetic sample using custom method...")
        synthetic_data = generate_synthetic_data_custom(
            original_data=train_data,
            n_samples=len(train_data)
        )
        generating_time = time.time() - start_time
        
        print("\nRunning TSTR evaluation...")
        tstr_results = {}
        for m_name, model in tstr_models.items():
            for met_name, metric_func in tstr_metrics.items():
                
                score_real, score_synth, gap = run_tstr_evaluation(
                    real_data=original_adult_df,  # Use the *full* original data
                    synthetic_data=synthetic_data,
                    target_column=target_column,
                    model=model,
                    metric_func=metric_func
                )
                
                # Log the TSTR scores for wandb
                base_name = f"TSTR_{m_name}_{met_name}"

                # Log the TSTR scores with flat keys
                tstr_results[f"{base_name}_Real_Score"] = score_real
                tstr_results[f"{base_name}_Synthetic_Score"] = score_synth
                tstr_results[f"{base_name}_Performance_Drop_%"] = gap
        
        print("TSTR evaluation complete.")

        metrics_qa = get_metrics(train_data, synthetic_data, holdout_data)
        report_data = evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path, metrics_qa, 0.0 ,generating_time,tstr_results)
        
        diag_props = {prop['Property']: prop['Score'] for prop in report_data['diagnostic_report']['properties']}
        qual_props = {prop['Property']: prop['Score'] for prop in report_data['quality_report']['properties']}
        
        
        wandb.log({
            'Data Validity': diag_props.get('Data Validity'),
            'Data Structure': diag_props.get('Data Structure'),
            'Overall Quality Score': report_data['quality_report'].get('overall_score'),
            'Column Shapes Score': qual_props.get('Column Shapes'),
            'Column Pair Trends Score': qual_props.get('Column Pair Trends'),
            "overall_accuracy": metrics_qa.accuracy.overall,
            "univariate_accuracy": metrics_qa.accuracy.univariate,
            "bivariate_accuracy": metrics_qa.accuracy.bivariate,
            "discriminator_auc": metrics_qa.similarity.discriminator_auc_training_synthetic,
            "identical_matches": metrics_qa.distances.ims_training,
            "dcr_training": metrics_qa.distances.dcr_training,
            "dcr_holdout": metrics_qa.distances.dcr_holdout,
            "dcr_share": metrics_qa.distances.dcr_share,
            "training_time": 0.0,
            "evaluation_time": generating_time,
            **tstr_results
        }, step=FINAL_EVAL_STEP)
        
        print(f"Evaluation scores logged/overwritten to W&B.")

        # 4. Finish the run
        run.finish()

        # Update training data for the next iteration
        current_training_data = synthetic_data
        
        print(f"{'='*20} FINISHED ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

if __name__ == "__main__":
    main()