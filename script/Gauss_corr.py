# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using a custom Gaussian Copula method, logging results to W&B.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import sys
from sklearn.model_selection import train_test_split
import time
import wandb  
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
import argparse

# This makes the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom functions from your correlator.py file
from src.correlator import generate_synthetic_data
from src.metrics import get_metrics, run_tstr_evaluation, evaluate_and_save_reports
from src.loader import load_and_prepare_data
from src.image_plotter import plot_marginals

def main(args):
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'Gaussian_correlated' 
    
    tstr_models = {
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    tstr_metrics = {
        "Accuracy": accuracy_score
    }
    
    TOTAL_ITERATIONS = args.iterations
    DATASET_NAME = args.dataset
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, f'metadata/{DATASET_NAME}/metadata.json')
    BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, f'models/{DATASET_NAME}', MODEL_TYPE)
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, f'reports/{DATASET_NAME}', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999 # For overwriting scores

    # --- Initial Setup ---
    original_adult_df, metadata, target_column = load_and_prepare_data(DATASET_NAME, METADATA_PATH)
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
            group=f"{DATASET_NAME}/{MODEL_TYPE}", 
            name=f"{DATASET_NAME}_{MODEL_TYPE}_iter-{i}", 
            config={
                "dataset": DATASET_NAME, 
                "model_type": MODEL_TYPE, 
                "iteration": i
            },
            id=run_id,
            resume="allow"
        )

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
        synthetic_data = generate_synthetic_data(
            original_data=train_data,
            n_samples=len(train_data)
        )
        generating_time = time.time() - start_time
        
        print("Plotting marginals...")
        plot_marginals(DATASET_NAME, MODEL_TYPE, i, train_data, synthetic_data)
        
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
    parser = argparse.ArgumentParser(description="Run a synthetic data generation pipeline.")
    
    parser.add_argument('-d', '--dataset', 
                        type=str, 
                        default='adults', 
                        help='Name of the dataset to process (must match a case in src/loader.py)')
    
    parser.add_argument('-i', '--iterations', 
                        type=int, 
                        default=1, 
                        help='Number of iterations to run.')
    
    args = parser.parse_args()
    main(args)