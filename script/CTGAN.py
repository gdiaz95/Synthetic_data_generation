# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using the CTGAN model. It logs training losses and evaluation
scores to Weights & Biases, allowing runs to be updated.
"""
import os
from sdv.single_table import CTGANSynthesizer
import sys
import time
from sklearn.model_selection import train_test_split
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import argparse
import random
import numpy as np
import torch
SEED = 42

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom metrics function
from src.metrics import get_metrics, run_tstr_evaluation, evaluate_and_save_reports
from src.loader import load_or_train_synthesizer, load_and_prepare_data
from dotenv import load_dotenv
from src.image_plotter import plot_marginals

# Load environment variables from .env file
load_dotenv()


def main(args):
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'CTGAN'
    TOTAL_ITERATIONS = args.iterations
    DATASET_NAME = args.dataset
    random.seed(SEED)
    np.random.seed(SEED)
    
    tstr_models = {
        "XGBoost Classifier": XGBClassifier(random_state=SEED, eval_metric='logloss')
    }
    tstr_metrics = {
        "Accuracy": accuracy_score
    }
 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, f'metadata/{DATASET_NAME}/metadata.json')
    BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, f'models/{DATASET_NAME}', MODEL_TYPE)
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, f'reports/{DATASET_NAME}', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999
    # --- Initial Setup ---
    original_adult_df, metadata, target_column = load_and_prepare_data(DATASET_NAME, METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} PROCESSING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")
        
        model_path = os.path.join(BASE_MODEL_DIR, str(i), 'synthesizer.pkl')
        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        run_id_path = os.path.join(os.path.dirname(model_path), 'wandb_run_id.txt')
        
        print(f"\nSplitting data into training and holdout sets for iteration {i}...")
        if i == 1:
            train_data, holdout_data = train_test_split(current_training_data, test_size=0.2, random_state=SEED+1)
        else:
            train_data = current_training_data
        print(f"Training data shape: {train_data.shape}, Holdout data shape: {holdout_data.shape}")

        # 1. Load or train the model (was_trained is ignored in the logging logic)
        synthesizer_to_fit = CTGANSynthesizer(metadata, verbose=True)
        torch.manual_seed(SEED + 1000 + i)

        # Pass it to the new generic function
        synthesizer, training_time = load_or_train_synthesizer(
            training_data=train_data,
            model_path=model_path,
            report_path=report_path,
            synthesizer_to_fit=synthesizer_to_fit
        )

        # 2. W&B Initialization (Maintain 10 runs)
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

        # 3. Save the new ID if it was created.
        if not run.resumed:
            with open(run_id_path, 'w') as f:
                f.write(run.id)

        # 4. LOGGING BLOCK: Log all metrics (Assuming loss_values always exists after load_or_train)
        
        # 4a. Log Losses (Time Series)
        try:
            loss_values_df = synthesizer._model.loss_values
            print("\n--- Logging detailed losses. ---")
            
            # Log the full history. Using 'step=Epoch' encourages W&B to overwrite 
            # points at the same step if the history is cleared.
            for index, row in loss_values_df.iterrows():
                wandb.log({
                    "Discriminator Loss": row['Discriminator Loss'],
                    "Generator Loss": row['Generator Loss']
                }, step=int(row['Epoch']))
            
            print("Training losses logged as time series for plotting to W&B.")
        except Exception as e:
            # This catch is necessary if 'loss_values' is not loaded with the model.
            print(f"Could not log detailed losses. The model may have been loaded: {e}")

        # 4b. Generate Sample and Evaluation Scores (Overwrite previous single point)
        print(f"\nIteration {i}: Generating synthetic sample and evaluating...")
        
        start_eval_time = time.time()
        # Generate data with the same size as the training set
        torch.manual_seed(SEED + 2000 + i)
        synthetic_data = synthesizer.sample(num_rows=len(train_data)) 
        evaluation_time = time.time() - start_eval_time
        
        print("Plotting marginals...")
        plot_marginals(DATASET_NAME, MODEL_TYPE, i, train_data, synthetic_data)
        
        print("\nRunning TSTR evaluation...")
        tstr_results = {}
        for m_name, model in tstr_models.items():
            for met_name, metric_func in tstr_metrics.items():
                
                score_real, score_synth, gap = run_tstr_evaluation(
                    real_data=original_adult_df,
                    synthetic_data=synthetic_data,
                    target_column=target_column,
                    model=model,
                    metric_func=metric_func,
                    random_state=SEED + 3000 + i
                )
                
                # Create a simple, flat name for the report
                base_name = f"TSTR_{m_name}_{met_name}"

                # Log the TSTR scores with flat keys
                tstr_results[f"{base_name}_Real_Score"] = score_real
                tstr_results[f"{base_name}_Synthetic_Score"] = score_synth
                tstr_results[f"{base_name}_Performance_Drop_%"] = gap
        
        print("TSTR evaluation complete.")

        print("Getting QA metrics...")
        # Get metrics using the train/holdout split
        metrics_qa = get_metrics(train_data, synthetic_data, holdout_data)

        report_data = evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path, metrics_qa, training_time, evaluation_time, tstr_results)

        diag_props = {prop['Property']: prop['Score'] for prop in report_data['diagnostic_report']['properties']}
        qual_props = {prop['Property']: prop['Score'] for prop in report_data['quality_report']['properties']}
        
        # Log evaluation scores at the FIXED, high step to force overwrite.
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
            "training_time": training_time,
            "evaluation_time": evaluation_time,
            **tstr_results
        }, step=FINAL_EVAL_STEP) 
        
        print(f"Evaluation scores logged/overwritten to W&B.")

        # 5. Finish the run
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