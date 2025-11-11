# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using the CopulaGAN model, logging all results to W&B.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import json
import pandas as pd
import sys
import time
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# This makes the script runnable from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom functions
from src.metrics import get_metrics, run_tstr_evaluation, evaluate_and_save_reports
from src.loader import load_or_train_synthesizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------
# 2. FUNCTION DEFINITIONS
# --------------------------------------------------------------------------

def load_and_prepare_data(metadata_path):
    """Loads the Adult dataset and the project's metadata."""
    print("Loading original dataset and metadata...")
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


def main():
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'CopulaGAN' # <-- CORRECT MODEL
    TOTAL_ITERATIONS = 10
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, 'metadata.json')
    BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', MODEL_TYPE)
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999

    tstr_models = {
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    tstr_metrics = {
        "Accuracy": accuracy_score
    }
    target_column = 'income'

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    holdout_data = None

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} PROCESSING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")
        
        model_path = os.path.join(BASE_MODEL_DIR, str(i), 'synthesizer.pkl')
        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        run_id_path = os.path.join(os.path.dirname(model_path), 'wandb_run_id.txt')

        # ADDED: Split data into training and holdout sets
        print(f"\nSplitting data into training and holdout sets for iteration {i}...")
        if i == 1:
            train_data, holdout_data = train_test_split(current_training_data, test_size=0.2, random_state=42)
        else:
            train_data = current_training_data
        print(f"Training data shape: {train_data.shape}, Holdout data shape: {holdout_data.shape}")

        synthesizer_to_fit = CopulaGANSynthesizer(metadata, verbose=True)
        
        synthesizer, training_time = load_or_train_synthesizer(
            training_data=train_data,
            model_path=model_path,
            report_path=report_path,
            synthesizer_to_fit=synthesizer_to_fit
        )
        
        # 2. W&B Initialization
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

        # 3. Save the new ID if it was created.
        if not run.resumed:
            with open(run_id_path, 'w') as f:
                f.write(run.id)

        # 4. LOGGING BLOCK
        
        # 4a. Log Losses (Time Series) - This is the same as CTGAN
        try:
            loss_values_df = synthesizer._model.loss_values
            print("\n--- Logging detailed losses. ---")
            
            for index, row in loss_values_df.iterrows():
                wandb.log({
                    "Generator Loss": row['Generator Loss'],
                    "Discriminator Loss": row['Discriminator Loss']
                }, step=int(row['Epoch']))
            
            print("Training losses logged as time series for plotting to W&B.")
        except Exception as e:
            print(f"Could not log detailed losses. The model may have been loaded: {e}")

        # MODIFIED: This whole block is updated
        # 4b. Generate Sample and Evaluation Scores
        print(f"\nIteration {i}: Generating synthetic sample and evaluating...")
        
        start_eval_time = time.time()
        # Generate data with the same size as the training set
        synthetic_data = synthesizer.sample(num_rows=len(train_data))
        evaluation_time = time.time() - start_eval_time

        # --- ADDED: Run the TSTR (Train on Synthetic, Test on Real) evaluation ---
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
                
                # Create a simple, flat name for the report
                base_name = f"TSTR_{m_name}_{met_name}"

                # Log the TSTR scores with flat keys
                tstr_results[f"{base_name}_Real_Score"] = score_real
                tstr_results[f"{base_name}_Synthetic_Score"] = score_synth
                tstr_results[f"{base_name}_Performance_Drop_%"] = gap
        
        print("TSTR evaluation complete.")
        # --- END OF ADDED SECTION ---

        print("Getting QA metrics...")
        # Get metrics using the train/holdout split
        metrics_qa = get_metrics(train_data, synthetic_data, holdout_data)
        
        # --- MODIFIED: Pass tstr_results to the report function ---
        report_data = evaluate_and_save_reports(
            original_adult_df, synthetic_data, metadata, report_path, 
            metrics_qa, training_time, evaluation_time, tstr_results
        )
        
        diag_props = {prop['Property']: prop['Score'] for prop in report_data['diagnostic_report']['properties']}
        qual_props = {prop['Property']: prop['Score'] for prop in report_data['quality_report']['properties']}
        
        # Log evaluation scores at the FIXED, high step to force overwrite.
        wandb.log({
            'Data Validity': diag_props.get('Data Validity'),
            'Data Structure': diag_props.get('Data Structure'),
            'Overall Quality Score': report_data['quality_report'].get('overall_score'),
            'Column Shapes Score': qual_props.get('Column Shapes'),
            'Column Pair Trends Score': qual_props.get('Column Pair Trends'),
            
            # ADDED: Log QA metrics and times
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
    main()