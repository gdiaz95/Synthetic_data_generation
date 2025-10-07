# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using the GaussianCopula model, logging evaluation results to W&B.
"""

# --------------------------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------------------------
import os
import json
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.utils import load_synthesizer
from sdv.single_table import GaussianCopulaSynthesizer # <-- CHANGED
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import wandb

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

def load_or_train_synthesizer(training_data, metadata, model_path):
    """Loads a synthesizer if it exists, otherwise fits a new one."""
    if os.path.exists(model_path):
        print(f"Found existing model at '{model_path}'. Loading...")
        synthesizer = load_synthesizer(model_path)
        print("Model loaded successfully.")
    else:
        print(f"No model found. Fitting a new model...")
        synthesizer = GaussianCopulaSynthesizer(metadata) # <-- CHANGED
        synthesizer.fit(training_data)
        print(f"Fitting complete. Saving model to '{model_path}'...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print("Model saved successfully.")
    return synthesizer

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path):
    """Generates reports, saves them, and returns the scores as a dictionary."""
    print(f"--- Evaluating and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    combined_report_data = {
        'diagnostic_report': {'properties': diagnostic_report.get_properties().to_dict('records')},
        'quality_report': {
            'overall_score': quality_report.get_score(),
            'properties': quality_report.get_properties().to_dict('records')
        }
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(combined_report_data, f, indent=4)
    print(f"Combined report saved successfully.")
    return combined_report_data

# --------------------------------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------------------------------

def main():
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'GaussianCopula' # <-- CHANGED
    TOTAL_ITERATIONS = 10
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    METADATA_PATH = os.path.join(PROJECT_ROOT, 'metadata.json')
    BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', MODEL_TYPE)
    BASE_REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports', MODEL_TYPE)
    FINAL_EVAL_STEP = 9999

    # --- Initial Setup ---
    original_adult_df, metadata = load_and_prepare_data(METADATA_PATH)
    current_training_data = original_adult_df.copy()

    # --- Iteration Loop ---
    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'='*20} PROCESSING ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")
        
        model_path = os.path.join(BASE_MODEL_DIR, str(i), 'synthesizer.pkl')
        report_path = os.path.join(BASE_REPORT_DIR, f'{i}.json')
        run_id_path = os.path.join(os.path.dirname(model_path), 'wandb_run_id.txt')

        # 1. Load or fit the model
        synthesizer = load_or_train_synthesizer(current_training_data, metadata, model_path)
        
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
        
        # NOTE: The loss logging block is intentionally removed for this model.
        
        # Generate Sample and Evaluation Scores
        print(f"\nIteration {i}: Generating synthetic sample and evaluating...")
        synthetic_data = synthesizer.sample(num_rows=len(original_adult_df))
        
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

        # 5. Finish the run
        run.finish()

        # Update training data for the next iteration
        current_training_data = synthetic_data
        
        print(f"{'='*20} FINISHED ITERATION {i}/{TOTAL_ITERATIONS} {'='*20}")

if __name__ == "__main__":
    main()