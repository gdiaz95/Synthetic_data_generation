# -*- coding: utf-8 -*-
"""
This script performs an iterative pipeline for generating and evaluating
synthetic data using the TVAE model. It logs training losses and evaluation
scores to Weights & Biases, allowing runs to be updated.
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
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import wandb  # <-- ADDED

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
        metadata.detect_table_from_dataframe(
            table_name='adult_data',
            data=adult_df
        )
        metadata.save_to_json(filepath=metadata_path)
        print(f"Metadata detected and saved to '{metadata_path}'.")

    print("Dataset loaded successfully.")
    return adult_df, metadata

def load_or_train_synthesizer(training_data, metadata, model_path):
    """
    Loads a synthesizer if it exists, otherwise trains a new one.
    """
    if os.path.exists(model_path):
        print(f"Found existing model at '{model_path}'. Loading...")
        synthesizer = load_synthesizer(model_path)
        print("Model loaded successfully.")
    else:
        print(f"No model found. Training a new model...")
        # Note: TVAESynthesizer doesn't have a verbose parameter like CTGAN
        synthesizer = TVAESynthesizer(metadata)
        synthesizer.fit(training_data)
        print(f"Training complete. Saving model to '{model_path}'...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print("Model saved successfully.")
    return synthesizer

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path):
    """Generates reports, saves them, and returns the scores as a dictionary."""
    print(f"--- Evaluating against ORIGINAL data and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    
    # This dictionary structure is slightly different from your original TVAE script
    # to match the CTGAN template exactly.
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
    return combined_report_data # <-- MODIFIED: Returns data for logging

# --------------------------------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------------------------------

def main():
    """Main function to orchestrate the iterative pipeline and W&B logging."""
    # --- Configuration ---
    MODEL_TYPE = 'TVAE'
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

        # 1. Load or train the model
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

        # --- THIS BLOCK IS NOW CORRECTLY INDENTED ---
        # 4a. Log Losses (Time Series)
        try:
            # 1. Get the raw loss data from the synthesizer
            loss_df = synthesizer._model.loss_values

            # 2. Calculate the average loss for each epoch
            epoch_losses = loss_df.groupby('Epoch')['Loss'].mean().reset_index()

            # 3. Loop through the clean, averaged data and log it to W&B
            for index, row in epoch_losses.iterrows():
                wandb.log({
                    "Average Epoch Loss": row['Loss']
                }, step=int(row['Epoch']))
            
            print("\nTraining losses successfully averaged and logged to W&B.")

        except Exception as e:
            print(f"\nCould not log detailed losses. The model may have been loaded or another error occurred: {e}")
        # --- END OF CORRECTED BLOCK ---

        # 4b. Generate Sample and Evaluation Scores (Overwrite previous single point)
        print(f"\nIteration {i}: Generating synthetic sample and evaluating...")
        synthetic_data = synthesizer.sample(num_rows=len(original_adult_df))
        
        report_data = evaluate_and_save_reports(original_adult_df, synthetic_data, metadata, report_path)
        
        diag_props = {prop['Property']: prop['Score'] for prop in report_data['diagnostic_report']['properties']}
        qual_props = {prop['Property']: prop['Score'] for prop in report_data['quality_report']['properties']}
        
        # Log evaluation scores at the FIXED, high step to force overwrite.
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