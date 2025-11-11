import os
import json
import time
from sdv.utils import load_synthesizer

def load_or_train_synthesizer(training_data, model_path, report_path, synthesizer_to_fit):
    """
    Loads a synthesizer from model_path if it exists, otherwise fits the
    provided synthesizer_to_fit and saves it.
    
    Returns the (loaded or trained) synthesizer and the training time.
    """
    
    # --- BLOCK 1: Try to load existing model ---
    if os.path.exists(model_path):
        print(f"Found existing model at '{model_path}'. Loading...")
        synthesizer = load_synthesizer(model_path)
        print("Model loaded successfully.")
        
        # Logic to read training_time from the existing report
        training_time = 0.0  # Default if report/key doesn't exist
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                training_time = report_data['times']['training_time']
                print(f"Read existing training time from report: {training_time}s")
            except Exception as e:
                print(f"Warning: Could not read training time from report '{report_path}'. Defaulting to 0.0. Error: {e}")
        else:
            print(f"Warning: Report file '{report_path}' not found. Defaulting training time to 0.0.")
            
        return synthesizer, training_time
    
    # --- BLOCK 2: Train a new model if not found ---
    else:
        print(f"No model found. Fitting a new model: {type(synthesizer_to_fit).__name__}")
        
        # Use the provided unfitted synthesizer
        synthesizer = synthesizer_to_fit 
        
        start_time = time.time()
        synthesizer.fit(training_data)
        training_time = time.time() - start_time
        
        print(f"Fitting complete. Saving model to '{model_path}'...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print("Model saved successfully.")
        
        return synthesizer, training_time

