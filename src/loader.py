import os
import json
import time
from contextlib import contextmanager
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sdv.metadata import Metadata
from sdv.utils import load_synthesizer

"""Utilities for loading datasets and persisting synthesizer artifacts."""

_THREAD_LIMITED_SYNTHESIZERS = {"CTGANSynthesizer", "CopulaGANSynthesizer", "TVAESynthesizer"}
_THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


@contextmanager
def _temporary_thread_limit_for_deep_sdv_models(synthesizer):
    """Temporarily clamp BLAS/OpenMP threads for deep SDV models."""
    model_name = type(synthesizer).__name__
    max_threads = os.getenv("SDG_MAX_BLAS_THREADS", "32")

    if model_name not in _THREAD_LIMITED_SYNTHESIZERS:
        yield
        return

    original_values = {key: os.environ.get(key) for key in _THREAD_ENV_VARS}
    try:
        for key in _THREAD_ENV_VARS:
            current_value = os.environ.get(key)
            if current_value is None:
                os.environ[key] = max_threads
            else:
                try:
                    if int(current_value) > int(max_threads):
                        os.environ[key] = max_threads
                except ValueError:
                    os.environ[key] = max_threads
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def _is_threading_related_error(error):
    """Heuristic check for OpenBLAS/MKL/OMP thread-related fit failures."""
    error_text = str(error).lower()
    thread_signals = (
        "openblas",
        "mkl",
        "omp",
        "thread",
        "blas",
        "memory regions",
        "num_threads",
    )
    return any(signal in error_text for signal in thread_signals)


def _fit_synthesizer_with_thread_fallback(synthesizer, training_data):
    """Try fit normally first; only apply thread cap and retry on thread errors."""
    model_name = type(synthesizer).__name__

    if model_name not in _THREAD_LIMITED_SYNTHESIZERS:
        synthesizer.fit(training_data)
        return

    try:
        synthesizer.fit(training_data)
    except Exception as error:
        if not _is_threading_related_error(error):
            raise

        print(
            "Detected a BLAS/OpenMP threading-related fit error. "
            "Retrying fit with capped thread env vars..."
        )
        with _temporary_thread_limit_for_deep_sdv_models(synthesizer):
            synthesizer.fit(training_data)

def load_or_train_synthesizer(training_data, model_path, report_path, synthesizer_to_fit):
    """
    Loads a synthesizer from model_path if it exists, otherwise fits the
    provided synthesizer_to_fit and saves it.
    
    Returns the (loaded or trained) synthesizer and the training time.
    """
    
    # --- BLOCK 1: Try to load existing model ---
    if os.path.exists(model_path):
        print(f"Found existing model at '{model_path}'. Loading...")
        
        # Custom logic: Bypass SDV loader ONLY for the custom synthesizer
        if type(synthesizer_to_fit).__name__ == "NonParamGaussianCopulaSynthesizer":
            import pickle
            with open(model_path, 'rb') as f:
                synthesizer = pickle.load(f)
        else:
            # Standard loader for SDV models (TVAE, GaussianCopula, etc.)
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
        _fit_synthesizer_with_thread_fallback(synthesizer, training_data)
        training_time = time.time() - start_time
        
        print(f"Fitting complete. Saving model to '{model_path}'...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print("Model saved successfully.")
        
        return synthesizer, training_time

def load_and_prepare_data(dataset_name, metadata_path):
    """
    Loads and prepares the specified dataset and its metadata.
    
    Returns the dataframe, metadata, and the dataset's target column.
    """
    print(f"--- Loading dataset: {dataset_name} ---")
    
    # Route to dataset-specific loading logic.
    if dataset_name == 'adults':
        # --- Adults Dataset Logic ---
        print("Loading original 'adults' dataset and metadata...")
        data = fetch_ucirepo(id=2)
        df = pd.concat([data.data.features, data.data.targets], axis=1)
        df['income'] = df['income'].str.strip().str.replace('.', '', regex=False)
        table_name = 'adult_data'
        target_column = 'income' # <-- Dataset-specific target
        # --- End Adults Logic ---
        
    elif dataset_name == 'car_evaluation':
        # --- Car Evaluation Dataset Logic ---
        print("Loading 'Car Evaluation' dataset and metadata...")
        data = fetch_ucirepo(id=19)
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Rename target for clarity (UCI uses 'class')
        df.rename(columns={'class': 'car_acceptability'}, inplace=True)

        table_name = 'car_evaluation_data'
        target_column = 'car_acceptability'
        
    elif dataset_name == 'balance_scale':
        # --- Balance Scale Dataset Logic ---
        print("Loading 'Balance Scale' dataset and metadata...")
        data = fetch_ucirepo(id=12)
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Rename target column for consistency
        df.rename(columns={'class': 'balance_class'}, inplace=True)

        table_name = 'balance_scale_data'
        target_column = 'balance_class'
        
    elif dataset_name == 'nursery':
        # --- Nursery Dataset Logic ---
        print("Loading 'Nursery' dataset and metadata...")
        data = fetch_ucirepo(id=26)
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Rename target for consistency
        df.rename(columns={'class': 'nursery_class'}, inplace=True)

        table_name = 'nursery_data'
        target_column = 'nursery_class'
    
    elif dataset_name == 'student_performance':
        print("Loading 'Student Performance' dataset and metadata...")
        data = fetch_ucirepo(id=320)
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        grade_cols = ['G1', 'G2', 'G3']
        df['avg_grade'] = df[grade_cols].astype(float).mean(axis=1)

        # Define pass/fail based on average grade
        df['performance_category'] = df['avg_grade'].apply(lambda x: 'pass' if x >= 10 else 'fail')

        # Drop the raw grade columns
        df.drop(columns=grade_cols, inplace=True)
        df.drop(columns=['avg_grade'], inplace=True)

        table_name = 'student_performance_data'
        target_column = 'performance_category'
        
        
    elif dataset_name == 'student_dropout_success':
        print("Loading 'Predict Students’ Dropout and Academic Success' dataset and metadata...")
        data = fetch_ucirepo(id=697)

        # Combine features and target(s)
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Ensure we have exactly one target column; rename it consistently

        original_target_name = data.data.targets.columns[0]
        df.rename(columns={original_target_name: 'student_outcome'}, inplace=True)

        table_name = 'student_dropout_success_data'
        target_column = 'student_outcome'

    # --- ADD NEW DATASETS HERE ---
    # elif dataset_name == 'your_next_dataset':
    #     print("Loading 'your_next_dataset'...")
    #     # df = ... (load your data)
    #     # ... (do your preprocessing)
    #     # table_name = 'some_table'
    #     # target_column = 'some_target'
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized in src/loader.py")

    # --- Generic Metadata Handling ---
    # Reuse existing metadata when present; otherwise infer and persist it.
    if os.path.exists(metadata_path):
        metadata = Metadata.load_from_json(filepath=metadata_path)
    else:
        print("Metadata not found. Detecting from dataframe...")
        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name=table_name, data=df)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata.save_to_json(filepath=metadata_path)
        print(f"New metadata saved to '{metadata_path}'")
        
    print(f"Dataset '{dataset_name}' loaded successfully.")
    
    # <-- RETURN THE TARGET COLUMN ---
    return df, metadata, target_column
