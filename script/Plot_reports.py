import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# --- 1. Shared Configuration ---
# Moved all shared settings here to be used by all functions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASETS_TO_PROCESS = [
    'car_evaluation',
    'balance_scale',  # Using 'balance_scale' as provided in your code
    'nursery',
    'student_performance',
    'student_dropout_success',
    'adults'
]

METHODS = [
    'CTGAN', 'GaussianCopula', 'CopulaGAN', 'TVAE', 'Gaussian_correlated'
]

COLOR_MAP = {
    'CTGAN': '#1f77b4',
    'GaussianCopula': '#ff7f0e',
    'CopulaGAN': '#2ca02c',
    'TVAE': '#d62728',
    'Gaussian_correlated': '#9467bd'
}

ALL_METRICS_TO_PLOT = [
    'Data Validity', 'Data Structure', 'Overall Score', 
    'Column Shapes', 'Column Pair Trends',
    'overall_accuracy', 'univariate_accuracy', 'bivariate_accuracy',
    'discriminator_auc', 'identical_matches', 'dcr_training',
    'dcr_holdout', 'dcr_share', 'training_time', 'evaluation_time',
    'TSTR_XGBoost Classifier_Accuracy_Real_Score',
    'TSTR_XGBoost Classifier_Accuracy_Synthetic_Score',
    'TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%'
]


# --- 2. Function 1: Adults-Only Iteration Plots (Your Original Function) ---
# RENAMED for clarity
def create_comparison_plots_adults_only():
    """
    Loads report data for ADULTS dataset only, generates iteration plots, 
    and saves to 'images/adults_comparison'.
    """
    print("--- Running Function 1: Adults-Only Iteration Plots ---")
    # --- 1. Configuration ---
    dataset_name = 'adults'
    
    report_base_dir = os.path.join(PROJECT_ROOT, 'reports', dataset_name)
    
    output_dir = os.path.join(PROJECT_ROOT, 'images', 'adults_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    run_id_file = os.path.join(output_dir, 'wandb.txt')
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
    
    run = wandb.init(
        project="synthetic-data-generation",
        job_type="analysis",
        group="comparison_reports",
        name=f"Adults_Report",
        id=run_id,
        resume="allow"
    )
    print(f" Initialized W&B run for reporting: {run.name}")

    if not run.resumed:
        with open(run_id_file, 'w') as f:
            f.write(run.id)
            
    # --- 2. Data Extraction ---
    all_scores_data = []
    print("\nStarting to parse report files...")

    for method in METHODS:
        for i in range(1, 11): # Hard-coded for 'adults'
            file_path = os.path.join(report_base_dir, method, f'{i}.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    report = json.load(f)
                    diag_props = {p['Property']: p['Score'] for p in report.get('diagnostic_report', {}).get('properties', [])}
                    qual_props = {p['Property']: p['Score'] for p in report.get('quality_report', {}).get('properties', [])}
                    metrics_qa = report.get('metrics_qa', {})
                    times = report.get('times', {})
                    tstr = report.get('tstr_evaluation', {})

                    all_scores_data.append({
                        'Iteration': i, 'Method': method,
                        **{metric: None for metric in ALL_METRICS_TO_PLOT}, 
                        'Data Validity': diag_props.get('Data Validity'),
                        'Data Structure': diag_props.get('Data Structure'),
                        'Overall Score': report.get('quality_report', {}).get('overall_score'),
                        'Column Shapes': qual_props.get('Column Shapes'),
                        'Column Pair Trends': qual_props.get('Column Pair Trends'),
                        'overall_accuracy': metrics_qa.get('overall_accuracy'),
                        'univariate_accuracy': metrics_qa.get('univariate_accuracy'),
                        'bivariate_accuracy': metrics_qa.get('bivariate_accuracy'),
                        'discriminator_auc': metrics_qa.get('discriminator_auc'),
                        'identical_matches': metrics_qa.get('identical_matches'),
                        'dcr_training': metrics_qa.get('dcr_training'),
                        'dcr_holdout': metrics_qa.get('dcr_holdout'),
                        'dcr_share': metrics_qa.get('dcr_share'),
                        'training_time': times.get('training_time'),
                        'evaluation_time': times.get('evaluation_time'),
                        'TSTR_XGBoost Classifier_Accuracy_Real_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Real_Score'),
                        'TSTR_XGBoost Classifier_Accuracy_Synthetic_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Synthetic_Score'),
                        'TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%': tstr.get('TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%'),
                    })

    # --- 3. Create DataFrame and Log Grouped Tables ---
    if not all_scores_data:
        print(f"No data was loaded from {report_base_dir}. Exiting.")
        run.finish()
        return

    scores_df = pd.DataFrame(all_scores_data)
    print("\nData successfully loaded into a DataFrame.")

    print("\nLogging scores to W&B in tables grouped by method...")
    for method in scores_df['Method'].unique():
        method_df = scores_df[scores_df['Method'] == method].reset_index(drop=True)
        table = wandb.Table(dataframe=method_df)
        run.log({f"{dataset_name}/method_scores/{method}_table": table})
    print("-> All method tables logged successfully.")

    # --- 4. Generate and Save Plots ---
    print("\nGenerating, saving, and logging comparison plots...")
    for metric in ALL_METRICS_TO_PLOT:
        if scores_df[metric].isnull().all():
            print(f"-> Skipping plot for '{metric}' (all data is NaN).")
            continue
            
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=scores_df, x='Iteration', y=metric, hue='Method', 
                     marker='o', palette=COLOR_MAP, hue_order=METHODS, linewidth=3, markersize=8)
        
        plt.title(f'{dataset_name.capitalize()}: {metric} Comparison Across Methods', fontsize=18, pad=20)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Score' if 'time' not in metric.lower() else 'Value', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(range(1, 11))
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({f"{dataset_name}/comparison_plots/{metric}": wandb.Image(plt)})
        filename = metric.lower().replace(' ', '_').replace('%', 'perc').replace('/', '_') + '_comparison.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"-> Plot for '{metric}' saved and logged.")

    run.finish()
    print(f"\n All comparison plots for '{dataset_name}' created and logged to W&B successfully.")


# --- 3. Function 2: Overall Average Plots (Your Original Function) ---

def create_average_plots():
    """
    Loads report data from ALL specified datasets, calculates the average score 
    for each method, and logs bar plots of these averages to W&B.
    """
    print("--- Running Function 2: Overall Average Plots ---")
    # --- 1. Configuration ---
    report_base_dir = os.path.join(PROJECT_ROOT, 'reports')
    output_dir = os.path.join(PROJECT_ROOT, 'images', 'report_average_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    run_id_file = os.path.join(output_dir, 'wandb.txt')
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
    
    run = wandb.init(
        project="synthetic-data-generation",
        job_type="analysis",
        group="comparison_reports",
        name=f"Overall_Average_Report",
        id=run_id,
        resume="allow"
    )
    print(f" Initialized W&B run for reporting: {run.name}")

    if not run.resumed:
        with open(run_id_file, 'w') as f:
            f.write(run.id)
            
    # --- 2. Data Extraction (CHANGED to loop datasets) ---
    all_scores_data = []
    print("\nStarting to parse report files from all datasets...")

    for dataset_name in DATASETS_TO_PROCESS:
        print(f"--- Processing dataset: {dataset_name} ---")
        dataset_report_dir = os.path.join(report_base_dir, dataset_name)
        if not os.path.exists(dataset_report_dir):
            print(f"Warning: Directory not found, skipping: {dataset_report_dir}")
            continue

        for method in METHODS:
            method_dir = os.path.join(dataset_report_dir, method)
            if not os.path.exists(method_dir):
                print(f"Warning: Method directory not found, skipping: {method_dir}")
                continue

            iterations_to_load = []
            if dataset_name == 'adults':
                iterations_to_load = ['1'] # Special rule: only iteration 1 for adults
            else:
                try:
                    found_files = [
                        f for f in os.listdir(method_dir) 
                        if f.endswith('.json') and f.split('.')[0].isdigit()
                    ]
                    iterations_to_load = [f.split('.')[0] for f in found_files]
                except OSError as e:
                    print(f"Error reading {method_dir}: {e}")
                    continue

            if not iterations_to_load:
                print(f"Warning: No iterations found for {dataset_name}/{method}")
                continue
                
            for i_str in iterations_to_load:
                file_path = os.path.join(method_dir, f'{i_str}.json')
                
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        report = json.load(f)
                        diag_props = {p['Property']: p['Score'] for p in report.get('diagnostic_report', {}).get('properties', [])}
                        qual_props = {p['Property']: p['Score'] for p in report.get('quality_report', {}).get('properties', [])}
                        metrics_qa = report.get('metrics_qa', {})
                        times = report.get('times', {})
                        tstr = report.get('tstr_evaluation', {})

                        all_scores_data.append({
                            'Dataset': dataset_name, 'Iteration': i_str, 'Method': method,
                            'Data Validity': diag_props.get('Data Validity'),
                            'Data Structure': diag_props.get('Data Structure'),
                            'Overall Score': report.get('quality_report', {}).get('overall_score'),
                            'Column Shapes': qual_props.get('Column Shapes'),
                            'Column Pair Trends': qual_props.get('Column Pair Trends'),
                            'overall_accuracy': metrics_qa.get('overall_accuracy'),
                            'univariate_accuracy': metrics_qa.get('univariate_accuracy'),
                            'bivariate_accuracy': metrics_qa.get('bivariate_accuracy'),
                            'discriminator_auc': metrics_qa.get('discriminator_auc'),
                            'identical_matches': metrics_qa.get('identical_matches'),
                            'dcr_training': metrics_qa.get('dcr_training'),
                            'dcr_holdout': metrics_qa.get('dcr_holdout'),
                            'dcr_share': metrics_qa.get('dcr_share'),
                            'training_time': times.get('training_time'),
                            'evaluation_time': times.get('evaluation_time'),
                            'TSTR_XGBoost Classifier_Accuracy_Real_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Real_Score'),
                            'TSTR_XGBoost Classifier_Accuracy_Synthetic_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Synthetic_Score'),
                            'TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%': tstr.get('TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%'),
                        })
                else:
                    print(f"Warning: File not found {file_path}")

    # --- 3. Create DataFrame and Calculate Averages ---
    if not all_scores_data:
        print("No data was loaded from any dataset. Exiting.")
        run.finish()
        return

    scores_df = pd.DataFrame(all_scores_data)
    # Convert numeric columns, forcing errors to NaN
    for col in scores_df.columns:
        if col not in ['Dataset', 'Method']:
             scores_df[col] = pd.to_numeric(scores_df[col], errors='coerce')
             
    print("\nFull data successfully loaded into a DataFrame.")

    avg_scores_df = scores_df.groupby('Method').mean(numeric_only=True).reset_index()

    print("\nLogging average scores to W&B table...")
    table = wandb.Table(dataframe=avg_scores_df)
    run.log({"average_method_scores": table})
    print("-> Average table logged successfully.")

    # --- 4. Generate and Save Plots (Bar Plots) ---
    print("\nGenerating, saving, and logging average comparison plots...")
    for metric in ALL_METRICS_TO_PLOT:
        if metric not in avg_scores_df.columns or avg_scores_df[metric].isnull().all():
            print(f"-> Skipping plot for '{metric}' (no data).")
            continue
            
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=avg_scores_df, 
            x='Method', 
            y=metric, 
            palette=COLOR_MAP, 
            order=METHODS,
            hue='Method',    
            legend=False
        )
        plt.title(f'Average {metric} Across All Datasets', fontsize=18, pad=20)
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Average Value', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        wandb.log({f"average_comparison_plots/{metric}": wandb.Image(plt)})
        filename = 'avg_' + metric.lower().replace(' ', '_').replace('%', 'perc').replace('/', '_') + '_comparison.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"-> Plot for 'Average {metric}' saved and logged.")

    run.finish()
    print("\n All average comparison plots created and logged to W&B successfully.")


# --- 4. Function 3: Per-Dataset Average/Single Plots (CORRECTED) ---

def create_dataset_average_plots():
    """
    Loops through each dataset and generates BAR plots, saving them to
    'images/<dataset_name>/metrics/'.
    - For 'adults', plots iteration 1.
    - For all other datasets, plots the average of all iterations.
    """
    print("--- Running Function 3: Per-Dataset Average/Single Plots ---")
    
    # --- W&B Initialization ---
    wandb_id_dir = os.path.join(PROJECT_ROOT, 'images', 'report_dataset_averages')
    os.makedirs(wandb_id_dir, exist_ok=True)
    run_id_file = os.path.join(wandb_id_dir, 'wandb.txt')
    
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
    
    run = wandb.init(
        project="synthetic-data-generation",
        job_type="analysis",
        group="comparison_reports", # Same group
        name=f"Dataset_Average_Reports", # New name
        id=run_id,
        resume="allow"
    )
    print(f" Initialized W&B run for reporting: {run.name}")
    
    if not run.resumed:
        with open(run_id_file, 'w') as f:
            f.write(run.id)

    # --- Main Dataset Loop ---
    report_base_dir_root = os.path.join(PROJECT_ROOT, 'reports')
    images_root = os.path.join(PROJECT_ROOT, 'images')

    for dataset_name in DATASETS_TO_PROCESS:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        report_base_dir = os.path.join(report_base_dir_root, dataset_name)
        output_dir = os.path.join(images_root, dataset_name, 'metrics')
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(report_base_dir):
            print(f"Warning: Directory not found, skipping: {report_base_dir}")
            continue

        # --- Data Extraction (for this dataset) ---
        all_scores_data = []
        print("Starting to parse report files...")

        for method in METHODS:
            method_dir = os.path.join(report_base_dir, method)
            if not os.path.exists(method_dir):
                print(f"Warning: Method dir not found, skipping: {method_dir}")
                continue
            
            iterations_to_load = []
            if dataset_name == 'adults':
                iterations_to_load = ['1'] # Special rule
            else:
                try:
                    found_files = [
                        f for f in os.listdir(method_dir) 
                        if f.endswith('.json') and f.split('.')[0].isdigit()
                    ]
                    iterations_to_load = [f.split('.')[0] for f in found_files]
                except OSError as e:
                    print(f"Error reading {method_dir}: {e}")
                    continue

            if not iterations_to_load:
                print(f"Warning: No iterations found for {dataset_name}/{method}")
                continue

            for i_str in iterations_to_load:
                file_path = os.path.join(method_dir, f'{i_str}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        report = json.load(f)
                        diag_props = {p['Property']: p['Score'] for p in report.get('diagnostic_report', {}).get('properties', [])}
                        qual_props = {p['Property']: p['Score'] for p in report.get('quality_report', {}).get('properties', [])}
                        metrics_qa = report.get('metrics_qa', {})
                        times = report.get('times', {})
                        tstr = report.get('tstr_evaluation', {})

                        # --- CORRECTED DATA APPEND BLOCK ---
                        all_scores_data.append({
                            'Iteration': int(i_str), 'Method': method,
                            'Data Validity': diag_props.get('Data Validity'),
                            'Data Structure': diag_props.get('Data Structure'),
                            'Overall Score': report.get('quality_report', {}).get('overall_score'),
                            'Column Shapes': qual_props.get('Column Shapes'),
                            'Column Pair Trends': qual_props.get('Column Pair Trends'),
                            'overall_accuracy': metrics_qa.get('overall_accuracy'),
                            'univariate_accuracy': metrics_qa.get('univariate_accuracy'),
                            'bivariate_accuracy': metrics_qa.get('bivariate_accuracy'),
                            'discriminator_auc': metrics_qa.get('discriminator_auc'),
                            'identical_matches': metrics_qa.get('identical_matches'),
                            'dcr_training': metrics_qa.get('dcr_training'),
                            'dcr_holdout': metrics_qa.get('dcr_holdout'),
                            'dcr_share': metrics_qa.get('dcr_share'),
                            'training_time': times.get('training_time'),
                            'evaluation_time': times.get('evaluation_time'),
                            'TSTR_XGBoost Classifier_Accuracy_Real_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Real_Score'),
                            'TSTR_XGBoost Classifier_Accuracy_Synthetic_Score': tstr.get('TSTR_XGBoost Classifier_Accuracy_Synthetic_Score'),
                            'TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%': tstr.get('TSTR_XGBoost Classifier_Accuracy_Performance_Drop_%'),
                        })
                else:
                    print(f"Warning: File not found {file_path}")
        
        # --- Create DataFrame and Log Tables ---
        if not all_scores_data:
            print(f"No data was loaded for {dataset_name}. Skipping plots.")
            continue

        # --- CORRECTED DATAFRAME CREATION ---
        scores_df = pd.DataFrame(all_scores_data)
        print("Data loaded into DataFrame.")

        # --- Calculate average for this dataset ---
        # If 'adults', this df only has iter 1. If not, it has all iters.
        # .mean() will correctly return the single value for 'adults'
        # or the average for all other datasets.
        dataset_avg_df = scores_df.groupby('Method').mean(numeric_only=True).reset_index()

        # Log the dataset-specific average table
        print("Logging scores to W&B table...")
        table = wandb.Table(dataframe=dataset_avg_df)
        run.log({f"{dataset_name}/method_scores/average_table": table})
        
        # --- Generate and Save Plots (Bar Plots) ---
        print("Generating, saving, and logging comparison plots...")
        
        for metric in ALL_METRICS_TO_PLOT:
            if metric not in dataset_avg_df.columns or dataset_avg_df[metric].isnull().all():
                print(f"-> Skipping plot for '{metric}' (no data).")
                continue
            
            plt.figure(figsize=(14, 8))

            sns.barplot(
                data=dataset_avg_df, 
                x='Method', 
                y=metric, 
                palette=COLOR_MAP, 
                order=METHODS,
                hue='Method',
                legend=False
            )
            
            plt.title(f'{dataset_name.capitalize()}: {metric} Comparison', fontsize=18, pad=20)
            plt.xlabel('Method', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            # Log to dataset-specific path
            wandb.log({f"{dataset_name}/comparison_plots/{metric}": wandb.Image(plt)})

            filename = metric.lower().replace(' ', '_').replace('%', 'perc').replace('/', '_') + '_comparison.png'
            # Save to new output path
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"-> Plot for '{metric}' saved to {save_path} and logged.")
    
    run.finish()
    print("\nAll dataset average/single plots created and logged to W&B successfully.")


# --- 5. Main Execution Block ---

if __name__ == '__main__':
    
    # 1. Runs the 'adults-only' plot (saves to images/adults_comparison)
    create_comparison_plots_adults_only()
    
    # 2. Runs the 'average' plots (saves to images/report_average_comparison)
    create_average_plots()

    # 3. Runs the per-dataset average plots (saves to images/<dataset>/metrics/)
    create_dataset_average_plots()