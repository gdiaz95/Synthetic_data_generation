import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import wandb
import time

def create_comparison_plots():
    """
    Loads report data, generates plots with consistent colors, and logs
    both the plots and grouped data tables to Weights & Biases.
    """
    # --- 1. Configuration ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    report_base_dir = os.path.join(project_root, 'reports')
    output_dir = os.path.join(project_root, 'images', 'report_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    run = wandb.init(
        project="synthetic-data-generation",
        job_type="analysis",
        name=f"Comparison_Report_{int(time.time())}"
    )
    print(f" Initialized W&B run for reporting: {run.name}")

    methods = [
        'CTGAN', 'GaussianCopula', 'CopulaGAN', 'TVAE', 'Gaussian_correlated'
    ]
    
    # --- ADDED: Define a permanent color map for consistent plots ---
    color_map = {
        'CTGAN': '#1f77b4',
        'GaussianCopula': '#ff7f0e',
        'CopulaGAN': '#2ca02c',
        'TVAE': '#d62728',
        'Gaussian_correlated': '#9467bd'
    }

    # --- 2. Data Extraction ---
    all_scores_data = []
    print("\nStarting to parse report files...")
    for method in methods:
        for i in range(1, 11):
            file_path = os.path.join(report_base_dir, method, str(i), f'{i}.json')
            if not os.path.exists(file_path):
                 file_path = os.path.join(report_base_dir, method, f'{i}.json') # Check alternate path

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    # ... (data loading logic is unchanged) ...
                    report = json.load(f)
                    diag_props = {p['Property']: p['Score'] for p in report['diagnostic_report']['properties']}
                    qual_props = {p['Property']: p['Score'] for p in report['quality_report']['properties']}
                    all_scores_data.append({
                        'Iteration': i, 'Method': method,
                        'Data Validity': diag_props.get('Data Validity'),
                        'Data Structure': diag_props.get('Data Structure'),
                        'Overall Score': report['quality_report'].get('overall_score'),
                        'Column Shapes': qual_props.get('Column Shapes'),
                        'Column Pair Trends': qual_props.get('Column Pair Trends')
                    })

    # --- 3. Create DataFrame and Log Grouped Tables ---
    if not all_scores_data:
        print("No data was loaded. Exiting.")
        run.finish()
        return

    scores_df = pd.DataFrame(all_scores_data)
    print("\nData successfully loaded into a DataFrame.")

    # --- CHANGED: Log data in separate tables for each method ---
    print("\nLogging scores to W&B in tables grouped by method...")
    for method in scores_df['Method'].unique():
        method_df = scores_df[scores_df['Method'] == method].reset_index(drop=True)
        table = wandb.Table(dataframe=method_df)
        run.log({f"method_scores/{method}_table": table})
    print("-> All method tables logged successfully.")

    # --- 4. Generate and Save Plots ---
    metrics_to_plot = [
        'Data Validity', 'Data Structure', 'Overall Score', 
        'Column Shapes', 'Column Pair Trends'
    ]

    print("\nGenerating, saving, and logging comparison plots...")
    for metric in metrics_to_plot:
        plt.figure(figsize=(14, 8))

        # --- CHANGED: Use the consistent color_map in the palette ---
        sns.lineplot(data=scores_df, x='Iteration', y=metric, hue='Method', 
                     marker='o', palette=color_map, hue_order=methods, linewidth=3, markersize=8)
        
        plt.title(f'{metric} Comparison Across Methods', fontsize=18, pad=20)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(range(1, 11))
        plt.ylim(0, 1.05)
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        wandb.log({f"comparison_plots/{metric}": wandb.Image(plt)})

        filename = metric.lower().replace(' ', '_') + '_comparison.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"-> Plot for '{metric}' saved and logged.")

    run.finish()
    print("\n All comparison plots created and logged to W&B successfully.")

if __name__ == '__main__':
    create_comparison_plots()