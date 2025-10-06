import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def create_comparison_plots():
    """
    Loads all JSON report data, aggregates it, and generates line plots
    comparing the performance of different synthetic data generation methods
    across multiple iterations.
    """
    # --- 1. Configuration (Updated for Portability) ---
    
    # Get the absolute path of the directory where this script is located.
    # This assumes the script is in the project's root folder.
    project_root = os.path.dirname(os.path.abspath(__file__)) + '/../'

    # Build all other paths based on this project_root.
    report_base_dir = os.path.join(project_root, 'reports')
    output_dir = os.path.join(project_root, 'images', 'report_comparison')
    os.makedirs(output_dir, exist_ok=True)


    methods = [
        'CTGAN',
        'GaussianCopula',
        'CopulaGAN',
        'TVAE',
        'Gaussian_correlated'
    ]
    iterations = range(1, 11)
    
    all_scores_data = []
    print("Starting to parse report files...")

    # --- 2. Data Extraction ---
    for method in methods:
        for i in iterations:
            file_path = os.path.join(report_base_dir, method, f'{i}.json')
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        report = json.load(f)
                        
                        diag_props = {prop['Property']: prop['Score'] for prop in report['diagnostic_report']['properties']}
                        qual_props = {prop['Property']: prop['Score'] for prop in report['quality_report']['properties']}

                        all_scores_data.append({
                            'Iteration': i,
                            'Method': method,
                            'Data Validity': diag_props.get('Data Validity'),
                            'Data Structure': diag_props.get('Data Structure'),
                            'Overall Score': report['quality_report'].get('overall_score'),
                            'Column Shapes': qual_props.get('Column Shapes'),
                            'Column Pair Trends': qual_props.get('Column Pair Trends')
                        })
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Warning: Could not parse file {file_path}. Error: {e}. Skipping.")
            else:
                print(f"Warning: File not found {file_path}. Skipping.")

    # --- 3. Create a single DataFrame ---
    if not all_scores_data:
        print("No data was loaded. Exiting.")
        return

    scores_df = pd.DataFrame(all_scores_data)
    print("\nData successfully loaded into a DataFrame.")

    # --- 4. Generate and Save Plots ---
    metrics_to_plot = [
        'Data Validity', 
        'Data Structure', 
        'Overall Score', 
        'Column Shapes', 
        'Column Pair Trends'
    ]

    print("\nGenerating and saving comparison plots...")
    for metric in metrics_to_plot:
        plt.figure(figsize=(14, 8))

        sns.lineplot(data=scores_df, x='Iteration', y=metric, hue='Method', marker='o', palette='tab10', linewidth=4,markersize=10)
        plt.title(f'{metric} Comparison Across Methods and Iterations', fontsize=18, pad=20)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(iterations)
        plt.ylim(0, 1.05)
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        filename = metric.lower().replace(' ', '_') + '_comparison.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"-> Plot saved: {save_path}")

    print("\nAll comparison plots created successfully.")

if __name__ == '__main__':
    create_comparison_plots()