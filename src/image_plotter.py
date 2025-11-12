import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_marginals(dataset_name, model, i, real_df, synth_df):
    save_dir = f"images/{dataset_name}/{model}/{i}"
    os.makedirs(save_dir, exist_ok=True)

    common_cols = [c for c in real_df.columns if c in synth_df.columns]

    for col in common_cols:
        real_col = real_df[col].dropna()
        synth_col = synth_df[col].dropna()

        safe_col = col.replace("/", "_").replace("\\", "_").replace(":", "_")

        plt.figure(figsize=(10, 6))

        if pd.api.types.is_numeric_dtype(real_col) and pd.api.types.is_numeric_dtype(synth_col):
            sns.kdeplot(real_col, fill=True, alpha=0.5, label="Original Real", color="steelblue")
            sns.kdeplot(synth_col, fill=True, alpha=0.5, label="Synthetic", color="orange")
            plt.ylabel("Density")
            plt.title(f"Numerical Distribution: {col}")

        else:
            real_counts = real_col.value_counts(normalize=True)
            synth_counts = synth_col.value_counts(normalize=True)
            combined = pd.DataFrame({
                "Original Real": real_counts,
                "Synthetic": synth_counts
            }).fillna(0)

            combined.plot(kind="bar", alpha=0.8)
            plt.title(f"Categorical Distribution: {col}")
            plt.ylabel("Proportion")

        plt.xlabel(col)
        plt.legend()
        plt.tight_layout()

        path = os.path.join(save_dir, f"{safe_col}.png")
        plt.savefig(path, dpi=150)
        plt.close()