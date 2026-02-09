
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

def plot_trends(results_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Plots trends for Cs and Dnss over time, grouped by Binder (color) and Condition (line style).
    
    This function:
    1. Filters outliers for Cs [0, 500%] and Dnss (median log +/- 2.5).
    2. Groups data by Cement and Condition.
    3. Plots each group with specific colors (CEM I=blue, CEM III=green, CEM V=red)
       and line styles (Standard=-, In situ=--, Autogenous=:).
    
    results_df: DataFrame containing [Cement, Condition, Age, Cs, Dnss]
    output_dir: Directory to save plots (trend_Cs.png, trend_Dnss.png)
    """
    output_dir = Path(output_dir)
    
    # Deterministic configuration
    COLORS = {
        "CEM I": "blue",
        "CEM III": "green", 
        "CEM V": "red"
    }
    
    LINE_STYLES = {
        "Standard": "-",      # Solid
        "In situ": "--",      # Dashed
        "Autogenous": ":"     # Dotted
    }
    
    # Ensure Age is numeric
    results_df["Age"] = pd.to_numeric(results_df["Age"])

    # --- Outlier Filtering ---
    # 1. Filter Cs: Range [0, 500%] (0 to 5.0)
    # Physical Cs shouldn't be negative or > 100% usually, but fit can drift. 5.0 is a safe cutoff for massive failures.
    original_len = len(results_df)
    results_df = results_df[(results_df["Cs"] >= 0) & (results_df["Cs"] <= 5.0)]
    
    # 2. Filter Dnss: Median log order of magnitude +/- 2
    # Dnss should be around ~1e-12.
    dnss_valid = results_df["Dnss"] > 0
    if dnss_valid.any():
        dnss_logs = np.log10(results_df.loc[dnss_valid, "Dnss"])
        median_log = dnss_logs.median()
        lower_log = median_log - 2.5 # S slightly wider to be safe
        upper_log = median_log + 2.5
        
        lower_bound = 10**lower_log
        upper_bound = 10**upper_log
        
        results_df = results_df[
            (results_df["Dnss"] >= lower_bound) & 
            (results_df["Dnss"] <= upper_bound)
        ]
    
    filtered_len = len(results_df)
    if filtered_len < original_len:
        print(f"Filtered {original_len - filtered_len} outliers from trend plots.")

    def _plot_metric(metric_col: str, ylabel: str, filename: str):
        plt.figure(figsize=(10, 6))
        
        # Group by Binder and Condition
        groups = results_df.groupby(["Cement", "Condition"])
        
        for (binder, condition), group in groups:
            # Sort by Age
            group = group.sort_values("Age")
            
            color = COLORS.get(binder, "black")
            linestyle = LINE_STYLES.get(condition, ":")
            
            label = f"{binder} - {condition}"
            
            plt.plot(group["Age"], group[metric_col], 
                     marker='o', 
                     color=color, 
                     linestyle=linestyle, 
                     label=label)
            
        plt.xlabel("Age (days)")
        plt.ylabel(ylabel)
        plt.title(f"Trend Analysis: {ylabel}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()

    # Plot Cs
    _plot_metric("Cs", "Surface Concentration (Cs) [%]", "trend_Cs.png")
    
    # Plot Dnss
    _plot_metric("Dnss", "Diffusion Coefficient (Dnss) [m^2/s]", "trend_Dnss.png")
