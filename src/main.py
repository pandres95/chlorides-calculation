
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chloride_model import fit_chloride_profile, calculate_x_alpha, interp_cross
from src.plotting import plot_profile
from src.excel_export import export_group_excel
from src.trends import plot_trends

def main() -> None:
    parser = argparse.ArgumentParser(description="Chloride Ingress Analysis")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="output", help="Directory for output files")
    parser.add_argument("--Ci", type=float, default=None, help="Global initial concentration override. If not set, uses defaults based on Cement Type.")
    args = parser.parse_args()

    # Default Ci values by Binder type
    # CEM I: 0.01370%
    # CEM III: 0.00950%
    # CEM V: 0.00975%
    CI_DEFAULTS = {
        "CEM I": 0.01370,
        "CEM III": 0.00950,
        "CEM V": 0.00975
    }

    # Load data
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Create output directories
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Columns expected in input
    # Based on input_data.csv: Binder,Condition,Exposure_Days,Depth_mm,Chloride_Mass_Pct
    group_cols = ['Binder', 'Condition', 'Exposure_Days']
    
    # Check if columns exist
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Input file missing columns: {missing_cols}")
        print(f"Expected: {group_cols}")
        return

    print(f"Processing {len(df)} rows from {args.input_file}...")

    for name, group in df.groupby(group_cols):
        binder, condition, exposure_days = name
        
        # Sort by depth
        group = group.sort_values("Depth_mm")
        
        # Data for fit (exclude depth 0, per constraint 2)
        fit_data = group[group["Depth_mm"] > 0]
        
        if fit_data.empty:
            print(f"Warning: No data for fit in group {name} (after removing depth 0)")
            continue

        # Prepare arrays
        raw_depths_mm = group["Depth_mm"].values
        raw_chlorides = group["Chloride_Mass_Pct"].values
        
        fit_depths_mm = fit_data["Depth_mm"].values
        fit_chlorides = fit_data["Chloride_Mass_Pct"].values
        
        # Convert to units for model
        fit_depths_m = fit_depths_mm / 1000.0
        t_seconds = float(exposure_days) * 24 * 3600
        
        # Determine Ci
        if args.Ci is not None:
             Ci = args.Ci
        else:
             # Strip potential whitespace
             binder_key = str(binder).strip()
             Ci = CI_DEFAULTS.get(binder_key, 0.0)
             if binder_key not in CI_DEFAULTS:
                 print(f"Warning: Unknown binder '{binder_key}', using Ci=0.0")

        Cs, Dnss, Cs_std, Dnss_std, r_squared = fit_chloride_profile(fit_depths_m, fit_chlorides, t_seconds, Ci=Ci)
        
        # Calculate derived metrics
        x_alpha_mm = np.nan
        x_cross_mm = None
        if not np.isnan(Dnss):
             alpha_m = calculate_x_alpha(Dnss, t_seconds)
             x_alpha_mm = alpha_m * 1000.0
             
             # Interpolate cross with Ci
             x_cross_mm = interp_cross(raw_depths_mm, raw_chlorides, Ci)

        # Sanitize filename
        safe_name = f"{binder}_{condition}_{exposure_days}".replace(" ", "_").replace("/", "-")
        
        # --- Plotting ---
        plot_path = plots_dir / f"{safe_name}.png"
        
        fitted_params = {
            'Cs': Cs, 'Dnss': Dnss,
            'Cs_std': Cs_std, 'Dnss_std': Dnss_std,
            'x_alpha_mm': x_alpha_mm,
            'x_cross_mm': x_cross_mm
        }
        
        data_group = {
            'raw_depths_mm': raw_depths_mm,
            'raw_chlorides': raw_chlorides,
            'fit_depths_mm': fit_depths_mm,
            'fit_chlorides': fit_chlorides,
            'Ci': Ci,
            't_seconds': t_seconds
        }
        
        try:
            plot_profile(data_group, fitted_params, plot_path)
        except Exception as e:
            print(f"Error plotting {name}: {e}")

        # --- Excel Export ---
        try:
             export_group_excel(output_dir, safe_name, fitted_params, data_group)
        except Exception as e:
             print(f"Error exporting excel {name}: {e}")

        # Store result
        results.append({
            "Cement": binder,
            "Condition": condition,
            "Age": exposure_days,
            "Cs": Cs,
            "Dnss": Dnss,
            "R_squared": r_squared,
            # Extra info useful for debug but not strictly requested in short list, put at end
            "Cs_std": Cs_std,
            "Dnss_std": Dnss_std
        })

    # Save report
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    
    # Reorder columns as requested: [Cement, Condition, Age, Cs, Dnss, R_squared]
    req_cols = ["Cement", "Condition", "Age", "Cs", "Dnss", "R_squared"]
    # Ensure they exist
    cols_to_save = [c for c in req_cols if c in results_df.columns]
    # append others
    cols_to_save += [c for c in results_df.columns if c not in cols_to_save]
    
    results_df = results_df[cols_to_save]
    
    results_path = output_dir / "results_report.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Processing complete. {len(results_df)} groups analyzed.")
    print(f"Report saved to: {results_path}")
    print(f"Plots saved to: {plots_dir}")

    # --- Trend Analysis ---
    try:
        plot_trends(results_df, output_dir)
        print(f"Trend plots saved to: {output_dir}")
    except Exception as e:
        print(f"Error plotting trends: {e}")

if __name__ == "__main__":
    main()
