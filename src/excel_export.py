
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from src.chloride_model import modele_diffusion

# =========================
# === EXPORT EXCEL
# =========================
def export_group_excel(output_dir: Union[str, Path], safe_name: str, params_dict: Dict[str, Any], raw_data: Dict[str, Any]) -> None:
    """
    Exports processing results to an Excel file.
    
    output_dir: Base output directory
    safe_name: Filename base (without extension)
    params_dict: Dictionary of fitted parameters (Cs, Dnss, etc.) and metrics
    raw_data: Dictionary containing raw data and fit inputs
    
    Produces sheets: 'params', 'fit_curve', and 'raw_data' as in original script.
    """
    output_dir = Path(output_dir)
    excels_dir = output_dir / "excels"
    excels_dir.mkdir(parents=True, exist_ok=True)
    excel_path = excels_dir / f"{safe_name}.xlsx"
    
    # Unpack
    Ci = float(raw_data['Ci'])
    Cs = float(params_dict['Cs'])
    Dnss = float(params_dict['Dnss'])
    t_seconds = float(raw_data['t_seconds'])
    t_days = t_seconds / (24 * 3600)
    
    # Re-calculate fit curve
    x_fit_m = np.linspace(0, 0.025, 400) 
    y_fit = modele_diffusion(x_fit_m, Cs, Dnss, Ci, t_seconds)
    x_fit_mm = x_fit_m * 1000.0
    
    # Create DataFrames
    df_params = pd.DataFrame({
        "Ci": [Ci],
        "Cs_opt": [Cs],
        "Cs_std": [params_dict.get('Cs_std', np.nan)],
        "Dnss_opt (m²/s)": [Dnss],
        "Dnss_std (m²/s)": [params_dict.get('Dnss_std', np.nan)],
        "t_days": [t_days],
        "alpha": [0.01],
        "x_alpha_mm": [params_dict.get('x_alpha_mm', np.nan)],
        "x_cross_exp_mm": [params_dict.get('x_cross_mm', np.nan)],
    })
    
    df_curve = pd.DataFrame({"x_mm": x_fit_mm, "y_fit": y_fit})
    
    df_raw = pd.DataFrame({
        "x_mm_total": raw_data['raw_depths_mm'],
        "chlorures_total": raw_data['raw_chlorides']
    })
    
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_params.to_excel(writer, index=False, sheet_name="params")
            df_curve.to_excel(writer, index=False, sheet_name="fit_curve")
            df_raw.to_excel(writer, index=False, sheet_name="raw_data")
    except Exception as e:
        print(f"Error saving excel {safe_name}: {e}")
