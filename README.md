# Chloride Ingress Analysis

This project performs non-linear regression on chloride profiles to extract $D_{nss}$ (Non-steady state diffusion coefficient) and $C_s$ (Surface concentration) based on Fick's Second Law.

## Requirements

- Python 3.13+
- Dependencies listed in `requirements.txt`

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script via the CLI:

```bash
python3 src/main.py <input_file> [--output_dir <output_directory>]
```

### Example

```bash
python3 src/main.py input/input_data.csv --Ci 0.0137 --output_dir output
```

## Input Format

The input CSV file must contain the following columns:
- `Binder` (or `Cement_Type`)
- `Condition` (`Test_Condition`)
- `Exposure_Days` (`Age_Days`)
- `Depth_mm`
- `Chloride_Mass_Pct`

## Outputs

The script generates:
1. **Report**: `<output_dir>/results_report.csv` containing fit parameters (`Cs`, `Dnss`, `R_squared`) for each group.
3. **Excel Exports**: `<output_dir>/excels/` containing detailed data and model curves for each group.
4. **Trend Plots**: `<output_dir>/trend_Cs.png` and `<output_dir>/trend_Dnss.png`, showing parameter evolution over time.

## Methodology

- **Model**: Fick's Second Law (Error Function solution).
- **Fitting**: Non-linear least squares regression (`scipy.optimize.curve_fit`).
- **Surface Skin Effect**: The first data point (depth = 0) is excluded from the fit but shown in plots.
- **Initial Concentration ($C_i$)**: Default values are determined by the `Binder` (Cement Type) as follows:
    - **CEM I**: 0.01370%
    - **CEM III**: 0.00950%
    - **CEM V**: 0.00975%
  - These can be overridden globally using the `--Ci` argument (e.g., `--Ci 0.0`). If a binder type is unknown, $C_i$ defaults to 0.0.
