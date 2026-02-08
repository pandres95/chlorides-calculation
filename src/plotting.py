
import matplotlib.pyplot as plt
import numpy as np
from .chloride_model import modele_diffusion

def plot_profile(data_group, fitted_params, output_path, alpha=0.01):
    """
    Generates and saves the plot for a specific group.
    
    data_group: dict containing:
        - raw_depths_mm: all depths
        - raw_chlorides: all chlorides
        - fit_depths_mm: depths used for fit
        - fit_chlorides: chlorides used for fit
        - Ci: initial concentration used
        - t_seconds: time in seconds
    fitted_params: dict containing Cs, Dnss, Cs_std, Dnss_std, x_alpha_mm, x_cross_mm, etc.
    output_path: path to save the PNG
    alpha: alpha value for x_alpha
    """
    
    # Unpack data
    raw_depths = np.array(data_group['raw_depths_mm'])
    raw_chlorides = np.array(data_group['raw_chlorides'])
    fit_depths = np.array(data_group['fit_depths_mm'])
    fit_chlorides = np.array(data_group['fit_chlorides'])
    Ci = data_group['Ci']
    t = data_group['t_seconds']
    
    Cs = fitted_params['Cs']
    Dnss = fitted_params['Dnss']
    Cs_std = fitted_params['Cs_std']
    Dnss_std = fitted_params['Dnss_std']
    x_alpha_mm = fitted_params['x_alpha_mm']
    x_cross_mm = fitted_params['x_cross_mm']

    # Generate fit curve
    # 0–25 mm range for plotting as in original script (xlim was 25)
    x_fit_m = np.linspace(0, 0.025, 400) 
    y_fit = modele_diffusion(x_fit_m, Cs, Dnss, Ci, t)
    x_fit_mm = x_fit_m * 1000.0
    
    target_conc = Ci + alpha * (Cs - Ci)

    plt.figure(figsize=(6.6, 4.8))
    
    # Plot data
    plt.scatter(raw_depths, raw_chlorides, label="Data (all)")
    plt.scatter(fit_depths, fit_chlorides, label="Data used for fit", marker="s")
    
    # Plot model
    plt.plot(x_fit_mm, y_fit, label="Model fit", linestyle="--")
    
    # Lines
    plt.axhline(Ci, linestyle=":", label="Ci (bulk)")
    
    if not np.isnan(x_alpha_mm):
        plt.axvline(x_alpha_mm, linestyle=":", label=f"xα (α={alpha:.2%}) ≈ {x_alpha_mm:.2f} mm")
        plt.scatter([x_alpha_mm], [target_conc], marker="x")
        
    if x_cross_mm is not None and not np.isnan(x_cross_mm):
        plt.axvline(x_cross_mm, linestyle="-.")
        plt.scatter([x_cross_mm], [Ci], marker="o")
        plt.text(x_cross_mm, Ci, f"  exp. cross ≈ {x_cross_mm:.2f} mm", va="bottom")

    # Text box
    texte = (
        f"Cs = {Cs:.5f} ± {Cs_std:.1e}\n"
        f"Dnss = {Dnss:.2e} ± {Dnss_std:.1e} m²/s\n"
        f"xα ≈ {x_alpha_mm:.2f} mm (α={alpha:.2%})"
    )
    plt.gca().text(0.02, 0.02, texte, fontsize=10, ha="left", va="bottom",
                   transform=plt.gca().transAxes, bbox=dict(facecolor="white", alpha=0.7))

    plt.xlabel("Depth (mm)")
    plt.ylabel("Chloride content")
    plt.legend(loc="upper right")
    plt.xlim(0, 25)
    plt.ylim(0, 1) # Might need adjustment if data goes higher, but original script had this
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close() # Close to free memory
