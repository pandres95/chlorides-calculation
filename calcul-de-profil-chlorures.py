# -*- coding: utf-8 -*-
"""
Ajustement d'un profil de chlorures (solution erf), calcul de x_alpha,
détection du croisement expérimental avec Ci, tracé + export PNG/Excel.

Dépendances : numpy, scipy, matplotlib, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv

# =========================
# === DONNÉES EXPÉRIMENTALES
# =========================
profondeur_mm_total = np.array([0, 0.98, 1.60, 2.32, 4.02, 5.64, 7.74])
chlorures_total     = np.array([0.58995,
0.64675,
0.62435,
0.4327,
0.3089,
0.2282,
0.16595]
)

# Retirer le premier point (0 mm) pour l'ajustement
profondeur_mm = profondeur_mm_total[1:]
chlorures     = chlorures_total[1:]

# Conversion mm -> m pour l'ajustement
profondeur_m = profondeur_mm / 1000.0

# =========================
# === PARAMÈTRES PHYSIQUES
# =========================
Ci     = 0.01370         # teneur de fond (bulk)
t_days = 34                 # durée d'exposition (jours)
t      = t_days * 24 * 3600 # en secondes
ALPHA  = 0.01               # 1 % au-dessus de Ci pour définir x_alpha

# Fichiers de sortie
FIG_PATH   = "profil_chlorures_fit_34j_setR.png"
EXCEL_PATH = "resultats_chlorures_34j_setR.xlsx"

# =========================
# === MODÈLE & AJUSTEMENT
# =========================
def modele_diffusion(x, Cs, Dnss):
    """
    Solution 1D semi-infinie (profil d'erreur) avec teneur de fond Ci.
    y(x) = Ci + (Cs - Ci) * (1 - erf( x / (2 * sqrt(Dnss * t)) ))
    """
    return Ci + (Cs - Ci) * (1.0 - erf(x / (2.0 * np.sqrt(Dnss * t))))

# ⚠️ Logique de fittage identique à l'autre script : mêmes p0, PAS de bornes
p0 = [0.02, 1e-12]  # (Cs, Dnss) initial
params, cov = curve_fit(modele_diffusion, profondeur_m, chlorures, p0=p0, maxfev=20000)
Cs_opt, Dnss_opt = params
Cs_std, Dnss_std = np.sqrt(np.diag(cov)) if cov is not None else (np.nan, np.nan)

# =========================
# === PROFONDEUR x_alpha & CROISEMENT EXP.
# =========================
# Définition : y(x_alpha) = Ci + ALPHA*(Cs - Ci)
z = erfinv(1.0 - ALPHA)                 # car erf(z) = 1 - ALPHA
x_alpha_m  = 2.0 * np.sqrt(Dnss_opt * t) * z
x_alpha_mm = x_alpha_m * 1000.0
target_conc = Ci + ALPHA * (Cs_opt - Ci)

def interp_cross(x_mm, y, level):
    """Première abscisse où le segment [i,i+1] croise 'level' (interp. linéaire)."""
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        x1, x2 = x_mm[i], x_mm[i + 1]
        if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
            w = (level - y1) / (y2 - y1)
            return x1 + w * (x2 - x1)
    return None

x_cross_exp_mm = interp_cross(profondeur_mm_total, chlorures_total, Ci)

# =========================
# === COURBE AJUSTÉE & TRACÉ
# =========================
x_fit_m  = np.linspace(0, 0.016, 400)    # 0–16 mm
y_fit    = modele_diffusion(x_fit_m, Cs_opt, Dnss_opt)
x_fit_mm = x_fit_m * 1000.0

plt.figure(figsize=(6.6, 4.8))
plt.scatter(profondeur_mm_total, chlorures_total, label="Data (all)")
plt.scatter(profondeur_mm, chlorures, label="Data used for fit", marker="s")
plt.plot(x_fit_mm, y_fit, label="Model fit", linestyle="--")
plt.axhline(Ci, linestyle=":", label="Ci (bulk)")
plt.axvline(x_alpha_mm, linestyle=":", label=f"xα (α={ALPHA:.2%}) ≈ {x_alpha_mm:.2f} mm")
plt.scatter([x_alpha_mm], [target_conc], marker="x")
if x_cross_exp_mm is not None:
    plt.axvline(x_cross_exp_mm, linestyle="-.")
    plt.scatter([x_cross_exp_mm], [Ci], marker="o")
    plt.text(x_cross_exp_mm, Ci, f"  exp. cross ≈ {x_cross_exp_mm:.2f} mm", va="bottom")

# Encart paramètres
texte = (
    f"Cs = {Cs_opt:.5f} ± {Cs_std:.1e}\n"
    f"Dnss = {Dnss_opt:.2e} ± {Dnss_std:.1e} m²/s\n"
    f"xα ≈ {x_alpha_mm:.2f} mm (α={ALPHA:.2%})"
)
plt.gca().text(0.02, 0.02, texte, fontsize=10, ha="left", va="bottom",
               transform=plt.gca().transAxes, bbox=dict(facecolor="white", alpha=0.7))

plt.xlabel("Depth (mm)")
plt.ylabel("Chloride content")
plt.legend(loc="upper right")
plt.xlim(0, 25)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.show()

# =========================
# === EXPORT EXCEL
# =========================
df_params = pd.DataFrame({
    "Ci": [Ci],
    "Cs_opt": [Cs_opt],
    "Cs_std": [Cs_std],
    "Dnss_opt (m²/s)": [Dnss_opt],
    "Dnss_std (m²/s)": [Dnss_std],
    "t_days": [t_days],
    "alpha": [ALPHA],
    "x_alpha_mm": [x_alpha_mm],
    "x_cross_exp_mm": [x_cross_exp_mm],
})
df_curve = pd.DataFrame({"x_mm": x_fit_mm, "y_fit": y_fit})
df_raw   = pd.DataFrame({"x_mm_total": profondeur_mm_total,
                         "chlorures_total": chlorures_total})

with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as writer:
    df_params.to_excel(writer, index=False, sheet_name="params")
    df_curve.to_excel(writer, index=False, sheet_name="fit_curve")
    df_raw.to_excel(writer, index=False, sheet_name="raw_data")

# =========================
# === LOG
# =========================
print("\n=== RÉSULTATS ===")
print(f"Cs (surface) : {Cs_opt:.5f} (± {Cs_std:.1e})")
print(f"Dnss (m²/s)  : {Dnss_opt:.3e} (± {Dnss_std:.1e})")
print(f"x_alpha (mm) : {x_alpha_mm:.2f}  (α={ALPHA:.2%})")
print("Croisement exp. avec Ci :", "non atteint" if x_cross_exp_mm is None else f"~ {x_cross_exp_mm:.2f} mm")
print(f"Figure : {FIG_PATH}")
print(f"Excel  : {EXCEL_PATH}")
