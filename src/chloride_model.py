
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv
from typing import Tuple, Optional, Union

def modele_diffusion(x: Union[float, NDArray[np.float64]], Cs: float, Dnss: float, Ci: float, t: float) -> Union[float, NDArray[np.float64]]:
    """
    Solution 1D semi-infinie (profil d'erreur) avec teneur de fond Ci.
    y(x) = Ci + (Cs - Ci) * (1 - erf( x / (2 * sqrt(Dnss * t)) ))
    
    x: depth in meters
    Cs: Surface concentration
    Dnss: Non-steady state diffusion coefficient (m^2/s)
    Ci: Initial concentration (background)
    t: time in seconds
    """
    # Note: The original script had `modele_diffusion(x, Cs, Dnss)` with Ci and t global.
    # Here we pass them as arguments or fixed parameters.
    # To use curve_fit properly, we might need a wrapper or lambda if Ci and t are fixed for the fit.
    return Ci + (Cs - Ci) * (1.0 - erf(x / (2.0 * np.sqrt(Dnss * t))))

def fit_chloride_profile(depths_m: NDArray[np.float64], chlorides: NDArray[np.float64], t_seconds: float, Ci: float = 0.0) -> Tuple[float, float, float, float, float]:
    """
    Fits the diffusion model to the data.
    
    depths_m: array of depths in meters (excluding 0 if needed, but this function just fits what it gets)
    chlorides: array of chloride concentrations
    t_seconds: exposure time in seconds
    Ci: Initial concentration (background)
    
    Returns: (Cs, Dnss, Cs_std, Dnss_std, R_squared)
    """
    
    # Wrapper for curve_fit that fixes Ci and t
    def model_to_fit(x: Union[float, NDArray[np.float64]], Cs: float, Dnss: float) -> Union[float, NDArray[np.float64]]:
        return modele_diffusion(x, Cs, Dnss, Ci, t_seconds)

    # Initial guesses from original script
    # p0 = [0.02, 1e-12]  # (Cs, Dnss) initial
    p0 = [0.02, 1e-12]
    
    try:
        params, cov = curve_fit(model_to_fit, depths_m, chlorides, p0=p0, maxfev=20000)
        Cs_opt, Dnss_opt = params
        if cov is not None:
            Cs_std, Dnss_std = np.sqrt(np.diag(cov))
        else:
            Cs_std, Dnss_std = np.nan, np.nan
        
        # Calculate R_squared
        residuals = chlorides - model_to_fit(depths_m, Cs_opt, Dnss_opt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((chlorides - np.mean(chlorides))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return Cs_opt, Dnss_opt, Cs_std, Dnss_std, r_squared
    except Exception as e:
        print(f"Fitting failed: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

def calculate_x_alpha(Dnss: float, t_seconds: float, alpha: float = 0.01) -> float:
    """
    Calculates the depth x_alpha where the concentration is Ci + alpha*(Cs - Ci).
    """
    if Dnss is None or np.isnan(Dnss) or Dnss <= 0:
        return np.nan
    z = erfinv(1.0 - alpha)
    x_alpha_m = 2.0 * np.sqrt(Dnss * t_seconds) * z
    return x_alpha_m

def interp_cross(depths_mm: NDArray[np.float64], chlorides: NDArray[np.float64], level: float) -> Optional[float]:
    """
    Première abscisse où le segment [i,i+1] croise 'level' (interp. linéaire).
    Original logic from the script.
    """
    depths_mm = np.array(depths_mm)
    chlorides = np.array(chlorides)
    
    for i in range(len(chlorides) - 1):
        y1, y2 = chlorides[i], chlorides[i + 1]
        x1, x2 = depths_mm[i], depths_mm[i + 1]
        if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
            w = (level - y1) / (y2 - y1)
            return x1 + w * (x2 - x1)
    return None
