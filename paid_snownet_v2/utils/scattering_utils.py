"""
Scattering utilities for SPM module.
Implements Mie scattering (Eq. 2-7) and Rayleigh scattering (Eq. 1).
File: utils/scattering_utils.py
"""

import numpy as np
import torch
import torch.nn.functional as F


def rayleigh_scattering_intensity(I0, wavelength, diameter, distance, refractive_index):
    """
    Compute Rayleigh scattering intensity (Equation 1).
    
    I_Rayleigh = I0 * (pi^2 * d^6) / (2 * R^2 * lambda^4) * ((n^2-1)/(n^2+2))^2
    
    Args:
        I0: Incident light intensity (normalized to 1.0)
        wavelength: Wavelength in meters (380nm-780nm visible)
        diameter: Particle diameter in meters (0.01um-0.1um for Rayleigh)
        distance: Observation distance in meters (1m-100m)
        refractive_index: Relative refractive index (~1.31 for ice)
    Returns:
        Rayleigh scattering intensity
    """
    n = refractive_index
    n_factor = ((n ** 2 - 1) / (n ** 2 + 2)) ** 2
    intensity = I0 * (np.pi ** 2 * diameter ** 6) / (2 * distance ** 2 * wavelength ** 4) * n_factor
    return intensity


def riccati_bessel_psi(n_order, rho):
    """
    Compute Riccati-Bessel function psi_n(rho) using recurrence (Eq. 5, 7).
    
    psi_n(rho) = rho * j_n(rho)
    Initial: psi_0 = sin(rho), psi_1 = sin(rho)/rho - cos(rho)
    Recurrence: psi_{n+1} = (2n+1)/rho * psi_n - psi_{n-1}
    """
    if isinstance(rho, (int, float)):
        rho = np.array([rho], dtype=np.float64)
    
    psi_prev = np.sin(rho)          # psi_0
    if n_order == 0:
        return psi_prev
    
    psi_curr = np.sin(rho) / rho - np.cos(rho)  # psi_1
    if n_order == 1:
        return psi_curr
    
    for n in range(1, n_order):
        psi_next = (2 * n + 1) / rho * psi_curr - psi_prev
        psi_prev = psi_curr
        psi_curr = psi_next
    
    return psi_curr


def riccati_bessel_chi(n_order, rho):
    """
    Compute Riccati-Bessel function chi_n(rho) = -rho * y_n(rho).
    Initial: chi_0 = cos(rho), chi_1 = cos(rho)/rho + sin(rho)
    """
    if isinstance(rho, (int, float)):
        rho = np.array([rho], dtype=np.float64)
    
    chi_prev = np.cos(rho)
    if n_order == 0:
        return chi_prev
    
    chi_curr = np.cos(rho) / rho + np.sin(rho)
    if n_order == 1:
        return chi_curr
    
    for n in range(1, n_order):
        chi_next = (2 * n + 1) / rho * chi_curr - chi_prev
        chi_prev = chi_curr
        chi_curr = chi_next
    
    return chi_curr


def riccati_bessel_xi(n_order, rho):
    """
    Compute xi_n = psi_n + i*chi_n (Eq. 6).
    zeta_n(rho) = rho * h_n^(1)(rho) = psi_n + i*chi_n
    """
    return riccati_bessel_psi(n_order, rho) + 1j * riccati_bessel_chi(n_order, rho)


def derivative_psi(n_order, rho, psi_n=None, psi_nm1=None):
    """Derivative of psi_n using recurrence."""
    if psi_n is None:
        psi_n = riccati_bessel_psi(n_order, rho)
    if psi_nm1 is None:
        if n_order == 0:
            psi_nm1 = np.zeros_like(rho)
        else:
            psi_nm1 = riccati_bessel_psi(n_order - 1, rho)
    return psi_nm1 - n_order / rho * psi_n


def derivative_xi(n_order, rho):
    """Derivative of xi_n."""
    psi_d = derivative_psi(n_order, rho)
    chi_n = riccati_bessel_chi(n_order, rho)
    if n_order == 0:
        chi_nm1 = np.zeros_like(rho)
    else:
        chi_nm1 = riccati_bessel_chi(n_order - 1, rho)
    chi_d = chi_nm1 - n_order / rho * chi_n
    return psi_d + 1j * chi_d


def mie_coefficients(n_order, x, m):
    """
    Compute Mie coefficients a_n and b_n (Equations 3, 4).
    
    a_n = [psi_n(mx)*psi'_n(x) - m*psi'_n(mx)*psi_n(x)] /
          [psi_n(mx)*xi'_n(x) - m*psi'_n(mx)*xi_n(x)]
    
    b_n = [m*psi'_n(mx)*psi_n(x) - psi_n(mx)*psi'_n(x)] /
          [m*psi'_n(mx)*xi_n(x) - psi_n(mx)*xi'_n(x)]
    
    Args:
        n_order: Order n
        x: Size parameter x = 2*pi*a/lambda
        m: Relative refractive index m = n_particle/n_medium
    """
    mx = m * x
    
    psi_n_mx = riccati_bessel_psi(n_order, mx)
    psi_n_x = riccati_bessel_psi(n_order, x)
    dpsi_mx = derivative_psi(n_order, mx)
    dpsi_x = derivative_psi(n_order, x)
    xi_n_x = riccati_bessel_xi(n_order, x)
    dxi_x = derivative_xi(n_order, x)
    
    # Equation 3
    a_n_num = psi_n_mx * dpsi_x - m * dpsi_mx * psi_n_x
    a_n_den = psi_n_mx * dxi_x - m * dpsi_mx * xi_n_x
    a_n = a_n_num / a_n_den
    
    # Equation 4
    b_n_num = m * dpsi_mx * psi_n_x - psi_n_mx * dpsi_x
    b_n_den = m * dpsi_mx * xi_n_x - psi_n_mx * dxi_x
    b_n = b_n_num / b_n_den
    
    return a_n, b_n


def mie_scattering_intensity(theta, x, m, n_max=None):
    """
    Compute Mie scattering intensity (Equation 2).
    
    I_Mie(theta) = (|i1|^2 + |i2|^2) / (2*k^2*r^2)
    
    Series truncated at n_max = x + 4*x^(1/3) + 2
    
    Args:
        theta: Scattering angle in radians
        x: Size parameter
        m: Relative refractive index
        n_max: Maximum order (auto-computed if None)
    """
    if n_max is None:
        n_max = int(x + 4 * x ** (1 / 3) + 2)
    n_max = max(n_max, 2)
    n_max = min(n_max, 200)  # Cap for computational feasibility
    
    i1 = np.zeros_like(theta, dtype=complex)
    i2 = np.zeros_like(theta, dtype=complex)
    
    for n in range(1, n_max + 1):
        a_n, b_n = mie_coefficients(n, x, m)
        factor = (2 * n + 1) / (n * (n + 1))
        # Simplified angular functions
        i1 += factor * (a_n + b_n)
        i2 += factor * (a_n + b_n)
    
    intensity = (np.abs(i1) ** 2 + np.abs(i2) ** 2) / 2.0
    return intensity


def compute_scattering_map(image_tensor, wavelength=550e-9, refractive_index=1.31):
    """
    Compute a scattering coefficient map for an image tensor.
    Used by SPM module to estimate per-pixel scattering effects.
    
    Args:
        image_tensor: (B, C, H, W) tensor
        wavelength: Reference wavelength (default 550nm green)
        refractive_index: Ice refractive index (~1.31)
    Returns:
        scattering_map: (B, 1, H, W) estimated scattering coefficients
    """
    B, C, H, W = image_tensor.shape
    # Use intensity as proxy for scattering estimation
    intensity = image_tensor.mean(dim=1, keepdim=True)
    
    # Rayleigh lambda^-4 dependence creates wavelength-dependent scattering
    # Normalize to [0, 1] range for network processing
    scattering_map = 1.0 - intensity  # Higher scattering in brighter (snow) regions
    return scattering_map


def compute_transmittance_map(depth_map, scattering_coeff):
    """
    Compute transmittance map: t(x,y) = exp(-beta(x,y) * d(x,y))
    
    Args:
        depth_map: (B, 1, H, W) estimated depth
        scattering_coeff: (B, 1, H, W) scattering coefficient beta
    Returns:
        transmittance: (B, 1, H, W) in [0, 1]
    """
    transmittance = torch.exp(-scattering_coeff * depth_map)
    return transmittance.clamp(0.0, 1.0)
