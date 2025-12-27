"""
Scattering Physics Utilities for PAID-SnowNet
Implements Equations (1)-(7) from the paper

This module provides physical models for light scattering in snow scenes:
- Rayleigh scattering for small particles (d << λ)
- Mie scattering for larger particles (d ~ λ)
- Combined model with automatic switching based on particle size
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy import special


class RayleighScattering(nn.Module):
    """
    Rayleigh Scattering Model for small particles (d << λ)
    Implements Equation (1): I ∝ λ^(-4)
    
    For particles much smaller than wavelength, intensity follows inverse
    fourth power law with wavelength.
    """
    
    def __init__(self, wavelength: float = 550e-9, refractive_index: float = 1.31):
        """
        Args:
            wavelength: Light wavelength in meters (default: 550nm green light)
            refractive_index: Particle refractive index (default: 1.31 for ice)
        """
        super().__init__()
        self.wavelength = wavelength
        self.n = refractive_index
        
        # Eq(1): Rayleigh scattering coefficient
        # K = (2π/λ)^4 * (n²-1)²/(n²+2)² * (d/2)^6
        self.register_buffer('k_wave', torch.tensor(2 * np.pi / wavelength))
        
    def forward(self, particle_diameter: torch.Tensor, 
                distance: torch.Tensor) -> torch.Tensor:
        """
        Compute Rayleigh scattering intensity.
        
        Eq(1): I_rayleigh = I_0 * (π²/2) * ((n²-1)/(n²+2))² * (d/λ)^4 * (1/r²)
        
        Args:
            particle_diameter: Particle diameter tensor [B, H, W] in meters
            distance: Distance from observer [B, H, W] in meters
            
        Returns:
            Scattered intensity [B, H, W]
        """
        n_sq = self.n ** 2
        polarizability_factor = ((n_sq - 1) / (n_sq + 2)) ** 2
        
        size_param = particle_diameter / self.wavelength
        
        # Eq(1): Intensity proportional to (d/λ)^4 and 1/r^2
        intensity = (np.pi ** 2 / 2) * polarizability_factor * \
                    (size_param ** 4) / (distance ** 2 + 1e-8)
        
        return intensity
    
    def scattering_cross_section(self, particle_diameter: torch.Tensor) -> torch.Tensor:
        """
        Compute Rayleigh scattering cross section.
        
        σ_s = (2π^5/3) * (d^6/λ^4) * ((n²-1)/(n²+2))²
        
        Args:
            particle_diameter: Particle diameter [B, H, W]
            
        Returns:
            Scattering cross section [B, H, W]
        """
        n_sq = self.n ** 2
        polarizability_factor = ((n_sq - 1) / (n_sq + 2)) ** 2
        
        sigma = (2 * np.pi ** 5 / 3) * \
                (particle_diameter ** 6 / self.wavelength ** 4) * \
                polarizability_factor
        
        return sigma


class MieScattering(nn.Module):
    """
    Mie Scattering Model for larger particles (d ~ λ)
    Implements Equations (2)-(7) from the paper
    
    Uses rigorous electromagnetic theory for spherical particles
    with size comparable to wavelength.
    """
    
    def __init__(self, wavelength: float = 550e-9, refractive_index: float = 1.31,
                 max_terms: int = 50):
        """
        Args:
            wavelength: Light wavelength in meters
            refractive_index: Complex refractive index of particle
            max_terms: Maximum terms in Mie series expansion
        """
        super().__init__()
        self.wavelength = wavelength
        self.n = refractive_index
        self.max_terms = max_terms
        
    def _compute_mie_coefficients(self, x: float, m: complex, 
                                   n_terms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mie coefficients a_n and b_n using Riccati-Bessel functions.
        
        Eq(3): a_n = [D_n(mx)/m + n/x] * ψ_n(x) - ψ_{n-1}(x)
                     ────────────────────────────────────────
                     [D_n(mx)/m + n/x] * ξ_n(x) - ξ_{n-1}(x)
        
        Eq(4): b_n = [m*D_n(mx) + n/x] * ψ_n(x) - ψ_{n-1}(x)
                     ────────────────────────────────────────
                     [m*D_n(mx) + n/x] * ξ_n(x) - ξ_{n-1}(x)
        
        Args:
            x: Size parameter 2πa/λ
            m: Relative refractive index
            n_terms: Number of terms
            
        Returns:
            Tuple of (a_n, b_n) coefficient arrays
        """
        a_n = np.zeros(n_terms, dtype=complex)
        b_n = np.zeros(n_terms, dtype=complex)
        
        mx = m * x
        
        # Compute Riccati-Bessel functions
        # ψ_n(x) = x * j_n(x), ξ_n(x) = x * h_n^(1)(x)
        for n in range(1, n_terms + 1):
            # Spherical Bessel functions
            jn_x = special.spherical_jn(n, x)
            jn_mx = special.spherical_jn(n, mx)
            yn_x = special.spherical_yn(n, x)
            
            jn_1_x = special.spherical_jn(n - 1, x) if n > 0 else np.sin(x) / x
            
            # Riccati-Bessel functions
            psi_n = x * jn_x
            psi_n_1 = x * jn_1_x if n > 0 else np.sin(x)
            xi_n = x * (jn_x + 1j * yn_x)
            xi_n_1 = x * (jn_1_x + 1j * special.spherical_yn(n - 1, x)) if n > 0 else \
                     (np.sin(x) - 1j * np.cos(x))
            
            # Logarithmic derivative D_n(mx) = d/d(mx)[ln(ψ_n(mx))]
            D_n = special.spherical_jn(n, mx, derivative=True) / \
                  (special.spherical_jn(n, mx) + 1e-10) * mx / mx
            
            # Eq(3) and Eq(4)
            numerator_a = (D_n / m + n / x) * psi_n - psi_n_1
            denominator_a = (D_n / m + n / x) * xi_n - xi_n_1
            
            numerator_b = (m * D_n + n / x) * psi_n - psi_n_1
            denominator_b = (m * D_n + n / x) * xi_n - xi_n_1
            
            a_n[n - 1] = numerator_a / (denominator_a + 1e-10)
            b_n[n - 1] = numerator_b / (denominator_b + 1e-10)
        
        return a_n, b_n
    
    def forward(self, particle_diameter: torch.Tensor,
                scattering_angle: torch.Tensor) -> torch.Tensor:
        """
        Compute Mie scattering intensity.
        
        Eq(2): I_mie(θ) = (λ²/4π²r²) * [|S_1(θ)|² + |S_2(θ)|²] / 2
        
        Args:
            particle_diameter: Particle diameter [B, H, W]
            scattering_angle: Scattering angle in radians [B, H, W]
            
        Returns:
            Scattered intensity [B, H, W]
        """
        # Convert to numpy for scipy functions
        d_np = particle_diameter.detach().cpu().numpy()
        theta_np = scattering_angle.detach().cpu().numpy()
        
        # Size parameter: x = πd/λ
        x = np.pi * d_np / self.wavelength
        
        # Number of terms: n_max = x + 4*x^(1/3) + 2
        n_max = int(np.max(x) + 4 * np.max(x) ** (1/3) + 2)
        n_max = min(n_max, self.max_terms)
        
        # Compute Mie coefficients for mean size parameter
        x_mean = np.mean(x)
        a_n, b_n = self._compute_mie_coefficients(x_mean, self.n, n_max)
        
        # Eq(5) and Eq(6): Amplitude functions S_1 and S_2
        # S_1(θ) = Σ (2n+1)/(n(n+1)) * [a_n*π_n(cosθ) + b_n*τ_n(cosθ)]
        # S_2(θ) = Σ (2n+1)/(n(n+1)) * [a_n*τ_n(cosθ) + b_n*π_n(cosθ)]
        
        cos_theta = np.cos(theta_np)
        S1 = np.zeros_like(cos_theta, dtype=complex)
        S2 = np.zeros_like(cos_theta, dtype=complex)
        
        for n in range(1, n_max + 1):
            # Eq(7): Angular functions π_n and τ_n
            # π_n(cosθ) = P_n^1(cosθ) / sinθ
            # τ_n(cosθ) = dP_n^1(cosθ)/dθ
            
            # Legendre polynomial derivatives
            P_n = special.lpmv(1, n, cos_theta)
            sin_theta = np.sin(theta_np) + 1e-10
            
            pi_n = P_n / sin_theta
            tau_n = n * cos_theta * pi_n - (n + 1) * \
                    special.lpmv(1, n - 1, cos_theta) / sin_theta if n > 1 else cos_theta
            
            coeff = (2 * n + 1) / (n * (n + 1))
            
            S1 += coeff * (a_n[n - 1] * pi_n + b_n[n - 1] * tau_n)
            S2 += coeff * (a_n[n - 1] * tau_n + b_n[n - 1] * pi_n)
        
        # Eq(2): Intensity
        intensity = (np.abs(S1) ** 2 + np.abs(S2) ** 2) / 2
        intensity = intensity * (self.wavelength ** 2) / (4 * np.pi ** 2)
        
        return torch.from_numpy(intensity).to(particle_diameter.device).float()
    
    def scattering_efficiency(self, particle_diameter: torch.Tensor) -> torch.Tensor:
        """
        Compute Mie scattering efficiency Q_sca.
        
        Q_sca = (2/x²) * Σ (2n+1) * (|a_n|² + |b_n|²)
        
        Args:
            particle_diameter: Particle diameter [B, H, W]
            
        Returns:
            Scattering efficiency [B, H, W]
        """
        d_np = particle_diameter.detach().cpu().numpy()
        x = np.pi * d_np / self.wavelength
        
        x_mean = np.mean(x)
        n_max = int(x_mean + 4 * x_mean ** (1/3) + 2)
        n_max = min(n_max, self.max_terms)
        
        a_n, b_n = self._compute_mie_coefficients(x_mean, self.n, n_max)
        
        Q_sca = 0
        for n in range(1, n_max + 1):
            Q_sca += (2 * n + 1) * (np.abs(a_n[n - 1]) ** 2 + np.abs(b_n[n - 1]) ** 2)
        
        Q_sca = 2 / (x_mean ** 2 + 1e-8) * Q_sca
        
        return torch.full_like(particle_diameter, Q_sca)


class CombinedScatteringModel(nn.Module):
    """
    Combined scattering model that automatically switches between
    Rayleigh and Mie regimes based on particle size parameter.
    
    - Rayleigh: x = 2πa/λ < 0.1 (small particles)
    - Mie: x >= 0.1 (larger particles)
    """
    
    def __init__(self, wavelength: float = 550e-9, refractive_index: float = 1.31,
                 transition_threshold: float = 0.1):
        """
        Args:
            wavelength: Light wavelength in meters
            refractive_index: Particle refractive index
            transition_threshold: Size parameter threshold for regime switching
        """
        super().__init__()
        self.wavelength = wavelength
        self.threshold = transition_threshold  # x = 2πa/λ threshold
        
        self.rayleigh = RayleighScattering(wavelength, refractive_index)
        self.mie = MieScattering(wavelength, refractive_index)
        
    def forward(self, particle_diameter: torch.Tensor,
                distance: torch.Tensor,
                scattering_angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scattering intensity using appropriate model.
        
        Args:
            particle_diameter: Particle diameter [B, H, W]
            distance: Distance from observer [B, H, W]
            scattering_angle: Optional scattering angle (for Mie)
            
        Returns:
            Scattered intensity [B, H, W]
        """
        # Size parameter
        x = np.pi * particle_diameter / self.wavelength
        
        # Create masks for each regime
        rayleigh_mask = (x < self.threshold).float()
        mie_mask = 1 - rayleigh_mask
        
        # Rayleigh contribution
        I_rayleigh = self.rayleigh(particle_diameter, distance)
        
        # Mie contribution (use forward scattering if angle not provided)
        if scattering_angle is None:
            scattering_angle = torch.zeros_like(particle_diameter)
        I_mie = self.mie(particle_diameter, scattering_angle)
        I_mie = I_mie / (distance ** 2 + 1e-8)
        
        # Combine with smooth transition
        intensity = rayleigh_mask * I_rayleigh + mie_mask * I_mie
        
        return intensity


def compute_scattering_coefficient_map(particle_density: torch.Tensor,
                                       particle_diameter: torch.Tensor,
                                       wavelength: float = 550e-9,
                                       refractive_index: float = 1.31) -> torch.Tensor:
    """
    Compute spatially-varying scattering coefficient β(x,y).
    
    β(x,y) = N(x,y) * σ_s(d)
    
    where N is particle number density and σ_s is scattering cross section.
    
    Args:
        particle_density: Number density map [B, H, W] in particles/m³
        particle_diameter: Diameter map [B, H, W] in meters
        wavelength: Light wavelength
        refractive_index: Particle refractive index
        
    Returns:
        Scattering coefficient map [B, H, W] in m⁻¹
    """
    rayleigh = RayleighScattering(wavelength, refractive_index)
    
    # Scattering cross section
    sigma_s = rayleigh.scattering_cross_section(particle_diameter)
    
    # Scattering coefficient
    beta = particle_density * sigma_s
    
    return beta


def compute_transmittance_map(scattering_coeff: torch.Tensor,
                              depth: torch.Tensor) -> torch.Tensor:
    """
    Compute transmittance map using Beer-Lambert law.
    
    t(x,y) = exp(-β(x,y) * d(x,y))
    
    Args:
        scattering_coeff: Scattering coefficient [B, H, W]
        depth: Optical depth [B, H, W]
        
    Returns:
        Transmittance map [B, H, W] in range [0, 1]
    """
    transmittance = torch.exp(-scattering_coeff * depth)
    return torch.clamp(transmittance, 0, 1)


# Initialize package
__all__ = [
    'RayleighScattering',
    'MieScattering', 
    'CombinedScatteringModel',
    'compute_scattering_coefficient_map',
    'compute_transmittance_map'
]
