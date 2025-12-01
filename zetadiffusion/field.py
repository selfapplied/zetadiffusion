"""
field.py

Implements the Riemann field equations, the completed zeta function xi(s),
and the Spectral Line Probe (XiSampler).
"""

import numpy as np
from scipy.special import gamma, zeta as riemann_zeta
from scipy.optimize import brentq

def xi(s):
    """
    Computes the completed Riemann zeta function xi(s).
    
    xi(s) = 1/2 * s * (s - 1) * pi^(-s/2) * Gamma(s/2) * zeta(s)
    """
    # Handle singularity at s=1 by returning limit value if hit exactly (unlikely with floats)
    # But xi is entire, so we rely on the cancellation.
    val = 0.5 * s * (s - 1) * np.power(np.pi, -0.5 * s) * gamma(0.5 * s) * riemann_zeta(s)
    return val

def critical_line_field(t):
    """
    Evaluates the field along the critical line s = 0.5 + i*t.
    Returns real part since xi is theoretically real on critical line.
    """
    s = 0.5 + 1j * t
    return np.real(xi(s))

class XiSampler:
    """
    Spectral Line Probe for sampling Xi(t) and detecting zeros.
    """
    def __init__(self, t_min, t_max, n_points):
        self.t_min = t_min
        self.t_max = t_max
        self.n_points = n_points
        self.t_grid = np.linspace(t_min, t_max, n_points)
        self.xi_values = None
        self.zeros = []
        
    def sample(self):
        """Samples Xi(t) on the configured grid."""
        s_vals = 0.5 + 1j * self.t_grid
        # xi(s) should be real on critical line
        self.xi_values = np.real(xi(s_vals))
        return self.xi_values

    def detect_zeros(self, threshold=1e-3):
        """
        Detects candidate zeros via sign changes and local minima.
        """
        if self.xi_values is None:
            self.sample()
            
        self.zeros = []
        
        # Sign changes
        signs = np.sign(self.xi_values)
        crossings = np.where(np.diff(signs))[0]
        
        for idx in crossings:
            # Linear interpolation for initial guess
            t1, t2 = self.t_grid[idx], self.t_grid[idx+1]
            y1, y2 = self.xi_values[idx], self.xi_values[idx+1]
            # Secant method approximation
            t_zero = t1 - y1 * (t2 - t1) / (y2 - y1)
            self.zeros.append((t_zero, 0.0)) # Store (t, val)
            
        return self.zeros

    def refine_zeros(self):
        """
        Refines detected zeros using root finding (Brent's method).
        """
        refined = []
        for t_est, _ in self.zeros:
            # bracket around estimate
            delta = (self.t_max - self.t_min) / self.n_points
            try:
                root = brentq(critical_line_field, t_est - delta, t_est + delta)
                refined.append((root, critical_line_field(root)))
            except ValueError:
                # If bracketing fails, keep estimate
                refined.append((t_est, critical_line_field(t_est)))
        self.zeros = refined
        return self.zeros

    def get_window(self, t_center, width):
        """
        Returns a slice of the data centered around t_center.
        """
        if self.xi_values is None:
            return None, None
        
        mask = np.abs(self.t_grid - t_center) <= width / 2
        return self.t_grid[mask], self.xi_values[mask]

def symmetry_error(s):
    return np.abs(xi(s) - xi(1 - s))

def noether_charge(s):
    return np.abs(xi(s))
