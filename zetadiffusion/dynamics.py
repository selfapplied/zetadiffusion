"""
dynamics.py

Formalizes bundle dynamics on X x S^1, circle map extraction,
and rotation number calculations using simplified modular types.
"""

import numpy as np
from scipy.interpolate import interp1d
from .types import to_radians, to_turns, Radians, Turns

class BundleSystem:
    """
    Represents the dynamical system on the bundle X x S^1.
    """
    def __init__(self, base_map, fiber_map_factory):
        self.base_map = base_map
        self.fiber_map = fiber_map_factory

    def step(self, x, theta):
        x_next = self.base_map(x)
        theta_next = self.fiber_map(x, theta)
        return x_next, theta_next

class CircleMapExtractor:
    """
    Extracts circle map dynamics from local field data.
    """
    def __init__(self, t_grid, xi_values, t_center, window_width):
        self.t_grid = t_grid
        self.xi_values = xi_values
        self.t_center = t_center
        self.width = window_width
        
        # Extract local window
        mask = np.abs(t_grid - t_center) <= window_width / 2
        self.local_t = t_grid[mask]
        self.local_xi = xi_values[mask]
        
        # Normalize Phi to be a perturbation
        if len(self.local_xi) > 0:
            min_val, max_val = np.min(self.local_xi), np.max(self.local_xi)
            if max_val > min_val:
                self.normalized_xi = (self.local_xi - min_val) / (max_val - min_val) - 0.5
            else:
                self.normalized_xi = np.zeros_like(self.local_xi)
        else:
            self.normalized_xi = np.array([0.0])

    def get_phi(self, k_coupling=1.0):
        """
        Returns a function Phi(theta) derived from the local Xi window.
        Phi takes Radians (as float-like), returns Radians perturbation.
        """
        if len(self.local_t) < 2:
            return lambda theta: 0.0
            
        # Interpolate normalized Xi over the window [0, 2pi]
        interp = interp1d(np.linspace(0, 2*np.pi, len(self.normalized_xi)), 
                          self.normalized_xi, 
                          kind='linear', 
                          fill_value="extrapolate")
        
        def phi(theta_val):
            # Expects raw float/Radians
            # Periodic wrapping
            theta_wrapped = np.mod(theta_val, 2*np.pi)
            return k_coupling * interp(theta_wrapped)
            
        return phi

    def rotation_number(self, omega, k_coupling=1.0, steps=1000, burn_in=100):
        """
        Computes rotation number rho(omega) for the extracted map.
        Input omega in Radians per step.
        Returns rho in Turns per step.
        """
        # Input omega is assumed to be in Radians per step
        # Ensure we are treating it as a float for the loop
        omega_val = float(omega)
        
        phi_func = self.get_phi(k_coupling)
        
        current = 0.0
        
        # Burn-in
        for _ in range(burn_in):
            pert = phi_func(current)
            current = current + omega_val + pert
            
        # Measurement (using lift)
        start_theta = current
        
        for _ in range(steps):
            pert = phi_func(current)
            current = current + omega_val + pert
            
        total_rotation_rad = current - start_theta
        
        # Convert total radians to turns
        # rho = (Total Radians / steps) / (2pi)
        rho_turns = (total_rotation_rad / steps) / (2 * np.pi)
        
        # Return explicit Turns type
        return to_turns(rho_turns)

    def scan_devil_staircase(self, omega_min=0, omega_max=1, n_omegas=100, k=0.5):
        """
        Generates the rho(omega) curve.
        Omega inputs here are treated as Turns (0..1).
        """
        omegas_turns = np.linspace(omega_min, omega_max, n_omegas)
        rhos = []
        for w_turn in omegas_turns:
            # Convert turn omega to radians for the map
            w_rad = w_turn * 2 * np.pi
            # Pass explicit Radians if we wanted, but float value suffices for loop
            rhos.append(self.rotation_number(w_rad, k_coupling=k))
        
        # Map rhos to Turns explicitly? 
        # The list contains Turns objects now.
        # But for plotting we likely want raw values or mixed array.
        # If we return a numpy array of Turns objects it might be awkward.
        # Let's return raw floats for the array, but the types are preserved internally.
        return omegas_turns, np.array([float(r) for r in rhos])

def rotation_number(fiber_map_x, theta0=0.0, limit_n=1000):
    """Legacy helper."""
    theta = theta0
    for _ in range(limit_n):
        theta = fiber_map_x(theta)
    return (theta - theta0) / (limit_n)
