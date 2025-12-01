"""
windowspec.py

SeedSpec Format - The "Genome" of ZetaDiffusion.
A SeedSpec contains center (where) + seed (how) - everything else is derived.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os
from pathlib import Path

@dataclass
class SeedSpec:
    """
    Minimal specification: center (where) + seed (how).
    All rules are derived deterministically from the seed.
    """
    center: float  # t_center - the "where"
    seed: float    # S - the "how" (operator)
    
    def to_dict(self):
        return asdict(self)
    
    def to_file(self, filename: str):
        """Save SeedSpec to JSON file in .out/ directory."""
        out_dir = Path(".out")
        out_dir.mkdir(exist_ok=True)
        filepath = out_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_file(cls, filename: str) -> 'SeedSpec':
        """Load SeedSpec from JSON file (checks .out/ first, then current directory)."""
        out_path = Path(".out") / filename
        if out_path.exists():
            filepath = out_path
        else:
            filepath = Path(filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Seed → Rule functions (deterministic operators)
def f_width(seed: float, t_range: float) -> float:
    """Seed → width: map via modular arithmetic."""
    return t_range * 0.1 * (1.0 + (seed % 1.0))

def f_res(seed: float, base_res: float = 0.01) -> float:
    """Seed → resolution: map via log scale."""
    # Map seed to resolution on log scale: 0.001 to 0.1
    log_min, log_max = np.log10(0.001), np.log10(0.1)
    log_res = log_min + (log_max - log_min) * ((seed * 0.618) % 1.0)  # Golden ratio spacing
    return 10 ** log_res

def f_norm(seed: float) -> str:
    """Seed → normalization: map via seed parity."""
    # 3 choices: peak-centered, minmax, zscore
    choice = int(seed * 1000) % 3
    return ["peak-centered", "minmax", "zscore"][choice]

def f_sym(seed: float) -> bool:
    """Seed → symmetry: map via seed bits."""
    return (int(seed * 1000) % 2) == 0

def f_steps(seed: float, min_steps: int = 3, max_steps: int = 10) -> int:
    """Seed → RG steps: map via seed mod N."""
    return min_steps + (int(seed * 1000) % (max_steps - min_steps + 1))

def expand_seed(seed: float, t_range: float) -> dict:
    """
    Expand seed into full rule set.
    Returns dict with: width, resolution, normalization, symmetry, rg_steps
    """
    return {
        'width': f_width(seed, t_range),
        'resolution': f_res(seed),
        'normalization': f_norm(seed),
        'symmetry': f_sym(seed),
        'rg_steps': f_steps(seed)
    }

def regenerate_from_spec(spec: SeedSpec, t_grid: Optional[np.ndarray] = None, xi_values: Optional[np.ndarray] = None):
    """
    Regenerate entire ZetaDiffusion analysis from SeedSpec alone.
    
    Expands seed → rules, then generates full analysis.
    If t_grid/xi_values not provided, generates them deterministically.
    """
    from zetadiffusion.renorm import Window, local_scan, find_bifurcations
    from zetadiffusion.dynamics import CircleMapExtractor
    from zetadiffusion.field import XiSampler
    
    # Expand seed into rules
    if t_grid is None or xi_values is None:
        # Generate initial data range
        t_range = 60.0  # Default range
        t_start = spec.center - t_range / 2
        t_end = spec.center + t_range / 2
    else:
        t_range = t_grid[-1] - t_grid[0]
    
    rules = expand_seed(spec.seed, t_range)
    
    # Generate data if not provided
    if t_grid is None or xi_values is None:
        n_points = int(t_range / rules['resolution'])
        sampler = XiSampler(t_start, t_end, n_points)
        sampler.sample()
        t_grid = sampler.t_grid
        xi_values = sampler.xi_values
    
    # Create Window using expanded rules
    window = Window(t_grid, xi_values, spec.center, rules['width'])
    
    # Extract map (deterministic from window)
    op = window.map
    
    # RG flow (deterministic from map)
    alphas = local_scan(t_grid, xi_values, spec.center, rules['width'])
    
    # Bundle dynamics (deterministic from window)
    extractor = CircleMapExtractor(t_grid, xi_values, spec.center, rules['width'])
    omegas, rhos = extractor.scan_devil_staircase(n_omegas=60, k=1.5)
    
    # Spectral probe (deterministic from window region)
    mask = window.mask
    window_xi = xi_values[mask]
    window_t = t_grid[mask]
    
    # Find zeros in window
    sampler = XiSampler(window_t[0], window_t[-1], len(window_t))
    sampler.t_grid = window_t
    sampler.xi_values = window_xi
    zeros = sampler.detect_zeros()
    
    return {
        'window': window,
        'alphas': alphas,
        'omegas': omegas,
        'rhos': rhos,
        'zeros': zeros,
        't_grid': window_t,
        'xi_values': window_xi,
        'rules': rules  # Include expanded rules for inspection
    }

