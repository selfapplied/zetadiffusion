"""
renorm.py

Implements the Renormalization Operator and Feigenbaum delta estimation.
"""

import numpy as np
from functools import cached_property
from itertools import islice
from scipy.interpolate import UnivariateSpline
from typing import Callable, Iterator, List, Optional, Tuple

# Universal constants
DELTA_F = 4.66920160910299067185320382
ALPHA_F = -2.50290787509589282228390287

DiffusionMap = Callable[[float], float]

class RGOperator:
    """Recursive RG operator: f_{n+1}(x) = alpha * f_n(f_n(x/alpha))"""
    def __init__(self, base: DiffusionMap, alpha: float = 1.0, depth: int = 0, prev: Optional['RGOperator'] = None):
        self.base = base
        self.alpha = alpha
        self.depth = depth
        self.prev = prev

    def __call__(self, x: float) -> float:
        if self.prev is None:
            return self.base(x)
        if np.abs(self.alpha) < 1e-10:
            return ALPHA_F * self.prev(self.prev(x / ALPHA_F))
        inner = x / self.alpha
        return self.alpha * self.prev(self.prev(inner))
    
    def __iter__(self) -> Iterator['RGOperator']:
        """Iterates the RG operator, computing alpha from f(1) at each step."""
        current = self
        while True:
            val_at_1 = current(1.0)
            alpha = ALPHA_F if np.abs(val_at_1) < 1e-6 else 1.0 / val_at_1
            current = RGOperator(base=self.base, alpha=alpha, depth=current.depth + 1, prev=current)
            yield current

def spline_map(x: np.ndarray, y: np.ndarray) -> RGOperator:
    """Creates an RGOperator from spline-fitted data."""
    unique_x, inverse_idx = np.unique(x, return_inverse=True)
    unique_y = np.array([np.mean(y[inverse_idx == i]) for i in range(len(unique_x))])
    
    spline = UnivariateSpline(unique_x, unique_y, k=min(3, len(unique_x)-1), s=0, ext=0)
    def f(x_val: float) -> float:
        return float(spline(x_val))
    def f_sym(x_val: float) -> float:
        return 0.5 * (f(x_val) + f(-x_val))
    return RGOperator(base=f_sym, alpha=1.0, depth=0)

class Window:
    """A window of data points around a center point."""
    def __init__(self, t_grid: np.ndarray, xi_values: np.ndarray, t_center: float, width: float):
        self.t_grid = t_grid
        self.xi_values = xi_values
        self.t_center = t_center
        self.width = width

    @property
    def mask(self) -> np.ndarray:
        return np.abs(self.t_grid - self.t_center) <= self.width / 2

    @property
    def t(self) -> np.ndarray:
        return self.t_grid[self.mask]

    @property
    def xi(self) -> np.ndarray:
        return self.xi_values[self.mask]

    @cached_property
    def peak_idx(self) -> int:
        return int(np.argmax(np.abs(self.xi)))

    @property
    def peak_t(self) -> float:
        return self.t[self.peak_idx]

    @property
    def peak_xi(self) -> float:
        return np.abs(self.xi[self.peak_idx])

    @property
    def x_norm(self) -> np.ndarray:
        return (self.t - self.peak_t) / (self.width / 2.0)

    @property
    def y_norm(self) -> np.ndarray:
        raw_y = -self.xi if self.xi[self.peak_idx] < 0 else self.xi
        return raw_y / self.peak_xi

    @property
    def precision(self) -> float:
        """Grid spacing precision (like continued fraction resolution)."""
        if len(self.t) < 2:
            return self.width
        return np.mean(np.diff(np.sort(self.t)))

    @property
    def steps(self) -> int:
        """Maximum RG steps needed to reach window precision (continued fraction convergence)."""
        # RG converges as delta^(-n), so n ~ log(precision) / log(delta)
        prec = max(self.precision, 1e-10)  # Safety: avoid log(0)
        return int(np.ceil(-np.log10(prec) / np.log10(DELTA_F))) + 2

    @property
    def tolerance(self) -> float:
        """Tolerance based on window precision."""
        return self.precision * 0.1  # 10% of grid spacing

    @cached_property
    def map(self) -> RGOperator:
        if len(self.xi) < 7:
            return RGOperator(base=lambda x: 0.0, alpha=1.0, depth=0)
            
        x_sym = np.concatenate([self.x_norm, -self.x_norm])
        y_sym = np.concatenate([self.y_norm, self.y_norm])
        
        sort_idx = np.argsort(x_sym)
        return spline_map(x_sym[sort_idx], y_sym[sort_idx])

def local_scan(t_grid: np.ndarray, xi_values: np.ndarray, t_center: float, width: float) -> List[float]:
    """Scans the local RG operator at a given center and width."""
    window = Window(t_grid, xi_values, t_center, width)
    return [op.alpha for op in islice(window.map, window.steps)]

def find_peaks(t_grid: np.ndarray, xi_values: np.ndarray, t_min: float, t_max: float) -> List[float]:
    """Identifies local maxima of |Xi| in the range."""
    window = Window(t_grid, xi_values, (t_min + t_max) / 2, t_max - t_min)
    if len(window.xi) < 3:
        return []
    abs_xi = np.abs(window.xi)
    is_peak = (abs_xi[1:-1] > abs_xi[:-2]) & (abs_xi[1:-1] > abs_xi[2:])
    return window.t[np.where(is_peak)[0] + 1].tolist()

def find_bifurcations(t_grid: np.ndarray, xi_values: np.ndarray, t_min: float, t_max: float, width: float = 5.0) -> List[Tuple[float, float]]:
    """Finds bifurcation points (peaks) and their Feigenbaum scaling factors."""
    peaks = find_peaks(t_grid, xi_values, t_min, t_max)
    results = []
    for t_peak in peaks:
        alphas = local_scan(t_grid, xi_values, t_peak, width)
        results.append((t_peak, alphas[-1] if alphas else np.nan))
    return results
