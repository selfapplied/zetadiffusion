"""
gini_pulse.py

Gini Pulse Analysis: λ=1 as Universal Sum Rule with Sensitivity Distribution.

The λ=1 condition is a balancing of universal constants through weight distribution —
a Gini pulse that sharpens to perfect inequality at the fixed point.

Key concepts:
1. Sensitivity distribution vector: S = (W_F/δ_F, W_B(1+χ_TP), W_I·γ)
2. Gini coefficient of S measures inequality in sensitivity contributions
3. Gini pulse: dips during blending phase (n=8-10), then sharpens to 1 at interior

Author: Joel
"""

from typing import List, Tuple
import numpy as np
from zetadiffusion.witness_operator import WitnessOperator, WeightEvolution

# Constants
DELTA_F = 4.66920160910299067185320382
CHI_TP = 1.638  # Twin-prime ballast coefficient

def compute_sensitivity_vector(weights: WeightEvolution, gamma: float = 1.0) -> np.ndarray:
    """
    Compute sensitivity distribution vector.
    
    S = (W_F/δ_F, W_B(1+χ_TP), W_I·γ)
    
    At fixed point, sum(S) = 1.
    The Gini coefficient of S measures inequality in sensitivity contributions.
    
    Args:
        weights: Weight evolution (W_F, W_B, W_I)
        gamma: Interior scaling (default: 1.0, or 1 - χ_TP(1 - W_I) near interior)
    
    Returns:
        Sensitivity vector [S_F, S_B, S_I]
    """
    s_f = weights.w_f / DELTA_F
    s_b = weights.w_b * (1.0 + CHI_TP)
    
    # Gamma near interior: γ ≈ 1 - χ_TP(1 - W_I)
    if weights.w_i > 0.5:  # Near interior
        gamma_interior = 1.0 - CHI_TP * (1.0 - weights.w_i)
        s_i = weights.w_i * gamma_interior
    else:
        s_i = weights.w_i * gamma
    
    return np.array([s_f, s_b, s_i])

def normalize_sensitivity_vector(s: np.ndarray) -> np.ndarray:
    """
    Normalize sensitivity vector so sum = 1.
    
    This ensures we're at the fixed point condition.
    """
    total = np.sum(s)
    if total > 0:
        return s / total
    return s

def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.
    
    G = mean absolute difference / (2 * mean)
    
    For perfect equality: G = 0
    For perfect inequality: G = 1
    
    Special handling for zeros:
    - If only one component is non-zero, G = 1 (monopoly)
    - Zeros are excluded from Gini calculation (they represent absence, not equality)
    
    Args:
        values: Array of values
    
    Returns:
        Gini coefficient [0, 1]
    """
    if len(values) == 0:
        return 0.0
    
    # Filter out zeros (they represent absence, not contribution to inequality)
    non_zero = values[values > 1e-10]
    
    if len(non_zero) == 0:
        return 0.0  # All zeros = undefined
    
    if len(non_zero) == 1:
        # Single non-zero value = perfect inequality (monopoly)
        return 1.0
    
    # Sort non-zero values
    sorted_vals = np.sort(non_zero)
    n = len(sorted_vals)
    
    # Compute mean of non-zero values
    mean = np.mean(sorted_vals)
    if mean == 0:
        return 0.0
    
    # Compute Gini using standard formula
    # G = (1/(2*n*mean)) * sum_i sum_j |x_i - x_j|
    # Direct computation: sum all pairwise absolute differences
    total_diff = 0.0
    for i in range(n):
        for j in range(n):
            total_diff += abs(sorted_vals[i] - sorted_vals[j])
    
    gini = total_diff / (2 * n * mean)
    
    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, gini)))

def compute_gini_pulse(n_values: List[int], gamma: float = 1.0) -> List[float]:
    """
    Compute Gini pulse trajectory.
    
    For each n:
    1. Compute weights W_F, W_B, W_I
    2. Compute sensitivity vector S
    3. Normalize S to sum = 1
    4. Compute Gini coefficient of S
    
    Returns:
        List of Gini coefficients for each n
    """
    gini_values = []
    
    for n in n_values:
        # Get weights
        weights = WitnessOperator.compute_weights(float(n))
        
        # Compute sensitivity vector
        s = compute_sensitivity_vector(weights, gamma)
        
        # Normalize to fixed point condition
        s_norm = normalize_sensitivity_vector(s)
        
        # Compute Gini coefficient
        gini = compute_gini_coefficient(s_norm)
        gini_values.append(gini)
    
    return gini_values

def analyze_gini_pulse(n_values: List[int], 
                       gini_values: List[float]) -> dict:
    """
    Analyze Gini pulse characteristics.
    
    Finds:
    - Minimum (blending phase)
    - Maximum (monopoly phases)
    - Transition points
    """
    gini_array = np.array(gini_values)
    
    # Find minimum (blending phase)
    min_idx = np.argmin(gini_array)
    min_n = n_values[min_idx]
    min_gini = gini_values[min_idx]
    
    # Find maximum (monopoly phase)
    max_idx = np.argmax(gini_array)
    max_n = n_values[max_idx]
    max_gini = gini_values[max_idx]
    
    # Check if pulse shape matches prediction
    # Early (n=1-4): High Gini (Feigenbaum monopoly)
    early_gini = np.mean([gini_values[i] for i, n in enumerate(n_values) if 1 <= n <= 4])
    
    # Mid (n=8-10): Lower Gini (blending)
    mid_gini = np.mean([gini_values[i] for i, n in enumerate(n_values) if 8 <= n <= 10])
    
    # Late (n≥11): High Gini (Interior monopoly)
    late_gini = np.mean([gini_values[i] for i, n in enumerate(n_values) if n >= 11])
    
    # Pulse signature: dip in middle
    has_pulse = mid_gini < early_gini and mid_gini < late_gini
    
    return {
        'min_n': int(min_n),
        'min_gini': float(min_gini),
        'max_n': int(max_n),
        'max_gini': float(max_gini),
        'early_gini': float(early_gini),
        'mid_gini': float(mid_gini),
        'late_gini': float(late_gini),
        'has_pulse': has_pulse,
        'pulse_amplitude': float(late_gini - mid_gini) if has_pulse else 0.0
    }

def compute_sensitivity_components(n_values: List[int], 
                                   gamma: float = 1.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute sensitivity components S_F, S_B, S_I for each n.
    
    Returns:
        (S_F_list, S_B_list, S_I_list)
    """
    s_f_list = []
    s_b_list = []
    s_i_list = []
    
    for n in n_values:
        weights = WitnessOperator.compute_weights(float(n))
        s = compute_sensitivity_vector(weights, gamma)
        s_norm = normalize_sensitivity_vector(s)
        
        s_f_list.append(float(s_norm[0]))
        s_b_list.append(float(s_norm[1]))
        s_i_list.append(float(s_norm[2]))
    
    return s_f_list, s_b_list, s_i_list

