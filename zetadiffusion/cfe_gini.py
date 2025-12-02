"""
cfe_gini.py

Continued Fraction Expansion (CFE) to Gini Pulse Mapping.

The Gini pulse is about inequality of influence.
A continued fraction is a time-ordered inequality series.

Key insight:
- Large partial quotients (a_k) → high inequality (dominance)
- Small partial quotients (a_k) → low inequality (egalitarian)
- The CFE spike corresponds to the mid-RG perturbation (blending zone)

The inequality of partial quotients matches the inequality of clock weights.

Author: Joel
"""

from typing import List, Tuple
import numpy as np
import math
from zetadiffusion.gini_pulse import compute_gini_coefficient

# Universal constants
PI = math.pi
DELTA_F = 4.66920160910299067185320382
ALPHA_F = 2.502907875095892822283902873218

def continued_fraction_expansion(x: float, max_terms: int = 20) -> List[int]:
    """
    Compute continued fraction expansion [a_0; a_1, a_2, ...] of x.
    
    x = a_0 + 1/(a_1 + 1/(a_2 + ...))
    
    Args:
        x: The number to expand
        max_terms: Maximum number of partial quotients
    
    Returns:
        List of partial quotients [a_0, a_1, a_2, ...]
    """
    if x < 0:
        # Handle negative numbers
        cf = continued_fraction_expansion(-x, max_terms)
        if cf:
            cf[0] = -cf[0]
        return cf
    
    result = []
    remaining = x
    
    for _ in range(max_terms):
        if abs(remaining) < 1e-10:
            break
        
        a_k = int(remaining)
        result.append(a_k)
        
        remaining = remaining - a_k
        if abs(remaining) < 1e-10:
            break
        
        remaining = 1.0 / remaining
    
    return result

def compute_cfe_gini_sequence(cfe: List[int]) -> List[float]:
    """
    Compute Gini coefficient sequence from continued fraction expansion.
    
    For each prefix [a_0; a_1, ..., a_n], compute Gini of {a_0, ..., a_n}.
    
    This generates the "inequality waveform" of the CFE.
    
    Args:
        cfe: List of partial quotients [a_0, a_1, ...]
    
    Returns:
        List of Gini coefficients for each prefix
    """
    gini_sequence = []
    
    for n in range(1, len(cfe) + 1):
        prefix = cfe[:n]
        if prefix:
            # Compute Gini of partial quotients
            gini = compute_gini_coefficient(np.array(prefix))
            gini_sequence.append(gini)
    
    return gini_sequence

def analyze_cfe_pulse(cfe: List[int], gini_sequence: List[float]) -> dict:
    """
    Analyze CFE for Gini pulse characteristics.
    
    Looks for:
    - Early phase (small terms, low Gini)
    - Spike term (large term, Gini jump)
    - Relaxation phase (smaller terms, Gini falls)
    - Pulse energy (amplitude of spike)
    
    Args:
        cfe: Partial quotients
        gini_sequence: Gini coefficients for each prefix
    
    Returns:
        Dictionary with pulse analysis
    """
    if not cfe or not gini_sequence:
        return {
            'has_pulse': False,
            'spike_index': None,
            'spike_value': None,
            'pulse_amplitude': 0.0
        }
    
    # Find spike (largest partial quotient)
    spike_index = np.argmax(cfe)
    spike_value = cfe[spike_index]
    
    # Early Gini (before spike)
    early_gini = np.mean(gini_sequence[:spike_index]) if spike_index > 0 else gini_sequence[0] if gini_sequence else 0.0
    
    # Spike Gini
    spike_gini = gini_sequence[spike_index] if spike_index < len(gini_sequence) else 0.0
    
    # Late Gini (after spike)
    late_gini = np.mean(gini_sequence[spike_index+1:]) if spike_index+1 < len(gini_sequence) else gini_sequence[-1] if gini_sequence else 0.0
    
    # Pulse amplitude
    pulse_amplitude = spike_gini - early_gini if spike_gini > early_gini else 0.0
    
    # Check if pulse shape matches prediction
    has_pulse = (spike_value > 10 and  # Large spike
                 spike_gini > early_gini and  # Gini jumps
                 late_gini < spike_gini)  # Gini relaxes
    
    return {
        'has_pulse': has_pulse,
        'spike_index': int(spike_index),
        'spike_value': int(spike_value),
        'spike_gini': float(spike_gini),
        'early_gini': float(early_gini),
        'late_gini': float(late_gini),
        'pulse_amplitude': float(pulse_amplitude),
        'cfe': cfe,
        'gini_sequence': gini_sequence
    }

def compute_universal_constants_cfe() -> dict:
    """
    Compute CFE and Gini sequences for universal constants.
    
    Constants:
    - π (pi)
    - 1/δ_F (inverse Feigenbaum delta)
    - 7/11 (mysterious ratio)
    
    Returns:
        Dictionary with CFE and Gini analysis for each constant
    """
    constants = {
        'pi': PI,
        '1/delta_F': 1.0 / DELTA_F,
        '7/11': 7.0 / 11.0
    }
    
    results = {}
    
    for name, value in constants.items():
        # Compute CFE
        cfe = continued_fraction_expansion(value, max_terms=20)
        
        # Compute Gini sequence
        gini_sequence = compute_cfe_gini_sequence(cfe)
        
        # Analyze pulse
        pulse_analysis = analyze_cfe_pulse(cfe, gini_sequence)
        
        results[name] = {
            'value': value,
            'cfe': cfe,
            'gini_sequence': gini_sequence,
            'pulse_analysis': pulse_analysis
        }
    
    return results

def overlay_cfe_with_dynamical_gini(cfe_gini: List[float], 
                                    dynamical_gini: List[float],
                                    cfe_indices: List[int] = None,
                                    dyn_indices: List[int] = None) -> dict:
    """
    Overlay CFE-Gini curve with dynamical Gini pulse.
    
    Check if pulse phases align.
    
    Args:
        cfe_gini: Gini sequence from CFE
        dynamical_gini: Gini sequence from weight dynamics
        cfe_indices: Optional indices for CFE (default: [0, 1, 2, ...])
        dyn_indices: Optional indices for dynamical (default: n values)
    
    Returns:
        Dictionary with alignment analysis
    """
    if cfe_indices is None:
        cfe_indices = list(range(len(cfe_gini)))
    if dyn_indices is None:
        dyn_indices = list(range(1, len(dynamical_gini) + 1))
    
    # Normalize sequences to same length for comparison
    min_len = min(len(cfe_gini), len(dynamical_gini))
    cfe_normalized = cfe_gini[:min_len]
    dyn_normalized = dynamical_gini[:min_len]
    
    # Compute correlation
    if len(cfe_normalized) > 1:
        correlation = np.corrcoef(cfe_normalized, dyn_normalized)[0, 1]
    else:
        correlation = 0.0
    
    # Find spike locations
    cfe_spike_idx = np.argmax(cfe_normalized) if cfe_normalized else None
    dyn_spike_idx = np.argmax(dyn_normalized) if dyn_normalized else None
    
    # Check alignment
    aligned = False
    if cfe_spike_idx is not None and dyn_spike_idx is not None:
        # Check if spikes are within 2 positions of each other
        aligned = abs(cfe_spike_idx - dyn_spike_idx) <= 2
    
    return {
        'correlation': float(correlation),
        'cfe_spike_index': int(cfe_spike_idx) if cfe_spike_idx is not None else None,
        'dyn_spike_index': int(dyn_spike_idx) if dyn_spike_idx is not None else None,
        'aligned': aligned,
        'cfe_normalized': cfe_normalized,
        'dyn_normalized': dyn_normalized
    }




