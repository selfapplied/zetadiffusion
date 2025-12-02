"""
concurrency_stability.py

9/11 Charge as Concurrency Stability Index.

The 9/11 charge Q₉₍₁₁₎ = tension / (ballast + 1) is a universal
tension-balancing operator that measures lock contention.

Key insight:
- 9's = tension units (spinning, retrying, almost-carries)
- 0's = ballast units (idle, yielding, spacing)
- Q < ~1 ⇒ stable (no contention)
- Q ≈ 1 ⇒ optimal flow (sweet spot)
- Q > ~1 ⇒ lock contention (too much tension)

The fixed point λ=1 happens when Q≈1, which is exactly where
contention vanishes - the concurrency sweet spot.

Author: Joel
"""

from typing import List, Dict, Tuple
import numpy as np
from zetadiffusion.digit_ballast import extract_ballast_and_units
from zetadiffusion.witness_operator import WitnessOperator

def compute_q_9_11(number: float) -> float:
    """
    Compute Q₉₍₁₁₎ = tension / (ballast + 1).
    
    This is the concurrency stability index:
    - Q < ~1: stable (no contention)
    - Q ≈ 1: optimal flow (sweet spot)
    - Q > ~1: lock contention (too much tension)
    
    Args:
        number: The live number to analyze
    
    Returns:
        Q₉₍₁₁₎ value
    """
    analysis = extract_ballast_and_units(number)
    return analysis['q_9_11']

def classify_concurrency_state(q_value: float) -> Tuple[str, str]:
    """
    Classify concurrency state from Q₉₍₁₁₎ value.
    
    Args:
        q_value: Q₉₍₁₁₎ value
    
    Returns:
        (state, description) tuple
    """
    if q_value < 0.5:
        return ("STABLE", "No contention - system has breathing room")
    elif q_value < 1.0:
        return ("OPTIMAL", "Sweet spot - optimal flow, no contention")
    elif q_value < 2.0:
        return ("MARGINAL", "Approaching contention - monitor closely")
    else:
        return ("CONTENTION", "Lock contention detected - too much tension")

def compute_concurrency_trajectory(numbers: List[float]) -> Dict:
    """
    Compute concurrency stability trajectory.
    
    Tracks Q₉₍₁₁₎ values across a sequence to detect contention patterns.
    
    Args:
        numbers: Sequence of live numbers
    
    Returns:
        Dictionary with trajectory data
    """
    q_values = []
    states = []
    ballasts = []
    units = []
    
    for num in numbers:
        analysis = extract_ballast_and_units(num)
        q_val = analysis['q_9_11']
        state, _ = classify_concurrency_state(q_val)
        
        q_values.append(q_val)
        states.append(state)
        ballasts.append(analysis['ballasts'])
        units.append(analysis['units'])
    
    # Find contention events (Q > 1)
    contention_indices = [i for i, q in enumerate(q_values) if q > 1.0]
    optimal_indices = [i for i, q in enumerate(q_values) if 0.5 <= q <= 1.0]
    
    return {
        'q_values': q_values,
        'states': states,
        'ballasts': ballasts,
        'units': units,
        'contention_count': len(contention_indices),
        'optimal_count': len(optimal_indices),
        'avg_q': float(np.mean(q_values)) if q_values else 0.0,
        'max_q': float(np.max(q_values)) if q_values else 0.0,
        'min_q': float(np.min(q_values)) if q_values else 0.0
    }

def detect_fixed_point(numbers: List[float], indices: List[int] = None) -> Dict:
    """
    Detect fixed point where Q₉₍₁₁₎ ≈ 1 (concurrency sweet spot).
    
    This corresponds to λ=1 in the renormalization operator,
    where contention vanishes.
    
    Args:
        numbers: Sequence of live numbers
        indices: Optional list of indices (n values)
    
    Returns:
        Dictionary with fixed point detection results
    """
    if indices is None:
        indices = list(range(1, len(numbers) + 1))
    
    trajectory = compute_concurrency_trajectory(numbers)
    
    # Find points where Q ≈ 1 (within tolerance)
    tolerance = 0.2
    fixed_point_indices = [
        i for i, q in enumerate(trajectory['q_values'])
        if abs(q - 1.0) < tolerance
    ]
    
    # Check if fixed point aligns with λ=1 (interior phase, n≥11)
    interior_fixed_points = [
        i for i in fixed_point_indices
        if i < len(indices) and indices[i] >= 11
    ]
    
    return {
        'fixed_point_indices': fixed_point_indices,
        'interior_fixed_points': interior_fixed_points,
        'fixed_point_count': len(fixed_point_indices),
        'interior_count': len(interior_fixed_points),
        'trajectory': trajectory,
        'aligned_with_lambda': len(interior_fixed_points) > 0
    }

def derive_backoff_algorithm(q_value: float, base_delay: float = 1.0) -> float:
    """
    Derive lock-free backoff delay from Q₉₍₁₁₎ value.
    
    When Q > 1 (contention), increase backoff.
    When Q < 1 (stable), decrease backoff.
    
    Args:
        q_value: Q₉₍₁₁₎ value
        base_delay: Base delay in microseconds
    
    Returns:
        Recommended backoff delay
    """
    if q_value < 0.5:
        # Very stable - minimal backoff
        return base_delay * 0.5
    elif q_value < 1.0:
        # Optimal - use base delay
        return base_delay
    elif q_value < 2.0:
        # Marginal contention - moderate backoff
        return base_delay * (1.0 + (q_value - 1.0))
    else:
        # High contention - exponential backoff
        return base_delay * (2.0 ** min(q_value - 1.0, 10.0))

def analyze_cas_loop(numbers: List[float], max_retries: int = 10) -> Dict:
    """
    Analyze CAS loop behavior using Q₉₍₁₁₎.
    
    Predicts contention waves in a compare-and-swap loop.
    
    Args:
        numbers: Sequence representing CAS attempts
        max_retries: Maximum retry budget
    
    Returns:
        Dictionary with CAS loop analysis
    """
    trajectory = compute_concurrency_trajectory(numbers)
    
    # Predict retry behavior
    retry_predictions = []
    for q_val in trajectory['q_values']:
        if q_val > 1.0:
            # Contention - will need retries
            predicted_retries = min(int(q_val * 2), max_retries)
        else:
            # Stable - should succeed quickly
            predicted_retries = 1
        
        retry_predictions.append(predicted_retries)
    
    return {
        'retry_predictions': retry_predictions,
        'avg_retries': float(np.mean(retry_predictions)) if retry_predictions else 0.0,
        'max_retries': max(retry_predictions) if retry_predictions else 0,
        'contention_waves': len([r for r in retry_predictions if r > 1]),
        'trajectory': trajectory
    }




