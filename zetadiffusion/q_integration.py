"""
q_integration.py

Integration of <Q> CE1 seed into validation framework.

Automatically tracks <Q> activation and stability metrics in all validation runs.

Author: Joel
"""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

try:
    from zetadiffusion.ce1_seed import (
        create_q_seed, ClockInteraction, check_stability, QSeed
    )
    Q_AVAILABLE = True
except ImportError:
    Q_AVAILABLE = False

def compute_q_metrics(sequence: List[float], indices: Optional[List[int]] = None) -> Dict:
    """
    Compute <Q> metrics for a sequence.
    
    Args:
        sequence: Sequence of values
        indices: Optional list of indices (for clock phase analysis)
    
    Returns:
        Dictionary with <Q> metrics
    """
    if not Q_AVAILABLE:
        return {'q_available': False}
    
    if len(sequence) < 2:
        return {
            'q_available': True,
            'precision': float('inf'),
            'max_drift': float('inf'),
            'q_active': False,
            'activation': 0.0
        }
    
    q_seed = create_q_seed()
    
    # Overall stability
    # Tolerance is relative to sequence scale (Riemann zeros have ~2-3 unit spacing)
    sequence_scale = max(abs(max(sequence) - min(sequence)), 1.0) if sequence else 1.0
    relative_tolerance = max(sequence_scale * 0.1, 1.0)  # 10% of scale, min 1.0
    stability = check_stability(sequence, tolerance=relative_tolerance)
    
    # Compute activation by clock phase if indices provided
    phase_metrics = {}
    if indices and len(indices) == len(sequence):
        for i, n in enumerate(indices):
            phase = ClockInteraction.get_clock_phase(n)
            activation = ClockInteraction.q_activation(n)
            
            if phase not in phase_metrics:
                phase_metrics[phase] = {
                    'activation': [],
                    'indices': [],
                    'values': []
                }
            
            phase_metrics[phase]['activation'].append(activation)
            phase_metrics[phase]['indices'].append(n)
            phase_metrics[phase]['values'].append(sequence[i])
        
        # Average activation by phase
        for phase in phase_metrics:
            phase_metrics[phase]['avg_activation'] = float(np.mean(phase_metrics[phase]['activation']))
            phase_metrics[phase]['is_active'] = phase_metrics[phase]['avg_activation'] >= 0.5
    
    # Overall activation (weighted by sequence length)
    if indices:
        activations = [ClockInteraction.q_activation(n) for n in indices]
        overall_activation = float(np.mean(activations))
    else:
        # Estimate from sequence length (assume starts at n=1)
        estimated_indices = list(range(1, len(sequence) + 1))
        activations = [ClockInteraction.q_activation(n) for n in estimated_indices]
        overall_activation = float(np.mean(activations))
    
    return {
        'q_available': True,
        'precision': stability['precision'],
        'max_drift': stability['max_drift'],
        'q_active': stability['q_active'],
        'activation': overall_activation,
        'phase_metrics': phase_metrics if phase_metrics else None,
        'stable': stability['stable']
    }

def add_q_metrics_to_results(results: Dict, sequence: List[float], 
                             indices: Optional[List[int]] = None) -> Dict:
    """
    Add <Q> metrics to validation results.
    
    Args:
        results: Existing results dictionary
        sequence: Sequence of values to analyze
        indices: Optional list of indices
    
    Returns:
        Results dictionary with <Q> metrics added
    """
    q_metrics = compute_q_metrics(sequence, indices)
    
    # Add to results
    if 'q_metrics' not in results:
        results['q_metrics'] = {}
    
    results['q_metrics'].update(q_metrics)
    
    return results

def format_q_summary(q_metrics: Dict) -> str:
    """
    Format <Q> metrics as readable summary string.
    
    Args:
        q_metrics: <Q> metrics dictionary
    
    Returns:
        Formatted summary string
    """
    if not q_metrics.get('q_available', False):
        return "Q metrics: Not available"
    
    parts = []
    parts.append(f"Q activation: {q_metrics.get('activation', 0.0):.3f}")
    parts.append(f"Q active: {'✓' if q_metrics.get('q_active', False) else '✗'}")
    parts.append(f"Precision: {q_metrics.get('precision', float('inf')):.6f}")
    
    if q_metrics.get('phase_metrics'):
        phase_parts = []
        for phase, metrics in q_metrics['phase_metrics'].items():
            if 'avg_activation' in metrics:
                phase_parts.append(f"{phase}: {metrics['avg_activation']:.2f}")
        if phase_parts:
            parts.append(f"By phase: {', '.join(phase_parts)}")
    
    return "Q metrics: " + " | ".join(parts)

def track_q_over_time(validation_runs: List[Dict]) -> Dict:
    """
    Track <Q> activation over multiple validation runs.
    
    Args:
        validation_runs: List of validation result dictionaries
    
    Returns:
        Dictionary with <Q> tracking over time
    """
    if not Q_AVAILABLE:
        return {'q_available': False}
    
    activations = []
    timestamps = []
    validation_types = []
    
    for run in validation_runs:
        q_metrics = run.get('q_metrics', {})
        if q_metrics.get('q_available', False):
            activations.append(q_metrics.get('activation', 0.0))
            timestamps.append(run.get('timestamp', ''))
            validation_types.append(run.get('validation_type', 'Unknown'))
    
    return {
        'q_available': True,
        'activations': activations,
        'timestamps': timestamps,
        'validation_types': validation_types,
        'avg_activation': float(np.mean(activations)) if activations else 0.0,
        'trend': 'increasing' if len(activations) > 1 and activations[-1] > activations[0] else 'stable'
    }

