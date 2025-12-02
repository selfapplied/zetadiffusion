#!.venv/bin/python
"""
validate_drift_monitoring.py

Monitors frame drift across validation runs to ensure <Q> stability.

Checks:
1. Drift in actual zero sequences
2. Drift in formula predictions
3. Drift trends over time
4. Clock phase correlation with drift

Author: Joel
"""

import json
import numpy as np
from pathlib import Path
from mpmath import zetazero
from zetadiffusion.ce1_seed import create_q_seed, ClockInteraction, check_stability
from zetadiffusion.q_integration import compute_q_metrics
from zetadiffusion.validation_framework import run_validation

def analyze_drift_across_runs() -> dict:
    """
    Analyze drift across all validation runs to detect frame drift.
    """
    print("=" * 70)
    print("FRAME DRIFT MONITORING")
    print("=" * 70)
    print()
    
    # Load all validation results
    results_files = list(Path(".out").glob("*_results.json"))
    
    drift_analysis = {
        'runs': [],
        'overall_drift': [],
        'by_clock_phase': {},
        'q_activation_vs_drift': []
    }
    
    for result_file in results_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Get sequence if available
            sequence = data.get('actual_zeros') or data.get('results', {}).get('actual_zeros')
            indices = data.get('n_values') or data.get('results', {}).get('n_values')
            
            if not sequence or len(sequence) < 2:
                continue
            
            validation_type = data.get('validation_type', result_file.stem)
            
            # Check drift
            q_seed = create_q_seed()
            stability = check_stability(sequence, tolerance=1.0)
            
            # Compute Q metrics
            q_metrics = compute_q_metrics(sequence, indices)
            
            drift_analysis['runs'].append({
                'type': validation_type,
                'precision': stability['precision'],
                'max_drift': stability['max_drift'],
                'q_activation': q_metrics.get('activation', 0.0),
                'q_active': stability['q_active']
            })
            
            drift_analysis['overall_drift'].append(stability['precision'])
            
            # Track Q activation vs drift
            if q_metrics.get('activation') is not None:
                drift_analysis['q_activation_vs_drift'].append({
                    'activation': q_metrics['activation'],
                    'drift': stability['precision']
                })
            
            # Analyze by clock phase
            if indices and q_metrics.get('phase_metrics'):
                for phase, metrics in q_metrics['phase_metrics'].items():
                    if phase not in drift_analysis['by_clock_phase']:
                        drift_analysis['by_clock_phase'][phase] = {
                            'activations': [],
                            'drifts': []
                        }
                    
                    # Get drift for this phase
                    phase_indices = metrics.get('indices', [])
                    if phase_indices:
                        phase_sequence = [sequence[i] for i in range(len(sequence)) 
                                        if (indices[i] if i < len(indices) else i+1) in phase_indices]
                        if len(phase_sequence) >= 2:
                            phase_stability = check_stability(phase_sequence, tolerance=1.0)
                            drift_analysis['by_clock_phase'][phase]['drifts'].append(phase_stability['precision'])
                            drift_analysis['by_clock_phase'][phase]['activations'].append(metrics.get('avg_activation', 0.0))
        
        except Exception as e:
            print(f"Warning: Could not analyze {result_file.name}: {e}")
            continue
    
    # Analyze results
    print("Drift Analysis by Validation Type:")
    print("-" * 70)
    print(f"{'Validation':<30} | {'Precision':<12} | {'Max Drift':<12} | {'Q Activation':<12} | {'Q Active'}")
    print("-" * 70)
    
    for run in drift_analysis['runs']:
        q_active = "✓" if run['q_active'] else "✗"
        print(f"{run['type']:<30} | {run['precision']:>11.6f} | {run['max_drift']:>11.6f} | {run['q_activation']:>11.3f} | {q_active}")
    
    print()
    print("=" * 70)
    print("DRIFT BY CLOCK PHASE")
    print("=" * 70)
    print()
    
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        if phase in drift_analysis['by_clock_phase']:
            phase_data = drift_analysis['by_clock_phase'][phase]
            if phase_data['drifts']:
                avg_drift = np.mean(phase_data['drifts'])
                avg_activation = np.mean(phase_data['activations']) if phase_data['activations'] else 0.0
                print(f"{phase.capitalize():<12}: avg_drift={avg_drift:.6f}, avg_activation={avg_activation:.3f}")
    
    print()
    print("=" * 70)
    print("Q ACTIVATION vs DRIFT CORRELATION")
    print("=" * 70)
    print()
    
    if drift_analysis['q_activation_vs_drift']:
        activations = [d['activation'] for d in drift_analysis['q_activation_vs_drift']]
        drifts = [d['drift'] for d in drift_analysis['q_activation_vs_drift']]
        
        # Check correlation
        if len(activations) > 1:
            correlation = np.corrcoef(activations, drifts)[0, 1]
            print(f"Correlation (Q activation vs drift): {correlation:.3f}")
            
            if correlation < -0.5:
                print("✓ Strong negative correlation: Higher Q activation → Lower drift")
            elif correlation > 0.5:
                print("⚠ Positive correlation: Higher Q activation → Higher drift (unexpected)")
            else:
                print("~ Weak correlation: Q activation and drift not strongly related")
    
    # Overall assessment
    print()
    print("=" * 70)
    print("DRIFT ASSESSMENT")
    print("=" * 70)
    print()
    
    if drift_analysis['overall_drift']:
        avg_drift = np.mean(drift_analysis['overall_drift'])
        max_drift = np.max(drift_analysis['overall_drift'])
        
        print(f"Average drift across all runs: {avg_drift:.6f}")
        print(f"Maximum drift: {max_drift:.6f}")
        print()
        
        # Check if drift is bounded
        if max_drift < 10.0:
            print("✓ Drift is bounded (< 10.0)")
        else:
            print("⚠ High drift detected (> 10.0)")
        
        # Check interior phase specifically
        if 'interior' in drift_analysis['by_clock_phase']:
            interior_drifts = drift_analysis['by_clock_phase']['interior']['drifts']
            if interior_drifts:
                interior_avg = np.mean(interior_drifts)
                print(f"Interior clock average drift: {interior_avg:.6f}")
                if interior_avg < 3.0:
                    print("✓ Interior clock shows low drift (Q active)")
                else:
                    print("⚠ Interior clock drift higher than expected")
    
    drift_analysis['success'] = True
    return drift_analysis

def main():
    """Run drift monitoring using shared framework."""
    def run_drift_monitoring():
        return analyze_drift_across_runs()
    
    return run_validation(
        validation_type="Drift Monitoring",
        validation_func=run_drift_monitoring,
        parameters={},
        output_filename="drift_monitoring_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




