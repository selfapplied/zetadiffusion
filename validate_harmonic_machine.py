#!.venv/bin/python
"""
validate_harmonic_machine.py

Validation of Harmonic Machine with embedded CE1 Stability Module.

Tests:
1. Basic recursion with stability module
2. Q activation at n ≥ 11
3. Drift suppression
4. Tempo maintenance
5. Convergence behavior

Author: Joel
"""

import numpy as np
from zetadiffusion.harmonic_machine import HarmonicMachine
from zetadiffusion.validation_framework import run_validation

def simple_recursion(t_n: float) -> float:
    """
    Simple recursion operator: t[n+1] = t[n] + 0.1 * sin(t[n])
    
    This creates a slowly evolving sequence that can drift.
    """
    return t_n + 0.1 * np.sin(t_n)

def test_harmonic_machine() -> dict:
    """
    Test Harmonic Machine with embedded CE1 Stability Module.
    """
    print("=" * 70)
    print("HARMONIC MACHINE VALIDATION")
    print("=" * 70)
    print()
    
    # Initialize machine
    machine = HarmonicMachine(
        recursion_op=simple_recursion,
        activation_threshold=11,
        drift_tolerance=1.0
    )
    
    # Evolve from initial state
    t_0 = 1.0
    n_steps = 30
    sequence = machine.evolve(t_0, n_steps)
    
    print(f"Initial state: t[0] = {t_0:.6f}")
    print(f"Evolved {n_steps} steps")
    print()
    
    # Check stability at different stages
    print("Stability Status by Stage:")
    print("-" * 70)
    print(f"{'n':<5} | {'t[n]':<12} | {'Q Active':<10} | {'Precision':<12} | {'Max Drift':<12}")
    print("-" * 70)
    
    # Check at key points
    check_points = [5, 10, 11, 15, 20, 25, 30]
    stability_log = []
    
    for n in check_points:
        if n <= len(sequence) - 1:
            # Temporarily set history to check status at this point
            machine.stability_module.sequence_history = sequence[:n+1]
            status = machine.get_stability_status()
            
            q_active_str = "✓" if status['q_active'] else "✗"
            precision_str = f"{status['precision']:.6f}" if status['precision'] != float('inf') else "∞"
            max_drift_str = f"{status['max_drift']:.6f}" if status['max_drift'] != float('inf') else "∞"
            
            print(f"{n:<5} | {sequence[n]:<12.6f} | {q_active_str:<10} | {precision_str:<12} | {max_drift_str:<12}")
            
            stability_log.append({
                'n': n,
                't_n': sequence[n],
                'q_active': status['q_active'],
                'precision': status['precision'],
                'max_drift': status['max_drift']
            })
    
    print()
    print("=" * 70)
    print("DRIFT ANALYSIS")
    print("=" * 70)
    print()
    
    # Compute drifts
    drifts = [abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
    avg_drift = np.mean(drifts) if drifts else 0.0
    max_drift = max(drifts) if drifts else 0.0
    
    print(f"Average drift: {avg_drift:.6f}")
    print(f"Maximum drift: {max_drift:.6f}")
    print()
    
    # Check if drift is bounded
    if max_drift < 10.0:
        print("✓ Drift is bounded (< 10.0)")
    else:
        print("⚠ High drift detected (> 10.0)")
    
    # Check Q activation
    final_status = machine.get_stability_status()
    if final_status['q_active']:
        print("✓ <Q> is active (stable recursion)")
    else:
        print("~ <Q> not yet active (still converging)")
    
    print()
    print("=" * 70)
    print("SEQUENCE BEHAVIOR")
    print("=" * 70)
    print()
    
    # Analyze before and after activation threshold
    threshold = machine.stability_module.activation_threshold
    
    before_activation = sequence[:threshold]
    after_activation = sequence[threshold:]
    
    if before_activation:
        before_drifts = [abs(before_activation[i+1] - before_activation[i]) 
                        for i in range(len(before_activation)-1)]
        before_avg = np.mean(before_drifts) if before_drifts else 0.0
        print(f"Before activation (n < {threshold}): avg_drift = {before_avg:.6f}")
    
    if after_activation:
        after_drifts = [abs(after_activation[i+1] - after_activation[i]) 
                       for i in range(len(after_activation)-1)]
        after_avg = np.mean(after_drifts) if after_drifts else 0.0
        print(f"After activation (n ≥ {threshold}): avg_drift = {after_avg:.6f}")
        
        if after_activation and before_activation:
            drift_change = after_avg - before_avg
            if drift_change < 0:
                print(f"✓ Drift reduced by {abs(drift_change):.6f} after activation")
            else:
                print(f"~ Drift increased by {drift_change:.6f} after activation")
    
    results = {
        'sequence': sequence,
        'stability_log': stability_log,
        'avg_drift': float(avg_drift),
        'max_drift': float(max_drift),
        'q_active': final_status['q_active'],
        'precision': final_status['precision'],
        'activation_threshold': threshold,
        'n_steps': n_steps
    }
    
    return results

def main():
    """Run validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Harmonic Machine",
        validation_func=test_harmonic_machine,
        parameters={
            'activation_threshold': 11,
            'drift_tolerance': 1.0
        },
        output_filename="harmonic_machine_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

