#!.venv/bin/python
"""
validate_witness_operator.py

Validation of Witness Operator <Q> as Stability Detector.

Tests:
1. Weight evolution W_F, W_B, W_I extraction
2. Stability exponent λ(n) computation
3. Witness value ⟨Q⟩ = λ - 1
4. Regime classification (coherent/marginal/decoherent)
5. Witness ledger creation
6. Guardian-witness coupling

Author: Joel
"""

import json
import numpy as np
from pathlib import Path
from zetadiffusion.witness_operator import (
    WitnessOperator, GuardianWitnessCoupling,
    extract_weight_evolution_from_data,
    compute_lambda_trajectory,
    compute_witness_trajectory
)
from zetadiffusion.guardian import SystemState
from zetadiffusion.validation_framework import run_validation

def validate_witness_operator() -> dict:
    """
    Validate Witness Operator implementation.
    """
    print("=" * 70)
    print("WITNESS OPERATOR VALIDATION")
    print("=" * 70)
    print()
    
    # Load validation data
    results_file = Path(".out/conjecture_9_1_3_results.json")
    if not results_file.exists():
        print("⚠ No validation data found. Using synthetic data.")
        n_values = list(range(1, 21))
        errors = [30.0 - 0.5 * n for n in n_values]  # Synthetic decreasing errors
    else:
        with open(results_file, 'r') as f:
            data = json.load(f)
        n_values = data.get('n_values', list(range(1, 21)))
        actual_zeros = data.get('actual_zeros', [])
        formula_zeros = data.get('formula_zeros', [])
        
        # Compute errors
        errors = []
        for i in range(min(len(actual_zeros), len(formula_zeros))):
            error = abs(actual_zeros[i] - formula_zeros[i])
            errors.append(error)
        
        # Pad if needed
        while len(errors) < len(n_values):
            errors.append(errors[-1] if errors else 0.0)
    
    print(f"Analyzing {len(n_values)} data points")
    print()
    
    # 1. Extract weight evolution
    print("=" * 70)
    print("WEIGHT EVOLUTION")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'W_F':<10} | {'W_B':<10} | {'W_I':<10} | {'Sum':<10}")
    print("-" * 70)
    
    weights_list = []
    for n in n_values:
        weights = WitnessOperator.compute_weights(float(n))
        weights_list.append(weights)
        print(f"{n:<5} | {weights.w_f:<10.6f} | {weights.w_b:<10.6f} | "
              f"{weights.w_i:<10.6f} | {weights.w_f + weights.w_b + weights.w_i:<10.6f}")
    
    print()
    
    # 2. Compute stability exponent λ(n)
    print("=" * 70)
    print("STABILITY EXPONENT λ(n)")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'λ(n)':<12} | {'⟨Q⟩':<12} | {'Regime':<15} | {'Coherence':<12}")
    print("-" * 70)
    
    lambda_trajectory = compute_lambda_trajectory(n_values)
    witness_trajectory = compute_witness_trajectory(n_values)
    
    for i, n in enumerate(n_values):
        lambda_val = lambda_trajectory[i]
        witness = witness_trajectory[i]
        print(f"{n:<5} | {lambda_val:<12.6f} | {witness.witness_value:<12.6f} | "
              f"{witness.regime:<15} | {witness.coherence:<12.6f}")
    
    print()
    
    # 3. Analyze trajectory
    print("=" * 70)
    print("TRAJECTORY ANALYSIS")
    print("=" * 70)
    print()
    
    # Find peak (maximum instability)
    lambda_peak_idx = np.argmax(lambda_trajectory)
    lambda_peak_n = n_values[lambda_peak_idx]
    lambda_peak_val = lambda_trajectory[lambda_peak_idx]
    
    print(f"Peak instability at n={lambda_peak_n}: λ = {lambda_peak_val:.6f}")
    if lambda_peak_n == 9:
        print("✓ Peak at n=9 matches prediction (maximum instability, error peak)")
    else:
        print(f"~ Peak at n={lambda_peak_n} (expected n=9)")
    
    # Check decay for n≥11
    interior_lambdas = [lambda_trajectory[i] for i, n in enumerate(n_values) if n >= 11]
    if interior_lambdas:
        interior_avg = np.mean(interior_lambdas)
        print(f"Interior average (n≥11): λ = {interior_avg:.6f}")
        if abs(interior_avg - 1.0) < 0.1:
            print("✓ Interior stabilizes (λ → 1⁻)")
        else:
            print(f"~ Interior λ = {interior_avg:.6f} (expected ≈ 1.0)")
    
    print()
    
    # 4. Witness ledger
    print("=" * 70)
    print("WITNESS LEDGER (Sample Entries)")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'λ':<10} | {'⟨Q⟩':<10} | {'Regime':<15} | {'Flux':<12} | {'Error':<12}")
    print("-" * 70)
    
    ledger_entries = []
    prev_error = None
    
    for i, n in enumerate(n_values):
        error = errors[i] if i < len(errors) else None
        entry = WitnessOperator.create_ledger_entry(
            n, error=error, prev_error=prev_error
        )
        ledger_entries.append(entry)
        
        if i < 5 or i >= len(n_values) - 5 or n in [7, 9, 11]:
            error_str = f"{error:.6f}" if error else "N/A"
            print(f"{n:<5} | {entry.lambda_value:<10.6f} | {entry.witness_value:<10.6f} | "
                  f"{entry.regime:<15} | {entry.boundary_flux:<12.6f} | {error_str:<12}")
        
        prev_error = error
    
    print()
    
    # 5. Guardian-witness coupling
    print("=" * 70)
    print("GUARDIAN-WITNESS COUPLING")
    print("=" * 70)
    print()
    
    coupling = GuardianWitnessCoupling(q_crit=0.1, beta_max=1.0)
    
    # Test at key points
    test_points = [5, 9, 11, 15, 20]
    print(f"{'n':<5} | {'⟨Q⟩':<12} | {'Regime':<15} | {'Guardian Status':<25} | {'Coupling':<10}")
    print("-" * 70)
    
    for n in test_points:
        if n in n_values:
            idx = n_values.index(n)
            witness = witness_trajectory[idx]
            
            # Create synthetic system state
            state = SystemState(
                coherence=0.5,
                chaos=0.1,
                stress=0.3,
                hurst=0.5
            )
            
            response = coupling.compute_response(witness, state)
            
            print(f"{n:<5} | {witness.witness_value:<12.6f} | {witness.regime:<15} | "
                  f"{response.status[:25]:<25} | {response.coupling:<10.6f}")
    
    print()
    
    # 6. Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Count regimes
    regime_counts = {}
    for witness in witness_trajectory:
        regime_counts[witness.regime] = regime_counts.get(witness.regime, 0) + 1
    
    print("Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} points")
    
    # Average coherence
    avg_coherence = np.mean([w.coherence for w in witness_trajectory])
    print(f"\nAverage coherence: {avg_coherence:.6f}")
    
    # Check predictions
    print("\nPredictions:")
    if lambda_peak_n == 9:
        print("  ✓ Peak instability at n=9")
    if interior_lambdas and abs(np.mean(interior_lambdas) - 1.0) < 0.1:
        print("  ✓ Interior stabilizes (λ → 1⁻)")
    
    results = {
        'n_values': n_values,
        'weights_evolution': [
            {'w_f': w.w_f, 'w_b': w.w_b, 'w_i': w.w_i} 
            for w in weights_list
        ],
        'lambda_trajectory': lambda_trajectory,
        'witness_trajectory': [
            {
                'witness_value': w.witness_value,
                'lambda_value': w.lambda_value,
                'regime': w.regime,
                'coherence': w.coherence
            }
            for w in witness_trajectory
        ],
        'ledger_entries': [
            {
                'iteration': e.iteration,
                'lambda_value': e.lambda_value,
                'witness_value': e.witness_value,
                'regime': e.regime,
                'coherence': e.coherence,
                'boundary_flux': e.boundary_flux,
                'error': e.error
            }
            for e in ledger_entries
        ],
        'lambda_peak_n': int(lambda_peak_n),
        'lambda_peak_value': float(lambda_peak_val),
        'interior_avg_lambda': float(np.mean(interior_lambdas)) if interior_lambdas else None,
        'avg_coherence': float(avg_coherence),
        'regime_counts': regime_counts
    }
    
    return results

def main():
    """Run validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Witness Operator",
        validation_func=validate_witness_operator,
        parameters={
            'q_crit': 0.1,
            'beta_max': 1.0
        },
        output_filename="witness_operator_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




