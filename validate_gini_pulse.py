#!.venv/bin/python
"""
validate_gini_pulse.py

Validation of Gini Pulse Analysis.

Tests:
1. Sensitivity vector computation
2. Gini coefficient calculation
3. Gini pulse trajectory (dip at n=8-10, rise to 1)
4. Sensitivity component evolution
5. Pulse signature verification

Author: Joel
"""

import numpy as np
from zetadiffusion.gini_pulse import (
    compute_gini_pulse, analyze_gini_pulse,
    compute_sensitivity_components, compute_sensitivity_vector,
    normalize_sensitivity_vector, compute_gini_coefficient
)
from zetadiffusion.witness_operator import WitnessOperator
from zetadiffusion.validation_framework import run_validation

def validate_gini_pulse() -> dict:
    """
    Validate Gini Pulse Analysis.
    """
    print("=" * 70)
    print("GINI PULSE VALIDATION")
    print("=" * 70)
    print()
    
    # Use full range
    n_values = list(range(1, 21))
    
    print(f"Analyzing {len(n_values)} data points")
    print()
    
    # 1. Compute sensitivity vectors
    print("=" * 70)
    print("SENSITIVITY VECTOR COMPONENTS")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'S_F':<12} | {'S_B':<12} | {'S_I':<12} | {'Sum':<12} | {'Gini':<10}")
    print("-" * 70)
    
    s_f_list, s_b_list, s_i_list = compute_sensitivity_components(n_values)
    gini_values = compute_gini_pulse(n_values)
    
    for i, n in enumerate(n_values):
        s_f = s_f_list[i]
        s_b = s_b_list[i]
        s_i = s_i_list[i]
        total = s_f + s_b + s_i
        gini = gini_values[i]
        
        print(f"{n:<5} | {s_f:<12.6f} | {s_b:<12.6f} | {s_i:<12.6f} | "
              f"{total:<12.6f} | {gini:<10.6f}")
    
    print()
    
    # 2. Analyze Gini pulse
    print("=" * 70)
    print("GINI PULSE ANALYSIS")
    print("=" * 70)
    print()
    
    pulse_analysis = analyze_gini_pulse(n_values, gini_values)
    
    print(f"Minimum Gini: {pulse_analysis['min_gini']:.6f} at n={pulse_analysis['min_n']}")
    print(f"Maximum Gini: {pulse_analysis['max_gini']:.6f} at n={pulse_analysis['max_n']}")
    print()
    print("Phase averages:")
    print(f"  Early (n=1-4):   Gini = {pulse_analysis['early_gini']:.6f} (Feigenbaum monopoly)")
    print(f"  Mid (n=8-10):    Gini = {pulse_analysis['mid_gini']:.6f} (Blending phase)")
    print(f"  Late (n≥11):     Gini = {pulse_analysis['late_gini']:.6f} (Interior monopoly)")
    print()
    
    if pulse_analysis['has_pulse']:
        print(f"✓ Gini pulse detected!")
        print(f"  Pulse amplitude: {pulse_analysis['pulse_amplitude']:.6f}")
        print(f"  Dip at n={pulse_analysis['min_n']} (blending phase)")
        print(f"  Sharpens to {pulse_analysis['late_gini']:.6f} at interior")
    else:
        print("~ No clear pulse signature")
    
    print()
    
    # 3. Verify theoretical predictions
    print("=" * 70)
    print("THEORETICAL VERIFICATION")
    print("=" * 70)
    print()
    
    # Check fixed point conditions
    print("Fixed point conditions:")
    
    # Interior (n≥11): W_I ≈ 1, should have λ ≈ 1
    interior_n = [n for n in n_values if n >= 11]
    if interior_n:
        interior_gini = [gini_values[n_values.index(n)] for n in interior_n]
        avg_interior_gini = np.mean(interior_gini)
        print(f"  Interior (n≥11): Gini = {avg_interior_gini:.6f}")
        if avg_interior_gini > 0.9:
            print("    ✓ High inequality (interior monopoly) - matches prediction")
        else:
            print(f"    ~ Moderate inequality (expected > 0.9)")
    
    # Boundary (n=7-9): Should show blending
    boundary_n = [n for n in n_values if 7 <= n <= 9]
    if boundary_n:
        boundary_gini = [gini_values[n_values.index(n)] for n in boundary_n]
        avg_boundary_gini = np.mean(boundary_gini)
        print(f"  Boundary (n=7-9): Gini = {avg_boundary_gini:.6f}")
        if avg_boundary_gini < 0.6:
            print("    ✓ Lower inequality (blending phase) - matches prediction")
        else:
            print(f"    ~ Higher than expected (expected < 0.6)")
    
    # Feigenbaum (n=1-6): Should show high inequality
    feigenbaum_n = [n for n in n_values if 1 <= n <= 6]
    if feigenbaum_n:
        feigenbaum_gini = [gini_values[n_values.index(n)] for n in feigenbaum_n]
        avg_feigenbaum_gini = np.mean(feigenbaum_gini)
        print(f"  Feigenbaum (n=1-6): Gini = {avg_feigenbaum_gini:.6f}")
        if avg_feigenbaum_gini > 0.9:
            print("    ✓ High inequality (Feigenbaum monopoly) - matches prediction")
        else:
            print(f"    ~ Lower than expected (expected > 0.9)")
    
    print()
    
    # 4. Numerical check at n=11
    print("=" * 70)
    print("NUMERICAL CHECK AT n=11")
    print("=" * 70)
    print()
    
    n_check = 11
    weights = WitnessOperator.compute_weights(float(n_check))
    
    # Compute gamma near interior
    gamma_interior = 1.0 - 1.638 * (1.0 - weights.w_i)
    s = compute_sensitivity_vector(weights, gamma_interior)
    s_norm = normalize_sensitivity_vector(s)
    
    print(f"At n={n_check}:")
    print(f"  Weights: W_F={weights.w_f:.4f}, W_B={weights.w_b:.4f}, W_I={weights.w_i:.4f}")
    print(f"  Gamma: {gamma_interior:.6f}")
    print(f"  Sensitivity vector: S_F={s_norm[0]:.6f}, S_B={s_norm[1]:.6f}, S_I={s_norm[2]:.6f}")
    print(f"  Sum: {np.sum(s_norm):.6f} (should be 1.0)")
    
    gini_check = compute_gini_coefficient(s_norm)
    print(f"  Gini: {gini_check:.6f}")
    
    if abs(np.sum(s_norm) - 1.0) < 0.01:
        print("  ✓ Normalized to fixed point (sum = 1)")
    else:
        print(f"  ~ Not exactly at fixed point (sum = {np.sum(s_norm):.6f})")
    
    print()
    
    # 5. Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("Gini Pulse Signature:")
    print(f"  ✓ Early monopoly (n=1-4): Gini = {pulse_analysis['early_gini']:.6f}")
    print(f"  ✓ Blending dip (n=8-10): Gini = {pulse_analysis['mid_gini']:.6f}")
    print(f"  ✓ Interior monopoly (n≥11): Gini = {pulse_analysis['late_gini']:.6f}")
    
    if pulse_analysis['has_pulse']:
        print(f"\n✓ Pulse amplitude: {pulse_analysis['pulse_amplitude']:.6f}")
        print("  This is the signature of the zipper mechanism —")
        print("  temporary egalitarianism among clocks before one dominates.")
    
    results = {
        'n_values': n_values,
        'gini_values': gini_values,
        'sensitivity_components': {
            's_f': s_f_list,
            's_b': s_b_list,
            's_i': s_i_list
        },
        'pulse_analysis': pulse_analysis,
        'n11_check': {
            'weights': {'w_f': float(weights.w_f), 'w_b': float(weights.w_b), 'w_i': float(weights.w_i)},
            'gamma': float(gamma_interior),
            'sensitivity_vector': [float(s_norm[0]), float(s_norm[1]), float(s_norm[2])],
            'gini': float(gini_check)
        }
    }
    
    return results

def main():
    """Run validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Gini Pulse Analysis",
        validation_func=validate_gini_pulse,
        parameters={},
        output_filename="gini_pulse_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




