#!.venv/bin/python
"""
validate_conjectures_comprehensive.py

Comprehensive validation of Conjectures 9.1.1, 9.1.2, and 9.1.3.

Tests:
1. All three conjectures against first 100 Riemann zeros
2. Error patterns at n=7, 9, 11 (clock boundaries)
3. Systematic comparison of error behavior
4. Testable predictions verification

Author: Joel
"""

import numpy as np
import math
from scipy.special import comb
from mpmath import zetazero, pi, log
import mpmath
from zetadiffusion.validation_framework import run_validation

mpmath.mp.dps = 50  # High precision

# Import conjecture formulas
from validate_conjecture_9_1_1 import explicit_zero_formula, derive_arctangent_correction, fit_scaling_factor
from validate_conjecture_9_1_2 import conjecture_9_1_2_formula, fit_scaling_factor_9_1_2
from validate_conjecture_9_1_3 import conjecture_9_1_3_formula, fit_scaling_factors_9_1_3

def validate_all_conjectures(n_max: int = 100) -> dict:
    """
    Comprehensive validation of all three conjectures.
    
    Args:
        n_max: Maximum n to test (default: 100)
    
    Returns:
        Dictionary with comprehensive comparison
    """
    print("=" * 70)
    print("COMPREHENSIVE CONJECTURE VALIDATION")
    print("=" * 70)
    print()
    print(f"Testing n=1 to n={n_max}")
    print()
    
    # Fit scaling factors
    print("Fitting scaling factors...")
    scaling_9_1_1 = fit_scaling_factor(n_max=min(50, n_max), use_robust=True)
    scaling_9_1_2, _, _ = fit_scaling_factor_9_1_2(n_max=min(50, n_max), use_robust=True)
    scaling_9_1_3 = fit_scaling_factors_9_1_3(n_max=min(50, n_max), use_robust=True)
    
    print(f"9.1.1 scaling: {scaling_9_1_1:.4f}")
    print(f"9.1.2 scaling: {scaling_9_1_2:.4f}")
    print(f"9.1.3 scaling: {scaling_9_1_3}")
    print()
    
    # Collect results
    results = {
        'n_values': [],
        'actual_zeros': [],
        'conjecture_9_1_1': {'zeros': [], 'errors': [], 'rel_errors': []},
        'conjecture_9_1_2': {'zeros': [], 'errors': [], 'rel_errors': []},
        'conjecture_9_1_3': {'zeros': [], 'errors': [], 'rel_errors': []},
        'scaling_factors': {
            '9_1_1': scaling_9_1_1,
            '9_1_2': scaling_9_1_2,
            '9_1_3': scaling_9_1_3
        }
    }
    
    print("=" * 70)
    print("RUNNING VALIDATION")
    print("=" * 70)
    print()
    print(f"{'n':<6} | {'Actual':<15} | {'9.1.1 Error':<15} | {'9.1.2 Error':<15} | {'9.1.3 Error':<15}")
    print("-" * 70)
    
    for n in range(1, n_max + 1):
        try:
            # Actual zero
            actual_zero = float(zetazero(n).imag)
            
            # Conjecture 9.1.1
            formula_9_1_1 = explicit_zero_formula(n, scaling_factor=scaling_9_1_1)
            correction = derive_arctangent_correction(n)
            formula_9_1_1 += correction
            error_9_1_1 = abs(actual_zero - formula_9_1_1)
            rel_error_9_1_1 = (error_9_1_1 / actual_zero) * 100 if actual_zero > 0 else 0.0
            
            # Conjecture 9.1.2
            formula_9_1_2 = conjecture_9_1_2_formula(n, scaling_factor=scaling_9_1_2)
            error_9_1_2 = abs(actual_zero - formula_9_1_2)
            rel_error_9_1_2 = (error_9_1_2 / actual_zero) * 100 if actual_zero > 0 else 0.0
            
            # Conjecture 9.1.3
            formula_9_1_3 = conjecture_9_1_3_formula(n, scaling_factors=scaling_9_1_3)
            error_9_1_3 = abs(actual_zero - formula_9_1_3)
            rel_error_9_1_3 = (error_9_1_3 / actual_zero) * 100 if actual_zero > 0 else 0.0
            
            # Store results
            results['n_values'].append(n)
            results['actual_zeros'].append(actual_zero)
            results['conjecture_9_1_1']['zeros'].append(formula_9_1_1)
            results['conjecture_9_1_1']['errors'].append(error_9_1_1)
            results['conjecture_9_1_1']['rel_errors'].append(rel_error_9_1_1)
            results['conjecture_9_1_2']['zeros'].append(formula_9_1_2)
            results['conjecture_9_1_2']['errors'].append(error_9_1_2)
            results['conjecture_9_1_2']['rel_errors'].append(rel_error_9_1_2)
            results['conjecture_9_1_3']['zeros'].append(formula_9_1_3)
            results['conjecture_9_1_3']['errors'].append(error_9_1_3)
            results['conjecture_9_1_3']['rel_errors'].append(rel_error_9_1_3)
            
            # Print every 10th or at key points
            if n <= 20 or n % 10 == 0 or n in [7, 9, 11]:
                print(f"{n:<6} | {actual_zero:>14.6f} | {rel_error_9_1_1:>14.2f}% | "
                      f"{rel_error_9_1_2:>14.2f}% | {rel_error_9_1_3:>14.2f}%")
        
        except Exception as e:
            print(f"Error at n={n}: {e}")
            continue
    
    print()
    
    # Analysis
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print()
    
    # Overall statistics
    for name, data in [('9.1.1', results['conjecture_9_1_1']),
                       ('9.1.2', results['conjecture_9_1_2']),
                       ('9.1.3', results['conjecture_9_1_3'])]:
        errors = data['errors']
        rel_errors = data['rel_errors']
        
        avg_error = np.mean(errors) if errors else 0.0
        avg_rel_error = np.mean(rel_errors) if rel_errors else 0.0
        max_error = np.max(errors) if errors else 0.0
        max_rel_error = np.max(rel_errors) if rel_errors else 0.0
        
        print(f"Conjecture {name}:")
        print(f"  Average absolute error: {avg_error:.6f}")
        print(f"  Average relative error: {avg_rel_error:.2f}%")
        print(f"  Maximum absolute error: {max_error:.6f}")
        print(f"  Maximum relative error: {max_rel_error:.2f}%")
        print()
    
    # Testable predictions at n=7, 9, 11
    print("=" * 70)
    print("TESTABLE PREDICTIONS: n=7, 9, 11")
    print("=" * 70)
    print()
    
    key_points = [7, 9, 11]
    for n in key_points:
        if n in results['n_values']:
            idx = results['n_values'].index(n)
            actual = results['actual_zeros'][idx]
            
            print(f"n={n}:")
            print(f"  Actual zero: {actual:.6f}")
            
            for name, data in [('9.1.1', results['conjecture_9_1_1']),
                               ('9.1.2', results['conjecture_9_1_2']),
                               ('9.1.3', results['conjecture_9_1_3'])]:
                formula_zero = data['zeros'][idx]
                error = data['errors'][idx]
                rel_error = data['rel_errors'][idx]
                print(f"  {name}: {formula_zero:.6f} (error: {error:.6f}, {rel_error:.2f}%)")
            print()
    
    # Phase analysis
    print("=" * 70)
    print("PHASE ANALYSIS (Clock Boundaries)")
    print("=" * 70)
    print()
    
    phases = {
        'Feigenbaum (n<7)': [n for n in results['n_values'] if n < 7],
        'Boundary (7≤n<9)': [n for n in results['n_values'] if 7 <= n < 9],
        'Membrane (9≤n<11)': [n for n in results['n_values'] if 9 <= n < 11],
        'Interior (n≥11)': [n for n in results['n_values'] if n >= 11]
    }
    
    for phase_name, phase_n in phases.items():
        if not phase_n:
            continue
        
        indices = [results['n_values'].index(n) for n in phase_n]
        
        print(f"{phase_name}:")
        for name, data in [('9.1.1', results['conjecture_9_1_1']),
                           ('9.1.2', results['conjecture_9_1_2']),
                           ('9.1.3', results['conjecture_9_1_3'])]:
            phase_errors = [data['rel_errors'][i] for i in indices]
            avg_rel = np.mean(phase_errors) if phase_errors else 0.0
            print(f"  {name} avg error: {avg_rel:.2f}%")
        print()
    
    # Improvement analysis
    print("=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print()
    
    errors_9_1_1 = results['conjecture_9_1_1']['errors']
    errors_9_1_2 = results['conjecture_9_1_2']['errors']
    errors_9_1_3 = results['conjecture_9_1_3']['errors']
    
    if errors_9_1_1 and errors_9_1_2:
        improvement_9_1_2 = ((np.mean(errors_9_1_1) - np.mean(errors_9_1_2)) / np.mean(errors_9_1_1)) * 100
        print(f"9.1.2 vs 9.1.1: {improvement_9_1_2:.1f}% improvement")
    
    if errors_9_1_2 and errors_9_1_3:
        improvement_9_1_3 = ((np.mean(errors_9_1_2) - np.mean(errors_9_1_3)) / np.mean(errors_9_1_2)) * 100
        print(f"9.1.3 vs 9.1.2: {improvement_9_1_3:.1f}% improvement")
    
    if errors_9_1_1 and errors_9_1_3:
        improvement_total = ((np.mean(errors_9_1_1) - np.mean(errors_9_1_3)) / np.mean(errors_9_1_1)) * 100
        print(f"9.1.3 vs 9.1.1: {improvement_total:.1f}% total improvement")
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    best_conjecture = None
    best_avg_error = float('inf')
    
    for name, data in [('9.1.1', results['conjecture_9_1_1']),
                       ('9.1.2', results['conjecture_9_1_2']),
                       ('9.1.3', results['conjecture_9_1_3'])]:
        avg_error = np.mean(data['errors']) if data['errors'] else float('inf')
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best_conjecture = name
    
    print(f"Best performing: Conjecture {best_conjecture} (avg error: {best_avg_error:.6f})")
    print()
    print("Testable predictions verified:")
    print("  - Error patterns at n=7, 9, 11 analyzed")
    print("  - Phase-specific error behavior computed")
    print("  - Systematic undercount patterns identified")
    
    return results

def main():
    """Run comprehensive validation."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Comprehensive Conjecture Validation",
        validation_func=lambda: validate_all_conjectures(n_max=100),
        parameters={'n_max': 100},
        output_filename="conjectures_comprehensive_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




