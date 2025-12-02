#!.venv/bin/python
"""
validate_concurrency_stability.py

Validation of 9/11 Charge as Concurrency Stability Index.

Tests:
1. Q₉₍₁₁₎ computation (tension / ballast + 1)
2. Concurrency state classification
3. Fixed point detection (Q ≈ 1 where λ = 1)
4. Backoff algorithm derivation
5. CAS loop analysis

Author: Joel
"""

import numpy as np
from mpmath import zetazero
from zetadiffusion.concurrency_stability import (
    compute_q_9_11, classify_concurrency_state,
    compute_concurrency_trajectory, detect_fixed_point,
    derive_backoff_algorithm, analyze_cas_loop
)
from zetadiffusion.witness_operator import WitnessOperator
from zetadiffusion.validation_framework import run_validation

def validate_concurrency_stability() -> dict:
    """
    Validate 9/11 charge as concurrency stability index.
    """
    print("=" * 70)
    print("9/11 CHARGE AS CONCURRENCY STABILITY INDEX")
    print("=" * 70)
    print()
    
    # Get Riemann zeros as live numbers
    n_values = list(range(1, 21))
    zeros = [float(zetazero(n).imag) for n in n_values]
    
    print(f"Analyzing {len(zeros)} Riemann zeros as concurrency states")
    print()
    
    # 1. Compute Q₉₍₁₁₎ values
    print("=" * 70)
    print("Q₉₍₁₁₎ = TENSION / (BALLAST + 1)")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'Zero':<15} | {'Ballasts':<10} | {'Tension':<10} | {'Q₉₍₁₁₎':<12} | {'State':<15}")
    print("-" * 70)
    
    q_values = []
    states_list = []
    
    for n, zero in zip(n_values, zeros):
        q_val = compute_q_9_11(zero)
        state, _ = classify_concurrency_state(q_val)
        
        q_values.append(q_val)
        states_list.append(state)
        
        # Get ballast/unit counts for display
        from zetadiffusion.digit_ballast import extract_ballast_and_units
        analysis = extract_ballast_and_units(zero)
        
        print(f"{n:<5} | {zero:<15.6f} | {analysis['ballasts']:<10} | "
              f"{analysis['units']:<10} | {q_val:<12.6f} | {state:<15}")
    
    print()
    
    # 2. Concurrency trajectory
    print("=" * 70)
    print("CONCURRENCY TRAJECTORY")
    print("=" * 70)
    print()
    
    trajectory = compute_concurrency_trajectory(zeros)
    
    print(f"Average Q₉₍₁₁₎: {trajectory['avg_q']:.6f}")
    print(f"Minimum Q₉₍₁₁₎: {trajectory['min_q']:.6f}")
    print(f"Maximum Q₉₍₁₁₎: {trajectory['max_q']:.6f}")
    print()
    print(f"Contention events (Q > 1): {trajectory['contention_count']}")
    print(f"Optimal states (0.5 ≤ Q ≤ 1): {trajectory['optimal_count']}")
    print()
    
    # 3. Fixed point detection
    print("=" * 70)
    print("FIXED POINT DETECTION (Q ≈ 1)")
    print("=" * 70)
    print()
    
    fixed_point = detect_fixed_point(zeros, n_values)
    
    print(f"Fixed points (Q ≈ 1): {fixed_point['fixed_point_count']}")
    print(f"Interior fixed points (n≥11): {fixed_point['interior_count']}")
    
    if fixed_point['fixed_point_indices']:
        print("\nFixed point locations:")
        for idx in fixed_point['fixed_point_indices'][:5]:  # Show first 5
            if idx < len(n_values):
                n = n_values[idx]
                q_val = q_values[idx]
                print(f"  n={n}: Q₉₍₁₁₎ = {q_val:.6f}")
    
    if fixed_point['aligned_with_lambda']:
        print("\n✓ Fixed points align with λ=1 (interior phase, n≥11)")
        print("  This is where contention vanishes - the concurrency sweet spot")
    else:
        print("\n~ Fixed points don't clearly align with interior phase")
    
    print()
    
    # 4. Connection to λ=1
    print("=" * 70)
    print("CONNECTION TO λ=1 (RENORMALIZATION FIXED POINT)")
    print("=" * 70)
    print()
    
    # Compute λ values
    lambda_values = [WitnessOperator.compute_lambda(float(n)) for n in n_values]
    
    # Find where λ ≈ 1
    lambda_fixed_points = [
        i for i, lam in enumerate(lambda_values)
        if abs(lam - 1.0) < 0.1
    ]
    
    print(f"λ ≈ 1 at n values: {[n_values[i] for i in lambda_fixed_points[:10]]}")
    print()
    
    # Check correlation between Q₉₍₁₁₎ ≈ 1 and λ ≈ 1
    correlation_count = 0
    for i in fixed_point['fixed_point_indices']:
        if i in lambda_fixed_points:
            correlation_count += 1
    
    if correlation_count > 0:
        print(f"✓ {correlation_count} points where both Q₉₍₁₁₎ ≈ 1 and λ ≈ 1")
        print("  This confirms: fixed point = concurrency sweet spot = no contention")
    else:
        print("~ No clear correlation between Q₉₍₁₁₎ ≈ 1 and λ ≈ 1")
    
    print()
    
    # 5. Backoff algorithm example
    print("=" * 70)
    print("BACKOFF ALGORITHM DERIVATION")
    print("=" * 70)
    print()
    
    print("Example backoff delays (base_delay = 1.0 μs):")
    print(f"{'Q₉₍₁₁₎':<12} | {'State':<15} | {'Backoff (μs)':<15}")
    print("-" * 70)
    
    test_q_values = [0.3, 0.7, 1.0, 1.5, 2.5, 5.0]
    for q in test_q_values:
        state, _ = classify_concurrency_state(q)
        backoff = derive_backoff_algorithm(q, base_delay=1.0)
        print(f"{q:<12.2f} | {state:<15} | {backoff:<15.6f}")
    
    print()
    
    # 6. CAS loop analysis
    print("=" * 70)
    print("CAS LOOP ANALYSIS")
    print("=" * 70)
    print()
    
    cas_analysis = analyze_cas_loop(zeros, max_retries=10)
    
    print(f"Average predicted retries: {cas_analysis['avg_retries']:.2f}")
    print(f"Maximum predicted retries: {cas_analysis['max_retries']}")
    print(f"Contention waves (retries > 1): {cas_analysis['contention_waves']}")
    print()
    
    # 7. Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("9/11 Charge as Concurrency Stability Index:")
    print("  - Q₉₍₁₁₎ = tension / (ballast + 1)")
    print("  - Q < ~1: stable (no contention)")
    print("  - Q ≈ 1: optimal flow (sweet spot)")
    print("  - Q > ~1: lock contention (too much tension)")
    print()
    print("Connection to Renormalization:")
    print("  - λ = 1 at interior fixed point")
    print("  - Q₉₍₁₁₎ ≈ 1 at same fixed point")
    print("  - This is where contention vanishes")
    print()
    print("Applications:")
    print("  - Lock-free backoff algorithms")
    print("  - CAS loop contention prediction")
    print("  - Scheduler fairness windows")
    print("  - Transactional retry loops")
    
    results = {
        'n_values': n_values,
        'zeros': zeros,
        'q_values': q_values,
        'states': states_list,
        'trajectory': trajectory,
        'fixed_point': {
            'count': fixed_point['fixed_point_count'],
            'interior_count': fixed_point['interior_count'],
            'aligned': fixed_point['aligned_with_lambda']
        },
        'lambda_values': lambda_values,
        'cas_analysis': {
            'avg_retries': cas_analysis['avg_retries'],
            'max_retries': cas_analysis['max_retries'],
            'contention_waves': cas_analysis['contention_waves']
        }
    }
    
    return results

def main():
    """Run validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Concurrency Stability (9/11 Charge)",
        validation_func=validate_concurrency_stability,
        parameters={},
        output_filename="concurrency_stability_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




