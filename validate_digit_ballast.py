#!.venv/bin/python
"""
validate_digit_ballast.py

Validation of Digit-Based Ballast Analysis (9/11 Conjecture).

Tests:
1. Ballast counting (0's)
2. Unit counting (9's as 1's)
3. 9/11 pattern detection
4. Ballast trajectory analysis

Author: Joel
"""

import json
import numpy as np
from pathlib import Path
from mpmath import zetazero
from zetadiffusion.digit_ballast import (
    extract_ballast_and_units,
    analyze_live_number_sequence,
    compute_ballast_trajectory,
    detect_9_11_pattern
)
from zetadiffusion.validation_framework import run_validation

def validate_digit_ballast() -> dict:
    """
    Validate digit-based ballast analysis.
    """
    print("=" * 70)
    print("DIGIT-BASED BALLAST ANALYSIS (9/11 CONJECTURE)")
    print("=" * 70)
    print()
    
    # Get Riemann zeros as "live numbers"
    n_values = list(range(1, 21))
    zeros = [float(zetazero(n).imag) for n in n_values]
    
    print(f"Analyzing {len(zeros)} Riemann zeros as live numbers")
    print()
    
    # 1. Analyze individual numbers
    print("=" * 70)
    print("BALLAST AND UNIT ANALYSIS")
    print("=" * 70)
    print()
    print(f"{'n':<5} | {'Zero':<15} | {'Ballasts (0)':<12} | {'Units (9→1)':<12} | {'Ballast Ratio':<12} | {'Unit Ratio':<12}")
    print("-" * 70)
    
    analyses = analyze_live_number_sequence(zeros)
    
    for i, (n, analysis) in enumerate(zip(n_values, analyses)):
        print(f"{n:<5} | {analysis['value']:<15.6f} | {analysis['ballasts']:<12} | "
              f"{analysis['units']:<12} | {analysis['ballast_ratio']:<12.6f} | "
              f"{analysis['unit_ratio']:<12.6f}")
    
    print()
    
    # 2. Compute trajectory
    print("=" * 70)
    print("BALLAST TRAJECTORY")
    print("=" * 70)
    print()
    
    trajectory = compute_ballast_trajectory(zeros)
    
    print(f"Total ballasts (0's): {trajectory['total_ballasts']}")
    print(f"Total units (9's): {trajectory['total_units']}")
    print(f"Average ballasts per number: {trajectory['avg_ballast']:.2f}")
    print(f"Average units per number: {trajectory['avg_units']:.2f}")
    print()
    
    # 3. Detect 9/11 pattern
    print("=" * 70)
    print("9/11 PATTERN DETECTION")
    print("=" * 70)
    print()
    
    pattern = detect_9_11_pattern(zeros, n_values)
    
    print(f"Ballast at n=9: {pattern['n9_ballast_avg']:.2f}")
    print(f"Units at n=11: {pattern['n11_units_avg']:.2f}")
    print()
    
    if pattern['has_ballast_at_9']:
        print("✓ Ballast accumulation detected at n=9")
    else:
        print("~ No clear ballast accumulation at n=9")
    
    if pattern['has_units_at_11']:
        print("✓ Unit activation detected at n=11")
    else:
        print("~ No clear unit activation at n=11")
    
    if pattern['pattern_detected']:
        print("\n✓ 9/11 pattern detected!")
        print("  Ballasts (0's) accumulate at n=9 (stabilization)")
        print("  Units (9's as 1's) activate at n=11 (interior clock)")
    else:
        print("\n~ 9/11 pattern not clearly detected")
    
    print()
    
    # 4. Detailed analysis at key points
    print("=" * 70)
    print("KEY POINT ANALYSIS")
    print("=" * 70)
    print()
    
    key_points = [1, 7, 9, 11, 15, 20]
    for n in key_points:
        if n in n_values:
            idx = n_values.index(n)
            analysis = analyses[idx]
            zero = zeros[idx]
            
            print(f"n={n}:")
            print(f"  Zero: {zero:.10f}")
            print(f"  Ballasts (0's): {analysis['ballasts']}")
            print(f"  Units (9's): {analysis['units']}")
            print(f"  Ballast ratio: {analysis['ballast_ratio']:.4f}")
            print(f"  Unit ratio: {analysis['unit_ratio']:.4f}")
            
            # Show digit distribution
            digit_counts = analysis['digit_counts']
            non_zero_digits = {d: c for d, c in digit_counts.items() if c > 0 and d != '0'}
            if non_zero_digits:
                top_digits = sorted(non_zero_digits.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top digits: {', '.join([f'{d}:{c}' for d, c in top_digits])}")
            print()
    
    # 5. Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("9/11 Conjecture Interpretation:")
    print("  - 0's act as 'ballasts' (stabilizing elements)")
    print("  - 9's act as '1's' (active/unit elements)")
    print("  - Live numbers encode stability/activity in their digit structure")
    print()
    
    if pattern['pattern_detected']:
        print("✓ Pattern confirmed:")
        print(f"  - Ballast accumulation (0's) at n=9: {pattern['n9_ballast_avg']:.2f}")
        print(f"  - Unit activation (9's) at n=11: {pattern['n11_units_avg']:.2f}")
        print("  - This matches the three-clock structure:")
        print("    * n=9: Boundary/membrane phase (ballast loading)")
        print("    * n=11: Interior phase (unit activation)")
    
    results = {
        'n_values': n_values,
        'zeros': zeros,
        'analyses': [
            {
                'n': n,
                'value': a['value'],
                'ballasts': a['ballasts'],
                'units': a['units'],
                'ballast_ratio': a['ballast_ratio'],
                'unit_ratio': a['unit_ratio']
            }
            for n, a in zip(n_values, analyses)
        ],
        'trajectory': {
            'total_ballasts': trajectory['total_ballasts'],
            'total_units': trajectory['total_units'],
            'avg_ballast': trajectory['avg_ballast'],
            'avg_units': trajectory['avg_units']
        },
        'pattern': {
            'n9_ballast_avg': pattern['n9_ballast_avg'],
            'n11_units_avg': pattern['n11_units_avg'],
            'pattern_detected': pattern['pattern_detected']
        }
    }
    
    return results

def main():
    """Run validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    return run_validation(
        validation_type="Digit Ballast Analysis",
        validation_func=validate_digit_ballast,
        parameters={},
        output_filename="digit_ballast_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




