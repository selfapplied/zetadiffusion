#!.venv/bin/python
"""
analyze_clock_execution_times.py

Analyzes whether execution times correlate with the three-clock structure.

Hypothesis: Computational complexity may change at clock boundaries.
- Feigenbaum clock: Fast computation (simple dynamics)
- Boundary clock: Slower (membrane formation adds complexity)
- Membrane transition: Slower (halocline computation)
- Interior clock: Fastest? (self-sustaining structure optimized)

Author: Joel
"""

import json
import numpy as np
from pathlib import Path
from mpmath import zetazero
import time

# Clock boundaries
N_FEIGENBAUM_MAX = 7.0
N_BOUNDARY_END = 9.0
N_MEMBRANE_END = 11.0

def measure_zero_computation_time(n: int, n_samples: int = 5) -> float:
    """
    Measure average time to compute zeta zero n.
    """
    times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        _ = float(zetazero(n).imag)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)

def analyze_clock_execution_times(n_max: int = 20) -> dict:
    """
    Analyze execution times by clock phase.
    """
    results = {
        'n_values': [],
        'computation_times': [],
        'clock_phase': [],
        'phase_stats': {}
    }
    
    print("=" * 70)
    print("CLOCK EXECUTION TIME ANALYSIS")
    print("=" * 70)
    print()
    print("Measuring zeta zero computation times by clock phase...")
    print()
    print(f"{'n':<6} | {'Clock Phase':<15} | {'Time (ms)':<12} | {'Cumulative':<12}")
    print("-" * 70)
    
    cumulative_time = 0.0
    
    for n in range(1, n_max + 1):
        # Determine clock phase
        if n < N_FEIGENBAUM_MAX:
            phase = 'feigenbaum'
        elif n < N_BOUNDARY_END:
            phase = 'boundary'
        elif n < N_MEMBRANE_END:
            phase = 'membrane'
        else:
            phase = 'interior'
        
        # Measure computation time
        comp_time = measure_zero_computation_time(n, n_samples=3)
        comp_time_ms = comp_time * 1000  # Convert to milliseconds
        cumulative_time += comp_time_ms
        
        results['n_values'].append(n)
        results['computation_times'].append(comp_time_ms)
        results['clock_phase'].append(phase)
        
        # Highlight transitions
        if n == int(N_FEIGENBAUM_MAX):
            marker = " ⚡"
        elif n == int(N_BOUNDARY_END):
            marker = " 🌀"
        elif n == int(N_MEMBRANE_END):
            marker = " ✨"
        else:
            marker = ""
        
        print(f"{n:<6} | {phase.capitalize():<15} | {comp_time_ms:>11.3f} | {cumulative_time:>11.3f}{marker}")
    
    # Calculate statistics by phase
    times = np.array(results['computation_times'])
    
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        phase_mask = np.array([p == phase for p in results['clock_phase']])
        if np.any(phase_mask):
            phase_times = times[phase_mask]
            results['phase_stats'][phase] = {
                'mean': float(np.mean(phase_times)),
                'median': float(np.median(phase_times)),
                'std': float(np.std(phase_times)),
                'min': float(np.min(phase_times)),
                'max': float(np.max(phase_times)),
                'count': int(np.sum(phase_mask))
            }
    
    print()
    print("=" * 70)
    print("STATISTICS BY CLOCK PHASE")
    print("=" * 70)
    print()
    
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        if phase in results['phase_stats']:
            stats = results['phase_stats'][phase]
            print(f"{phase.capitalize()} Clock (n={stats['count']} points):")
            print(f"  Mean:   {stats['mean']:.3f} ms")
            print(f"  Median: {stats['median']:.3f} ms")
            print(f"  Std:    {stats['std']:.3f} ms")
            print(f"  Range:  {stats['min']:.3f} - {stats['max']:.3f} ms")
            print()
    
    # Check for correlation
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Compare phases
    if 'feigenbaum' in results['phase_stats'] and 'interior' in results['phase_stats']:
        feigenbaum_mean = results['phase_stats']['feigenbaum']['mean']
        interior_mean = results['phase_stats']['interior']['mean']
        ratio = interior_mean / feigenbaum_mean if feigenbaum_mean > 0 else 0
        
        print(f"Interior / Feigenbaum time ratio: {ratio:.3f}")
        if ratio < 1.0:
            print("  → Interior clock is FASTER (self-sustaining structure optimized)")
        elif ratio > 1.0:
            print("  → Interior clock is SLOWER (more complex computation)")
        else:
            print("  → No significant difference")
        print()
    
    # Check for jumps at boundaries
    print("Time jumps at clock boundaries:")
    for i in range(1, len(results['n_values'])):
        n = results['n_values'][i]
        prev_n = results['n_values'][i-1]
        prev_phase = results['clock_phase'][i-1]
        curr_phase = results['clock_phase'][i]
        
        if prev_phase != curr_phase:
            prev_time = results['computation_times'][i-1]
            curr_time = results['computation_times'][i]
            jump = curr_time - prev_time
            jump_pct = (jump / prev_time * 100) if prev_time > 0 else 0
            
            print(f"  n={prev_n}→{n} ({prev_phase}→{curr_phase}): "
                  f"{jump:+.3f} ms ({jump_pct:+.1f}%)")
    
    results['success'] = True
    return results

def main():
    """Run clock execution time analysis."""
    results = analyze_clock_execution_times(n_max=20)
    
    # Save results
    output_file = Path(".out/clock_execution_times.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"✓ Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()

