#!.venv/bin/python
"""
generate_conjecture_comparison_plots.py

Generate diagnostic plots comparing all three conjectures.

Plots:
1. Error trajectories (all three conjectures)
2. Phase transitions at n=7, 9, 11
3. Error slopes by phase
4. Improvement over baseline

Author: Joel
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_comprehensive_results():
    """Load comprehensive validation results."""
    results_file = Path(".out/conjectures_comprehensive_results.json")
    if not results_file.exists():
        raise FileNotFoundError("Run validate_conjectures_comprehensive.py first")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_error_trajectories(results, output_dir):
    """Plot error trajectories for all three conjectures."""
    n_values = results['n_values']
    
    errors_9_1_1 = results['conjecture_9_1_1']['rel_errors']
    errors_9_1_2 = results['conjecture_9_1_2']['rel_errors']
    errors_9_1_3 = results['conjecture_9_1_3']['rel_errors']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(n_values, errors_9_1_1, 'b-', label='Conjecture 9.1.1', linewidth=2, alpha=0.7)
    ax.plot(n_values, errors_9_1_2, 'r-', label='Conjecture 9.1.2', linewidth=2, alpha=0.7)
    ax.plot(n_values, errors_9_1_3, 'g-', label='Conjecture 9.1.3', linewidth=2, alpha=0.7)
    
    # Mark clock boundaries
    ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='n=7 (Boundary)')
    ax.axvline(x=9, color='gray', linestyle='--', alpha=0.5, label='n=9 (Membrane)')
    ax.axvline(x=11, color='gray', linestyle='--', alpha=0.5, label='n=11 (Interior)')
    
    ax.set_xlabel('n (Zero Index)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Conjecture Comparison: Error Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'conjecture_comparison_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_phase_transitions(results, output_dir):
    """Plot error behavior at phase transitions."""
    n_values = np.array(results['n_values'])
    
    errors_9_1_1 = np.array(results['conjecture_9_1_1']['rel_errors'])
    errors_9_1_2 = np.array(results['conjecture_9_1_2']['rel_errors'])
    errors_9_1_3 = np.array(results['conjecture_9_1_3']['rel_errors'])
    
    # Focus on transition region (n=1-20)
    mask = n_values <= 20
    n_trans = n_values[mask]
    e1_trans = errors_9_1_1[mask]
    e2_trans = errors_9_1_2[mask]
    e3_trans = errors_9_1_3[mask]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(n_trans, e1_trans, 'b-o', label='9.1.1', markersize=4, linewidth=2)
    ax.plot(n_trans, e2_trans, 'r-s', label='9.1.2', markersize=4, linewidth=2)
    ax.plot(n_trans, e3_trans, 'g-^', label='9.1.3', markersize=4, linewidth=2)
    
    # Highlight key points
    for n_key in [7, 9, 11]:
        if n_key in n_trans:
            idx = list(n_trans).index(n_key)
            ax.plot(n_key, e1_trans[idx], 'bo', markersize=10, alpha=0.5)
            ax.plot(n_key, e2_trans[idx], 'rs', markersize=10, alpha=0.5)
            ax.plot(n_key, e3_trans[idx], 'g^', markersize=10, alpha=0.5)
            ax.text(n_key, max(e1_trans[idx], e2_trans[idx], e3_trans[idx]) * 1.1,
                   f'n={n_key}', ha='center', fontsize=10, fontweight='bold')
    
    # Phase regions
    ax.axvspan(1, 7, alpha=0.1, color='blue', label='Feigenbaum')
    ax.axvspan(7, 9, alpha=0.1, color='orange', label='Boundary')
    ax.axvspan(9, 11, alpha=0.1, color='yellow', label='Membrane')
    ax.axvspan(11, 20, alpha=0.1, color='green', label='Interior')
    
    ax.set_xlabel('n (Zero Index)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Phase Transitions: Error Behavior at n=7, 9, 11', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'conjecture_phase_transitions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_improvement_analysis(results, output_dir):
    """Plot improvement of 9.1.2 and 9.1.3 over 9.1.1."""
    n_values = np.array(results['n_values'])
    
    errors_9_1_1 = np.array(results['conjecture_9_1_1']['rel_errors'])
    errors_9_1_2 = np.array(results['conjecture_9_1_2']['rel_errors'])
    errors_9_1_3 = np.array(results['conjecture_9_1_3']['rel_errors'])
    
    # Compute improvement ratios
    improvement_9_1_2 = ((errors_9_1_1 - errors_9_1_2) / errors_9_1_1) * 100
    improvement_9_1_3 = ((errors_9_1_1 - errors_9_1_3) / errors_9_1_1) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(n_values, improvement_9_1_2, 'r-', label='9.1.2 vs 9.1.1', linewidth=2, alpha=0.7)
    ax.plot(n_values, improvement_9_1_3, 'g-', label='9.1.3 vs 9.1.1', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Mark clock boundaries
    ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=9, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=11, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('n (Zero Index)', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Improvement Analysis: 9.1.2 and 9.1.3 vs 9.1.1', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'conjecture_improvement.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Generate all comparison plots."""
    output_dir = Path(".out")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading comprehensive results...")
    results = load_comprehensive_results()
    
    print("Generating plots...")
    
    plot1 = plot_error_trajectories(results, output_dir)
    print(f"✓ Error trajectories: {plot1}")
    
    plot2 = plot_phase_transitions(results, output_dir)
    print(f"✓ Phase transitions: {plot2}")
    
    plot3 = plot_improvement_analysis(results, output_dir)
    print(f"✓ Improvement analysis: {plot3}")
    
    print("\n✅ All plots generated!")

if __name__ == "__main__":
    main()

