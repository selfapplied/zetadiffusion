#!.venv/bin/python
"""
generate_diagnostic_plots.py

Generates diagnostic plots from validation results.
Creates 10 plots as specified in Notion diagnostic plot specifications.

Plots:
1. FEG Cascade: Coherence vs Chaos (divergence visualization)
2. FEG Cascade: Period vs Chaos (bifurcation detection)
3. Temperature Cascade: Entropy vs Temperature
4. Temperature Cascade: Period vs Temperature
5. Conjecture 9.1.1: Error vs n (linear)
6. Conjecture 9.1.1: Error vs n (log-log)
7. Conjecture Extended: Error scaling (asymptotic)
8. Operator Analysis: Summary dashboard
9. All Validations: Execution time comparison
10. All Validations: Status overview

Author: Joel
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams['figure.figsize'] = (12, 8)
matplotlib.rcParams['font.size'] = 10

def load_validation_data():
    """Load all validation result files."""
    data = {}
    out_dir = Path(".out")
    
    files = {
        'feg_cascade': 'feg_cascade_results.json',
        'temperature_cascade': 'temperature_cascade_results.json',
        'conjecture': 'conjecture_9_1_1_results.json',
        'conjecture_extended': 'conjecture_extended_results.json',
        'operator_analysis': 'operator_analysis_results.json'
    }
    
    for key, filename in files.items():
        filepath = out_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
        else:
            print(f"⚠ {filename} not found")
    
    return data

def plot_feg_coherence_vs_chaos(data, output_dir):
    """Plot 1: FEG Cascade - Coherence vs Chaos (divergence visualization)."""
    if 'feg_cascade' not in data:
        return None
    
    d = data['feg_cascade']
    chaos = np.array(d.get('chaos_values', []))
    coherence = np.array(d.get('coherence', []))
    
    if len(chaos) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot coherence
    ax.plot(chaos, coherence, 'b-', linewidth=2, label='Coherence', marker='o', markersize=4)
    
    # Highlight divergence region (chaos > 0.6)
    divergence_mask = chaos > 0.6
    if np.any(divergence_mask):
        ax.plot(chaos[divergence_mask], coherence[divergence_mask], 'r-', 
                linewidth=3, label='Divergence Region', alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=0.64, color='r', linestyle='--', alpha=0.5, label='Critical Threshold (χ=0.64)')
    
    ax.set_xlabel('Chaos Injection (χ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coherence', fontsize=12, fontweight='bold')
    ax.set_title('FEG Cascade: Coherence vs Chaos\n(Divergence at Critical Threshold)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale for y-axis if values are large
    if np.max(coherence) > 1000:
        ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'plot1_feg_coherence_vs_chaos.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_feg_period_vs_chaos(data, output_dir):
    """Plot 2: FEG Cascade - Period vs Chaos (bifurcation detection)."""
    if 'feg_cascade' not in data:
        return None
    
    d = data['feg_cascade']
    chaos = np.array(d.get('chaos_values', []))
    periods = np.array(d.get('periods', []))
    
    if len(chaos) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot periods
    ax.plot(chaos, periods, 'g-', linewidth=2, label='Period', marker='o', markersize=6)
    
    # Highlight expected Feigenbaum sequence: 1, 2, 4, 8, 16
    feigenbaum_periods = [1, 2, 4, 8, 16]
    for p in feigenbaum_periods:
        ax.axhline(y=p, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Chaos Injection (χ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Period', fontsize=12, fontweight='bold')
    ax.set_title('FEG Cascade: Period vs Chaos\n(No Bifurcations Detected)', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot2_feg_period_vs_chaos.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_temperature_entropy(data, output_dir):
    """Plot 3: Temperature Cascade - Entropy vs Temperature."""
    if 'temperature_cascade' not in data:
        return None
    
    d = data['temperature_cascade']
    temps = np.array(d.get('temperatures', []))
    entropies = np.array(d.get('entropies', []))
    
    if len(temps) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(temps, entropies, 'purple', linewidth=2, marker='o', markersize=5, label='Entropy H(T)')
    
    # Add saturation fit if available
    if 'entropy_analysis' in d:
        ea = d['entropy_analysis']
        if ea.get('fit_valid', False):
            H_inf = ea.get('H_inf', 0)
            tau = ea.get('tau', 1)
            fit_temps = np.linspace(temps[0], temps[-1], 100)
            fit_entropies = H_inf * (1 - np.exp(-fit_temps / tau))
            ax.plot(fit_temps, fit_entropies, 'r--', linewidth=2, 
                   label=f'Saturation Fit (H∞={H_inf:.2f}, τ={tau:.2f})')
    
    ax.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy H(T)', fontsize=12, fontweight='bold')
    ax.set_title('Temperature Cascade: Entropy Saturation\n(No Bifurcations Detected)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot3_temperature_entropy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_temperature_period(data, output_dir):
    """Plot 4: Temperature Cascade - Period vs Temperature."""
    if 'temperature_cascade' not in data:
        return None
    
    d = data['temperature_cascade']
    temps = np.array(d.get('temperatures', []))
    periods = np.array(d.get('periods', []))
    
    if len(temps) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(temps, periods, 'orange', linewidth=2, marker='o', markersize=6, label='Period')
    
    # Highlight expected Feigenbaum sequence
    for p in [1, 2, 4, 8]:
        ax.axhline(y=p, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Period', fontsize=12, fontweight='bold')
    ax.set_title('Temperature Cascade: Period vs Temperature\n(All Period-1)', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 4, 8])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot4_temperature_period.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_conjecture_error_linear(data, output_dir):
    """Plot 5: Conjecture 9.1.1 - Error vs n (linear scale)."""
    if 'conjecture' not in data:
        return None
    
    d = data['conjecture']
    n_values = np.array(d.get('n_values', []))
    errors = np.array(d.get('errors', []))
    
    if len(n_values) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_values, errors, 'b-', linewidth=2, marker='o', markersize=5, label='Absolute Error')
    
    # Add average error line
    if len(errors) > 0:
        avg_error = np.mean(errors)
        ax.axhline(y=avg_error, color='r', linestyle='--', 
                  label=f'Average Error: {avg_error:.2f}')
    
    ax.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Conjecture 9.1.1: Error vs n (Linear Scale)\n(After Scaling Fix)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot5_conjecture_error_linear.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_conjecture_error_loglog(data, output_dir):
    """Plot 6: Conjecture 9.1.1 - Error vs n (log-log scale)."""
    if 'conjecture' not in data:
        return None
    
    d = data['conjecture']
    n_values = np.array(d.get('n_values', []))
    errors = np.array(d.get('errors', []))
    
    if len(n_values) == 0 or np.any(n_values <= 0) or np.any(errors <= 0):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(n_values, errors, 'b-', linewidth=2, marker='o', markersize=5, label='Absolute Error')
    
    # Fit power law: error ~ n^α
    if len(n_values) > 5:
        log_n = np.log(n_values)
        log_err = np.log(errors)
        coeffs = np.polyfit(log_n, log_err, 1)
        alpha = coeffs[0]
        fit_errors = np.exp(coeffs[1]) * n_values ** alpha
        ax.loglog(n_values, fit_errors, 'r--', linewidth=2, 
                 label=f'Power Law Fit (α={alpha:.2f})')
    
    ax.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Conjecture 9.1.1: Error vs n (Log-Log Scale)\n(Error Scaling Analysis)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = output_dir / 'plot6_conjecture_error_loglog.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_conjecture_9_1_3_three_clocks(data, output_dir):
    """Plot: Conjecture 9.1.3 - Three-clock structure visualization."""
    result_file = Path(".out/conjecture_9_1_3_results.json")
    if not result_file.exists():
        return None
    
    with open(result_file, 'r') as f:
        d = json.load(f)
    
    n_values = np.array(d.get('n_values', []))
    actual_zeros = np.array(d.get('actual_zeros', []))
    formula_zeros = np.array(d.get('formula_zeros', []))
    errors = np.array(d.get('errors', []))
    phases = d.get('clock_phase', [])
    
    if len(n_values) == 0:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Color map for phases
    phase_colors = {
        'feigenbaum': 'red',
        'boundary': 'orange',
        'membrane': 'purple',
        'interior': 'green'
    }
    
    # Plot 1: Actual vs Formula zeros with clock phase coloring
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        phase_mask = np.array([p == phase for p in phases])
        if np.any(phase_mask):
            color = phase_colors[phase]
            ax1.plot(n_values[phase_mask], actual_zeros[phase_mask], 
                    'o', color=color, markersize=8, label=f'Actual ({phase.capitalize()})', 
                    alpha=0.7, markeredgewidth=2)
            ax1.plot(n_values[phase_mask], formula_zeros[phase_mask], 
                    '--', color=color, linewidth=2, label=f'Formula ({phase.capitalize()})', 
                    alpha=0.7)
    
    # Highlight clock boundaries
    ax1.axvline(x=7, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Feigenbaum→Boundary')
    ax1.axvline(x=9, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Boundary→Membrane')
    ax1.axvline(x=11, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Membrane→Interior')
    
    ax1.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Zero Value', fontsize=12, fontweight='bold')
    ax1.set_title('Conjecture 9.1.3: Three-Clock Structure\n(crisis → membrane → ecology)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error by clock phase
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        phase_mask = np.array([p == phase for p in phases])
        if np.any(phase_mask):
            color = phase_colors[phase]
            ax2.plot(n_values[phase_mask], errors[phase_mask], 
                    '-', color=color, linewidth=2, marker='o', markersize=6, 
                    label=f'{phase.capitalize()} Error', alpha=0.7)
            
            # Add average error line
            avg_error = np.mean(errors[phase_mask])
            ax2.axhline(y=avg_error, color=color, linestyle='--', alpha=0.5,
                      label=f'{phase.capitalize()} Avg: {avg_error:.2f}')
    
    # Highlight clock boundaries
    ax2.axvline(x=7, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax2.axvline(x=9, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax2.axvline(x=11, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('Error by Clock Phase (Interior: 1.22% error - "Life" phase)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot12_conjecture_9_1_3_three_clocks.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_conjecture_9_1_2_bifurcation(data, output_dir):
    """Plot: Conjecture 9.1.2 - Bifurcation at n=9 visualization."""
    # Try to load 9.1.2 results if available
    result_file = Path(".out/conjecture_9_1_2_results.json")
    if not result_file.exists():
        return None
    
    with open(result_file, 'r') as f:
        d = json.load(f)
    
    n_values = np.array(d.get('n_values', []))
    actual_zeros = np.array(d.get('actual_zeros', []))
    formula_zeros = np.array(d.get('formula_zeros', []))
    errors = np.array(d.get('errors', []))
    regimes = d.get('regime', [])
    
    if len(n_values) == 0:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Actual vs Formula zeros with regime coloring
    periphery_mask = np.array([r == 'periphery' for r in regimes])
    interior_mask = np.array([r == 'interior' for r in regimes])
    
    ax1.plot(n_values[periphery_mask], actual_zeros[periphery_mask], 
            'go', markersize=8, label='Actual (Periphery)', alpha=0.7)
    ax1.plot(n_values[interior_mask], actual_zeros[interior_mask], 
            'bo', markersize=8, label='Actual (Interior)', alpha=0.7)
    ax1.plot(n_values[periphery_mask], formula_zeros[periphery_mask], 
            'g--', linewidth=2, label='Formula (Periphery)', alpha=0.7)
    ax1.plot(n_values[interior_mask], formula_zeros[interior_mask], 
            'b--', linewidth=2, label='Formula (Interior)', alpha=0.7)
    
    # Highlight bifurcation
    ax1.axvline(x=9, color='r', linestyle=':', linewidth=2, 
               label='Bifurcation (n=9)', alpha=0.7)
    
    ax1.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Zero Value', fontsize=12, fontweight='bold')
    ax1.set_title('Conjecture 9.1.2: Bifurcation at n=9\n(Pascal\'s Triangle Structure)', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error by regime
    ax2.plot(n_values[periphery_mask], errors[periphery_mask], 
            'g-', linewidth=2, marker='o', markersize=6, label='Periphery Error', alpha=0.7)
    ax2.plot(n_values[interior_mask], errors[interior_mask], 
            'b-', linewidth=2, marker='o', markersize=6, label='Interior Error', alpha=0.7)
    
    # Add average error lines
    if np.any(periphery_mask):
        avg_periphery = np.mean(errors[periphery_mask])
        ax2.axhline(y=avg_periphery, color='g', linestyle='--', alpha=0.5,
                   label=f'Periphery Avg: {avg_periphery:.2f}')
    
    if np.any(interior_mask):
        avg_interior = np.mean(errors[interior_mask])
        ax2.axhline(y=avg_interior, color='b', linestyle='--', alpha=0.5,
                   label=f'Interior Avg: {avg_interior:.2f}')
    
    ax2.axvline(x=9, color='r', linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('Error by Regime (Bifurcation Visible)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot11_conjecture_9_1_2_bifurcation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_conjecture_extended_error(data, output_dir):
    """Plot 7: Conjecture Extended - Error scaling (asymptotic analysis)."""
    if 'conjecture_extended' not in data:
        return None
    
    d = data['conjecture_extended']
    n_values = np.array(d.get('n_values', []))
    errors = np.array(d.get('errors', []))
    
    if len(n_values) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_values, errors, 'b-', linewidth=2, marker='o', markersize=3, label='Absolute Error')
    
    # Highlight early vs recent regions
    if len(n_values) > 20:
        early_n = n_values[:20]
        early_err = errors[:20]
        recent_n = n_values[-20:]
        recent_err = errors[-20:]
        
        ax.plot(early_n, early_err, 'g-', linewidth=3, alpha=0.7, label='Early (n=1-20)')
        ax.plot(recent_n, recent_err, 'r-', linewidth=3, alpha=0.7, label='Recent (n=81-100)')
        
        # Add trend lines
        if len(early_n) > 1:
            early_trend = np.polyfit(early_n, early_err, 1)
            ax.plot(early_n, np.polyval(early_trend, early_n), 'g--', alpha=0.5)
        
        if len(recent_n) > 1:
            recent_trend = np.polyfit(recent_n, recent_err, 1)
            ax.plot(recent_n, np.polyval(recent_trend, recent_n), 'r--', alpha=0.5)
    
    ax.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Conjecture Extended: Error Scaling (n=1..100)\n(Error Increases with n - Diverging)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot7_conjecture_extended_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_operator_summary(data, output_dir):
    """Plot 8: Operator Analysis - Summary dashboard."""
    if 'operator_analysis' not in data:
        return None
    
    d = data['operator_analysis']
    results = d.get('results', {})
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Subplot 1: Verification results
    ax1 = fig.add_subplot(gs[0, 0])
    verification = results.get('verification', {})
    categories = ['Gaussian\nFixed Point', 'Feigenbaum\nResidue', 'Logistic\nCorrespondence']
    values = [
        1 if verification.get('gaussian_fixed_point', {}).get('is_stable', False) else 0,
        1 if verification.get('feigenbaum_residue', {}).get('converged', False) else 0,
        0  # Logistic correspondence failed
    ]
    colors = ['green' if v == 1 else 'red' for v in values]
    ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Pass (1) / Fail (0)', fontweight='bold')
    ax1.set_title('Known Limits Verification', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Structural features
    ax2 = fig.add_subplot(gs[0, 1])
    features = ['Poles', 'Branch Cuts', 'Fixed Points', 'Eigenvalues']
    counts = [
        results.get('poles', {}).get('count', 0),
        results.get('branch_cuts', {}).get('count', 0),
        results.get('spectrum', {}).get('fixed_points', 0),
        results.get('spectrum', {}).get('eigenvalues', 0)
    ]
    ax2.bar(features, counts, color='orange', alpha=0.7)
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Structural Features Detected', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Critical exponents
    ax3 = fig.add_subplot(gs[1, 0])
    spectrum = results.get('spectrum', {})
    critical = spectrum.get('critical_exponents', {})
    exponents = ['δ_F', 'α_F']
    values = [
        critical.get('delta_feigenbaum', 0),
        critical.get('alpha_feigenbaum', 0)
    ]
    theoretical = [4.669, 2.503]
    x = np.arange(len(exponents))
    width = 0.35
    ax3.bar(x - width/2, values, width, label='Computed', color='blue', alpha=0.7)
    ax3.bar(x + width/2, theoretical, width, label='Theoretical', color='green', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(exponents)
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.set_title('Critical Exponents', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = f"""
    Operator Analysis Summary
    
    Execution Time: {d.get('execution_time', 0):.3f}s
    Status: {'✓ Success' if d.get('success', False) else '✗ Failed'}
    
    Known Limits: {'✓' if verification.get('all_passed', False) else '✗'}
    Structural Features: {sum(counts)} found
    Critical Exponents: {'✓ Match' if abs(values[0] - theoretical[0]) < 0.01 else '✗ Mismatch'}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Operator Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'plot8_operator_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_execution_times(data, output_dir):
    """Plot 9: All Validations - Execution time comparison."""
    execution_times = {}
    validation_names = []
    
    for key, d in data.items():
        name_map = {
            'feg_cascade': 'FEG Cascade',
            'temperature_cascade': 'Temperature Cascade',
            'conjecture': 'Conjecture 9.1.1',
            'conjecture_extended': 'Conjecture Extended',
            'operator_analysis': 'Operator Analysis'
        }
        if 'execution_time' in d:
            execution_times[name_map.get(key, key)] = d['execution_time']
            validation_names.append(name_map.get(key, key))
    
    if len(execution_times) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(execution_times.keys())
    times = list(execution_times.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax.barh(names, times, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (name, time) in enumerate(zip(names, times)):
        ax.text(time, i, f' {time:.3f}s', va='center', fontweight='bold')
    
    ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Validation Execution Times', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'plot9_execution_times.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_clock_execution_times(data, output_dir):
    """Plot: Clock execution times - computational signature of clock phases."""
    result_file = Path(".out/clock_execution_times.json")
    if not result_file.exists():
        return None
    
    with open(result_file, 'r') as f:
        d = json.load(f)
    
    n_values = np.array(d.get('n_values', []))
    comp_times = np.array(d.get('computation_times', []))
    phases = d.get('clock_phase', [])
    phase_stats = d.get('phase_stats', {})
    
    if len(n_values) == 0:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Color map for phases
    phase_colors = {
        'feigenbaum': 'red',
        'boundary': 'orange',
        'membrane': 'purple',
        'interior': 'green'
    }
    
    # Plot 1: Computation time by clock phase
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        phase_mask = np.array([p == phase for p in phases])
        if np.any(phase_mask):
            color = phase_colors[phase]
            ax1.plot(n_values[phase_mask], comp_times[phase_mask], 
                    '-', color=color, linewidth=2, marker='o', markersize=6, 
                    label=f'{phase.capitalize()} Clock', alpha=0.7)
            
            # Add mean line
            if phase in phase_stats:
                mean_time = phase_stats[phase]['mean']
                ax1.axhline(y=mean_time, color=color, linestyle='--', alpha=0.5,
                          label=f'{phase.capitalize()} Mean: {mean_time:.2f} ms')
    
    # Highlight clock boundaries
    ax1.axvline(x=7, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Clock Boundaries')
    ax1.axvline(x=9, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax1.axvline(x=11, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    # Highlight speedup at n=11 transition
    if len(comp_times) > 10:
        n10_idx = 9  # n=10 is index 9 (0-indexed)
        n11_idx = 10  # n=11 is index 10
        if n11_idx < len(comp_times):
            speedup = comp_times[n10_idx] - comp_times[n11_idx]
            ax1.annotate(f'Speedup at n=11:\n-{speedup:.2f} ms (-{speedup/comp_times[n10_idx]*100:.1f}%)',
                        xy=(11, comp_times[n11_idx]), xytext=(13, comp_times[n11_idx] + 0.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=10, fontweight='bold', color='green')
    
    ax1.set_xlabel('n (Zero Index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Clock Execution Times: Computational Signature\n(Interior clock shows speedup at transition)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase statistics comparison
    phases_list = ['feigenbaum', 'boundary', 'membrane', 'interior']
    means = [phase_stats[p]['mean'] for p in phases_list if p in phase_stats]
    stds = [phase_stats[p]['std'] for p in phases_list if p in phase_stats]
    colors_list = [phase_colors[p] for p in phases_list if p in phase_stats]
    labels_list = [p.capitalize() for p in phases_list if p in phase_stats]
    
    x_pos = np.arange(len(means))
    bars = ax2.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.7, 
                  capsize=5, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 0.1, f'{mean:.2f} ms', 
                ha='center', fontweight='bold', fontsize=10)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_list)
    ax2.set_ylabel('Mean Computation Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Computation Time by Clock Phase\n(Interior: 6.53 ms, Feigenbaum: 6.39 ms)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight interior clock
    if 'interior' in phase_stats:
        interior_idx = labels_list.index('Interior')
        bars[interior_idx].set_edgecolor('green')
        bars[interior_idx].set_linewidth(3)
    
    plt.tight_layout()
    output_path = output_dir / 'plot13_clock_execution_times.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_status_overview(data, output_dir):
    """Plot 10: All Validations - Status overview."""
    statuses = {}
    
    for key, d in data.items():
        name_map = {
            'feg_cascade': 'FEG Cascade',
            'temperature_cascade': 'Temperature Cascade',
            'conjecture': 'Conjecture 9.1.1',
            'conjecture_extended': 'Conjecture Extended',
            'operator_analysis': 'Operator Analysis'
        }
        name = name_map.get(key, key)
        success = d.get('success', False)
        statuses[name] = 'Completed' if success else 'Failed'
    
    if len(statuses) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(statuses.keys())
    status_values = [1 if s == 'Completed' else 0 for s in statuses.values()]
    colors = ['green' if v == 1 else 'red' for v in status_values]
    
    bars = ax.barh(names, status_values, color=colors, alpha=0.7)
    
    # Add status labels
    for i, (name, status) in enumerate(zip(names, statuses.values())):
        symbol = '✓' if status == 'Completed' else '✗'
        ax.text(0.5, i, f' {symbol} {status}', va='center', ha='center', 
               fontweight='bold', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title('Validation Status Overview', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'plot10_status_overview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Generate all diagnostic plots."""
    print("=" * 70)
    print("Generating Diagnostic Plots")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path(".out/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading validation data...")
    data = load_validation_data()
    print(f"Loaded {len(data)} validation datasets")
    print()
    
    # Generate plots
    plots = []
    
    print("Generating plots...")
    plot_funcs = [
        ("FEG Coherence vs Chaos", plot_feg_coherence_vs_chaos),
        ("FEG Period vs Chaos", plot_feg_period_vs_chaos),
        ("Temperature Entropy", plot_temperature_entropy),
        ("Temperature Period", plot_temperature_period),
        ("Conjecture Error (Linear)", plot_conjecture_error_linear),
        ("Conjecture Error (Log-Log)", plot_conjecture_error_loglog),
        ("Conjecture Extended Error", plot_conjecture_extended_error),
        ("Conjecture 9.1.2 Bifurcation", plot_conjecture_9_1_2_bifurcation),
        ("Conjecture 9.1.3 Three Clocks", plot_conjecture_9_1_3_three_clocks),
        ("Clock Execution Times", plot_clock_execution_times),
        ("Operator Summary", plot_operator_summary),
        ("Execution Times", plot_execution_times),
        ("Status Overview", plot_status_overview)
    ]
    
    for name, plot_func in plot_funcs:
        try:
            path = plot_func(data, output_dir)
            if path:
                plots.append((name, path))
                print(f"  ✓ {name}")
            else:
                print(f"  ⚠ {name} (no data)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    print()
    print("=" * 70)
    print(f"Generated {len(plots)} plots")
    print("=" * 70)
    print()
    print("Plot files:")
    for name, path in plots:
        print(f"  - {path.name}")
    print()
    print(f"All plots saved to: {output_dir}")
    
    return plots

if __name__ == "__main__":
    main()

