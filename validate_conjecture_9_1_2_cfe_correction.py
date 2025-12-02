"""
validate_conjecture_9_1_2_cfe_correction.py

Use Continued Fraction Expansion (CFE) to refine Conjecture 9.1.2 predictions.

Strategy:
1. Compute actual/predicted ratios for known zeros
2. Find CFE of these ratios
3. Use CFE convergents to derive correction factors
4. Apply corrections to improve predictions

Author: Joel
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from zetadiffusion.cfe_gini import continued_fraction_expansion

# Import the three-term formula
import sys
sys.path.insert(0, '.')
from validate_conjecture_9_1_2_three_term import (
    riemann_zero_conjecture_9_1_2,
    ACTUAL_ZEROS
)

DELTA_F = 4.669201609102990
ALPHA_F = 2.502907875095892


def cfe_to_convergent(cfe: list, depth: int = None) -> float:
    """
    Convert continued fraction [a_0; a_1, a_2, ...] to rational convergent.
    
    Uses recurrence: h_n = a_n * h_{n-1} + h_{n-2}
    """
    if not cfe:
        return 0.0
    
    if depth is None:
        depth = len(cfe)
    else:
        depth = min(depth, len(cfe))
    
    if depth == 0:
        return 0.0
    
    # Initialize
    h_prev2, h_prev1 = 0, 1
    k_prev2, k_prev1 = 1, 0
    
    for i in range(depth):
        a_i = cfe[i]
        h_curr = a_i * h_prev1 + h_prev2
        k_curr = a_i * k_prev1 + k_prev2
        
        h_prev2, h_prev1 = h_prev1, h_curr
        k_prev2, k_prev1 = k_prev1, k_curr
    
    if k_prev1 == 0:
        return float('inf')
    
    return h_prev1 / k_prev1


def compute_ratio_cfe(n_max: int = 30) -> dict:
    """
    Compute actual/predicted ratios and their CFE.
    
    Returns:
        Dictionary with ratios, CFEs, and analysis
    """
    results = []
    
    for n in range(1, min(n_max + 1, len(ACTUAL_ZEROS) + 1)):
        actual = ACTUAL_ZEROS[n-1]
        predicted = riemann_zero_conjecture_9_1_2(n)
        
        if predicted > 0:
            ratio = actual / predicted
        else:
            ratio = 1.0
        
        # Compute CFE of ratio
        cfe = continued_fraction_expansion(ratio, max_terms=10)
        
        # Get first few convergents
        convergents = []
        for depth in range(1, min(len(cfe) + 1, 5)):
            conv = cfe_to_convergent(cfe, depth)
            convergents.append(conv)
        
        results.append({
            'n': n,
            'actual': actual,
            'predicted': predicted,
            'ratio': ratio,
            'cfe': cfe,
            'convergents': convergents,
            'error': abs(actual - predicted) / actual * 100
        })
    
    return {'results': results}


def find_correction_pattern(ratio_data: dict) -> dict:
    """
    Analyze CFE patterns to find systematic correction factors.
    
    Strategy:
    - Look for common CFE prefixes across n values
    - Identify regime-specific corrections
    - Find universal correction constants
    """
    results = ratio_data['results']
    
    # Group by regime
    regime_ratios = {
        'dynamics': [],      # n < 7
        'ballast': [],       # 7 <= n < 9
        'transition': [],    # 9 <= n < 12
        'emergence': []      # n >= 12
    }
    
    for r in results:
        n = r['n']
        ratio = r['ratio']
        
        if n < 7:
            regime_ratios['dynamics'].append(ratio)
        elif n < 9:
            regime_ratios['ballast'].append(ratio)
        elif n < 12:
            regime_ratios['transition'].append(ratio)
        else:
            regime_ratios['emergence'].append(ratio)
    
    # Compute mean ratios per regime
    regime_means = {}
    for regime, ratios in regime_ratios.items():
        if ratios:
            regime_means[regime] = np.mean(ratios)
            regime_means[f'{regime}_cfe'] = continued_fraction_expansion(
                np.mean(ratios), max_terms=8
            )
    
    # Find universal correction (overall mean)
    all_ratios = [r['ratio'] for r in results]
    universal_ratio = np.mean(all_ratios)
    universal_cfe = continued_fraction_expansion(universal_ratio, max_terms=10)
    
    return {
        'regime_means': regime_means,
        'universal_ratio': universal_ratio,
        'universal_cfe': universal_cfe,
        'universal_convergent': cfe_to_convergent(universal_cfe, depth=3)
    }


def apply_cfe_correction(n: int, base_prediction: float, correction_data: dict) -> float:
    """
    Apply CFE-derived correction to base prediction.
    
    Uses regime-specific or universal correction based on n.
    """
    # Determine regime
    if n < 7:
        regime = 'dynamics'
    elif n < 9:
        regime = 'ballast'
    elif n < 12:
        regime = 'transition'
    else:
        regime = 'emergence'
    
    # Get regime-specific correction if available
    regime_means = correction_data['regime_means']
    if f'{regime}_cfe' in regime_means:
        cfe = regime_means[f'{regime}_cfe']
        correction = cfe_to_convergent(cfe, depth=3)
    else:
        # Fall back to universal correction
        correction = correction_data['universal_convergent']
    
    return base_prediction * correction


def validate_with_cfe_correction(n_max: int = 30) -> dict:
    """
    Run validation with CFE-derived corrections.
    """
    print("="*80)
    print("CONJECTURE 9.1.2 WITH CFE CORRECTION")
    print("="*80)
    print()
    
    # Step 1: Compute ratios and CFEs
    print("Step 1: Computing actual/predicted ratios and CFEs...")
    ratio_data = compute_ratio_cfe(n_max)
    
    # Step 2: Find correction patterns
    print("Step 2: Analyzing CFE patterns for corrections...")
    correction_data = find_correction_pattern(ratio_data)
    
    print(f"\nUniversal correction ratio: {correction_data['universal_ratio']:.6f}")
    print(f"Universal CFE: {correction_data['universal_cfe'][:8]}")
    print(f"Universal convergent (depth 3): {correction_data['universal_convergent']:.6f}")
    print()
    
    print("Regime-specific corrections:")
    for regime in ['dynamics', 'ballast', 'transition', 'emergence']:
        if f'{regime}_cfe' in correction_data['regime_means']:
            mean_ratio = correction_data['regime_means'][regime]
            cfe = correction_data['regime_means'][f'{regime}_cfe']
            convergent = cfe_to_convergent(cfe, depth=3)
            print(f"  {regime:12s}: mean_ratio={mean_ratio:.6f}, convergent={convergent:.6f}")
    print()
    
    # Step 3: Apply corrections and validate
    print("Step 3: Applying CFE corrections...")
    print("="*80)
    print()
    
    corrected_results = []
    
    for n in range(1, min(n_max + 1, len(ACTUAL_ZEROS) + 1)):
        actual = ACTUAL_ZEROS[n-1]
        base_pred = riemann_zero_conjecture_9_1_2(n)
        corrected_pred = apply_cfe_correction(n, base_pred, correction_data)
        
        error_before = abs(actual - base_pred) / actual * 100
        error_after = abs(actual - corrected_pred) / actual * 100
        improvement = error_before - error_after
        
        corrected_results.append({
            'n': n,
            'actual': actual,
            'base_pred': base_pred,
            'corrected_pred': corrected_pred,
            'error_before': error_before,
            'error_after': error_after,
            'improvement': improvement
        })
        
        # Regime indicator
        if n < 7:
            regime = "I:Dynamics"
        elif n < 9:
            regime = "II:Ballast"
        elif n < 12:
            regime = "II+III:Trans"
        else:
            regime = "III:Emerge"
        
        print(f"n={n:2d} [{regime:12s}]: "
              f"base={base_pred:7.2f}, corrected={corrected_pred:7.2f}, actual={actual:7.2f}")
        print(f"      Error: {error_before:6.2f}% → {error_after:6.2f}% "
              f"(improvement: {improvement:+.2f}%)")
        print()
    
    # Summary statistics
    errors_before = [r['error_before'] for r in corrected_results]
    errors_after = [r['error_after'] for r in corrected_results]
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mean error (before): {np.mean(errors_before):.2f}%")
    print(f"Mean error (after):  {np.mean(errors_after):.2f}%")
    print(f"Improvement:        {np.mean(errors_before) - np.mean(errors_after):.2f}%")
    print()
    
    return {
        'ratio_data': ratio_data,
        'correction_data': correction_data,
        'corrected_results': corrected_results
    }


def plot_cfe_correction_results(data: dict):
    """Generate plots showing CFE correction effectiveness."""
    results = data['corrected_results']
    
    n_vals = [r['n'] for r in results]
    errors_before = [r['error_before'] for r in results]
    errors_after = [r['error_after'] for r in results]
    improvements = [r['improvement'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error comparison
    axes[0, 0].plot(n_vals, errors_before, 'o-', label='Before CFE correction', 
                    linewidth=2, markersize=6, color='red', alpha=0.7)
    axes[0, 0].plot(n_vals, errors_after, 's-', label='After CFE correction', 
                    linewidth=2, markersize=6, color='green', alpha=0.7)
    axes[0, 0].axvline(7, color='orange', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(9, color='red', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(11, color='green', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Zero Index n')
    axes[0, 0].set_ylabel('Relative Error (%)')
    axes[0, 0].set_title('Error: Before vs After CFE Correction')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Improvement
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[0, 1].bar(n_vals, improvements, color=colors, alpha=0.6)
    axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel('Zero Index n')
    axes[0, 1].set_ylabel('Error Improvement (%)')
    axes[0, 1].set_title('CFE Correction Improvement')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Actual vs Corrected
    actual = [r['actual'] for r in results]
    corrected = [r['corrected_pred'] for r in results]
    axes[1, 0].scatter(actual, corrected, s=50, alpha=0.6, color='green')
    axes[1, 0].plot([min(actual), max(actual)], [min(actual), max(actual)], 
                    'r--', linewidth=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual Zero Location')
    axes[1, 0].set_ylabel('Corrected Prediction')
    axes[1, 0].set_title('Actual vs CFE-Corrected Prediction')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Ratio distribution
    ratios = [r['actual'] / r['base_pred'] for r in results]
    axes[1, 1].plot(n_vals, ratios, 'o-', linewidth=2, markersize=6)
    axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Ideal ratio = 1')
    axes[1, 1].axhline(np.mean(ratios), color='green', linestyle='--', 
                       label=f'Mean ratio = {np.mean(ratios):.3f}')
    axes[1, 1].set_xlabel('Zero Index n')
    axes[1, 1].set_ylabel('Actual / Predicted Ratio')
    axes[1, 1].set_title('Prediction Ratio (should be ~1)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run validation with CFE correction
    data = validate_with_cfe_correction(n_max=30)
    
    # Generate plots
    fig = plot_cfe_correction_results(data)
    plt.savefig('conjecture_9_1_2_cfe_correction.png', dpi=150, bbox_inches='tight')
    
    print()
    print("="*80)
    print("Validation complete. Plot saved to 'conjecture_9_1_2_cfe_correction.png'")
    print("="*80)

