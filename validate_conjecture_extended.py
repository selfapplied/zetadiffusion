#!.venv/bin/python
"""
validate_conjecture_extended.py

Extended Conjecture 9.1.1 validation to n=100 for asymptotic analysis.
Tests convergence of correction term and formula accuracy.

Author: Joel
"""

from validate_conjecture_9_1_1 import verify_conjecture
from zetadiffusion.validation_framework import run_validation

def main():
    """Run extended conjecture validation using shared framework."""
    def run_extended():
        print("=" * 70)
        print("EXTENDED CONJECTURE 9.1.1 VALIDATION")
        print("Testing n ∈ [1, 100] for asymptotic convergence")
        print("=" * 70)
        
        # Run extended validation
        results = verify_conjecture(n_max=100)
        results['n_max'] = 100
        
        # Analyze asymptotic behavior
        if 'errors' in results and len(results['errors']) > 20:
            # Check if error decreases with n
            recent_errors = results['errors'][-20:]
            early_errors = results['errors'][:20]
            
            avg_recent = sum(recent_errors) / len(recent_errors)
            avg_early = sum(early_errors) / len(early_errors)
            
            results['asymptotic_analysis'] = {
                'early_avg_error': float(avg_early),
                'recent_avg_error': float(avg_recent),
                'error_reduction': float((avg_early - avg_recent) / avg_early if avg_early > 0 else 0),
                'converging': bool(avg_recent < avg_early)
            }
            
            print(f"\n{'='*70}")
            print("ASYMPTOTIC ANALYSIS")
            print(f"{'='*70}")
            print(f"Early (n=1-20) avg error: {avg_early:.4f}")
            print(f"Recent (n=81-100) avg error: {avg_recent:.4f}")
            print(f"Error reduction: {results['asymptotic_analysis']['error_reduction']*100:.1f}%")
            print(f"Converging: {results['asymptotic_analysis']['converging']}")
        
        return results
    
    return run_validation(
        validation_type="Conjecture 9.1.1 Extended",
        validation_func=run_extended,
        parameters={'n_max': 100},
        output_filename="conjecture_extended_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

