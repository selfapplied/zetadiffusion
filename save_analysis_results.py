#!.venv/bin/python
"""
save_analysis_results.py

Saves operator analysis results to JSON for Notion upload.
Uses shared validation framework.
"""

from zetadiffusion.operator_analysis import run_all_analysis
from zetadiffusion.validation_framework import run_validation

def convert_results(results):
    """Convert operator analysis results to JSON-serializable format."""
    return {
        "verification": {
            "all_passed": bool(results["verification"].all_passed),
            "gaussian_fixed_point": {
                "is_stable": bool(results["verification"].gaussian_fixed_point["is_stable"]),
                "coherence_change": float(results["verification"].gaussian_fixed_point["coherence_change"])
            },
            "feigenbaum_residue": {
                "chaos_residue": float(results["verification"].feigenbaum_residue["chaos_residue"]),
                "converged": bool(results["verification"].feigenbaum_residue["converged"])
            },
            "logistic_correspondence": {
                "periods_detected": int(results["verification"].logistic_correspondence["periods_detected"]),
                "correspondence_error": float(results["verification"].logistic_correspondence["correspondence_error"])
            }
        },
        "branch_cuts": {
            "count": len(results["branch_cuts"].branch_cuts),
            "phase_discontinuities": len(results["branch_cuts"].phase_discontinuities)
        },
        "poles": {
            "count": len(results["poles"].poles),
            "real_axis": len(results["poles"].pole_locations["real_axis"]),
            "imaginary_axis": len(results["poles"].pole_locations["imaginary_axis"]),
            "complex_plane": len(results["poles"].pole_locations["complex_plane"])
        },
        "spectrum": {
            "fixed_points": len(results["spectrum"].fixed_points),
            "eigenvalues": len(results["spectrum"].eigenvalues),
            "feigenbaum_eigenvals": len(results["spectrum"].feigenbaum_eigenvalues),
            "critical_exponents": results["spectrum"].critical_exponents
        }
    }

def main():
    """Run operator analysis using shared framework."""
    def run_analysis():
        print("Running operator analysis and saving results...")
        results = run_all_analysis()
        return convert_results(results)
    
    return run_validation(
        validation_type="Operator Analysis",
        validation_func=run_analysis,
        parameters={"tasks": 4},
        output_filename="operator_analysis_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

