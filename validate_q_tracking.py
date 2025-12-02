#!.venv/bin/python
"""
validate_q_tracking.py

Tracks <Q> activation across all validation runs.
Shows how <Q> evolves over time and across different validations.

Author: Joel
"""

import json
from pathlib import Path
from zetadiffusion.q_integration import track_q_over_time, format_q_summary
from zetadiffusion.validation_framework import run_validation

def load_all_validation_runs() -> list:
    """Load all validation result files."""
    runs = []
    out_dir = Path(".out")
    
    result_files = list(out_dir.glob("*_results.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                runs.append(data)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    return runs

def analyze_q_tracking() -> dict:
    """
    Analyze <Q> activation across all validation runs.
    """
    print("=" * 70)
    print("Q ACTIVATION TRACKING")
    print("=" * 70)
    print()
    
    # Load all runs
    runs = load_all_validation_runs()
    print(f"Loaded {len(runs)} validation runs")
    print()
    
    # Track <Q> over time
    tracking = track_q_over_time(runs)
    
    if not tracking.get('q_available', False):
        print("Q metrics not available in any runs")
        return tracking
    
    # Display tracking
    print("Q Activation by Run:")
    print("-" * 70)
    print(f"{'Run':<30} | {'Activation':<12} | {'Timestamp'}")
    print("-" * 70)
    
    for i, (val_type, activation, timestamp) in enumerate(zip(
        tracking['validation_types'],
        tracking['activations'],
        tracking['timestamps']
    )):
        timestamp_short = timestamp[:10] if timestamp else "N/A"
        print(f"{val_type:<30} | {activation:>11.3f} | {timestamp_short}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average Q activation: {tracking['avg_activation']:.3f}")
    print(f"Trend: {tracking['trend']}")
    print()
    
    # Show runs with Q metrics
    print("Runs with Q Metrics:")
    print("-" * 70)
    for run in runs:
        q_metrics = run.get('q_metrics', {})
        if q_metrics.get('q_available', False):
            val_type = run.get('validation_type', 'Unknown')
            summary = format_q_summary(q_metrics)
            print(f"{val_type}:")
            print(f"  {summary}")
            print()
    
    return tracking

def main():
    """Run Q tracking analysis using shared framework."""
    def run_q_tracking():
        return analyze_q_tracking()
    
    return run_validation(
        validation_type="Q Activation Tracking",
        validation_func=run_q_tracking,
        parameters={},
        output_filename="q_tracking_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)




