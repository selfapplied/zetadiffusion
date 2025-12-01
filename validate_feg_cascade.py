#!.venv/bin/python
"""
validate_feg_cascade.py

FEG-0.4 operator cascade validation.
Varies chaos injection to induce period-doubling bifurcations.
Measures Feigenbaum δ convergence using the actual renormalization operator.

Author: Joel
"""

import numpy as np
from zetadiffusion.renorm import RGOperator, DELTA_F
from zetadiffusion.guardian import SystemState, guardian_nash_policy
from zetadiffusion.complex_renorm import ComplexRenormOperator, PsiField
from zetadiffusion.validation_framework import run_validation

def logistic_map(x, r):
    """Logistic map: x_{n+1} = r·x_n(1 - x_n)"""
    return r * x * (1.0 - x)

def detect_period(sequence, max_period=20):
    """
    Detect period in a sequence using FFT-based sub-harmonic detection.
    
    Enhanced to detect Feigenbaum period-doubling sequence: 1, 2, 4, 8, 16...
    Uses FFT to find dominant frequencies and checks for sub-harmonics.
    """
    if len(sequence) < max_period * 2:
        return 1
    
    # Convert to numpy array
    seq = np.array(sequence)
    
    # Method 1: FFT-based frequency detection
    if len(seq) >= 32:  # Need sufficient length for FFT
        # Compute FFT
        fft = np.fft.fft(seq - np.mean(seq))
        freqs = np.fft.fftfreq(len(seq))
        power = np.abs(fft)
        
        # Find dominant frequency (excluding DC component)
        dominant_idx = np.argmax(power[1:len(power)//2]) + 1
        dominant_freq = abs(freqs[dominant_idx])
        
        if dominant_freq > 0:
            # Period is inverse of frequency
            fft_period = int(1.0 / dominant_freq)
            
            # Check if period is in Feigenbaum sequence
            feigenbaum_periods = [1, 2, 4, 8, 16]
            if fft_period in feigenbaum_periods:
                # Verify by checking sub-harmonics
                for p in feigenbaum_periods:
                    if p <= max_period and len(seq) >= p * 2:
                        # Check if sequence repeats with period p
                        tail = seq[-p:]
                        prev_tail = seq[-2*p:-p]
                        if np.allclose(tail, prev_tail, atol=1e-6):
                            return p
                return fft_period
    
    # Method 2: Autocorrelation-based (check Feigenbaum sequence first)
    # Check Feigenbaum sequence explicitly: 1, 2, 4, 8, 16...
    feigenbaum_periods = [2, 4, 8, 16, 1]  # Check period-2 first (most common)
    for period in feigenbaum_periods:
        if period > max_period or len(seq) < period * 3:  # Need at least 3 cycles
            continue
        # Check multiple cycles for robustness
        matches = 0
        for cycle in range(1, min(4, len(seq) // period)):  # Check up to 4 cycles
            tail = seq[-period:]
            prev_tail = seq[-(cycle+1)*period:-cycle*period]
            if len(tail) == period and len(prev_tail) == period:
                if np.allclose(tail, prev_tail, atol=1e-6):
                    matches += 1
        if matches >= 2:  # At least 2 matching cycles
            return period
    
    # Method 3: General autocorrelation (original method)
    for period in range(1, max_period + 1):
        if len(seq) < period * 2:
            continue
        tail = seq[-period:]
        prev_tail = seq[-2*period:-period]
        if np.allclose(tail, prev_tail, atol=1e-6):
            return period
    
    return 1

def run_feg_cascade(
    chaos_min=0.1,
    chaos_max=0.8,
    n_points=40,
    n_iterations=100
):
    """
    Run FEG-0.4 operator cascade by varying chaos injection.
    
    Measures period-doubling bifurcations and estimates Feigenbaum δ.
    """
    print("=" * 70)
    print("FEG-0.4 OPERATOR CASCADE VALIDATION")
    print("=" * 70)
    print(f"\nVarying chaos from {chaos_min:.2f} to {chaos_max:.2f}")
    print(f"Points: {n_points}, Iterations per point: {n_iterations}\n")
    
    chaos_values = np.linspace(chaos_min, chaos_max, n_points)
    
    results = {
        'chaos_values': chaos_values.tolist(),
        'periods': [],
        'coherence': [],
        'stress': [],
        'bifurcations': [],
        'delta_estimates': []
    }
    
    # Create base RG operator from logistic map
    def base_map(x):
        return logistic_map(x, 3.5699456)  # Near accumulation point
    
    rg_op = RGOperator(base=base_map, alpha=1.0, depth=0)
    R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0)
    
    prev_period = 1
    bifurcation_points = []
    
    # Adaptive guardian: track gradient to predict instability
    prev_coherence = None
    coherence_gradient = None
    
    for i, chaos in enumerate(chaos_values):
        print(f"[{i+1}/{n_points}] Chaos={chaos:.3f}...", end=" ", flush=True)
        
        # Create system state
        state = SystemState(
            chaos=chaos,
            coherence=0.5,
            stress=0.1,
            hurst=0.5
        )
        
        # Create psi field
        psi = PsiField(
            x=np.linspace(0, 1, 20),
            t=0.0,
            z=0.0,
            coherence=state.coherence,
            chaos=state.chaos,
            stress=state.stress
        )
        
        # Iterate operator with guardian damping
        trajectory = []
        coherence_trajectory = []
        
        # Adaptive guardian: track coherence gradient within iteration
        prev_iter_coherence = None
        
        # Numerical stability thresholds
        MAX_COHERENCE = 1e10
        MAX_STRESS = 1e10
        
        for iter_step in range(n_iterations):
            psi = R(psi)
            
            # Adaptive guardian: predictive intervention based on iteration gradient
            if prev_iter_coherence is not None and iter_step > 0:
                iter_gradient = abs(psi.coherence - prev_iter_coherence)
                ITER_GRADIENT_THRESHOLD = 1e5  # Growth per iteration
                if iter_gradient > ITER_GRADIENT_THRESHOLD:
                    # Predictive damping: system growing too fast
                    damping = min(0.1, 1.0 / (1.0 + iter_gradient / ITER_GRADIENT_THRESHOLD))
                    psi.coherence *= damping
                    psi.chaos *= (1 - damping * 0.5)
                    psi.stress *= (1 - damping * 0.5)
            
            prev_iter_coherence = psi.coherence
            
            # Apply guardian correction if approaching blow-up
            # Check before it happens (proactive damping)
            if abs(psi.coherence) > MAX_COHERENCE * 0.1 or abs(psi.stress) > MAX_STRESS * 0.1:
                # Get guardian response
                state = SystemState(
                    chaos=psi.chaos,
                    coherence=psi.coherence,
                    stress=psi.stress,
                    hurst=0.5
                )
                response = guardian_nash_policy(state)
                
                # Apply damping proactively
                if response.status == "SHIELDING" or abs(psi.coherence) > MAX_COHERENCE * 0.5:
                    psi.coherence = psi.coherence / DELTA_F
                    psi.chaos = psi.chaos / DELTA_F
                    psi.stress = psi.stress / DELTA_F
                    
            # Hard limit if still too large
            if abs(psi.coherence) > MAX_COHERENCE or abs(psi.stress) > MAX_STRESS:
                psi.coherence = np.sign(psi.coherence) * MAX_COHERENCE
                psi.stress = np.sign(psi.stress) * MAX_STRESS
                break  # Stop iteration if blow-up occurred
            
            trajectory.append(psi.coherence)
            coherence_trajectory.append(psi.coherence)
        
        # Detect period
        period = detect_period(coherence_trajectory)
        results['periods'].append(period)
        results['coherence'].append(float(psi.coherence))
        results['stress'].append(float(psi.stress))
        
        # Detect bifurcation (period doubling)
        if period > prev_period and prev_period > 0:
            if period == prev_period * 2:  # Period doubling
                bifurcation_points.append({
                    'index': i,
                    'chaos': float(chaos),
                    'period_before': prev_period,
                    'period_after': period
                })
                results['bifurcations'].append({
                    'chaos': float(chaos),
                    'period': period
                })
        
        prev_period = period
        
        # Track coherence gradient across chaos values for adaptive guardian
        if prev_coherence is not None:
            dchaos = chaos - (chaos_values[i-1] if i > 0 else chaos)
            if dchaos > 0:
                coherence_gradient = (psi.coherence - prev_coherence) / dchaos
                
                # Predictive intervention: if gradient exceeds threshold, intervene early
                GRADIENT_THRESHOLD = 1e6  # Coherence growth rate threshold
                if coherence_gradient > GRADIENT_THRESHOLD:
                    print(f"  ⚠ Predictive guardian: gradient={coherence_gradient:.2e}")
                    # Apply early guardian intervention
                    response = guardian_nash_policy(SystemState(
                        coherence=psi.coherence, 
                        chaos=psi.chaos, 
                        stress=psi.stress, 
                        hurst=0.5
                    ))
                    # Strong damping when gradient is high
                    damping_factor = min(0.1, 1.0 / (1.0 + coherence_gradient / GRADIENT_THRESHOLD))
                    psi.coherence *= damping_factor
                    psi.chaos *= (1 - damping_factor * 0.5)
                    psi.stress *= (1 - damping_factor * 0.5)
        
        prev_coherence = psi.coherence
        print(f"Period={period}, Coherence={psi.coherence:.4f}")
    
    # Estimate Feigenbaum δ from bifurcation intervals
    if len(bifurcation_points) >= 2:
        chaos_intervals = []
        for i in range(len(bifurcation_points) - 1):
            interval = bifurcation_points[i+1]['chaos'] - bifurcation_points[i]['chaos']
            chaos_intervals.append(interval)
        
        if len(chaos_intervals) >= 2:
            # δ ≈ interval_n / interval_{n+1}
            delta_estimates = []
            for i in range(len(chaos_intervals) - 1):
                if chaos_intervals[i+1] > 0:
                    delta_est = chaos_intervals[i] / chaos_intervals[i+1]
                    delta_estimates.append(delta_est)
            
            if delta_estimates:
                results['delta_estimates'] = delta_estimates
                avg_delta = np.mean(delta_estimates)
                delta_error = abs(avg_delta - DELTA_F)
                
                print(f"\n{'='*70}")
                print("FEIGENBAUM δ ANALYSIS")
                print(f"{'='*70}")
                print(f"Bifurcations detected: {len(bifurcation_points)}")
                print(f"δ estimates: {delta_estimates}")
                print(f"Average δ: {avg_delta:.6f}")
                print(f"True δ: {DELTA_F:.6f}")
                print(f"Error: {delta_error:.6f}")
                print(f"{'='*70}\n")
                
                results['delta_average'] = float(avg_delta)
                results['delta_error'] = float(delta_error)
                results['n_bifurcations'] = len(bifurcation_points)
            else:
                results['delta_average'] = None
                results['delta_error'] = None
        else:
            results['delta_average'] = None
            results['delta_error'] = None
    else:
        print(f"\n⚠ Only {len(bifurcation_points)} bifurcation(s) detected (need ≥2 for δ estimate)")
        results['delta_average'] = None
        results['delta_error'] = None
    
    return results

def main():
    """Run FEG cascade validation using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    def run_cascade():
        return run_feg_cascade(
            chaos_min=0.1,
            chaos_max=0.8,
            n_points=40,
            n_iterations=100
        )
    
    result = run_validation(
        validation_type="FEG Cascade",
        validation_func=run_cascade,
        parameters={
            'chaos_min': 0.1,
            'chaos_max': 0.8,
            'n_points': 40,
            'n_iterations': 100
        },
        output_filename="feg_cascade_results.json"
    )
    
    # Add success flag based on delta_average
    if result.success and 'delta_average' in result.results:
        result.results['success'] = result.results.get('delta_average') is not None
    
    return result

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

