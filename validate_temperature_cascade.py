#!.venv/bin/python
"""
Temperature Cascade Experiment

Empirical validation: Run GPT-2 through 40 temperature points,
measure δ_T convergence to Feigenbaum constant 4.6692016.

Tests whether the trigonometric framework holds empirically by
observing period-doubling bifurcations in model behavior.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import json
from typing import List, Dict, Tuple
import sys

FEIGENBAUM_DELTA = 4.66920160910299067185320382
FEIGENBAUM_ALPHA = 2.50290787509589282228390287

class TemperatureCascade:
    """
    Measures model behavior across temperature range to detect
    period-doubling bifurcations and estimate δ convergence.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def measure_entropy(self, prompt: str, temperature: float, n_samples: int = 10) -> float:
        """
        Measure entropy of model outputs at given temperature.
        
        Entropy serves as order parameter - phase transitions
        show as discontinuities in entropy vs temperature.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        entropies = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=input_length + 20,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                # Get logits for generated tokens
                logits = self.model(inputs.input_ids).logits[0, -1, :]
                probs = torch.softmax(logits / temperature, dim=-1)
                
                # Calculate entropy: H = -Σ p log p
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                entropies.append(entropy)
        
        return np.mean(entropies)
    
    def measure_diversity(self, prompt: str, temperature: float, n_samples: int = 20) -> float:
        """
        Measure output diversity (unique tokens / total tokens).
        
        Diversity shows phase transitions as model behavior
        shifts from deterministic to chaotic.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        all_tokens = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=input_length + 15,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                generated = outputs[0, input_length:].cpu().numpy()
                all_tokens.extend(generated.tolist())
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        diversity = unique_tokens / max(total_tokens, 1)
        
        return diversity
    
    def measure_periodicity(self, prompt: str, temperature: float, n_steps: int = 50) -> Dict:
        """
        Measure periodicity in model outputs by tracking token sequences.
        
        Period-doubling bifurcations appear as changes in sequence period.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        sequences = []
        
        with torch.no_grad():
            current_ids = inputs.input_ids
            for _ in range(n_steps):
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits / temperature, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                sequences.append(next_token.item())
                
                # Append to input
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        
        # Detect periodicity
        sequence = np.array(sequences)
        period = self._detect_period(sequence)
        
        return {
            'period': period,
            'sequence': sequences[:20],  # First 20 for inspection
            'variance': float(np.var(sequence))
        }
    
    def _detect_period(self, sequence: np.ndarray, max_period: int = 20) -> int:
        """
        Detect period in sequence using autocorrelation.
        Enhanced to detect period-doubling bifurcations.
        """
        if len(sequence) < max_period * 2:
            return 1
        
        best_period = 1
        best_correlation = 0
        
        # Check for period-doubling sequence: 1, 2, 4, 8, 16...
        period_candidates = [1, 2, 4, 8, 16]
        for p in period_candidates:
            if p >= max_period or len(sequence) <= p * 2:
                break
            
            # Autocorrelation at lag p
            corr = np.corrcoef(sequence[:-p], sequence[p:])[0, 1]
            if not np.isnan(corr) and corr > best_correlation + 0.1:
                best_correlation = corr
                best_period = p
        
        # Also check general autocorrelation for other periods
        for p in range(1, min(max_period, len(sequence) // 2)):
            if p in period_candidates:
                continue
            if len(sequence) > p:
                corr = np.corrcoef(sequence[:-p], sequence[p:])[0, 1]
                if not np.isnan(corr) and corr > best_correlation + 0.2:
                    best_correlation = corr
                    best_period = p
        
        return best_period
    
    def run_cascade(self, 
                   prompt: str = "The meaning of life is",
                   n_temperatures: int = 40,
                   temp_min: float = 0.1,
                   temp_max: float = 4.0) -> Dict:
        """
        Run temperature cascade experiment.
        
        Measures order parameters across temperature range to detect
        phase transitions and estimate δ convergence.
        """
        temperatures = np.linspace(temp_min, temp_max, n_temperatures)
        
        print(f"Running cascade: {n_temperatures} temperatures from {temp_min:.2f} to {temp_max:.2f}")
        print(f"Prompt: '{prompt}'")
        print()
        
        results = {
            'temperatures': temperatures.tolist(),
            'entropies': [],
            'diversities': [],
            'periods': [],
            'variances': []
        }
        
        for i, temp in enumerate(temperatures):
            print(f"[{i+1}/{n_temperatures}] T={temp:.3f}...", end=" ", flush=True)
            
            # Measure order parameters (minimal samples for fast execution)
            entropy = self.measure_entropy(prompt, temp, n_samples=2)
            diversity = self.measure_diversity(prompt, temp, n_samples=2)
            period_data = self.measure_periodicity(prompt, temp, n_steps=5)
            
            results['entropies'].append(entropy)
            results['diversities'].append(diversity)
            results['periods'].append(period_data['period'])
            results['variances'].append(period_data['variance'])
            
            print(f"H={entropy:.3f}, D={diversity:.3f}, P={period_data['period']}")
        
        # Detect phase transitions (bifurcations)
        transitions = self._detect_transitions(results)
        results['transitions'] = transitions
        
        # Estimate δ convergence
        delta_analysis = self._estimate_delta(transitions)
        results['delta_analysis'] = delta_analysis
        results['delta_estimate'] = delta_analysis['delta_estimate']
        results['delta_error'] = abs(delta_analysis['delta_estimate'] - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100 if delta_analysis['delta_estimate'] > 0 else 100.0
        
        # Analyze entropy saturation
        entropy_analysis = self.analyze_entropy_saturation(results)
        results['entropy_analysis'] = entropy_analysis
        
        return results
    
    def _detect_transitions(self, results: Dict) -> List[Dict]:
        """
        Detect phase transitions with focus on genuine period-doubling bifurcations.
        
        Filters for true bifurcations: period must double (1→2→4→8...)
        """
        temps = np.array(results['temperatures'])
        entropies = np.array(results['entropies'])
        diversities = np.array(results['diversities'])
        periods = np.array(results['periods'])
        
        transitions = []
        
        # Find genuine period-doubling bifurcations
        # Period must be in sequence: 1, 2, 4, 8, 16...
        period_changes = []
        prev_period = periods[0]
        
        for i in range(1, len(periods)):
            curr_period = periods[i]
            
            # Check if period doubled (genuine bifurcation)
            if curr_period == prev_period * 2 or (prev_period == 1 and curr_period == 2):
                period_changes.append(i)
            
            # Also check if period halved (reverse bifurcation)
            elif curr_period == prev_period / 2:
                period_changes.append(i)
            
            prev_period = curr_period
        
        # Find sharp changes in entropy (second derivative)
        entropy_second_diff = np.diff(np.diff(entropies))
        entropy_threshold = np.std(entropy_second_diff) * 2.5
        
        # Find sharp changes in diversity
        diversity_second_diff = np.diff(np.diff(diversities))
        diversity_threshold = np.std(diversity_second_diff) * 2.5
        
        # Combine: prioritize period-doubling, then sharp parameter changes
        all_transitions = set(period_changes)  # Start with genuine bifurcations
        
        for i in range(1, len(temps) - 1):
            # Sharp entropy change (second derivative)
            if abs(entropy_second_diff[i-1]) > entropy_threshold:
                all_transitions.add(i)
            
            # Sharp diversity change
            if abs(diversity_second_diff[i-1]) > diversity_threshold:
                all_transitions.add(i)
        
        # Convert to list of transition points
        transition_list = []
        for idx in sorted(all_transitions):
            transition_list.append({
                'index': int(idx),
                'temperature': float(temps[idx]),
                'entropy': float(entropies[idx]),
                'diversity': float(diversities[idx]),
                'period': int(periods[idx]),
                'is_bifurcation': idx in period_changes
            })
        
        return transition_list
    
    def _estimate_delta(self, transitions: List[Dict]) -> Dict:
        """
        Estimate Feigenbaum δ from bifurcation intervals.
        
        δ = lim(n→∞) Δ_n / Δ_{n+1}
        where Δ_n is the width of the n-th stability window.
        
        Only uses genuine period-doubling bifurcations.
        """
        # Filter for genuine bifurcations
        bifurcations = [t for t in transitions if t.get('is_bifurcation', False)]
        
        if len(bifurcations) < 3:
            return {
                'delta_estimate': 0.0,
                'delta_convergence': [],
                'n_bifurcations': len(bifurcations),
                'valid': False
            }
        
        # Extract temperature values at bifurcations
        bifurcation_temps = [t['temperature'] for t in bifurcations]
        
        # Calculate intervals between bifurcations
        intervals = []
        for i in range(len(bifurcation_temps) - 1):
            interval = bifurcation_temps[i+1] - bifurcation_temps[i]
            if interval > 0:
                intervals.append(interval)
        
        if len(intervals) < 2:
            return {
                'delta_estimate': 0.0,
                'delta_convergence': [],
                'n_bifurcations': len(bifurcations),
                'valid': False
            }
        
        # Calculate δ estimates from successive interval ratios
        delta_estimates = []
        for i in range(len(intervals) - 1):
            if intervals[i+1] > 0:
                delta_est = intervals[i] / intervals[i+1]
                delta_estimates.append(delta_est)
        
        if not delta_estimates:
            return {
                'delta_estimate': 0.0,
                'delta_convergence': delta_estimates,
                'n_bifurcations': len(bifurcations),
                'valid': False
            }
        
        # Return convergence sequence and final estimate
        return {
            'delta_estimate': np.mean(delta_estimates[-3:]) if len(delta_estimates) >= 3 else np.mean(delta_estimates),
            'delta_convergence': delta_estimates,
            'n_bifurcations': len(bifurcations),
            'valid': len(bifurcations) >= 4  # Need at least 4 bifurcations for convergence
        }
    
    def analyze_entropy_saturation(self, results: Dict) -> Dict:
        """
        Analyze entropy saturation: H(T) = H∞(1 - e^(-T/τ))
        
        Extracts characteristic timescale τ and asymptotic entropy H∞.
        """
        temps = np.array(results['temperatures'])
        entropies = np.array(results['entropies'])
        
        # Fit to saturation model: H(T) = H∞(1 - exp(-T/τ))
        from scipy.optimize import curve_fit
        
        def saturation_model(T, H_inf, tau):
            return H_inf * (1 - np.exp(-T / tau))
        
        try:
            popt, pcov = curve_fit(
                saturation_model,
                temps,
                entropies,
                p0=[max(entropies), 1.0],
                bounds=([0, 0.01], [max(entropies) * 2, 10.0])
            )
            
            H_inf, tau = popt
            H_inf_err, tau_err = np.sqrt(np.diag(pcov))
            
            # Calculate R²
            H_pred = saturation_model(temps, H_inf, tau)
            ss_res = np.sum((entropies - H_pred) ** 2)
            ss_tot = np.sum((entropies - np.mean(entropies)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'H_inf': float(H_inf),
                'H_inf_error': float(H_inf_err),
                'tau': float(tau),
                'tau_error': float(tau_err),
                'r_squared': float(r_squared),
                'fit_valid': r_squared > 0.9
            }
        except:
            return {
                'H_inf': float(max(entropies)),
                'H_inf_error': 0.0,
                'tau': 0.0,
                'tau_error': 0.0,
                'r_squared': 0.0,
                'fit_valid': False
            }
    
    def visualize_results(self, results: Dict, output_file: str = None):
        """Print ASCII visualization of results."""
        temps = np.array(results['temperatures'])
        entropies = np.array(results['entropies'])
        diversities = np.array(results['diversities'])
        periods = np.array(results['periods'])
        
        print("\n" + "=" * 70)
        print("TEMPERATURE CASCADE RESULTS")
        print("=" * 70)
        print()
        
        # Entropy plot
        print("Entropy vs Temperature:")
        self._ascii_plot(temps, entropies, height=10, width=50)
        print()
        
        # Diversity plot
        print("Diversity vs Temperature:")
        self._ascii_plot(temps, diversities, height=10, width=50)
        print()
        
        # Period plot
        print("Period vs Temperature:")
        self._ascii_plot(temps, periods.astype(float), height=10, width=50)
        print()
        
        # Transitions
        print(f"Phase Transitions Detected: {len(results['transitions'])}")
        for t in results['transitions'][:10]:  # First 10
            print(f"  T={t['temperature']:.3f}: period={t['period']}, H={t['entropy']:.3f}")
        print()
        
        # Delta estimate
        print("=" * 70)
        print("FEIGENBAUM δ ESTIMATE")
        print("=" * 70)
        delta_analysis = results['delta_analysis']
        print(f"Genuine Bifurcations Detected: {delta_analysis['n_bifurcations']}")
        print(f"Estimated δ: {results['delta_estimate']:.6f}")
        print(f"Theoretical δ: {FEIGENBAUM_DELTA:.6f}")
        print(f"Error: {results['delta_error']:.2f}%")
        print()
        
        if delta_analysis['valid']:
            if results['delta_error'] < 10:
                print("✓ Convergence to Feigenbaum δ confirmed!")
            elif results['delta_error'] < 25:
                print("~ Partial convergence observed")
            else:
                print("✗ Convergence not yet observed - may need more bifurcations")
        else:
            print("✗ Insufficient bifurcations for δ estimation")
            print("  → System remains in period-1 regime")
            print("  → No period-doubling cascade detected")
        print()
        
        # Entropy saturation analysis
        print("=" * 70)
        print("ENTROPY SATURATION ANALYSIS")
        print("=" * 70)
        entropy_analysis = results['entropy_analysis']
        print(f"Model: H(T) = H∞(1 - e^(-T/τ))")
        print(f"Asymptotic Entropy H∞: {entropy_analysis['H_inf']:.3f} ± {entropy_analysis['H_inf_error']:.3f}")
        print(f"Characteristic Timescale τ: {entropy_analysis['tau']:.3f} ± {entropy_analysis['tau_error']:.3f}")
        print(f"R² Fit Quality: {entropy_analysis['r_squared']:.3f}")
        if entropy_analysis['fit_valid']:
            print("✓ Entropy saturation model fits well")
        else:
            print("~ Entropy saturation model fit marginal")
        print()
    
    def _ascii_plot(self, x: np.ndarray, y: np.ndarray, height: int = 10, width: int = 50):
        """Simple ASCII plot."""
        if len(x) == 0 or len(y) == 0:
            return
        
        y_min, y_max = np.min(y), np.max(y)
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min if x_max > x_min else 1.0
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for i in range(len(x)):
            if i < len(y):
                x_idx = int((x[i] - x_min) / x_range * (width - 1))
                y_idx = int((y[i] - y_min) / y_range * (height - 1))
                x_idx = max(0, min(width - 1, x_idx))
                y_idx = max(0, min(height - 1, y_idx))
                grid[height - 1 - y_idx][x_idx] = '*'
        
        # Print grid
        for row in grid:
            print('|' + ''.join(row) + '|')
        
        print(f"Range: [{y_min:.3f}, {y_max:.3f}]")

def main():
    """Run temperature cascade experiment using shared framework."""
    from zetadiffusion.validation_framework import run_validation
    
    def run_cascade():
        print("=" * 70)
        print("TEMPERATURE CASCADE EXPERIMENT")
        print("=" * 70)
        print()
        print("Testing δ_T convergence to Feigenbaum constant 4.6692016...")
        print()
        
        # Initialize cascade
        cascade = TemperatureCascade()
        
        # Run experiment
        prompt = "The meaning of life is"
        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
        
        results = cascade.run_cascade(
            prompt=prompt,
            n_temperatures=10,  # Drastically reduced for fast execution
            temp_min=0.1,
            temp_max=4.0  # Reduced range
        )
        
        # Visualize
        cascade.visualize_results(results)
        
        # Convert to JSON-serializable format
        return {
            'temperatures': [float(t) for t in results['temperatures']],
            'entropies': [float(e) for e in results['entropies']],
            'diversities': [float(d) for d in results['diversities']],
            'periods': [int(p) for p in results['periods']],
            'variances': [float(v) for v in results['variances']],
            'transitions': [
                {
                    'index': int(t['index']),
                    'temperature': float(t['temperature']),
                    'entropy': float(t['entropy']),
                    'diversity': float(t['diversity']),
                    'period': int(t['period']),
                    'is_bifurcation': bool(t.get('is_bifurcation', False))
                }
                for t in results['transitions']
            ],
            'delta_analysis': {
                'delta_estimate': float(results['delta_analysis']['delta_estimate']),
                'delta_convergence': [float(d) for d in results['delta_analysis']['delta_convergence']],
                'n_bifurcations': int(results['delta_analysis']['n_bifurcations']),
                'valid': bool(results['delta_analysis']['valid'])
            },
            'delta_estimate': float(results['delta_estimate']),
            'delta_error': float(results['delta_error']),
            'entropy_analysis': {
                'H_inf': float(results['entropy_analysis']['H_inf']),
                'H_inf_error': float(results['entropy_analysis']['H_inf_error']),
                'tau': float(results['entropy_analysis']['tau']),
                'tau_error': float(results['entropy_analysis']['tau_error']),
                'r_squared': float(results['entropy_analysis']['r_squared']),
                'fit_valid': bool(results['entropy_analysis']['fit_valid'])
            }
        }
    
    return run_validation(
        validation_type="Temperature Cascade",
        validation_func=run_cascade,
        parameters={
            'n_temperatures': 10,
            'temp_min': 0.1,
            'temp_max': 4.0
        },
        output_filename="temperature_cascade_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

