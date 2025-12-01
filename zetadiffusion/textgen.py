"""
textgen.py

Fractal Text Generation using Guardian Nash Policy.
Text as signal: Guardian modulates coherence (structure) vs chaos (exploration).
"""

import numpy as np
from typing import List, Tuple, Dict
from zetadiffusion.guardian import (
    SystemState, guardian_nash_policy, calculate_fixed_points,
    FEIGENBAUM_DELTA
)

class TextSignal:
    """Text treated as a signal - characters as values in a dynamical system."""
    
    def __init__(self, text: str):
        self.text = text
        self.chars = list(text)
        self.signal = np.array([ord(c) for c in text], dtype=float)
        self.normalized = (self.signal - np.min(self.signal)) / (np.max(self.signal) - np.min(self.signal) + 1e-10)
    
    def coherence(self) -> float:
        """Measure structural coherence: repetition, patterns, grammar-like structure."""
        if len(self.chars) < 2:
            return 0.0
        
        # Measure pattern repetition (higher = more coherent)
        patterns = {}
        for i in range(len(self.chars) - 1):
            pair = (self.chars[i], self.chars[i+1])
            patterns[pair] = patterns.get(pair, 0) + 1
        
        # Coherence = normalized pattern diversity (lower diversity = higher coherence)
        max_patterns = len(self.chars) - 1
        pattern_diversity = len(patterns) / max_patterns if max_patterns > 0 else 0
        coherence = 1.0 - pattern_diversity  # Invert: less diversity = more coherence
        
        return coherence
    
    def chaos(self) -> float:
        """Measure exploration/chaos: character diversity, novelty."""
        if len(self.chars) == 0:
            return 0.0
        
        unique_chars = len(set(self.chars))
        total_chars = len(self.chars)
        diversity = unique_chars / total_chars
        
        # Chaos = high diversity (exploration)
        return diversity
    
    def stress(self) -> float:
        """Measure stress: deviation from expected patterns."""
        if len(self.normalized) < 2:
            return 0.0
        
        # Stress = variance in normalized signal
        variance = np.var(self.normalized)
        return min(1.0, variance * 2.0)  # Scale to [0, 1]
    
    def hurst_estimate(self) -> float:
        """Estimate Hurst exponent from text signal (memory/persistence)."""
        if len(self.normalized) < 10:
            return 0.5  # Default: random walk
        
        # Simplified Hurst estimation: autocorrelation at lag 1
        signal = self.normalized
        if len(signal) < 2:
            return 0.5
        
        # Autocorrelation
        mean_signal = np.mean(signal)
        centered = signal - mean_signal
        
        if len(centered) < 2:
            return 0.5
        
        autocorr = np.correlate(centered[:-1], centered[1:])[0] / (len(centered) - 1)
        variance = np.var(centered)
        
        if variance < 1e-10:
            return 0.5
        
        # Map autocorrelation to Hurst [0, 1]
        # Positive autocorr → H > 0.5 (persistent)
        # Negative autocorr → H < 0.5 (anti-persistent)
        hurst = 0.5 + (autocorr / variance) * 0.3
        hurst = max(0.0, min(1.0, hurst))
        
        return hurst

class FractalTextGenerator:
    """Generates text fractally using Guardian Nash Policy."""
    
    def __init__(self, seed_text: str = "The"):
        self.seed_text = seed_text
        self.current_text = TextSignal(seed_text)
        self.history: List[TextSignal] = [self.current_text]
        self.guardian_history: List[Dict] = []
    
    def generate_step(self, char_pool: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?") -> str:
        """
        Generate one character using Guardian Nash Policy.
        
        The Guardian controls the balance between:
        - Coherence (following patterns, structure)
        - Chaos (exploration, novelty)
        """
        # Get current system state
        coherence = self.current_text.coherence()
        chaos = self.current_text.chaos()
        stress = self.current_text.stress()
        hurst = self.current_text.hurst_estimate()
        
        # Create Guardian state
        guardian_state = SystemState(
            coherence=coherence,
            chaos=chaos,
            stress=stress,
            hurst=hurst,
            gamma=1.0,
            delta=0.5
        )
        
        # Get Guardian response
        response = guardian_nash_policy(guardian_state)
        
        # Store Guardian decision
        self.guardian_history.append({
            'coherence': coherence,
            'chaos': chaos,
            'stress': stress,
            'hurst': hurst,
            'coupling': response.coupling,
            'status': response.status
        })
        
        # Generate next character based on Guardian coupling strength
        # High coupling β: Follow patterns (coherence)
        # Low coupling β: Explore (chaos)
        # Use coupling value to blend between modes
        
        # Normalize coupling to [0, 1] for blending
        # Typical β_res is ~2.7, so normalize by that
        coupling_norm = min(1.0, response.coupling / 3.0)
        
        # Blend: coupling_norm = 1.0 → fully coherent, 0.0 → fully chaotic
        if np.random.random() < coupling_norm:
            # Follow patterns (weighted by coupling, memory controlled by Hurst)
            next_char = self._coherent_char(char_pool, hurst=hurst)
        else:
            # Explore (weighted by 1 - coupling)
            next_char = self._chaotic_char(char_pool)
        
        # Add character
        new_text = self.current_text.text + next_char
        self.current_text = TextSignal(new_text)
        self.history.append(self.current_text)
        
        return next_char
    
    def _coherent_char(self, char_pool: str, hurst: float = 0.5) -> str:
        """Generate character following patterns (coherence mode).
        
        Hurst controls memory: H > 0.5 = remember longer, H < 0.5 = forget quickly.
        """
        if len(self.current_text.chars) < 2:
            return np.random.choice(list(char_pool))
        
        # Hurst controls lookback window: high H = longer memory
        # Map H [0, 1] to lookback [1, min(20, text_length)]
        max_lookback = min(20, len(self.current_text.chars))
        lookback = max(1, int(hurst * max_lookback))
        
        # Find most common transitions (weighted by recency for low H)
        transitions = {}
        weights = {}
        
        start_idx = max(0, len(self.current_text.chars) - lookback)
        for i in range(start_idx, len(self.current_text.chars) - 1):
            from_char = self.current_text.chars[i]
            to_char = self.current_text.chars[i+1]
            
            # Weight by recency (more recent = higher weight for low H)
            recency_weight = (i - start_idx + 1) / lookback
            if hurst < 0.5:
                # Anti-persistent: weight recent more
                weight = recency_weight
            else:
                # Persistent: weight all equally
                weight = 1.0
            
            if from_char not in transitions:
                transitions[from_char] = {}
                weights[from_char] = {}
            
            transitions[from_char][to_char] = transitions[from_char].get(to_char, 0) + weight
            weights[from_char][to_char] = weights[from_char].get(to_char, 0) + weight
        
        # Use last character(s) to predict next
        last_char = self.current_text.chars[-1]
        if last_char in transitions:
            # Most common transition from last_char
            next_options = transitions[last_char]
            if next_options:
                # Weight by frequency, but add some randomness for variety
                chars = list(next_options.keys())
                weights = list(next_options.values())
                # Normalize weights
                total = sum(weights)
                if total > 0:
                    probs = [w / total for w in weights]
                    # Bias toward most common but allow some exploration
                    next_char = np.random.choice(chars, p=probs)
                    return next_char
        
        # Fallback: prefer common characters (spaces, vowels)
        common_chars = " eaoiutnshrdlcmfwypgvbkjqxz"
        common_in_pool = [c for c in common_chars if c in char_pool]
        if common_in_pool:
            return np.random.choice(common_in_pool)
        
        return np.random.choice(list(char_pool))
    
    def _chaotic_char(self, char_pool: str) -> str:
        """Generate character randomly (chaos/exploration mode)."""
        return np.random.choice(list(char_pool))
    
    def generate(self, n_chars: int = 100) -> str:
        """Generate n characters using fractal Guardian-controlled process."""
        char_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?"
        
        for _ in range(n_chars):
            self.generate_step(char_pool)
        
        return self.current_text.text
    
    def get_statistics(self) -> Dict:
        """Get generation statistics."""
        if not self.guardian_history:
            return {}
        
        couplings = [h['coupling'] for h in self.guardian_history]
        statuses = [h['status'] for h in self.guardian_history]
        
        resonance_count = statuses.count("RESONANCE: Equilibrium Tracking")
        shielding_count = statuses.count("SHIELDING: Persistence Risk Detected")
        
        return {
            'total_chars': len(self.current_text.text),
            'final_coherence': self.current_text.coherence(),
            'final_chaos': self.current_text.chaos(),
            'final_stress': self.current_text.stress(),
            'final_hurst': self.current_text.hurst_estimate(),
            'resonance_steps': resonance_count,
            'shielding_steps': shielding_count,
            'mean_coupling': np.mean(couplings) if couplings else 0.0
        }

