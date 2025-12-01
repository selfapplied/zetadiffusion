"""
ce1_seed.py

CE1 Seed Structure: Minimal generative units with bracket-topology.

A CE1 seed defines:
- How it behaves
- What it transforms into
- What invariants it carries
- How it composes with neighboring structures

Author: Joel
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class CE1Seed:
    """
    CE1 Seed: A minimal generative unit with bracket-topology.
    
    Attributes:
        witness: The literal bracket object (e.g., "<Q>")
        invariant: What it preserves (e.g., "t[n+1] - t[n] = 0")
        morphism: How it behaves under operators (e.g., "(Δₙ) <Q> = <Q>")
        role: Function in the system (e.g., "stabilizer/precision")
    """
    witness: str
    invariant: str
    morphism: str
    role: str
    
    def __repr__(self) -> str:
        return f"Seed<{self.witness}>"
    
    def compose(self, other: 'CE1Seed') -> 'CE1Seed':
        """
        CE1 composition: <Q> ⊕ X = X (neutral composition).
        
        Precision doesn't alter other seeds - it's a "no-effect" witness.
        """
        # <Q> composes neutrally
        if self.witness == "<Q>":
            return other
        if other.witness == "<Q>":
            return self
        
        # Default: return self (can be extended for other seeds)
        return self
    
    def is_idempotent(self) -> bool:
        """
        Check idempotence: <Q> ⊕ <Q> = <Q>.
        
        Precision stacked on precision doesn't accumulate.
        """
        return True
    
    def apply_morphism(self, morphism_name: str) -> 'CE1Seed':
        """
        Apply morphism to seed.
        
        For <Q>: (Δₙ) <Q> = <Q> (stable under index-shift).
        """
        if morphism_name == "Δₙ" and self.witness == "<Q>":
            # Index-shift morphism leaves <Q> unchanged
            return self
        
        # Default: return self unchanged
        return self

class QSeed(CE1Seed):
    """
    <Q> Seed: Precision/stability witness.
    
    Invariant: t[n+1] - t[n] = 0 (no drift)
    Morphism: (Δₙ) <Q> = <Q> (stable under index-shift)
    Role: Stabilizer/precision - marks where recursion reaches stable tempo
    """
    
    def __init__(self):
        super().__init__(
            witness="<Q>",
            invariant="t[n+1] - t[n] = 0",
            morphism="(Δₙ) <Q> = <Q>",
            role="stabilizer/precision"
        )
    
    def check_invariant(self, t_n: float, t_n_plus_1: float, tolerance: float = 1e-6) -> bool:
        """
        Check if invariant holds: t[n+1] - t[n] = 0.
        
        Args:
            t_n: Value at index n
            t_n_plus_1: Value at index n+1
            tolerance: Numerical tolerance for equality
        
        Returns:
            True if invariant holds (no drift)
        """
        drift = abs(t_n_plus_1 - t_n)
        return drift < tolerance
    
    def compute_precision(self, sequence: list[float]) -> float:
        """
        Compute precision measure: average step-size stability.
        
        Lower values = more stable = <Q> active.
        
        Args:
            sequence: Sequence of values t[0], t[1], ..., t[n]
        
        Returns:
            Average absolute drift (lower is better)
        """
        if len(sequence) < 2:
            return float('inf')
        
        drifts = [abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
        return np.mean(drifts) if drifts else float('inf')

class CE1Algebra:
    """
    CE1 Algebra: Operations on seeds.
    
    Defines composition, morphisms, and clock interactions.
    """
    
    @staticmethod
    def compose(seed1: CE1Seed, seed2: CE1Seed) -> CE1Seed:
        """
        CE1 composition: <Q> ⊕ X = X (neutral).
        
        Precision doesn't alter other seeds.
        """
        return seed1.compose(seed2)
    
    @staticmethod
    def idempotent(seed: CE1Seed) -> bool:
        """
        Check idempotence: <Q> ⊕ <Q> = <Q>.
        """
        return seed.is_idempotent()
    
    @staticmethod
    def apply_morphism(seed: CE1Seed, morphism: str) -> CE1Seed:
        """
        Apply morphism to seed.
        
        For <Q>: (Δₙ) <Q> = <Q>
        """
        return seed.apply_morphism(morphism)

class ClockInteraction:
    """
    Clock interaction with CE1 seeds.
    
    <Q> behavior by clock phase:
    - Feigenbaum clock: Suppressed (not activated)
    - Boundary clock: Weak witness (beginning to form)
    - Interior clock: Dominant (recursion reaches stable tempo)
    """
    
    # Clock boundaries (from Conjecture 9.1.3)
    FEIGENBAUM_MAX = 7.0
    BOUNDARY_END = 9.0
    MEMBRANE_END = 11.0
    
    @staticmethod
    def get_clock_phase(n: float) -> str:
        """Get clock phase for index n."""
        if n < ClockInteraction.FEIGENBAUM_MAX:
            return "feigenbaum"
        elif n < ClockInteraction.BOUNDARY_END:
            return "boundary"
        elif n < ClockInteraction.MEMBRANE_END:
            return "membrane"
        else:
            return "interior"
    
    @staticmethod
    def q_activation(n: float) -> float:
        """
        <Q> activation strength by clock phase.
        
        Returns:
            Activation strength [0, 1]:
            - Feigenbaum: 0.0 (suppressed)
            - Boundary: 0.3 (weak witness)
            - Membrane: 0.6 (forming)
            - Interior: 1.0 (dominant)
        """
        phase = ClockInteraction.get_clock_phase(n)
        
        if phase == "feigenbaum":
            return 0.0
        elif phase == "boundary":
            # Linear interpolation in boundary phase
            progress = (n - ClockInteraction.FEIGENBAUM_MAX) / (ClockInteraction.BOUNDARY_END - ClockInteraction.FEIGENBAUM_MAX)
            return 0.3 * progress
        elif phase == "membrane":
            # Linear interpolation in membrane phase
            progress = (n - ClockInteraction.BOUNDARY_END) / (ClockInteraction.MEMBRANE_END - ClockInteraction.BOUNDARY_END)
            return 0.3 + 0.3 * progress
        else:  # interior
            return 1.0
    
    @staticmethod
    def is_q_active(n: float, threshold: float = 0.5) -> bool:
        """
        Check if <Q> is active at index n.
        
        Args:
            n: Index
            threshold: Activation threshold (default 0.5)
        
        Returns:
            True if <Q> is active (activation >= threshold)
        """
        return ClockInteraction.q_activation(n) >= threshold

def create_q_seed() -> QSeed:
    """Create a <Q> seed instance."""
    return QSeed()

def check_stability(sequence: list[float], tolerance: float = 1e-6) -> dict:
    """
    Check if sequence exhibits <Q> stability (no drift).
    
    Args:
        sequence: Sequence of values
        tolerance: Numerical tolerance for drift
    
    Returns:
        Dictionary with stability metrics
    """
    q_seed = create_q_seed()
    
    if len(sequence) < 2:
        return {
            'stable': False,
            'precision': float('inf'),
            'max_drift': float('inf'),
            'q_active': False
        }
    
    # Compute precision
    precision = q_seed.compute_precision(sequence)
    
    # Check all consecutive pairs
    drifts = [abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
    max_drift = max(drifts) if drifts else float('inf')
    
    # Sequence is stable if all drifts are below tolerance
    stable = max_drift < tolerance
    
    # <Q> is active if precision is low (stable tempo)
    q_active = precision < tolerance
    
    return {
        'stable': stable,
        'precision': precision,
        'max_drift': max_drift,
        'q_active': q_active,
        'drifts': drifts
    }

