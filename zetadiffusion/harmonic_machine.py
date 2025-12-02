"""
harmonic_machine.py

Harmonic Machine Architecture with embedded CE1 Stability Module.

Three layers:
1. Resonant Bracket Grammar - CE1 bracket roles and flows
2. Morphism Generators - Operators (composition, shift, reflection, blending)
3. Harmonic Dynamics - Iterative/recursive update loop

Fourth submodule (embedded):
4. CE1 Stability Module - Precision gate, rule activator, alignment operator

Author: Joel
"""

from typing import Callable, Optional, Dict, Any
import numpy as np
from zetadiffusion.ce1_seed import QSeed, create_q_seed, check_stability, ClockInteraction

# ============================================================================
# Layer 1: Resonant Bracket Grammar
# ============================================================================

class ResonantBracket:
    """
    Resonant Bracket Grammar: CE1 brackets define roles and flows.
    
    <Q> is a universal stabilizer - harmonic-neutral, like a tonic.
    """
    
    @staticmethod
    def is_stabilizer(bracket: str) -> bool:
        """Check if bracket is a stabilizer (e.g., <Q>)."""
        return bracket == "<Q>"
    
    @staticmethod
    def compose_brackets(bracket1: str, bracket2: str) -> str:
        """
        Compose brackets: <Q> ⊕ X = X (neutral composition).
        
        <Q> is harmonic-neutral - doesn't disturb other brackets.
        """
        if bracket1 == "<Q>" or bracket2 == "<Q>":
            # <Q> composes neutrally
            return bracket1 if bracket2 == "<Q>" else bracket2
        # Default: return first (can be extended)
        return bracket1

# ============================================================================
# Layer 2: Morphism Generators
# ============================================================================

class MorphismGenerator:
    """
    Morphism Generators: Operators like composition, shift, reflection, blending.
    
    <Q> acts as a modifier that "locks" a morphism to stable step-size.
    """
    
    @staticmethod
    def q_lock(morphism: Callable[[float], float], align_op: Callable[[float], float]) -> Callable[[float], float]:
        """
        Q-lock: (Q-lock)(f) := Op_align ∘ (f)
        
        Apply morphism normally, then enforce t[n+1] = t[n].
        This is a safety rail for morphisms.
        """
        def locked_morphism(t_n: float) -> float:
            # Apply morphism
            t_next = morphism(t_n)
            # Then enforce alignment (stability)
            return align_op(t_next)
        return locked_morphism
    
    @staticmethod
    def apply_with_stability(morphism: Callable[[float], float], 
                            t_n: float,
                            q_active: bool,
                            align_op: Callable[[float], float]) -> float:
        """
        Apply morphism with optional Q-lock.
        
        If Q is active, lock the morphism to stable step-size.
        """
        if q_active:
            locked = MorphismGenerator.q_lock(morphism, align_op)
            return locked(t_n)
        else:
            return morphism(t_n)

# ============================================================================
# Layer 3: Harmonic Dynamics
# ============================================================================

class HarmonicDynamics:
    """
    Harmonic Dynamics: The iterative/recursive update loop.
    
    Main dynamic: t[n+1] = R[t[n]]
    
    With Stability Module embedded:
    - Before n=11 → machine runs normally
    - Approaching convergence → <Q> begins to form
    - <Q> active → alignment operator keeps recursion steady
    """
    
    def __init__(self, 
                 recursion_op: Callable[[float], float],
                 stability_module: 'CE1StabilityModule'):
        """
        Initialize Harmonic Dynamics with recursion operator and stability module.
        
        Args:
            recursion_op: The main recursion operator R[t[n]]
            stability_module: The embedded CE1 Stability Module
        """
        self.recursion_op = recursion_op
        self.stability_module = stability_module
    
    def step(self, t_n: float, n: int) -> float:
        """
        Single step of harmonic dynamics with embedded stability.
        
        Logic:
        if activate(n):     // n ≥ 11
            if <Q> active:     // t[n+1] = t[n]
                t[n+1] = align(t[n])    // enforce stability
            else:
                t[n+1] = gate(t[n])     // allow convergence
        else:
            t[n+1] = R(t[n])              // normal recursion
        """
        # Check if stability module should activate
        if self.stability_module.should_activate(n):
            # Check if Q is active (stable recursion)
            q_active = self.stability_module.is_q_active(t_n)
            
            if q_active:
                # <Q> active → alignment operator keeps recursion steady
                return self.stability_module.align(t_n)
            else:
                # Allow convergence through precision gate
                return self.stability_module.gate(t_n)
        else:
            # Normal recursion before n=11
            return self.recursion_op(t_n)
    
    def evolve(self, t_0: float, n_steps: int) -> list[float]:
        """
        Evolve harmonic dynamics over n_steps.
        
        Returns sequence of states: [t[0], t[1], ..., t[n_steps]]
        """
        sequence = [t_0]
        t_current = t_0
        
        for n in range(1, n_steps + 1):
            t_next = self.step(t_current, n)
            sequence.append(t_next)
            t_current = t_next
        
        return sequence

# ============================================================================
# Layer 4: CE1 Stability Module (Embedded)
# ============================================================================

class CE1StabilityModule:
    """
    CE1 Stability Module: Precision gate + rule activator + alignment operator.
    
    Module_stability := {
        seed: <Q>,
        gate: Gate_precision,
        activate: Rule_activate_precision,
        align: Op_align
    }
    
    This is the machine's "metronome" - not a governor, just steady tempo.
    """
    
    def __init__(self, 
                 activation_threshold: int = 11,
                 drift_tolerance: float = 1.0):
        """
        Initialize CE1 Stability Module.
        
        Args:
            activation_threshold: n ≥ this value activates stability (default: 11)
            drift_tolerance: Tolerance for drift detection (default: 1.0)
        """
        self.activation_threshold = activation_threshold
        self.drift_tolerance = drift_tolerance
        self.q_seed = create_q_seed()
        self.sequence_history: list[float] = []
    
    def should_activate(self, n: int) -> bool:
        """
        Rule_activate_precision: Activate stability module at n ≥ threshold.
        
        Before n=11 → machine runs normally.
        At n=11 → stability module activates.
        """
        return n >= self.activation_threshold
    
    def gate(self, t_n: float) -> float:
        """
        Gate_precision: Drift-stopping gate.
        
        Allows convergence but prevents drift explosion.
        This is the "allow convergence" path when Q is not yet active.
        """
        # Simple gate: clamp to prevent explosion
        # Can be extended with more sophisticated damping
        if abs(t_n) > 1e6:
            # Prevent explosion
            return np.sign(t_n) * 1e6
        return t_n
    
    def align(self, t_n: float) -> float:
        """
        Op_align: Alignment operator that enforces stable tempo.
        
        Enforces t[n+1] = t[n] when <Q> is active.
        This is the "enforce stability" path.
        """
        # If we have history, maintain stable step-size
        if len(self.sequence_history) >= 2:
            # Use last step-size as reference
            last_step = self.sequence_history[-1] - self.sequence_history[-2]
            return t_n + last_step  # Maintain step-size
        else:
            # No history yet, just return current value
            return t_n
    
    def is_q_active(self, t_n: float) -> bool:
        """
        Check if <Q> is active (stable recursion).
        
        <Q> is active if:
        - We have enough history (≥ 2 points)
        - Drift is bounded (stable tempo)
        """
        # Add current value to history
        self.sequence_history.append(t_n)
        
        # Need at least 2 points to check stability
        if len(self.sequence_history) < 2:
            return False
        
        # Check stability using Q seed
        stability = check_stability(self.sequence_history, tolerance=self.drift_tolerance)
        return stability['q_active']
    
    def reset_history(self):
        """Reset sequence history (useful for new runs)."""
        self.sequence_history = []

# ============================================================================
# Harmonic Machine (Complete System)
# ============================================================================

class HarmonicMachine:
    """
    Harmonic Machine: Complete system with embedded CE1 Stability Module.
    
    Architecture:
    1. Resonant Bracket Grammar - CE1 bracket roles
    2. Morphism Generators - Operators with Q-lock
    3. Harmonic Dynamics - Recursive update loop
    4. CE1 Stability Module - Precision/stability subsystem
    
    This gives the machine:
    - Prevented drift
    - Maintained tempo
    - Consistent recursion behavior
    - Smoother convergence
    - Lower variance in morphism chains
    """
    
    def __init__(self, 
                 recursion_op: Callable[[float], float],
                 activation_threshold: int = 11,
                 drift_tolerance: float = 1.0):
        """
        Initialize Harmonic Machine.
        
        Args:
            recursion_op: Main recursion operator R[t[n]]
            activation_threshold: n ≥ this activates stability (default: 11)
            drift_tolerance: Tolerance for drift detection (default: 1.0)
        """
        # Initialize stability module
        self.stability_module = CE1StabilityModule(
            activation_threshold=activation_threshold,
            drift_tolerance=drift_tolerance
        )
        
        # Initialize harmonic dynamics with embedded stability
        self.dynamics = HarmonicDynamics(recursion_op, self.stability_module)
        
        # Initialize bracket grammar and morphism generators
        self.bracket_grammar = ResonantBracket()
        self.morphism_gen = MorphismGenerator()
    
    def step(self, t_n: float, n: int) -> float:
        """
        Single step of harmonic machine.
        
        This is the main entry point - all layers work together.
        """
        return self.dynamics.step(t_n, n)
    
    def evolve(self, t_0: float, n_steps: int) -> list[float]:
        """
        Evolve harmonic machine over n_steps.
        
        Returns sequence: [t[0], t[1], ..., t[n_steps]]
        """
        # Reset stability module history for clean run
        self.stability_module.reset_history()
        return self.dynamics.evolve(t_0, n_steps)
    
    def get_stability_status(self) -> Dict[str, Any]:
        """
        Get current stability status.
        
        Returns dictionary with:
        - q_active: Whether <Q> is active
        - precision: Current precision measure
        - max_drift: Maximum drift observed
        - activation_threshold: n threshold for activation
        """
        if len(self.stability_module.sequence_history) < 2:
            return {
                'q_active': False,
                'precision': float('inf'),
                'max_drift': float('inf'),
                'activation_threshold': self.stability_module.activation_threshold
            }
        
        stability = check_stability(
            self.stability_module.sequence_history,
            tolerance=self.stability_module.drift_tolerance
        )
        
        return {
            'q_active': stability['q_active'],
            'precision': stability['precision'],
            'max_drift': stability['max_drift'],
            'activation_threshold': self.stability_module.activation_threshold
        }




