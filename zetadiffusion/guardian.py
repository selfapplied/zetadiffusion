"""
guardian.py

Guardian Nash Policy Kernel - Game-theoretic stability control for FEG-0.2.
Implements the Guardian-Entropy game with Hurst-modulated switching strategy.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

# Universal constants
FEIGENBAUM_DELTA = 4.66920160910299067185320382
G_MAX = 1.000  # Gini Coefficient Limit
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant

# Guardian utility weights
WEIGHT_COHERENCE = 1.0  # a: weight on preserving coherence
WEIGHT_CHAOS = 2.5      # b: weight on suppressing chaos (high penalty)
COST_COUPLING = 0.05    # c: cost of maintaining high coupling

@dataclass
class SystemState:
    """Current state of the coherence-chaos system."""
    coherence: float      # C: current coherence level
    chaos: float          # λ: current Lyapunov/chaos level
    stress: float         # G: current Gini/stress metric [0, 1]
    hurst: float         # H: Hurst exponent [0, 1]
    gamma: float = 1.0   # Chaos growth rate
    delta: float = 0.5   # Coherence decay rate

@dataclass
class GuardianResponse:
    """Guardian's strategic response."""
    coupling: float       # β: optimal coupling strength
    status: str          # "RESONANCE" or "SHIELDING"
    threshold: float     # G_crit(H): critical stress threshold
    beta_res: float      # Optimal resonance coupling

def calculate_beta_res(gamma: float, delta: float) -> float:
    """
    Calculate optimal resonance coupling β_res from Nash equilibrium.
    
    Solves: β_res = ((b*γ - a*δ) / (2*c))^(1/3)
    
    This is the cube-root Nash solution that balances:
    - High coherence (a*C*)
    - Low chaos (b*λ*)
    - Low coupling cost (c*β²)
    """
    numerator = (WEIGHT_CHAOS * gamma) - (WEIGHT_COHERENCE * delta)
    
    if numerator <= 0:
        # Constraint violation: chaos penalty doesn't exceed coherence gain
        # Force minimal coupling to avoid singularity
        return 0.001
    
    return (numerator / (2 * COST_COUPLING)) ** (1/3)

def calculate_g_crit(hurst: float) -> float:
    """
    Calculate Hurst-modulated critical stress threshold G_crit(H).
    
    Uses Feigenbaum Delta as scaling factor for persistence sensitivity.
    
    H > 0.5 (Persistent): Lower threshold (shield earlier)
    H < 0.5 (Anti-persistent): Higher threshold (more permissive)
    H = 0.5 (Random): G_crit = G_max
    """
    # Risk factor: tanh(δ_F * (H - 0.5))
    # Maps H to [-1, 1] range, centered at H=0.5
    risk_factor = np.tanh(FEIGENBAUM_DELTA * (hurst - 0.5))
    
    # Modulate G_max: subtract up to 20% for high persistence
    return G_MAX * (1.0 - (0.2 * risk_factor))

def guardian_nash_policy(state: SystemState) -> GuardianResponse:
    """
    Guardian Nash Policy Kernel.
    
    Implements the game-theoretic equilibrium strategy:
    - Calculates optimal resonance coupling β_res
    - Computes Hurst-modulated critical threshold G_crit(H)
    - Chooses RESONANCE or SHIELDING mode based on current stress
    
    Args:
        state: Current system state (coherence, chaos, stress, hurst, etc.)
        
    Returns:
        GuardianResponse with optimal coupling and strategy
    """
    # Calculate optimal resonance coupling
    beta_res = calculate_beta_res(state.gamma, state.delta)
    
    # Calculate Hurst-modulated threshold
    g_crit = calculate_g_crit(state.hurst)
    
    # Nash strategy: compare current stress to threshold
    if state.stress >= g_crit:
        # SHIELDING MODE: Defect/Decouple
        # Guardian refuses to play when table is rigged
        coupling = 0.0
        status = "SHIELDING: Persistence Risk Detected"
    else:
        # RESONANCE MODE: Cooperate/Couple
        # System orbits stable fixed point (C*, λ*)
        coupling = beta_res
        status = "RESONANCE: Equilibrium Tracking"
    
    return GuardianResponse(
        coupling=coupling,
        status=status,
        threshold=g_crit,
        beta_res=beta_res
    )

def calculate_fixed_points(beta: float, gamma: float = 1.0, delta: float = 0.5) -> Tuple[float, float]:
    """
    Calculate Lotka-Volterra fixed points for given coupling.
    
    Non-trivial equilibrium:
    C* = δ / β
    λ* = γ / β
    
    Returns:
        (C*, λ*): Coherence and chaos at equilibrium
    """
    if beta < 1e-10:
        # Decoupled: both diverge
        return (np.inf, np.inf)
    
    c_star = delta / beta
    lambda_star = gamma / beta
    
    return (c_star, lambda_star)

def guardian_potential(stress: float, coupling: float) -> float:
    """
    Guardian Potential V_G(G, β).
    
    Barrier potential that scales as system approaches chaos limit.
    V_G ~ 1 / (G_max - G) when coupling is active.
    
    This is the "smart potential" that enforces boundary awareness.
    """
    if stress >= G_MAX:
        return np.inf  # Hard boundary
    
    if coupling < 1e-10:
        return 0.0  # No potential when decoupled
    
    # Barrier potential: stronger near boundary
    barrier = 1.0 / (G_MAX - stress + 1e-10)
    return coupling * barrier

def calculate_guardian_utility(state: SystemState, beta: float) -> float:
    """
    Calculate Guardian's utility U_G(β) for given coupling.
    
    U_G = a*C* - b*λ* - c*β²
    
    Where (C*, λ*) are the fixed points for coupling β.
    """
    c_star, lambda_star = calculate_fixed_points(beta, state.gamma, state.delta)
    
    # Utility components
    coherence_gain = WEIGHT_COHERENCE * c_star
    chaos_penalty = WEIGHT_CHAOS * lambda_star
    coupling_cost = COST_COUPLING * (beta ** 2)
    
    return coherence_gain - chaos_penalty - coupling_cost

def directional_gradient(state: SystemState, metric_tensor: np.ndarray = None) -> float:
    """
    Calculate directional gradient for Guardian awareness.
    
    Vector_Dir = ∇(S_int) - ∇(S_ext)
    
    If > 0: moving toward boundary (danger)
    If < 0: consolidating internal coherence (safety)
    """
    # Simplified: gradient of entropy difference
    # Internal entropy (coherence) vs external entropy (chaos)
    s_int = -state.coherence  # Negative entropy = order
    s_ext = state.chaos       # Positive entropy = disorder
    
    # Gradient approximation
    gradient = s_ext - s_int
    
    return gradient

def flux_across_boundary(state: SystemState, coupling: float) -> float:
    """
    Calculate flux Φ across boundary ∂Ω.
    
    Flux_Φ = Integral(Vector_Dir · dA) over ∂Ω
    
    Simplified: flux proportional to coupling and stress gradient.
    """
    gradient = directional_gradient(state)
    flux = coupling * gradient * state.stress
    
    return flux

def apply_guardian_correction(state: SystemState, response: GuardianResponse) -> SystemState:
    """
    Apply Guardian correction to system state.
    
    If flux exceeds threshold, activate damping field:
    x_{n+1} = (x_n / δ_F) + Correction_Term
    """
    flux = flux_across_boundary(state, response.coupling)
    threshold = 0.1  # Flux threshold
    
    if flux > threshold and response.status == "SHIELDING":
        # Apply Feigenbaum renormalization correction
        correction = state.coherence / FEIGENBAUM_DELTA
        state.coherence = correction
        # Reset chaos to prevent blow-up
        state.chaos = state.chaos / FEIGENBAUM_DELTA
    
    return state








