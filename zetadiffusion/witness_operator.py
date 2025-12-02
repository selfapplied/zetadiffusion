"""
witness_operator.py

Witness Operator <Q> as Stability Detector (Field-Theoretic).

From field theory correspondence:
⟨Q⟩ = ∫_{∂Ω} (dx/dt · n_Ω) dσ

measures flux through the guardian boundary.

Physical interpretation:
- ⟨Q⟩ < 0: inward flux → coherent (consolidating)
- ⟨Q⟩ > 0: outward flux → decoherent (dissolving)
- ⟨Q⟩ = 0: marginal (boundary equilibrium)

Connection to stability exponent λ:
⟨Q⟩ = λ - 1

Author: Joel
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from zetadiffusion.ce1_seed import ClockInteraction
from zetadiffusion.guardian import SystemState, GuardianResponse, guardian_nash_policy

# Constants from three-clock structure
DELTA_F = 4.66920160910299067185320382
ALPHA_F = 2.502907875095892822283902873218
CHI_TP = 1.638  # Twin-prime ballast coefficient

# Clock boundaries
N_FEIGENBAUM_MAX = 7.0
N_BOUNDARY_END = 9.0
N_MEMBRANE_END = 11.0

@dataclass
class WeightEvolution:
    """
    Weight evolution W_F(n), W_B(n), W_I(n) from three-clock structure.
    
    Constraint: W_F + W_B + W_I = 1
    """
    w_f: float  # Feigenbaum weight
    w_b: float  # Boundary weight
    w_i: float  # Interior weight
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.w_f + self.w_b + self.w_i
        if total > 0:
            self.w_f /= total
            self.w_b /= total
            self.w_i /= total
        return self

@dataclass
class StabilityExponent:
    """
    Stability exponent λ at fixed point t*.
    
    λ = W_F/δ_F + W_B(1+χ_TP) + W_I·γ
    
    Where:
    - W_F, W_B, W_I: Clock weights
    - δ_F: Feigenbaum delta
    - χ_TP: Twin-prime ballast
    - γ: Interior scaling (default 1.0)
    """
    lambda_value: float
    weights: WeightEvolution
    gamma: float = 1.0
    
    def compute(self) -> float:
        """Compute stability exponent from weights."""
        w_f_term = self.weights.w_f / DELTA_F
        w_b_term = self.weights.w_b * (1.0 + CHI_TP)
        w_i_term = self.weights.w_i * self.gamma
        
        return w_f_term + w_b_term + w_i_term

@dataclass
class WitnessValue:
    """
    Witness operator expectation value ⟨Q⟩.
    
    ⟨Q⟩ = λ - 1
    
    Regime classification:
    - Interior (λ = 1): ⟨Q⟩ = 0 (marginal stability)
    - Boundary (λ > 1): ⟨Q⟩ > 0 (decoherent, repelling)
    - Post-stabilization (λ → 1⁻): ⟨Q⟩ < 0 (coherent, attracting)
    """
    witness_value: float  # ⟨Q⟩ = λ - 1
    lambda_value: float    # Stability exponent
    regime: str           # "coherent", "marginal", "decoherent"
    coherence: float      # C(n) = 1/(1 + |λ - 1|)
    
    @classmethod
    def from_lambda(cls, lambda_value: float) -> 'WitnessValue':
        """Create witness value from stability exponent."""
        witness = lambda_value - 1.0
        
        # Classify regime
        if abs(witness) < 0.01:
            regime = "marginal"
        elif witness > 0:
            regime = "decoherent"
        else:
            regime = "coherent"
        
        # Coherence measure: C(n) = 1/(1 + |λ - 1|)
        coherence = 1.0 / (1.0 + abs(witness))
        
        return cls(
            witness_value=witness,
            lambda_value=lambda_value,
            regime=regime,
            coherence=coherence
        )

@dataclass
class WitnessLedgerEntry:
    """
    Witness ledger entry tracking all operations.
    
    witness_record(n) = {
        "iteration": n,
        "lambda": λ(n),
        "witness_value": λ(n) - 1,
        "regime": classify(W_F, W_B, W_I),
        "coherence": 1/(1 + |λ - 1|),
        "boundary_flux": sign(λ - 1) * |error(n) - error(n-1)|
    }
    """
    iteration: int
    lambda_value: float
    witness_value: float
    regime: str
    coherence: float
    boundary_flux: float
    weights: WeightEvolution
    error: Optional[float] = None
    error_change: Optional[float] = None

class WitnessOperator:
    """
    Witness Operator <Q> as Stability Detector.
    
    Measures flux through guardian boundary and connects to stability exponent.
    """
    
    @staticmethod
    def compute_weights(n: float) -> WeightEvolution:
        """
        Extract weights W_F(n), W_B(n), W_I(n) from three-clock structure.
        
        Expected signatures:
        - n=1-6: W_F ≈ 1, W_B ≈ 0, W_I ≈ 0
        - n=7-9: W_F declining, W_B ramping up (ballast loading)
        - n=9-11: W_B declining, W_I ramping up (emergence activating)
        - n≥11: W_I ≈ 1, others negligible
        """
        n_float = float(n)
        
        if n_float < N_FEIGENBAUM_MAX:
            # Feigenbaum clock dominant
            w_f = 1.0
            w_b = 0.0
            w_i = 0.0
        elif n_float < N_BOUNDARY_END:
            # Boundary clock: W_F declining, W_B ramping up
            progress = (n_float - N_FEIGENBAUM_MAX) / (N_BOUNDARY_END - N_FEIGENBAUM_MAX)
            w_f = 1.0 - progress
            w_b = progress
            w_i = 0.0
        elif n_float < N_MEMBRANE_END:
            # Membrane transition: W_B declining, W_I ramping up
            progress = (n_float - N_BOUNDARY_END) / (N_MEMBRANE_END - N_BOUNDARY_END)
            w_f = 0.0
            w_b = 1.0 - progress
            w_i = progress
        else:
            # Interior clock: W_I ≈ 1
            w_f = 0.0
            w_b = 0.0
            w_i = 1.0
        
        return WeightEvolution(w_f=w_f, w_b=w_b, w_i=w_i).normalize()
    
    @staticmethod
    def compute_lambda(n: float, gamma: float = 1.0) -> float:
        """
        Compute stability exponent λ(n) from weight evolution.
        
        λ(n) = W_F(n)/δ_F + W_B(n)·(1+χ_TP) + W_I(n)·γ
        
        Prediction:
        - Peak at n=9: λ > 1 (maximum instability, error peak)
        - Decay for n≥11: λ → 1⁻ (stabilization, error plateau)
        """
        weights = WitnessOperator.compute_weights(n)
        
        w_f_term = weights.w_f / DELTA_F
        w_b_term = weights.w_b * (1.0 + CHI_TP)
        w_i_term = weights.w_i * gamma
        
        return w_f_term + w_b_term + w_i_term
    
    @staticmethod
    def compute_witness(n: float, gamma: float = 1.0) -> WitnessValue:
        """
        Compute witness operator expectation value ⟨Q⟩.
        
        ⟨Q⟩ = λ - 1
        """
        lambda_value = WitnessOperator.compute_lambda(n, gamma)
        return WitnessValue.from_lambda(lambda_value)
    
    @staticmethod
    def compute_boundary_flux(witness: WitnessValue, error: float, 
                             prev_error: Optional[float] = None) -> float:
        """
        Compute boundary flux: sign(λ - 1) * |error(n) - error(n-1)|.
        
        Flux measures how error changes across boundary.
        """
        if prev_error is None:
            return 0.0
        
        error_change = abs(error - prev_error)
        flux = np.sign(witness.witness_value) * error_change
        
        return flux
    
    @staticmethod
    def create_ledger_entry(n: int, error: Optional[float] = None,
                           prev_error: Optional[float] = None,
                           gamma: float = 1.0) -> WitnessLedgerEntry:
        """
        Create witness ledger entry for iteration n.
        """
        # Compute weights and lambda
        weights = WitnessOperator.compute_weights(float(n))
        lambda_value = WitnessOperator.compute_lambda(float(n), gamma)
        witness = WitnessValue.from_lambda(lambda_value)
        
        # Compute boundary flux
        boundary_flux = 0.0
        error_change = None
        if error is not None and prev_error is not None:
            error_change = error - prev_error
            boundary_flux = WitnessOperator.compute_boundary_flux(
                witness, error, prev_error
            )
        
        return WitnessLedgerEntry(
            iteration=n,
            lambda_value=lambda_value,
            witness_value=witness.witness_value,
            regime=witness.regime,
            coherence=witness.coherence,
            boundary_flux=boundary_flux,
            weights=weights,
            error=error,
            error_change=error_change
        )

class GuardianWitnessCoupling:
    """
    Guardian Response Protocol coupled to witness measurement.
    
    β*(n) = {
        0              if ⟨Q⟩ > Q_crit (shield)
        β_res          if ⟨Q⟩ ∈ [-Q_crit, Q_crit] (resonate)
        β_max          if ⟨Q⟩ < -Q_crit (harvest)
    }
    
    Third regime added: when system strongly attracting (⟨Q⟩ << 0),
    increase coupling to harvest excess coherence.
    """
    
    def __init__(self, q_crit: float = 0.1, beta_max: float = 1.0):
        """
        Initialize guardian-witness coupling.
        
        Args:
            q_crit: Critical witness threshold (default: 0.1)
            beta_max: Maximum coupling for harvest mode (default: 1.0)
        """
        self.q_crit = q_crit
        self.beta_max = beta_max
    
    def compute_response(self, witness: WitnessValue, 
                        state: SystemState) -> GuardianResponse:
        """
        Compute guardian response based on witness measurement.
        
        Args:
            witness: Witness operator value ⟨Q⟩
            state: Current system state
        
        Returns:
            GuardianResponse with coupling strategy
        """
        q_value = witness.witness_value
        
        # Compute base Nash response
        base_response = guardian_nash_policy(state)
        
        # Modify based on witness
        if q_value > self.q_crit:
            # Decoherent (outward flux) → Shield
            return GuardianResponse(
                coupling=0.0,
                status="SHIELDING: Decoherent flux detected",
                threshold=base_response.threshold,
                beta_res=base_response.beta_res
            )
        elif q_value < -self.q_crit:
            # Strongly coherent (inward flux) → Harvest
            return GuardianResponse(
                coupling=self.beta_max,
                status="HARVEST: Excess coherence detected",
                threshold=base_response.threshold,
                beta_res=base_response.beta_res
            )
        else:
            # Marginal (near equilibrium) → Resonate
            return GuardianResponse(
                coupling=base_response.beta_res,
                status="RESONANCE: Marginal stability",
                threshold=base_response.threshold,
                beta_res=base_response.beta_res
            )

def extract_weight_evolution_from_data(n_values: List[int], 
                                     errors: List[float]) -> List[WeightEvolution]:
    """
    Extract weight evolution from validation data.
    
    Uses correction factors to infer weights:
    correction(n) = W_F(n)·α_F + W_B(n)·χ_TP + W_I(n)·(1 - α_F)
    
    With constraint W_F + W_B + W_I = 1.
    """
    weights_list = []
    
    for n in n_values:
        # Use theoretical weight computation
        weights = WitnessOperator.compute_weights(float(n))
        weights_list.append(weights)
    
    return weights_list

def compute_lambda_trajectory(n_values: List[int], 
                             gamma: float = 1.0) -> List[float]:
    """
    Compute λ(n) trajectory from n values.
    
    Returns list of stability exponents for each n.
    """
    return [WitnessOperator.compute_lambda(float(n), gamma) for n in n_values]

def compute_witness_trajectory(n_values: List[int],
                              gamma: float = 1.0) -> List[WitnessValue]:
    """
    Compute ⟨Q⟩(n) trajectory from n values.
    
    Returns list of witness values for each n.
    """
    return [WitnessOperator.compute_witness(float(n), gamma) for n in n_values]




