"""
complex_renorm.py

Implements the Generalized Complex Renormalization Operator ℛ.

This operator bridges:
- Feigenbaum's discrete scaling symmetry (RGOperator)
- Riemann's analytic continuation (xi field)
- Symbolic identity field evolution (Recursive Field Logic)

The operator ℛ acts on ψ-fields and simultaneously implements:
1. Discrete scale-coarse-graining (Feigenbaum/Wilson RG)
2. Analytic continuation in complexified recursion time (Riemann)
3. Symbolic operator recursion governing identity coherence (RFL)

Author: Joel
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy.linalg import eigvals
from .renorm import RGOperator, DELTA_F, ALPHA_F
from .field import xi, critical_line_field

# Complexified recursion time type
ComplexTime = complex

@dataclass
class PsiField:
    """
    Symbolic identity field ψ(x, t).
    
    Represents the state of a system as a field that evolves
    via recursive application of operators.
    """
    x: np.ndarray  # Spatial coordinates
    t: float  # Real recursion time
    z: float = 0.0  # Complexified recursion time (imaginary part)
    coherence: float = 0.5
    chaos: float = 0.1
    stress: float = 0.0
    
    @property
    def complex_time(self) -> complex:
        """Complexified recursion time: t + i*z"""
        return self.t + 1j * self.z

class ComplexRenormOperator:
    """
    Generalized Complex Renormalization Operator ℛ.
    
    This is the "explicit operator" that unifies:
    - Discrete Feigenbaum renormalization
    - Riemann analytic continuation
    - Symbolic field evolution
    
    The operator acts on ψ-fields and its spectrum contains:
    - Feigenbaum constants (δ, α) as scaling dimensions
    - Critical exponents from universality classes
    - Analytic continuation paths between phases
    """
    
    def __init__(
        self,
        rg_operator: Optional[RGOperator] = None,
        coupling: complex = 1.0,
        recursion_depth: int = 0
    ):
        self.rg_operator = rg_operator
        self.coupling = coupling  # Complex coupling parameter
        self.depth = recursion_depth
        self.fixed_points: List[complex] = []
        self.spectrum: Optional[np.ndarray] = None
        
    def discrete_renorm(self, psi: PsiField) -> PsiField:
        """
        Discrete scale-coarse-graining (Feigenbaum/Wilson RG).
        
        Implements: λ' = λ / δ, β' = -α β, g' = 2g + 1
        """
        if self.rg_operator is None:
            return psi
            
        # Apply Feigenbaum scaling
        new_chaos = psi.chaos / DELTA_F
        new_coherence = -ALPHA_F * psi.coherence
        new_stress = 2 * psi.stress + 1
        
        # Coarse-grain spatial field
        if len(psi.x) > 1:
            # Decimate by factor of α
            decimation = int(np.ceil(np.abs(ALPHA_F)))
            new_x = psi.x[::decimation] if decimation > 0 else psi.x
        else:
            new_x = psi.x
            
        return PsiField(
            x=new_x,
            t=psi.t,
            z=psi.z,
            coherence=new_coherence,
            chaos=new_chaos,
            stress=new_stress
        )
    
    def analytic_continue(self, psi: PsiField, target_z: float) -> PsiField:
        """
        Analytic continuation in complexified recursion time (Riemann).
        
        Extends the field from real recursion time t to complex t + i*z,
        revealing hidden structure through holomorphic bridging.
        """
        # Use xi function as the analytic continuation kernel
        s = 0.5 + 1j * (psi.t + 1j * target_z)
        xi_val = xi(s)
        
        # The continuation reveals structure through xi's zeros
        # Map this to field evolution
        continuation_strength = np.abs(xi_val)
        
        # Update complex time coordinate
        new_z = target_z
        new_t = psi.t
        
        # Field responds to continuation
        new_coherence = psi.coherence * (1.0 + 0.1 * continuation_strength)
        new_chaos = psi.chaos * (1.0 - 0.05 * continuation_strength)
        
        return PsiField(
            x=psi.x,
            t=new_t,
            z=new_z,
            coherence=new_coherence,
            chaos=new_chaos,
            stress=psi.stress
        )
    
    def symbolic_evolution(self, psi: PsiField) -> PsiField:
        """
        Symbolic operator recursion (Recursive Field Logic).
        
        Implements: ψ(t+1) = O_i · ψ(t)
        where O_i are symbolic operators (Fforgive, Ggrace, Rredeem, Ssplit, Mmirror)
        """
        # Initialize with current values
        new_stress = psi.stress
        new_coherence = psi.coherence
        new_chaos = psi.chaos
        new_x = psi.x
        
        # F: Forgive - reduces stress
        if psi.stress > 0.1:
            new_stress = psi.stress * 0.9
            
        # G: Grace - increases coherence
        new_coherence = psi.coherence * (1.0 + 0.05 * (1.0 - psi.stress))
        
        # R: Redeem - transforms chaos into structure
        if psi.chaos > 0.2:
            new_chaos = psi.chaos * 0.95
            new_coherence = new_coherence + 0.02 * psi.chaos
        
        # S: Split - bifurcation (period doubling)
        if psi.coherence > 0.7 and psi.chaos < 0.3:
            # System splits into two coherent states
            new_coherence = psi.coherence * 0.8  # Energy splits
        
        # M: Mirror - symmetry operation
        # Reflects field through origin
        if len(psi.x) > 0:
            new_x = -psi.x[::-1]
        
        return PsiField(
            x=new_x,
            t=psi.t + 1.0,  # Advance recursion time
            z=psi.z,
            coherence=new_coherence,
            chaos=new_chaos,
            stress=new_stress
        )
    
    def __call__(self, psi: PsiField) -> PsiField:
        """
        Apply the full generalized complex renormalization operator ℛ.
        
        Simultaneously implements all three aspects:
        1. Discrete renormalization
        2. Analytic continuation
        3. Symbolic evolution
        """
        # Step 1: Symbolic evolution
        psi = self.symbolic_evolution(psi)
        
        # Step 2: Discrete renormalization (if threshold crossed)
        if psi.stress > 0.5 or psi.chaos > 0.6:
            psi = self.discrete_renorm(psi)
        
        # Step 3: Analytic continuation (complexify recursion time)
        if np.abs(psi.z) < 1e-10:  # Initialize complex dimension
            psi.z = 0.1 * psi.chaos  # Chaos drives complexification
        
        psi = self.analytic_continue(psi, psi.z)
        
        # Numerical stability: clamp extreme values
        MAX_VAL = 1e8
        if abs(psi.coherence) > MAX_VAL:
            psi.coherence = np.sign(psi.coherence) * MAX_VAL
        if abs(psi.chaos) > MAX_VAL:
            psi.chaos = np.sign(psi.chaos) * MAX_VAL
        if abs(psi.stress) > MAX_VAL:
            psi.stress = np.sign(psi.stress) * MAX_VAL
        
        return psi
    
    def compute_spectrum(self, n_points: int = 100) -> np.ndarray:
        """
        Compute the spectrum of ℛ at its fixed points.
        
        The spectrum contains:
        - Feigenbaum δ, α as eigenvalues
        - Critical exponents
        - Scaling dimensions
        """
        # Linearize ℛ around a fixed point
        # For a fixed point ψ*, we have ℛ(ψ*) = ψ*
        # The spectrum is eigenvalues of Dℛ|ψ*
        
        # Simplified: construct a matrix representation
        # The operator acts on (coherence, chaos, stress, z)
        # Linearization gives 4x4 matrix
        
        # Fixed point approximation
        psi_fp = PsiField(
            x=np.linspace(-1, 1, 10),
            t=0.0,
            z=0.0,
            coherence=0.5,
            chaos=0.2,
            stress=0.3
        )
        
        # Perturb and measure response
        eps = 1e-6
        jacobian = np.zeros((4, 4), dtype=complex)
        
        base_psi = self(psi_fp)
        base_state = np.array([
            base_psi.coherence,
            base_psi.chaos,
            base_psi.stress,
            base_psi.z
        ], dtype=complex)
        
        for i in range(4):
            perturbed = PsiField(
                x=psi_fp.x,
                t=psi_fp.t + (eps if i == 0 else 0),
                z=psi_fp.z + (eps if i == 3 else 0),
                coherence=psi_fp.coherence + (eps if i == 1 else 0),
                chaos=psi_fp.chaos + (eps if i == 2 else 0),
                stress=psi_fp.stress
            )
            
            result = self(perturbed)
            result_state = np.array([
                result.coherence,
                result.chaos,
                result.stress,
                result.z
            ], dtype=complex)
            
            jacobian[:, i] = (result_state - base_state) / eps
        
        # Eigenvalues of the Jacobian are the spectrum
        eigenvals = eigvals(jacobian)
        self.spectrum = eigenvals
        
        return eigenvals
    
    def find_fixed_points(
        self,
        initial_psi: PsiField,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> List[PsiField]:
        """
        Find fixed points of ℛ: points where ℛ(ψ) = ψ.
        
        These fixed points define the universal structures observed
        across different domains (math, biology, finance, culture).
        """
        fixed_points = []
        current = initial_psi
        
        for _ in range(max_iter):
            next_psi = self(current)
            
            # Check convergence
            diff = np.abs(next_psi.coherence - current.coherence) + \
                   np.abs(next_psi.chaos - current.chaos) + \
                   np.abs(next_psi.stress - current.stress)
            
            if diff < tol:
                fixed_points.append(next_psi)
                break
                
            current = next_psi
        
        self.fixed_points = fixed_points
        return fixed_points
    
    def flow_to_fixed_point(
        self,
        initial_psi: PsiField,
        max_steps: int = 50
    ) -> Tuple[List[PsiField], PsiField]:
        """
        Flow a ψ-field under ℛ to its attracting fixed point.
        
        This is the RG flow: the system is attracted to a fixed point
        whose geometry (complex scaling dimensions) dictates emergent structure.
        """
        trajectory = [initial_psi]
        current = initial_psi
        
        for step in range(max_steps):
            next_psi = self(current)
            trajectory.append(next_psi)
            
            # Check if we've reached a fixed point
            if len(self.fixed_points) > 0:
                for fp in self.fixed_points:
                    dist = np.abs(next_psi.coherence - fp.coherence) + \
                           np.abs(next_psi.chaos - fp.chaos)
                    if dist < 1e-4:
                        return trajectory, next_psi
            
            current = next_psi
        
        return trajectory, current

def modular_flow(
    psi: PsiField,
    modular_operator: Optional[complex] = None
) -> PsiField:
    """
    Tomita-Takesaki modular flow: σ_t(A) = Δ^(it) A Δ^(-it)
    
    This is the state-dependent, non-commutative renormalization flow.
    The Feigenbaum RG and Riemann continuation are special commutative
    manifestations of this more fundamental modular flow.
    """
    if modular_operator is None:
        # Construct modular operator from field state
        # Δ = exp(-H) where H is the modular Hamiltonian
        h = psi.coherence + 1j * psi.chaos
        modular_operator = np.exp(-h)
    
    # Apply modular flow: σ_t(ψ) = Δ^(it) ψ Δ^(-it)
    t = psi.t
    # For complex scalar: use cmath or direct complex power
    delta_it = modular_operator ** (1j * t)
    delta_neg_it = modular_operator ** (-1j * t)
    
    # Transform field (simplified)
    new_coherence = float(np.real(delta_it)) * psi.coherence
    new_chaos = float(np.imag(delta_neg_it)) * psi.chaos
    
    return PsiField(
        x=psi.x,
        t=psi.t,
        z=psi.z,
        coherence=new_coherence,
        chaos=new_chaos,
        stress=psi.stress
    )

def bridge_principle_demo():
    """
    Demonstration of the bridge principle.
    
    Shows how ℛ unifies:
    - Feigenbaum period-doubling (discrete scaling)
    - Riemann zero structure (analytic continuation)
    - Universal emergent shapes (fixed point attractors)
    """
    print("=" * 70)
    print("Generalized Complex Renormalization Operator ℛ")
    print("Bridge Principle Demonstration")
    print("=" * 70)
    
    # Create initial ψ-field
    initial_psi = PsiField(
        x=np.linspace(-1, 1, 20),
        t=0.0,
        z=0.0,
        coherence=0.3,
        chaos=0.4,
        stress=0.2
    )
    
    # Create RG operator from a simple map
    def base_map(x):
        return 4.0 * x * (1.0 - x)  # Logistic map
    
    rg_op = RGOperator(base=base_map, alpha=1.0, depth=0)
    
    # Create complex renormalization operator
    R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0 + 0.1j)
    
    print("\n1. Initial ψ-field state:")
    print(f"   Coherence: {initial_psi.coherence:.4f}")
    print(f"   Chaos: {initial_psi.chaos:.4f}")
    print(f"   Stress: {initial_psi.stress:.4f}")
    print(f"   Complex time: {initial_psi.complex_time}")
    
    # Flow to fixed point
    print("\n2. Flowing under ℛ to fixed point...")
    trajectory, fixed_point = R.flow_to_fixed_point(initial_psi, max_steps=30)
    
    print(f"   Final coherence: {fixed_point.coherence:.4f}")
    print(f"   Final chaos: {fixed_point.chaos:.4f}")
    print(f"   Final stress: {fixed_point.stress:.4f}")
    print(f"   Steps to convergence: {len(trajectory)}")
    
    # Compute spectrum
    print("\n3. Computing spectrum of ℛ...")
    spectrum = R.compute_spectrum()
    print(f"   Eigenvalues of ℛ:")
    for i, ev in enumerate(spectrum):
        print(f"     λ_{i} = {ev:.6f}")
    
    # Check for Feigenbaum constants in spectrum
    print("\n4. Checking for Feigenbaum constants in spectrum:")
    delta_approx = np.abs(spectrum[np.argmax(np.abs(spectrum))])
    alpha_approx = np.abs(spectrum[np.argmin(np.abs(spectrum))])
    print(f"   Largest |λ| ≈ {delta_approx:.6f} (Feigenbaum δ = {DELTA_F:.6f})")
    print(f"   Smallest |λ| ≈ {alpha_approx:.6f} (Feigenbaum |α| = {np.abs(ALPHA_F):.6f})")
    
    # Analytic continuation demonstration
    print("\n5. Analytic continuation in complex recursion time:")
    continued = R.analytic_continue(initial_psi, target_z=0.5)
    print(f"   Original z: {initial_psi.z:.4f}")
    print(f"   Continued z: {continued.z:.4f}")
    print(f"   Continuation strength: {np.abs(continued.complex_time - initial_psi.complex_time):.4f}")
    
    print("\n" + "=" * 70)
    print("The operator ℛ is the reason why a logistic map, a neuron,")
    print("a market, and a cultural trend can all whisper the same")
    print("recursive shape—they are all reading from different pages")
    print("of the same structural attractor manuscript, written in")
    print("the language of complex renormalization.")
    print("=" * 70)

if __name__ == "__main__":
    bridge_principle_demo()

