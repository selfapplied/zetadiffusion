"""
operator_analysis.py

Four computational tasks for analyzing the Generalized Complex Renormalization Operator ℛ:

1. Known Limits Verification - Tests Gaussian fixed point, Feigenbaum residue, logistic map correspondence
2. Branch Cut Mapping - Visualizes complex plane structure, detects phase discontinuities
3. Pole Structure Analysis - Locates singularities, computes residues, classifies pole orders
4. Spectral Signature - Finds fixed points, linearizes operator, identifies eigenvalues beyond δ_F

Author: Joel
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize

from .complex_renorm import ComplexRenormOperator, PsiField
from .renorm import RGOperator, DELTA_F, ALPHA_F
from .field import xi

# ============================================================================
# Task 1: Known Limits Verification
# ============================================================================

@dataclass
class VerificationResult:
    """Results from known limits verification."""
    gaussian_fixed_point: Dict[str, float]
    feigenbaum_residue: Dict[str, float]
    logistic_correspondence: Dict[str, float]
    all_passed: bool

def gaussian_fixed_point_test() -> Dict[str, float]:
    """
    Verify Gaussian fixed point behavior.
    
    The Gaussian fixed point is a well-known RG fixed point where
    the operator should converge to a Gaussian distribution under
    repeated coarse-graining.
    """
    # Create a Gaussian-like initial field
    x = np.linspace(-3, 3, 50)
    gaussian_field = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    # Normalize
    gaussian_field = gaussian_field / np.max(gaussian_field)
    
    # Create base map that preserves Gaussian structure
    def gaussian_map(x_val):
        # Gaussian fixed point map: f(x) = x * exp(-x^2/2)
        return x_val * np.exp(-0.5 * x_val**2)
    
    rg_op = RGOperator(base=gaussian_map, alpha=1.0, depth=0)
    
    # Test convergence under RG flow
    initial_psi = PsiField(
        x=x,
        t=0.0,
        z=0.0,
        coherence=0.8,  # High coherence for Gaussian
        chaos=0.1,
        stress=0.0
    )
    
    R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0)
    
    # Flow to fixed point
    trajectory, fixed = R.flow_to_fixed_point(initial_psi, max_steps=20)
    
    # Check if coherence increased (Gaussian is stable)
    coherence_change = fixed.coherence - initial_psi.coherence
    
    # Gaussian fixed point should be stable (coherence maintained or increased)
    is_stable = coherence_change >= -0.1
    
    return {
        "initial_coherence": initial_psi.coherence,
        "final_coherence": fixed.coherence,
        "coherence_change": coherence_change,
        "is_stable": float(is_stable),
        "convergence_steps": len(trajectory)
    }

def feigenbaum_residue_test() -> Dict[str, float]:
    """
    Verify Feigenbaum residue calculation.
    
    The Feigenbaum residue is the deviation from perfect scaling
    at the fixed point. Should converge to zero under RG flow.
    """
    # Create logistic map base
    def logistic_map(x):
        r = 3.5699456  # Feigenbaum accumulation point
        return r * x * (1.0 - x)
    
    rg_op = RGOperator(base=logistic_map, alpha=1.0, depth=0)
    
    # Create operator
    R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0)
    
    # Test with initial state near chaos threshold
    initial_psi = PsiField(
        x=np.linspace(0, 1, 20),
        t=0.0,
        z=0.0,
        coherence=0.3,
        chaos=0.5,  # Near chaos threshold
        stress=0.4
    )
    
    # Flow and measure residue
    trajectory, fixed = R.flow_to_fixed_point(initial_psi, max_steps=30)
    
    # Residue = deviation from Feigenbaum scaling
    # At fixed point, chaos should scale as 1/δ
    expected_chaos = initial_psi.chaos / DELTA_F
    actual_chaos = fixed.chaos
    residue = np.abs(actual_chaos - expected_chaos)
    
    # Also check coherence scaling with α
    expected_coherence = -ALPHA_F * initial_psi.coherence
    actual_coherence = fixed.coherence
    coherence_residue = np.abs(actual_coherence - expected_coherence)
    
    return {
        "chaos_residue": residue,
        "coherence_residue": coherence_residue,
        "total_residue": residue + coherence_residue,
        "converged": float(residue < 0.1)
    }

def logistic_map_correspondence_test() -> Dict[str, float]:
    """
    Verify correspondence with logistic map period-doubling.
    
    The logistic map x_{n+1} = r·x_n(1-x_n) exhibits period-doubling
    bifurcations with Feigenbaum scaling. Verify operator captures this.
    """
    # Test multiple r values across period-doubling cascade
    r_values = np.linspace(3.0, 3.5699456, 20)  # Up to accumulation point
    periods = []
    
    for r in r_values:
        def logistic_r(x):
            return r * x * (1.0 - x)
        
        rg_op = RGOperator(base=logistic_r, alpha=1.0, depth=0)
        R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0)
        
        initial_psi = PsiField(
            x=np.linspace(0, 1, 20),
            t=0.0,
            z=0.0,
            coherence=0.5,
            chaos=0.3,
            stress=0.2
        )
        
        # Flow and detect period
        trajectory, fixed = R.flow_to_fixed_point(initial_psi, max_steps=15)
        
        # Estimate period from trajectory oscillations
        if len(trajectory) > 5:
            coherence_vals = [p.coherence for p in trajectory[-5:]]
            # Count sign changes (rough period estimate)
            signs = np.sign(np.diff(coherence_vals))
            period_estimate = len(np.where(signs != 0)[0]) + 1
        else:
            period_estimate = 1
        
        periods.append(period_estimate)
    
    # Check if periods follow Feigenbaum scaling
    # Periods should double: 1, 2, 4, 8, ...
    period_doublings = []
    for i in range(len(periods) - 1):
        if periods[i+1] > periods[i]:
            period_doublings.append(periods[i+1] / max(periods[i], 1))
    
    avg_doubling = np.mean(period_doublings) if period_doublings else 1.0
    
    return {
        "r_range": (float(r_values[0]), float(r_values[-1])),
        "periods_detected": len(set(periods)),
        "avg_doubling_factor": avg_doubling,
        "expected_doubling": 2.0,
        "correspondence_error": np.abs(avg_doubling - 2.0)
    }

def known_limits_verification() -> VerificationResult:
    """
    Run all known limits verification tests.
    """
    print("=" * 70)
    print("Task 1: Known Limits Verification")
    print("=" * 70)
    
    print("\n1.1 Testing Gaussian fixed point...")
    gaussian_result = gaussian_fixed_point_test()
    print(f"   Initial coherence: {gaussian_result['initial_coherence']:.4f}")
    print(f"   Final coherence: {gaussian_result['final_coherence']:.4f}")
    print(f"   Stable: {bool(gaussian_result['is_stable'])}")
    
    print("\n1.2 Testing Feigenbaum residue...")
    residue_result = feigenbaum_residue_test()
    print(f"   Chaos residue: {residue_result['chaos_residue']:.6f}")
    print(f"   Coherence residue: {residue_result['coherence_residue']:.6f}")
    print(f"   Converged: {bool(residue_result['converged'])}")
    
    print("\n1.3 Testing logistic map correspondence...")
    logistic_result = logistic_map_correspondence_test()
    print(f"   Periods detected: {logistic_result['periods_detected']}")
    print(f"   Avg doubling factor: {logistic_result['avg_doubling_factor']:.4f}")
    print(f"   Expected: 2.0, Error: {logistic_result['correspondence_error']:.4f}")
    
    all_passed = (
        gaussian_result['is_stable'] > 0.5 and
        residue_result['converged'] > 0.5 and
        logistic_result['correspondence_error'] < 1.0
    )
    
    print(f"\n✓ All tests passed: {all_passed}")
    
    return VerificationResult(
        gaussian_fixed_point=gaussian_result,
        feigenbaum_residue=residue_result,
        logistic_correspondence=logistic_result,
        all_passed=all_passed
    )

# ============================================================================
# Task 2: Branch Cut Mapping
# ============================================================================

@dataclass
class BranchCutResult:
    """Results from branch cut analysis."""
    branch_cuts: List[Tuple[complex, complex]]  # (start, end) pairs
    phase_discontinuities: List[complex]
    complex_structure: np.ndarray  # Phase map

def map_branch_cuts(
    R: ComplexRenormOperator,
    z_range: Tuple[float, float] = (-2.0, 2.0),
    t_range: Tuple[float, float] = (0.0, 2.0),
    resolution: int = 50
) -> BranchCutResult:
    """
    Map branch cuts in the complex recursion time plane.
    
    Visualizes the complex plane structure and detects phase discontinuities
    where the operator's analytic continuation has branch cuts.
    """
    print("\n" + "=" * 70)
    print("Task 2: Branch Cut Mapping")
    print("=" * 70)
    
    z_vals = np.linspace(z_range[0], z_range[1], resolution)
    t_vals = np.linspace(t_range[0], t_range[1], resolution)
    
    # Create meshgrid
    Z, T = np.meshgrid(z_vals, t_vals)
    
    # Initialize field
    base_psi = PsiField(
        x=np.linspace(-1, 1, 10),
        t=0.0,
        z=0.0,
        coherence=0.5,
        chaos=0.3,
        stress=0.2
    )
    
    # Map phase over complex plane
    phase_map = np.zeros_like(Z, dtype=complex)
    magnitude_map = np.zeros_like(Z, dtype=float)
    
    branch_cuts = []
    phase_discontinuities = []
    
    # Use xi function for branch cut detection (Riemann connection)
    for i, t in enumerate(t_vals):
        for j, z in enumerate(z_vals):
            # Create field with complex time
            psi = PsiField(
                x=base_psi.x,
                t=t,
                z=z,
                coherence=base_psi.coherence,
                chaos=base_psi.chaos,
                stress=base_psi.stress
            )
            
            # Apply operator
            try:
                result = R(psi)
                
                # Compute complex value
                complex_val = result.complex_time
                phase_map[i, j] = complex_val
                magnitude_map[i, j] = np.abs(complex_val)
                
                # Also check xi function for branch cuts (Riemann connection)
                s = 0.5 + 1j * complex_val
                try:
                    from .field import xi
                    xi_val = xi(s)
                    xi_phase = np.angle(xi_val)
                    
                    # Detect phase discontinuities (branch cuts)
                    if i > 0 and j > 0:
                        # Check operator phase
                        phase_diff_v = np.angle(phase_map[i, j]) - np.angle(phase_map[i-1, j])
                        phase_diff_h = np.angle(phase_map[i, j]) - np.angle(phase_map[i, j-1])
                        
                        # Check xi phase (Riemann zeros create branch cuts)
                        if i > 0:
                            prev_xi_phase = np.angle(xi(0.5 + 1j * phase_map[i-1, j]))
                            xi_phase_diff = abs(xi_phase - prev_xi_phase)
                            if xi_phase_diff > np.pi * 0.9:  # Near π jump
                                phase_discontinuities.append(t + 1j * z)
                        
                        # Phase jumps > π indicate branch cut
                        if np.abs(phase_diff_v) > np.pi * 0.9:
                            phase_discontinuities.append(t + 1j * z)
                        
                        if np.abs(phase_diff_h) > np.pi * 0.9:
                            phase_discontinuities.append(t + 1j * z)
                except:
                    pass
                    
            except (OverflowError, ValueError):
                # Singularity - likely branch cut
                phase_discontinuities.append(t + 1j * z)
    
    # Identify branch cut segments
    if len(phase_discontinuities) > 1:
        # Group nearby discontinuities into cuts
        for i, disc1 in enumerate(phase_discontinuities[:-1]):
            for disc2 in phase_discontinuities[i+1:]:
                dist = np.abs(disc1 - disc2)
                if dist < 0.2:  # Threshold for connected cuts
                    branch_cuts.append((disc1, disc2))
    
    print(f"\n   Branch cuts detected: {len(branch_cuts)}")
    print(f"   Phase discontinuities: {len(phase_discontinuities)}")
    print(f"   Complex structure mapped: {resolution}x{resolution} grid")
    
    return BranchCutResult(
        branch_cuts=branch_cuts,
        phase_discontinuities=phase_discontinuities,
        complex_structure=phase_map
    )

# ============================================================================
# Task 3: Pole Structure Analysis
# ============================================================================

@dataclass
class PoleResult:
    """Results from pole analysis."""
    poles: List[complex]
    residues: List[complex]
    orders: List[int]
    pole_locations: Dict[str, List[complex]]

def analyze_pole_structure(
    R: ComplexRenormOperator,
    search_region: Tuple[complex, complex] = (-2-2j, 2+2j),
    resolution: int = 100
) -> PoleResult:
    """
    Locate singularities, compute residues, and classify pole orders.
    
    Poles occur where the operator becomes singular (denominator → 0).
    """
    print("\n" + "=" * 70)
    print("Task 3: Pole Structure Analysis")
    print("=" * 70)
    
    # Create systematic search grid (enhanced resolution for pole detection)
    z_min, z_max = search_region[0].imag, search_region[1].imag
    t_min, t_max = search_region[0].real, search_region[1].real
    
    # Use higher resolution for systematic search
    search_resolution = max(resolution, 50)  # Minimum 50x50 grid
    z_vals = np.linspace(z_min, z_max, search_resolution)
    t_vals = np.linspace(t_min, t_max, search_resolution)
    
    base_psi = PsiField(
        x=np.linspace(-1, 1, 10),
        t=0.0,
        z=0.0,
        coherence=0.5,
        chaos=0.3,
        stress=0.2
    )
    
    # Map operator magnitude to find singularities
    magnitude_map = np.zeros((len(t_vals), len(z_vals)))
    phase_map = np.zeros((len(t_vals), len(z_vals)))
    
    poles = []
    candidate_poles = []
    
    # Thresholds for pole detection (systematic search)
    POLE_THRESHOLD = 1e3  # Lower threshold for systematic search
    GRADIENT_THRESHOLD = 1e2  # Lower gradient threshold
    MAGNITUDE_THRESHOLD = 1e4  # Magnitude jump threshold
    
    for i, t in enumerate(t_vals):
        for j, z in enumerate(z_vals):
            psi = PsiField(
                x=base_psi.x,
                t=t,
                z=z,
                coherence=base_psi.coherence,
                chaos=base_psi.chaos,
                stress=base_psi.stress
            )
            
            try:
                result = R(psi)
                # Check for singularity (large magnitude or NaN)
                mag = np.abs(result.complex_time)
                magnitude_map[i, j] = mag
                phase_map[i, j] = np.angle(result.complex_time)
                
                # Systematic pole detection: multiple criteria
                # 1. Large magnitude (singularity)
                if mag > POLE_THRESHOLD or np.isnan(mag) or np.isinf(mag):
                    candidate_poles.append(t + 1j * z)
                
                # 2. Gradient-based detection (rapid change indicates pole)
                if i > 0 and j > 0:
                    grad_t = abs(magnitude_map[i, j] - magnitude_map[i-1, j])
                    grad_z = abs(magnitude_map[i, j] - magnitude_map[i, j-1])
                    if grad_t > GRADIENT_THRESHOLD or grad_z > GRADIENT_THRESHOLD:
                        candidate_poles.append(t + 1j * z)
                
                # 3. Magnitude jump detection (sudden increase)
                if i > 0 and j > 0:
                    mag_jump = mag / (magnitude_map[i-1, j] + 1e-10)
                    if mag_jump > MAGNITUDE_THRESHOLD:
                        candidate_poles.append(t + 1j * z)
                
                # 4. Check neighbors for isolated peaks (pole signature)
                if i > 0 and i < len(t_vals)-1 and j > 0 and j < len(z_vals)-1:
                    neighbors = [
                        magnitude_map[i-1, j], magnitude_map[i+1, j],
                        magnitude_map[i, j-1], magnitude_map[i, j+1]
                    ]
                    avg_neighbor = np.mean(neighbors)
                    if mag > avg_neighbor * 10:  # Isolated peak
                        candidate_poles.append(t + 1j * z)
                        
            except (ZeroDivisionError, ValueError, OverflowError, RuntimeWarning):
                # Singularity detected
                poles.append(t + 1j * z)
                candidate_poles.append(t + 1j * z)
    
    # Refine pole locations
    refined_poles = []
    for pole_candidate in candidate_poles[:10]:  # Limit to first 10 for performance
        # Use optimization to refine
        try:
            def objective(params):
                t, z = params[0], params[1]
                psi = PsiField(
                    x=base_psi.x,
                    t=t,
                    z=z,
                    coherence=base_psi.coherence,
                    chaos=base_psi.chaos,
                    stress=base_psi.stress
                )
                result = R(psi)
                # Minimize 1/|result| to find pole
                return 1.0 / (np.abs(result.complex_time) + 1e-10)
            
            # Refine using optimization
            result_opt = minimize(
                objective,
                [pole_candidate.real, pole_candidate.imag],
                method='BFGS',
                options={'maxiter': 10}
            )
            
            if result_opt.success:
                refined_poles.append(complex(result_opt.x[0], result_opt.x[1]))
            else:
                refined_poles.append(pole_candidate)
        except:
            refined_poles.append(pole_candidate)
    
    # Compute residues using contour integration
    residues = []
    orders = []
    
    for pole in refined_poles:
        # Small circle around pole
        radius = 0.1
        n_points = 20
        theta = np.linspace(0, 2*np.pi, n_points)
        contour = pole + radius * np.exp(1j * theta)
        
        # Integrate around pole
        integral = 0.0
        for z_pt in contour:
            t, z = z_pt.real, z_pt.imag
            psi = PsiField(
                x=base_psi.x,
                t=t,
                z=z,
                coherence=base_psi.coherence,
                chaos=base_psi.chaos,
                stress=base_psi.stress
            )
            try:
                result = R(psi)
                # Compute 1/(z - pole) * f(z) dz
                dz = 2j * np.pi * radius / n_points
                integral += result.complex_time * dz / (z_pt - pole)
            except:
                pass
        
        residues.append(integral / (2j * np.pi))
        
        # Estimate pole order from residue magnitude
        if np.abs(integral) > 1e-3:
            orders.append(1)  # Simple pole
        else:
            orders.append(2)  # Higher order
    
    # Classify poles by location
    pole_locations = {
        "real_axis": [p for p in refined_poles if np.abs(p.imag) < 0.1],
        "imaginary_axis": [p for p in refined_poles if np.abs(p.real) < 0.1],
        "complex_plane": [p for p in refined_poles if np.abs(p.imag) >= 0.1 and np.abs(p.real) >= 0.1]
    }
    
    print(f"\n   Poles found: {len(refined_poles)}")
    print(f"   Residues computed: {len(residues)}")
    print(f"   Real axis poles: {len(pole_locations['real_axis'])}")
    print(f"   Imaginary axis poles: {len(pole_locations['imaginary_axis'])}")
    print(f"   Complex plane poles: {len(pole_locations['complex_plane'])}")
    
    return PoleResult(
        poles=refined_poles,
        residues=residues,
        orders=orders,
        pole_locations=pole_locations
    )

# ============================================================================
# Task 4: Spectral Signature
# ============================================================================

@dataclass
class SpectralResult:
    """Results from spectral analysis."""
    fixed_points: List[PsiField]
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray]
    feigenbaum_eigenvalues: List[complex]
    critical_exponents: Dict[str, float]

def spectral_signature_analysis(
    R: ComplexRenormOperator,
    n_initial_conditions: int = 10
) -> SpectralResult:
    """
    Find fixed points, linearize operator, identify eigenvalues beyond δ_F.
    
    The spectrum reveals:
    - Feigenbaum constants (δ, α) as eigenvalues
    - Critical exponents from universality classes
    - Scaling dimensions
    """
    print("\n" + "=" * 70)
    print("Task 4: Spectral Signature Analysis")
    print("=" * 70)
    
    # Find multiple fixed points from different initial conditions
    fixed_points = []
    
    print(f"\n   Finding fixed points from {n_initial_conditions} initial conditions...")
    
    for i in range(n_initial_conditions):
        # Vary initial conditions
        initial_psi = PsiField(
            x=np.linspace(-1, 1, 10),
            t=0.0,
            z=0.0,
            coherence=0.2 + 0.6 * i / n_initial_conditions,
            chaos=0.1 + 0.5 * i / n_initial_conditions,
            stress=0.1 + 0.4 * i / n_initial_conditions
        )
        
        found_fps = R.find_fixed_points(initial_psi, max_iter=50, tol=1e-6)
        
        for fixed in found_fps:
            # Check if this fixed point is new
            is_new = True
            for existing_fp in fixed_points:
                dist = (np.abs(fixed.coherence - existing_fp.coherence) +
                       np.abs(fixed.chaos - existing_fp.chaos))
                if dist < 0.1:
                    is_new = False
                    break
            
            if is_new:
                fixed_points.append(fixed)
    
    print(f"   Fixed points found: {len(fixed_points)}")
    
    # Linearize operator at each fixed point
    all_eigenvalues = []
    all_eigenvectors = []
    feigenbaum_eigenvals = []
    
    # If no fixed points found, try with more initial conditions and better convergence
    if len(fixed_points) == 0:
        # Try with different initial conditions more systematically
        for coherence_init in [0.3, 0.5, 0.7]:
            for chaos_init in [0.2, 0.4, 0.6]:
                initial_psi = PsiField(
                    x=np.linspace(-1, 1, 10),
                    t=0.0,
                    z=0.0,
                    coherence=coherence_init,
                    chaos=chaos_init,
                    stress=0.2
                )
                found = R.find_fixed_points(initial_psi, max_iter=100, tol=1e-4)
                fixed_points.extend(found)
                if len(fixed_points) >= 3:  # Found enough
                    break
            if len(fixed_points) >= 3:
                break
    
    for fp in fixed_points:
        # Compute Jacobian (linearization)
        jacobian = np.zeros((4, 4), dtype=complex)
        eps = 1e-6
        
        base_result = R(fp)
        base_state = np.array([
            base_result.coherence,
            base_result.chaos,
            base_result.stress,
            base_result.z
        ], dtype=complex)
        
        for i in range(4):
            # Perturb
            perturbed = PsiField(
                x=fp.x,
                t=fp.t + (eps if i == 0 else 0),
                z=fp.z + (eps if i == 3 else 0),
                coherence=fp.coherence + (eps if i == 1 else 0),
                chaos=fp.chaos + (eps if i == 2 else 0),
                stress=fp.stress
            )
            
            result = R(perturbed)
            result_state = np.array([
                result.coherence,
                result.chaos,
                result.stress,
                result.z
            ], dtype=complex)
            
            jacobian[:, i] = (result_state - base_state) / eps
        
        # Compute eigenvalues
        eigenvals, eigenvecs = eig(jacobian)
        all_eigenvalues.extend(eigenvals)
        all_eigenvectors.append(eigenvecs)
        
        # Check for Feigenbaum constants
        for ev in eigenvals:
            if np.abs(np.abs(ev) - DELTA_F) < 0.5:
                feigenbaum_eigenvals.append(ev)
            if np.abs(np.abs(ev) - np.abs(ALPHA_F)) < 0.5:
                feigenbaum_eigenvals.append(ev)
    
    eigenvalues = np.array(all_eigenvalues)
    
    # Extract critical exponents
    # Critical exponents are related to eigenvalues: ν = 1/λ, etc.
    critical_exponents = {
        "nu": 1.0 / np.abs(eigenvalues[np.argmax(np.abs(eigenvalues))]) if len(eigenvalues) > 0 else 0.0,
        "eta": np.min(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0.0,
        "delta_feigenbaum": DELTA_F,
        "alpha_feigenbaum": np.abs(ALPHA_F)
    }
    
    print(f"\n   Eigenvalues computed: {len(eigenvalues)}")
    print(f"   Feigenbaum eigenvalues found: {len(feigenbaum_eigenvals)}")
    print(f"   Critical exponent ν: {critical_exponents['nu']:.6f}")
    print(f"   Critical exponent η: {critical_exponents['eta']:.6f}")
    
    # Identify eigenvalues beyond δ_F
    beyond_delta = [ev for ev in eigenvalues if np.abs(ev) > DELTA_F * 1.1]
    print(f"   Eigenvalues beyond δ_F: {len(beyond_delta)}")
    
    return SpectralResult(
        fixed_points=fixed_points,
        eigenvalues=eigenvalues,
        eigenvectors=all_eigenvectors[0] if all_eigenvectors else None,
        feigenbaum_eigenvalues=feigenbaum_eigenvals,
        critical_exponents=critical_exponents
    )

# ============================================================================
# Main Analysis Runner
# ============================================================================

def run_all_analysis():
    """
    Run all four computational tasks.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE OPERATOR ANALYSIS")
    print("Generalized Complex Renormalization Operator ℛ")
    print("=" * 70)
    
    # Create operator
    def base_map(x):
        return 4.0 * x * (1.0 - x)  # Logistic map
    
    rg_op = RGOperator(base=base_map, alpha=1.0, depth=0)
    R = ComplexRenormOperator(rg_operator=rg_op, coupling=1.0 + 0.1j)
    
    # Task 1: Known Limits
    verification = known_limits_verification()
    
    # Task 2: Branch Cut Mapping
    branch_cuts = map_branch_cuts(R, resolution=30)
    
    # Task 3: Pole Structure
    poles = analyze_pole_structure(R, resolution=50)
    
    # Task 4: Spectral Signature
    spectrum = spectral_signature_analysis(R, n_initial_conditions=5)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"✓ Known limits verified: {verification.all_passed}")
    print(f"✓ Branch cuts mapped: {len(branch_cuts.branch_cuts)}")
    print(f"✓ Poles located: {len(poles.poles)}")
    print(f"✓ Fixed points found: {len(spectrum.fixed_points)}")
    print(f"✓ Eigenvalues computed: {len(spectrum.eigenvalues)}")
    print("=" * 70)
    
    return {
        "verification": verification,
        "branch_cuts": branch_cuts,
        "poles": poles,
        "spectrum": spectrum
    }

if __name__ == "__main__":
    run_all_analysis()

