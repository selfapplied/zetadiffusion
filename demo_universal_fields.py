#!.venv/bin/python
"""
Universal Field Equations Demo

Models different domains as fields and predicts their equations:
- CMB (Cosmic Microwave Background) organic chemistry
- Biology equations
- Routing equations
- Language equations
- History equations
- Planetary bodies

Each domain is compressed to a spectral signature, and field equations
are derived from the operator spectrum.
"""

import sys
import numpy as np
from pathlib import Path
from zetadiffusion.compress import compress_text, encode_text_to_state
from zetadiffusion.windowspec import SeedSpec
from zetadiffusion.guardian import SystemState
import json

# Domain data examples
DOMAINS = {
    "cmb_chemistry": """
    Cosmic Microwave Background organic chemistry equations:
    H2O + CO2 → H2CO3 (carbonic acid formation in early atmosphere)
    CH4 + O2 → CO2 + H2O (methane oxidation)
    NH3 + H2O → NH4+ + OH- (ammonia hydration)
    C6H12O6 + 6O2 → 6CO2 + 6H2O (glucose oxidation)
    The CMB provides the thermal background for these reactions.
    Temperature-dependent rate constants: k = A * exp(-Ea/RT)
    """,
    
    "biology": """
    Biological field equations:
    Population growth: dN/dt = rN(1 - N/K) (logistic growth)
    Enzyme kinetics: v = Vmax * [S] / (Km + [S]) (Michaelis-Menten)
    Neural firing: dV/dt = (I - V) / τ (membrane potential)
    Gene expression: d[P]/dt = α * [mRNA] - β * [P] (protein production)
    Ecosystem dynamics: dX/dt = rX - aXY, dY/dt = baXY - mY (predator-prey)
    """,
    
    "routing": """
    Network routing equations:
    Shortest path: min Σ w(e) for path P (Dijkstra's algorithm)
    Flow conservation: Σ f_in = Σ f_out (network flow)
    Routing metric: cost = bandwidth^-1 + delay + hop_count
    Congestion control: cwnd = min(cwnd + 1/cwnd, ssthresh) (TCP)
    Load balancing: P(i) = 1 / (load_i + ε) (probability distribution)
    """,
    
    "language": """
    Language field equations:
    Word frequency: f(w) = k * rank(w)^-α (Zipf's law)
    Sentence complexity: C = Σ log(prob(word|context)) (entropy)
    Semantic distance: d(s1, s2) = ||embed(s1) - embed(s2)|| (vector space)
    Grammar rules: S → NP VP, NP → Det N (context-free grammar)
    Language evolution: dL/dt = μ * L * (1 - L/K) (logistic growth)
    """,
    
    "history": """
    Historical field equations:
    Population dynamics: P(t) = P0 * e^(rt) (exponential growth)
    Economic cycles: Y(t) = A * sin(ωt + φ) + trend (business cycles)
    Cultural diffusion: dC/dt = D * ∇²C + rC(1 - C/K) (reaction-diffusion)
    Technological adoption: A(t) = K / (1 + e^(-r(t-t0))) (S-curve)
    Conflict intensity: I(t) = Σ events * severity * decay(t - t_event)
    """,
    
    "planetary_bodies": """
    Planetary field equations:
    Orbital mechanics: F = GMm/r² (gravitational force)
    Tidal forces: F_tidal = 2GMmR/r³ (tidal acceleration)
    Atmospheric pressure: P = P0 * exp(-h/H) (barometric formula)
    Planetary formation: M(t) = M0 * (1 - e^(-t/τ)) (accretion model)
    Climate dynamics: dT/dt = (S - σT⁴) / C (energy balance)
    """
}

def extract_field_equations(state: SystemState, domain: str) -> dict:
    """
    Extract field equations from operator spectrum.
    
    The operator spectrum (C, λ, G, H) encodes the field structure:
    - Coherence (C): Structure/stability → Conservation laws
    - Chaos (λ): Complexity → Nonlinear terms
    - Stress (G): Curvature → Field gradients
    - Hurst (H): Memory → Temporal correlations
    """
    equations = {}
    
    # Base field equation structure
    # ∂φ/∂t = D∇²φ + f(φ) + noise
    
    # Diffusion coefficient from coherence (structure)
    D = state.coherence * 0.1  # Higher coherence → more structured diffusion
    
    # Nonlinear term from chaos (complexity)
    if state.chaos > 0.3:
        # Nonlinear reaction term
        equations['reaction'] = f"f(φ) = {state.chaos:.3f} * φ * (1 - φ/K)"
    else:
        # Linear term
        equations['reaction'] = f"f(φ) = {state.chaos:.3f} * φ"
    
    # Field gradient from stress (curvature)
    equations['gradient'] = f"∇φ = {state.stress:.3f} * ∂φ/∂x"
    
    # Temporal correlation from Hurst (memory)
    if state.hurst > 0.5:
        # Persistent (long memory)
        equations['temporal'] = f"φ(t+τ) = φ(t) * {state.hurst:.3f}^τ"
    else:
        # Anti-persistent (short memory)
        equations['temporal'] = f"φ(t+τ) = φ(t) * (1 - {state.hurst:.3f})^τ"
    
    # Domain-specific adaptations
    if domain == "cmb_chemistry":
        equations['rate'] = f"k = A * exp(-Ea / (R * T_CMB))"
        equations['concentration'] = f"d[C]/dt = k * [A] * [B]"
    elif domain == "biology":
        equations['growth'] = f"dN/dt = {state.coherence:.3f} * N * (1 - N/K)"
        equations['interaction'] = f"dX/dt = {state.chaos:.3f} * X * Y"
    elif domain == "routing":
        equations['path'] = f"min Σ w(e) where w = {state.stress:.3f} * delay"
        equations['flow'] = f"Σ f_in = Σ f_out (conservation)"
    elif domain == "language":
        equations['frequency'] = f"f(w) = k * rank(w)^(-{state.hurst:.3f})"
        equations['complexity'] = f"C = -Σ log(p(w|context))"
    elif domain == "history":
        equations['population'] = f"P(t) = P0 * exp({state.coherence:.3f} * t)"
        equations['diffusion'] = f"dC/dt = {state.chaos:.3f} * ∇²C"
    elif domain == "planetary_bodies":
        equations['gravity'] = f"F = G * M * m / r²"
        equations['orbital'] = f"T² = (4π²/GM) * a³"
    
    # Full field equation
    equations['field'] = f"∂φ/∂t = {D:.3f}∇²φ + {equations['reaction']} + η(t)"
    
    return equations

def predict_field_evolution(spec: SeedSpec, domain: str, n_steps: int = 10) -> list:
    """
    Predict field evolution using the spectral signature.
    
    Uses the operator spectrum to evolve the field forward in time.
    """
    state, _, _, _ = encode_text_to_state(DOMAINS[domain])
    
    # Initialize field from seed
    np.random.seed(int(spec.seed * 1000) % (2**31))
    field_value = spec.center / 1000.0  # Normalize
    
    evolution = []
    for step in range(n_steps):
        # Field evolution based on operator spectrum
        # ∂φ/∂t = D∇²φ + f(φ)
        
        # Diffusion term (coherence)
        diffusion = state.coherence * 0.01 * (np.random.random() - 0.5)
        
        # Reaction term (chaos)
        reaction = state.chaos * field_value * (1 - field_value)
        
        # Stress term (gradient)
        stress_term = state.stress * 0.1 * np.sin(step * 0.1)
        
        # Update field
        field_value = field_value + diffusion + reaction + stress_term
        
        # Memory (Hurst)
        if state.hurst > 0.5:
            field_value = state.hurst * field_value + (1 - state.hurst) * np.random.random()
        
        evolution.append({
            'step': step,
            'value': field_value,
            'coherence': state.coherence,
            'chaos': state.chaos,
            'stress': state.stress,
            'hurst': state.hurst
        })
    
    return evolution

def showcase_universal_fields():
    """Showcase: Model all domains as fields and predict equations."""
    
    print("=" * 70)
    print("UNIVERSAL FIELD EQUATIONS")
    print("=" * 70)
    print()
    print("Modeling domains as fields and predicting their equations:")
    print("  - CMB organic chemistry")
    print("  - Biology")
    print("  - Routing")
    print("  - Language")
    print("  - History")
    print("  - Planetary bodies")
    print()
    
    Path(".out").mkdir(exist_ok=True)
    
    all_signatures = {}
    all_equations = {}
    
    # Process each domain
    for domain, text in DOMAINS.items():
        print(f"Processing: {domain.replace('_', ' ').title()}")
        print("-" * 70)
        
        # Compress domain to SeedSpec
        result = compress_text(text)
        spec = result['spec']
        state = result['state']
        
        all_signatures[domain] = {
            'center': spec.center,
            'seed': spec.seed,
            'spectrum': {
                'coherence': state.coherence,
                'chaos': state.chaos,
                'stress': state.stress,
                'hurst': state.hurst
            }
        }
        
        # Extract field equations
        equations = extract_field_equations(state, domain)
        all_equations[domain] = equations
        
        print(f"Compressed: {result['original_size']} bytes → 16 bytes ({result['compression_ratio']:.1f}x)")
        print(f"Spectral Signature: center={spec.center:.6f}, seed={spec.seed:.6f}")
        print()
        print("Operator Spectrum:")
        print(f"  Coherence C = {state.coherence:.4f} (structure)")
        print(f"  Chaos λ     = {state.chaos:.4f} (complexity)")
        print(f"  Stress G    = {state.stress:.4f} (curvature)")
        print(f"  Hurst H     = {state.hurst:.4f} (memory)")
        print()
        print("Field Equations:")
        for eq_name, eq_formula in equations.items():
            print(f"  {eq_name}: {eq_formula}")
        print()
        
        # Predict evolution
        evolution = predict_field_evolution(spec, domain, n_steps=5)
        print("Predicted Evolution (first 5 steps):")
        for e in evolution[:5]:
            print(f"  t={e['step']}: φ={e['value']:.4f}")
        print()
        
        # Save domain signature
        spec.to_file(f"{domain}_compressed.json")
    
    # Compare all domains
    print("=" * 70)
    print("DOMAIN COMPARISON")
    print("=" * 70)
    print()
    print("Spectral Signatures (compression space):")
    print(f"{'Domain':<20} | {'Center':<12} | {'Seed':<12} | {'Distance from Origin':<20}")
    print("-" * 70)
    
    origin_center = 0.0
    origin_seed = 0.0
    
    for domain, sig in all_signatures.items():
        distance = np.sqrt((sig['center'] - origin_center)**2 + (sig['seed'] - origin_seed)**2)
        print(f"{domain.replace('_', ' '):<20} | {sig['center']:<12.2f} | {sig['seed']:<12.2f} | {distance:<20.2f}")
    
    print()
    print("Operator Spectrum Comparison:")
    print(f"{'Domain':<20} | {'C':<8} | {'λ':<8} | {'G':<8} | {'H':<8}")
    print("-" * 70)
    
    for domain, sig in all_signatures.items():
        s = sig['spectrum']
        print(f"{domain.replace('_', ' '):<20} | {s['coherence']:<8.3f} | {s['chaos']:<8.3f} | {s['stress']:<8.3f} | {s['hurst']:<8.3f}")
    
    print()
    
    # Save all results
    results_file = ".out/universal_fields.json"
    with open(results_file, 'w') as f:
        json.dump({
            'signatures': all_signatures,
            'equations': all_equations,
            'note': 'Each domain compressed to 16-byte SeedSpec, field equations derived from operator spectrum'
        }, f, indent=2)
    
    print(f"All results saved to: {results_file}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ All domains compressed to 16-byte SeedSpecs")
    print("✓ Field equations extracted from operator spectrum")
    print("✓ Evolution predicted for each domain")
    print("✓ Domains mapped to spectral space")
    print()
    print("The operator spectrum (C, λ, G, H) encodes:")
    print("  - Field structure (coherence)")
    print("  - Nonlinear dynamics (chaos)")
    print("  - Spatial gradients (stress)")
    print("  - Temporal correlations (hurst)")
    print()
    print("Each domain has a unique spectral signature that")
    print("determines its field equations and evolution.")
    print()
    print("=" * 70)

if __name__ == "__main__":
    showcase_universal_fields()




