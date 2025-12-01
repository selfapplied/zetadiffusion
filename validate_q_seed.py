#!.venv/bin/python
"""
validate_q_seed.py

Validates <Q> CE1 seed structure and clock interactions.

Tests:
1. Seed structure and invariants
2. CE1 algebra (composition, idempotence)
3. Clock phase interactions
4. Stability detection on actual zero sequences

Author: Joel
"""

import numpy as np
from mpmath import zetazero
from zetadiffusion.ce1_seed import (
    create_q_seed, CE1Algebra, ClockInteraction, check_stability
)
from zetadiffusion.validation_framework import run_validation

def test_seed_structure() -> dict:
    """Test 1: Seed structure and basic properties."""
    q = create_q_seed()
    
    return {
        'witness': q.witness,
        'invariant': q.invariant,
        'morphism': q.morphism,
        'role': q.role,
        'idempotent': q.is_idempotent()
    }

def test_ce1_algebra() -> dict:
    """Test 2: CE1 algebra operations."""
    q = create_q_seed()
    
    # Create a dummy seed for composition test
    from zetadiffusion.ce1_seed import CE1Seed
    dummy = CE1Seed(
        witness="<X>",
        invariant="some invariant",
        morphism="some morphism",
        role="test"
    )
    
    # Composition: <Q> ⊕ X = X
    composed = CE1Algebra.compose(q, dummy)
    composition_neutral = composed.witness == dummy.witness
    
    # Idempotence: <Q> ⊕ <Q> = <Q>
    q_composed = CE1Algebra.compose(q, q)
    idempotent = q_composed.witness == q.witness
    
    # Morphism: (Δₙ) <Q> = <Q>
    morphed = CE1Algebra.apply_morphism(q, "Δₙ")
    morphism_stable = morphed.witness == q.witness
    
    return {
        'composition_neutral': composition_neutral,
        'idempotent': idempotent,
        'morphism_stable': morphism_stable
    }

def test_clock_interactions() -> dict:
    """Test 3: Clock phase interactions with <Q>."""
    test_indices = [1, 5, 7, 8, 9, 10, 11, 15, 20]
    
    results = {
        'indices': [],
        'phases': [],
        'activations': [],
        'active': []
    }
    
    for n in test_indices:
        phase = ClockInteraction.get_clock_phase(n)
        activation = ClockInteraction.q_activation(n)
        is_active = ClockInteraction.is_q_active(n)
        
        results['indices'].append(n)
        results['phases'].append(phase)
        results['activations'].append(float(activation))
        results['active'].append(is_active)
    
    return results

def test_stability_on_zeros(n_max: int = 20) -> dict:
    """Test 4: Check <Q> stability on actual Riemann zero sequences."""
    q = create_q_seed()
    
    # Get actual zeros
    zeros = [float(zetazero(n).imag) for n in range(1, n_max + 1)]
    
    # Check stability
    stability = check_stability(zeros, tolerance=1.0)  # 1.0 unit tolerance
    
    # Check by clock phase
    phase_stability = {}
    for phase in ['feigenbaum', 'boundary', 'membrane', 'interior']:
        phase_indices = []
        for n in range(1, n_max + 1):
            if ClockInteraction.get_clock_phase(n) == phase:
                phase_indices.append(n - 1)  # 0-indexed
        
        if phase_indices:
            phase_zeros = [zeros[i] for i in phase_indices]
            phase_stab = check_stability(phase_zeros, tolerance=1.0)
            phase_stability[phase] = {
                'precision': phase_stab['precision'],
                'max_drift': phase_stab['max_drift'],
                'q_active': phase_stab['q_active']
            }
    
    return {
        'overall': stability,
        'by_phase': phase_stability,
        'zeros': zeros[:10]  # First 10 for reference
    }

def verify_q_seed() -> dict:
    """
    Verify <Q> CE1 seed structure and behavior.
    """
    print("=" * 70)
    print("CE1 SEED <Q> VALIDATION")
    print("=" * 70)
    print()
    
    # Test 1: Seed structure
    print("Test 1: Seed Structure")
    print("-" * 70)
    structure = test_seed_structure()
    print(f"Witness: {structure['witness']}")
    print(f"Invariant: {structure['invariant']}")
    print(f"Morphism: {structure['morphism']}")
    print(f"Role: {structure['role']}")
    print(f"Idempotent: {structure['idempotent']}")
    print()
    
    # Test 2: CE1 Algebra
    print("Test 2: CE1 Algebra")
    print("-" * 70)
    algebra = test_ce1_algebra()
    print(f"Composition neutral (<Q> ⊕ X = X): {algebra['composition_neutral']}")
    print(f"Idempotent (<Q> ⊕ <Q> = <Q>): {algebra['idempotent']}")
    print(f"Morphism stable ((Δₙ) <Q> = <Q>): {algebra['morphism_stable']}")
    print()
    
    # Test 3: Clock interactions
    print("Test 3: Clock Phase Interactions")
    print("-" * 70)
    clock_test = test_clock_interactions()
    print(f"{'n':<6} | {'Phase':<12} | {'Activation':<12} | {'Active'}")
    print("-" * 70)
    for i, n in enumerate(clock_test['indices']):
        phase = clock_test['phases'][i]
        activation = clock_test['activations'][i]
        active = "✓" if clock_test['active'][i] else "✗"
        print(f"{n:<6} | {phase:<12} | {activation:>11.3f} | {active}")
    print()
    
    # Test 4: Stability on actual zeros
    print("Test 4: Stability on Riemann Zeros")
    print("-" * 70)
    stability = test_stability_on_zeros(n_max=20)
    print(f"Overall precision: {stability['overall']['precision']:.6f}")
    print(f"Overall max drift: {stability['overall']['max_drift']:.6f}")
    print(f"<Q> active overall: {stability['overall']['q_active']}")
    print()
    print("By clock phase:")
    for phase, metrics in stability['by_phase'].items():
        print(f"  {phase.capitalize():<12}: precision={metrics['precision']:.6f}, "
              f"max_drift={metrics['max_drift']:.6f}, active={metrics['q_active']}")
    print()
    
    # Summary
    all_tests_passed = (
        algebra['composition_neutral'] and
        algebra['idempotent'] and
        algebra['morphism_stable']
    )
    
    print("=" * 70)
    if all_tests_passed:
        print("✓ All CE1 algebra tests passed")
    else:
        print("✗ Some CE1 algebra tests failed")
    print("=" * 70)
    
    return {
        'structure': structure,
        'algebra': algebra,
        'clock_interactions': clock_test,
        'stability': stability,
        'success': all_tests_passed
    }

def main():
    """Run <Q> seed validation using shared framework."""
    def run_q_validation():
        return verify_q_seed()
    
    return run_validation(
        validation_type="CE1 Seed <Q>",
        validation_func=run_q_validation,
        parameters={'n_max': 20},
        output_filename="q_seed_results.json"
    )

if __name__ == "__main__":
    result = main()
    if not result.success:
        exit(1)

