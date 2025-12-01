#!.venv/bin/python
"""
demo.py

ZetaDiffusion Lab - Deterministic analysis from SeedSpec.
Usage: python demo.py [seed]                    # Run analysis with seed
       python demo.py --regenerate seedspec.json # Regenerate from SeedSpec
"""

import sys
import numpy as np
from zetadiffusion.field import XiSampler, critical_line_field
from zetadiffusion.dynamics import CircleMapExtractor
from zetadiffusion.renorm import find_bifurcations, local_scan, ALPHA_F
from zetadiffusion.energy import ManifoldState, TopologicalHarvester
from zetadiffusion.windowspec import SeedSpec, expand_seed, regenerate_from_spec

def ascii_plot(y_values, height=10, width=60, title="Plot"):
    """Simple ASCII plot."""
    y = np.array(y_values)
    if len(y) == 0:
        print("No data.")
        return
        
    valid_mask = np.isfinite(y)
    if not np.any(valid_mask):
        print("No valid data.")
        return
        
    min_y, max_y = np.min(y[valid_mask]), np.max(y[valid_mask])
    if np.isclose(max_y, min_y):
        normalized = np.full_like(y, height // 2, dtype=int)
    else:
        normalized = np.zeros_like(y, dtype=int)
        normalized[valid_mask] = ((y[valid_mask] - min_y) / (max_y - min_y) * (height - 1)).astype(int)
    
    print(f"\n--- {title} ---")
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    indices = np.linspace(0, len(y) - 1, width).astype(int)
    for col, idx in enumerate(indices):
        if not valid_mask[idx]: continue
        row = height - 1 - normalized[idx]
        grid[row][col] = '*'
        
    for row in grid:
        print("".join(row))
    print(f"Range: [{min_y:.4f}, {max_y:.4f}]")

def run_lab(seed: float = 1.0, silent: bool = False):
    """
    Run ZetaDiffusion analysis deterministically from seed.
    Returns analysis results that can be regenerated from SeedSpec.
    """
    # Generate t_grid and xi_values deterministically
    t_start, t_end = 0.0, 60.0
    n_points = 1000
    
    sampler = XiSampler(t_start, t_end, n_points)
    sampler.sample()
    t_grid = sampler.t_grid
    xi_values = sampler.xi_values
    
    # Derive center and width from seed
    t_range = t_end - t_start
    t_center = t_start + (seed % 1.0) * t_range
    width = t_range * 0.1 * (1.0 + (seed % 1.0))
    
    # Create SeedSpec
    spec = SeedSpec(center=t_center, seed=seed)
    rules = expand_seed(seed, t_range)
    
    if not silent:
        print(f"Initializing ZetaDiffusion Lab (seed={seed:.6f})...")
        print(f"\nSeedSpec: center={t_center:.2f}, seed={seed:.6f}")
        print(f"Expanded rules: {rules}")
    
    # --- I. Spectral Line Probe ---
    if not silent:
        print("\nI. Spectral Line Probe (Xi Sampler)")
    window_sampler = XiSampler(t_grid[0], t_grid[-1], n_points=len(t_grid))
    window_sampler.t_grid = t_grid
    window_sampler.xi_values = xi_values
    zeros = window_sampler.detect_zeros()
    
    if not silent:
        ascii_plot(xi_values, title=f"Xi(0.5 + it) [{t_grid[0]:.1f}, {t_grid[-1]:.1f}]")
        print(f"Detected {len(zeros)} candidate zeros.")
        if zeros:
            print(f"First few zeros: {[z[0] for z in zeros[:5]]}")
    
    # --- II. Bundle Dynamics ---
    if not silent:
        print(f"\nII. Bundle Dynamics (Circle Map Extraction at t={t_center:.2f})")
    extractor = CircleMapExtractor(t_grid, xi_values, t_center, width)
    k_coupling = 0.5 + (seed % 1.0) * 1.5
    omegas, rhos = extractor.scan_devil_staircase(n_omegas=60, k=k_coupling)
    if not silent:
        ascii_plot(rhos, title="Devil's Staircase (Rotation Number vs Omega)")
    
    # --- III. Local RG Operator ---
    if not silent:
        print(f"\nIII. Local RG Operator (Feigenbaum Scanner at t={t_center:.2f})")
    scan_width = t_range * 0.3
    t_min = max(t_grid[0], t_center - scan_width/2)
    t_max = min(t_grid[-1], t_center + scan_width/2)
    
    results = find_bifurcations(t_grid, xi_values, t_min, t_max, width=width)
    mean_alpha = np.nan
    
    if not results:
        alphas = local_scan(t_grid, xi_values, t_center, width)
        if alphas:
            mean_alpha = alphas[-1]
            if not silent:
                print(f"Local alpha estimate: {mean_alpha:.4f}")
                print(f"Target Universal Alpha: {ALPHA_F:.4f}")
    else:
        peaks, alphas = zip(*results)
        mean_alpha = np.nanmean(alphas)
        if not silent:
            ascii_plot(alphas, title="Local RG Scaling Factor (Alpha estimate)")
            print(f"Target Universal Alpha: {ALPHA_F:.4f}")
            print(f"Mean estimated alpha: {mean_alpha:.4f}")
    
    # --- IV. Topological Energy Harvesting ---
    if not silent:
        print("\nIV. Topological Energy Harvesting (Negentropic Engine)")
        print("Simulating loading/shock cycle...")
    
    grid_size = int(50 + (seed * 100) % 100)
    n_steps = int(30 + (seed * 50) % 50)
    efficiency = 0.7 + (seed % 1.0) * 0.2
    
    state = ManifoldState(curvature_grid=np.zeros(grid_size), viscosity_gamma=0.5)
    harvester = TopologicalHarvester(efficiency=efficiency)
    
    works = []
    num_shocks = 0
    
    np.random.seed(int(seed * 1000) % (2**31))
    for t in range(n_steps):
        injection_scale = 0.05 + (seed * 0.1) % 0.1
        injection = np.random.normal(injection_scale, 0.05, grid_size) * (t % 15)
        state.curvature_grid += injection
        mach = np.sqrt(state.potential_energy) / 5.0
        
        report = harvester.process_event(state, mach)
        works.append(harvester.accumulated_work)
        if report.get('status') == 'SHOCK_HARVESTED':
            num_shocks += 1
            if not silent:
                print(f"T={t:2d} | SHOCK! | Energy: {report.get('energy_released', 0):.2f} | Work: {report.get('work_captured', 0):.2f}")
    
    if not silent:
        ascii_plot(works, title="Accumulated Harvested Work (Insight)")
    
    # --- V. Guardian Nash Policy (Game-Theoretic Stability) ---
    if not silent:
        print("\nV. Guardian Nash Policy (Game-Theoretic Stability Control)")
    
    from zetadiffusion.guardian import (
        SystemState, GuardianResponse, guardian_nash_policy,
        calculate_fixed_points, calculate_guardian_utility
    )
    
    # Derive system state from analysis
    # Coherence from accumulated work (normalized)
    coherence = min(1.0, harvester.accumulated_work / 1000.0) if works else 0.5
    # Chaos from mean alpha deviation
    chaos = abs(mean_alpha - ALPHA_F) / abs(ALPHA_F) if not np.isnan(mean_alpha) else 0.1
    # Stress from Gini-like metric (inequality in curvature)
    stress = np.std(state.curvature_grid) / (np.mean(np.abs(state.curvature_grid)) + 1e-10)
    stress = min(1.0, stress)  # Clamp to [0, 1]
    # Hurst from rotation number persistence
    hurst = np.mean(rhos) if len(rhos) > 0 else 0.5
    
    guardian_state = SystemState(
        coherence=coherence,
        chaos=chaos,
        stress=stress,
        hurst=hurst,
        gamma=1.0,
        delta=0.5
    )
    
    response = guardian_nash_policy(guardian_state)
    c_star, lambda_star = calculate_fixed_points(response.coupling, guardian_state.gamma, guardian_state.delta)
    utility = calculate_guardian_utility(guardian_state, response.coupling)
    
    if not silent:
        print(f"System State:")
        print(f"  Coherence C = {coherence:.4f}")
        print(f"  Chaos λ = {chaos:.4f}")
        print(f"  Stress G = {stress:.4f}")
        print(f"  Hurst H = {hurst:.4f}")
        print(f"\nGuardian Response:")
        print(f"  Status: {response.status}")
        print(f"  Coupling β = {response.coupling:.6f}")
        print(f"  Threshold G_crit(H) = {response.threshold:.4f}")
        print(f"  Optimal β_res = {response.beta_res:.6f}")
        print(f"  Fixed Point: C* = {c_star:.4f}, λ* = {lambda_star:.4f}")
        print(f"  Guardian Utility U_G = {utility:.4f}")
    
    return {
        'zeros': zeros,
        'omegas': omegas,
        'rhos': rhos,
        'alphas': alphas if 'alphas' in locals() else [],
        'mean_alpha': mean_alpha,
        'final_work': works[-1] if works else 0.0,
        'num_shocks': num_shocks,
        't_center': t_center,
        'seed': seed,
        'spec': spec,
        'guardian_response': response,
        'guardian_state': guardian_state
    }

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regenerate":
        # Regenerate from SeedSpec
        if len(sys.argv) < 3:
            print("Usage: python demo.py --regenerate seedspec.json")
            sys.exit(1)
        
        spec = SeedSpec.from_file(sys.argv[2])
        print(f"Regenerating from SeedSpec: center={spec.center:.2f}, seed={spec.seed:.6f}")
        
        results = regenerate_from_spec(spec)
        
        print(f"\nRegenerated Analysis:")
        print(f"  Zeros found: {len(results['zeros'])}")
        print(f"  RG alphas: {len(results['alphas'])} values")
        print(f"  Rotation numbers: {len(results['rhos'])} values")
        print(f"  Window size: {len(results['t_grid'])} points")
        
        ascii_plot(results['xi_values'], title="Regenerated Xi(0.5 + it)")
        if len(results['alphas']) > 0:
            ascii_plot(results['alphas'], title="Regenerated RG Alphas")
        if len(results['rhos']) > 0:
            ascii_plot(results['rhos'], title="Regenerated Devil's Staircase")
    elif len(sys.argv) > 1 and sys.argv[1] == "--textgen":
        # Fractal text generation using Guardian Nash Policy
        from zetadiffusion.textgen import FractalTextGenerator
        
        seed_text = sys.argv[2] if len(sys.argv) > 2 else "The"
        n_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        
        print(f"Fractal Text Generation (Guardian Nash Policy)")
        print(f"Seed: '{seed_text}'")
        print(f"Generating {n_chars} characters...")
        print()
        
        generator = FractalTextGenerator(seed_text)
        generated = generator.generate(n_chars)
        
        print("Generated Text:")
        print("=" * 60)
        print(generated)
        print("=" * 60)
        
        stats = generator.get_statistics()
        print(f"\nGeneration Statistics:")
        print(f"  Total characters: {stats['total_chars']}")
        print(f"  Final coherence: {stats['final_coherence']:.4f}")
        print(f"  Final chaos: {stats['final_chaos']:.4f}")
        print(f"  Final stress: {stats['final_stress']:.4f}")
        print(f"  Final Hurst: {stats['final_hurst']:.4f}")
        print(f"  Resonance steps: {stats['resonance_steps']}")
        print(f"  Shielding steps: {stats['shielding_steps']}")
        print(f"  Mean coupling β: {stats['mean_coupling']:.4f}")
        
        # Show Guardian decisions over time
        if len(generator.guardian_history) > 0:
            print(f"\nGuardian Decision History (last 10 steps):")
            for i, h in enumerate(generator.guardian_history[-10:]):
                status_short = "RES" if "RESONANCE" in h['status'] else "SHLD"
                print(f"  Step {len(generator.guardian_history)-10+i}: {status_short} | "
                      f"C={h['coherence']:.3f} λ={h['chaos']:.3f} G={h['stress']:.3f} "
                      f"H={h['hurst']:.3f} β={h['coupling']:.3f}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--compress":
        # Compression mode: compress text to SeedSpec
        from zetadiffusion.compress import compress_text, decompress_text
        
        if len(sys.argv) < 3:
            print("Usage: python demo.py --compress <text>")
            print("   or: python demo.py --compress --file <filename>")
            sys.exit(1)
        
        if len(sys.argv) > 2 and sys.argv[2] == "--file":
            # Compress file
            filename = sys.argv[3] if len(sys.argv) > 3 else "demo.py"
            with open(filename, 'r') as f:
                text = f.read()
            print(f"Compressing file: {filename}")
        else:
            # Compress provided text
            text = " ".join(sys.argv[2:])
            print(f"Compressing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        print(f"Original size: {len(text)} characters ({len(text.encode('utf-8'))} bytes)")
        print()
        
        # Compress
        result = compress_text(text)
        spec = result['spec']
        state = result['state']
        
        print(f"Compressed to SeedSpec:")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print(f"  Compression: {result['original_size']} → {result['compressed_size']} bytes")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print()
        print(f"System State:")
        print(f"  Coherence C = {state.coherence:.4f}")
        print(f"  Chaos λ = {state.chaos:.4f}")
        print(f"  Stress G = {state.stress:.4f}")
        print(f"  Hurst H = {state.hurst:.4f}")
        print()
        
        # Save SeedSpec
        spec_file = f"compressed_{spec.seed:.6f}.json"
        spec.to_file(spec_file)
        print(f"SeedSpec saved to: .out/{spec_file}")
        print()
        
        # Decompress (lossy)
        print("Decompressing (lossy approximation)...")
        decompressed = decompress_text(spec, target_length=len(text))
        
        print("\nOriginal text:")
        print("=" * 60)
        print(text[:200] + ("..." if len(text) > 200 else ""))
        print("=" * 60)
        
        print("\nDecompressed text:")
        print("=" * 60)
        print(decompressed[:200] + ("..." if len(decompressed) > 200 else ""))
        print("=" * 60)
        
        # Calculate similarity
        min_len = min(len(text), len(decompressed))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if text[i] == decompressed[i])
            similarity = (matches / min_len) * 100
            print(f"\nSimilarity: {similarity:.1f}% ({matches}/{min_len} characters match)")
            print(f"  Note: This is lossy compression - approximation expected")
    elif len(sys.argv) > 1 and sys.argv[1] == "--fixed-point":
        # Fixed point compression: find (seed, center) where R(seed, center) = text exactly
        from zetadiffusion.compress import compress_text_fixed_point, regenerate_text_from_state
        
        if len(sys.argv) < 3:
            print("Usage: python demo.py --fixed-point <text>")
            print("   or: python demo.py --fixed-point --file <filename>")
            sys.exit(1)
        
        if len(sys.argv) > 2 and sys.argv[2] == "--file":
            filename = sys.argv[3] if len(sys.argv) > 3 else "demo.py"
            with open(filename, 'r') as f:
                text = f.read()
            print(f"Finding fixed point for file: {filename}")
        else:
            text = " ".join(sys.argv[2:])
            print(f"Finding fixed point for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        print(f"Original size: {len(text)} characters ({len(text.encode('utf-8'))} bytes)")
        print()
        print("Searching for fixed point (seed, center) where R(seed, center) = text...")
        print("This may take a while for longer texts...")
        print()
        
        # Find fixed point
        result = compress_text_fixed_point(text, max_iterations=10000)
        spec = result['spec']
        state = result['state']
        found = result.get('fixed_point_found', False)
        
        print(f"Fixed point search complete:")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print(f"  Compression: {result['original_size']} → {result['compressed_size']} bytes")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print()
        print(f"System State:")
        print(f"  Coherence C = {state.coherence:.4f}")
        print(f"  Chaos λ = {state.chaos:.4f}")
        print(f"  Stress G = {state.stress:.4f}")
        print(f"  Hurst H = {state.hurst:.4f}")
        print()
        
        # Verify fixed point
        regenerated = regenerate_text_from_state(spec.center, spec.seed, len(text))
        exact_match = (regenerated == text)
        
        if exact_match:
            print("✓ Fixed point found! Exact regeneration verified.")
        else:
            print("⚠ Fixed point not found (exact match). Best approximation:")
            min_len = min(len(text), len(regenerated))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if text[i] == regenerated[i])
                similarity = (matches / min_len) * 100
                print(f"  Similarity: {similarity:.1f}% ({matches}/{min_len} characters match)")
        
        # Save SeedSpec
        spec_file = f"fixedpoint_{spec.seed:.6f}.json"
        spec.to_file(spec_file)
        print(f"\nSeedSpec saved to: .out/{spec_file}")
        print(f"  To regenerate: python demo.py --regenerate {spec_file}")
        
        # Show comparison
        print("\nOriginal text:")
        print("=" * 60)
        print(text[:200] + ("..." if len(text) > 200 else ""))
        print("=" * 60)
        
        print("\nRegenerated text:")
        print("=" * 60)
        print(regenerated[:200] + ("..." if len(regenerated) > 200 else ""))
        print("=" * 60)
    else:
        # Normal run mode
        seed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
        
        result = run_lab(seed)
        
        # Save SeedSpec
        spec = result['spec']
        spec_file = f"seedspec_{seed:.6f}.json"
        spec.to_file(spec_file)
        
        rules = expand_seed(seed, 60.0)
        print(f"\nSeedSpec saved to .out/{spec_file}")
        print(f"  center={spec.center:.2f}, seed={spec.seed:.6f}")
        print(f"  Expanded rules: {rules}")
        print(f"  To regenerate: python demo.py --regenerate {spec_file}")
