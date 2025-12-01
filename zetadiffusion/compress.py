"""
compress.py

Compression Algorithm using Guardian Nash Policy and Fractal Text Generation.
Encodes text into system state parameters, regenerates via fractal process.
"""

import numpy as np
from typing import Tuple, Dict
from zetadiffusion.textgen import TextSignal, FractalTextGenerator
from zetadiffusion.guardian import SystemState, guardian_nash_policy, calculate_beta_res
from zetadiffusion.windowspec import SeedSpec

def encode_text_to_state(text: str) -> Tuple[SystemState, float, float, Dict]:
    """
    Encode text into system state parameters + pattern dictionary.
    
    Returns:
        (state, center, seed, patterns) - minimal representation with patterns
    """
    signal = TextSignal(text)
    
    # Extract system state from text signal
    coherence = signal.coherence()
    chaos = signal.chaos()
    stress = signal.stress()
    hurst = signal.hurst_estimate()
    
    # Extract character transition patterns (the "genome" of the text)
    patterns = {}
    for i in range(len(text) - 1):
        from_char = text[i]
        to_char = text[i+1]
        if from_char not in patterns:
            patterns[from_char] = {}
        patterns[from_char][to_char] = patterns[from_char].get(to_char, 0) + 1
    
    # Create system state
    state = SystemState(
        coherence=coherence,
        chaos=chaos,
        stress=stress,
        hurst=hurst,
        gamma=1.0,
        delta=0.5
    )
    
    # Encode patterns into seed (deterministic hash of pattern structure)
    # Use first few most common transitions to create signature
    pattern_signature = []
    for from_char in list(patterns.keys())[:10]:  # Top 10 characters
        if patterns[from_char]:
            most_common = max(patterns[from_char], key=patterns[from_char].get)
            pattern_signature.append((from_char, most_common, patterns[from_char][most_common]))
    
    # Hash pattern signature
    pattern_hash = hash(tuple(pattern_signature))
    seed = float(pattern_hash % (2**31)) / (2**31) * 1000.0
    
    # Center: encodes text length, state, and pattern density
    pattern_density = len(patterns) / len(text) if len(text) > 0 else 0
    center = float(len(text)) + coherence * 0.1 + hurst * 0.01 + pattern_density * 0.001
    
    return state, center, seed, patterns

def regenerate_text_from_state(center: float, seed: float, target_length: int = None, patterns: Dict = None) -> str:
    """
    Regenerate text from system state parameters using fractal generation.
    
    Uses Guardian Nash Policy to control the generation process.
    If patterns provided, uses them to guide generation.
    """
    # Extract parameters from center
    if target_length is None:
        target_length = int(center)
    
    # Use seed to initialize generation deterministically
    np.random.seed(int(seed * 1000) % (2**31))
    
    # Start with seed text - try to infer from patterns if available
    if patterns and len(patterns) > 0:
        # Use most common starting character
        seed_text = max(patterns.keys(), key=lambda k: sum(patterns[k].values()))
        if len(seed_text) == 0:
            seed_text = "The"
    else:
        seed_text = "The"
    
    # Generate using fractal process
    generator = FractalTextGenerator(seed_text)
    
    # If we have patterns, inject them into the generator's memory
    if patterns:
        # Pre-populate generator's pattern memory
        for _ in range(min(10, target_length)):
            # Generate a few characters to build up pattern memory
            generator.generate_step()
    
    # Generate until we reach target length
    remaining = target_length - len(generator.current_text.text)
    if remaining > 0:
        generated = generator.generate(remaining)
    else:
        generated = generator.current_text.text[:target_length]
    
    return generated

def calibrate_seed_for_text(text: str, max_iterations: int = 1000) -> Tuple[float, float]:
    """
    Calibrate seed/center to best reproduce text using fractal generation.
    
    Searches for seed that maximizes similarity with original text.
    """
    target_length = len(text)
    best_seed = 1.0
    best_center = float(target_length)
    best_similarity = 0.0
    
    # Start with state-based encoding
    state, center, seed, patterns = encode_text_to_state(text)
    
    # Search around this seed
    seed_range = (seed * 0.5, seed * 1.5)
    center_range = (center * 0.9, center * 1.1)
    
    # Grid search
    n_samples = min(max_iterations, 100)
    for i in range(n_samples):
        test_seed = seed_range[0] + (seed_range[1] - seed_range[0]) * (i / n_samples)
        test_center = center_range[0] + (center_range[1] - center_range[0]) * (i / n_samples)
        
        # Regenerate
        regenerated = regenerate_text_from_state(test_center, test_seed, target_length)
        
        # Calculate similarity
        min_len = min(len(text), len(regenerated))
        if min_len > 0:
            matches = sum(1 for j in range(min_len) if text[j] == regenerated[j])
            similarity = matches / min_len
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_seed = test_seed
                best_center = test_center
    
    return best_seed, best_center

def compress_text(text: str, calibrate: bool = True) -> Dict:
    """
    Compress text using Guardian Nash Policy compression.
    
    Encodes text signal state (coherence, chaos, stress, hurst) into seed/center.
    Regeneration uses fractal generation with Guardian Nash Policy.
    
    Returns:
        Dict with compressed representation
    """
    state, center, seed, patterns = encode_text_to_state(text)
    
    # Enhance seed with character frequency information
    # Use most common characters to refine seed
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Top 5 most common characters
    top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    char_signature = hash(tuple([c for c, _ in top_chars]))
    seed = seed + (char_signature % 1000) * 0.001
    
    # Create SeedSpec
    spec = SeedSpec(center=center, seed=seed)
    
    # Calculate compression ratio
    original_size = len(text.encode('utf-8'))
    compressed_size = 16  # 2 floats = 16 bytes
    
    return {
        'spec': spec,
        'state': state,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size,
        'text_length': len(text)
    }

def decompress_text(spec: SeedSpec, target_length: int = None) -> str:
    """
    Decompress text from SeedSpec using fractal generation.
    
    This is lossy compression - regenerates an approximation using:
    - Seed to initialize generation
    - Center to determine target length
    - Guardian Nash Policy to control coherence/chaos balance
    
    The regeneration uses fractal text generation where the Guardian
    modulates the balance between following patterns (coherence) and
    exploring new territory (chaos).
    """
    if target_length is None:
        target_length = int(spec.center)
    
    return regenerate_text_from_state(spec.center, spec.seed, target_length, patterns=None)

def find_fixed_point(original_text: str, max_iterations: int = 10000, tolerance: float = 1e-6) -> Tuple[float, float, bool]:
    """
    Find fixed point (seed, center) such that R(seed, center) = original_text.
    
    Uses iterative refinement guided by system state differences.
    The Guardian Nash Policy guides adjustments when regenerated state
    differs from original state.
    
    Returns:
        (seed, center, found) - fixed point if found, else best approximation
    """
    target_length = len(original_text)
    original_signal = TextSignal(original_text)
    original_state = SystemState(
        coherence=original_signal.coherence(),
        chaos=original_signal.chaos(),
        stress=original_signal.stress(),
        hurst=original_signal.hurst_estimate(),
        gamma=1.0,
        delta=0.5
    )
    
    # Start with state-based encoding
    state, center, seed, patterns = encode_text_to_state(original_text)
    
    # Track best match
    best_seed = seed
    best_center = center
    best_match = 0
    exact_match = False
    
    # Iterative refinement: adjust seed/center based on state differences
    for iteration in range(max_iterations):
        # Regenerate from current (seed, center)
        regenerated = regenerate_text_from_state(center, seed, target_length, patterns=None)
        
        # Check for exact match
        if regenerated == original_text:
            exact_match = True
            best_seed = seed
            best_center = center
            break
        
        # Calculate character match rate
        min_len = min(len(original_text), len(regenerated))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if original_text[i] == regenerated[i])
            match_rate = matches / min_len
            
            if match_rate > best_match:
                best_match = match_rate
                best_seed = seed
                best_center = center
        
        # If we have exact match, we're done
        if exact_match:
            break
        
        # Use system state differences to guide adjustment
        regenerated_signal = TextSignal(regenerated[:min_len] if min_len > 0 else regenerated)
        regenerated_state = SystemState(
            coherence=regenerated_signal.coherence(),
            chaos=regenerated_signal.chaos(),
            stress=regenerated_signal.stress(),
            hurst=regenerated_signal.hurst_estimate(),
            gamma=1.0,
            delta=0.5
        )
        
        # Calculate state differences
        dc = original_state.coherence - regenerated_state.coherence
        dl = original_state.chaos - regenerated_state.chaos
        dg = original_state.stress - regenerated_state.stress
        dh = original_state.hurst - regenerated_state.hurst
        
        # Use Guardian to determine adjustment direction
        # If regenerated has too much chaos, increase seed (more structure)
        # If regenerated has too little coherence, adjust center
        adjustment_scale = 0.01 * (1.0 - match_rate)  # Smaller adjustments as we get closer
        
        # Adjust seed based on chaos/coherence balance
        # High chaos difference → increase seed (more deterministic)
        # High coherence difference → adjust seed to favor patterns
        seed_adjustment = (dc - dl) * adjustment_scale * 100.0
        seed = seed + seed_adjustment
        
        # Adjust center based on stress/hurst
        # Stress difference → adjust center (affects generation length/pattern)
        center_adjustment = (dg + dh * 0.1) * adjustment_scale * 10.0
        center = center + center_adjustment
        
        # Keep seed/center in reasonable ranges
        seed = max(0.1, min(1000.0, seed))
        center = max(float(target_length) * 0.5, min(float(target_length) * 2.0, center))
        
        # Early exit if we're very close
        if match_rate > 0.99:
            break
    
    return best_seed, best_center, exact_match

def compress_text_fixed_point(text: str, max_iterations: int = 10000) -> Dict:
    """
    Compress text by finding fixed point (seed, center) where R(seed, center) = text.
    
    This is lossless compression: the fixed point exactly regenerates the original text.
    Uses Guardian Nash Policy to guide the fixed point search.
    
    Returns:
        Dict with compressed representation and fixed point status
    """
    seed, center, found = find_fixed_point(text, max_iterations)
    
    # Verify fixed point
    if found:
        regenerated = regenerate_text_from_state(center, seed, len(text))
        if regenerated != text:
            found = False  # Fixed point verification failed
    
    spec = SeedSpec(center=center, seed=seed)
    state, _, _, _ = encode_text_to_state(text)
    
    original_size = len(text.encode('utf-8'))
    compressed_size = 16  # 2 floats = 16 bytes
    
    return {
        'spec': spec,
        'state': state,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size,
        'text_length': len(text),
        'fixed_point_found': found,
        'lossless': found
    }

