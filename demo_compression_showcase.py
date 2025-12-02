#!.venv/bin/python
"""
Compression Showcase - The Coolest Demo

Demonstrates the power of SeedSpec compression by:
1. Compressing the FEG-0.4 Field Manual (the system's own documentation)
2. Compressing demo.py (the system compressing itself!)
3. Compressing famous texts and showing their spectral signatures
4. Visualizing the compression space
"""

import sys
import os
from pathlib import Path
from zetadiffusion.compress import compress_text, encode_text_to_state
from zetadiffusion.windowspec import SeedSpec
from zetadiffusion.guardian import SystemState

def showcase_compression():
    """The coolest compression demo."""
    
    print("=" * 70)
    print("ZETADIFFUSION COMPRESSION SHOWCASE")
    print("=" * 70)
    print()
    
    # Ensure .out directory exists
    Path(".out").mkdir(exist_ok=True)
    
    # Demo 1: Compress the FEG-0.4 Field Manual (meta!)
    print("Demo 1: Compressing the System's Own Documentation")
    print("-" * 70)
    
    if Path("FEG-0.4_Field_Manual.md").exists():
        with open("FEG-0.4_Field_Manual.md", 'r') as f:
            manual_text = f.read()
        
        result = compress_text(manual_text)
        spec = result['spec']
        state = result['state']
        
        print(f"Text: FEG-0.4 Field Manual")
        print(f"  Size: {result['original_size']:,} bytes")
        print(f"  Compressed: {result['compressed_size']} bytes (SeedSpec)")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print()
        print(f"Spectral Signature:")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print()
        print(f"Operator Spectrum:")
        print(f"  Coherence C = {state.coherence:.6f}")
        print(f"  Chaos λ     = {state.chaos:.6f}")
        print(f"  Stress G    = {state.stress:.6f}")
        print(f"  Hurst H     = {state.hurst:.6f}")
        print()
        
        # Save
        spec_file = "manual_compressed.json"
        spec.to_file(spec_file)
        print(f"Saved to: .out/{spec_file}")
        print()
        
        manual_signature = (spec.center, spec.seed)
    else:
        print("FEG-0.4_Field_Manual.md not found, skipping...")
        print()
        manual_signature = None
    
    # Demo 2: Compress demo.py (self-compression!)
    print("Demo 2: Compressing the System Itself")
    print("-" * 70)
    
    if Path("demo.py").exists():
        with open("demo.py", 'r') as f:
            demo_text = f.read()
        
        result = compress_text(demo_text)
        spec = result['spec']
        state = result['state']
        
        print(f"Text: demo.py (the system compressing itself!)")
        print(f"  Size: {result['original_size']:,} bytes")
        print(f"  Compressed: {result['compressed_size']} bytes")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print()
        print(f"Spectral Signature:")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print()
        
        spec_file = "demo_compressed.json"
        spec.to_file(spec_file)
        print(f"Saved to: .out/{spec_file}")
        print()
        
        demo_signature = (spec.center, spec.seed)
    else:
        demo_signature = None
    
    # Demo 3: Famous texts and their signatures
    print("Demo 3: Spectral Signatures of Famous Texts")
    print("-" * 70)
    
    famous_texts = [
        ("Shakespeare", "To be or not to be, that is the question."),
        ("Einstein", "Imagination is more important than knowledge."),
        ("Turing", "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human."),
        ("Feynman", "I think I can safely say that nobody understands quantum mechanics."),
        ("Shannon", "Information is the resolution of uncertainty."),
    ]
    
    signatures = []
    for name, text in famous_texts:
        result = compress_text(text)
        spec = result['spec']
        state = result['state']
        signatures.append((name, spec.center, spec.seed, state))
        
        print(f"{name:12} | center={spec.center:10.6f} | seed={spec.seed:10.6f} | C={state.coherence:.4f} λ={state.chaos:.4f}")
    
    print()
    
    # Demo 4: Compression space visualization
    print("Demo 4: Compression Space")
    print("-" * 70)
    print("All texts map to points in (center, seed) space.")
    print("Each point is a unique spectral signature.")
    print()
    
    if manual_signature and demo_signature:
        print("Distance between Manual and Demo:")
        center_diff = abs(manual_signature[0] - demo_signature[0])
        seed_diff = abs(manual_signature[1] - demo_signature[1])
        distance = (center_diff**2 + seed_diff**2)**0.5
        print(f"  Δcenter = {center_diff:.6f}")
        print(f"  Δseed   = {seed_diff:.6f}")
        print(f"  Distance = {distance:.6f}")
        print()
    
    # Demo 5: The recursive compression (compress the compressor!)
    print("Demo 5: Recursive Compression")
    print("-" * 70)
    
    if Path("zetadiffusion/compress.py").exists():
        with open("zetadiffusion/compress.py", 'r') as f:
            compressor_text = f.read()
        
        result = compress_text(compressor_text)
        spec = result['spec']
        
        print(f"Text: compress.py (compressing the compressor!)")
        print(f"  Size: {result['original_size']:,} bytes")
        print(f"  Compressed: {result['compressed_size']} bytes")
        print(f"  Ratio: {result['compression_ratio']:.1f}x")
        print()
        print(f"Spectral Signature:")
        print(f"  center = {spec.center:.10f}")
        print(f"  seed   = {spec.seed:.10f}")
        print()
        
        spec_file = "compressor_compressed.json"
        spec.to_file(spec_file)
        print(f"Saved to: .out/{spec_file}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Every text has a unique spectral signature (center, seed).")
    print("The signature encodes the operator spectrum:")
    print("  - Coherence (stability)")
    print("  - Chaos (complexity)")
    print("  - Stress (curvature)")
    print("  - Hurst (memory)")
    print()
    print("Compression is not just size reduction—")
    print("it's a mapping to the operator spectrum.")
    print()
    print("The system can compress itself, its documentation,")
    print("and any text into a single (center, seed) pair.")
    print()
    print("=" * 70)

if __name__ == "__main__":
    showcase_compression()







