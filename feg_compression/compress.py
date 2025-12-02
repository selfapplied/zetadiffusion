"""
feg_compression.compress

Core compression functions for FEG Compression library.

Author: Joel
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Import from zetadiffusion (parent package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from zetadiffusion.compress import (
    compress_text as _compress_text,
    decompress_text as _decompress_text,
    compress_text_fixed_point as _compress_text_fixed_point
)
from zetadiffusion.windowspec import SeedSpec

def compress(data: str, lossless: bool = False, max_iterations: int = 10000) -> Dict:
    """
    Compress text data using FEG compression.
    
    Args:
        data: Text data to compress
        lossless: If True, use fixed-point search for lossless compression
        max_iterations: Maximum iterations for fixed-point search
    
    Returns:
        Dictionary with compression results:
        - 'spec': SeedSpec object
        - 'compressed_size': Size in bytes (always 16)
        - 'original_size': Original size in bytes
        - 'compression_ratio': Compression ratio
        - 'lossless': Whether compression is lossless
    """
    if lossless:
        result = _compress_text_fixed_point(data, max_iterations)
    else:
        result = _compress_text(data, calibrate=True)
    
    return {
        'spec': result['spec'],
        'compressed_size': result['compressed_size'],
        'original_size': result['original_size'],
        'compression_ratio': result['compression_ratio'],
        'lossless': result.get('lossless', False) or result.get('fixed_point_found', False)
    }

def decompress(spec: SeedSpec, target_length: Optional[int] = None) -> str:
    """
    Decompress data from a SeedSpec.
    
    Args:
        spec: SeedSpec object containing compressed data
        target_length: Target length for decompressed data (default: from spec.center)
    
    Returns:
        Decompressed text data
    """
    return _decompress_text(spec, target_length)

def compress_file(input_path: str, output_path: Optional[str] = None, 
                 lossless: bool = False) -> Dict:
    """
    Compress a file using FEG compression.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (default: input_path + '.feg')
        lossless: If True, use lossless compression
    
    Returns:
        Compression result dictionary
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read()
    
    # Compress
    result = compress(data, lossless=lossless)
    
    # Save SeedSpec
    if output_path is None:
        output_path = str(input_file) + '.feg'
    
    result['spec'].to_file(output_path)
    
    result['input_file'] = str(input_file)
    result['output_file'] = output_path
    
    return result

def decompress_file(input_path: str, output_path: Optional[str] = None,
                   target_length: Optional[int] = None) -> str:
    """
    Decompress a file from a SeedSpec.
    
    Args:
        input_path: Path to .feg file
        output_path: Path to output file (default: input_path without .feg)
        target_length: Target length for decompressed data
    
    Returns:
        Decompressed text data
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load SeedSpec
    spec = SeedSpec.from_file(str(input_file))
    
    # Decompress
    data = decompress(spec, target_length)
    
    # Save output
    if output_path is None:
        if input_file.suffix == '.feg':
            output_path = str(input_file.with_suffix(''))
        else:
            output_path = str(input_file) + '.decompressed'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(data)
    
    return data




