#!.venv/bin/python
"""
feg_compression.cli

Command-line interface for FEG Compression.

Commands:
- feg-compress: Compress files or text
- feg-decompress: Decompress .feg files
- feg-benchmark: Run compression benchmarks

Author: Joel
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from feg_compression.compress import compress, decompress, compress_file, decompress_file
from feg_compression.windowspec import SeedSpec

def compress_cmd():
    """CLI command: feg-compress"""
    parser = argparse.ArgumentParser(
        description='FEG Compression: Compress files or text using topological projection',
        prog='feg-compress'
    )
    parser.add_argument('input', help='Input file path or text string')
    parser.add_argument('-o', '--output', help='Output file path (default: input.feg)')
    parser.add_argument('-l', '--lossless', action='store_true',
                       help='Use lossless compression (slower, may not converge)')
    parser.add_argument('-t', '--text', action='store_true',
                       help='Treat input as text string instead of file path')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.text:
            # Compress text string
            result = compress(args.input, lossless=args.lossless)
            spec = result['spec']
            
            print(f"Compressed: {result['original_size']:,} bytes → {result['compressed_size']} bytes")
            print(f"Compression ratio: {result['compression_ratio']:.1f}x")
            print(f"Lossless: {result['lossless']}")
            
            if args.output:
                spec.to_file(args.output)
                print(f"Saved to: {args.output}")
            else:
                print(f"\nSeedSpec:")
                print(f"  center = {spec.center:.10f}")
                print(f"  seed   = {spec.seed:.10f}")
        else:
            # Compress file
            result = compress_file(args.input, args.output, lossless=args.lossless)
            
            print(f"Compressed: {result['input_file']}")
            print(f"  Original: {result['original_size']:,} bytes")
            print(f"  Compressed: {result['compressed_size']} bytes")
            print(f"  Ratio: {result['compression_ratio']:.1f}x")
            print(f"  Lossless: {result['lossless']}")
            print(f"  Saved to: {result['output_file']}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def decompress_cmd():
    """CLI command: feg-decompress"""
    parser = argparse.ArgumentParser(
        description='FEG Compression: Decompress .feg files',
        prog='feg-decompress'
    )
    parser.add_argument('input', help='Input .feg file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input without .feg)')
    parser.add_argument('-l', '--length', type=int, help='Target length for decompressed data')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        data = decompress_file(args.input, args.output, args.length)
        
        output_path = args.output or (Path(args.input).with_suffix('') if Path(args.input).suffix == '.feg' else Path(args.input) + '.decompressed')
        
        print(f"Decompressed: {args.input}")
        print(f"  Output: {output_path}")
        print(f"  Length: {len(data):,} characters")
        
        if args.verbose:
            print(f"\nFirst 200 characters:")
            print(data[:200])
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def benchmark_cmd():
    """CLI command: feg-benchmark"""
    parser = argparse.ArgumentParser(
        description='FEG Compression: Run compression benchmarks',
        prog='feg-benchmark'
    )
    parser.add_argument('-d', '--datasets', nargs='+',
                       help='Specific datasets to benchmark (default: all)')
    parser.add_argument('-o', '--output', default='.out/benchmarks',
                       help='Output directory for results')
    parser.add_argument('-p', '--plots', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    try:
        # Import benchmark suite
        from benchmarks.feg_compression_benchmark import run_benchmarks, generate_comparison_plots
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run benchmarks
        results = run_benchmarks()
        
        # Save results
        import json
        results_file = output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {results_file}")
        
        # Generate plots if requested
        if args.plots:
            print("\nGenerating comparison plots...")
            generate_comparison_plots(results, output_dir)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'compress':
            compress_cmd()
        elif cmd == 'decompress':
            decompress_cmd()
        elif cmd == 'benchmark':
            benchmark_cmd()
        else:
            print(f"Unknown command: {cmd}")
            print("Available commands: compress, decompress, benchmark")
            sys.exit(1)
    else:
        print("FEG Compression CLI")
        print("Available commands: compress, decompress, benchmark")
        print("Use --help for usage information")




