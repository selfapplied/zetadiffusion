#!.venv/bin/python
"""
real_world_datasets.py

Week 2: Real-world dataset benchmarks for FEG Compression.

Datasets:
1. Genomics: FASTA files
2. Financial: CSV tick data
3. Satellite: Image metadata

Author: Joel
"""

import sys
from pathlib import Path
import json
import csv
from typing import List, Tuple, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.feg_compression_benchmark import benchmark_compression, generate_comparison_plots

def generate_fasta_sequence(length: int = 10000, sequence_id: str = "test") -> str:
    """
    Generate a simulated FASTA sequence.
    
    Args:
        length: Sequence length
        sequence_id: FASTA header ID
    
    Returns:
        FASTA format string
    """
    bases = ['A', 'T', 'G', 'C']
    sequence = ''.join(np.random.choice(bases, size=length))
    
    # Add some structure (repeats, motifs)
    motif = "ATGCGATCG"
    sequence = sequence[:length//2] + motif + sequence[length//2:]
    
    fasta = f">{sequence_id}\n"
    # Wrap at 80 characters per line
    for i in range(0, len(sequence), 80):
        fasta += sequence[i:i+80] + "\n"
    
    return fasta

def generate_financial_tick_data(n_ticks: int = 10000) -> str:
    """
    Generate simulated financial tick data (CSV format).
    
    Args:
        n_ticks: Number of ticks
    
    Returns:
        CSV format string
    """
    base_price = 100.0
    ticks = []
    
    # Header
    ticks.append("timestamp,symbol,price,volume,bid,ask\n")
    
    for i in range(n_ticks):
        timestamp = f"2025-12-01T{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}"
        symbol = "AAPL"
        price = base_price + np.sin(i * 0.001) * 5 + np.random.normal(0, 0.1)
        volume = int(np.random.exponential(1000))
        bid = price - 0.01
        ask = price + 0.01
        
        ticks.append(f"{timestamp},{symbol},{price:.2f},{volume},{bid:.2f},{ask:.2f}\n")
    
    return "".join(ticks)

def generate_satellite_metadata(n_images: int = 1000) -> str:
    """
    Generate simulated satellite image metadata (JSON format).
    
    Args:
        n_images: Number of images
    
    Returns:
        JSON format string
    """
    metadata = {
        'mission': 'Earth Observation',
        'satellite': 'EO-SAT-1',
        'images': []
    }
    
    for i in range(n_images):
        image = {
            'id': f'IMG_{i:06d}',
            'timestamp': f'2025-12-01T{i//3600:02d}:{(i%3600)//60:02d}:00Z',
            'coordinates': {
                'lat': 37.7749 + np.random.normal(0, 0.1),
                'lon': -122.4194 + np.random.normal(0, 0.1),
                'altitude': 500000 + np.random.normal(0, 1000)
            },
            'resolution': {
                'width': 2048,
                'height': 2048,
                'pixel_size': 0.5
            },
            'bands': ['RGB', 'NIR', 'SWIR'],
            'cloud_cover': np.random.uniform(0, 0.3),
            'quality_score': np.random.uniform(0.8, 1.0)
        }
        metadata['images'].append(image)
    
    return json.dumps(metadata, indent=2)

def load_real_world_datasets() -> List[Tuple[str, bytes]]:
    """
    Load real-world benchmark datasets.
    
    Returns:
        List of (name, data_bytes) tuples
    """
    datasets = []
    
    # 1. Genomics: FASTA sequence
    print("Generating FASTA sequence...")
    fasta_data = generate_fasta_sequence(length=10000, sequence_id="chr1")
    datasets.append(("Genomics FASTA", fasta_data.encode('utf-8')))
    
    # 2. Financial: Tick data
    print("Generating financial tick data...")
    financial_data = generate_financial_tick_data(n_ticks=10000)
    datasets.append(("Financial Ticks", financial_data.encode('utf-8')))
    
    # 3. Satellite: Image metadata
    print("Generating satellite metadata...")
    satellite_data = generate_satellite_metadata(n_images=1000)
    datasets.append(("Satellite Metadata", satellite_data.encode('utf-8')))
    
    return datasets

def run_real_world_benchmarks() -> dict:
    """
    Run benchmarks on real-world datasets.
    """
    print("=" * 70)
    print("FEG COMPRESSION: REAL-WORLD DATASET BENCHMARKS")
    print("=" * 70)
    print()
    
    datasets = load_real_world_datasets()
    all_results = {}
    
    for name, data in datasets:
        print(f"Benchmarking: {name}")
        print(f"  Original size: {len(data):,} bytes")
        print("  Running compression methods...")
        
        results = benchmark_compression(data, name)
        all_results[name] = results
        
        # Print summary
        print(f"\n  Results:")
        for method, method_results in results['methods'].items():
            if 'error' not in method_results:
                ratio = method_results['ratio']
                size = method_results['compressed_size']
                time_taken = method_results['time']
                print(f"    {method:10} | Ratio: {ratio:8.1f}x | Size: {size:8} bytes | Time: {time_taken:.4f}s")
            else:
                print(f"    {method:10} | ERROR: {method_results['error']}")
        print()
    
    return all_results

def main():
    """Run real-world benchmarks."""
    output_dir = Path(".out/benchmarks/real_world")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_real_world_benchmarks()
    
    # Save results
    results_file = output_dir / 'real_world_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to: {results_file}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(results, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("REAL-WORLD BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    
    for dataset, dataset_results in results.items():
        print(f"{dataset}:")
        best_method = None
        best_ratio = 0
        
        for method, method_results in dataset_results['methods'].items():
            if 'error' not in method_results:
                ratio = method_results['ratio']
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_method = method
        
        if best_method:
            print(f"  Best: {best_method} ({best_ratio:.1f}x)")
        print()

if __name__ == "__main__":
    main()




