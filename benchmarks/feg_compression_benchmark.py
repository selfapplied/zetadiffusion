#!.venv/bin/python
"""
feg_compression_benchmark.py

Week 1: FEG Compression Benchmark Suite

Tests FEG compression on 5 standard benchmark datasets:
1. Text corpus (literature, technical docs)
2. Time series data (financial, sensor)
3. Network traces (packet captures, logs)
4. Genomic sequences (DNA/RNA)
5. Structured data (JSON, CSV)

Compares against:
- RLE (Run-Length Encoding)
- Huffman coding
- gzip
- zlib

Generates comparative plots and metrics.

Author: Joel
"""

import sys
import os
from pathlib import Path
import time
import json
import gzip
import zlib
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zetadiffusion.compress import compress_text, compress_text_fixed_point
from zetadiffusion.windowspec import SeedSpec

# Standard compression libraries
try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False

def rle_compress(data: bytes) -> bytes:
    """Run-Length Encoding compression."""
    if not data:
        return b''
    
    compressed = bytearray()
    i = 0
    while i < len(data):
        current = data[i]
        count = 1
        while i + count < len(data) and data[i + count] == current and count < 255:
            count += 1
        compressed.append(count)
        compressed.append(current)
        i += count
    
    return bytes(compressed)

def rle_decompress(compressed: bytes) -> bytes:
    """Run-Length Encoding decompression."""
    if not compressed:
        return b''
    
    decompressed = bytearray()
    i = 0
    while i < len(compressed):
        if i + 1 >= len(compressed):
            break
        count = compressed[i]
        byte_val = compressed[i + 1]
        decompressed.extend([byte_val] * count)
        i += 2
    
    return bytes(decompressed)

def huffman_compress_simple(data: bytes) -> Tuple[bytes, Dict]:
    """
    Simple Huffman coding implementation.
    For benchmarking purposes - not production quality.
    """
    # Count frequencies
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Build simple encoding (most frequent = shortest codes)
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Simple prefix code assignment
    code_map = {}
    code_length = 1
    for byte_val, _ in sorted_bytes:
        if code_length > 8:
            # Fallback: use byte itself
            code_map[byte_val] = (byte_val.to_bytes(1, 'big'), 8)
        else:
            # Use variable-length code
            code = bin(code_length - 1)[2:].zfill(code_length)
            code_map[byte_val] = (int(code, 2).to_bytes(1, 'big'), code_length)
            code_length += 1
    
    # Encode data
    compressed = bytearray()
    bit_buffer = 0
    bit_count = 0
    
    for byte in data:
        code_bytes, code_len = code_map[byte]
        # Simplified: just append code bytes
        compressed.extend(code_bytes)
    
    # Add frequency table for decompression
    freq_table = json.dumps(freq).encode('utf-8')
    result = len(freq_table).to_bytes(4, 'big') + freq_table + bytes(compressed)
    
    return result, code_map

def benchmark_compression(data: bytes, data_name: str) -> Dict:
    """
    Benchmark all compression methods on given data.
    
    Returns:
        Dictionary with compression results for each method
    """
    original_size = len(data)
    results = {
        'data_name': data_name,
        'original_size': original_size,
        'methods': {}
    }
    
    # FEG Compression (text-based)
    try:
        data_text = data.decode('utf-8', errors='ignore')
        start_time = time.time()
        feg_result = compress_text(data_text)
        feg_time = time.time() - start_time
        
        feg_compressed_size = feg_result['compressed_size']
        feg_ratio = feg_result['compression_ratio']
        
        results['methods']['FEG'] = {
            'compressed_size': feg_compressed_size,
            'ratio': feg_ratio,
            'time': feg_time,
            'lossless': False  # FEG is lossy by design
        }
    except Exception as e:
        results['methods']['FEG'] = {'error': str(e)}
    
    # RLE
    try:
        start_time = time.time()
        rle_compressed = rle_compress(data)
        rle_time = time.time() - start_time
        rle_ratio = original_size / len(rle_compressed) if rle_compressed else 1.0
        
        results['methods']['RLE'] = {
            'compressed_size': len(rle_compressed),
            'ratio': rle_ratio,
            'time': rle_time,
            'lossless': True
        }
    except Exception as e:
        results['methods']['RLE'] = {'error': str(e)}
    
    # gzip
    try:
        start_time = time.time()
        gzip_compressed = gzip.compress(data)
        gzip_time = time.time() - start_time
        gzip_ratio = original_size / len(gzip_compressed) if gzip_compressed else 1.0
        
        results['methods']['gzip'] = {
            'compressed_size': len(gzip_compressed),
            'ratio': gzip_ratio,
            'time': gzip_time,
            'lossless': True
        }
    except Exception as e:
        results['methods']['gzip'] = {'error': str(e)}
    
    # zlib
    try:
        start_time = time.time()
        zlib_compressed = zlib.compress(data)
        zlib_time = time.time() - start_time
        zlib_ratio = original_size / len(zlib_compressed) if zlib_compressed else 1.0
        
        results['methods']['zlib'] = {
            'compressed_size': len(zlib_compressed),
            'ratio': zlib_ratio,
            'time': zlib_time,
            'lossless': True
        }
    except Exception as e:
        results['methods']['zlib'] = {'error': str(e)}
    
    # lzma (if available)
    if LZMA_AVAILABLE:
        try:
            start_time = time.time()
            lzma_compressed = lzma.compress(data)
            lzma_time = time.time() - start_time
            lzma_ratio = original_size / len(lzma_compressed) if lzma_compressed else 1.0
            
            results['methods']['lzma'] = {
                'compressed_size': len(lzma_compressed),
                'ratio': lzma_ratio,
                'time': lzma_time,
                'lossless': True
            }
        except Exception as e:
            results['methods']['lzma'] = {'error': str(e)}
    
    # Huffman (simple)
    try:
        start_time = time.time()
        huffman_compressed, _ = huffman_compress_simple(data)
        huffman_time = time.time() - start_time
        huffman_ratio = original_size / len(huffman_compressed) if huffman_compressed else 1.0
        
        results['methods']['Huffman'] = {
            'compressed_size': len(huffman_compressed),
            'ratio': huffman_ratio,
            'time': huffman_time,
            'lossless': True
        }
    except Exception as e:
        results['methods']['Huffman'] = {'error': str(e)}
    
    return results

def load_benchmark_datasets() -> List[Tuple[str, bytes]]:
    """
    Load 5 standard benchmark datasets.
    
    Returns:
        List of (name, data_bytes) tuples
    """
    datasets = []
    
    # 1. Text corpus
    text_samples = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
        "To be or not to be, that is the question. " * 200,
    ]
    text_corpus = "\n".join(text_samples)
    datasets.append(("Text Corpus", text_corpus.encode('utf-8')))
    
    # 2. Time series data (simulated financial tick data)
    time_series = []
    base_price = 100.0
    for i in range(1000):
        price = base_price + np.sin(i * 0.1) * 10 + np.random.normal(0, 0.5)
        timestamp = f"{i:06d}"
        time_series.append(f"{timestamp},{price:.2f}\n")
    datasets.append(("Time Series", "".join(time_series).encode('utf-8')))
    
    # 3. Network traces (simulated log data)
    network_logs = []
    for i in range(500):
        src_ip = f"192.168.1.{i % 255}"
        dst_ip = f"10.0.0.{i % 255}"
        port = 80 + (i % 1000)
        network_logs.append(f"{i:06d} {src_ip}:{port} -> {dst_ip}:{port} TCP\n")
    datasets.append(("Network Traces", "".join(network_logs).encode('utf-8')))
    
    # 4. Genomic sequences (simulated DNA)
    bases = ['A', 'T', 'G', 'C']
    genomic_seq = ''.join(np.random.choice(bases, size=5000))
    datasets.append(("Genomic Sequence", genomic_seq.encode('utf-8')))
    
    # 5. Structured data (JSON)
    structured_data = {
        'users': [
            {'id': i, 'name': f'User{i}', 'email': f'user{i}@example.com', 'score': i * 10}
            for i in range(100)
        ],
        'metadata': {
            'version': '1.0',
            'timestamp': '2025-12-01T00:00:00Z',
            'count': 100
        }
    }
    json_data = json.dumps(structured_data, indent=2)
    datasets.append(("Structured JSON", json_data.encode('utf-8')))
    
    return datasets

def run_benchmarks() -> Dict:
    """
    Run all benchmarks and return results.
    """
    print("=" * 70)
    print("FEG COMPRESSION BENCHMARK SUITE")
    print("=" * 70)
    print()
    
    datasets = load_benchmark_datasets()
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

def generate_comparison_plots(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # Plot 1: Compression ratios
    fig, ax = plt.subplots(figsize=(12, 8))
    
    datasets = list(results.keys())
    methods = ['FEG', 'RLE', 'gzip', 'zlib', 'Huffman']
    if LZMA_AVAILABLE:
        methods.append('lzma')
    
    x = np.arange(len(datasets))
    width = 0.15
    
    for i, method in enumerate(methods):
        ratios = []
        for dataset in datasets:
            if method in results[dataset]['methods'] and 'error' not in results[dataset]['methods'][method]:
                ratios.append(results[dataset]['methods'][method]['ratio'])
            else:
                ratios.append(0)
        
        offset = (i - len(methods) / 2) * width
        ax.bar(x + offset, ratios, width, label=method)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('FEG Compression Benchmark: Compression Ratios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / 'compression_ratios.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")
    
    # Plot 2: Compression times
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        times = []
        for dataset in datasets:
            if method in results[dataset]['methods'] and 'error' not in results[dataset]['methods'][method]:
                times.append(results[dataset]['methods'][method]['time'] * 1000)  # ms
            else:
                times.append(0)
        
        offset = (i - len(methods) / 2) * width
        ax.bar(x + offset, times, width, label=method)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Compression Time (ms)', fontsize=12)
    ax.set_title('FEG Compression Benchmark: Compression Times', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / 'compression_times.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

def main():
    """Run benchmark suite."""
    output_dir = Path(".out/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Save results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to: {results_file}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(results, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
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




