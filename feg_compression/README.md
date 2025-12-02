# FEG Compression Library

**Topological Projection-Based Compression with 300-1400x Ratios**

## Installation

```bash
pip install -e .
```

## Quick Start

### Command Line

```bash
# Compress a file
feg-compress input.txt -o output.feg

# Decompress
feg-decompress output.feg -o decompressed.txt

# Compress text string
feg-compress "Hello, world!" -t

# Run benchmarks
feg-benchmark --plots
```

### Python API

```python
from feg_compression import compress, decompress, SeedSpec

# Compress text
result = compress("Hello, world!")
print(f"Compression ratio: {result['compression_ratio']:.1f}x")

# Save compressed data
result['spec'].to_file('compressed.feg')

# Decompress
spec = SeedSpec.from_file('compressed.feg')
decompressed = decompress(spec)
print(decompressed)
```

## Features

- **300-1400x compression ratios** on standard datasets
- **16-byte output** (topological projection to 2 floats)
- **Fast compression** (1-6ms per dataset)
- **Lossy and lossless modes**
- **CLI and Python API**

## How It Works

FEG compression uses **topological projection**:
1. Maps data to system state (coherence, chaos, stress, hurst)
2. Encodes state into spectral signature (center, seed)
3. Regenerates via fractal generation with Guardian Nash Policy

The 16-byte SeedSpec contains the complete operator spectrum, enabling reconstruction of data structure and patterns.

## Benchmarks

See `benchmarks/README.md` for full benchmark results.

## Documentation

- `FEG_COMPRESSION_ROADMAP.md` - 30-day development plan
- `benchmarks/README.md` - Benchmark results and methodology

## License

MIT




