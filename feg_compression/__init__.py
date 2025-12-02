"""
FEG Compression: Topological Projection-Based Compression

A compression library that uses topological projection to encode data
into operator spectrum parameters, achieving 300-1400x compression ratios.

Author: Joel
"""

__version__ = "0.1.0"

from feg_compression.compress import compress, decompress, compress_file, decompress_file
from feg_compression.windowspec import SeedSpec

__all__ = [
    'compress',
    'decompress',
    'compress_file',
    'decompress_file',
    'SeedSpec',
    '__version__',
]




