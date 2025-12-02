"""
feg_compression.windowspec

SeedSpec wrapper for feg_compression package.

Author: Joel
"""

import sys
from pathlib import Path

# Import from zetadiffusion (parent package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from zetadiffusion.windowspec import SeedSpec

__all__ = ['SeedSpec']




