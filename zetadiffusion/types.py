"""
types.py

Type definitions for modular arithmetic domains.
Enforces explicit conversion and unit awareness.
"""

import numpy as np
from typing import TypeVar, Union, cast

# Base classes that behave like floats but carry unit metadata
class Radians(float):
    """The Circle Group S1 (Radians)."""
    units = 2 * np.pi
    
    def __new__(cls, value):
        return super().__new__(cls, value)

class Turns(float):
    """The Unit Interval (Turns)."""
    units = 1.0

    def __new__(cls, value):
        return super().__new__(cls, value)

def to_radians(x: Union[float, np.ndarray]) -> Union[Radians, np.ndarray]:
    """Converts value to Radians (mod 2pi)."""
    val = np.mod(x, 2 * np.pi)
    if np.isscalar(val):
        return Radians(val)
    return val

def to_turns(x: Union[float, np.ndarray]) -> Union[Turns, np.ndarray]:
    """Converts value to Turns (mod 1.0)."""
    val = np.mod(x, 1.0)
    if np.isscalar(val):
        return Turns(val)
    return val

def to_int(x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Converts to standard integer."""
    return np.array(x).astype(int) if not np.isscalar(x) else int(x)

def units(x: Union[Radians, Turns]) -> float:
    "The modulus unit of the typed value."
    return x.units