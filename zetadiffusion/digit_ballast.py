"""
digit_ballast.py

Digit-Based Ballast Analysis: 9/11 Conjecture.

The 9/11 conjecture is about counting:
- 0's as ballasts (stabilizing elements)
- 9's as 1's (active/unit elements)
- When dealing with a "live number"

A "live number" is a number actively being processed or transformed,
where its digit structure encodes information about stability and activity.

Author: Joel
"""

from typing import List, Tuple, Dict
import numpy as np

def count_digits(number: float, base: int = 10) -> Dict[str, int]:
    """
    Count digits in a number.
    
    Args:
        number: The number to analyze
        base: Base for digit representation (default: 10)
    
    Returns:
        Dictionary with digit counts
    """
    # Convert to string representation
    num_str = f"{number:.15f}".replace('.', '').replace('-', '')
    
    digit_counts = {}
    for digit in '0123456789':
        digit_counts[digit] = num_str.count(digit)
    
    return digit_counts

def extract_ballast_and_units(number: float) -> Dict[str, int]:
    """
    Extract ballast (0's) and units (9's as 1's) from a live number.
    
    The 9/11 conjecture:
    - 0's count as ballasts (stabilizing elements, idle slots, breathing room)
    - 9's count as tension units (active/unit elements, almost-carries, pressure)
    
    Concurrency semantics:
    - 0's = idle, yielding, spacing, parking
    - 9's = spinning, retrying, almost-carry (almost CAS success)
    
    Args:
        number: The live number to analyze
    
    Returns:
        Dictionary with:
        - 'ballasts': count of 0's
        - 'units': count of 9's (treated as tension units)
        - 'total_digits': total digit count
        - 'ballast_ratio': ratio of ballasts to total
        - 'unit_ratio': ratio of units to total
        - 'q_9_11': Q₉₍₁₁₎ = tension / (ballast + 1) (concurrency stability index)
    """
    digit_counts = count_digits(number)
    
    ballasts = digit_counts.get('0', 0)
    units = digit_counts.get('9', 0)  # 9's count as tension units
    
    total_digits = sum(digit_counts.values())
    
    ballast_ratio = ballasts / total_digits if total_digits > 0 else 0.0
    unit_ratio = units / total_digits if total_digits > 0 else 0.0
    
    # Q₉₍₁₁₎ = tension / (ballast + 1)
    # This is the concurrency stability index
    q_9_11 = units / (ballasts + 1.0) if (ballasts + 1.0) > 0 else float('inf')
    
    return {
        'ballasts': ballasts,
        'units': units,
        'total_digits': total_digits,
        'ballast_ratio': ballast_ratio,
        'unit_ratio': unit_ratio,
        'q_9_11': q_9_11,
        'digit_counts': digit_counts
    }

def analyze_live_number_sequence(numbers: List[float]) -> List[Dict]:
    """
    Analyze a sequence of live numbers for ballast and unit patterns.
    
    Args:
        numbers: Sequence of numbers to analyze
    
    Returns:
        List of analysis dictionaries for each number
    """
    analyses = []
    for num in numbers:
        analysis = extract_ballast_and_units(num)
        analysis['value'] = num
        analyses.append(analysis)
    
    return analyses

def compute_ballast_trajectory(numbers: List[float]) -> Dict:
    """
    Compute ballast trajectory across a sequence.
    
    Tracks how ballast (0's) and units (9's) evolve.
    
    Args:
        numbers: Sequence of numbers
    
    Returns:
        Dictionary with trajectory data
    """
    analyses = analyze_live_number_sequence(numbers)
    
    ballasts = [a['ballasts'] for a in analyses]
    units = [a['units'] for a in analyses]
    ballast_ratios = [a['ballast_ratio'] for a in analyses]
    unit_ratios = [a['unit_ratio'] for a in analyses]
    
    return {
        'ballasts': ballasts,
        'units': units,
        'ballast_ratios': ballast_ratios,
        'unit_ratios': unit_ratios,
        'analyses': analyses,
        'avg_ballast': float(np.mean(ballasts)) if ballasts else 0.0,
        'avg_units': float(np.mean(units)) if units else 0.0,
        'total_ballasts': sum(ballasts),
        'total_units': sum(units)
    }

def detect_9_11_pattern(numbers: List[float], indices: List[int] = None) -> Dict:
    """
    Detect 9/11 pattern in a sequence of live numbers.
    
    Looks for:
    - Ballast accumulation (0's) around n=9
    - Unit activation (9's as 1's) around n=11
    - Transition structure
    
    Args:
        numbers: Sequence of numbers
        indices: Optional list of indices (n values)
    
    Returns:
        Dictionary with pattern detection results
    """
    if indices is None:
        indices = list(range(1, len(numbers) + 1))
    
    trajectory = compute_ballast_trajectory(numbers)
    
    # Find n=9 and n=11 regions
    n9_indices = [i for i, n in enumerate(indices) if 8.5 <= n <= 9.5]
    n11_indices = [i for i, n in enumerate(indices) if 10.5 <= n <= 11.5]
    
    # Analyze ballast at n=9
    n9_ballasts = [trajectory['ballasts'][i] for i in n9_indices if i < len(trajectory['ballasts'])]
    n9_ballast_avg = np.mean(n9_ballasts) if n9_ballasts else 0.0
    
    # Analyze units at n=11
    n11_units = [trajectory['units'][i] for i in n11_indices if i < len(trajectory['units'])]
    n11_units_avg = np.mean(n11_units) if n11_units else 0.0
    
    # Check for pattern
    has_ballast_at_9 = n9_ballast_avg > np.mean(trajectory['ballasts']) if trajectory['ballasts'] else False
    has_units_at_11 = n11_units_avg > np.mean(trajectory['units']) if trajectory['units'] else False
    
    return {
        'n9_ballast_avg': float(n9_ballast_avg),
        'n11_units_avg': float(n11_units_avg),
        'has_ballast_at_9': has_ballast_at_9,
        'has_units_at_11': has_units_at_11,
        'trajectory': trajectory,
        'pattern_detected': has_ballast_at_9 and has_units_at_11
    }

