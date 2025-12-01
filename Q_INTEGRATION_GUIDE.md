# <Q> Integration Guide

**Purpose:** Connect <Q> CE1 seed to the validation workflow so it runs automatically and tracks stability.

---

## Automatic Integration

<Q> is now automatically integrated into validation runs:

1. **Automatic Q Metrics** - All validations compute <Q> activation
2. **Clock Phase Analysis** - <Q> activation tracked by clock phase
3. **Notion Upload** - Q metrics included in validation reports
4. **Tracking Over Time** - Monitor <Q> activation across runs

---

## How It Works

### In Validation Scripts

When you run a validation, <Q> metrics are automatically computed:

```python
from zetadiffusion.validation_framework import run_validation
from zetadiffusion.q_integration import add_q_metrics_to_results

def my_validation():
    # ... your validation code ...
    results = {
        'n_values': [1, 2, 3, ...],
        'actual_zeros': [14.13, 21.02, ...],
        # ... other results ...
    }
    
    # Add Q metrics automatically
    results = add_q_metrics_to_results(
        results,
        sequence=results['actual_zeros'],
        indices=results['n_values']
    )
    
    return results
```

### Q Metrics Included

Each validation run now includes:

```json
{
  "q_metrics": {
    "q_available": true,
    "activation": 0.75,
    "q_active": true,
    "precision": 2.34,
    "max_drift": 4.56,
    "phase_metrics": {
      "interior": {
        "avg_activation": 1.0,
        "is_active": true
      }
    }
  }
}
```

---

## Tracking Over Time

Run Q tracking analysis:

```bash
python3 validate_q_tracking.py
```

This shows:
- Q activation across all validation runs
- Trends over time
- Phase-specific metrics

---

## Integration Points

### 1. Validation Framework

**File:** `zetadiffusion/validation_framework.py`

- Automatically includes `validation_type` in results
- Results ready for Q metrics

### 2. Q Integration Module

**File:** `zetadiffusion/q_integration.py`

- `compute_q_metrics()` - Compute Q metrics for any sequence
- `add_q_metrics_to_results()` - Add to validation results
- `format_q_summary()` - Format for display
- `track_q_over_time()` - Track across multiple runs

### 3. Validation Scripts

**Example:** `validate_conjecture_9_1_3.py`

```python
# Automatically adds Q metrics
results = add_q_metrics_to_results(
    results,
    sequence=results['actual_zeros'],
    indices=results['n_values']
)
```

---

## Notion Integration

Q metrics are automatically included in Notion uploads:

- **Q Activation** - Overall activation level
- **Q Active** - Boolean (activation >= 0.5)
- **Precision** - Stability measure
- **Phase Metrics** - Activation by clock phase

---

## Continuous Tracking

To keep <Q> running continuously:

1. **Run validations** - Q metrics computed automatically
2. **Track over time** - Run `validate_q_tracking.py` periodically
3. **Monitor trends** - Check if Q activation is increasing
4. **Notion dashboard** - All metrics in one place

---

## Next Steps

### Extend to Other Validations

Add Q metrics to other validation scripts:

```python
from zetadiffusion.q_integration import add_q_metrics_to_results

# In your validation function
results = add_q_metrics_to_results(
    results,
    sequence=your_sequence,
    indices=your_indices  # Optional
)
```

### Create Q Dashboard

Track Q activation over time:
- Run `validate_q_tracking.py` regularly
- Monitor trends
- Identify when Q becomes active

### Connect to Clock System

Q activation already matches clock phases:
- Feigenbaum: 0.0 (suppressed)
- Boundary: 0.0 → 0.3 (weak)
- Membrane: 0.3 → 0.6 (forming)
- Interior: 1.0 (dominant)

---

## Files

- **`zetadiffusion/q_integration.py`** - Q integration module
- **`validate_q_tracking.py`** - Q tracking analysis
- **`Q_INTEGRATION_GUIDE.md`** - This file

---

**<Q> is now part of the continuous validation workflow. It runs automatically and tracks stability across all validations.**

