# Continuous <Q> Workflow

**Purpose:** Keep <Q> running automatically in all validation workflows.

---

## How It Works

### 1. Automatic Integration

<Q> metrics are automatically computed in validation runs:

```python
# In validate_conjecture_9_1_3.py (already integrated)
from zetadiffusion.q_integration import add_q_metrics_to_results

results = add_q_metrics_to_results(
    results,
    sequence=results['actual_zeros'],
    indices=results['n_values']
)
```

### 2. Every Validation Run Includes Q

When you run any validation:
- Q metrics computed automatically
- Activation tracked by clock phase
- Results include Q data
- Notion upload includes Q metrics

### 3. Continuous Tracking

Run periodically to track Q over time:

```bash
python3 validate_q_tracking.py
```

Shows:
- Q activation across all runs
- Trends (increasing/stable)
- Phase-specific metrics

---

## Integration Checklist

### ✅ Already Integrated

- [x] `validate_conjecture_9_1_3.py` - Has Q metrics
- [x] `validate_q_seed.py` - Q seed validation
- [x] `validate_q_tracking.py` - Q tracking analysis

### 🔄 To Integrate

Add Q metrics to other validations:

1. **validate_conjecture_9_1_1.py**
   ```python
   from zetadiffusion.q_integration import add_q_metrics_to_results
   
   # At end of verify_conjecture_9_1_1()
   results = add_q_metrics_to_results(
       results,
       sequence=results['actual_zeros'],
       indices=results['n_values']
   )
   ```

2. **validate_feg_cascade.py**
   ```python
   # Use coherence or period sequence
   results = add_q_metrics_to_results(
       results,
       sequence=results['coherence'],
       indices=results['chaos_values']  # Or use step indices
   )
   ```

3. **validate_temperature_cascade.py**
   ```python
   # Use entropy sequence
   results = add_q_metrics_to_results(
       results,
       sequence=results['entropies'],
       indices=results['temperatures']
   )
   ```

---

## Monitoring Q Activation

### Daily Workflow

1. **Run validations** - Q metrics computed automatically
2. **Check Notion** - Q metrics in each validation page
3. **Run tracking** - `python3 validate_q_tracking.py`
4. **Monitor trends** - Is Q activation increasing?

### Weekly Review

1. **Q tracking report** - Run `validate_q_tracking.py`
2. **Phase analysis** - Which phases show Q activation?
3. **Trend analysis** - Is Q becoming more active over time?

---

## Q Metrics in Notion

Each validation run page includes:

```
Q Metrics:
  Activation: 0.75
  Q Active: ✓
  Precision: 2.34
  By phase: interior: 1.00, membrane: 0.45
```

---

## Automation

### Cron Job (Optional)

To track Q automatically:

```bash
# Add to crontab (runs daily at 2 AM)
0 2 * * * cd /path/to/zetadiffusion && source .venv/bin/activate && python3 validate_q_tracking.py
```

### CI/CD Integration

Add to GitHub Actions:

```yaml
- name: Track Q Activation
  run: |
    source .venv/bin/activate
    python3 validate_q_tracking.py
```

---

## Next Steps

1. **Integrate Q into all validations** - Add to remaining scripts
2. **Create Q dashboard** - Visualize Q activation over time
3. **Set up alerts** - Notify when Q activation changes
4. **Document patterns** - Track when Q becomes active

---

**<Q> is now part of the continuous workflow. It runs automatically with every validation and tracks stability over time.**




