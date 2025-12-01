# Diagnostic Plots Summary

**Generated:** December 1, 2025  
**Location:** `.out/plots/`  
**Total Plots:** 10

---

## Generated Plots

### FEG Cascade Analysis
1. **plot1_feg_coherence_vs_chaos.png** - Coherence vs Chaos (divergence visualization)
   - Shows coherence explosion at χ ≈ 0.64
   - Highlights critical threshold region
   - Log scale for large values

2. **plot2_feg_period_vs_chaos.png** - Period vs Chaos (bifurcation detection)
   - Shows all period-1 (no bifurcations)
   - Highlights expected Feigenbaum sequence (1, 2, 4, 8, 16)

### Temperature Cascade Analysis
3. **plot3_temperature_entropy.png** - Entropy vs Temperature
   - Shows entropy saturation (H∞ ≈ 11.6)
   - Includes saturation fit curve
   - No bifurcations detected

4. **plot4_temperature_period.png** - Period vs Temperature
   - Shows all period-1 across temperature range
   - Highlights expected Feigenbaum sequence

### Conjecture 9.1.1 Analysis
5. **plot5_conjecture_error_linear.png** - Error vs n (linear scale)
   - Shows error after scaling fix
   - Average error line included
   - Error reduced from 42% to 18.57%

6. **plot6_conjecture_error_loglog.png** - Error vs n (log-log scale)
   - Power law fit analysis
   - Error scaling behavior
   - Reveals asymptotic trends

7. **plot7_conjecture_extended_error.png** - Extended error scaling (n=1..100)
   - Early vs recent error comparison
   - Trend lines showing divergence
   - Error increases with n (not converging)

### Operator Analysis
8. **plot8_operator_summary.png** - Summary dashboard
   - 4-panel overview:
     - Known limits verification
     - Structural features detected
     - Critical exponents comparison
     - Summary statistics

### System Overview
9. **plot9_execution_times.png** - Execution time comparison
   - Bar chart of all validation run times
   - Performance metrics

10. **plot10_status_overview.png** - Status overview
    - All validations completion status
    - Visual success/failure indicators

---

## Key Findings Visualized

### FEG Cascade
- **Divergence clearly visible** at χ = 0.64
- **No bifurcations** despite adaptive guardian
- Coherence bounded but still high (~123k)

### Temperature Cascade
- **Entropy saturation** well-fitted (R² = 0.966)
- **No period-doubling** across full range
- System remains in period-1 regime

### Conjecture 9.1.1
- **Error reduction** visible after scaling fix
- **Log-log plot** shows power law behavior
- **Extended plot** confirms error divergence

### Operator Analysis
- **Critical exponents match** (δ_F, α_F exact)
- **No structural features** detected (0 poles, 0 cuts)
- **Known limits verified** (partial success)

---

## Next Steps

1. **Upload plots to Notion** analysis document
2. **Review diagnostic patterns** for insights
3. **Identify failure modes** from visualizations
4. **Plan fixes** based on plot findings

---

**All plots ready for Notion upload.**

