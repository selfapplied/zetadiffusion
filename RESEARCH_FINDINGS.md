# Research Findings & Diagnostics

**Date:** December 1, 2025  
**Status:** Comprehensive analysis of validation results and root causes

> **Note:** For current validation status, see VALIDATION_STATUS.md  
> **Note:** For project overview, see PROJECT_STATUS.md

---

## Executive Summary

Three critical issues identified across validation runs:

1. **FEG Cascade Numerical Instability** - System diverges at χ ≈ 0.64 despite stabilization attempts
2. **Temperature Cascade No Bifurcations** - GPT-2 remains in period-1 regime across full temperature range
3. **Conjecture 9.1.1 Systematic Undercounting** - Formula error increases with n (diverging, not converging)

---

## 1. FEG Cascade: Numerical Divergence at Critical Chaos

### Observed Behavior

**Pattern:**
- System stable for χ ∈ [0.1, 0.62]
- Coherence explodes to 1e8 (clamped) at χ ≈ 0.64
- All subsequent points (χ > 0.64) remain at clamped value
- **Zero bifurcations detected** before or after divergence

**Execution Data:**
- Run time: 0.030s (fails fast)
- Bifurcations detected: 0
- δ estimate: N/A (insufficient data)

### Root Cause Analysis

**Hypothesis 1: Guardian Intervention Too Late**
- Current threshold: 10% of max (1e8)
- System may cross instability boundary before guardian activates
- **Evidence:** Coherence jumps from ~0.65 to 1e8 in single iteration

**Hypothesis 2: Discrete Renormalization Trigger**
- `ComplexRenormOperator.__call__()` applies `discrete_renorm` when `stress > 0.5` or `chaos > 0.6`
- At χ = 0.64, chaos threshold crossed → discrete renormalization activates
- **Issue:** Discrete renormalization may amplify rather than dampen at critical point

**Hypothesis 3: Complex Dimension Initialization**
- `psi.z` initialized as `0.1 * psi.chaos` when `|psi.z| < 1e-10`
- At critical point, complex dimension may cause analytic continuation to diverge
- **Evidence:** System stable until complex dimension activates

### Diagnostic Evidence

```python
# From validate_feg_cascade.py output:
[31/40] Chaos=0.638... Period=1, Coherence=100000000.0000
[32/40] Chaos=0.656... Period=1, Coherence=100000000.0000
```

**Key Observation:** Coherence jumps directly to clamp value, suggesting:
- Single iteration blow-up (not gradual)
- Guardian intervention ineffective (already at clamp)
- System enters unstable attractor basin

### Recommended Fixes

**Priority 1: Adaptive Threshold Guardian**
- Reduce guardian trigger from 10% to 1% of threshold
- Monitor gradient: `d(coherence)/d(chaos)` 
- Intervene when gradient > threshold (predictive, not reactive)

**Priority 2: Conditional Discrete Renormalization**
- Only apply `discrete_renorm` when system is in stable basin
- Check stability before renormalization: `|eigenvalue| < 1`
- Skip renormalization if system near critical point

**Priority 3: Complex Dimension Smoothing**
- Initialize `psi.z` gradually: `psi.z = 0.01 * psi.chaos` (10x smaller)
- Add damping: `psi.z *= 0.9` after each iteration
- Prevent sudden complex dimension jumps

---

## 2. Temperature Cascade: No Period-Doubling Detected

### Observed Behavior

**Pattern:**
- All temperatures (T ∈ [0.1, 4.0]): Period = 1
- Entropy saturates: H(T) → 11.6 as T → ∞
- Diversity increases: D(T) → 1.0 (full vocabulary)
- **Zero bifurcations** across entire range

**Execution Data:**
- Temperatures tested: 10 (reduced for speed)
- Bifurcations detected: 0
- δ estimate: N/A
- Entropy fit: R² = 0.966 (excellent saturation model)

### Root Cause Analysis

**Hypothesis 1: Wrong System for Feigenbaum Universality**
- GPT-2 sampling temperature ≠ renormalization group flow
- Feigenbaum δ emerges from **RG fixed point**, not model sampling
- **Evidence:** NEXT_STEPS.md notes "temperature cascade on GPT-2 is the wrong system"

**Hypothesis 2: Insufficient Resolution**
- Only 10 temperature points (reduced for speed)
- Period-doubling may occur in narrow windows (< 0.1 temperature units)
- **Evidence:** Previous runs with 80 points also showed no bifurcations

**Hypothesis 3: Model Architecture Limitation**
- GPT-2 may not exhibit period-doubling in token sequences
- Periodicity detection algorithm may be too coarse
- **Evidence:** All periods = 1 suggests detection algorithm issue OR genuine period-1 behavior

### Diagnostic Evidence

```
Period vs Temperature:
|*    *    *     *    *     *    *     *    *     *|
Range: [1.000, 1.000]
```

**Key Observation:** Perfect period-1 across all temperatures suggests:
- Either system genuinely in period-1 regime
- Or period detection algorithm insufficiently sensitive

### Recommended Fixes

**Priority 1: Use Correct System (FEG-0.4)**
- Temperature cascade should use `RGOperator` with chaos injection
- Vary chaos parameter (λ), not GPT-2 temperature
- Measure period-doubling in operator iterations, not token sequences

**Priority 2: Improve Period Detection**
- Current algorithm: Autocorrelation with lag p ∈ [1, 20]
- **Enhancement:** Use FFT to detect sub-harmonic frequencies
- Check for period-2, period-4, period-8 explicitly (Feigenbaum sequence)

**Priority 3: Extended Range Experiment**
- Test T ∈ [0.01, 20.0] with 100+ points
- Focus on low-T regime (T < 1.0) where order should emerge
- Use longer sequences (n_steps = 100+) for better period detection

---

## 3. Conjecture 9.1.1: Systematic Undercounting with Diverging Error

### Observed Behavior

**Pattern:**
- n=1-20: Average error ~30 units (~63% relative)
- n=80-100: Average error ~63 units (~42% relative)
- **Error increases 110%** (diverging, not converging)
- Correction factor: 0.0 → 0.82 → 3×10⁻¹¹ (exponentially decays)

**Execution Data:**
- n_max: 100
- Average absolute error: 51.56 units
- Maximum absolute error: 64.03 units
- Average relative error: 42.25%

### Root Cause Analysis

**Hypothesis 1: Missing Higher-Order Terms**
- Current formula: `t_n = πn/log(2π) + O(tan⁻¹(n))`
- Correction term `tan⁻¹(n)` decays to zero, but error grows
- **Evidence:** Error increases linearly with n, suggesting missing O(n) term

**Hypothesis 2: Binomial Edge Effects Incomplete**
- Correction derived from binomial expansion edge effects
- May need next-order terms: O(n²), O(log n), or O(1/n)
- **Evidence:** Relative error decreases (42% vs 63%) but absolute error increases

**Hypothesis 3: Formula Fundamentally Underestimates**
- Formula systematically undercounts zeros by factor ~2.5
- May need multiplicative correction: `t_n = C × πn/log(2π) + ...`
- **Evidence:** Formula zeros consistently ~40-50% of actual zeros

### Diagnostic Evidence

```
n=1:   Actual=14.13, Formula=2.49, Error=82%
n=50:  Actual=143.11, Formula=87.14, Error=39%
n=100: Actual=236.52, Formula=172.50, Error=27%
```

**Key Observations:**
1. **Relative error decreases** (82% → 27%) - formula scaling improves
2. **Absolute error increases** (11.6 → 64.0) - missing terms accumulate
3. **Correction factor decays** (0.82 → 3×10⁻¹¹) - current correction insufficient

### Recommended Fixes

**Priority 1: Add Linear Correction Term**
- Current: `t_n = πn/log(2π) + O(tan⁻¹(n))`
- Proposed: `t_n = πn/log(2π) + α·n + O(tan⁻¹(n))`
- Fit α from n=1..100 data: `α ≈ (error_n - error_1) / (n - 1)`

**Priority 2: Multiplicative Scaling Factor**
- Proposed: `t_n = C × [πn/log(2π) + O(tan⁻¹(n))]`
- Fit C from ratio: `C = mean(actual_n / formula_n) ≈ 1.4`
- **Evidence:** Formula consistently ~70% of actual (1/1.4 ≈ 0.7)

**Priority 3: Higher-Order Bernoulli Terms**
- Expand correction using full Bernoulli expansion
- Include O(log n) and O(1/n) terms from asymptotic expansion
- Derive from rigorous asymptotic analysis of zero-counting function

---

## 4. Operator Analysis: Missing Structural Features

### Observed Behavior

**Pattern:**
- Known limits: ✓ Gaussian fixed point, ✓ Feigenbaum residue
- Structural features: ✗ 0 poles, ✗ 0 branch cuts, ✗ 0 fixed points
- Critical exponents: δ_F = 4.669, α_F = 2.503 (exact matches)

**Execution Data:**
- Execution time: 0.047s
- Poles found: 0
- Branch cuts: 0
- Fixed points: 0
- Eigenvalues: 0

### Root Cause Analysis

**Hypothesis 1: Detection Thresholds Too High**
- Pole detection: Requires `|operator(psi)| > 1e6`
- May miss poles with smaller residues
- **Evidence:** System stable, suggesting poles may be in stable region

**Hypothesis 2: Search Space Insufficient**
- Fixed point search: 5 initial conditions
- May miss fixed points in other regions of phase space
- **Evidence:** Gaussian fixed point found (known), but no others

**Hypothesis 3: Operator Implementation Incomplete**
- `ComplexRenormOperator` may not fully implement analytic continuation
- Branch cuts require complex plane exploration
- **Evidence:** Branch cut detection uses xi function, but no cuts found

### Recommended Fixes

**Priority 1: Systematic Pole Search**
- Grid search: z ∈ [-10, 10] × [-10, 10] with resolution 0.1
- Use gradient-based detection: `|∇operator| > threshold`
- Check for poles at zeros of denominator functions

**Priority 2: Enhanced Branch Cut Detection**
- Use Riemann surface analysis: track phase discontinuities
- Explore complex plane along rays: arg(z) = constant
- Check for branch points where function becomes multi-valued

**Priority 3: Fixed Point Continuation**
- Use homotopy continuation from known fixed points
- Track fixed points as parameters vary
- Detect bifurcations in fixed point structure

---

## 5. Cross-Cutting Issues

### Execution Time Tracking
- ✅ **Fixed:** All runs now include execution_time
- All validations track timing automatically via framework

### Notion Upload Consistency
- ✅ **Fixed:** Shared framework eliminates duplicate upload logic
- All runs auto-upload with consistent metadata

### Numerical Stability
- ⚠️ **Partial:** FEG cascade still diverges despite clamping
- Guardian intervention needs predictive (not reactive) approach

---

## Next Steps Priority

### Immediate (This Week)
1. **FEG Stabilization** - Implement adaptive threshold guardian
2. **Conjecture Revision** - Add linear correction term and fit parameters
3. **Operator Pole Search** - Systematic grid search for poles

### Short-Term (Next Week)
4. **Temperature Cascade Replacement** - Use FEG-0.4 system instead of GPT-2
5. **Period Detection Enhancement** - FFT-based sub-harmonic detection
6. **Branch Cut Analysis** - Riemann surface exploration

### Medium-Term (Next Month)
7. **Theoretical Revision** - Rigorous asymptotic analysis of Conjecture 9.1.1
8. **Operator Completeness** - Verify full analytic continuation implementation
9. **Validation Suite Expansion** - Add tests for known chaotic systems (Lorenz, Rössler)

---

## Key Insights

1. **FEG divergence is predictable** - System crosses instability boundary at χ ≈ 0.64, guardian must intervene earlier

2. **Temperature cascade is wrong system** - GPT-2 sampling ≠ RG flow, need to use FEG-0.4 operator directly

3. **Conjecture needs multiplicative scaling** - Formula systematically undercounts by ~40%, needs C ≈ 1.4 factor

4. **Operator analysis incomplete** - Structural features (poles, cuts) require systematic search, not random sampling

---

**Document Status:** Complete  
**Last Updated:** December 1, 2025  
**Next Review:** After implementing Priority 1 fixes

