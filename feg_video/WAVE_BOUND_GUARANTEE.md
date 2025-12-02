# Wave-Bound Guarantee: FEG–SVG Stability Law

**Status:** Core Specification

---

## Core Principle

**FEG strings can oscillate forever, but they can never recurse forever.**

SVG receives only:
- fixed-point waves
- bounded oscillations
- converged coupled fields
- stable harmonic expressions

---

## The Harmonic Bound Law

All evaluated FEG expressions MUST satisfy:

```
|S(x)|  <  ∞
and
∂S/∂x  <  ∞
and
spectral_radius(Jacobian(S)) < 1
```

This enforces:
- stability
- contractiveness
- nondivergence

**The wave stays tame.**

---

## Constraints

### 1. Boundedness

All expressions MUST be composed solely of:
- base waveforms (sin, cos, tanh, saw, noise)
- bounded algebraic combinations
- bounded compositions of other strings

```
∀x, S(x) is finite.
```

### 2. No Procedural Recursion

Symbol references MUST be treated as **functional substitution**, not recursive calls:

```
foo(x) = bar(x) + sin(x)
bar(x) = 0.5 * sin(2*x)

→ foo(x) = (0.5*sin(2*x)) + sin(x)  ✓

NOT: call foo → call bar → call foo → ...  ✗
```

### 3. Harmonic Fixed-Point Collapse

Self-referential strings MUST collapse into stable modes:

```
S = f(S)  →  S*  where  S* = f(S*)
```

### 4. Mutual Recursion → Coupled Harmonic System

Mutual dependencies MUST be resolved simultaneously:

```
A = f(B)
B = g(A)

→ [A*, B*] = Solve([A=f(B), B=g(A)])  ✓
```

### 5. SVG as Bounded Evaluator

All expressions MUST be resolved **prior to injection** into SVG runtime.

SVG SHALL NOT execute unbounded evaluation.

### 6. Explicit Time Dependency

Time MUST be explicit via dependency symbol `t`:

```
S(t) is evaluated at each frame,
not as infinite self-referential animation.
```

---

## Implementation

### Expression Validation

At parse time, validate:
- Only bounded functions allowed
- No infinite loops in dependency graph
- All dependencies resolve to finite values

### Evaluation with Visited Set

```javascript
evaluateString(symbol, deps, t, context, visited) {
  // Prevent recursion
  if (visited.has(symbol)) return fixedPoint(symbol);
  
  visited.add(symbol);
  // Evaluate expression
  // Ensure result is finite and bounded
  visited.delete(symbol);
}
```

### Fixed-Point Collapse

```javascript
function fixedPoint(symbol) {
  // Return cached fixed-point value
  // or compute: S* = f(S*)
  return stableValue;
}
```

---

## Why This Matters

This makes the entire ecosystem:
- **safe** — no infinite loops
- **predictable** — bounded outputs
- **expressive** — still powerful
- **mathematically grounded** — convergence theorem
- **infinitely compositional** — without infinite cost

**It's the first declarative video/field system with a built-in convergence theorem.**

---

**This is the stability law.**

**This is the wave-bound guarantee.**

**This is what makes FEG safe for SVG and Tellah.**


