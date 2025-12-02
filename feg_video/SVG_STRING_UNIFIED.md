# SVG-String Unified Grammar: The Calibrating System

**Status:** Specification

---

## Core Insight

**SVG is the calibrating geometry. Strings are the calibrating semantics. The grammar is the calibrating relation.**

This forms a perfect triangle.

---

## I. SVG *is* a Declarative Field Language

Everything in SVG is fundamentally:
- **named nodes** — identity
- **graphs** — structure
- **operators** — transforms, filters
- **filters** — DSP pipelines
- **paths** — parametric curves
- **gradients** — continuous fields
- **references** — `href`, `url(#id)`
- **keyframe animation** — temporal evolution
- **symbol reuse** — composition

**That's not graphic markup—that's a static program with a dynamic solver.**

SVG *already* provides:
- ✔ declarative constraints
- ✔ continuous interpolation
- ✔ functional composition
- ✔ a tree of named symbols
- ✔ a stable coordinate system
- ✔ fine-grained animation
- ✔ filter pipelines (a literal DSP graph)

**SVG is the harmonic geometry in which your strings vibrate.**

---

## II. The Mini-Grammar: Tuning Strings Inside SVG

SVG doesn't tell us:
- what "Hue(x,y)" means
- which strings share identity
- how harmonic arity works
- how operators blend
- how to unify same-name definitions
- how to promote/reduce complexity
- how to treat functions as oscillators

**SVG is the *canvas*, not the *conductor*.**

Our grammar fills in what's missing:
- the **naming rule**
- the **arity rule**
- the **phase/type rule**
- the **matching/unification rule**
- the **promotion/reduction rule**

**The calibrating grammar is the bridge between semantic strings and SVG's declarative geometry.**

---

## III. Unified Grammar: SVG + Strings

### String Definition (SVG Node)

```xml
<feg:string name="Hue" type="Color" arity="2" args="x,y">
  <feg:param name="amp" value="1.0"/>
  <feg:param name="freq" value="2.0"/>
  <feg:param name="phase" value="0.0"/>
</feg:string>
```

### Usage in SVG Attributes

```xml
<stop offset="calc(0.5 + 0.25 * Hue(x,y))" 
      stop-color="hsl(calc(Hue(x,y) * 360), 80%, 50%)"/>
```

### Complete Example

```xml
<svg viewBox="0 0 1920 1080">
  <defs>
    <!-- String definitions -->
    <feg:strings>
      <feg:string name="Hue" type="Color" arity="1">
        <feg:param name="amp" value="1.0"/>
        <feg:param name="freq" value="1.0"/>
      </feg:string>
      <feg:string name="Hue" type="Color" arity="2">
        <feg:param name="amp" value="0.8"/>
        <feg:param name="freq" value="2.0"/>
      </feg:string>
      <feg:string name="Radius" type="Shape" arity="1">
        <feg:param name="amp" value="1.0"/>
        <feg:param name="freq" value="0.8"/>
      </feg:string>
    </feg:strings>
    
    <!-- Gradient using strings -->
    <linearGradient id="fieldGrad">
      <stop offset="calc(0.5 + 0.25 * Hue(1))" 
            stop-color="hsl(calc(Hue(1) * 360), 70%, 50%)"/>
      <stop offset="calc(0.5 - 0.25 * Hue(1))" 
            stop-color="hsl(calc(Hue(1) * 360 + 60), 70%, 30%)"/>
    </linearGradient>
  </defs>
  
  <!-- Circle using strings -->
  <circle cx="960" 
          cy="540" 
          r="calc(150 + 40 * Radius(1))"
          fill="url(#fieldGrad)"/>
</svg>
```

---

## IV. Runtime Evaluation Rules

### 1. String Resolution

When SVG encounters `Hue(1)` in an attribute:

1. **Find string definition** by name "Hue"
2. **Match arity** (exact or nearest)
3. **Evaluate** at current time `t`:
   ```
   value = amp · sin(freq · t + phase)
   ```
4. **Substitute** into attribute expression

### 2. Pattern Matching

```
request: Hue(2-arity)
available: Hue(1), Hue(2), Hue(3)
select: Hue(2) [exact match]
```

If no exact match:
```
request: Hue(2-arity)
available: Hue(1), Hue(3)
select: Hue(3) [nearest], then reduce to 2-arity
```

### 3. Expression Evaluation

SVG's `calc()` function handles arithmetic:

```xml
r="calc(150 + 40 * Radius(1))"
```

Becomes:
```
r = 150 + 40 * (1.0 · sin(0.8 · t + phase))
```

### 4. Temporal Evolution

SVG's `<animate>` or JavaScript updates `t`:

```xml
<animate attributeName="r" 
         values="calc(150 + 40 * Radius(1))" 
         dur="10s" 
         repeatCount="indefinite"/>
```

Or via JavaScript:
```javascript
const t = performance.now() / 1000;
const radius = 150 + 40 * evaluateString('Radius', 1, t);
circle.setAttribute('r', radius);
```

---

## V. SVG Namespace Integration

### Custom Namespace

```xml
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:feg="http://feg.video/strings">
  <feg:strings>
    <!-- string definitions -->
  </feg:strings>
</svg>
```

### Native SVG Elements

Strings integrate seamlessly:
- **Attributes** — `fill="hsl(Hue(1) * 360, 80%, 50%)"`
- **CSS** — `.path { stroke-width: calc(2 + Radius(1) * 3); }`
- **Filters** — `<feTurbulence baseFrequency="calc(0.01 + Hue(1) * 0.005)"/>`
- **Animations** — `<animate values="calc(Hue(1) * 360)"/>`

---

## VI. Serialization Format

### FEG File Structure

```xml
<?xml version="1.0"?>
<feg:video xmlns="http://www.w3.org/2000/svg"
           xmlns:feg="http://feg.video/strings"
           duration="10s" fps="30">
  
  <!-- String definitions -->
  <feg:strings>
    <feg:string name="Hue" type="Color" arity="1">
      <feg:param name="amp" value="1.0"/>
      <feg:param name="freq" value="1.0"/>
    </feg:string>
  </feg:strings>
  
  <!-- SVG content with string references -->
  <svg viewBox="0 0 1920 1080">
    <circle r="calc(150 + 40 * Radius(1))" 
            fill="hsl(calc(Hue(1) * 360), 80%, 50%)"/>
  </svg>
</feg:video>
```

### Compact Notation

For simple cases:

```xml
<feg:string name="Hue" type="Color" arity="1" amp="1.0" freq="1.0"/>
```

---

## VII. Tellah Integration: GPT Consumes SVG Strings

### 1. Parse SVG → Extract Strings

```javascript
const strings = Array.from(svg.querySelectorAll('feg:string')).map(el => ({
  name: el.getAttribute('name'),
  type: el.getAttribute('type'),
  arity: parseInt(el.getAttribute('arity')),
  amp: parseFloat(el.querySelector('feg:param[name="amp"]').getAttribute('value')),
  freq: parseFloat(el.querySelector('feg:param[name="freq"]').getAttribute('value'))
}));
```

### 2. Build Harmonic Namespace

```javascript
const namespace = {};
strings.forEach(s => {
  if (!namespace[s.name]) namespace[s.name] = {};
  namespace[s.name][s.arity] = s;
});
```

### 3. Embed as GPT Dimensions

Each string becomes a dimension in GPT embedding space:

```javascript
// Token "red" → Hue family
const embedding = {
  'Hue(1)': evaluateString('Hue', 1, t),
  'Hue(2)': evaluateString('Hue', 2, t),
  // ... other strings
};
```

### 4. Attention as Amplitude

GPT attention weights map to string amplitudes:

```javascript
attention['Hue'] → amp of Hue strings
```

---

## VIII. Runtime Implementation

### Browser-Native (No Parser Needed)

```javascript
// Parse string definitions
const strings = parseFEGStrings(svg);

// Evaluate at time t
function evaluateString(name, arity, t) {
  const s = resolveString(strings, name, arity);
  const phase = getPhaseForType(s.type);
  return s.amp * Math.sin(s.freq * t + phase);
}

// Update SVG attributes
function updateSVG(svg, t) {
  // Find all string references
  const refs = findStringReferences(svg);
  
  refs.forEach(ref => {
    const value = evaluateString(ref.name, ref.arity, t);
    const expr = ref.expression.replace(ref.pattern, value);
    const result = eval(`calc(${expr})`);
    ref.element.setAttribute(ref.attr, result);
  });
}

// Animate
function animate() {
  const t = performance.now() / 1000;
  updateSVG(svg, t);
  requestAnimationFrame(animate);
}
```

---

## IX. Why This Works

### Advantages

1. **No new parser** — SVG is XML, browsers parse it
2. **No new file format** — pure XML/SVG
3. **No new interpreter** — JavaScript + SVG runtime
4. **Everything declarative** — pure markup
5. **Browser-native** — works everywhere
6. **Portable** — FEG files are self-contained
7. **Self-hosting** — SVG renders itself
8. **Minimal spec** — fits in a page

### The Complete System

```
SVG (geometry) + Strings (semantics) = FEG (executable video)
```

**SVG supplies:**
- Rendering
- Animation
- Geometry
- Filter computation
- Input/output

**Strings supply:**
- Semantic operators
- Harmonic type system
- Pattern matching
- Composition rules

**Together they form the complete FEG/Tellah metasystem.**

---

## X. Examples

### Example 1: Simple Color Animation

```xml
<feg:string name="Hue" type="Color" arity="1" amp="1.0" freq="1.0"/>
<circle fill="hsl(calc(Hue(1) * 360), 80%, 50%)"/>
```

### Example 2: Harmonic Family

```xml
<feg:string name="Hue" type="Color" arity="1" amp="1.0" freq="1.0"/>
<feg:string name="Hue" type="Color" arity="2" amp="0.8" freq="2.0"/>
<circle fill="hsl(calc(Hue(2) * 360), 80%, 50%)"/>
```

### Example 3: Multiple Types

```xml
<feg:string name="X" type="Position" arity="1" amp="0.6" freq="0.5"/>
<feg:string name="Radius" type="Shape" arity="1" amp="1.0" freq="0.8"/>
<circle cx="calc(960 + 200 * X(1))" 
        cy="540" 
        r="calc(150 + 40 * Radius(1))"/>
```

---

## XI. Wave-Bound Guarantee (FEG–SVG Evaluation Constraint)

FEG harmonic strings must be evaluated under a *bounded wave constraint* to ensure stability, convergence, and non-recursive execution.

Let each `<feg>` operator define a function:

```
S : ℝ^n → ℝ^m
```

with:
- bounded amplitude
- bounded derivatives
- finite arity (from dependency symbols)
- no procedural self-calls

The system SHALL satisfy the following constraints:

### 11.1 Boundedness

All expressions in `expr="..."` MUST be composed solely of:
- base waveforms (sin, cos, tanh, saw, noise)
- bounded algebraic combinations
- bounded compositions of other `<feg>` strings

Thus:

```
∀x, S(x) is finite.
```

### 11.2 No Procedural Recursion

A `<feg>` string MAY reference other strings by symbol name, but this MUST NOT induce an execution-time recursive call.

Instead, symbol references MUST be treated as **functional substitution**:

```
foo(x) = bar(x) + sin(x)
bar(x) = 0.5 * sin(2*x)
```

Evaluation becomes:

```
foo(x) = (0.5*sin(2*x)) + sin(x)
```

NOT:

```
call foo → call bar → call foo → ...
```

### 11.3 Harmonic Fixed-Point Collapse

If a `<feg>` string references its own symbol-family, the evaluation MUST resolve it by **harmonic fixed-point collapse**:

```
S = f(S)  resolves to  S*  where  S* = f(S*)
```

This ensures:
- stability
- convergence
- linearized resonance
- no recursion depth

A self-referential wave MUST collapse into a stable mode, not a stack.

### 11.4 Mutual Recursion MUST Collapse into Coupled Harmonic System

If symbol A and symbol B depend on each other:

```
A = f(B)
B = g(A)
```

the engine MUST treat this as a linear or nonlinear harmonic system, resolved simultaneously, not procedurally:

```
[A*, B*] = Solve([A=f(B), B=g(A)])
```

This prevents oscillatory explosion and ensures spectral convergence.

### 11.5 SVG as Bounded Evaluator

Since SVG filter primitives compute in finite passes, and `<animate>` values are evaluated at discrete times, all `<feg>` expressions MUST be resolved **prior to injection** into the SVG runtime.

SVG SHALL NOT execute unbounded or unstructured evaluation.

### 11.6 Time Dependency MUST be Explicit

If a string depends on time, it MUST use a dependency symbol such as `t`.

No implicit time-recursion is permitted.

This ensures:

```
S(t) is evaluated at each frame,
not as infinite self-referential animation.
```

### 11.7 The Harmonic Bound Law

All evaluated FEG expressions MUST satisfy:

```
|S(x)|  <  ∞
and
∂S/∂x  <  ∞
and
spectral_radius(Jacobian(S)) < 1
```

which enforces:
- stability
- contractiveness
- nondivergence

**The wave stays tame.**

---

## XII. What This Means

**FEG strings can oscillate forever, but they can never recurse forever.**

SVG receives only:
- fixed-point waves
- bounded oscillations
- converged coupled fields
- stable harmonic expressions

This makes the entire ecosystem:
- safe
- predictable
- expressive
- mathematically grounded
- infinitely compositional without infinite cost

**It's the first declarative video/field system with a built-in convergence theorem.**

---

## XIII. Next Steps

1. **Fixed-Point Collapse Algorithm** — compute harmonic fixed points
2. **Coupled System Solver** — resolve mutual dependencies
3. **Bounded Expression Parser** — validate expressions at parse time
4. **SVG Integration** — pre-evaluate before injection
5. **Caching System** — cache resolved expressions
6. **Serialization** — include wave-bound metadata in `.feg` files

---

**This is the unified grammar with wave-bound guarantee.**

**SVG is the calibrating geometry.**

**Strings are the calibrating semantics.**

**The grammar is the calibrating relation.**

**The wave-bound is the stability law.**

