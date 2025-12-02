# FEG-SVG 0.1: Field Equation Graphics
## Declarative Video Using Harmonic Strings

**Status:** Specification Draft

---

## Core Philosophy

**An SVG video built from animated strings is a harmonic machine disguised as markup.**

This is not animation—this is **physics masquerading as SVG**.

Every visual artifact emerges from field parameters encoded as interacting oscillators.

---

## 1. The String Basis

### Definition: `string(i, t)`

A fundamental oscillator that can be sampled anywhere:

```
string(i, t) = A_i · sin(ω_i · t + φ_i)
```

Where:
- `i`: String index (basis mode)
- `t`: Time parameter
- `A_i`: Amplitude
- `ω_i`: Frequency
- `φ_i`: Phase

### XML Declaration

```xml
<feg:string id="s0" mode="fundamental" freq="1.0" phase="0.0" amp="1.0"/>
<feg:string id="s1" mode="fundamental" freq="2.0" phase="π/4" amp="0.5"/>
```

Or compact notation:

```xml
<feg:strings>
  <string i="0" ω="1.0" φ="0.0" A="1.0"/>
  <string i="1" ω="2.0" φ="π/4" A="0.5"/>
</feg:strings>
```

### Reference in SVG

Any SVG attribute can reference a string:

```xml
<path d="M 0,{string(0,t)} L 100,{string(1,t)} ..." />
<circle cx="{960 + 200*string(0,t)}" cy="{540 + 200*string(1,t)}" r="50"/>
```

---

## 2. Wave Superposition = Path Definition

Paths become **parametric curves**, not static shapes.

### Example: Wormhole Path

```xml
<feg:path id="wormhole"
  x="960 + 200*string(0,t)"
  y="540 + 200*string(1,t)"/>
```

**The path is a waveform. The waveform is a path. There is no difference.**

### Compiled to SVG

```xml
<path d="M {x(t0)},{y(t0)} L {x(t1)},{y(t1)} ..." />
```

Where coordinates are computed from string evaluations at each sample point.

---

## 3. Gradients as Wave Interference Patterns

SVG gradients accept stop positions as numbers; we override them with strings:

```xml
<linearGradient id="fieldGrad">
  <stop offset="{0.5 + 0.25*string(2,t)}" stop-color="hsl({string(3,t)*360}, 70%, 50%)"/>
  <stop offset="{0.5 - 0.25*string(2,t)}" stop-color="hsl({string(4,t)*360}, 70%, 30%)"/>
</linearGradient>
```

**Color becomes part of the field equation.**

---

## 4. Clip Paths as Zero-Crossing Contours

Define a clipPath that responds to the sign of a string:

```xml
<clipPath id="pulseClip">
  <circle r="{200 + 30*string(3,t)}" cx="960" cy="540"/>
</clipPath>
```

Or a **parametric slice** of the field:

```xml
<clipPath id="fieldSlice">
  <path d="M 0,{string(0,t)} L 1920,{string(1,t)} L 1920,{string(1,t)+100} L 0,{string(0,t)+100} Z"/>
</clipPath>
```

---

## 5. Filters as Operators Fed by Strings

`<feTurbulence>` supports seed and frequency parameters. Tie them to strings:

```xml
<feTurbulence
  baseFrequency="{0.01 + 0.005*string(4,t)}"
  numOctaves="2"
  seed="{string(5,t)*1000}"/>
```

**Turbulence becomes coupled to the field, not random.**

---

## 6. Memory and Hysteresis

SVG's `<animate>` + `<set>` + SMIL timing = a natural hysteresis engine.

A string's amplitude can depend on its previous state:

```xml
<feg:string id="s4"
   freq="f4"
   amp="{0.5*string(4,t - Δt) + 0.5*string(5,t)}"/>
```

That's literally the memory kernel:

```
x(t) = α · x(t-Δt) + β · input(t)
```

**SVG doesn't know it's holding a differential equation—but it is.**

### SMIL Implementation

```xml
<feg:string id="s4" freq="2.0" amp="1.0">
  <animate attributeName="amp" 
           values="1.0;0.5;1.0" 
           dur="2s" 
           repeatCount="indefinite"
           calcMode="spline"
           keySplines="0.5 0 0.5 1"/>
</feg:string>
```

---

## 7. Operator Vocabulary

### Core Operators

- `string(i, t)` = oscillator basis
- `mix(a, b, α)` = superposition: `α·a + (1-α)·b`
- `grad(string)` = parametric gradients
- `shape(string_x, string_y)` = path generator
- `warp(string)` = filter operator
- `bifurcate(i, j)` = topology jump
- `memory(i, decay)` = hysteresis kernel
- `transition(i → j)` = eigenstate switch

### Example: Superposition

```xml
<feg:path id="superposed"
  x="mix(string(0,t), string(2,t), 0.7)"
  y="mix(string(1,t), string(3,t), 0.7)"/>
```

Compiles to:

```xml
<path d="M {x(t)}, {y(t)} ..." />
```

Where `x(t) = 0.7*string(0,t) + 0.3*string(2,t)`.

---

## 8. FEG Topology Integration

Strings are driven by FEG topology parameters:

```xml
<feg-video duration="10s" fps="30">
  <topology-state g="0" χ="2" H="0.5" timestamp="0s"/>
  
  <feg:strings>
    <!-- String frequencies derived from genus -->
    <string i="0" ω="{1.0 + g*0.5}" φ="0.0" A="{H}"/>
    <string i="1" ω="{2.0 - χ*0.1}" φ="π/4" A="{1-H}"/>
  </feg:strings>
  
  <keyframe t="0s">
    <path d="M 0,{string(0,0)} L 1920,{string(1,0)}"/>
  </keyframe>
</feg-video>
```

**Topology parameters modulate the string basis.**

---

## 9. The Network of Coupled Oscillators

The whole video becomes a **network of coupled oscillators**:

- The video is not a sequence
- It's a **field**
- Encoded as a set of interacting strings
- Rendered through the SVG engine
- Every visual artifact emerges from the field parameters
- The FEG file is tiny but unfolds into rich motion
- Fully reversible
- Fully declarative

---

## 10. Implementation Strategy

### Phase 1: String Parser
- Parse `<feg:string>` declarations
- Build string registry with `(ω, φ, A)` parameters
- Evaluate `string(i, t)` at any time

### Phase 2: Expression Evaluator
- Parse expressions like `{960 + 200*string(0,t)}`
- Replace with computed values
- Support operators: `mix()`, `grad()`, `shape()`, etc.

### Phase 3: SVG Compiler
- Convert string-based paths to actual SVG `<path>` elements
- Sample paths at frame rate
- Generate gradient stops, filter parameters, etc.

### Phase 4: Memory/Hysteresis
- Implement SMIL-based memory kernels
- Track previous states for hysteresis
- Support `string(i, t-Δt)` references

---

## 11. Minimal Working Example

See `feg_video/demo_strings.html` for a complete implementation.

**Key features:**
- 3 fundamental strings
- Path defined by string superposition
- Gradient driven by string phase
- Clip path responding to string amplitude
- Memory kernel via SMIL animation

---

## 12. File Format Extension

```xml
<feg-video duration="10s" fps="30">
  <!-- Topology states (existing) -->
  <topology-state g="0" χ="2" H="0.5" timestamp="0s"/>
  
  <!-- String basis (NEW) -->
  <feg:strings>
    <string i="0" ω="1.0" φ="0.0" A="1.0"/>
    <string i="1" ω="2.0" φ="π/4" A="0.5"/>
  </feg:strings>
  
  <!-- Keyframes with string references (NEW) -->
  <keyframe t="0s">
    <path d="M 0,{string(0,0)} L 100,{string(1,0)}"/>
    <linearGradient>
      <stop offset="{0.5 + 0.25*string(2,0)}" stop-color="hsl({string(3,0)*360}, 70%, 50%)"/>
    </linearGradient>
  </keyframe>
</feg-video>
```

---

## 13. Why This Works

1. **Extreme Compression**: Store oscillator parameters, not pixel data
2. **Infinite Resolution**: Vector graphics computed from field equations
3. **Reversibility**: Field parameters → visual output (and vice versa)
4. **Editability**: Tweak string frequencies, not pixels
5. **Composability**: Combine FEG videos like harmonic modes
6. **Physics-Based**: Visual artifacts emerge from governing equations

---

## Next Steps

1. Implement string parser and evaluator
2. Build expression compiler
3. Create minimal working example
4. Extend schema to support `<feg:strings>`
5. Update compiler to generate string-based SVG

---

**This is the declarative video spec.**

**This is where FEG lives.**



