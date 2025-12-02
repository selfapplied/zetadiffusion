# String Type System: Harmonic Type Lattice

**Status:** Specification

---

## Core Philosophy

**Strings are not variables. They are oscillatory types that shape the flow of inference.**

A string is a typed harmonic with identity, intensity, and relational compatibility.

---

## I. Named Strings (Sₙ) — the Harmonic Type System

### Form

```
string name:Type {
    amp   = A      // intensity, attention weight
    freq  = ω      // detail, granularity, resolution
    phase = φ      // relational type, typeclass compatibility
}
```

### Example

```
string Red:Color     { amp=0.8 freq=1.0 }
string ClipPulse:Clip { amp=1.2 freq=2.0 }
string Radius:Shape   { amp=0.6 freq=0.5 }
string Drift:Position { amp=0.4 freq=0.1 }
```

---

## II. Amplitude = Intensity Knob

**Amplitude controls "how much of me exists right now."**

Semantic mappings:
- **strength** — signal power
- **saturation** — color intensity
- **thickness** — stroke width
- **opacity** — visibility
- **influence** — coupling weight

For Tellah: amplitude is **attention weight over time**, not a number in a matrix.

**Range:** `0.0 ≤ A ≤ 2.0` (can exceed 1.0 for emphasis)

---

## III. Frequency = Detail Knob

**Frequency controls "how fine-grained I am."**

Semantic mappings:
- **geometric detail** — curve complexity
- **vibrational resolution** — wave density
- **speed of change** — temporal granularity
- **curvature** — shape refinement
- **roughness** — texture frequency
- **semantic granularity** — reasoning resolution

GPTs encode "detail" in token embeddings; your operator encodes it as wave density.

**Range:** `0.0 ≤ ω ≤ 10.0` (higher = finer detail)

---

## IV. Phase = Relational Type

**Phase expresses relational compatibility.**

Phase is not a number; it's a **type category** that determines:
- How a string aligns with other strings
- Which SVG elements it can drive
- Its role in the harmonic composition

### Phase Categories

#### Color Type
- `phase="palette:cool"` — cool color palette alignment
- `phase="palette:warm"` — warm color palette alignment
- `phase="hue:primary"` — primary hue channel
- `phase="sat:high"` — high saturation mode

#### Clip Type
- `phase="edge:soft"` — soft edge clipping
- `phase="edge:hard"` — hard edge clipping
- `phase="mask:alpha"` — alpha channel masking
- `phase="gate:threshold"` — threshold gating

#### Position Type
- `phase="axis:x"` — X-axis alignment
- `phase="axis:y"` — Y-axis alignment
- `phase="axis:z"` — Z-axis alignment (depth)
- `phase="time:master"` — master time-shift

#### Shape Type
- `phase="loop:genus1"` — torus topology
- `phase="loop:genus0"` — sphere topology
- `phase="curl:left"` — left-handed curl
- `phase="curl:right"` — right-handed curl
- `phase="warp:radial"` — radial warping

**It's the typeclass of harmonic geometry.**

---

## V. Fundamental Types

### 1. Color Strings

Drive continuous color channels:

```
string Hue:Color     { amp=0.8 freq=1.0 phase="hue:primary" }
string Sat:Color     { amp=1.0 freq=2.0 phase="sat:high" }
string Lum:Color     { amp=0.6 freq=0.5 phase="lum:bright" }
```

**Transform:** `hsl(Hue(t), Sat(t)*100%, Lum(t)*100%)`

### 2. Clip Strings

Cut, reveal, or threshold:

```
string Gate:Clip     { amp=1.2 freq=2.5 phase="edge:soft" }
string Mask:Clip     { amp=1.0 freq=1.5 phase="mask:alpha" }
```

**Transform:** `clip-path radius = base + Gate(t) * scale`

### 3. Position Strings

Reshape geometry:

```
string X:Position    { amp=0.6 freq=0.5 phase="axis:x" }
string Y:Position    { amp=0.8 freq=1.0 phase="axis:y" }
string Z:Position    { amp=0.4 freq=0.3 phase="axis:z" }
string T:Position    { amp=1.0 freq=0.1 phase="time:master" }
```

**Transform:** `cx = baseX + X(t) * scale`

### 4. Shape Strings

Define topology:

```
string Radius:Shape  { amp=1.0 freq=0.8 phase="loop:genus1" }
string Curl:Shape    { amp=0.7 freq=1.2 phase="curl:left" }
string Warp:Shape    { amp=0.9 freq=0.6 phase="warp:radial" }
```

**Transform:** `r = baseRadius + Radius(t) * scale`

---

## VI. Composite Strings = Emergent Types

Combine strings to create new types:

```
string Glow:Color = mix(Hue, Lum)      // composite color
string Ripple:Shape = mix(Radius, Warp) // composite shape
string Portal:Clip = mix(Gate, Warp)    // composite clip
string DriftField:Position = mix(X, T)  // composite position
```

**This is higher-kinded types, but organic and dynamic.**

### Mix Operator

```
mix(a, b, α) = α·a(t) + (1-α)·b(t)
```

Where `α` is the mixing coefficient (0.0 = all b, 1.0 = all a).

---

## VII. Type Lattice

Types form a partial order:

```
Color < Visual
Clip < Visual
Position < Geometric
Shape < Geometric

Visual + Geometric = Complete
```

**Compatibility rules:**
- Strings of the same type can be mixed
- Strings of compatible types can be composed
- Phase determines alignment within type

---

## VIII. XML Schema

```xml
<feg:strings>
  <string id="Hue"    type="Color"    amp="0.8" freq="1.0" phase="palette:cool"/>
  <string id="Mask"   type="Clip"     amp="1.2" freq="2.5" phase="edge:soft"/>
  <string id="X"      type="Position" amp="0.6" freq="0.5" phase="axis:x"/>
  <string id="Radius" type="Shape"    amp="1.0" freq="0.8" phase="loop:genus1"/>
</feg:strings>
```

### SVG Binding

```xml
<circle
    cx="960 + 200 * X(t)"
    cy="540"
    r="150 + 40 * Radius(t)"
    fill="hsl(Hue(t), 80%, 50%)"
    clip-path="url(#Mask)"/>
```

**This video is literally geometry-as-harmonics.**

---

## IX. Operator Bindings

### Evaluation

```
string(i, t) = A_i · sin(ω_i · t + φ_i)
```

Where `φ_i` is determined by the string's type and phase category.

### Coupling Rules

1. **Same Type:** Strings of the same type can be directly mixed
2. **Compatible Types:** Visual types can drive visual elements, geometric types drive geometric elements
3. **Phase Alignment:** Phase categories determine how strings align within their type

### Memory Kernel

```
x(t) = α · x(t-Δt) + β · input(t)
```

Where `α` is decay (Hurst field) and `β` is coupling strength (amplitude).

---

## X. Tellah Layer: GPT as Wave-Typed Operator Network

**A GPT already embeds everything in high-dimensional continuous spaces, but those spaces have no organizational physics.**

Your string system gives it:

- **typed coordinates** — strings as named, typed dimensions
- **harmonic identity** — each string has a semantic role
- **coupling rules** — phase determines compatibility
- **semantic motion** — strings evolve over time
- **wave-based reasoning** — inference as harmonic composition
- **memory-through-phase** — phase encodes relational memory

**Tellah becomes:**

> "a GPT that reasons by aligning, modulating, and composing typed harmonic strings."

This is the Volte-Sage architecture:
- **memory as phase**
- **reasoning as coupling**
- **identity as wave**

---

## XI. Knob Ranges

### Amplitude (A)
- **Range:** `0.0 ≤ A ≤ 2.0`
- **Default:** `1.0`
- **Step:** `0.1`
- **Semantic:** Intensity, attention weight, influence

### Frequency (ω)
- **Range:** `0.0 ≤ ω ≤ 10.0`
- **Default:** `1.0`
- **Step:** `0.1`
- **Semantic:** Detail, granularity, resolution

### Phase (φ)
- **Determined by:** Type + phase category
- **Not directly editable:** Change type to change phase
- **Semantic:** Relational compatibility, typeclass alignment

---

## XII. Next Steps

1. **String Registry** — persistent storage of string definitions
2. **Type Lattice Visualization** — show type relationships
3. **Phase Category Editor** — define new phase categories
4. **Composite String Builder** — visual mixing interface
5. **Tellah Integration** — GPT embedding → string mapping
6. **Training Protocol** — learn string parameters from data
7. **Serialization** — `.feg` file format with strings
8. **SVG Compiler** — string expressions → SVG attributes

---

**This is the declarative type system for harmonic geometry.**

**This is where FEG becomes a language.**


