# Composed Types: Strings as Type Constructors

**Status:** Specification

---

## Core Insight

**Each string definition is a new composed type.**

Strings aren't just oscillators—they're **type constructors** that create new types in the harmonic type system.

---

## I. Strings as Type Constructors

### Definition

When you define a string:

```xml
<feg:string name="Hue" type="Color" arity="2">
  <feg:param index="0" amp="1.0" freq="2.0"/>
  <feg:param index="1" amp="0.8" freq="1.5"/>
</feg:string>
```

You're defining a **new type**: `Hue(2)`

This type has:
- **Name**: `Hue` (the type constructor)
- **Arity**: `2` (number of dependencies)
- **Phase**: `Color` (determined by type category)
- **Parameters**: Each dependency has its own freq/amp

### Type Notation

```
type Hue(1) = Color
type Hue(2) = Color
type Radius(1) = Shape
type Radius(2) = Shape
```

Each is a **distinct type** in the harmonic type lattice.

---

## II. Type Families

### Same Name = Type Family

```
type Hue(1) = Color  // 1-arity type
type Hue(2) = Color  // 2-arity type (different type!)
type Hue(3) = Color  // 3-arity type (different type!)
```

These are **different types** that share:
- Same type constructor name (`Hue`)
- Same phase category (`Color`)
- Different arity (different complexity)

### Type Family Structure

```
Hue {
  Hue(1): Color  // simple color cycle
  Hue(2): Color  // two-color interference
  Hue(3): Color  // three-color palette
}
```

---

## III. Type Composition

### Mixing Types

You can compose types:

```
type Glow = mix(Hue(1), Lum(1))  // composite type
type Ripple = mix(Radius(1), Warp(1))  // composite type
```

### Type Promotion/Reduction

```
Hue(1) → Hue(2)  // promote (add dependency)
Hue(3) → Hue(2)  // reduce (remove dependency)
```

### Type Unification

Pattern matching unifies types:

```
request: Hue(2)
available: Hue(1), Hue(2), Hue(3)
unify: Hue(2) ✓
```

---

## IV. Type Lattice

### Partial Order

Types form a lattice:

```
Hue(1) < Hue(2) < Hue(3)  // complexity order
Color < Visual              // phase hierarchy
Shape < Geometric           // phase hierarchy
```

### Type Compatibility

- **Same family, same arity**: Exact match
- **Same family, different arity**: Compatible (can promote/reduce)
- **Different family, same phase**: Compatible (can mix)
- **Different phase**: Incompatible (unless explicitly composed)

---

## V. Type Evaluation

### Type as Function

Each type evaluates to a value:

```
Hue(2)(t) = (A₁·sin(ω₁·t + φ₁) + A₂·sin(ω₂·t + φ₂)) / 2
```

Where:
- `A₁, A₂` are amplitudes of dependencies
- `ω₁, ω₂` are frequencies of dependencies
- `φ₁, φ₂` are phases (staggered)

### Type Application

In SVG:

```xml
<circle fill="hsl(calc(Hue(2)(t) * 360), 80%, 50%)"/>
```

The type `Hue(2)` is applied at time `t`.

---

## VI. Type System Properties

### 1. Type Identity

Each string definition creates a **unique type**:
- `Hue(1)` ≠ `Hue(2)` (different types)
- `Hue(1)` ≠ `Radius(1)` (different families)

### 2. Type Families

Types with the same constructor form a **family**:
- `Hue(1)`, `Hue(2)`, `Hue(3)` are all in the `Hue` family

### 3. Type Composition

Types can be composed:
- `mix(Hue(1), Hue(2))` creates a new type
- `mix(Hue(1), Radius(1))` creates a composite type

### 4. Type Matching

Pattern matching finds compatible types:
- Exact match: `Hue(2)` → `Hue(2)`
- Nearest match: `Hue(2)` → `Hue(3)` (then reduce)

---

## VII. XML Representation

### Type Definition

```xml
<feg:string name="Hue" type="Color" arity="2">
  <!-- This defines type Hue(2) -->
  <feg:param index="0" amp="1.0" freq="2.0"/>
  <feg:param index="1" amp="0.8" freq="1.5"/>
</feg:string>
```

### Type Usage

```xml
<circle fill="hsl(calc(Hue(2)(t) * 360), 80%, 50%)"/>
```

---

## VIII. Type System for Tellah

### GPT Embedding

Each type becomes a dimension:

```javascript
embedding = {
  'Hue(1)': Hue(1)(t),
  'Hue(2)': Hue(2)(t),
  'Radius(1)': Radius(1)(t),
  // ... all defined types
}
```

### Type Inference

Tellah can infer types from context:
- "red circle" → `Hue(1)` + `Radius(1)`
- "colorful shape" → `Hue(2)` + `Radius(2)`

### Type Learning

Tellah can learn new types:
- Observe pattern → define new type
- Generalize from examples → create type family

---

## IX. Why This Matters

### 1. Type Safety

Each string is a **typed value**, not just a number.

### 2. Type Composition

Types can be combined, mixed, promoted, reduced.

### 3. Type Families

Related types form families with shared semantics.

### 4. Type Matching

Pattern matching finds compatible types automatically.

### 5. Type Learning

Tellah can learn and extend the type system.

---

## X. Example: Type System in Action

### Define Types

```xml
<feg:string name="Hue" type="Color" arity="1">
  <feg:param index="0" amp="1.0" freq="1.0"/>
</feg:string>

<feg:string name="Hue" type="Color" arity="2">
  <feg:param index="0" amp="1.0" freq="2.0"/>
  <feg:param index="1" amp="0.8" freq="1.5"/>
</feg:string>
```

This creates:
- `type Hue(1) = Color`
- `type Hue(2) = Color`

### Use Types

```xml
<!-- Use exact type -->
<circle fill="hsl(calc(Hue(1)(t) * 360), 80%, 50%)"/>

<!-- Pattern matching finds Hue(2) -->
<circle fill="hsl(calc(Hue(2)(t) * 360), 80%, 50%)"/>
```

### Compose Types

```xml
<!-- Mix two types -->
<circle fill="hsl(calc(mix(Hue(1), Hue(2), 0.5)(t) * 360), 80%, 50%)"/>
```

---

## XI. The Complete Picture

**Strings are type constructors.**

Each string definition:
1. Creates a new type in the harmonic type system
2. Defines the type's structure (arity, dependencies)
3. Specifies the type's phase (Color, Shape, Position, Clip)
4. Provides the type's evaluation function

**The type system is built from these composed types.**

---

**This is the harmonic type system.**

**Each string is a new composed type.**

**The type lattice emerges from string definitions.**


