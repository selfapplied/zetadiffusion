# The Calibrating Grammar: Three Rules

**Status:** Core Specification

---

## The Irreducible Algebra

A string is defined by:
1. **a symbol** (its name)
2. **a set of dependency symbols** (its arity / its inputs)
3. **an expression** using those dependencies

**That expression *is itself* a new string-type** that enters the global type set.

---

## Formal Definition

```
String := ( symbol, deps, expr )
```

Where:
- **symbol** is the name
- **deps** is a finite ordered set of symbols
- **expr** is some expression built from deps and other strings

This captures everything:
- arity = |deps|
- phase/type = inferred from context
- frequency = structural arity
- resonance = name matching
- compositionality = dependency graph
- semantics = expr

**Elegant. No knobs. No metadata.**

---

## The Three Rules

### Rule 1 — Definition

```
A string is a named expression over dependency symbols.
```

### Rule 2 — Expansion

```
Each expression produces a new string-type in the symbol's type family.
```

### Rule 3 — Alignment

```
Strings unify by matching symbols;
strings dispatch by matching dependency patterns.
```

---

## From These Three Axioms

You get:
- polymorphism
- resonance
- harmonic complexity
- type inference
- dynamic dispatch
- recursive compositionality
- self-extending language
- executable FEG video
- Tellah wave-based reasoning

**Everything else is emergent behavior.**

---

## Examples

### Simple String

```
Hue(x) := sin(x)
```

Creates type `Hue(1)` with dependency `{x}`.

### Multi-Dependency String

```
Hue(x,y) := sin(x) + 0.5*sin(y*2)
```

Creates type `Hue(2)` with dependencies `{x, y}`.

### Composed String

```
Glow(x,y) := Hue(x) + Lum(y)
```

Creates type `Glow(2)` that composes `Hue(1)` and `Lum(1)`.

### Complex Expression

```
Ripple(t,a,b) := Radius(t) * Warp(a,b) + Curl(t*2)
```

Creates type `Ripple(3)` with dependencies `{t, a, b}`.

---

## SVG Representation

```xml
<feg-string symbol="Hue" deps="x,y" expr="sin(x) + 0.5*sin(y*2)"/>
```

Usage:

```xml
<circle fill="hsl(calc(Hue(x,y)*360), 80%, 50%)"/>
```

---

## The Dependency Set IS the Frequency

You don't need:
- explicit "frequency"
- explicit "complexity"
- explicit "arity"
- explicit "typeclass"

It all falls out from:

```
deps = {x, y, z}
```

So:
- |deps| = 3 → tri-harmonic string
- order/dependency shape = harmonic signature
- symbol hierarchy = semantic root
- expression = internal recipe

**Frequency in physics is "how many cycles are interacting."**

**Here, frequency = "how many dependency symbols are interacting."**

---

## Tellah Embedding Rule

Tellah can embed each string as:
- **symbol** → embedding centroid
- **deps** → embedding axes
- **expr** → operator
- **type** → harmonic cluster ID
- **frequency** → tensor dimensionality
- **resonance** → dot-product / cosine similarity in harmonic space

**What GPTs do with tokens, Tellah does with *waves*.**

---

## Self-Extending Type System

Every new expression is a new string-type:

```
Hue(x,y)
Hue(x,y,z)
Hue(x) + Tint(y)
Mask(a,b).smooth()
Radius(t) * Warp(a,b,c)
```

Each definition:
- expands the type universe
- creates a new node in the harmonic graph
- adds a new arity-pattern to the symbol family
- enriches the semantic field

**The system becomes *self-calibrating* as more strings accumulate.**

**This is how a living language evolves.**

---

## Why This Works

This mirrors:
- OPIC operator families
- CE1 bracket expansions
- GPT token unification
- functional overloading
- type inference in ML/OCaml
- category theory (objects + arrows)
- musical harmonics (fundamental + overtones + combinations)

**But yours is simpler and more primitive.**

---

**This is the core.**

**This is the irreducible algebra.**

**Everything else is emergent.**


