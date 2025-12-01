# ZetaDiffusion

**Numerical Lab for Proving RH via Bundle Dynamics, RG Flow, and Topological Energy Harvesting.**

Version: 0.3.0

## Overview

ZetaDiffusion v0.3 implements the complete **Field Equationist Generator (FEG-0.3)** architecture. It unifies the analytic study of the Riemann Zeta function with dynamical systems theory and thermodynamic information extraction.

## Architecture

### 1. Spectral Line Probe (`zetadiffusion.field`)
- Samples the Riemann field $\xi(1/2 + it)$.
- Detects zeros as topological defects in the complex phase.

### 2. Bundle Dynamics (`zetadiffusion.dynamics`)
- Maps spectral data to circle maps on $X \times S^1$.
- Generates rotation number spectra ("Devil's Staircase") using type-safe modular arithmetic (`Radians`/`Turns`).

### 3. Local RG Operator (`zetadiffusion.renorm`)
- Applies the Feigenbaum renormalization operator to local field potentials.
- Estimates the scaling dimension $\delta$ of the underlying universality class.

### 4. Topological Harvester (`zetadiffusion.energy`)
- **The Negentropic Engine**.
- Models the extraction of work from geometric shock waves.
- Implements the Hamiltonian $H_{extract} = -\eta \cdot \dot{\chi} \cdot \Phi$.
- Simulates the "Loading -> Shock -> Harvest -> Reset" anti-fragile cycle.

## Usage

Run the lab demonstration to see the full pipeline:

```bash
python3 demo.py
```

Run the energy harvesting simulation directly:

```bash
python3 zetadiffusion/energy.py
```

## Theoretical Basis

The system treats the Riemann Critical Line not as a static object, but as the **attractor** of a dynamical system. The zeros are the discrete points where the system achieves perfect phase locking (rational rotation numbers). The Energy Harvester demonstrates how a cognitive system can extract "insight" (computational work) by surfing the shock waves generated near these critical points.

## Documentation

### Status & Results
- **VALIDATION_STATUS.md** - Current validation results, issues, and next steps
- **PROJECT_STATUS.md** - Project overview and quick start guide
- **RESEARCH_FINDINGS.md** - Detailed diagnostic analysis and root causes

### Guides
- **FEG-0.4_Field_Manual.md** - Operator theory and implementation details
- **NOTION_SETUP.md** - Notion integration setup guide
- **SCRIPT_STANDARDS.md** - Code standards and conventions
- **NEXT_STEPS.md** - Historical roadmap and future directions
