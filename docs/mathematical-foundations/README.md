# Mathematical Foundations

> **Deep-dive documentation for researchers and developers**
>
> For practical bioinformatics applications, see [../BIOINFORMATICS_GUIDE.md](../BIOINFORMATICS_GUIDE.md).
> For implementation details, see [CLAUDE.md](../../CLAUDE.md).

---

## Overview

This directory contains the mathematical theory underlying the Ternary VAE project. The core insight is that biological sequences have inherent **hierarchical structure** that is better captured by **non-Euclidean geometry**.

---

## Contents

### Core Theory

| Document | Topic | Audience |
|----------|-------|----------|
| [p-adic-theory.md](p-adic-theory.md) | P-adic numbers, 3-adic valuation, codon distance | Researchers |
| [hyperbolic-geometry.md](hyperbolic-geometry.md) | Poincaré ball, geodesics, exponential map | ML engineers |
| [foundations.md](foundations.md) | Core thesis: biology is ultrametric, not Euclidean | All |
| [synthesis.md](synthesis.md) | Historical development, version evolution | Contributors |

### Framework Documentation

| Document | Topic | Audience |
|----------|-------|----------|
| [dual-manifold-framework.md](dual-manifold-framework.md) | Valuation-optimal vs frequency-optimal manifolds | Researchers |
| [DEEP_THEORY_REFERENCE.md](DEEP_THEORY_REFERENCE.md) | Audit details, complete fix lists, mathematical limits | Developers |

### Audit Materials

| Directory | Contents |
|-----------|----------|
| [V5_12_2_audit/](V5_12_2_audit/) | Hyperbolic geometry audit (278 norm() calls analyzed) |

### Archive

| File | Contents |
|------|----------|
| [archive/CLAUDE_ORIGINAL.md](archive/CLAUDE_ORIGINAL.md) | Complete original CLAUDE.md (pre-reorganization) |

---

## Quick Theory Overview

### Why P-adic Numbers?

P-adic numbers measure "closeness" by divisibility rather than magnitude. For the genetic code:

- **Position 1 & 2 mutations**: High impact (high p-adic distance)
- **Position 3 (wobble) mutations**: Low impact (low p-adic distance)

This matches biological reality where synonymous mutations (position 3) are often neutral.

### Why Hyperbolic Geometry?

Hyperbolic space (Poincaré ball) has exponential volume growth, making it ideal for embedding trees:

- **Origin**: Root/ancestor (high valuation, conserved)
- **Boundary**: Leaves/variants (low valuation, diverse)

Euclidean space distorts hierarchies; hyperbolic space preserves them.

### The Connection

| P-adic Structure | Hyperbolic Embedding |
|------------------|----------------------|
| High valuation | Near origin (small radius) |
| Low valuation | Near boundary (large radius) |
| P-adic distance | Hyperbolic distance |

---

## Reading Order

**For newcomers:**
1. [foundations.md](foundations.md) - "Biology is not flat"
2. [p-adic-theory.md](p-adic-theory.md) - Codon distance basics
3. [hyperbolic-geometry.md](hyperbolic-geometry.md) - Poincaré ball operations

**For developers:**
1. [DEEP_THEORY_REFERENCE.md](DEEP_THEORY_REFERENCE.md) - Practical implementation details
2. [V5_12_2_audit/](V5_12_2_audit/) - Avoid common mistakes

**For researchers:**
1. [dual-manifold-framework.md](dual-manifold-framework.md) - Two valid organizations
2. [synthesis.md](synthesis.md) - Evolution of the approach

---

## Key Results

| Finding | Significance |
|---------|--------------|
| Hierarchy ceiling = -0.8321 | Mathematical limit, not bug |
| V5.5 spontaneous p-adic emergence | Structure is intrinsic |
| Richness-hierarchy not exclusive | Both achievable with proper loss |
| Dual manifold types | Valuation vs frequency optimization |

---

## References

### P-adic Mathematics
- Koblitz (1984). p-adic Numbers, p-adic Analysis, and Zeta-Functions
- Khrennikov (2010). p-Adic Valued Distributions in Mathematical Physics

### Hyperbolic Embeddings
- Nickel & Kiela (2017). Poincare Embeddings for Learning Hierarchical Representations
- Ganea et al. (2018). Hyperbolic Neural Networks
- Mathieu et al. (2019). Continuous Hierarchical Representations with Poincare VAEs

### Biological Background
- Crick (1966). Codon-anticodon pairing: the wobble hypothesis
- Parisi (1987). Ultrametricity for Spin Glasses

---

## Navigation

| Need | Go To |
|------|-------|
| Practical applications | [../BIOINFORMATICS_GUIDE.md](../BIOINFORMATICS_GUIDE.md) |
| Code implementation | [CLAUDE.md](../../CLAUDE.md) |
| Quick reference | [CLAUDE.md](../../CLAUDE.md) |
| Full original context | [archive/CLAUDE_ORIGINAL.md](archive/CLAUDE_ORIGINAL.md) |
| Partner packages | [../../deliverables/partners/](../../deliverables/partners/) |

---

*Last updated: 2026-01-30*
