# Ultrametric Antigen AI

**High-throughput mutation screening 300-18,000x faster than physics-based methods**

[![Version](https://img.shields.io/badge/version-5.12.5-blue.svg)](docs/mathematical-foundations/README.md)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-lightgrey.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-101%2B%20passing-brightgreen.svg)](deliverables/tests/)

---

## What We Do

We accelerate drug discovery and pathogen surveillance by screening millions of mutations in seconds - work that takes physics-based tools hours or days.

| Capability | Our Speed | Traditional (FoldX/Rosetta) | Advantage |
|------------|-----------|----------------------------|-----------|
| **Protein stability (DDG)** | <0.1 sec | 30 sec - 30 min | 300-18,000x faster |
| **No structure required** | Sequence only | Requires PDB structure | Lower cost, broader applicability |
| **Complementary detection** | Finds Rosetta-blind mutations | Misses 23.6% of cases | Orthogonal signal |

**Applications**: Antimicrobial peptide design, protein engineering, arbovirus surveillance, HIV drug resistance

---

## Validated Results

### Production-Ready Partner Packages

| Package | Application | Performance | Validation | Status |
|---------|-------------|-------------|------------|--------|
| **Antimicrobial Peptides** | Design AMPs for WHO priority pathogens | r=0.656 (5 models, all p<0.001) | 5-fold CV, N=425 | 90% Ready |
| **Protein Stability** | Predict mutation effects (DDG) | LOO ρ=0.521 (N=52) | Bootstrap CI: [0.21, 0.80] | 95% Ready |
| **Arbovirus Surveillance** | Pan-arbovirus primer design | 7 viruses, CDC recovery 60% | Wet-lab primer validation | 90% Ready |
| **HIV Research** | Drug resistance screening | 200K sequences analyzed | Stanford HIVdb integration | Complete |

### Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **AMP General Model** | r=0.608 (p=2.4e-44) | Outperforms sklearn baseline (0.56) |
| **DDG Prediction** | ρ=0.521 LOO-CV | Comparable to BLOSUM (0.41), faster than FoldX |
| **Force Constants** | r=0.86 | Codon encoder physical properties |
| **Compression** | 1,230x | 19,683 operations → 16-d hyperbolic space |

---

## Honest Limitations

We believe transparency builds trust. Here's where we stand:

| Limitation | Details | Mitigation |
|------------|---------|------------|
| **DDG benchmark gap** | Full S669 (N=669): ρ=0.37-0.40, below ESM-1v (0.51) | Use as fast pre-screen, refine with physics tools |
| **S. aureus AMP model** | Only moderate confidence (r=0.35) | Use general model for S. aureus instead |
| **No wet-lab validation** | All results computational | 8 AMP candidates ready for experimental testing |
| **Research stage** | No clinical deployment yet | Production APIs and inference tested |

---

## Why This Approach?

### The Problem
Standard deep learning uses Euclidean geometry - great for images, suboptimal for hierarchical biological data (phylogenies, protein families, mutation trees).

### Our Solution
**Hyperbolic geometry** + **p-adic number theory** = geometry that matches data structure.

- Mutations close in evolutionary space → close in hyperbolic space
- Hierarchical relationships preserved automatically
- Captures signals that Euclidean models miss

### Unique Capabilities

1. **Rosetta-blind detection**: Identifies geometrically strained mutations that physics tools score as stable
2. **Codon-level signal**: Captures evolutionary constraints beyond one-hot encoding (+75-216% improvement on 3 HIV drugs)
3. **Multi-disease generalization**: Single architecture tested on 9 diseases (mean Spearman 0.54)

---

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run antimicrobial peptide design
python deliverables/partners/antimicrobial_peptides/scripts/B1_pathogen_specific_design.py

# Run DDG prediction
python deliverables/partners/protein_stability_ddg/scripts/C4_mutation_effect_predictor.py

# Run arbovirus primer design
python deliverables/partners/arbovirus_surveillance/scripts/A2_pan_arbovirus_primers.py
```

### Interactive Demos

- [Full Platform Demo](deliverables/demos/full_platform_demo.ipynb)
- [AMP Navigator](deliverables/partners/antimicrobial_peptides/notebooks/brizuela_amp_navigator.ipynb)
- [Protein Stability Explorer](deliverables/partners/protein_stability_ddg/notebooks/colbes_scoring_function.ipynb)

---

## For Different Audiences

| You Are... | Start Here |
|------------|-----------|
| **Investor/Partner** | [Business Case](docs/content/stakeholders/investors.md) - Market opportunity, defensibility, funding |
| **Bioinformatician** | [Bioinformatics Guide](docs/BIOINFORMATICS_GUIDE.md) - No math required |
| **ML Engineer** | [CLAUDE_DEV.md](deliverables/docs/CLAUDE_DEV.md) - Architecture, training, extending |
| **Researcher** | [Partner Packages](deliverables/partners/) - Ready-to-use tools |

---

## Project Structure

```
src/                    # Core ML framework
├── core/              # P-adic math, ternary operations
├── geometry/          # Poincaré ball, hyperbolic distances
├── models/            # VAE architectures
├── encoders/          # Codon, peptide, segment encoders
└── training/          # Training loops, optimizers

deliverables/          # Production-ready packages
├── partners/          # 4 validated research packages
├── shared/            # Common utilities
└── results/           # Publication-quality figures

research/              # Scientific exploration
├── diseases/          # HIV (200K sequences), arbovirus
└── codon-encoder/     # Embedding extraction
```

---

## Technical Foundation

Built on the [3-adic-ml](https://github.com/Ai-Whisperers/3-adic-ml) mathematical framework.

| Component | Description |
|-----------|-------------|
| **TernaryVAE** | Dual-encoder architecture with homeostatic controller |
| **Poincaré Ball** | 16-d hyperbolic latent space (geoopt-backed) |
| **P-adic Valuation** | Hierarchy encoding for ternary operations |
| **Mixed Precision** | torch.compile with 3-4x speedup |

<details>
<summary><strong>Validated Checkpoints</strong></summary>

| Checkpoint | Coverage | Hierarchy | Best For |
|------------|----------|-----------|----------|
| `homeostatic_rich` | 100% | -0.8321 | DDG prediction, semantic reasoning |
| `v5_12_4` | 100% | -0.82 | General purpose |
| `v5_11_structural` | 100% | -0.74 | Contact prediction (AUC=0.67) |

</details>

<details>
<summary><strong>Mathematical Background</strong></summary>

**P-adic Hierarchy**: The 3-adic valuation v_3(n) counts powers of 3 in n, creating natural tree structure.

**Hyperbolic Realization**: Poincaré ball provides exponential volume growth and proper geodesic distances.

**Ultrametric Property**: d(x,z) ≤ max(d(x,y), d(y,z)) - creates perfect hierarchical clustering matching phylogenetic trees.

</details>

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.0 | 2.1+ |
| RAM | 8GB | 16GB |
| GPU | Optional | CUDA 11.8+ |

```bash
pip install -e ".[all]"
```

---

## License

- **Code**: [PolyForm Non-Commercial 1.0.0](LICENSE) - Academic/non-profit use permitted; commercial use requires license
- **Outputs**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) - Free for any reuse with attribution

---

## Citation

```bibtex
@software{ultrametric_antigen_ai,
  author = {{AI Whisperers}},
  title = {Ultrametric Antigen AI: Hyperbolic VAEs for Bioinformatics},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ultrametric-antigen-AI}
}
```

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/Ai-Whisperers/ultrametric-antigen-AI/issues)
- **Commercial licensing**: ai.whisperer.wvdp@gmail.com
- **Collaboration**: See [Partner Packages](deliverables/partners/) for research partnerships

---

*Version 5.12.5 · Updated 2026-02-04*

---

## The P-Adic Ecosystem

This repository is part of a tri-fold ecosystem exploring the intersection of p-adic mathematics, ternary logic, and high-performance computing:

*   **[3-Adic ML](https://github.com/gesttaltt/3-adic-ml)**: Mathematical foundation and framework for p-adic Variational Autoencoders and geometric deep learning.
*   **[3-Adic Bioinformatics](https://github.com/gesttaltt/3-adic-bioinformatics)**: (This Repo) Application of ultrametric geometry to genomic sequences, protein folding, and biological hierarchy analysis.
*   **[Ternary Engine](https://github.com/gesttaltt/ternary-engine)**: High-performance C++/C backend for native ternary arithmetic and efficient p-adic valuation processing.

## Status & Engagement

**Current Phase**: Active Low-Profile Research

We are investigating how p-adic geometry can better represent the hierarchical nature of biological sequences. While the code is now public to facilitate scientific transparency, we are maintaining a focused development environment.

*   **Proposals**: We are not actively seeking investment or commercial acquisition. Our goal is the advancement of computational biology through non-Euclidean mathematics.
*   **Collaboration**: We welcome collaboration from bioinformatics researchers and computational biologists interested in non-standard sequence representations.
