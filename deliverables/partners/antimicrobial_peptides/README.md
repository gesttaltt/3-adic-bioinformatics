# Antimicrobial Peptide Design Package

**Design AMPs for WHO priority pathogens with validated ML models**

[![Status](https://img.shields.io/badge/status-90%25%20ready-green.svg)]()
[![Models](https://img.shields.io/badge/models-5%20validated-blue.svg)]()

---

## Why This Package?

| Capability | Value |
|------------|-------|
| **5 pathogen-specific models** | All statistically significant (p<0.001) |
| **Mean correlation r=0.656** | Outperforms sklearn baseline (0.56) |
| **Multi-objective optimization** | Balance activity, toxicity, synthesis cost |
| **Wet-lab ready candidates** | 8 peptides with selectivity index >1.0 |

**Target pathogens**: *A. baumannii*, *P. aeruginosa*, *E. coli*, *S. aureus*, *H. pylori* (WHO priority)

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Package Structure](#package-structure)
- [Easy Implementation Tools](#easy-implementation-tools)
- [Demo Results Summary](#demo-results-summary)
- [What's Included](#whats-included)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Output Formats](#output-formats)
- [Integration with VAE](#integration-with-vae)
- [Validation Checklist](#validation-checklist)
- [Known Limitations](#known-limitations)
- [Scientific Background](#scientific-background)
- [Citation](#citation)
- [Contact](#contact)

---

## Executive Summary

This package provides a comprehensive toolkit for antimicrobial peptide (AMP) design using NSGA-II multi-objective optimization in the VAE latent space. It includes three specialized tools:

1. **B1: Pathogen-Specific AMP Design** - Design AMPs targeting specific pathogens (e.g., *A. baumannii*)
2. **B8: Microbiome-Safe AMPs** - Design selective peptides that kill pathogens while sparing commensals
3. **B10: Synthesis Optimization** - Balance antimicrobial activity with synthesis feasibility

### Validated Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Mean Spearman** | **0.656** | 5-fold CV |
| Std Spearman | 0.060 | Consistent across folds |
| Best fold | 0.737 | Fold 2 |
| PeptideVAE status | PASSED | Beats sklearn baseline (0.56) |

---

## Package Structure

```
antimicrobial_peptides/
├── README.md                      # Main documentation
├── VALIDATION_FINDINGS.md         # Validation details
├── scripts/                       # Optimization tools
│   ├── B1_pathogen_specific_design.py
│   ├── B8_microbiome_safe_amps.py
│   ├── B10_synthesis_optimization.py
│   ├── predict_mic.py
│   └── latent_nsga2.py
├── notebooks/                     # Interactive navigator
│   └── amp_navigator.ipynb
├── results/                       # Generated peptides
│   ├── pathogen_specific/
│   ├── microbiome_safe/
│   └── synthesis_optimized/
├── checkpoints_definitive/        # Model checkpoints
├── models/                        # Pathogen-specific models
├── docs/                          # User guides
│   ├── B1_USER_GUIDE.md
│   ├── B8_USER_GUIDE.md
│   ├── B10_USER_GUIDE.md
│   └── LIMITATIONS_AND_FUTURE_WORK.md
├── src/                           # Shared library code
├── training/                      # Training scripts
└── validation/                    # Validation scripts
```

---

## Easy Implementation Tools

### B1: Pathogen-Specific AMP Design

Design antimicrobial peptides optimized for specific WHO priority pathogens.

```bash
# Single pathogen optimization
python scripts/B1_pathogen_specific_design.py \
    --pathogen A_baumannii \
    --output results/pathogen_specific/

# All WHO priority pathogens
python scripts/B1_pathogen_specific_design.py \
    --all-pathogens \
    --output results/pathogen_specific/
```

**Target Pathogens Supported:**
| CLI Name | Full Name | Priority | Resistance Pattern |
|----------|-----------|----------|-------------------|
| `A_baumannii` | *Acinetobacter baumannii* | Critical | Carbapenem-resistant |
| `P_aeruginosa` | *Pseudomonas aeruginosa* | Critical | Carbapenem-resistant |
| `Enterobacteriaceae` | *Enterobacteriaceae* (group) | Critical | Carbapenem-resistant |
| `S_aureus` | *Staphylococcus aureus* | High | Methicillin/Vancomycin-resistant |
| `H_pylori` | *Helicobacter pylori* | High | Clarithromycin-resistant |

### B8: Microbiome-Safe AMPs

Design peptides with selectivity for pathogens over beneficial commensals.

```bash
# Skin microbiome optimization
python scripts/B8_microbiome_safe_amps.py \
    --context skin \
    --output results/microbiome_safe/

# Gut microbiome optimization
python scripts/B8_microbiome_safe_amps.py \
    --context gut \
    --output results/microbiome_safe/
```

**Key Metric - Selectivity Index (SI):**
```
SI = Geometric Mean(Commensal MICs) / Geometric Mean(Pathogen MICs)
SI > 1 = Selective for pathogens (good)
SI > 4 = Clinically relevant selectivity (target)
```

### B10: Synthesis Optimization

Optimize peptides for both activity AND ease of synthesis.

```bash
# Standard synthesis optimization
python scripts/B10_synthesis_optimization.py \
    --output results/synthesis_optimized/

# Shorter peptides (easier to synthesize)
python scripts/B10_synthesis_optimization.py \
    --max-length 20 \
    --output results/synthesis_optimized/

# Dry run (without VAE model)
python scripts/B10_synthesis_optimization.py \
    --dry-run \
    --output results/synthesis_optimized/
```

**Synthesis Metrics Optimized:**
- Aggregation propensity
- Racemization risk
- Difficult coupling sequences
- Estimated cost per mg

---

## Demo Results Summary

### B1 Results - *Acinetobacter baumannii*

| Rank | Sequence | Charge | Hydro | Activity | Toxicity |
|------|----------|--------|-------|----------|----------|
| 1 | HFHTSFFFSTKVYETSHTHY | +2 | 0.09 | 4.04 | 0.0 |
| 2 | KHPHYTYYGAKTHKRVSQVK | +6.5 | -0.33 | 0.23 | 0.0 |
| 3 | KHPGYTYYGAKSHKRVSQVK | +6 | -0.30 | 0.19 | 0.0 |

### B8 Results - Microbiome-Safe

| Sequence | Charge | SI | Pathogen MIC | Commensal MIC |
|----------|--------|----|--------------| --------------|
| HNHWHMNWKKKKAYAHKPGR | +8 | 1.26 | 9.5 | 13.6 |
| RRTTHKHHCMSWRYKKAPHT | +8 | 1.26 | 10.1 | 14.5 |

### B10 Results - Synthesis-Optimized

| Sequence | Activity | Difficulty | Coupling | Cost |
|----------|----------|------------|----------|------|
| HRGTGKRTIKKLAVAGKFGA | 0.908 | 14.79 | 50.9% | $36.50 |
| GKRSLALGKRVLNCGARKGN | 0.882 | 14.62 | 51.5% | $36.50 |

---

## What's Included

### 1. Core Scripts

| File | Description | Lines |
|------|-------------|-------|
| `scripts/latent_nsga2.py` | NSGA-II optimizer core | 490 |
| `scripts/B1_pathogen_specific_design.py` | Pathogen-specific design | ~400 |
| `scripts/B8_microbiome_safe_amps.py` | Microbiome-safe design | ~400 |
| `scripts/B10_synthesis_optimization.py` | Synthesis optimization | ~400 |
| `scripts/predict_mic.py` | MIC prediction tool | ~200 |

### 2. Interactive Notebook

| File | Description |
|------|-------------|
| `notebooks/amp_navigator.ipynb` | Visualization and exploration |

### 3. Results

| File | Description |
|------|-------------|
| `results/pareto_peptides.csv` | 100 Pareto-optimal solutions |
| `results/pathogen_specific/*.json` | B1 demo results |
| `results/microbiome_safe/*.json` | B8 demo results |
| `results/synthesis_optimized/*.csv` | B10 demo results |

### 4. Documentation

| File | Description |
|------|-------------|
| `docs/B1_USER_GUIDE.md` | Pathogen-specific design guide |
| `docs/B8_USER_GUIDE.md` | Microbiome-safe design guide |
| `docs/B10_USER_GUIDE.md` | Synthesis optimization guide |
| `docs/LIMITATIONS_AND_FUTURE_WORK.md` | Known limitations |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy torch pandas matplotlib seaborn deap
```

### Step 2: Run All Demos

```bash
# B1: Pathogen-Specific
python scripts/B1_pathogen_specific_design.py

# B8: Microbiome-Safe
python scripts/B8_microbiome_safe_amps.py

# B10: Synthesis-Optimized
python scripts/B10_synthesis_optimization.py
```

### Step 3: Explore Results

```bash
jupyter notebook notebooks/amp_navigator.ipynb
```

---

## Technical Details

### NSGA-II Multi-Objective Optimization

All three tools use the NSGA-II algorithm operating in the VAE latent space:

```
Latent Space (16D) --> NSGA-II Optimization --> Decoded Peptides
```

**Key Innovation:**
- Traditional: Discrete sequence mutations (20^L combinations)
- Our approach: Continuous latent space optimization

### Objective Functions

#### B1: Pathogen-Specific
| Objective | Description | Goal |
|-----------|-------------|------|
| Activity | Predicted MIC against target | Minimize |
| Toxicity | Hemolysis prediction | Minimize |
| Stability | VAE reconstruction quality | Maximize |

#### B8: Microbiome-Safe
| Objective | Description | Goal |
|-----------|-------------|------|
| Pathogen Activity | MIC against pathogens | Minimize |
| Commensal Sparing | MIC against commensals | Maximize |
| Selectivity Index | Ratio of above | Maximize |
| Toxicity | Host cell safety | Minimize |

#### B10: Synthesis-Optimized
| Objective | Description | Goal |
|-----------|-------------|------|
| Activity | Antimicrobial potency | Maximize |
| Synthesis Difficulty | Aggregation + racemization | Minimize |
| Cost | Estimated $/mg | Minimize |

### Feature Engineering

Each peptide is featurized with 12 physicochemical descriptors plus 5 p-adic valuation features:

| Feature Group | Features |
|---------------|---------|
| Physicochemical (12) | length, charge, hydrophobicity, volume, positive/negative/aromatic/aliphatic/polar/hydrophobic fractions, amphipathicity, hydrophobic moment |
| P-adic (5) | mean valuation, max valuation, valuation variance, valuation product, valuation gradient |
| Amino acid composition (20) | per-AA frequency (aac_A … aac_Y) |

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Individuals per generation |
| `generations` | 50 | Evolutionary iterations |
| `crossover_prob` | 0.9 | SBX crossover probability |
| `mutation_prob` | 0.1 | Per-gene mutation rate |
| `latent_bounds` | (-3, 3) | Valid latent coordinate range |

---

## Output Formats

### Pareto Front CSV

```csv
rank,sequence,net_charge,hydrophobicity,activity_score,toxicity,synthesis_difficulty
1,HFHTSFFFSTKVYETSHTHY,2.0,0.09,4.04,0.0,12.5
2,KHPHYTYYGAKTHKRVSQVK,6.5,-0.33,0.23,0.0,14.2
```

### Results JSON

```json
{
  "objective": "Pathogen-specific AMP design",
  "target_pathogen": "Acinetobacter_baumannii",
  "nsga2_config": {
    "population": 100,
    "generations": 50
  },
  "pareto_front_size": 15,
  "candidates": [
    {
      "rank": 1,
      "sequence": "HFHTSFFFSTKVYETSHTHY",
      "net_charge": 2.0,
      "hydrophobicity": 0.09,
      "predicted_activity": 4.04,
      "toxicity_score": 0.0,
      "latent_vector": [0.23, -0.45, ...]
    }
  ]
}
```

---

## Integration with VAE

### Using Real VAE Checkpoint

```python
from src.vae_interface import VAEInterface

# Load trained VAE
vae = VAEInterface(checkpoint_path="checkpoints_definitive/best_production.pt")

# Decode latent vector to sequence
z = np.array([0.23, -0.45, ...])  # From optimization
sequence = vae.decode_latent(z)

# Encode existing peptide
z_encoded = vae.encode_sequence("KLWKKLKKALK")
```

### Custom Objective Functions

```python
from scripts.latent_nsga2 import LatentNSGA2, OptimizationConfig

def my_activity_objective(z):
    """Custom activity prediction."""
    sequence = vae.decode_latent(z)
    return your_activity_model.predict(sequence)

def my_toxicity_objective(z):
    """Custom toxicity prediction."""
    sequence = vae.decode_latent(z)
    return your_toxicity_model.predict(sequence)

# Configure optimizer
config = OptimizationConfig(
    latent_dim=16,
    population_size=200,
    generations=100
)

optimizer = LatentNSGA2(
    config=config,
    objective_functions=[my_activity_objective, my_toxicity_objective]
)

# Run optimization
pareto_front = optimizer.run(verbose=True)
```

---

## Validation Checklist

### B1: Pathogen-Specific
- [x] Script runs without errors
- [x] NSGA-II completes 50 generations
- [x] Pareto front contains 10+ candidates
- [x] All candidates have positive charge (+2 to +8)
- [x] Toxicity scores are < 0.5 for all

### B8: Microbiome-Safe
- [x] Selectivity Index > 1.0 for top candidates
- [x] Pathogen MICs < Commensal MICs
- [x] No duplicate sequences in output

### B10: Synthesis-Optimized
- [x] Synthesis difficulty < 20 for top candidates
- [x] No difficult coupling motifs (Asp-X, His-His)
- [x] Estimated cost < $50/mg

---

## Model Validation

### Per-Pathogen Model Performance

All 5 models are statistically significant after dataset expansion and validation fixes.

| Pathogen | N | Pearson r | p-value | Confidence |
|----------|--:|:---------:|:-------:|:----------:|
| General | 425 | 0.606 | 5.6e-44 | **HIGH** |
| P. aeruginosa | 100 | 0.507 | 7.2e-08 | **HIGH** |
| E. coli | 133 | 0.493 | 1.6e-09 | **HIGH** |
| A. baumannii | 88 | 0.461 | 6.2e-06 | **HIGH** |
| S. aureus | 104 | 0.343 | 3.6e-04 | **MODERATE** |

**Source:** `validation/results/comprehensive_validation.json` (5-fold CV, includes p-adic features)

### Methodology Notes

| Component | Method | Validated |
|-----------|--------|:---------:|
| MIC Prediction | PeptideVAE ML | **YES** (5/5 models) |
| Toxicity | Heuristic (charge, hydrophobicity) | **NO** |
| Stability | Proxy (reconstruction quality) | **NO** |
| Pathogen specificity | DRAMP database labels | PARTIAL |

**Note:** S. aureus has MODERATE confidence - use for ranking candidates, combine with general model for robust predictions.

---

## Scientific Background

### Why Multi-Objective Optimization?

AMP design inherently involves trade-offs:
- High activity often correlates with toxicity
- Stable peptides may be harder to synthesize
- Broad-spectrum may mean less selectivity

NSGA-II discovers the Pareto front: solutions where improving one objective necessarily worsens another.

### P-adic Latent Space Advantage

The Ternary VAE embeds peptides in a hyperbolic space where:
- Center = more stable sequences
- Edge = more variable sequences
- P-adic valuation correlates with evolutionary stability

This provides implicit regularization toward stable, "natural-like" peptides.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ternary_vae_amp,
  author = {{AI Whisperers}},
  title = {Multi-Objective Antimicrobial Peptide Design},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics},
  note = {Part of the Ternary VAE Bioinformatics project}
}
```

---

## Honest Limitations

| Limitation | Details | Mitigation |
|------------|---------|------------|
| **S. aureus model** | Moderate confidence (r=0.35) | Use general model for S. aureus predictions |
| **Toxicity prediction** | Heuristic (charge/hydrophobicity), not ML | Validate experimentally before synthesis |
| **Stability proxy** | Uses VAE reconstruction quality | Not a direct stability measurement |
| **No wet-lab validation** | All results computational | 8 candidates ready for experimental testing |

---

## Contact

- **Repository:** [github.com/Ai-Whisperers/ternary-vaes-bioinformatics](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics)
- **Issues:** GitHub Issues
- **Email:** ai.whisperer.wvdp@gmail.com
- **References:** NSGA-II (Deb et al. 2002), DRAMP, APD3, DBAASP databases

---

*Version 2.1 · Updated 2026-02-04*
*Part of the [Ultrametric Antigen AI](../../../README.md) project*
