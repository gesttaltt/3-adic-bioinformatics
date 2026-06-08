# Ultrametric Antigen AI - Bioinformatics Applications

**Doc-Type:** AI Context (Bioinformatics) · Version 1.0 · 2026-02-03

---

## What This Project Does

Sequence-only predictions for bioinformatics applications using learned codon embeddings. No structure required.

**Core Capability:** Extract meaningful features from genetic sequences by learning hierarchical relationships in hyperbolic space.

---

## Validated Results

| Application | Metric | Value | Dataset | Status |
|-------------|--------|-------|---------|--------|
| **DDG Prediction** | LOO Spearman | 0.52-0.58 | ProTherm subset (N=52, NOT S669) | Production |
| **Contact Prediction** | AUC-ROC | 0.67 | Insulin B-chain | Research |
| **AMP Fitness** | Pearson r | 0.61 | DRAMP | Production |
| **Force Constants** | Correlation | 0.86 | AA properties | Validated |

---

## Partner Packages

### Protein Stability (DDG)

**Location:** `deliverables/partners/protein_stability_ddg/`
**Status:** 95% Ready

Predict protein stability changes (ΔΔG) from single amino acid mutations.

```bash
# Predict mutation effect
python scripts/C4_mutation_effect_predictor.py --mutations mutations.csv
```

**Validated Performance:**
| Metric | N=52 (curated) | N=669 (full) |
|--------|:--------------:|:------------:|
| Spearman | 0.52 | 0.37-0.40 |
| p-value | <0.001 | <0.001 |

**Strengths:**
- Rosetta-blind detection (23.6% of cases Rosetta misses)
- Neutral→charged mutations: +159% vs baseline
- No structure required

**Limitations:**
- Charge reversal mutations: method fails
- Proline mutations: method fails
- N=669 does NOT outperform ESM-1v/FoldX

---

### Antimicrobial Peptides

**Location:** `deliverables/partners/antimicrobial_peptides/`
**Status:** 90% Ready

Multi-objective AMP design using NSGA-II optimization in VAE latent space.

```bash
# Pathogen-specific design
python scripts/B1_pathogen_specific_design.py --pathogen A_baumannii

# Microbiome-safe AMPs
python scripts/B8_microbiome_safe_amps.py --context gut

# Synthesis optimization
python scripts/B10_synthesis_optimization.py
```

**Per-Pathogen Model Performance:**

| Pathogen | N | Pearson r | Confidence |
|----------|--:|:---------:|:----------:|
| General | 425 | 0.608 | HIGH |
| P. aeruginosa | 100 | 0.506 | HIGH |
| E. coli | 133 | 0.492 | HIGH |
| A. baumannii | 88 | 0.463 | HIGH |
| S. aureus | 104 | 0.348 | MODERATE |

**All 5 models statistically significant** (p < 0.05)

**Note:** S. aureus has MODERATE confidence - combine with general model for robust predictions.

---

### Arbovirus Surveillance

**Location:** `deliverables/partners/arbovirus_surveillance/`
**Status:** 90% Ready

Pan-arbovirus primer design for 7 viruses.

```bash
# Design primers
python scripts/A2_pan_arbovirus_primers.py --use-ncbi
```

**Coverage:**
- Dengue (all 4 serotypes)
- Zika
- Chikungunya
- Yellow Fever

**Key Finding:** DENV-4 cryptic diversity (71.7% identity vs 95-98% other serotypes) makes universal primer design challenging - addressed through dual-layer architecture.

---

### HIV Drug Resistance

**Location:** `deliverables/partners/hiv_research_package/`
**Status:** Complete

Stanford HIVdb API integration for resistance prediction.

```bash
# TDR screening
python scripts/H6_tdr_screening.py

# Long-acting selection
python scripts/H7_la_selection.py
```

---

## Checkpoints for Bioinformatics

| Task | Recommended Checkpoint | Why |
|------|------------------------|-----|
| DDG prediction | `homeostatic_rich` | High richness for mutation effects |
| Contact prediction | `v5_11_structural` | Collapsed shells for pairwise distances |
| AMP optimization | `v5_12_4/best_Q.pt` | General purpose |
| Codon embeddings | `trained_codon_encoder.pt` | Direct sequence encoding |

---

## TrainableCodonEncoder

Direct codon-to-embedding encoder for sequence analysis.

```python
from src.encoders import TrainableCodonEncoder
import torch

encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
ckpt = torch.load('research/codon-encoder/training/results/trained_codon_encoder.pt')
encoder.load_state_dict(ckpt['model_state_dict'])
encoder.eval()

# Get amino acid embeddings
aa_embs = encoder.get_all_amino_acid_embeddings()

# Compute distance between amino acids
dist = encoder.compute_aa_distance('A', 'V')  # hyperbolic distance
```

**Performance:** LOO Spearman 0.61 on DDG prediction (+105% over baseline)

---

## DDG Prediction Details

### Mutation-Type Performance

| Mutation Type | Performance vs Baseline | Use? |
|--------------|:-----------------------:|:----:|
| neutral → charged | **+159%** | YES |
| hydrophobic → polar | +52% | YES |
| size_change | +28% | MAYBE |
| charge_reversal | -737% | NO |
| proline_mutations | -89% | NO |

### Rosetta-Blind Detection

The method catches 23.6% of cases where Rosetta fails - complementary to structure-based methods.

### Honest Performance Disclosure

- **N=52 curated subset:** rho=0.52 (what ships in ValidatedDDGPredictor)
- **N=669 full dataset:** rho=0.37-0.40 (fair literature comparison)
- ESM-1v (0.51), FoldX (0.48) are benchmarked on N=669

---

## Contact Prediction Discovery

Pairwise hyperbolic distances predict residue-residue 3D contacts.

| Checkpoint | Richness | AUC-ROC | Best For |
|------------|:--------:|:-------:|----------|
| v5_11_structural | ~0.003 | **0.67** | Contacts |
| homeostatic_rich | 0.007 | 0.59 | DDG |

**Tradeoff:** Low richness (collapsed shells) = better contacts. High richness = better DDG.

---

## Physical Invariants Discovered

| Invariant | Correlation | Significance |
|-----------|:-----------:|--------------|
| Dimension 13 → mass, volume | rho = -0.695 | "Physics dimension" |
| Radial position → AA mass | rho = +0.760 | Emergent property |
| Force constant formula | rho = 0.860 | `k = radius × mass / 100` |

---

## Data Leakage: Fixed

All validation scripts now use proper sklearn Pipeline pattern:

```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=100))
])
y_pred = cross_val_predict(pipeline, X, y, cv=len(y))  # Scaler inside CV
```

---

## Validation Sync Script

Ensure documentation matches canonical JSON sources:

```bash
python3 deliverables/scripts/sync_validation_docs.py --report
```

---

## Quick Links

| Resource | Location |
|----------|----------|
| Protein Stability | `deliverables/partners/protein_stability_ddg/` |
| Antimicrobial Peptides | `deliverables/partners/antimicrobial_peptides/` |
| Arbovirus Surveillance | `deliverables/partners/arbovirus_surveillance/` |
| HIV Research | `deliverables/partners/hiv_research_package/` |
| Validation Status | `deliverables/partners/CLAUDE.md` |

---

*For technical implementation details: [CLAUDE_DEV.md](CLAUDE_DEV.md)*
*For mathematical theory: [../../docs/mathematical-foundations/](../../docs/mathematical-foundations/)*
