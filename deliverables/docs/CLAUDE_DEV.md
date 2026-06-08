# Ultrametric Antigen AI - Developer Reference

**Doc-Type:** AI Context (Developer) · Version 1.0 · 2026-02-03

---

## Architecture Overview

The TernaryVAE embeds 19,683 ternary operations (3^9) into a 16-dimensional hyperbolic Poincaré ball where radial position encodes 3-adic valuation.

**Current Version:** 5.12.5
**Status:** Production-optimized

---

## Dual-Encoder System (v5.11+)

### VAE-A (Coverage Encoder)
- **Role:** Reconstruct all 19,683 operations
- **Behavior:** May learn frequency-based ordering
- **Freezing:** Freeze to preserve coverage while training VAE-B

### VAE-B (Hierarchy Encoder)
- **Role:** Learn hierarchical radial ordering
- **Behavior:** Negative correlation for p-adic structure
- **Training:** Benefits from slower learning rate (encoder_b_lr_scale=0.1)

### DifferentiableController
- **Architecture:** 8→32→32→6 MLP
- **Outputs:** rho, weight_geodesic, beta_A, beta_B, tau
- **Enable:** `use_controller=True`

### HomeostasisController
- **Role:** Training orchestrator for freeze/unfreeze
- **Triggers:** Coverage thresholds, hierarchy plateau detection
- **Q-metric:** `Q = dist_corr + 1.5 × |hierarchy|`

---

## V5.12.5 Production Optimizations

### Performance Gains

| Optimization | Speedup | Implementation |
|--------------|---------|----------------|
| torch.compile | 1.4-2.0x | PyTorch 2.x Inductor |
| Mixed Precision | 2.0x + 20-30% VRAM | FP16 autocast |
| Per-Parameter LR | Better hierarchy | encoder_b_lr_scale=0.1 |
| Grokking Detection | Real-time monitoring | Phase classification |

**Measured:** 57 seconds vs ~2-3 minutes baseline (5 epochs, 19,683 ops)

### Configuration

```yaml
# src/configs/v5_12_4_fixed_checkpoint.yaml
torch_compile:
  enabled: true
  backend: eager
  mode: default

mixed_precision:
  enabled: true
  dtype: float16
  init_scale: 65536.0

option_c:
  enabled: true
  encoder_b_lr_scale: 0.1
  encoder_a_lr_scale: 0.05
```

### Training Command

```bash
python src/scripts/training/train_v5_12.py \
    --config src/configs/v5_12_4_fixed_checkpoint.yaml \
    --epochs 100
```

---

## V5.12.4 Improved Components

`ImprovedEncoder` and `ImprovedDecoder` (`src/models/improved_components.py`):

- SiLU activation (smoother gradients than ReLU)
- LayerNorm for stable training
- Dropout (default 0.1)
- logvar clamping [-10, 2] to prevent KL collapse

**Enable:**
```yaml
encoder_type: improved
decoder_type: improved
```

---

## Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Coverage** | % of 19,683 ops correctly reconstructed | 100% |
| **Hierarchy** | Spearman(valuation, radius) | -0.83 to -1.0 |
| **Richness** | Avg within-level radius variance | >0.005 |
| **Q-metric** | dist_corr + 1.5×\|hierarchy\| | Maximize |

### Hierarchy Ceiling: -0.8321

The Spearman correlation cannot exceed -0.8321 with any within-level variance because v=0 contains 66.7% of samples (13,122 of 19,683).

---

## Checkpoint Reference

| Checkpoint | Coverage | Hier_B | Use Case |
|------------|:--------:|:------:|----------|
| `homeostatic_rich` | 100% | -0.8321 | DDG, semantic reasoning |
| `v5_12_4/best_Q.pt` | 100% | -0.82 | General purpose |
| `v5_11_structural` | 100% | -0.74 | Contact prediction |
| `v5_11_progressive` | 100% | +0.78 | Compression (frequency-optimal) |
| `v5_5/best.pt` | 97.1% | -0.30 | Foundation topology |

### Dual Manifold Framework

Both manifold types are mathematically valid:

| Type | Hierarchy | Optimizes For |
|------|:---------:|---------------|
| Valuation-optimal | Negative (-0.8 to -1.0) | P-adic semantic structure |
| Frequency-optimal | Positive (+0.6 to +0.8) | Shannon information |

---

## Critical Pattern: Hyperbolic Distance

```python
# CORRECT - Hyperbolic distance from origin
from src.geometry import poincare_distance
origin = torch.zeros_like(z_hyp)
radius = poincare_distance(z_hyp, origin, c=curvature)

# WRONG - Euclidean norm on hyperbolic embeddings
radius = torch.norm(z_hyp, dim=-1)  # DO NOT USE
```

### V5.12.2 Audit Status

- **Core files:** All fixed
- **Research scripts:** ~40 files still use Euclidean norm()
- **Audit docs:** `docs/mathematical-foundations/V5_12_2_audit/`

---

## Training Parameters

```python
# Loss weights (from homeostatic_rich)
hierarchy_weight = 5.0      # Push toward target radii
coverage_weight = 1.0       # Maintain reconstruction
richness_weight = 2.0       # Preserve within-level variance
separation_weight = 3.0     # Ensure level ordering

# Freeze strategy
freeze_encoder_a = True     # Preserve coverage
freeze_encoder_b = False    # Allow B to learn hierarchy
encoder_b_lr_scale = 0.1    # Slower adaptation

# Variance control
variance_weight = 50-100    # Higher = more collapse
min_richness_ratio = 0.5    # Keep 50% of original variance
```

---

## File Locations

### Core Model
| Purpose | Location |
|---------|----------|
| Main model | `src/models/ternary_vae.py` |
| Improved components | `src/models/improved_components.py` |
| Homeostasis controller | `src/models/homeostasis.py` |

### Geometry
| Purpose | Location |
|---------|----------|
| Poincaré operations | `src/geometry/` |
| **DEPRECATED** | `src/core/geometry_utils.py` |

### Training
| Purpose | Location |
|---------|----------|
| Main script | `src/scripts/training/train_v5_12.py` |
| Homeostatic rich | `src/scripts/epsilon_vae/train_homeostatic_rich.py` |
| Configs | `src/configs/` |

### Encoders
| Purpose | Location |
|---------|----------|
| TrainableCodonEncoder | `src/encoders/trainable_codon_encoder.py` |
| Codon encoder | `src/encoders/codon_encoder.py` |

---

## Grokking Detection

Real-time training dynamics monitoring:

```python
from src.training.grokking_detector import GrokDetector

grok_detector = GrokDetector()
grok_analysis = grok_detector.update(EpochMetrics(
    epoch=epoch, train_loss=loss, correlation=hierarchy_B
))
# Phases: WARMUP → MEMORIZATION → GROKKING
```

---

## Mixed Precision Training

```python
from src.training.optimizations import MixedPrecisionTrainer, MixedPrecisionConfig

mp_trainer = MixedPrecisionTrainer(MixedPrecisionConfig(
    enabled=True, dtype='float16', init_scale=65536.0
))
```

---

## Per-Parameter Learning Rates

```python
param_groups = [
    {"params": encoder_A_params, "lr": base_lr * 0.05},  # Slower coverage
    {"params": encoder_B_params, "lr": base_lr * 0.10},  # Hierarchy learning
    {"params": projection_params, "lr": base_lr}         # Fast adaptation
]
```

---

## Quick Evaluation

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY
from src.geometry import poincare_distance
from scipy.stats import spearmanr

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.99,
    curvature=1.0, use_controller=True, use_dual_projection=True
)
model.load_state_dict(torch.load('checkpoints/homeostatic_rich/best.pt')['model_state_dict'])

# Get embeddings
out = model(ops, compute_control=False)

# Compute radii (hyperbolic)
origin = torch.zeros_like(out['z_B_hyp'])
radii_B = poincare_distance(out['z_B_hyp'], origin, c=1.0)

# Compute hierarchy
valuations = TERNARY.valuation(indices)
hierarchy = spearmanr(valuations.cpu(), radii_B.cpu())[0]
```

---

## Partner Packages Integration

All partner packages use `src.core.padic_math` for consistency.

| Package | Key Integration |
|---------|-----------------|
| protein_stability_ddg | TrainableCodonEncoder + Ridge |
| antimicrobial_peptides | PeptideVAE + NSGA-II |
| arbovirus_surveillance | TrainableCodonEncoder + primer design |
| hiv_research_package | Stanford HIVdb API |

---

## Deprecated Modules

**`src/core/geometry_utils.py`** - DEPRECATED as of V5.12.2

```python
# OLD (deprecated)
from src.core.geometry_utils import poincare_distance

# NEW (use this)
from src.geometry import poincare_distance, exp_map_zero
```

---

## Known Issues

- Graph breaks in torch.compile from `.item()` calls (minor)

## Recently Fixed

- `torch.cuda.amp` → `torch.amp` migration (`src/training/optimizations.py`)
- AlphaFold `/api/prediction` sunset 2026-06-25: `AlphaFoldStructureLoader` now resolves
  download URLs via API (`modelEntityId`/`entryId`/`pdbUrl`) with direct-URL fallback
  (`src/encoders/alphafold_encoder.py`)
- V5.12.2 hyperbolic audit complete: all 258 `.norm()` calls verified correct

---

## Remaining Tasks

| Priority | Task |
|:--------:|------|
| 1 | Fix tier numbering in docs |
| 2 | Publication figures organization |

---

## External APIs

**AlphaFold API** - Sunset **2026-06-25**. Use `modelEntityId` not `entryId`.

---

## Documentation Links

| Document | Purpose |
|----------|---------|
| [CLAUDE_BIO.md](CLAUDE_BIO.md) | Bioinformatics applications |
| [CLAUDE_LITE.md](CLAUDE_LITE.md) | Quick reference |
| [Mathematical Foundations](../../docs/mathematical-foundations/) | Deep theory |
| [V5.12.2 Audit](../../docs/mathematical-foundations/V5_12_2_audit/) | Hyperbolic fixes |

---

*For mathematical theory: [../../docs/mathematical-foundations/](../../docs/mathematical-foundations/)*
*Original full context: [../../docs/mathematical-foundations/archive/CLAUDE_ORIGINAL.md](../../docs/mathematical-foundations/archive/CLAUDE_ORIGINAL.md)*


======================================================================BIOINFORMATICS-APPLICATIONS=======================================================================================================


# Bioinformatics Applications

**Doc-Type:** AI Context (Bioinformatics) · Version 1.0 · 2026-02-03

---

## What This Project Makes Possible

Sequence-only predictions for bioinformatics applications using *learned codon embeddings* over 3-adic embedding spaces. No previous structure required.

**Core Capability:** Extract meaningful features from genetic sequences by learning hierarchical relationships in hyperbolic space.

---

## Validated Results

| Application | Metric | Value | Dataset | Status |
|-------------|--------|-------|---------|--------|
| **DDG Prediction** | LOO Spearman | 0.52-0.58 | S669 (N=52) | Production |
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
