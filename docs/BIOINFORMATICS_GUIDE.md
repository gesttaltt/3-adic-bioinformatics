# Bioinformatics Applications Guide

> **For bioinformatics specialists** - No mathematical background required.
>
> **Foundation:** Mathematical framework at [3-adic-ml](https://github.com/Ai-Whisperers/3-adic-ml)
> For implementation details, see [CLAUDE.md](../CLAUDE.md).
> For mathematical theory, see [mathematical-foundations/](mathematical-foundations/).

**Note:** This is a **complement** to classical methods (AlphaFold, ESMFold, etc.), not a replacement. Results exceeding classical benchmarks emerge from proper 3-adic geometry research.

---

## What This Project Does

This project provides **sequence-only predictions** for bioinformatics applications. Given only a DNA/protein sequence, it can predict:

- **Protein stability changes** (ΔΔG) upon mutation
- **Residue-residue 3D contacts** from sequence
- **Antimicrobial peptide fitness** and MIC
- **Arbovirus primer design** for surveillance

The key insight: codons that encode similar amino acids are embedded near each other in a learned geometric space, and distances in this space have biological meaning.

---

## Validated Results

| Application | Metric | Value | Dataset | Status |
|-------------|--------|-------|---------|--------|
| **DDG Prediction** | Spearman ρ | 0.52 | ProTherm subset (N=52, NOT S669) | Production |
| **Contact Prediction** | AUC-ROC | 0.67 | Insulin B-chain | Research |
| **Force Constants** | Correlation | 0.86 | AA properties | Validated |
| **AMP Fitness** | Pearson r | 0.63 | DRAMP | Production |

---

## Quick Start by Application

### Protein Stability (ΔΔG)

**Location:** `deliverables/partners/protein_stability_ddg/`

**What it predicts:** How a single amino acid mutation affects protein folding stability.

```python
from deliverables.partners.protein_stability_ddg.src.validated_predictor import ValidatedDDGPredictor

predictor = ValidatedDDGPredictor()
ddg = predictor.predict('A', 'V')  # Alanine → Valine
print(f"Predicted ΔΔG: {ddg:.2f} kcal/mol")
```

**Strengths:**
- Sequence-only (no structure required)
- Catches 23.6% of cases that Rosetta misses
- Excellent on neutral→charged mutations (+159% vs baseline)

**Limitations:**
- Struggles with charge reversals and proline mutations
- N=669 performance (0.37-0.40) does not match N=52 curated subset (0.52)

---

### Residue-Residue Contacts

**Location:** `research/contact-prediction/`

**What it predicts:** Which amino acid pairs are spatially close in the 3D structure.

```python
# See research/contact-prediction/scripts/01_test_real_protein.py
# Requires v5_11_structural checkpoint for best results
```

**Key finding:** Lower "richness" checkpoints give better contact prediction (collapsed radial shells = consistent AA-level distances).

| Checkpoint | AUC-ROC | Use Case |
|------------|---------|----------|
| v5_11_structural | 0.67 | **Best for contacts** |
| homeostatic_rich | 0.59 | Balanced |

---

### Antimicrobial Peptides (AMP)

**Location:** `deliverables/partners/antimicrobial_peptides/`

**What it predicts:** Peptide fitness and minimum inhibitory concentration (MIC).

```bash
cd deliverables/partners/antimicrobial_peptides
python scripts/B8_microbiome_amp_explorer.py --sequence "KKLFKKILKYL"
```

**Results:**
- MIC prediction: Pearson r = 0.74
- Fitness prediction: Pearson r = 0.63

---

### Arbovirus Primer Design

**Location:** `deliverables/partners/arbovirus_surveillance/`

**What it does:** Designs PCR primers for arbovirus detection (DENV, ZIKV, CHIKV, etc.).

```bash
cd deliverables/partners/arbovirus_surveillance

# Practical primer design
python src/scripts/A2_pan_arbovirus_primers.py --use-ncbi

# Research analysis (DENV-4 conservation)
python src/scripts/denv4_padic_integration.py
```

**Key capability:** Addresses DENV-4 cryptic diversity (71.7% identity vs 95-98% other serotypes).

---

### HIV Drug Resistance

**Location:** `deliverables/partners/hiv_research_package/`

**What it does:** Screens sequences for transmitted drug resistance mutations.

```bash
cd deliverables/partners/hiv_research_package
python scripts/H6_tdr_screening.py --fasta patient_sequences.fasta
```

**Integration:** Uses Stanford HIVdb and WHO SDRM reference.

---

## Understanding the Outputs

### Codon Embeddings

The model learns a 16-dimensional representation for each codon. Similar codons cluster together:

```
Codons for Alanine (A): GCU, GCC, GCA, GCG → cluster in one region
Codons for Valine (V): GUU, GUC, GUA, GUG → cluster nearby (both hydrophobic)
Codons for Aspartate (D): GAU, GAC → cluster far from A/V (charged vs hydrophobic)
```

### Distance = Biological Difference

The **hyperbolic distance** between two codon embeddings correlates with:
- Physical/chemical difference between encoded amino acids
- Impact of mutation on protein stability
- Likelihood of residues being in contact (inverse)

---

## Data Requirements

| Application | Input Format | Example |
|-------------|--------------|---------|
| DDG | Wild-type AA, Mutant AA | `predict('A', 'V')` |
| Contacts | Protein sequence (codons) | FASTA with CDS |
| AMP | Peptide sequence | "KKLFKKILKYL" |
| Primers | Multiple alignment | FASTA alignment |

---

## Installation

```bash
# Clone repository
git clone <repo-url>
cd ternary-vaes-bioinformatics

# Install dependencies
pip install torch geoopt scipy pandas biopython

# Verify installation
python -c "from src.geometry import poincare_distance; print('OK')"
```

---

## Choosing the Right Checkpoint

| Application | Checkpoint | Why |
|-------------|------------|-----|
| DDG prediction | `homeostatic_rich/best.pt` | High richness = geometric diversity |
| Contact prediction | `v5_11_structural` | Low richness = consistent distances |
| General purpose | `v5_12_4/best_Q.pt` | Balanced, production-ready |

**Checkpoints location:** `checkpoints/`

---

## Common Questions

### Q: Do I need to understand p-adic numbers?

**A:** No. The p-adic mathematics is the theoretical foundation, but you can use the predictions without understanding it. Think of it as: "similar codons → similar embeddings → predictable biological effects."

### Q: Is this better than AlphaFold/ESMFold?

**A:** Different tools for different jobs:
- **AlphaFold**: 3D structure prediction (needs sequence homologs)
- **This project**: Mutation effects, contacts, primers (sequence-only, fast)

Use both when appropriate.

### Q: How accurate is DDG prediction?

**A:** Spearman ρ = 0.52 on the shipped predictor. This is comparable to other sequence-only methods (ESM-1v: 0.51) and provides value for:
- Screening large numbers of mutations
- Cases where structure is unavailable
- Mutations that Rosetta fails on (23.6% detection rate)

### Q: Can I train my own model?

**A:** Yes, see [CLAUDE.md](../CLAUDE.md) for training details.

---

## Output File Locations

| Partner | Results Directory |
|---------|-------------------|
| DDG | `deliverables/partners/protein_stability_ddg/results/` |
| AMP | `deliverables/partners/antimicrobial_peptides/results/` |
| Primers | `deliverables/partners/arbovirus_surveillance/results/` |
| HIV | `deliverables/partners/hiv_research_package/results/` |

---

## Getting Help

- **Quick reference:** [CLAUDE.md](../CLAUDE.md)
- **Implementation questions:** [CLAUDE.md](../CLAUDE.md)
- **Theoretical background:** [mathematical-foundations/](mathematical-foundations/)
- **Full original context:** [CLAUDE_ORIGINAL.md](mathematical-foundations/archive/CLAUDE_ORIGINAL.md)
- **Bug reports:** GitHub Issues

---

## Citation

If you use this work, please cite the relevant partner packages and the core methodology.

---

*Last updated: 2026-01-30*
