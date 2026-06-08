# Pre-Send Validation Checklist

**Purpose:** Verify claims in each email category are backed by validated evidence before sending.

**Last Updated:** 2026-01-26

---

## Category: ARBOVIRUS (Emails 04, 13-15, 46-60)

### Validation Status: ✅ READY TO SEND

| Claim | Evidence | File | Verified |
|-------|----------|------|:--------:|
| DENV-4 has 71.7% identity (vs 95-98% others) | Pairwise divergence analysis | `arbovirus_surveillance/VALIDATION_MASTER_REPORT.md` | ✅ |
| 97.4% of DENV-4 genome lacks conserved windows | Multi-strain entropy analysis | `arbovirus_surveillance/results/` | ✅ |
| Pan-arbovirus primers for 7 viruses | Primer CSV files exist | `arbovirus_surveillance/pan_arbovirus_primers/` | ✅ |
| CDC primer recovery 60% | In-silico PCR validation | `arbovirus_surveillance/VALIDATION_MASTER_REPORT.md` | ✅ |
| P-adic integration with TrainableCodonEncoder | Checkpoint used | `research/codon-encoder/training/results/` | ✅ |

### Pre-Send Checks:
- [x] Scientific finding is novel and validated
- [x] No overstated claims
- [x] Multiple orthogonal validation approaches documented
- [x] Emails reference specific, verifiable numbers

### Notes:
This is the strongest package. The DENV-4 cryptic diversity discovery explains a real diagnostic gap.

---

## Category: PROTEIN STABILITY / DDG (Emails 01, 05-09, 17-30)

### Validation Status: ⚠️ NEEDS CAVEAT ADDED

| Claim | Evidence | File | Verified | Caveat Needed |
|-------|----------|------|:--------:|:-------------:|
| LOO Spearman ρ=0.585 | Bootstrap validation | `protein_stability_ddg/validation/results/scientific_metrics.json` | ✅ | **N=52 only** |
| p-value < 0.001 | Permutation test | Same file | ✅ | - |
| 95% CI [0.34, 0.77] | Bootstrap n=1000 | Same file | ✅ | - |
| "Catches mutations Rosetta misses" | 12/52 discordant cases | `protein_stability_ddg/rosetta_blind/` | ✅ | Honest |
| Competitive with literature | **NOT TRUE on N=669** | `VALIDATION_SUMMARY.md` | ⚠️ | **Must clarify** |

### Pre-Send Checks:
- [x] Core metric (ρ=0.585) is real and reproducible
- [ ] **Emails must state "N=52 curated subset"**
- [ ] **Do NOT imply competitive with Rosetta (0.69) or ESM-1v (0.51) on full dataset**
- [x] Discordant cases claim is honest (different failure modes)

### Required Email Edits:
Add to relevant emails: "on a ProTherm benchmark subset (N=52) — not a subset of S669 by Pancotti et al. 2022"

### Honest Framing:
- ✅ "Different failure modes than Rosetta" - TRUE
- ✅ "Sequence-only, no structure needed" - TRUE
- ⚠️ "Competitive performance" - FALSE on N=669 (ρ=0.37-0.40)

---

## Category: ANTIMICROBIAL PEPTIDES (Emails 02, 10, 31-45)

### Validation Status: ⚠️ PARTIAL - SOME CLAIMS UNSUPPORTED

| Claim | Evidence | File | Verified | Issue |
|-------|----------|------|:--------:|:-----:|
| Mean Spearman ρ=0.656 | Cross-validation | `antimicrobial_peptides/cv_results_definitive.json` | ✅ | Pooled only |
| E. coli r=0.39 | Validation batch | Same file | ✅ | p<0.001 |
| Acinetobacter r=0.52 | Validation batch | Same file | ✅ | p=0.019 |
| **Pseudomonas prediction** | r=0.05 | Same file | ❌ | **p=0.82** |
| **Staphylococcus prediction** | r=0.17 | Same file | ❌ | **p=0.15** |
| Toxicity prediction | Heuristic only | - | ⚠️ | Not ML model |
| Synthesis difficulty | Heuristic only | - | ⚠️ | Uncalibrated |

### Pre-Send Checks:
- [x] General/pooled MIC prediction works (ρ=0.656)
- [ ] **Do NOT claim pathogen-specific accuracy for Pseudomonas or Staphylococcus**
- [ ] **Clarify toxicity/synthesis are heuristics, not validated models**
- [x] E. coli and Acinetobacter claims are solid

### Required Email Edits:
- Change "MIC prediction" to "MIC prediction for E. coli and general datasets"
- Add caveat: "Pathogen-specific models under development"

### Honest Framing:
- ✅ "Novel embedding approach" - TRUE
- ✅ "E. coli MIC prediction" - TRUE (r=0.39, p<0.001)
- ⚠️ "All pathogens" - FALSE (2/5 non-significant)

---

## Category: HIV RESEARCH (Emails 03, 11-12, 61-75)

### Validation Status: ⚠️ READY WITH CAVEATS

| Claim | Evidence | File | Verified | Notes |
|-------|----------|------|:--------:|:-----:|
| Stanford HIVdb API integrated | API tested | Direct GraphQL test | ✅ | **API v10.1 (2026-01-18)** |
| TDR screening functional | H6 script | `scripts/H6_tdr_screening.py` | ✅ | Works with real NT sequences |
| Demo mode | Local fallback | Same script | ✅ | Graceful fallback |
| Batch processing | FASTA support | stanford_client.py | ✅ | analyze_fasta_file() |

### API Verification (2026-01-26):
```
Status: CONNECTED
Version: HIVDB_10.1 (published 2026-01-18)
Endpoint: https://hivdb.stanford.edu/graphql
```

### Pre-Send Checks:
- [x] Stanford HIVdb API connectivity verified
- [x] GraphQL query type fixed (mutation → query)
- [x] Real HIV sequences analyzed successfully
- [ ] Minor response parsing issues (non-blocking)

### Remaining Minor Issues (Non-blocking):
1. ResistanceLevel.from_text() expects string, API returns int - needs type coercion
2. Demo sequences could use more variety
3. Mutation attribute mismatch between scripts/models

### Recommendation:
**CAN SEND** - API integration is verified. The email to Robert Shafer (HIVdb maintainer) is appropriate - we're asking for guidance, not claiming perfection.

---

## Category: VAE / HYPERBOLIC ML (Emails 07, 76-90)

### Validation Status: ✅ READY TO SEND

| Claim | Evidence | File | Verified |
|-------|----------|------|:--------:|
| Hierarchy correlation ρ=-0.83 | Training logs | `checkpoints/v5_12_4/` | ✅ |
| 100% coverage (19,683 ops) | Model evaluation | Same | ✅ |
| Ultrametric compliance 82.8% | V5.5 analysis | `checkpoints/v5_5/V5_5_ANALYSIS.md` | ✅ |
| Contact prediction AUC=0.67 | Insulin validation | `research/contact-prediction/` | ✅ |

### Pre-Send Checks:
- [x] Core VAE metrics reproducible
- [x] Hyperbolic geometry claims accurate
- [x] P-adic theory connection valid

---

## Category: P-ADIC ML THEORY (Emails 16, 91-100)

### Validation Status: ✅ READY TO SEND

| Claim | Evidence | File | Verified |
|-------|----------|------|:--------:|
| 3-adic valuation → radial ordering | Training achieves monotonic ordering | Model checkpoints | ✅ |
| P-adic + Poincaré combination | Novel theoretical contribution | Architecture docs | ✅ |
| Downstream biological predictions | DDG, contact, AMP results | Various | ✅ |

### Pre-Send Checks:
- [x] Theoretical claims are novel
- [x] Mathematical framework is sound
- [x] Practical applications demonstrated

---

## Send Priority Summary

| Priority | Category | Emails | Status | Action |
|:--------:|----------|--------|:------:|--------|
| **1** | Arbovirus | 04, 13-15, 46-60 | ✅ SEND | Ready now |
| **2** | Hyperbolic/P-adic ML | 07, 16, 86-100 | ✅ SEND | Ready now |
| **3** | Protein DDG | 01, 05-09, 17-30 | ⚠️ EDIT | Add N=52 caveat |
| **4** | AMP | 02, 10, 31-45 | ⚠️ EDIT | Clarify pathogen scope |
| **5** | HIV | 03, 11-12, 61-75 | ❌ HOLD | Verify API first |

---

## Quick Reference: What NOT to Claim

| Category | Avoid Saying | Say Instead |
|----------|--------------|-------------|
| DDG | "Competitive with Rosetta" | "Different failure modes than Rosetta" |
| DDG | "ρ=0.585" (alone) | "ρ=0.585 on N=52 curated subset" |
| AMP | "All pathogens" | "E. coli and general datasets" |
| AMP | "Validated toxicity prediction" | "Toxicity heuristics" |
| HIV | "Complete and tested" | "Integration ready for validation" |

---

*This checklist must be reviewed before each email batch is sent.*
