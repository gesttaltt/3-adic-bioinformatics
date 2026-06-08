# Publication Figures

Main paper figures for the Ultrametric Antigen AI / Ternary VAE project.

## Figure Index

| File | Title | Section | Generator |
|------|-------|---------|-----------|
| `figure_1_padic_hierarchy.png` | P-adic Valuation Hierarchy in Hyperbolic Space | Methods — Geometry | `src/scripts/visualization/analyze_3adic_deep.py` |
| `figure_2_hiv_resistance.png` | HIV Drug Resistance Prediction (Geometric Isolation) | Results — HIV | `src/scripts/visualization/` |
| `figure_3_amp_pareto.png` | AMP Design Pareto Front (NSGA-II) | Results — AMP | `deliverables/partners/antimicrobial_peptides/scripts/B1_pathogen_specific_design.py` |
| `figure_4_arbovirus_conservation.png` | Arbovirus Genomic Conservation & DENV-4 Diversity | Results — Arbovirus | `deliverables/partners/arbovirus_surveillance/scripts/A2_pan_arbovirus_primers.py` |
| `figure_5_rosetta_blind.png` | Rosetta-Blind Mutation Detection | Results — DDG | `deliverables/partners/protein_stability_ddg/scripts/C1_rosetta_blind_detection.py` |
| `figure_6_codon_physics.png` | Emergent Physical Properties in Codon Embeddings | Discussion | `src/scripts/visualization/analyze_3adic_deep.py` |

## Related Figure Sets

| Location | Contents |
|----------|----------|
| `research/results/publication/` | HIV paper figures (figure1–4 PDF+PNG, LaTeX tables) |
| `research/results/figures/` | HIV attention/mutation analysis plots |
| `research/diseases/hiv/public_medical_paper/images/` | All 33 HIV paper diagrams (regenerate with `src/visualization/generate_paper_*.py`) |
| `deliverables/partners/arbovirus_surveillance/results/phylogenetic/` | DENV-4 phylogenetic figures |
| `checkpoints/epsilon_vae_analysis/` | Training trajectory and embedding evolution plots |

## Regenerating

```bash
# HIV paper diagrams (charts, diagrams, flowcharts)
python src/visualization/generate_paper_charts.py
python src/visualization/generate_paper_diagrams.py
python src/visualization/generate_paper_flowcharts.py

# Main paper figures
python src/scripts/visualization/analyze_3adic_deep.py
```
