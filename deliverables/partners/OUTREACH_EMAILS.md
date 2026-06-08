# Outreach Email Templates

**Doc-Type:** Email Templates · Version 1.0 · 2026-01-26 · AI Whisperers

Ready-to-send email templates for each research domain. Copy, customize, and send.

---

## Quick Reference

| Domain | Package | Key Metric | Primary Audience |
|--------|---------|------------|------------------|
| Protein Stability | `protein_stability_ddg/` | LOO rho=0.585 | Structural biologists, protein engineers |
| Arbovirus Surveillance | `arbovirus_surveillance/` | 7 viruses covered | Epidemiologists, public health labs |
| Antimicrobial Peptides | `antimicrobial_peptides/` | Spearman=0.656 | Antibiotic researchers, pharma |
| HIV Clinical Tools | `hiv_research_package/` | Stanford HIVdb integrated | HIV clinicians, treatment centers |

---

## 1. Protein Stability (DDG) Prediction

### Subject Lines (choose one)
- Sequence-only DDG prediction - validated on S669 benchmark
- Novel geometric approach to protein stability prediction
- Complement your Rosetta/FoldX workflow with sequence-based screening

### Email Template

```
Subject: Sequence-only DDG prediction - validated on S669 benchmark

Dear [Dr./Prof. Name],

I came across your work on [specific paper/topic] and thought you might be interested in a new approach to protein stability prediction we've developed.

Our tool uses p-adic geometric embeddings to predict DDG from sequence alone - no 3D structure required. Key results:

- LOO Spearman rho = 0.585 (p < 0.001) on ProTherm benchmark subset (N=52, not comparable to S669 N=669)
- <0.1 second per mutation (vs minutes for physics-based methods)
- Detects "Rosetta-blind" instabilities that structure-based methods miss

The tool is particularly useful for:
- High-throughput screening (filter 1000s of candidates before FoldX/Rosetta)
- Cases where no structure is available
- Identifying hidden instability in apparently stable designs

Repository: https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
Package: deliverables/partners/protein_stability_ddg/

Important caveat: On the full N=669 dataset, our method achieves rho=0.37-0.40, which is competitive but not superior to ESM-1v (0.51) or FoldX (0.48). The N=52 curated subset result should not be directly compared.

Would you be interested in trying it on your data? Happy to discuss the methodology or help with integration.

Best regards,
[Your name]
AI Whisperers Team
```

### Contact Categories
- Protein engineers (rational design)
- Computational biologists (stability prediction)
- Pharmaceutical researchers (therapeutic proteins)
- Enzyme engineers (industrial applications)

---

## 2. Arbovirus Surveillance

### Subject Lines (choose one)
- Pan-arbovirus primer design tool + DENV-4 diversity discovery
- Hyperbolic trajectory forecasting for dengue surveillance
- New tool: RT-PCR primers for 7 arboviruses with evolutionary tracking

### Email Template

```
Subject: Pan-arbovirus primer design tool + DENV-4 diversity discovery

Dear [Dr./Prof. Name],

Your work on [arbovirus surveillance/dengue diagnostics/tropical disease monitoring] caught my attention. We've developed a toolkit that might complement your surveillance efforts.

Our package includes:
1. Pan-arbovirus RT-PCR primer design (DENV-1/2/3/4, Zika, Chikungunya, Mayaro)
2. Hyperbolic trajectory forecasting for serotype dominance prediction
3. Primer stability scanning to identify mutation-resistant targets

Key discovery: Our analysis revealed that 97.4% of DENV-4 sequences have NO conserved 25bp windows - explaining why DENV-4 is often missed in diagnostic panels. This "cryptic diversity" (71.7% identity vs 95-98% for other serotypes) requires clade-specific detection strategies.

The tools work with NCBI sequences and integrate with standard surveillance workflows.

Repository: https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
Package: deliverables/partners/arbovirus_surveillance/

Would you be interested in testing this on your regional sequences? Particularly relevant for South American surveillance programs where DENV-4 circulation is increasing.

Best regards,
[Your name]
AI Whisperers Team
```

### Contact Categories
- Public health laboratories (mosquito-borne disease)
- Epidemiologists (dengue forecasting)
- Diagnostic developers (PCR panel design)
- WHO/PAHO collaborators (surveillance programs)

---

## 3. Antimicrobial Peptide Design

### Subject Lines (choose one)
- Multi-objective AMP optimization in VAE latent space
- NSGA-II peptide design for WHO priority pathogens
- Microbiome-safe antimicrobial peptide discovery tool

### Email Template

```
Subject: Multi-objective AMP optimization in VAE latent space

Dear [Dr./Prof. Name],

I noticed your research on [antimicrobial peptides/antibiotic resistance/AMP discovery] and wanted to share a tool we've developed for peptide optimization.

Our package uses NSGA-II multi-objective optimization in a VAE latent space to design AMPs that balance:
- Antimicrobial activity (validated: Spearman rho = 0.656)
- Toxicity (heuristic-based)
- Synthesis feasibility

Three specialized tools included:
1. B1: Pathogen-specific design (A. baumannii, P. aeruginosa, Enterobacteriaceae, S. aureus, H. pylori)
2. B8: Microbiome-safe AMPs (selectivity index > 1.0 for pathogen vs commensal)
3. B10: Synthesis-optimized peptides (aggregation, racemization, cost prediction)

Important limitation: Pseudomonas and Staphylococcus models have insufficient data (p > 0.1). Acinetobacter, Escherichia, and general models are well-validated.

Repository: https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
Package: deliverables/partners/antimicrobial_peptides/

Would this be useful for your peptide discovery pipeline? Happy to discuss the NSGA-II approach or help validate candidates.

Best regards,
[Your name]
AI Whisperers Team
```

### Contact Categories
- Antibiotic researchers (AMR crisis)
- Pharmaceutical companies (peptide therapeutics)
- Academic labs (AMP discovery)
- Microbiome researchers (selective antimicrobials)

---

## 4. HIV Clinical Decision Support

### Subject Lines (choose one)
- TDR screening + LA injectable eligibility tools for HIV clinics
- Clinical decision support for HIV treatment optimization
- Stanford HIVdb-integrated resistance screening tools

### Email Template

```
Subject: TDR screening + LA injectable eligibility tools for HIV clinics

Dear [Dr./Prof. Name],

Your work on [HIV treatment/drug resistance/clinical care] prompted me to reach out about clinical decision support tools we've developed.

Our HIV package includes:
1. H6: Transmitted Drug Resistance (TDR) screening for treatment-naive patients
   - WHO-defined SDRM detection
   - First-line regimen recommendations
   - Full drug susceptibility report

2. H7: Long-Acting Injectable (CAB/RPV-LA) eligibility assessment
   - Viral suppression verification
   - Resistance risk evaluation
   - Pharmacokinetic adequacy scoring
   - Adherence history integration

Both tools integrate with Stanford HIVdb for mutation interpretation and can connect to EMR systems for clinical workflow integration.

Repository: https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
Package: deliverables/partners/hiv_research_package/

Would these tools be useful in your clinical setting? We're looking for pilot sites to validate real-world performance.

Best regards,
[Your name]
AI Whisperers Team
```

### Contact Categories
- HIV treatment centers
- Clinical researchers (ART optimization)
- Public health programs (TDR surveillance)
- Pharmaceutical (LA injectable programs)

---

## Follow-Up Templates

### No Response After 1 Week

```
Subject: Re: [Original subject line]

Dear [Dr./Prof. Name],

Following up on my previous email about [tool name]. I know you're busy, so I'll keep this brief:

The tool is freely available at: https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics

If you'd prefer a 15-minute demo call instead of exploring on your own, I'm happy to arrange that.

Best regards,
[Your name]
```

### Positive Response

```
Subject: Re: [Original subject line]

Dear [Dr./Prof. Name],

Great to hear you're interested! Here's how to get started:

1. Clone the repository: git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
2. Navigate to: deliverables/partners/[package_name]/
3. Follow the Quick Start in README.md

I'm also available for:
- A 30-minute walkthrough call
- Help integrating with your existing pipeline
- Custom analysis on your data

What would be most helpful?

Best regards,
[Your name]
```

---

## Contact List References

See these files for full contact lists by domain:

| File | Contents |
|------|----------|
| [CONTACT_LIST_200_RESEARCHERS.md](CONTACT_LIST_200_RESEARCHERS.md) | 200 contacts across 8 categories |
| [CONTACTS_VERIFIED_EMAILS.csv](CONTACTS_VERIFIED_EMAILS.csv) | CSV for mail merge |
| [OUTREACH_PITCH_DECKS.md](OUTREACH_PITCH_DECKS.md) | Detailed pitch materials |

---

## Email Best Practices

1. **Personalize the first line** - Reference their specific work
2. **Be honest about limitations** - Note caveats upfront
3. **Provide clear next steps** - Link to repo, offer demo
4. **Keep it brief** - Under 200 words for initial contact
5. **Follow up once** - If no response after 1 week
6. **Track responses** - Note who engages for future releases

---

*Last updated: 2026-01-26*
