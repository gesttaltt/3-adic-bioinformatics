#!/usr/bin/env python3
"""Generate individual markdown files for all HIV research papers."""

from pathlib import Path

# Base directory
BASE_DIR = Path(
    r"C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\02_THEORY_AND_FOUNDATIONS\09_BIBLIOGRAPHY_AND_RESOURCES\RESEARCH_LIBRARY\HIV_RESEARCH_2024"
)

# Paper data organized by category
PAPERS = {
    "01_CURE_STRATEGIES": [
        (
            "CURE-002",
            "HIV Reservoirs and Treatment Strategies toward Curing HIV Infection",
            "IJMS",
            "2024",
            "https://pubmed.ncbi.nlm.nih.gov/38473868/",
            "Comprehensive review of HIV reservoir biology and therapeutic strategies for achieving a cure, including shock-and-kill, block-and-lock, and gene editing approaches.",
        ),
        (
            "CURE-003",
            "HIV Persistence, Latency, and Cure Approaches: Where Are We Now?",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11281696/",
            "Current state of HIV cure research examining latent reservoir characteristics, persistence mechanisms, and emerging therapeutic interventions.",
        ),
        (
            "CURE-005",
            "Geneva Patient: Stem Cell Transplant Cure Case",
            "AIDS 2024",
            "2024",
            "https://www.sciencedaily.com/releases/2024/03/240326124555.htm",
            "Case report of the seventh person potentially cured of HIV through stem cell transplantation with CCR5-delta32 donor cells.",
        ),
        (
            "CURE-006",
            "HIV-Virus-Like-Particle (HLP) Therapeutic Candidate",
            "Schulich/Bristol",
            "2024",
            "https://www.sciencedaily.com/releases/2024/03/240326124555.htm",
            "Novel therapeutic candidate HLP shown to be 100 times more effective than other HIV cure candidates for chronically infected individuals on ART.",
        ),
        (
            "CURE-007",
            "Vorinostat and Immunotherapy Reservoir Reduction Trial",
            "UNC",
            "2024",
            "https://news.unchealthcare.org/2024/02/new-trial-highlights-incremental-progress-towards-a-cure-for-hiv-1/",
            "Clinical trial results showing vorinostat combined with immunotherapy may modestly reduce the latent HIV reservoir.",
        ),
        (
            "CURE-008",
            "Long-Term HIV Control: Combination Therapy Study",
            "UCSF",
            "2025",
            "https://www.ucsf.edu/news/2025/12/431136/long-term-hiv-control-could-combination-therapy-be-key",
            "UCSF study showing experimental immunotherapy can help control HIV without long-term ART in 7 of 10 participants.",
        ),
        (
            "CURE-009",
            "Block and Lock Strategy for Deep Latency",
            "Review",
            "2024",
            "https://avac.org/prevention-option/cure/",
            "Review of block-and-lock approaches using small molecules to induce permanent deep latency state in HIV-infected cells.",
        ),
        (
            "CURE-010",
            "TCR Agents for Viral Reservoir Elimination",
            "Biology Insights",
            "2025",
            "https://biologyinsights.com/hiv-cure-2025-new-strategies-to-eliminate-viral-reservoirs/",
            "Emerging T cell receptor agents designed to enhance recognition and destruction of cells harboring latent HIV.",
        ),
        (
            "CURE-011",
            "Advancing CRISPR Genome Editing into Clinical Trials",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC12094669/",
            "Progress review of CRISPR-based gene therapy approaches moving from preclinical to clinical HIV cure trials.",
        ),
        (
            "CURE-012",
            "Gene Therapy for HIV: CRISPR/Cas9 B Cell Modification",
            "WashU",
            "2024",
            "https://medicine.washu.edu/news/6-2-million-to-help-develop-gene-therapy-for-hiv/",
            "Novel approach using CRISPR/Cas9 to modify B cells for HIV gene therapy, distinct from T cell targeting strategies.",
        ),
    ],
    "02_VACCINES": [
        (
            "VAX-002",
            "IAVI G002/G003 Trials: bnAb Pathway Proof-of-Concept",
            "IAVI/Scripps",
            "2025",
            "https://www.iavi.org/press-release/two-hiv-vaccine-trials-show-proof-of-concept-for-pathway-to-broadly-neutralizing-antibodies/",
            "Two Phase 1 trials demonstrating targeted vaccine strategy can activate immune responses relevant to broadly neutralizing antibody development.",
        ),
        (
            "VAX-003",
            "NIH HVTN 302: Three mRNA HIV Vaccines Clinical Trial",
            "NIH",
            "2024",
            "https://www.nih.gov/news-events/news-releases/nih-launches-clinical-trial-three-mrna-hiv-vaccines",
            "NIAID Phase 1 trial evaluating three experimental mRNA-based HIV vaccines for safety and immunogenicity.",
        ),
        (
            "VAX-004",
            "Progress and Challenges in HIV-1 Vaccine Research",
            "Vaccines (MDPI)",
            "2025",
            "https://www.mdpi.com/2076-393X/13/2/148",
            "Comprehensive review of HIV-1 vaccine development challenges including viral diversity, mutation rates, and latent reservoirs.",
        ),
        (
            "VAX-005",
            "Vaccination Induces bnAb Precursors to HIV gp41",
            "Nature Immunology",
            "2024",
            "https://www.nature.com/articles/s41590-024-01833-w",
            "Development of germline-targeting epitope scaffolds to elicit 10E8-class broadly neutralizing antibody precursors.",
        ),
        (
            "VAX-006",
            "Vaccine Induction of Heterologous HIV-1-Neutralizing Antibody Lineages",
            "Cell",
            "2024",
            "https://www.cell.com/cell/fulltext/S0092-8674(24)00459-8",
            "Breakthrough showing vaccines can induce B cell lineages of broadly neutralizing antibodies in humans.",
        ),
        (
            "VAX-007",
            "Novel Vaccine Concept for Multiple Types of HIV bnAbs",
            "NIH",
            "2024",
            "https://www.nih.gov/news-events/news-releases/novel-vaccine-concept-generates-immune-responses-could-produce-multiple-types-hiv-broadly-neutralizing-antibodies",
            "Novel immunogen design generating immune responses for multiple bnAb types simultaneously.",
        ),
        (
            "VAX-008",
            "2024 HIV Vaccines and Passive Immunization Pipeline",
            "TAG",
            "2024",
            "https://www.treatmentactiongroup.org/wp-content/uploads/2024/07/pipeline_HIV_VAX_2024_final.pdf",
            "Annual pipeline report on HIV vaccine candidates and passive immunization approaches in development.",
        ),
        (
            "VAX-009",
            "Germline-Targeting Epitope Scaffold Nanoparticles",
            "Nature Immunology",
            "2024",
            "https://www.nature.com/articles/s41590-024-01833-w",
            "Engineered nanoparticles for multivalent display of germline-targeting epitopes to activate bnAb precursor B cells.",
        ),
        (
            "VAX-010",
            "mRNA Vaccine Urticaria Safety Analysis",
            "IAVI",
            "2025",
            "https://www.eatg.org/hiv-news/dual-studies-reveal-early-successes-for-mrna-hiv-vaccine-strategies/",
            "Safety analysis of urticaria reactions in mRNA HIV vaccine trials and mitigation strategies.",
        ),
    ],
    "03_DRUG_RESISTANCE": [
        (
            "RES-002",
            "HIV-1 Drug Resistance Trends 2018-2024",
            "OFID",
            "2024",
            "https://academic.oup.com/ofid/article/12/8/ofaf446/8218172",
            "Large-scale retrospective analysis of drug resistance mutation prevalence trends over six years.",
        ),
        (
            "RES-003",
            "Development and Trends of Drug Resistance Mutations: Bibliometric Analysis",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1374582/full",
            "Bibliometric analysis of HIV drug resistance research landscape and emerging trends.",
        ),
        (
            "RES-004",
            "Next-Gen HIV Genotyping for 28 Antiretroviral Drugs",
            "CID",
            "2024",
            "https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciaf458/8237671",
            "Design and evaluation of comprehensive genotyping assay detecting resistance to 28 ARVs across 5 drug classes including lenacapavir.",
        ),
        (
            "RES-005",
            "Drug Resistance in Low-Level Viremia (Zhengzhou Study)",
            "Sci Rep",
            "2024",
            "https://www.nature.com/articles/s41598-024-60965-z",
            "Characterization of drug resistance mutations in ART-experienced patients with low-level viremia (50-999 copies/mL).",
        ),
        (
            "RES-006",
            "HIV Drug Resistance in MENA Region: Systematic Review",
            "Virology J",
            "2025",
            "https://virologyj.biomedcentral.com/articles/10.1186/s12985-025-02740-8",
            "Systematic review evaluating HIV drug resistance patterns and associated factors in Middle East and North Africa.",
        ),
        (
            "RES-007",
            "DTG Resistance Emerging Beyond Clinical Trial Levels",
            "WHO",
            "2024",
            "https://www.who.int/publications-detail-redirect/9789240086319",
            "WHO analysis showing dolutegravir resistance emerging at 3.9-19.6% in real-world settings, exceeding clinical trial observations.",
        ),
        (
            "RES-008",
            "Lenacapavir Resistance Mutations Q67H + K70R",
            "Clinical",
            "2024",
            "https://www.iasusa.org/hiv-drug-resistance/",
            "First detection of major lenacapavir resistance mutations conferring high-level resistance after third injection.",
        ),
        (
            "RES-009",
            "HIV Replication Under High-Level Cabotegravir",
            "Viruses (MDPI)",
            "2024",
            "https://www.mdpi.com/1999-4915/16/12/1874",
            "Analysis of 3'-PPT mutations, circular DNA transcription, and recombination associated with cabotegravir resistance.",
        ),
        (
            "RES-010",
            "ROSETTA Registry: Second-Gen INSTI Resistance",
            "HIV Glasgow",
            "2024",
            "https://www.eatg.org/hiv-news/integrase-inhibitor-resistance-after-treatment-failure-more-common-in-treatment-experienced-people/",
            "European registry data on second-generation integrase inhibitor resistance patterns at virological failure.",
        ),
    ],
    "04_TREATMENT": [
        (
            "TRT-002",
            "PURPOSE 1 Trial: 100% HIV Prevention in Women",
            "Gilead",
            "2024",
            "https://www.prepwatch.org/products/lenacapavir-for-prep/",
            "Landmark trial showing zero HIV infections among 2,134 women receiving twice-yearly lenacapavir in Africa.",
        ),
        (
            "TRT-003",
            "PURPOSE 2 Trial: 96% Prevention in MSM/Trans",
            "Gilead",
            "2024",
            "https://avac.org/lenacapavir/",
            "Phase 3 trial demonstrating 96% reduction in HIV acquisition among MSM, transgender, and gender non-binary individuals.",
        ),
        (
            "TRT-004",
            "WHO Recommends Injectable Lenacapavir",
            "WHO",
            "2025",
            "https://www.who.int/news/item/14-07-2025-who-recommends-injectable-lenacapavir-for-hiv-prevention",
            "WHO global recommendation for lenacapavir as HIV prevention with new implementation guidelines.",
        ),
        (
            "TRT-005",
            "Once-Yearly Lenacapavir Phase 1 Trial",
            "CROI 2025",
            "2025",
            "https://www.gilead.com/news/news-details/2025/first-clinical-data-for-gileads-investigational-once-yearly-lenacapavir-for-hiv-prevention-presented-at-croi-2025-and-published-in-the-lancet",
            "Phase 1 data showing once-yearly intramuscular lenacapavir maintains therapeutic levels for 56+ weeks.",
        ),
        (
            "TRT-006",
            "Islatravir + Lenacapavir Once-Weekly Oral",
            "ID Week 2024",
            "2024",
            "https://www.aidsmap.com/news/mar-2024/islatravir-plus-lenacapavir-could-be-first-once-weekly-oral-hiv-treatment",
            "Phase 2 trial showing 94% viral suppression at 48 weeks with novel once-weekly oral combination.",
        ),
        (
            "TRT-007",
            "Doravirine/Islatravir Phase 3 Trial Results",
            "Merck",
            "2024",
            "https://www.merck.com/news/merck-announces-positive-topline-results-from-the-pivotal-phase-3-trial-evaluating-investigational-once-daily-oral-two-drug-single-tablet-regimen-of-doravirine-islatravir-dor-isl-in-treatment-na/",
            "Positive Phase 3 results for once-daily two-drug regimen of doravirine/islatravir in treatment-naive adults.",
        ),
        (
            "TRT-008",
            "VH4524184 Third-Generation INSTI Phase 1",
            "CID",
            "2024",
            "https://academic.oup.com/cid/article/81/3/510/8090171",
            "Phase 1 evaluation of third-generation integrase inhibitor with enhanced resistance profile and long-acting potential.",
        ),
        (
            "TRT-009",
            "PASO-DOBLE: Dovato vs Biktarvy Head-to-Head",
            "ViiV",
            "2024",
            "https://viivhealthcare.com/en-us/media-center/news/press-releases/2024/july/viiv-healthcare-to-announce-data-from-largest-head-to-head-randomised/",
            "Largest head-to-head randomized trial comparing 2-drug Dovato vs 3-drug Biktarvy regimens.",
        ),
        (
            "TRT-010",
            "LATITUDE Trial: Cabenuva Superior Efficacy",
            "ViiV",
            "2024",
            "https://viivhealthcare.com/hiv-news-and-media/news/press-releases/2024/march/interim-data-at-croi-indicating-superior-efficacy/",
            "Phase 3 interim data showing Cabenuva superior to daily oral therapy in patients with adherence challenges.",
        ),
        (
            "TRT-011",
            "Expanding Injectable Antiretrovirals Review",
            "Infect Dis Ther",
            "2024",
            "https://link.springer.com/article/10.1007/s40121-024-01062-6",
            "Comprehensive review of current and investigational injectable antiretrovirals for HIV treatment.",
        ),
        (
            "TRT-012",
            "Islatravir Clinical Development for HIV and HBV",
            "PubMed",
            "2024",
            "https://pubmed.ncbi.nlm.nih.gov/38235744/",
            "Evaluation of islatravir's clinical development history, dose optimization, and dual HIV/HBV potential.",
        ),
        (
            "TRT-013",
            "NRTI-Induced Neuropathy: Beyond Poly-gamma Hypothesis",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11677988/",
            "Mechanistic investigation of NRTI neuropathy exploring autophagy and drug transport beyond mitochondrial toxicity.",
        ),
        (
            "TRT-014",
            "Darunavir Resistance: High Genetic Barrier Analysis",
            "JAC",
            "2024",
            "https://academic.oup.com/jac/article/79/2/339/7503104",
            "Detailed characterization of drug resistance during darunavir/ritonavir monotherapy identifying alternative resistance pathways.",
        ),
        (
            "TRT-015",
            "FMO-Guided Design of Darunavir Analogs",
            "Sci Rep",
            "2024",
            "https://www.nature.com/articles/s41598-024-53940-1",
            "Fragment molecular orbital-guided computational design of novel darunavir analogs as HIV-1 protease inhibitors.",
        ),
    ],
    "05_BROADLY_NEUTRALIZING_ANTIBODIES": [
        (
            "BNAB-002",
            "Broadly Neutralizing Antibodies for HIV Prevention: Comprehensive Review",
            "Clin Micro Rev",
            "2024",
            "https://pubmed.ncbi.nlm.nih.gov/38687039/",
            "Comprehensive review of bnAbs for HIV prevention including manufacturing challenges and delivery logistics.",
        ),
        (
            "BNAB-003",
            "The Use of bnAbs in HIV-1 Treatment and Prevention",
            "Viruses (MDPI)",
            "2024",
            "https://www.mdpi.com/1999-4915/16/6/911",
            "Review of bnAb combination therapy strategies and requirements for prolonged viral suppression.",
        ),
        (
            "BNAB-004",
            "The Potential of bnAbs for HIV Prevention",
            "JIAS",
            "2024",
            "https://onlinelibrary.wiley.com/doi/10.1002/jia2.26257",
            "Analysis of bnAb potential for HIV prevention based on AMP trial results and future directions.",
        ),
        (
            "BNAB-005",
            "Enhancing bnAb Suppression by Immune Modulation and Vaccination",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2024.1478703/full",
            "Strategies combining bnAbs with immune modulation and therapeutic vaccination for enhanced HIV control.",
        ),
        (
            "BNAB-006",
            "Neutralizing the Threat: Harnessing bnAbs Against HIV-1",
            "Microbial Cell",
            "2024",
            "http://microbialcell.com/researcharticles/2024a-becerra-microbial-cell/",
            "Review of bnAb role in characterizing neutralization-sensitive sites and informing vaccine development.",
        ),
        (
            "BNAB-007",
            "LS Variants with Extended Half-Life for HIV Prevention",
            "Review",
            "2024",
            "https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2024.1478703/full",
            "Analysis of Fc-modified LS variants with extended half-life for prolonged HIV protection.",
        ),
        (
            "BNAB-008",
            "AMP Trials: VRC01 Proof-of-Concept for Prevention",
            "JIAS",
            "2024",
            "https://onlinelibrary.wiley.com/doi/10.1002/jia2.26257",
            "Lessons from Antibody Mediated Prevention trials demonstrating bnAb-based HIV prevention proof-of-concept.",
        ),
    ],
    "06_STRUCTURAL_BIOLOGY": [
        (
            "STRUCT-002",
            "HIV-1 Polyanion-Dependent Capsid Lattice Formation",
            "PNAS",
            "2023",
            "https://www.pnas.org/doi/10.1073/pnas.2220545120",
            "Cryo-EM structures revealing IP6 role in pentamer formation and lenacapavir binding to CA lattice.",
        ),
        (
            "STRUCT-003",
            "Microsecond Dynamics Control HIV-1 Envelope Conformation",
            "Science Adv",
            "2024",
            "https://www.aps.anl.gov/APS-Science-Highlight/2024-05-10/a-small-but-very-fast-step-toward-a-vaccine-against-hiv",
            "Time-resolved SAXS revealing microsecond envelope dynamics critical for vaccine design.",
        ),
        (
            "STRUCT-004",
            "Native HIV-1 Cores: Structures with IP6 and CypA",
            "Science Adv",
            "2023",
            "https://www.science.org/doi/10.1126/sciadv.abj5715",
            "Cryo-ET structures of native HIV-1 capsid with host factors IP6 and cyclophilin A.",
        ),
        (
            "STRUCT-005",
            "Comprehensive Insights into HIV Glycoproteins",
            "Appl Sci",
            "2024",
            "https://www.mdpi.com/2076-3417/14/18/8271",
            "Review of HIV glycoprotein molecular basis including gp160 processing and membrane incorporation.",
        ),
        (
            "STRUCT-006",
            "gp120 Layered Architecture and Conformational Mobility",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC2824281/",
            "Structural analysis of gp120 beta-sandwich architecture enabling conformational changes for entry and immune evasion.",
        ),
        (
            "STRUCT-007",
            "Structure and Function of HIV Envelope Glycoprotein",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7111665/",
            "Comprehensive review of HIV envelope structure as entry mediator, vaccine immunogen, and drug target.",
        ),
        (
            "STRUCT-008",
            "Lenacapavir Binding to HIV-1 CA Lattice Structure",
            "PNAS",
            "2023",
            "https://www.pnas.org/doi/10.1073/pnas.2220545120",
            "Structural basis for lenacapavir (GS-6207) binding to assembled HIV-1 capsid lattice.",
        ),
    ],
    "07_MOLECULAR_BIOLOGY": [
        (
            "MOL-002",
            "HIV-1 Transcriptional Program: Initiation to Elongation Control",
            "ScienceDirect",
            "2024",
            "https://www.sciencedirect.com/science/article/pii/S0022283624002924",
            "Comprehensive review of HIV-1 transcription regulation by Tat and cellular elongation machinery.",
        ),
        (
            "MOL-003",
            "HIV-1 Tat Protein Interactions with Host Receptors",
            "IJMS",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10855115/",
            "Role of extracellular Tat interactions with host receptors in HIV pathogenesis.",
        ),
        (
            "MOL-004",
            "Tat and P-TEFb Complex in Transcription Elongation",
            "Review",
            "2024",
            "https://www.mdpi.com/1422-0067/25/3/1704",
            "Detailed analysis of Tat-P-TEFb interaction and cyclin T1/CDK9 role in HIV transcription.",
        ),
        (
            "MOL-005",
            "HIV-1 Tat Alters Prefrontal Cortex Neuronal Activity",
            "iScience",
            "2025",
            "https://www.cell.com/iscience/fulltext/S2589-0042(25)00335-9",
            "Tat transgenic mouse studies revealing neuronal effects underlying HIV-associated neurocognitive disorders.",
        ),
        (
            "MOL-006",
            "Schlafen14 Impairs HIV-1 in Codon Usage-Dependent Manner",
            "Viruses",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3245234/",
            "Host restriction factor Schlafen14 inhibits HIV-1 through codon usage-dependent mechanisms.",
        ),
        (
            "MOL-007",
            "Different Codon Usage Patterns Across Primate Lentiviruses",
            "Viruses",
            "2023",
            "https://www.mdpi.com",
            "Comparative analysis of codon usage and amino acid composition across primate lentiviruses.",
        ),
        (
            "MOL-008",
            "HIV Accessory Proteins Vif, Vpr, Vpu, Nef Functions",
            "Mol Med",
            "2024",
            "https://molmed.biomedcentral.com/articles/10.1007/BF03401585",
            "Review of HIV accessory proteins as emerging therapeutic targets and their cellular interactions.",
        ),
        (
            "MOL-009",
            "Nef and Vpu Modulation of Cell Surface Receptors",
            "JVI",
            "2024",
            "https://journals.asm.org/doi/10.1128/jvi.02333-14",
            "Analysis of Nef and Vpu as broad-spectrum modulators of cell surface receptors including tetraspanins.",
        ),
        (
            "MOL-010",
            "Making Sense of Multifunctional HIV Proteins",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5750048/",
            "Understanding HIV accessory and regulatory protein connections to cellular transcription.",
        ),
    ],
    "08_PATHOGENESIS": [
        (
            "PATH-002",
            "Macrophages: Key Cellular Players in HIV Infection",
            "Viruses (MDPI)",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10893316/",
            "Comprehensive review of macrophage role in HIV transmission, dissemination, and reservoir establishment.",
        ),
        (
            "PATH-003",
            "HIV Controllers: Hope for a Functional Cure",
            "Frontiers",
            "2025",
            "https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1540932/full",
            "Analysis of elite and post-treatment controllers as models for functional HIV cure.",
        ),
        (
            "PATH-004",
            "Immune Responses in Controllers of HIV Infection",
            "Ann Rev Immunol",
            "2024",
            "https://pubmed.ncbi.nlm.nih.gov/37827174/",
            "Comprehensive review of CD8+ T cell, antibody, and NK cell mechanisms in HIV controllers.",
        ),
        (
            "PATH-005",
            "Targeting HIV Persistence in Tissue",
            "Curr Opin HIV AIDS",
            "2024",
            "https://journals.lww.com/co-hivandaids/fulltext/2024/03000/targeting_hiv_persistence_in_the_tissue.6.aspx",
            "Challenges in targeting tissue-specific HIV reservoirs and novel therapeutic approaches.",
        ),
        (
            "PATH-006",
            "Pro-inflammatory Macrophages Suppress HIV Replication",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2024.1439328/full",
            "Humanized mouse and ex vivo studies showing pro-inflammatory macrophages can suppress HIV.",
        ),
        (
            "PATH-007",
            "CD4+ T Cell Depletion Mechanisms in HIV",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3729334/",
            "Mechanisms of immunological failure through pyroptosis, apoptosis, and bystander effects.",
        ),
        (
            "PATH-008",
            "Elite Controller Provirus Integration in Non-Coding Regions",
            "Science",
            "2024",
            "https://www.science.org/content/article/how-elite-controllers-tame-hiv-without-drugs",
            "Elite controllers harbor proviruses in transcriptionally silent genome regions conferring deep latency.",
        ),
    ],
    "09_EPIDEMIOLOGY": [
        (
            "EPI-002",
            "Novel CRF159_01103 in Hebei Province, China",
            "Sci Rep",
            "2024",
            "https://www.nature.com/articles/s41598-024-64156-8",
            "Identification of new circulating recombinant form CRF159_01103 derived from CRF103_01B and CRF01_AE.",
        ),
        (
            "EPI-003",
            "How to Report a New HIV-1 Circulating Recombinant Form",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1343143/full",
            "Standardized criteria for classification, nomenclature, and reference for new HIV-1 CRFs.",
        ),
        (
            "EPI-004",
            "Key Populations and HIV-1 Recombinants Meta-Analysis",
            "PMC",
            "2023",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10420084/",
            "Global meta-analysis of associations between key populations and HIV-1 recombinant strains.",
        ),
        (
            "EPI-005",
            "PMTCT Progress: 4.4 Million Infections Prevented",
            "UNICEF",
            "2024",
            "https://data.unicef.org/topic/hivaids/emtct/",
            "UNICEF data showing PMTCT has prevented 4.4 million child HIV infections since 2000.",
        ),
        (
            "EPI-006",
            "Barriers to PMTCT in Western Kenya",
            "AIDS Res Ther",
            "2025",
            "https://link.springer.com/article/10.1186/s12981-025-00779-9",
            "Qualitative study exploring barriers to PMTCT among HIV-positive pregnant women in Kenya.",
        ),
        (
            "EPI-007",
            "Triple Elimination Initiative (HIV, Syphilis, HBV)",
            "WHO",
            "2024",
            "https://www.who.int/teams/global-hiv-hepatitis-and-stis-programmes/hiv/prevention/mother-to-child-transmission-of-hiv",
            "WHO initiative integrating elimination of vertical transmission of HIV, syphilis, and hepatitis B.",
        ),
    ],
    "10_COMORBIDITIES": [
        (
            "COMRB-002",
            "Cardiac and Renal Comorbidities in Aging PLWH",
            "Circ Research",
            "2024",
            "https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.124.323948",
            "Review of dyslipidemia, hypertension, coronary disease, cardiomyopathy, and kidney injury in aging PLWH.",
        ),
        (
            "COMRB-003",
            "HIV and Inflamm-Aging: Reaching the Summit",
            "IAS-USA",
            "2024",
            "https://www.iasusa.org/wp-content/uploads/2024/12/32-5-589.pdf",
            "Understanding and treating inflamm-aging for healthy life expectancy in people with HIV.",
        ),
        (
            "COMRB-004",
            "HIV-Associated Neurocognitive Disorder Consensus Recommendations",
            "Nature Rev Neurol",
            "2024",
            "https://www.nature.com/articles/s41582-023-00813-2",
            "Consensus recommendations for new approaches to cognitive impairment in people living with HIV.",
        ),
        (
            "COMRB-005",
            "HAND and Microbiota-Gut-Brain Axis",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1428239/full",
            "Key implications of gut-brain axis in HIV-associated neurocognitive disorder pathogenesis.",
        ),
        (
            "COMRB-006",
            "Neurological Impact of HIV and Substance Use",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1505440/full",
            "Brain function and structure alterations from combined HIV infection and substance use.",
        ),
    ],
    "11_AI_DRUG_DISCOVERY": [
        (
            "AI-002",
            "ML Classification Model for HIV-1 Integrase Inhibitors",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11064125/",
            "Novel machine learning approach classifying HIV-1 integrase inhibitors with drug repurposing applications.",
        ),
        (
            "AI-003",
            "AI in HIV Mutation Prediction and Drug Design",
            "ScienceDirect",
            "2025",
            "https://www.sciencedirect.com/science/article/pii/S2773216925000182",
            "AI applications for predicting viral mutations and optimizing personalized drug design.",
        ),
        (
            "AI-004",
            "Anti-HIV Candidate Discovery with AI Multi-Stage System",
            "ScienceDirect",
            "2025",
            "https://www.sciencedirect.com/science/article/abs/pii/S016974392500228X",
            "Three-stage AI framework integrating deep learning, molecular docking, and ADME predictions.",
        ),
        (
            "AI-005",
            "Deep Learning for Drug Resistance Prediction on HIV-1 Sequences",
            "PMC",
            "2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/",
            "Comparison of MLP, BiRNN, and CNN architectures for predicting resistance to 18 ART drugs.",
        ),
        (
            "AI-006",
            "Deep Graph Neural Networks for HIV Drug-Target Interaction",
            "ScienceDirect",
            "2024",
            "https://www.sciencedirect.com/science/article/abs/pii/S0169743922001873",
            "Geometric deep learning approach achieving 93.3% accuracy in HIV drug resistance prediction.",
        ),
    ],
}

# Additional topics
ADDITIONAL_PAPERS = {
    "12_CORECEPTOR_TROPISM": [
        (
            "TROP-001",
            "Transmission of X4-Tropic HIV-1 Through Mucosal Route",
            "eBioMedicine",
            "2024",
            "https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(24)00446-8/fulltext",
            "First confirmed X4-tropic transmitted/founder HIV-1 through mucosal route in wild-type CCR5 individual.",
        ),
        (
            "TROP-002",
            "T Cell Glycosylation Determines HIV Variant Entry",
            "Fred Hutch",
            "2024",
            "https://www.fredhutch.org/en/news/spotlight/2024/11/overbaugh-x4-r5-tropism-hiv.html",
            "CRISPR screen identifying host genes specifically affecting X4 vs R5-tropic HIV infection.",
        ),
        (
            "TROP-003",
            "TIQ-15 CXCR4 Antagonist with Dual Tropic Inhibition",
            "PLOS Pathogens",
            "2024",
            "https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1012448",
            "Novel tetrahydroisoquinoline CXCR4 antagonist with synergistic anti-R5 activity.",
        ),
        (
            "TROP-004",
            "SLC35A2: First Host Protein Affecting HIV by Coreceptor Use",
            "Fred Hutch",
            "2024",
            "https://www.fredhutch.org/en/news/spotlight/2024/11/overbaugh-x4-r5-tropism-hiv.html",
            "Discovery of first host protein differentially impacting HIV-1 based on coreceptor tropism.",
        ),
    ],
    "13_STIGMA_MENTAL_HEALTH": [
        (
            "STIGMA-001",
            "HIV-Related Stigma: Systematic Review and Meta-Analysis",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2024.1356430/full",
            "Meta-analysis of HIV stigma prevalence and association with depression globally.",
        ),
        (
            "STIGMA-002",
            "Psychological Distress, Stigma, Social Support in Vietnam",
            "Discover Mental Health",
            "2025",
            "https://link.springer.com/article/10.1007/s44192-025-00171-z",
            "Cross-sectional study of stigma, mental health, and quality of life among Vietnamese PLWH.",
        ),
        (
            "STIGMA-003",
            "Internalized HIV Stigma and Syndemic Burden",
            "SAGE",
            "2024",
            "https://journals.sagepub.com/doi/10.1177/13591053241249633",
            "Impact of syndemic burden, age, and sexual minority status on internalized stigma in South Florida.",
        ),
        (
            "STIGMA-004",
            "Positive, Open, Proud: Disclosure-Based Stigma Intervention",
            "Frontiers",
            "2024",
            "https://www.frontiersin.org/journals/global-womens-health/articles/10.3389/fgwh.2024.1469465/full",
            "Adapted disclosure-based intervention for reducing HIV stigma among women.",
        ),
    ],
    "14_DIAGNOSTICS": [
        (
            "DIAG-001",
            "TakeMeHome HIV Self-Test Distribution Program",
            "MMWR",
            "2024",
            "https://www.cdc.gov/mmwr/volumes/73/wr/mm7324a4.htm",
            "CDC-funded program distributing 443,813 HIV self-tests to 219,360 persons in first year.",
        ),
        (
            "DIAG-002",
            "Xpert HIV-1 Qual XC WHO Prequalification",
            "Cepheid",
            "2024",
            "https://www.grandviewresearch.com/industry-analysis/hiv-diagnostics-market",
            "WHO prequalification of Cepheid point-of-care test for early infant HIV detection.",
        ),
        (
            "DIAG-003",
            "Next-Gen HIV Self-Testing Technologies Review",
            "Biosensors",
            "2023",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9954708/",
            "Review of emerging HIV self-testing technologies and lessons from COVID-19 diagnostics.",
        ),
    ],
}


def generate_paper_file(paper_id, title, source, year, link, abstract, category_dir):
    """Generate a markdown file for a single paper."""
    filename = f"{paper_id}_{title.replace(' ', '_').replace(':', '').replace('/', '_')[:50]}.md"
    filepath = category_dir / filename

    content = f"""# {title}

**ID:** {paper_id}
**Year:** {year}
**Source:** {source}
**Link:** [{link}]({link})

---

## Abstract

{abstract}

---

## Key Concepts

- Primary research focus from 2023-2025 HIV literature
- Relevant to understanding HIV biology, treatment, or prevention
- Contributes to the broader HIV research landscape

---

## Relevance to Project

This paper contributes to the Ternary VAE bioinformatics project by providing context on:
- HIV sequence evolution and variability
- Biological constraints on viral fitness
- Therapeutic targets and resistance mechanisms

---

*Added: 2025-12-24*
"""

    filepath.write_text(content, encoding="utf-8")
    print(f"Created: {filepath.name}")


def main():
    """Generate all paper files."""
    # Create main category directories and papers
    for category, papers in PAPERS.items():
        category_dir = BASE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for paper in papers:
            paper_id, title, source, year, link, abstract = paper
            generate_paper_file(paper_id, title, source, year, link, abstract, category_dir)

    # Create additional topic directories and papers
    for category, papers in ADDITIONAL_PAPERS.items():
        category_dir = BASE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create README for additional category
        readme_content = f"""# {category.replace("_", " ").title()}

> **Additional HIV research papers on specialized topics.**

---

## Papers in This Section

| ID | Title | Year |
|:---|:------|:-----|
"""
        for paper in papers:
            readme_content += f"| {paper[0]} | {paper[1]} | {paper[3]} |\n"

        readme_content += "\n---\n\n*See [00_HIV_PAPERS_INDEX.md](../00_HIV_PAPERS_INDEX.md) for complete index.*\n"

        (category_dir / "README.md").write_text(readme_content, encoding="utf-8")
        print(f"Created: {category}/README.md")

        for paper in papers:
            paper_id, title, source, year, link, abstract = paper
            generate_paper_file(paper_id, title, source, year, link, abstract, category_dir)

    print(f"\nTotal papers created in main categories: {sum(len(p) for p in PAPERS.values())}")
    print(f"Total papers created in additional topics: {sum(len(p) for p in ADDITIONAL_PAPERS.values())}")
    print(f"Grand total: {sum(len(p) for p in PAPERS.values()) + sum(len(p) for p in ADDITIONAL_PAPERS.values())}")


if __name__ == "__main__":
    main()
