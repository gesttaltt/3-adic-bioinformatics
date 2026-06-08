# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Disease registry with configurations for all supported diseases."""

from __future__ import annotations

from src.diseases.base import DiseaseConfig, DiseaseType, TaskType


class DiseaseRegistry:
    """Registry of all supported disease configurations.

    This registry provides a central place to define and access
    disease-specific configurations including:
    - Data sources (databases, APIs)
    - Loss function weights
    - Task definitions
    - Feature extractors
    """

    _diseases: dict[str, DiseaseConfig] = {}
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the registry with all disease configurations."""
        if cls._initialized:
            return

        # ================================================================
        # HIV / AIDS
        # ================================================================
        cls._diseases["hiv"] = DiseaseConfig(
            name="hiv",
            display_name="HIV/AIDS",
            disease_type=DiseaseType.VIRAL,
            tasks=[
                TaskType.RESISTANCE,
                TaskType.ESCAPE,
                TaskType.BINDING,
                TaskType.IMMUNOGENICITY,
            ],
            data_sources={
                "stanford_hivdb": "https://hivdb.stanford.edu/",
                "los_alamos": "https://www.hiv.lanl.gov/",
                "catnap": "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/",
                "pdb_structures": "https://www.rcsb.org/",
            },
            loss_weights={
                "glycan_shield": 0.15,
                "escape": 0.20,
                "resistance": 0.25,
                "binding": 0.15,
                "coevolution": 0.10,
                "padic_geodesic": 0.15,
            },
            special_losses=[
                "SentinelGlycanLoss",
                "CoEvolutionLoss",
                "DrugInteractionLoss",
            ],
            codon_features=[
                "resistance_mutations",
                "glycan_sites",
                "ctl_epitopes",
                "cd4_binding_residues",
            ],
            structure_features=[
                "env_trimer",
                "cd4_interface",
                "bnab_epitopes",
                "fusion_peptide",
            ],
            api_endpoints={
                "hivdb_resistance": "https://hivdb.stanford.edu/graphql",
                "lanl_sequence": "https://www.hiv.lanl.gov/components/sequence/HIV/search/search.html",
            },
            metadata={
                "drugs": ["PI", "NRTI", "NNRTI", "INSTI", "EI"],
                "proteins": ["gag", "pol", "env", "nef", "tat", "rev"],
                "key_targets": ["RT", "PR", "IN", "gp120", "gp41"],
            },
        )

        # ================================================================
        # Rheumatoid Arthritis
        # ================================================================
        cls._diseases["ra"] = DiseaseConfig(
            name="ra",
            display_name="Rheumatoid Arthritis",
            disease_type=DiseaseType.AUTOIMMUNE,
            tasks=[
                TaskType.BINDING,
                TaskType.PTM,
                TaskType.IMMUNOGENICITY,
            ],
            data_sources={
                "iedb": "https://www.iedb.org/",
                "uniprot": "https://www.uniprot.org/",
                "hla_database": "https://www.ebi.ac.uk/ipd/imgt/hla/",
                "pdb_structures": "https://www.rcsb.org/",
            },
            loss_weights={
                "autoimmune": 0.25,
                "hla_binding": 0.20,
                "citrullination": 0.20,
                "cross_reactivity": 0.15,
                "padic_geodesic": 0.20,
            },
            special_losses=[
                "AutoimmuneCodonRegularizer",
                "CD4CD8AwareRegularizer",
            ],
            codon_features=[
                "citrullination_sites",
                "hla_binding_motifs",
                "ptm_sites",
                "epitope_regions",
            ],
            structure_features=[
                "hla_peptide_complex",
                "tcr_interface",
                "antigen_binding_groove",
            ],
            api_endpoints={
                "iedb_epitope": "https://www.iedb.org/api/",
                "netmhcpan": "https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1",
            },
            metadata={
                "hla_alleles": ["DRB1*04:01", "DRB1*04:04", "DRB1*01:01"],
                "autoantigens": ["collagen", "fibrinogen", "vimentin", "enolase"],
                "ptm_types": ["citrullination", "carbamylation", "acetylation"],
            },
        )

        # ================================================================
        # Neurodegeneration (Alzheimer's, Parkinson's, ALS)
        # ================================================================
        cls._diseases["neuro"] = DiseaseConfig(
            name="neuro",
            display_name="Neurodegeneration",
            disease_type=DiseaseType.NEURODEGENERATIVE,
            tasks=[
                TaskType.AGGREGATION,
                TaskType.PTM,
                TaskType.STABILITY,
            ],
            data_sources={
                "uniprot": "https://www.uniprot.org/",
                "pdb_structures": "https://www.rcsb.org/",
                "alphafold": "https://alphafold.ebi.ac.uk/",
                "phosphosite": "https://www.phosphosite.org/",
            },
            loss_weights={
                "aggregation": 0.25,
                "phosphorylation": 0.20,
                "stability": 0.20,
                "toxicity": 0.15,
                "padic_geodesic": 0.20,
            },
            special_losses=[
                "AggregationPropensityLoss",
                "PTMPredictionLoss",
            ],
            codon_features=[
                "phospho_sites",
                "aggregation_prone_regions",
                "prion_like_domains",
                "amyloid_cores",
            ],
            structure_features=[
                "tau_filaments",
                "alpha_synuclein",
                "amyloid_beta",
                "tdp43",
            ],
            api_endpoints={
                "alphafold": "https://alphafold.ebi.ac.uk/api/",
                "phosphosite": "https://www.phosphosite.org/",
            },
            metadata={
                "diseases": ["alzheimers", "parkinsons", "als", "ftd", "huntingtons"],
                "proteins": ["tau", "alpha_synuclein", "amyloid_beta", "tdp43", "sod1"],
                "ptm_types": ["phosphorylation", "ubiquitination", "acetylation"],
            },
        )

        # ================================================================
        # Cancer / Immuno-oncology
        # ================================================================
        cls._diseases["cancer"] = DiseaseConfig(
            name="cancer",
            display_name="Cancer/Immuno-oncology",
            disease_type=DiseaseType.CANCER,
            tasks=[
                TaskType.IMMUNOGENICITY,
                TaskType.BINDING,
                TaskType.EXPRESSION,
                TaskType.ESCAPE,
            ],
            data_sources={
                "tcga": "https://portal.gdc.cancer.gov/",
                "cosmic": "https://cancer.sanger.ac.uk/cosmic",
                "iedb": "https://www.iedb.org/",
                "tcia": "https://www.cancerimagingarchive.net/",
            },
            loss_weights={
                "neoantigen": 0.25,
                "mhc_binding": 0.20,
                "tcr_recognition": 0.20,
                "immune_escape": 0.15,
                "padic_geodesic": 0.20,
            },
            special_losses=[
                "NeoantigenLoss",
                "MHCBindingLoss",
                "ImmuneEscapeLoss",
            ],
            codon_features=[
                "mutation_sites",
                "neoepitopes",
                "mhc_binding_motifs",
                "tcr_contact_residues",
            ],
            structure_features=[
                "mhc_peptide_complex",
                "tcr_pmhc",
                "tumor_antigens",
            ],
            api_endpoints={
                "gdc": "https://api.gdc.cancer.gov/",
                "cosmic": "https://cancer.sanger.ac.uk/cosmic/api",
            },
            metadata={
                "cancer_types": ["melanoma", "nsclc", "crc", "breast", "pancreatic"],
                "checkpoint_targets": ["PD1", "PDL1", "CTLA4", "LAG3", "TIM3"],
                "therapies": ["checkpoint_inhibitors", "car_t", "cancer_vaccines"],
            },
        )

        # ================================================================
        # Emerging Infectious Diseases
        # ================================================================
        cls._diseases["emerging"] = DiseaseConfig(
            name="emerging",
            display_name="Emerging Pathogens",
            disease_type=DiseaseType.VIRAL,
            tasks=[
                TaskType.ESCAPE,
                TaskType.BINDING,
                TaskType.STABILITY,
                TaskType.IMMUNOGENICITY,
            ],
            data_sources={
                "gisaid": "https://www.gisaid.org/",
                "ncbi_virus": "https://www.ncbi.nlm.nih.gov/labs/virus/",
                "nextstrain": "https://nextstrain.org/",
                "pdb_structures": "https://www.rcsb.org/",
            },
            loss_weights={
                "variant_escape": 0.25,
                "receptor_binding": 0.20,
                "antibody_escape": 0.20,
                "fitness": 0.15,
                "padic_geodesic": 0.20,
            },
            special_losses=[
                "VariantEscapeLoss",
                "FitnessLoss",
            ],
            codon_features=[
                "rbd_mutations",
                "spike_mutations",
                "immune_epitopes",
                "ace2_binding",
            ],
            structure_features=[
                "spike_protein",
                "rbd_ace2_interface",
                "neutralizing_epitopes",
            ],
            api_endpoints={
                "nextstrain": "https://nextstrain.org/api/",
                "covdb": "https://covdb.stanford.edu/",
            },
            metadata={
                "pathogens": ["sars_cov_2", "influenza", "rsv", "mpox", "ebola"],
                "key_proteins": ["spike", "hemagglutinin", "fusion_protein"],
                "variants": ["omicron", "delta", "ba.2.86", "jn.1"],
            },
        )

        # ================================================================
        # Bacterial Infections
        # ================================================================
        cls._diseases["bacterial"] = DiseaseConfig(
            name="bacterial",
            display_name="Bacterial Infections",
            disease_type=DiseaseType.BACTERIAL,
            tasks=[
                TaskType.RESISTANCE,
                TaskType.ESCAPE,
                TaskType.BINDING,
            ],
            data_sources={
                "card": "https://card.mcmaster.ca/",
                "ncbi_amr": "https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/",
                "uniprot": "https://www.uniprot.org/",
            },
            loss_weights={
                "amr": 0.30,
                "virulence": 0.20,
                "efflux": 0.15,
                "target_modification": 0.15,
                "padic_geodesic": 0.20,
            },
            special_losses=[
                "AMRLoss",
                "VirulenceLoss",
            ],
            codon_features=[
                "resistance_genes",
                "virulence_factors",
                "efflux_pumps",
                "porins",
            ],
            structure_features=[
                "ribosome_binding",
                "penicillin_binding",
                "gyrase_interface",
            ],
            api_endpoints={
                "card": "https://card.mcmaster.ca/api/",
                "ncbi_amr": "https://www.ncbi.nlm.nih.gov/pathogens/api/",
            },
            metadata={
                "pathogens": ["ecoli", "staph_aureus", "mtb", "pseudomonas", "klebsiella"],
                "drug_classes": ["beta_lactams", "aminoglycosides", "fluoroquinolones"],
                "resistance_mechanisms": ["enzymatic", "target_modification", "efflux"],
            },
        )

        cls._initialized = True

    @classmethod
    def get(cls, name: str) -> DiseaseConfig:
        """Get disease configuration by name.

        Args:
            name: Disease identifier (e.g., "hiv", "ra")

        Returns:
            DiseaseConfig for the disease

        Raises:
            KeyError: If disease not found
        """
        cls._initialize()
        if name not in cls._diseases:
            available = ", ".join(cls._diseases.keys())
            raise KeyError(f"Disease '{name}' not found. Available: {available}")
        return cls._diseases[name]

    @classmethod
    def list_diseases(cls) -> list[str]:
        """List all registered disease names."""
        cls._initialize()
        return list(cls._diseases.keys())

    @classmethod
    def list_by_type(cls, disease_type: DiseaseType) -> list[str]:
        """List diseases of a specific type."""
        cls._initialize()
        return [name for name, config in cls._diseases.items() if config.disease_type == disease_type]

    @classmethod
    def register(cls, config: DiseaseConfig) -> None:
        """Register a new disease configuration.

        Args:
            config: Disease configuration to register
        """
        cls._initialize()
        cls._diseases[config.name] = config

    @classmethod
    def get_all_tasks(cls) -> set[TaskType]:
        """Get all unique task types across diseases."""
        cls._initialize()
        tasks = set()
        for config in cls._diseases.values():
            tasks.update(config.tasks)
        return tasks

    @classmethod
    def get_diseases_for_task(cls, task: TaskType) -> list[str]:
        """Get diseases that have a specific task."""
        cls._initialize()
        return [name for name, config in cls._diseases.items() if task in config.tasks]


def get_disease_config(name: str) -> DiseaseConfig:
    """Convenience function to get disease configuration.

    Args:
        name: Disease identifier

    Returns:
        DiseaseConfig for the disease
    """
    return DiseaseRegistry.get(name)
