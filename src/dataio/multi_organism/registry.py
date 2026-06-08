"""Registry for organism-specific data loaders.

Provides a central registry for all organism loaders and utilities
for loading data across organisms.
"""

from __future__ import annotations

from typing import Any

from .base import OrganismLoader, OrganismType, SequenceRecord


class OrganismRegistry:
    """Registry for organism-specific data loaders."""

    _loaders: dict[OrganismType, type[OrganismLoader]] = {}
    _instances: dict[OrganismType, OrganismLoader] = {}

    @classmethod
    def register(cls, organism: OrganismType):
        """Decorator to register a loader for an organism."""

        def decorator(loader_class: type[OrganismLoader]):
            cls._loaders[organism] = loader_class
            return loader_class

        return decorator

    @classmethod
    def get_loader(
        cls,
        organism: OrganismType,
        **kwargs,
    ) -> OrganismLoader:
        """Get loader instance for an organism.

        Args:
            organism: Organism type
            **kwargs: Loader configuration

        Returns:
            OrganismLoader instance
        """
        # Check for cached instance (without kwargs)
        if not kwargs and organism in cls._instances:
            return cls._instances[organism]

        # Create new instance
        if organism not in cls._loaders:
            # Try to import the loader module
            cls._try_import_loader(organism)

        if organism not in cls._loaders:
            raise ValueError(f"No loader registered for {organism.name}")

        loader = cls._loaders[organism](organism, **kwargs)

        # Cache instance if no kwargs
        if not kwargs:
            cls._instances[organism] = loader

        return loader

    @classmethod
    def _try_import_loader(cls, organism: OrganismType):
        """Try to import loader module for organism."""
        # Map organism types to module names
        module_map = {
            OrganismType.HIV: "src.data.multi_organism.loaders.hiv_loader",
            OrganismType.HBV: "src.data.multi_organism.loaders.hbv_loader",
            OrganismType.HCV: "src.data.multi_organism.loaders.hcv_loader",
            OrganismType.INFLUENZA: "src.data.multi_organism.loaders.flu_loader",
            OrganismType.SARS_COV_2: "src.data.multi_organism.loaders.covid_loader",
            OrganismType.TB: "src.data.multi_organism.loaders.tb_loader",
            OrganismType.MRSA: "src.data.multi_organism.loaders.mrsa_loader",
            OrganismType.MALARIA: "src.data.multi_organism.loaders.malaria_loader",
            OrganismType.ANTIBODY: "src.data.multi_organism.loaders.antibody_loader",
            OrganismType.TCR: "src.data.multi_organism.loaders.tcr_loader",
        }

        if organism in module_map:
            try:
                import importlib

                importlib.import_module(module_map[organism])
            except ImportError as e:
                print(f"Warning: Could not import loader for {organism.name}: {e}")

    @classmethod
    def list_available(cls) -> list[OrganismType]:
        """List all organisms with registered loaders."""
        return list(cls._loaders.keys())

    @classmethod
    def load_multi_organism(
        cls,
        organisms: list[OrganismType],
        max_per_organism: int | None = None,
    ) -> list[SequenceRecord]:
        """Load sequences from multiple organisms.

        Args:
            organisms: List of organism types
            max_per_organism: Maximum sequences per organism

        Returns:
            Combined list of sequence records
        """
        all_records = []

        for organism in organisms:
            try:
                loader = cls.get_loader(organism, max_sequences=max_per_organism)
                records = loader.load_sequences()
                all_records.extend(records)
            except ValueError as e:
                print(f"Skipping {organism.name}: {e}")

        return all_records

    @classmethod
    def get_statistics(
        cls,
        organisms: list[OrganismType] | None = None,
    ) -> dict[str, Any]:
        """Get statistics for one or more organisms.

        Args:
            organisms: List of organisms (None = all available)

        Returns:
            Dictionary of statistics per organism
        """
        if organisms is None:
            organisms = cls.list_available()

        stats = {}
        for organism in organisms:
            try:
                loader = cls.get_loader(organism)
                stats[organism.name] = loader.get_statistics()
            except ValueError:
                stats[organism.name] = {"error": "Loader not available"}

        return stats


# Convenience functions
def load_organism(organism: OrganismType, **kwargs) -> OrganismLoader:
    """Load data for a single organism."""
    return OrganismRegistry.get_loader(organism, **kwargs)


def load_all_viruses(max_per_organism: int = 1000) -> list[SequenceRecord]:
    """Load sequences from all virus types."""
    virus_types = [
        OrganismType.HIV,
        OrganismType.HBV,
        OrganismType.HCV,
        OrganismType.INFLUENZA,
        OrganismType.SARS_COV_2,
    ]
    return OrganismRegistry.load_multi_organism(virus_types, max_per_organism)


def load_all_bacteria(max_per_organism: int = 1000) -> list[SequenceRecord]:
    """Load sequences from all bacteria types."""
    bacteria_types = [
        OrganismType.TB,
        OrganismType.MRSA,
        OrganismType.ECOLI,
    ]
    return OrganismRegistry.load_multi_organism(bacteria_types, max_per_organism)


def load_all_proteins(max_per_type: int = 1000) -> list[SequenceRecord]:
    """Load sequences from all protein types."""
    protein_types = [
        OrganismType.ANTIBODY,
        OrganismType.TCR,
        OrganismType.KINASE,
        OrganismType.GPCR,
    ]
    return OrganismRegistry.load_multi_organism(protein_types, max_per_type)
