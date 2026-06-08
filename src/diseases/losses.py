# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-disease loss functions combining all available losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.diseases.registry import DiseaseRegistry


class MultiDiseaseLoss(nn.Module):
    """Unified loss function combining all disease-specific losses.

    This loss dynamically combines:
    - Base p-adic geodesic loss (always active)
    - Disease-specific losses based on config
    - Task-specific losses (resistance, escape, binding, etc.)

    Example:
        loss_fn = MultiDiseaseLoss(diseases=["hiv", "ra", "neuro"])
        total_loss, loss_dict = loss_fn(
            embeddings=z,
            sequences=seqs,
            labels=labels,
            disease="hiv"
        )
    """

    def __init__(
        self,
        diseases: list[str],
        base_loss_weight: float = 0.3,
        device: str = "cuda",
    ):
        """Initialize multi-disease loss.

        Args:
            diseases: List of disease names to include
            base_loss_weight: Weight for base p-adic geodesic loss
            device: Device to use
        """
        super().__init__()
        self.diseases = diseases
        self.base_loss_weight = base_loss_weight
        self.device = device

        # Load disease configs
        self.configs = {d: DiseaseRegistry.get(d) for d in diseases}

        # Initialize base losses (always active)
        self._init_base_losses()

        # Initialize disease-specific losses
        self._init_disease_losses()

    def _init_base_losses(self) -> None:
        """Initialize base losses that apply to all diseases."""
        # Import here to avoid circular imports
        try:
            from src.losses.padic_geodesic import PAdicGeodesicLoss

            self.padic_geodesic = PAdicGeodesicLoss(
                curvature=1.0,
                max_target_distance=3.0,
                n_pairs=2000,
            )
        except ImportError:
            self.padic_geodesic = None

        try:
            from src.losses.radial_stratification import RadialHierarchyLoss

            self.radial_hierarchy = RadialHierarchyLoss(
                inner_radius=0.1,
                outer_radius=0.85,
            )
        except ImportError:
            self.radial_hierarchy = None

    def _init_disease_losses(self) -> None:
        """Initialize disease-specific losses."""
        self.disease_losses = {}

        for disease_name, _config in self.configs.items():
            losses = {}

            # HIV-specific losses
            if disease_name == "hiv":
                try:
                    from src.losses.glycan_loss import SentinelGlycanLoss

                    losses["glycan_shield"] = SentinelGlycanLoss()
                except ImportError:
                    pass

                try:
                    from src.losses.coevolution_loss import CoEvolutionLoss

                    losses["coevolution"] = CoEvolutionLoss()
                except ImportError:
                    pass

                try:
                    from src.losses.drug_interaction import DrugInteractionLoss

                    losses["drug_interaction"] = DrugInteractionLoss()
                except ImportError:
                    pass

            # Autoimmune-specific losses (RA)
            elif disease_name == "ra":
                try:
                    from src.losses.autoimmunity import AutoimmuneCodonRegularizer

                    losses["autoimmune"] = AutoimmuneCodonRegularizer()
                except ImportError:
                    pass

                try:
                    from src.losses.autoimmunity import CD4CD8AwareRegularizer

                    losses["cd4cd8"] = CD4CD8AwareRegularizer()
                except ImportError:
                    pass

            # Neurodegeneration-specific losses
            elif disease_name == "neuro":
                # Add aggregation and PTM losses when available
                pass

            # Cancer-specific losses
            elif disease_name == "cancer":
                # Add neoantigen and MHC binding losses when available
                pass

            self.disease_losses[disease_name] = nn.ModuleDict(losses)

        # Convert to ModuleDict for proper parameter registration
        self.disease_losses = nn.ModuleDict(self.disease_losses)

    def forward(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor | None = None,
        sequences: torch.Tensor | None = None,
        labels: dict[str, torch.Tensor] | None = None,
        disease: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            embeddings: Latent embeddings (batch, dim) or (batch, seq, dim)
            indices: Operation indices for p-adic valuation
            sequences: Codon sequences (batch, seq_len)
            labels: Task-specific labels
            disease: Current disease (for disease-specific losses)
            **kwargs: Additional arguments for specific losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)

        # Base losses (always active)
        if self.padic_geodesic is not None and indices is not None:
            geodesic_loss = self.padic_geodesic(embeddings, indices)
            losses["padic_geodesic"] = geodesic_loss
            total_loss = total_loss + self.base_loss_weight * geodesic_loss

        if self.radial_hierarchy is not None and indices is not None:
            radial_loss = self.radial_hierarchy(embeddings, indices)
            losses["radial_hierarchy"] = radial_loss
            total_loss = total_loss + 0.2 * radial_loss

        # Disease-specific losses
        if disease is not None and disease in self.disease_losses:
            config = self.configs[disease]
            disease_loss_modules = self.disease_losses[disease]

            for loss_name, loss_module in disease_loss_modules.items():
                weight = config.get_loss_weight(loss_name, 0.1)
                try:
                    # Each loss may have different signatures
                    if hasattr(loss_module, "forward"):
                        loss_value = loss_module(
                            embeddings=embeddings,
                            sequences=sequences,
                            labels=labels,
                            **kwargs,
                        )
                        if isinstance(loss_value, dict):
                            loss_value = loss_value.get("total", sum(loss_value.values()))
                        losses[f"{disease}_{loss_name}"] = loss_value
                        total_loss = total_loss + weight * loss_value
                except Exception:
                    # Skip losses that fail (may need specific inputs)
                    pass

        losses["total"] = total_loss
        return total_loss, losses

    def get_active_losses(self, disease: str | None = None) -> list[str]:
        """Get list of active loss names for a disease."""
        active = ["padic_geodesic", "radial_hierarchy"]
        if disease and disease in self.disease_losses:
            active.extend([f"{disease}_{name}" for name in self.disease_losses[disease]])
        return active


class UnifiedTrainingLoss(nn.Module):
    """Complete training loss for multi-disease VAE.

    Combines:
    - VAE losses (reconstruction, KL)
    - Structure losses (radial hierarchy, geodesic)
    - Disease-specific losses
    - Task losses (classification, regression)
    """

    def __init__(
        self,
        diseases: list[str],
        vae_weight: float = 1.0,
        structure_weight: float = 0.5,
        disease_weight: float = 0.3,
        task_weight: float = 0.2,
    ):
        """Initialize unified loss.

        Args:
            diseases: List of disease names
            vae_weight: Weight for VAE losses
            structure_weight: Weight for structure losses
            disease_weight: Weight for disease-specific losses
            task_weight: Weight for task prediction losses
        """
        super().__init__()
        self.vae_weight = vae_weight
        self.structure_weight = structure_weight
        self.disease_weight = disease_weight
        self.task_weight = task_weight

        # Multi-disease loss component
        self.disease_loss = MultiDiseaseLoss(diseases)

        # Task losses
        self.task_losses = nn.ModuleDict(
            {
                "classification": nn.CrossEntropyLoss(),
                "regression": nn.MSELoss(),
                "ranking": nn.MarginRankingLoss(margin=0.1),
            }
        )

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        disease: str,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute complete training loss.

        Args:
            model_output: Dictionary with VAE outputs
                - "z": latent embeddings
                - "mu": mean
                - "logvar": log variance
                - "recon": reconstruction
                - "predictions": task predictions
            targets: Dictionary with target values
                - "input": original input
                - "indices": operation indices
                - "labels": task labels
            disease: Current disease name
            **kwargs: Additional arguments

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=model_output["z"].device)

        # VAE losses
        if "recon" in model_output and "input" in targets:
            recon_loss = nn.functional.cross_entropy(
                model_output["recon"].view(-1, model_output["recon"].size(-1)),
                targets["input"].view(-1),
            )
            losses["reconstruction"] = recon_loss
            total_loss = total_loss + self.vae_weight * recon_loss

        if "mu" in model_output and "logvar" in model_output:
            kl_loss = -0.5 * torch.mean(
                1 + model_output["logvar"] - model_output["mu"].pow(2) - model_output["logvar"].exp()
            )
            losses["kl_divergence"] = kl_loss
            total_loss = total_loss + self.vae_weight * 0.1 * kl_loss

        # Structure losses (via multi-disease loss)
        struct_loss, struct_losses = self.disease_loss(
            embeddings=model_output["z"],
            indices=targets.get("indices"),
            sequences=targets.get("sequences"),
            labels=targets.get("labels"),
            disease=disease,
            **kwargs,
        )
        losses.update(struct_losses)
        total_loss = total_loss + self.structure_weight * struct_loss

        # Task losses
        if "predictions" in model_output and "labels" in targets:
            for task_name, pred in model_output["predictions"].items():
                if task_name in targets["labels"]:
                    target = targets["labels"][task_name]
                    if target.dtype in [torch.long, torch.int]:
                        task_loss = self.task_losses["classification"](pred, target)
                    else:
                        task_loss = self.task_losses["regression"](pred, target)
                    losses[f"task_{task_name}"] = task_loss
                    total_loss = total_loss + self.task_weight * task_loss

        losses["total"] = total_loss
        return total_loss, losses
