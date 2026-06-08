# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss Registry - Dynamic composition pattern with plugin support.

This module provides a registry-based approach to loss composition
that eliminates the God Object anti-pattern in DualVAELoss.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                       LossRegistry                          │
    │                                                             │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │ ReconLoss    │  │ KLLoss       │  │ PAdicLoss    │ ...  │
    │  └──────────────┘  └──────────────┘  └──────────────┘      │
    │           │                │                │               │
    │           └────────────────┼────────────────┘               │
    │                            ▼                                │
    │                    ┌────────────────┐                       │
    │                    │ compose()       │                       │
    │                    │ returns total   │                       │
    │                    │ + all metrics   │                       │
    │                    └────────────────┘                       │
    └─────────────────────────────────────────────────────────────┘

Benefits:
    - Each loss is independently testable
    - Weights can be modified at runtime (StateNet control)
    - Easy to add/remove losses via configuration
    - Clean separation of concerns
    - No conditional spaghetti
    - Plugin support for extensibility

Usage:
    # Standard usage
    registry = LossRegistry()
    registry.register('recon', ReconstructionLoss(weight=1.0))
    registry.register('kl', KLDivergenceLoss(weight=0.1))

    # During training
    result = registry.compose(outputs, targets, batch_indices=indices)
    total_loss = result.loss
    metrics = result.metrics

    # StateNet weight update
    registry.set_weight('kl', new_weight)

Plugin System:
    # Define a new loss with decorator
    @LossComponentRegistry.register("my_custom_loss")
    class MyCustomLoss(LossComponent):
        ...

    # Auto-discover plugins
    LossComponentRegistry.discover_plugins(Path("plugins/losses"))
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .base import LossComponent, LossResult

if TYPE_CHECKING:
    from src.config.schema import TrainingConfig

logger = logging.getLogger(__name__)


class LossRegistry(nn.Module):
    """Dynamic loss composition registry.

    Manages a collection of LossComponent instances and composes them
    into a single loss with aggregated metrics.
    """

    def __init__(self):
        """Initialize empty registry."""
        super().__init__()
        self._losses: OrderedDict[str, LossComponent] = OrderedDict()
        self._enabled: dict[str, bool] = {}
        self._weight_overrides: dict[str, float] = {}

    def register(self, name: str, loss: LossComponent, enabled: bool = True) -> LossRegistry:
        """Register a loss component.

        Args:
            name: Unique identifier for this loss
            loss: The loss component instance
            enabled: Whether this loss is enabled by default

        Returns:
            Self for chaining

        Raises:
            ValueError: If name already registered
        """
        if name in self._losses:
            raise ValueError(f"Loss '{name}' already registered")

        # Register as a submodule for proper parameter tracking
        self.add_module(name, loss)
        self._losses[name] = loss
        self._enabled[name] = enabled

        return self

    def unregister(self, name: str) -> LossRegistry:
        """Remove a loss component.

        Args:
            name: Identifier of loss to remove

        Returns:
            Self for chaining
        """
        if name in self._losses:
            delattr(self, name)
            del self._losses[name]
            del self._enabled[name]
            self._weight_overrides.pop(name, None)

        return self

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a loss.

        Args:
            name: Loss identifier
            enabled: Whether to enable
        """
        if name in self._enabled:
            self._enabled[name] = enabled

    def set_weight(self, name: str, weight: float) -> None:
        """Override the weight for a loss.

        This allows external control (e.g., StateNet) to adjust
        loss weights without modifying the loss component itself.

        Args:
            name: Loss identifier
            weight: New weight value
        """
        self._weight_overrides[name] = weight

    def get_weight(self, name: str) -> float:
        """Get current effective weight for a loss.

        Args:
            name: Loss identifier

        Returns:
            Current weight (override if set, else component weight)
        """
        if name in self._weight_overrides:
            return self._weight_overrides[name]
        if name in self._losses:
            return self._losses[name].weight
        return 0.0

    def reset_weight(self, name: str) -> None:
        """Reset weight override to use component's original weight.

        Args:
            name: Loss identifier
        """
        self._weight_overrides.pop(name, None)

    def get_loss(self, name: str) -> LossComponent | None:
        """Get a registered loss by name.

        Args:
            name: Loss identifier

        Returns:
            LossComponent or None if not found
        """
        return self._losses.get(name)

    def list_losses(self) -> list[str]:
        """List all registered loss names.

        Returns:
            List of loss identifiers
        """
        return list(self._losses.keys())

    def list_enabled(self) -> list[str]:
        """List all enabled loss names.

        Returns:
            List of enabled loss identifiers
        """
        return [name for name, enabled in self._enabled.items() if enabled]

    def compose(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compose all enabled losses into single result.

        This is the main entry point during training. It:
        1. Iterates through enabled losses
        2. Computes each loss with current weights
        3. Aggregates into single loss + merged metrics

        Args:
            outputs: Model outputs dictionary
            targets: Target values
            **kwargs: Additional arguments passed to each loss

        Returns:
            LossResult with total loss and all metrics
        """
        if "z_A" in outputs:
            device = outputs["z_A"].device
        elif "logits_A" in outputs:
            device = outputs["logits_A"].device
        else:
            # Fallback to first available tensor or default
            device = next(iter(outputs.values())).device if outputs else torch.device("cpu")
        total_loss = torch.tensor(0.0, device=device)
        all_metrics: dict[str, float] = {}

        for name, loss_component in self._losses.items():
            # Skip disabled losses
            if not self._enabled.get(name, True):
                continue

            # Skip if component reports disabled
            if not loss_component.enabled():
                continue

            # Compute loss
            result = loss_component(outputs, targets, **kwargs)

            # Apply weight override if set
            effective_weight = self._weight_overrides.get(name, result.weight)
            weighted_loss = effective_weight * result.loss

            # Accumulate
            total_loss = total_loss + weighted_loss

            # Merge metrics with loss name prefix
            for key, value in result.metrics.items():
                metric_key = f"{name}/{key}"
                all_metrics[metric_key] = value

            # Also record the loss value itself
            all_metrics[f"{name}/loss"] = result.loss.item()
            all_metrics[f"{name}/weight"] = effective_weight

        return LossResult(
            loss=total_loss,
            metrics=all_metrics,
            weight=1.0,  # Total is already weighted
        )

    def compose_with_gradients(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> tuple:
        """Compose losses and return individual gradients for analysis.

        Useful for gradient balancing and StateNet feedback.

        Args:
            outputs: Model outputs
            targets: Target values
            **kwargs: Additional arguments

        Returns:
            Tuple of (total_loss, metrics, gradient_norms)
        """
        result = self.compose(outputs, targets, **kwargs)

        # Compute per-loss gradient norms if needed
        grad_norms: dict[str, float] = {}

        return result.loss, result.metrics, grad_norms


class LossGroup:
    """Grouping of related losses for organization.

    Example:
        reconstruction_group = LossGroup('reconstruction')
        reconstruction_group.add('ce_A', recon_loss_A)
        reconstruction_group.add('ce_B', recon_loss_B)
        registry.register_group(reconstruction_group)
    """

    def __init__(self, name: str):
        """Initialize loss group.

        Args:
            name: Group name (used as prefix in metrics)
        """
        self.name = name
        self._losses: dict[str, LossComponent] = {}

    def add(self, name: str, loss: LossComponent) -> LossGroup:
        """Add a loss to this group.

        Args:
            name: Loss name within group
            loss: Loss component

        Returns:
            Self for chaining
        """
        self._losses[name] = loss
        return self

    @property
    def losses(self) -> dict[str, LossComponent]:
        """Return all losses in this group."""
        return self._losses


def create_registry_from_config(config: dict[str, Any]) -> LossRegistry:
    """Factory to create LossRegistry from configuration.

    This is the main entry point for creating a configured registry.

    Args:
        config: Configuration dictionary with loss settings

    Returns:
        Configured LossRegistry

    Example config:
        {
            'losses': {
                'reconstruction': {'enabled': True, 'weight': 1.0},
                'kl': {'enabled': True, 'weight': 0.1, 'free_bits': 0.0},
                'padic_ranking': {'enabled': True, 'weight': 0.5, ...}
            }
        }
    """
    from .components import (
        EntropyLossComponent,
        KLDivergenceLossComponent,
        PAdicHyperbolicLossComponent,
        PAdicRankingLossComponent,
        ReconstructionLossComponent,
        RepulsionLossComponent,
    )

    registry = LossRegistry()
    losses_config = config.get("losses", {})

    # Register each loss type if enabled
    loss_factories: dict[str, Callable] = {
        "reconstruction": lambda cfg: ReconstructionLossComponent(weight=cfg.get("weight", 1.0)),
        "kl": lambda cfg: KLDivergenceLossComponent(weight=cfg.get("weight", 0.1), free_bits=cfg.get("free_bits", 0.0)),
        "entropy": lambda cfg: EntropyLossComponent(weight=cfg.get("weight", 0.01)),
        "repulsion": lambda cfg: RepulsionLossComponent(weight=cfg.get("weight", 0.01), sigma=cfg.get("sigma", 0.5)),
        "padic_ranking": lambda cfg: PAdicRankingLossComponent(weight=cfg.get("weight", 0.5), config=cfg),
        "padic_hyperbolic": lambda cfg: PAdicHyperbolicLossComponent(weight=cfg.get("weight", 0.5), config=cfg),
    }

    for loss_name, factory in loss_factories.items():
        loss_cfg = losses_config.get(loss_name, {})
        enabled = loss_cfg.get("enabled", False)

        if enabled:
            loss_component = factory(loss_cfg)
            registry.register(loss_name, loss_component, enabled=True)

    return registry


class LossComponentRegistry:
    """Global registry for loss component classes with plugin support.

    This class provides a decorator-based registration system that allows
    new loss components to be added without modifying existing code.

    Usage:
        # Register a new loss component
        @LossComponentRegistry.register("my_loss")
        class MyLoss(LossComponent):
            ...

        # Later, create an instance
        loss_cls = LossComponentRegistry.get("my_loss")
        loss = loss_cls(weight=0.5)

        # List all registered losses
        print(LossComponentRegistry.list_all())
    """

    _registry: dict[str, type[LossComponent]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[LossComponent]], type[LossComponent]]:
        """Decorator to register a loss component class.

        Args:
            name: Unique identifier for this loss type

        Returns:
            Decorator function

        Example:
            @LossComponentRegistry.register("custom_ranking")
            class CustomRankingLoss(LossComponent):
                ...
        """

        def decorator(loss_cls: type[LossComponent]) -> type[LossComponent]:
            if name in cls._registry:
                logger.warning(f"Overwriting existing loss registration: {name}")
            cls._registry[name] = loss_cls
            logger.debug(f"Registered loss component: {name} -> {loss_cls.__name__}")
            return loss_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[LossComponent]:
        """Get a registered loss component class by name.

        Args:
            name: Loss identifier

        Returns:
            Loss component class

        Raises:
            KeyError: If loss not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(f"Unknown loss component: '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered loss component names.

        Returns:
            List of registered loss names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a loss component is registered.

        Args:
            name: Loss identifier

        Returns:
            True if registered
        """
        return name in cls._registry

    @classmethod
    def discover_plugins(cls, plugin_dir: Path) -> int:
        """Auto-discover and load loss plugins from a directory.

        Scans the specified directory for Python modules and imports them.
        Any classes decorated with @LossComponentRegistry.register will
        be automatically registered.

        Args:
            plugin_dir: Directory containing plugin modules

        Returns:
            Number of plugins loaded

        Example:
            # plugins/losses/my_custom_loss.py contains:
            # @LossComponentRegistry.register("my_custom")
            # class MyCustomLoss(LossComponent): ...

            count = LossComponentRegistry.discover_plugins(Path("plugins/losses"))
            print(f"Loaded {count} plugins")
        """
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return 0

        loaded = 0
        for module_info in pkgutil.iter_modules([str(plugin_dir)]):
            try:
                # Import the module - decorators will auto-register
                module_name = f"plugins.losses.{module_info.name}"
                importlib.import_module(module_name)
                loaded += 1
                logger.info(f"Loaded loss plugin: {module_info.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {module_info.name}: {e}")

        return loaded

    @classmethod
    def create_from_config(cls, name: str, config: dict[str, Any], **kwargs) -> LossComponent:
        """Create a loss component instance from configuration.

        Args:
            name: Registered loss name
            config: Configuration dictionary
            **kwargs: Additional constructor arguments

        Returns:
            Configured loss component instance
        """
        loss_cls = cls.get(name)

        # Merge config with kwargs
        all_args = {**config, **kwargs}

        # Extract weight separately if present
        weight = all_args.pop("weight", 1.0)

        return loss_cls(weight=weight, **all_args)


def create_registry_with_plugins(
    config: dict[str, Any],
    plugin_dir: Path | None = None,
) -> LossRegistry:
    """Create a LossRegistry with plugin support.

    This is an enhanced version of create_registry_from_config that
    also loads plugins from a specified directory.

    Args:
        config: Configuration dictionary
        plugin_dir: Optional directory containing loss plugins

    Returns:
        Configured LossRegistry with plugins loaded
    """
    # Discover plugins first
    if plugin_dir is not None:
        LossComponentRegistry.discover_plugins(plugin_dir)

    # Create base registry
    registry = create_registry_from_config(config)

    # Add any custom losses from config that use registered plugins
    custom_losses = config.get("custom_losses", {})
    for name, loss_config in custom_losses.items():
        if not loss_config.get("enabled", True):
            continue

        loss_type = loss_config.get("type", name)
        if LossComponentRegistry.is_registered(loss_type):
            loss = LossComponentRegistry.create_from_config(loss_type, loss_config)
            registry.register(name, loss, enabled=True)

    return registry


def create_registry_from_training_config(config: TrainingConfig) -> LossRegistry:
    """Create LossRegistry from TrainingConfig dataclass.

    This bridges the new config schema (src.config.TrainingConfig) with
    the loss registry system.

    Args:
        config: TrainingConfig instance with loss_weights and ranking config

    Returns:
        Configured LossRegistry

    Example:
        from src.config import load_config
        from src.losses import create_registry_from_training_config

        config = load_config("config.yaml")
        registry = create_registry_from_training_config(config)
    """
    from .components import (
        EntropyLossComponent,
        KLDivergenceLossComponent,
        PAdicHyperbolicLossComponent,
        PAdicRankingLossComponent,
        ReconstructionLossComponent,
        RepulsionLossComponent,
    )

    registry = LossRegistry()
    weights = config.loss_weights

    # Reconstruction (always enabled if weight > 0)
    if weights.reconstruction > 0:
        registry.register(
            "reconstruction",
            ReconstructionLossComponent(weight=weights.reconstruction),
            enabled=True,
        )

    # KL Divergence
    if weights.kl_divergence > 0:
        registry.register(
            "kl",
            KLDivergenceLossComponent(
                weight=weights.kl_divergence,
                free_bits=config.free_bits,
            ),
            enabled=True,
        )

    # Entropy regularization
    if weights.entropy > 0:
        registry.register(
            "entropy",
            EntropyLossComponent(weight=weights.entropy),
            enabled=True,
        )

    # Repulsion loss
    if weights.repulsion > 0:
        registry.register(
            "repulsion",
            RepulsionLossComponent(
                weight=weights.repulsion,
                sigma=DEFAULT_REPULSION_SIGMA,
            ),
            enabled=True,
        )

    # Ranking loss (p-adic based)
    if weights.ranking > 0:
        ranking_cfg = {
            "margin": config.ranking.margin,
            "n_triplets": config.ranking.n_triplets,
            "hard_negative_ratio": config.ranking.hard_negative_ratio,
        }
        registry.register(
            "padic_ranking",
            PAdicRankingLossComponent(weight=weights.ranking, config=ranking_cfg),
            enabled=True,
        )

    # Hyperbolic losses
    if weights.radial > 0 or weights.geodesic > 0:
        hyperbolic_cfg = {
            "weight_radial": weights.radial,
            "weight_geodesic": weights.geodesic,
            "curvature": config.geometry.curvature,
            "max_radius": config.geometry.max_radius,
        }
        # Use radial weight as main weight if both present
        main_weight = weights.radial if weights.radial > 0 else weights.geodesic
        registry.register(
            "padic_hyperbolic",
            PAdicHyperbolicLossComponent(weight=main_weight, config=hyperbolic_cfg),
            enabled=True,
        )

    return registry


# Import constants for default values
from src.config.constants import DEFAULT_REPULSION_SIGMA

__all__ = [
    "LossRegistry",
    "LossGroup",
    "LossComponentRegistry",
    "create_registry_from_config",
    "create_registry_with_plugins",
    "create_registry_from_training_config",
]
