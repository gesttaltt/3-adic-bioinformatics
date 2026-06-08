# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Checkpoint loading utilities with numpy version compatibility.

This module provides backward-compatible checkpoint loading that handles
numpy version changes (numpy._core -> numpy.core), as well as comprehensive
metadata and metrics extraction from checkpoints with varying formats.

Single responsibility: Checkpoint I/O with version compatibility and
unified metadata extraction.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


class NumpyBackwardsCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy._core -> numpy.core renaming.

    PyTorch checkpoints saved with newer numpy versions use numpy._core
    which older versions cannot load. This unpickler handles the renaming.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Override find_class to handle numpy module renaming.

        Args:
            module: Module name from pickle
            name: Class name from pickle

        Returns:
            The resolved class
        """
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint_compat(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with numpy version compatibility.

    Attempts standard torch.load first, falls back to custom unpickler
    if numpy._core ModuleNotFoundError is encountered.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ModuleNotFoundError: If a non-numpy module is missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            with open(path, "rb") as f:
                unpickler = NumpyBackwardsCompatUnpickler(f)
                return unpickler.load()
        raise


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: Path | str,
) -> None:
    """Save checkpoint to file.

    Args:
        checkpoint: Checkpoint dictionary to save
        path: Destination path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def get_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract model state dict from checkpoint, handling various key formats.

    Different checkpoint versions use different key names:
    - model_state_dict: New standard format (homeostatic_rich)
    - model_state: v5.11 format (v5_11_homeostasis, v5_11_structural)
    - model: v5.5 format

    Args:
        checkpoint: Full checkpoint dictionary

    Returns:
        Model state dict

    Raises:
        ValueError: If no model state can be found
    """
    # Try different key formats in order of preference
    for key in ["model_state_dict", "model_state", "model"]:
        if key in checkpoint:
            return checkpoint[key]

    # If no known key found, check if checkpoint looks like a state dict
    if isinstance(checkpoint, dict) and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise ValueError(f"Cannot find model state in checkpoint. Keys: {list(checkpoint.keys())}")


def extract_model_state(
    checkpoint: dict[str, Any],
    prefix: str,
    strip_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract model state dict with optional prefix filtering.

    Args:
        checkpoint: Full checkpoint dictionary
        prefix: Key prefix to filter (e.g., 'encoder_A.')
        strip_prefix: Whether to remove prefix from keys

    Returns:
        Filtered state dict
    """
    model_state = get_model_state_dict(checkpoint)
    prefix_dot = f"{prefix}." if not prefix.endswith(".") else prefix

    filtered = {}
    for key, value in model_state.items():
        if key.startswith(prefix_dot):
            new_key = key[len(prefix_dot) :] if strip_prefix else key
            filtered[new_key] = value

    return filtered


def _to_python_type(value: Any) -> Any:
    """Convert numpy/torch types to Python native types for serialization."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {k: _to_python_type(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_type(v) for v in value]
    return value


@dataclass
class CheckpointMetrics:
    """Unified metrics from checkpoint, normalized across different formats.

    Attributes:
        coverage: Reconstruction accuracy (0-1)
        hierarchy: Spearman correlation of radii vs valuation (negative = correct)
        hierarchy_A: VAE-A hierarchy (radial_corr_A)
        hierarchy_B: VAE-B hierarchy (radial_corr_B)
        richness: Within-level variance (geometric diversity)
        r_v0: Mean radius for valuation 0 (outermost)
        r_v9: Mean radius for valuation 9 (innermost)
        mean_radius: Average radius across all operations
        std_radius: Standard deviation of radii
        distance_corr_A: Distance correlation for VAE-A
        distance_corr_B: Distance correlation for VAE-B
        Q: Composite quality score (dist_corr + 1.5 * |hierarchy|)
        raw: Original metrics dict for additional fields
    """

    coverage: float | None = None
    hierarchy: float | None = None
    hierarchy_A: float | None = None
    hierarchy_B: float | None = None
    richness: float | None = None
    r_v0: float | None = None
    r_v9: float | None = None
    mean_radius: float | None = None
    std_radius: float | None = None
    distance_corr_A: float | None = None
    distance_corr_B: float | None = None
    Q: float | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key in [
            "coverage",
            "hierarchy",
            "hierarchy_A",
            "hierarchy_B",
            "richness",
            "r_v0",
            "r_v9",
            "mean_radius",
            "std_radius",
            "distance_corr_A",
            "distance_corr_B",
            "Q",
        ]:
            val = getattr(self, key)
            if val is not None:
                result[key] = val
        return result

    def __repr__(self) -> str:
        parts = []
        if self.coverage is not None:
            parts.append(f"cov={self.coverage * 100:.1f}%")
        if self.hierarchy is not None:
            parts.append(f"hier={self.hierarchy:.4f}")
        elif self.hierarchy_B is not None:
            parts.append(f"hier_B={self.hierarchy_B:.4f}")
        if self.richness is not None:
            parts.append(f"rich={self.richness:.6f}")
        if self.Q is not None:
            parts.append(f"Q={self.Q:.3f}")
        return f"CheckpointMetrics({', '.join(parts)})"


@dataclass
class CheckpointInfo:
    """Comprehensive checkpoint information with unified access.

    Attributes:
        path: Path to the checkpoint file
        epoch: Training epoch when saved
        metrics: Normalized metrics
        config: Training configuration
        recommended_usage: List of recommended use cases
        usage_metadata: Detailed usage information
        train_metrics: Training-time metrics (loss, etc.)
        homeostasis_state: Homeostasis controller state if present
        composite_score: Overall quality score (tau-weighted)
        tau: Temperature/threshold parameter
        raw_checkpoint: Full original checkpoint for custom access
    """

    path: Path
    epoch: int | None = None
    metrics: CheckpointMetrics = field(default_factory=CheckpointMetrics)
    config: dict = field(default_factory=dict)
    recommended_usage: list[str] = field(default_factory=list)
    usage_metadata: dict = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)
    homeostasis_state: dict = field(default_factory=dict)
    composite_score: float | None = None
    tau: float | None = None
    richness_ratio: float | None = None
    raw_checkpoint: dict = field(default_factory=dict, repr=False)

    def get_model_state(self) -> dict[str, torch.Tensor]:
        """Get model state dict from this checkpoint."""
        return get_model_state_dict(self.raw_checkpoint)

    def summary(self) -> str:
        """Generate human-readable summary of checkpoint."""
        lines = [f"Checkpoint: {self.path.name}"]
        if self.epoch is not None:
            lines.append(f"  Epoch: {self.epoch}")
        lines.append(f"  Metrics: {self.metrics}")
        if self.recommended_usage:
            lines.append(f"  Recommended: {', '.join(self.recommended_usage[:2])}")
        if self.composite_score is not None:
            lines.append(f"  Composite Score: {self.composite_score:.4f}")
        return "\n".join(lines)


def get_checkpoint_metrics(checkpoint: dict[str, Any]) -> CheckpointMetrics:
    """Extract and normalize metrics from checkpoint.

    Handles different metric key formats across checkpoint versions:
    - hierarchy vs radial_corr_A/radial_corr_B
    - r_v0/r_v9 vs radius_v0/radius_v9
    - Various coverage formats

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        CheckpointMetrics with normalized values
    """
    raw_metrics = checkpoint.get("metrics", {})

    # Extract coverage
    coverage = raw_metrics.get("coverage")
    if coverage is not None:
        coverage = float(coverage)

    # Extract hierarchy (different formats)
    hierarchy = raw_metrics.get("hierarchy")
    hierarchy_A = raw_metrics.get("radial_corr_A")
    hierarchy_B = raw_metrics.get("radial_corr_B")

    # Normalize to float
    if hierarchy is not None:
        hierarchy = float(hierarchy)
    if hierarchy_A is not None:
        hierarchy_A = float(hierarchy_A)
    if hierarchy_B is not None:
        hierarchy_B = float(hierarchy_B)

    # If no explicit hierarchy, use radial_corr_B (primary for p-adic)
    if hierarchy is None and hierarchy_B is not None:
        hierarchy = hierarchy_B

    # Extract richness
    richness = raw_metrics.get("richness")
    if richness is not None:
        richness = float(richness)

    # Extract radius bounds (different key formats)
    r_v0 = raw_metrics.get("r_v0") or raw_metrics.get("radius_v0")
    r_v9 = raw_metrics.get("r_v9") or raw_metrics.get("radius_v9")
    if r_v0 is not None:
        r_v0 = float(r_v0)
    if r_v9 is not None:
        r_v9 = float(r_v9)

    # Extract radius statistics
    mean_radius = raw_metrics.get("mean_radius") or raw_metrics.get("mean_radius_A")
    std_radius = raw_metrics.get("std_radius") or raw_metrics.get("radius_range_A")
    if mean_radius is not None:
        mean_radius = float(mean_radius)
    if std_radius is not None:
        std_radius = float(std_radius)

    # Extract distance correlation
    distance_corr_A = raw_metrics.get("distance_corr_A")
    distance_corr_B = raw_metrics.get("distance_corr_B")
    if distance_corr_A is not None:
        distance_corr_A = float(distance_corr_A)
    if distance_corr_B is not None:
        distance_corr_B = float(distance_corr_B)

    # Compute Q if we have the components
    Q = None
    if distance_corr_A is not None and hierarchy is not None:
        Q = distance_corr_A + 1.5 * abs(hierarchy)

    return CheckpointMetrics(
        coverage=coverage,
        hierarchy=hierarchy,
        hierarchy_A=hierarchy_A,
        hierarchy_B=hierarchy_B,
        richness=richness,
        r_v0=r_v0,
        r_v9=r_v9,
        mean_radius=mean_radius,
        std_radius=std_radius,
        distance_corr_A=distance_corr_A,
        distance_corr_B=distance_corr_B,
        Q=Q,
        raw=_to_python_type(raw_metrics),
    )


def get_checkpoint_info(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> CheckpointInfo:
    """Load checkpoint and extract comprehensive metadata.

    This is the main entry point for loading checkpoints with full
    metadata extraction. It handles all checkpoint format variations.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        CheckpointInfo with all extracted metadata

    Raises:
        FileNotFoundError: If checkpoint doesn't exist

    Example:
        >>> info = get_checkpoint_info("checkpoints/v5_11_homeostasis/best.pt")
        >>> print(info.metrics)
        CheckpointMetrics(cov=99.9%, hier=-0.7429, Q=1.081)
        >>> print(info.recommended_usage)
        ['production systems (coverage priority)', 'incremental/online learning']
        >>> model.load_state_dict(info.get_model_state(), strict=False)
    """
    path = Path(path)
    checkpoint = load_checkpoint_compat(path, map_location=map_location)

    # Extract epoch
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        epoch = int(epoch)

    # Extract metrics
    metrics = get_checkpoint_metrics(checkpoint)

    # Extract config
    config = _to_python_type(checkpoint.get("config", {}))

    # Extract usage information
    recommended_usage = checkpoint.get("recommended_usage", [])
    if isinstance(recommended_usage, str):
        recommended_usage = [recommended_usage]
    usage_metadata = _to_python_type(checkpoint.get("usage_metadata", {}))

    # Extract training metrics
    train_metrics = _to_python_type(checkpoint.get("train_metrics", {}))

    # Extract homeostasis state
    homeostasis_state = _to_python_type(checkpoint.get("homeostasis_state", {}))

    # Extract scores
    composite_score = checkpoint.get("composite_score")
    if composite_score is not None:
        composite_score = float(composite_score)

    tau = checkpoint.get("tau")
    if tau is not None:
        tau = float(tau)

    richness_ratio = checkpoint.get("richness_ratio")
    if richness_ratio is not None:
        richness_ratio = float(richness_ratio)

    return CheckpointInfo(
        path=path,
        epoch=epoch,
        metrics=metrics,
        config=config,
        recommended_usage=recommended_usage,
        usage_metadata=usage_metadata,
        train_metrics=train_metrics,
        homeostasis_state=homeostasis_state,
        composite_score=composite_score,
        tau=tau,
        richness_ratio=richness_ratio,
        raw_checkpoint=checkpoint,
    )


def list_checkpoints(
    base_dir: Path | str,
    pattern: str = "*/best.pt",
) -> list[CheckpointInfo]:
    """List all checkpoints in a directory with their metadata.

    Args:
        base_dir: Base directory containing checkpoint subdirectories
        pattern: Glob pattern for checkpoint files

    Returns:
        List of CheckpointInfo objects, sorted by path

    Example:
        >>> checkpoints = list_checkpoints("checkpoints")
        >>> for ckpt in checkpoints:
        ...     print(f"{ckpt.path.parent.name}: {ckpt.metrics}")
    """
    base_dir = Path(base_dir)
    checkpoints = []

    for path in sorted(base_dir.glob(pattern)):
        try:
            info = get_checkpoint_info(path)
            checkpoints.append(info)
        except Exception as e:
            # Skip invalid checkpoints but log warning
            print(f"Warning: Could not load {path}: {e}")

    return checkpoints


def compare_checkpoints(
    checkpoint_paths: list[Path | str],
    map_location: str | torch.device = "cpu",
) -> str:
    """Generate comparison table of multiple checkpoints.

    Args:
        checkpoint_paths: List of paths to compare
        map_location: Device to map tensors to

    Returns:
        Formatted comparison string

    Example:
        >>> print(compare_checkpoints([
        ...     "checkpoints/v5_11_homeostasis/best.pt",
        ...     "checkpoints/homeostatic_rich/best.pt",
        ... ]))
    """
    infos = [get_checkpoint_info(p, map_location) for p in checkpoint_paths]

    # Build header
    lines = [
        "| Checkpoint | Epoch | Coverage | Hierarchy | Richness | Q |",
        "|------------|-------|----------|-----------|----------|---|",
    ]

    for info in infos:
        name = info.path.parent.name
        epoch = str(info.epoch) if info.epoch is not None else "-"
        cov = f"{info.metrics.coverage * 100:.1f}%" if info.metrics.coverage else "-"
        hier = f"{info.metrics.hierarchy:.4f}" if info.metrics.hierarchy else "-"
        rich = f"{info.metrics.richness:.6f}" if info.metrics.richness else "-"
        Q = f"{info.metrics.Q:.3f}" if info.metrics.Q else "-"

        lines.append(f"| {name} | {epoch} | {cov} | {hier} | {rich} | {Q} |")

    return "\n".join(lines)


@dataclass
class ArchitectureConfig:
    """Architecture configuration extracted from checkpoint.

    Used to ensure model architecture matches checkpoint before loading.
    Mismatched architectures cause silent failures with strict=False.
    """

    latent_dim: int = 16
    hidden_dim: int = 64
    curvature: float = 1.0
    max_radius: float = 0.95
    use_controller: bool = False
    use_dual_projection: bool = True
    n_projection_layers: int = 1

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "ArchitectureConfig":
        """Extract architecture config from checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            ArchitectureConfig with values from checkpoint
        """
        config = checkpoint.get("config", {})

        # Handle nested config (some checkpoints have config.config)
        if isinstance(config, dict) and "config" in config and config["config"] is None:
            # The actual config values are at top level of config dict
            pass

        return cls(
            latent_dim=config.get("latent_dim", 16),
            hidden_dim=config.get("hidden_dim", config.get("projection_hidden_dim", 64)),
            curvature=config.get("curvature", 1.0),
            max_radius=config.get("max_radius", 0.95),
            use_controller=config.get("use_controller", False),
            use_dual_projection=config.get("dual_projection", True),
            n_projection_layers=config.get("projection_layers", 1),
        )

    @classmethod
    def from_model_state(cls, state_dict: dict[str, torch.Tensor]) -> "ArchitectureConfig":
        """Infer architecture config from model state dict keys and shapes.

        Args:
            state_dict: Model state dictionary

        Returns:
            ArchitectureConfig inferred from state dict
        """
        # Check for controller
        use_controller = any(k.startswith("controller.") for k in state_dict)

        # Infer hidden_dim from encoder weight shape
        hidden_dim = 64
        for key in ["encoder_A.encoder.0.weight", "encoder_A.layers.0.weight"]:
            if key in state_dict:
                hidden_dim = state_dict[key].shape[0]
                break

        # Infer latent_dim from fc_mu weight shape
        latent_dim = 16
        if "encoder_A.fc_mu.weight" in state_dict:
            latent_dim = state_dict["encoder_A.fc_mu.weight"].shape[0]

        # Count projection layers by checking direction_net weights
        n_projection_layers = 1
        for i in range(10):
            if f"projection.proj_A.direction_net.{i * 2}.weight" in state_dict:
                n_projection_layers = i // 2 + 1

        return cls(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_controller=use_controller,
            n_projection_layers=n_projection_layers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for model construction."""
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "curvature": self.curvature,
            "max_radius": self.max_radius,
            "use_controller": self.use_controller,
            "use_dual_projection": self.use_dual_projection,
        }


class CheckpointArchitectureMismatch(Exception):
    """Raised when model architecture doesn't match checkpoint."""

    pass


def validate_checkpoint_architecture(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
    raise_on_mismatch: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """Validate that model architecture matches checkpoint.

    This catches the silent failure bug where strict=False loading
    leaves projection layers randomly initialized.

    Args:
        model_state: State dict from model.state_dict()
        checkpoint_state: State dict from checkpoint
        raise_on_mismatch: Whether to raise exception on mismatch

    Returns:
        Tuple of (is_valid, missing_keys, extra_keys)

    Raises:
        CheckpointArchitectureMismatch: If mismatch and raise_on_mismatch=True
    """
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())

    missing_keys = sorted(model_keys - checkpoint_keys)
    extra_keys = sorted(checkpoint_keys - model_keys)

    # Critical keys that MUST match (projection layers)
    critical_missing = [k for k in missing_keys if "projection" in k or "manifold" in k]

    is_valid = len(critical_missing) == 0

    if not is_valid and raise_on_mismatch:
        msg = (
            f"Checkpoint architecture mismatch!\n"
            f"  Missing {len(missing_keys)} keys ({len(critical_missing)} critical)\n"
            f"  Extra {len(extra_keys)} keys\n"
            f"  Critical missing: {critical_missing[:5]}...\n"
            f"  This will cause random initialization of projection layers.\n"
            f"  Fix: Match model architecture to checkpoint config."
        )
        raise CheckpointArchitectureMismatch(msg)

    return is_valid, missing_keys, extra_keys


def load_checkpoint_safe(
    path: Path | str,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    validate_architecture: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Load checkpoint with architecture validation.

    This is the recommended way to load checkpoints. It validates that
    the model architecture matches the checkpoint before loading, preventing
    the silent failure bug with strict=False.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        map_location: Device to map tensors to
        strict: Whether to use strict loading (recommended: True)
        validate_architecture: Whether to validate architecture match

    Returns:
        Tuple of (checkpoint_dict, was_strict_load_successful)

    Raises:
        CheckpointArchitectureMismatch: If architecture validation fails

    Example:
        >>> model = TernaryVAEV5_11_PartialFreeze(**arch_config.to_dict())
        >>> ckpt, strict_ok = load_checkpoint_safe("best.pt", model)
        >>> if not strict_ok:
        ...     print("Warning: Some keys didn't match")
    """
    checkpoint = load_checkpoint_compat(path, map_location)
    checkpoint_state = get_model_state_dict(checkpoint)
    model_state = model.state_dict()

    # Validate architecture
    if validate_architecture:
        validate_checkpoint_architecture(model_state, checkpoint_state, raise_on_mismatch=True)

    # Try strict load
    try:
        model.load_state_dict(checkpoint_state, strict=True)
        return checkpoint, True
    except RuntimeError:
        if strict:
            raise
        # Fall back to non-strict
        model.load_state_dict(checkpoint_state, strict=False)
        return checkpoint, False


def get_checkpoint_architecture(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> ArchitectureConfig:
    """Get architecture config needed to load a checkpoint.

    Use this to determine what model architecture to create before loading.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        ArchitectureConfig with required architecture settings

    Example:
        >>> arch = get_checkpoint_architecture("v5_11_homeostasis/best.pt")
        >>> print(f"Need: curvature={arch.curvature}, controller={arch.use_controller}")
        >>> model = TernaryVAEV5_11_PartialFreeze(**arch.to_dict())
    """
    checkpoint = load_checkpoint_compat(path, map_location)

    # First try to get from stored config
    arch = ArchitectureConfig.from_checkpoint(checkpoint)

    # Validate against state dict
    checkpoint_state = get_model_state_dict(checkpoint)
    inferred = ArchitectureConfig.from_model_state(checkpoint_state)

    # Use inferred values for fields that can be determined from state dict
    arch.use_controller = inferred.use_controller
    arch.latent_dim = inferred.latent_dim
    arch.hidden_dim = inferred.hidden_dim

    return arch


__all__ = [
    # Core loading
    "NumpyBackwardsCompatUnpickler",
    "load_checkpoint_compat",
    "save_checkpoint",
    # Model state extraction
    "get_model_state_dict",
    "extract_model_state",
    # Architecture validation (NEW)
    "ArchitectureConfig",
    "CheckpointArchitectureMismatch",
    "validate_checkpoint_architecture",
    "load_checkpoint_safe",
    "get_checkpoint_architecture",
    # Comprehensive metadata
    "CheckpointMetrics",
    "CheckpointInfo",
    "get_checkpoint_metrics",
    "get_checkpoint_info",
    # Utilities
    "list_checkpoints",
    "compare_checkpoints",
]
