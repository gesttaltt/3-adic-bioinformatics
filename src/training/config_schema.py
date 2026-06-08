# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Configuration schema with typed validation for Ternary VAE training.

This module provides:
- Typed dataclasses for all configuration sections
- Validation functions to catch misconfigurations before training
- Sensible defaults for optional parameters

Single responsibility: Configuration typing and validation only.
"""

from dataclasses import dataclass, field
from typing import Any

from src.config.paths import CHECKPOINTS_DIR, RUNS_DIR


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 9
    latent_dim: int = 16
    rho_min: float = 0.1
    rho_max: float = 0.7
    lambda3_base: float = 0.3
    lambda3_amplitude: float = 0.15
    eps_kl: float = 0.0005
    gradient_balance: bool = True
    adaptive_scheduling: bool = True
    use_statenet: bool = True
    statenet_lr_scale: float = 0.1
    statenet_lambda_scale: float = 0.02
    statenet_ranking_scale: float = 0.3
    statenet_hyp_sigma_scale: float = 0.05
    statenet_hyp_curvature_scale: float = 0.02


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adamw"
    lr_start: float = 0.001
    weight_decay: float = 0.0001
    lr_schedule: list[dict[str, float]] = field(default_factory=list)


@dataclass
class VAEConfig:
    """VAE-specific parameters (A or B)."""

    beta_start: float = 0.3
    beta_end: float = 0.8
    beta_warmup_epochs: int = 50
    temp_start: float = 1.0
    temp_end: float = 0.3
    temp_cyclic: bool = False
    temp_boost_amplitude: float = 0.5
    temp_phase4: float = 0.3
    entropy_weight: float = 0.0
    repulsion_weight: float = 0.0


@dataclass
class ContinuousFeedbackConfig:
    """Continuous feedback configuration for adaptive ranking."""

    enabled: bool = True
    base_ranking_weight: float = 0.4
    coverage_threshold: float = 92.0
    coverage_sensitivity: float = 0.05
    coverage_trend_sensitivity: float = 1.0
    min_ranking_weight: float = 0.1
    max_ranking_weight: float = 0.8
    coverage_ema_alpha: float = 0.95


@dataclass
class HyperbolicPriorConfig:
    """Hyperbolic prior configuration."""

    homeostatic: bool = True
    latent_dim: int = 16
    curvature: float = 2.0
    prior_sigma: float = 1.0
    max_norm: float = 0.95
    sigma_min: float = 0.3
    sigma_max: float = 2.0
    curvature_min: float = 0.5
    curvature_max: float = 4.0
    adaptation_rate: float = 0.01


@dataclass
class HyperbolicReconConfig:
    """Hyperbolic reconstruction loss configuration."""

    homeostatic: bool = True
    mode: str = "weighted_ce"
    curvature: float = 2.0
    max_norm: float = 0.95
    geodesic_weight: float = 0.3
    radius_weighting: bool = True
    radius_power: float = 2.0
    weight: float = 0.5
    geodesic_weight_min: float = 0.1
    geodesic_weight_max: float = 0.8
    radius_power_min: float = 1.0
    radius_power_max: float = 4.0
    adaptation_rate: float = 0.01


@dataclass
class CentroidConfig:
    """Hyperbolic centroid loss configuration."""

    max_level: int = 4
    curvature: float = 2.0
    max_norm: float = 0.95
    weight: float = 0.2


@dataclass
class HyperbolicV10Config:
    """v5.10 hyperbolic modules configuration."""

    use_hyperbolic_prior: bool = False
    use_hyperbolic_recon: bool = False
    use_centroid_loss: bool = False
    prior: HyperbolicPriorConfig = field(default_factory=HyperbolicPriorConfig)
    recon: HyperbolicReconConfig = field(default_factory=HyperbolicReconConfig)
    centroid: CentroidConfig = field(default_factory=CentroidConfig)


@dataclass
class RankingHyperbolicConfig:
    """Hyperbolic ranking loss configuration."""

    base_margin: float = 0.05
    margin_scale: float = 0.15
    n_triplets: int = 500
    hard_negative_ratio: float = 0.5
    curvature: float = 2.0
    radial_weight: float = 0.4
    max_norm: float = 0.95
    weight: float = 0.5


@dataclass
class PAdicLossesConfig:
    """p-Adic losses configuration."""

    enable_metric_loss: bool = False
    metric_loss_weight: float = 0.0
    enable_ranking_loss_hyperbolic: bool = True
    enable_norm_loss: bool = False
    norm_loss_weight: float = 0.0
    ranking_hyperbolic: RankingHyperbolicConfig = field(default_factory=RankingHyperbolicConfig)
    hyperbolic_v10: HyperbolicV10Config = field(default_factory=HyperbolicV10Config)


@dataclass
class ControllerConfig:
    """Training controller parameters."""

    temp_lag: int = 30
    beta_phase_lag: float = 0.785
    entropy_ema_alpha: float = 0.9
    dH_dt_threshold: float = 0.05


@dataclass
class PhaseTransitionsConfig:
    """Training phase transition epochs."""

    entropy_expansion_end: int = 40
    consolidation_end: int = 120
    resonant_coupling_end: int = 250
    ultra_exploration_start: int = 250
    statenet_warm_start: int = 20


@dataclass
class TorchCompileConfig:
    """TorchInductor compilation settings."""

    enabled: bool = False
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Required sections
    model: ModelConfig
    optimizer: OptimizerConfig
    vae_a: VAEConfig
    vae_b: VAEConfig

    # Training parameters
    seed: int = 42
    batch_size: int = 256
    num_workers: int = 0
    total_epochs: int = 300
    grad_clip: float = 1.0
    free_bits: float = 0.3

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Early stopping (loss-based)
    patience: int = 150
    min_delta: float = 0.0001

    # Coverage plateau detection (for manifold approach)
    coverage_plateau_patience: int = 100
    coverage_plateau_min_delta: float = 0.0005

    # Evaluation intervals
    eval_num_samples: int = 1000
    eval_interval: int = 20
    coverage_check_interval: int = 5

    # Logging and checkpointing
    log_interval: int = 1
    log_dir: str = "logs"
    checkpoint_freq: int = 10
    checkpoint_dir: str = str(CHECKPOINTS_DIR / "v5_10")
    tensorboard_dir: str = str(RUNS_DIR)
    experiment_name: str | None = None
    histogram_interval: int = 10
    embedding_interval: int = 50  # Log embeddings every N epochs (0 to disable)
    embedding_n_samples: int = 5000  # Number of samples for embedding visualization

    # Optional sections
    continuous_feedback: ContinuousFeedbackConfig = field(default_factory=ContinuousFeedbackConfig)
    padic_losses: PAdicLossesConfig = field(default_factory=PAdicLossesConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    phase_transitions: PhaseTransitionsConfig = field(default_factory=PhaseTransitionsConfig)
    torch_compile: TorchCompileConfig = field(default_factory=TorchCompileConfig)

    # Targets (informational)
    target_coverage_percent: float = 99.7
    target_ranking_correlation: float = 0.99


def _build_nested_config(raw: dict[str, Any], cls, defaults: dict[str, Any] | None = None) -> Any:
    """Build a dataclass from raw dict, using defaults for missing keys."""
    if raw is None:
        raw = {}
    if defaults is None:
        defaults = {}

    merged = {**defaults, **raw}

    # Get field types for nested dataclasses
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

    kwargs = {}
    for key, value in merged.items():
        if key in field_types:
            field_type = field_types[key]
            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _build_nested_config(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def validate_config(raw_config: dict[str, Any]) -> TrainingConfig:
    """Validate raw YAML dict and return typed TrainingConfig.

    Args:
        raw_config: Raw configuration dictionary from YAML

    Returns:
        Validated TrainingConfig dataclass

    Raises:
        ConfigValidationError: If required keys are missing or values are invalid
    """
    errors = []

    # Check required top-level keys
    required_keys = ["model", "optimizer", "vae_a", "vae_b"]
    for key in required_keys:
        if key not in raw_config:
            errors.append(f"Missing required section: '{key}'")

    if errors:
        raise ConfigValidationError("Configuration errors:\n  " + "\n  ".join(errors))

    # Build nested configs
    try:
        model = _build_nested_config(raw_config.get("model", {}), ModelConfig)
        optimizer = _build_nested_config(raw_config.get("optimizer", {}), OptimizerConfig)
        vae_a = _build_nested_config(raw_config.get("vae_a", {}), VAEConfig)
        vae_b = _build_nested_config(raw_config.get("vae_b", {}), VAEConfig)

        # Optional sections
        continuous_feedback = _build_nested_config(raw_config.get("continuous_feedback", {}), ContinuousFeedbackConfig)
        controller = _build_nested_config(raw_config.get("controller", {}), ControllerConfig)
        phase_transitions = _build_nested_config(raw_config.get("phase_transitions", {}), PhaseTransitionsConfig)
        torch_compile = _build_nested_config(raw_config.get("torch_compile", {}), TorchCompileConfig)

        # p-Adic losses (deeply nested)
        padic_raw = raw_config.get("padic_losses", {})
        ranking_hyp = _build_nested_config(padic_raw.get("ranking_hyperbolic", {}), RankingHyperbolicConfig)

        hyp_v10_raw = padic_raw.get("hyperbolic_v10", {})
        hyp_prior = _build_nested_config(hyp_v10_raw.get("prior", {}), HyperbolicPriorConfig)
        hyp_recon = _build_nested_config(hyp_v10_raw.get("recon", {}), HyperbolicReconConfig)
        centroid = _build_nested_config(hyp_v10_raw.get("centroid", {}), CentroidConfig)

        hyperbolic_v10 = HyperbolicV10Config(
            use_hyperbolic_prior=hyp_v10_raw.get("use_hyperbolic_prior", False),
            use_hyperbolic_recon=hyp_v10_raw.get("use_hyperbolic_recon", False),
            use_centroid_loss=hyp_v10_raw.get("use_centroid_loss", False),
            prior=hyp_prior,
            recon=hyp_recon,
            centroid=centroid,
        )

        padic_losses = PAdicLossesConfig(
            enable_metric_loss=padic_raw.get("enable_metric_loss", False),
            metric_loss_weight=padic_raw.get("metric_loss_weight", 0.0),
            enable_ranking_loss_hyperbolic=padic_raw.get("enable_ranking_loss_hyperbolic", True),
            enable_norm_loss=padic_raw.get("enable_norm_loss", False),
            norm_loss_weight=padic_raw.get("norm_loss_weight", 0.0),
            ranking_hyperbolic=ranking_hyp,
            hyperbolic_v10=hyperbolic_v10,
        )

    except Exception as e:
        raise ConfigValidationError(f"Error building config: {e}")

    # Value validations
    if raw_config.get("batch_size", 256) < 1:
        errors.append("batch_size must be >= 1")
    if raw_config.get("total_epochs", 300) < 1:
        errors.append("total_epochs must be >= 1")
    if not (0 < raw_config.get("train_split", 0.8) <= 1):
        errors.append("train_split must be in (0, 1]")

    splits_sum = (
        raw_config.get("train_split", 0.8) + raw_config.get("val_split", 0.1) + raw_config.get("test_split", 0.1)
    )
    if abs(splits_sum - 1.0) > 0.001:
        errors.append(f"Data splits must sum to 1.0, got {splits_sum}")

    if model.latent_dim < 2:
        errors.append("model.latent_dim must be >= 2")
    if not (0 <= model.rho_min < model.rho_max <= 1):
        errors.append("model.rho_min < model.rho_max required, both in [0, 1]")

    if errors:
        raise ConfigValidationError("Configuration errors:\n  " + "\n  ".join(errors))

    # Build final config
    config = TrainingConfig(
        model=model,
        optimizer=optimizer,
        vae_a=vae_a,
        vae_b=vae_b,
        seed=raw_config.get("seed", 42),
        batch_size=raw_config.get("batch_size", 256),
        num_workers=raw_config.get("num_workers", 0),
        total_epochs=raw_config.get("total_epochs", 300),
        grad_clip=raw_config.get("grad_clip", 1.0),
        free_bits=raw_config.get("free_bits", 0.3),
        train_split=raw_config.get("train_split", 0.8),
        val_split=raw_config.get("val_split", 0.1),
        test_split=raw_config.get("test_split", 0.1),
        patience=raw_config.get("patience", 150),
        min_delta=raw_config.get("min_delta", 0.0001),
        eval_num_samples=raw_config.get("eval_num_samples", 1000),
        eval_interval=raw_config.get("eval_interval", 20),
        coverage_check_interval=raw_config.get("coverage_check_interval", 5),
        log_interval=raw_config.get("log_interval", 1),
        log_dir=raw_config.get("log_dir", "logs"),
        checkpoint_freq=raw_config.get("checkpoint_freq", 10),
        checkpoint_dir=raw_config.get("checkpoint_dir", str(CHECKPOINTS_DIR / "v5_10")),
        tensorboard_dir=raw_config.get("tensorboard_dir", "runs"),
        experiment_name=raw_config.get("experiment_name"),
        histogram_interval=raw_config.get("histogram_interval", 10),
        embedding_interval=raw_config.get("embedding_interval", 50),
        embedding_n_samples=raw_config.get("embedding_n_samples", 5000),
        continuous_feedback=continuous_feedback,
        padic_losses=padic_losses,
        controller=controller,
        phase_transitions=phase_transitions,
        torch_compile=torch_compile,
        target_coverage_percent=raw_config.get("target_coverage_percent", 99.7),
        target_ranking_correlation=raw_config.get("target_ranking_correlation", 0.99),
    )

    return config


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Convert TrainingConfig back to dict for passing to existing code.

    This allows gradual migration - validated config can still be used
    with code expecting raw dicts.

    Args:
        config: Validated TrainingConfig

    Returns:
        Dictionary representation compatible with existing code
    """
    from dataclasses import asdict

    return asdict(config)


__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "VAEConfig",
    "PAdicLossesConfig",
    "ConfigValidationError",
    "validate_config",
    "config_to_dict",
]
