# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE - Canonical V5.11 Architecture.

This package implements Dual Neural VAEs for learning 3-adic algebraic
structure in ternary operation space using hyperbolic geometry.

Core Modules:
- config: Centralized configuration (constants, schema, loader)
- models: VAE architectures (canonical v5.11 with frozen encoder)
- losses: Loss functions including p-adic geodesic loss
- training: Trainers, schedulers, monitoring, callbacks
- data: Ternary operation generation and datasets
- metrics: Hyperbolic correlation metrics
- observability: Logging, metrics buffer, async TensorBoard writer

Advanced Modules (Production):
- graphs: Hyperbolic graph neural networks
- topology: Persistent homology and topological data analysis
- information: Fisher information geometry and natural gradients
- contrastive: P-adic contrastive learning
- physics: Statistical physics and spin glass methods
- tropical: Tropical geometry for neural network analysis
- categorical: Category theory for neural networks
- meta: Meta-learning (MAML, Reptile)
- equivariant: SO(3)/SE(3)-equivariant networks and codon symmetry
- diffusion: Discrete diffusion models for codon sequences
"""

__version__ = "5.12.0"
__author__ = "AI Whisperers"
__license__ = "PolyForm-Noncommercial-1.0.0"

# Configuration (new centralized system)
# Advanced Modules (experimental - lazy imports to avoid heavy dependencies at startup)
# These are imported on demand when accessed
from ._experimental import (
    categorical,
    contrastive,
    diffusion,
    equivariant,
    graphs,
    information,
    meta,
    physics,
    topology,
    tropical,
)
from .config import TrainingConfig, load_config, save_config

# Metrics
from .core.metrics import compute_ranking_correlation_hyperbolic

# Data
from .dataio import TernaryOperationDataset, generate_all_ternary_operations

# Losses (registry pattern)
from .losses import (
    LossRegistry,
    create_registry_from_config,
    create_registry_from_training_config,
)

# Canonical model (V5.11)
from .models.ternary_vae import (
    TernaryVAEV5_11,
    TernaryVAEV5_11_OptionC,
    TernaryVAEV5_11_PartialFreeze,
)

# Training
from .training import HyperbolicVAETrainer, TernaryVAETrainer, TrainingMonitor
from .training.callbacks import CallbackList, CheckpointCallback, EarlyStoppingCallback

# Observability
from .utils.observability import MetricsBuffer, setup_logging
from .utils.observability.logging import get_logger

# Canonical aliases (after imports)
TernaryVAE = TernaryVAEV5_11
TernaryVAE_PartialFreeze = TernaryVAEV5_11_PartialFreeze
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC  # Deprecated alias

__all__ = [
    # Configuration
    "TrainingConfig",
    "load_config",
    "save_config",
    # Canonical (V5.11)
    "TernaryVAE",
    "TernaryVAE_PartialFreeze",
    "TernaryVAE_OptionC",  # Deprecated alias
    "TernaryVAEV5_11",
    "TernaryVAEV5_11_PartialFreeze",
    "TernaryVAEV5_11_OptionC",  # Deprecated alias
    # Data
    "generate_all_ternary_operations",
    "TernaryOperationDataset",
    # Losses
    "LossRegistry",
    "create_registry_from_config",
    "create_registry_from_training_config",
    # Training
    "TernaryVAETrainer",
    "HyperbolicVAETrainer",
    "TrainingMonitor",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    # Observability
    "setup_logging",
    "get_logger",
    "MetricsBuffer",
    # Metrics
    "compute_ranking_correlation_hyperbolic",
    # Advanced Modules
    "graphs",
    "topology",
    "information",
    "contrastive",
    "physics",
    "tropical",
    "categorical",
    "meta",
    "equivariant",
    "diffusion",
]
