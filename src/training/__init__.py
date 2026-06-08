# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Training orchestration components.

This module contains components for managing the training process:
- BaseTrainer: Abstract base with defensive patterns (safe division, val_loader guards)
- TernaryVAETrainer: Main training loop (single responsibility)
- HyperbolicVAETrainer: Hyperbolic geometry trainer (canonical)
- Schedulers: Parameter scheduling (temperature, beta, learning rate)
- Monitor: Logging and metrics tracking (TensorBoard + file)
- ConfigSchema: Typed configuration validation
- Environment: Pre-training environment checks
- CheckpointManager: Async checkpoint saving (consolidated from artifacts/)
- Optimizers: Riemannian and multi-objective optimizers (consolidated from optimizers/)
"""

from .base import STATENET_KEYS, BaseTrainer
from .checkpoint_manager import AsyncCheckpointSaver, CheckpointManager
from .config_schema import ConfigValidationError, ModelConfig, TrainingConfig, config_to_dict, validate_config
from .curriculum import AdaptiveCurriculum, CurriculumState
from .data import StratifiedBatchSampler, TernaryDataset, create_stratified_batches
from .environment import EnvironmentStatus, require_valid_environment, validate_environment
from .grokking_detector import (
    EpochMetrics,
    GrokAnalysis,
    GrokDetector,
    GrokDetectorConfig,
    LocalComplexityEstimator,
    TrainingPhase,
    WeightNormTracker,
    analyze_training_log,
)
from .hyperbolic_trainer import HyperbolicVAETrainer
from .monitor import TrainingMonitor
from .schedulers import BetaScheduler, LearningRateScheduler, TemperatureScheduler, cyclic_schedule, linear_schedule
from .self_supervised import (
    ContrastiveHead,
    MaskedSequenceModeling,
    MutationHead,
    PretrainingObjective,
    SelfSupervisedConfig,
    SelfSupervisedModel,
    SelfSupervisedPretrainer,
    SequenceAugmenter,
    SequenceDecoder,
    SequenceEncoder,
)
from .trainer import TernaryVAETrainer
from .transfer_pipeline import (
    AdapterLayer,
    DiseaseHead,
    LoRALayer,
    MultiDiseaseModel,
    SharedEncoder,
    TransferConfig,
    TransferLearningPipeline,
    TransferStrategy,
)

__all__ = [
    # Base trainer
    "BaseTrainer",
    "STATENET_KEYS",
    # Trainers
    "TernaryVAETrainer",
    "HyperbolicVAETrainer",
    # Checkpoint management (consolidated from artifacts/)
    "CheckpointManager",
    "AsyncCheckpointSaver",
    # Schedulers
    "TemperatureScheduler",
    "BetaScheduler",
    "LearningRateScheduler",
    "linear_schedule",
    "cyclic_schedule",
    # Data and sampling
    "TernaryDataset",
    "StratifiedBatchSampler",
    "create_stratified_batches",
    # Monitoring
    "TrainingMonitor",
    # Config validation
    "TrainingConfig",
    "ModelConfig",
    "ConfigValidationError",
    "validate_config",
    "config_to_dict",
    # Environment validation
    "EnvironmentStatus",
    "validate_environment",
    "require_valid_environment",
    # Curriculum learning
    "AdaptiveCurriculum",
    "CurriculumState",
    # Grokking detection
    "GrokDetector",
    "GrokDetectorConfig",
    "GrokAnalysis",
    "EpochMetrics",
    "TrainingPhase",
    "LocalComplexityEstimator",
    "WeightNormTracker",
    "analyze_training_log",
    # Self-supervised pre-training
    "SelfSupervisedPretrainer",
    "SelfSupervisedConfig",
    "SelfSupervisedModel",
    "SequenceEncoder",
    "SequenceDecoder",
    "ContrastiveHead",
    "MaskedSequenceModeling",
    "MutationHead",
    "SequenceAugmenter",
    "PretrainingObjective",
    # Transfer learning pipeline
    "TransferLearningPipeline",
    "TransferConfig",
    "TransferStrategy",
    "MultiDiseaseModel",
    "SharedEncoder",
    "DiseaseHead",
    "AdapterLayer",
    "LoRALayer",
]
