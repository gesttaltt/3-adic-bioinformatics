# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-task learning module.

Provides architectures for jointly predicting multiple related tasks:
- Drug resistance across multiple drugs
- Cross-resistance patterns
- Escape mutations
- Fitness costs

Key components:
- MultiTaskResistancePredictor: Main MTL model for drug resistance
- TaskHead: Individual task-specific heads
- CrossTaskAttention: Attention between task representations
- GradNorm: Automatic task weighting
"""

from src.models.mtl.gradnorm import GradNormOptimizer
from src.models.mtl.resistance_predictor import (
    MTLConfig,
    MultiTaskResistancePredictor,
)
from src.models.mtl.task_heads import (
    ClassificationHead,
    CrossTaskAttention,
    RegressionHead,
    TaskHead,
)

__all__ = [
    "MultiTaskResistancePredictor",
    "MTLConfig",
    "TaskHead",
    "ClassificationHead",
    "RegressionHead",
    "CrossTaskAttention",
    "GradNormOptimizer",
]
