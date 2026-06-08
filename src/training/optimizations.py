# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training optimizations for improved performance and efficiency.

Implements:
- Mixed Precision Training (FP16/BF16)
- Gradient Checkpointing
- Stochastic Weight Averaging (SWA)
- Dynamic Batching
- Gradient Accumulation

References:
    - Micikevicius et al., "Mixed Precision Training" (2018)
    - Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
    - Izmailov et al., "Averaging Weights Leads to Wider Optima" (2018)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Mixed Precision Training
# =============================================================================


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training.

    Attributes:
        enabled: Whether to use mixed precision
        dtype: Data type for forward pass ('float16' or 'bfloat16')
        init_scale: Initial loss scale for gradient scaler
        growth_factor: Scale growth factor
        backoff_factor: Scale reduction factor on overflow
        growth_interval: Steps between scale increases
    """

    enabled: bool = True
    dtype: str = "float16"
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000


class MixedPrecisionTrainer:
    """Mixed precision training wrapper.

    Handles automatic mixed precision (AMP) with dynamic loss scaling.
    Supports both FP16 and BF16 (on supported hardware).

    Example:
        >>> mp_trainer = MixedPrecisionTrainer(config)
        >>> for batch in train_loader:
        ...     with mp_trainer.autocast():
        ...         outputs = model(batch)
        ...         loss = criterion(outputs, targets)
        ...     mp_trainer.backward(loss)
        ...     mp_trainer.step(optimizer)
    """

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        """Initialize mixed precision trainer.

        Args:
            config: Mixed precision configuration
        """
        self.config = config or MixedPrecisionConfig()

        if self.config.enabled:
            # Determine dtype
            if self.config.dtype == "bfloat16":
                self.dtype = torch.bfloat16
                # BF16 doesn't need loss scaling
                self.scaler = None
            else:
                self.dtype = torch.float16
                self.scaler = GradScaler(
                    device="cuda",
                    init_scale=self.config.init_scale,
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                )
        else:
            self.dtype = torch.float32
            self.scaler = None

        self._enabled = self.config.enabled

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision.

        Yields:
            Context with mixed precision enabled
        """
        if self._enabled:
            with autocast(device_type="cuda", dtype=self.dtype):
                yield
        else:
            yield

    def backward(self, loss: Tensor) -> None:
        """Backward pass with optional loss scaling.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(
        self,
        optimizer: Optimizer,
        clip_grad_norm: Optional[float] = None,
        parameters: Optional[Iterator[Tensor]] = None,
    ) -> Optional[float]:
        """Optimizer step with optional gradient clipping.

        Args:
            optimizer: Optimizer to step
            clip_grad_norm: Max gradient norm (optional)
            parameters: Model parameters for gradient clipping

        Returns:
            Gradient norm if clipping, else None
        """
        grad_norm = None

        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)

            if clip_grad_norm is not None and parameters is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, clip_grad_norm
                ).item()

            # Step optimizer with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_grad_norm is not None and parameters is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, clip_grad_norm
                ).item()

            optimizer.step()

        return grad_norm

    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        if self.scaler is not None:
            return {"scaler": self.scaler.state_dict()}
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])


# =============================================================================
# Gradient Checkpointing
# =============================================================================


class CheckpointedModule(nn.Module):
    """Wrapper that applies gradient checkpointing to a module.

    Trades computation for memory by not storing intermediate activations.
    Activations are recomputed during backward pass.

    Example:
        >>> encoder = CheckpointedModule(Encoder(), num_segments=4)
        >>> output = encoder(input)  # Memory-efficient forward
    """

    def __init__(
        self,
        module: nn.Module,
        num_segments: int = 2,
        preserve_rng_state: bool = True,
    ):
        """Initialize checkpointed module.

        Args:
            module: Module to wrap
            num_segments: Number of checkpoint segments
            preserve_rng_state: Preserve RNG state across checkpoints
        """
        super().__init__()
        self.module = module
        self.num_segments = num_segments
        self.preserve_rng_state = preserve_rng_state

        # Try to split into sequential segments
        if isinstance(module, nn.Sequential):
            self._is_sequential = True
        else:
            self._is_sequential = False

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass with checkpointing.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Module output
        """
        if not self.training:
            # No checkpointing during inference
            return self.module(*args, **kwargs)

        if self._is_sequential and len(args) == 1 and not kwargs:
            # Use efficient sequential checkpointing
            return checkpoint_sequential(
                self.module,
                self.num_segments,
                args[0],
                preserve_rng_state=self.preserve_rng_state,
            )
        else:
            # Wrap entire module
            return checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                preserve_rng_state=self.preserve_rng_state,
                **kwargs,
            )


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_layers: Optional[List[str]] = None,
    num_segments: int = 2,
) -> nn.Module:
    """Apply gradient checkpointing to model layers.

    Args:
        model: Model to modify
        checkpoint_layers: Names of layers to checkpoint (None = auto-detect)
        num_segments: Number of segments per checkpointed layer

    Returns:
        Model with checkpointing applied
    """
    if checkpoint_layers is None:
        # Auto-detect large sequential modules
        checkpoint_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and len(list(module.children())) > 2:
                checkpoint_layers.append(name)

    for name in checkpoint_layers:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with checkpointed version
        original = getattr(parent, parts[-1])
        checkpointed = CheckpointedModule(original, num_segments=num_segments)
        setattr(parent, parts[-1], checkpointed)

        logger.info(f"Applied gradient checkpointing to {name}")

    return model


# =============================================================================
# Stochastic Weight Averaging (SWA)
# =============================================================================


@dataclass
class SWAConfig:
    """Configuration for Stochastic Weight Averaging.

    Attributes:
        swa_start: Epoch to start SWA
        swa_lr: Learning rate for SWA phase
        swa_freq: Frequency of model averaging (epochs)
        anneal_epochs: Epochs to anneal LR to swa_lr
        anneal_strategy: LR annealing strategy ('linear' or 'cos')
    """

    swa_start: int = 75
    swa_lr: float = 0.001
    swa_freq: int = 1
    anneal_epochs: int = 10
    anneal_strategy: str = "cos"


class SWAWrapper:
    """Stochastic Weight Averaging wrapper.

    Maintains an exponential moving average of model weights during
    the final phase of training for better generalization.

    Example:
        >>> swa = SWAWrapper(model, optimizer, config)
        >>> for epoch in range(total_epochs):
        ...     train_epoch(model)
        ...     swa.step(epoch)
        >>> swa.finalize()  # Update batch norm stats
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: Optional[SWAConfig] = None,
        device: str = "cuda",
    ):
        """Initialize SWA wrapper.

        Args:
            model: Model to average
            optimizer: Training optimizer
            config: SWA configuration
            device: Device for computation
        """
        self.config = config or SWAConfig()
        self.device = device

        # Create averaged model
        self.swa_model = AveragedModel(model, device=device)

        # Create SWA scheduler
        self.swa_scheduler = SWALR(
            optimizer,
            swa_lr=self.config.swa_lr,
            anneal_epochs=self.config.anneal_epochs,
            anneal_strategy=self.config.anneal_strategy,
        )

        self._swa_active = False
        self._update_count = 0

    def step(self, epoch: int) -> bool:
        """Update SWA state for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            True if SWA update was performed
        """
        if epoch < self.config.swa_start:
            return False

        self._swa_active = True

        # Update scheduler
        self.swa_scheduler.step()

        # Update averaged model
        if (epoch - self.config.swa_start) % self.config.swa_freq == 0:
            self.swa_model.update_parameters(self.swa_model.module)
            self._update_count += 1
            return True

        return False

    def finalize(self, train_loader: DataLoader) -> nn.Module:
        """Finalize SWA by updating batch norm statistics.

        Args:
            train_loader: Training data loader for BN update

        Returns:
            Finalized SWA model
        """
        if self._update_count == 0:
            logger.warning("No SWA updates performed")
            return self.swa_model.module

        # Update batch norm running stats
        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)

        logger.info(f"SWA finalized after {self._update_count} updates")
        return self.swa_model

    def get_swa_model(self) -> AveragedModel:
        """Get the averaged model."""
        return self.swa_model

    def is_active(self) -> bool:
        """Check if SWA is currently active."""
        return self._swa_active

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "swa_model": self.swa_model.state_dict(),
            "swa_scheduler": self.swa_scheduler.state_dict(),
            "update_count": self._update_count,
            "swa_active": self._swa_active,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.swa_model.load_state_dict(state_dict["swa_model"])
        self.swa_scheduler.load_state_dict(state_dict["swa_scheduler"])
        self._update_count = state_dict["update_count"]
        self._swa_active = state_dict["swa_active"]


# =============================================================================
# Gradient Accumulation
# =============================================================================


class GradientAccumulator:
    """Gradient accumulation for effective larger batch sizes.

    Accumulates gradients over multiple micro-batches before
    performing an optimizer step.

    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=4)
        >>> for batch in train_loader:
        ...     loss = compute_loss(batch) / accumulator.accumulation_steps
        ...     loss.backward()
        ...     if accumulator.step():
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """

    def __init__(self, accumulation_steps: int = 1):
        """Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps before optimizer update
        """
        self.accumulation_steps = accumulation_steps
        self._current_step = 0

    def step(self) -> bool:
        """Increment step counter.

        Returns:
            True if optimizer should step
        """
        self._current_step += 1

        if self._current_step >= self.accumulation_steps:
            self._current_step = 0
            return True

        return False

    def should_step(self) -> bool:
        """Check if next step() will trigger optimizer update."""
        return self._current_step == self.accumulation_steps - 1

    def reset(self) -> None:
        """Reset step counter."""
        self._current_step = 0

    def get_loss_scale(self) -> float:
        """Get loss scaling factor for accumulation."""
        return 1.0 / self.accumulation_steps


# =============================================================================
# Dynamic Batching
# =============================================================================


class DynamicBatchSampler:
    """Dynamic batch sampler based on sequence length.

    Groups sequences of similar length to minimize padding waste
    and maximize GPU utilization.

    Example:
        >>> sampler = DynamicBatchSampler(dataset, max_tokens=8192)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        max_tokens: int,
        length_fn: Optional[Callable[[Any], int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Initialize dynamic batch sampler.

        Args:
            dataset: Dataset to sample from
            max_tokens: Maximum tokens per batch
            length_fn: Function to get sequence length from sample
            shuffle: Whether to shuffle batches
            drop_last: Drop incomplete final batch
        """
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Default length function
        if length_fn is None:
            self.length_fn = lambda x: len(x) if hasattr(x, "__len__") else 1
        else:
            self.length_fn = length_fn

        # Pre-compute lengths and sort indices
        self._compute_batches()

    def _compute_batches(self) -> None:
        """Compute batches based on sequence lengths."""
        # Get lengths for all samples
        lengths = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if isinstance(sample, dict):
                # Try common keys
                for key in ["input", "sequence", "x", "tokens"]:
                    if key in sample:
                        lengths.append(self.length_fn(sample[key]))
                        break
                else:
                    lengths.append(1)
            elif isinstance(sample, (list, tuple)):
                lengths.append(self.length_fn(sample[0]))
            else:
                lengths.append(self.length_fn(sample))

        # Sort by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Create batches
        self.batches = []
        current_batch = []
        current_max_len = 0

        for idx in sorted_indices:
            sample_len = lengths[idx]

            # Check if adding this sample exceeds budget
            new_max_len = max(current_max_len, sample_len)
            new_batch_tokens = new_max_len * (len(current_batch) + 1)

            if new_batch_tokens > self.max_tokens and current_batch:
                self.batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        # Handle last batch
        if current_batch:
            if not self.drop_last or len(current_batch) > 0:
                self.batches.append(current_batch)

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches.

        Yields:
            List of indices for each batch
        """
        if self.shuffle:
            import random
            random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.batches)


# =============================================================================
# Training Utilities
# =============================================================================


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Estimate memory usage for training.

    Args:
        model: Model to estimate
        batch_size: Training batch size
        seq_length: Sequence length
        dtype: Data type

    Returns:
        Dict with memory estimates in GB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as parameters)
    grad_memory = param_memory

    # Optimizer state (assume Adam: 2x parameters for momentum + variance)
    optimizer_memory = 2 * param_memory

    # Activation memory (rough estimate: 2x forward pass)
    bytes_per_element = 4 if dtype == torch.float32 else 2
    activation_memory = 2 * batch_size * seq_length * 1024 * bytes_per_element

    # Total
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        "parameters_gb": param_memory / 1e9,
        "gradients_gb": grad_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_gb": total_memory / 1e9,
    }


def find_optimal_batch_size(
    model: nn.Module,
    sample_input: Tensor,
    max_batch_size: int = 512,
    target_utilization: float = 0.8,
    device: str = "cuda",
) -> int:
    """Find optimal batch size for GPU memory.

    Args:
        model: Model to test
        sample_input: Sample input tensor (without batch dim)
        max_batch_size: Maximum batch size to try
        target_utilization: Target GPU memory utilization
        device: Device for testing

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return max_batch_size

    model = model.to(device)
    model.train()

    optimal_size = 1

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if batch_size > max_batch_size:
            break

        try:
            torch.cuda.empty_cache()

            # Create batch
            batch = sample_input.unsqueeze(0).repeat(batch_size, *[1] * (sample_input.dim()))
            batch = batch.to(device)

            # Forward + backward
            output = model(batch)
            if isinstance(output, dict):
                loss = sum(v.sum() for v in output.values() if isinstance(v, Tensor))
            else:
                loss = output.sum()
            loss.backward()

            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

            if memory_used < target_utilization:
                optimal_size = batch_size
            else:
                break

        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            raise

        finally:
            torch.cuda.empty_cache()

    return optimal_size


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric and stops training when it stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current metric value

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
