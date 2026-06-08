# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Parameter schedulers for training.

This module provides scheduling functionality for training parameters:
- Temperature scheduling (linear and cyclic)
- Beta (KL weight) scheduling with warmup
- Learning rate scheduling

P3 FIX: Replaced numpy with math module to avoid CPU-GPU sync issues.

Single responsibility: Parameter scheduling logic only.
"""

import math
from typing import Any


def linear_schedule(
    epoch: int,
    start_val: float,
    end_val: float,
    total_epochs: int,
    start_epoch: int = 0,
) -> float:
    """Linear scheduling from start_val to end_val.

    Args:
        epoch: Current epoch
        start_val: Starting value
        end_val: Ending value
        total_epochs: Total epochs for schedule
        start_epoch: Epoch to start schedule (default: 0)

    Returns:
        Scheduled value
    """
    if epoch < start_epoch:
        return start_val
    progress = min((epoch - start_epoch) / total_epochs, 1.0)
    return start_val + (end_val - start_val) * progress


def cyclic_schedule(epoch: int, base_val: float, amplitude: float, period: int) -> float:
    """Cyclic scheduling: base ± amplitude with given period.

    P3 FIX: Uses math.cos instead of numpy.cos to avoid potential
    CPU-GPU sync issues when used with torch tensors.

    Args:
        epoch: Current epoch
        base_val: Base value
        amplitude: Amplitude of oscillation
        period: Period of cycle in epochs

    Returns:
        Scheduled value
    """
    phase = (epoch % period) / period * 2 * math.pi
    return base_val + amplitude * math.cos(phase)


class TemperatureScheduler:
    """Temperature scheduling for VAEs.

    Supports:
    - Linear annealing
    - Cyclic modulation
    - Phase-dependent boosting
    """

    def __init__(self, config: dict[str, Any], phase_4_start: int, temp_lag: int = 0):
        """Initialize temperature scheduler.

        Args:
            config: Configuration dict with vae_a and vae_b sections
            phase_4_start: Epoch when phase 4 starts
            temp_lag: Lag for VAE-B temperature (default: 0)
        """
        self.config = config
        self.phase_4_start = phase_4_start
        self.temp_lag = temp_lag

    def get_temperature(self, epoch: int, vae: str = "A") -> float:
        """Get temperature with proper Phase 4 support.

        Args:
            epoch: Current epoch
            vae: Which VAE ('A' or 'B')

        Returns:
            Temperature value
        """
        if vae == "A":
            # Chaotic regime: cyclic with boost in Phase 4
            base_temp = linear_schedule(
                epoch,
                self.config["vae_a"]["temp_start"],
                self.config["vae_a"]["temp_end"],
                self.config["total_epochs"],
            )

            if self.config["vae_a"].get("temp_cyclic", False):
                # Phase 1-3: Small cyclic modulation
                amplitude = 0.1 * base_temp

                # Phase 4: Enhanced exploration with temp_boost_amplitude
                if epoch >= self.phase_4_start and "temp_boost_amplitude" in self.config["vae_a"]:
                    amplitude = self.config["vae_a"]["temp_boost_amplitude"]

                period = 30
                return max(0.1, cyclic_schedule(epoch, base_temp, amplitude, period))

            return base_temp
        else:
            # Frozen regime: monotonic with Phase 4 boost
            epoch_lagged = max(0, epoch - self.temp_lag)

            # Phase 1-3: Normal annealing
            if epoch < self.phase_4_start:
                return linear_schedule(
                    epoch_lagged,
                    self.config["vae_b"]["temp_start"],
                    self.config["vae_b"]["temp_end"],
                    self.config["total_epochs"],
                )
            else:
                # Phase 4: Use temp_phase4 if specified
                if "temp_phase4" in self.config["vae_b"]:
                    return self.config["vae_b"]["temp_phase4"]
                else:
                    return self.config["vae_b"]["temp_end"]


class BetaScheduler:
    """Beta (KL weight) scheduling with warmup.

    Implements β-VAE warmup to prevent posterior collapse:
    - Warmup phase: β increases from 0 to target over warmup_epochs
    - After warmup: β follows configured schedule
    """

    def __init__(self, config: dict[str, Any], beta_phase_lag: float = 0.0):
        """Initialize beta scheduler.

        Args:
            config: Configuration dict with vae_a and vae_b sections
            beta_phase_lag: Phase lag for VAE-B beta (default: 0.0)
        """
        self.config = config
        self.beta_phase_lag = beta_phase_lag

    def get_beta(self, epoch: int, vae: str = "A") -> float:
        """Get beta with KL warmup and phase offset for VAE-B.

        Args:
            epoch: Current epoch
            vae: Which VAE ('A' or 'B')

        Returns:
            Beta value
        """
        if vae == "A":
            # Get warmup parameters
            warmup_epochs = self.config["vae_a"].get("beta_warmup_epochs", 0)

            if warmup_epochs > 0 and epoch < warmup_epochs:
                # Warmup: linearly increase from 0 to beta_start
                beta_target = self.config["vae_a"]["beta_start"]
                return (epoch / warmup_epochs) * beta_target
            else:
                # Normal schedule after warmup
                return linear_schedule(
                    epoch - warmup_epochs,
                    self.config["vae_a"]["beta_start"],
                    self.config["vae_a"]["beta_end"],
                    self.config["total_epochs"] - warmup_epochs,
                )
        else:
            # VAE-B warmup
            warmup_epochs = self.config["vae_b"].get("beta_warmup_epochs", 0)

            if warmup_epochs > 0 and epoch < warmup_epochs:
                beta_target = self.config["vae_b"]["beta_start"]
                return (epoch / warmup_epochs) * beta_target
            else:
                # After warmup, use phase offset from VAE-A
                beta_A = self.get_beta(epoch, "A")
                return beta_A * abs(math.sin(self.beta_phase_lag))


class LearningRateScheduler:
    """Learning rate scheduling from config.

    Supports epoch-based step scheduling from configuration.
    """

    def __init__(self, lr_schedule: list[dict[str, Any]]):
        """Initialize learning rate scheduler.

        Args:
            lr_schedule: List of {'epoch': int, 'lr': float} dicts
        """
        self.lr_schedule = lr_schedule

    def get_lr(self, epoch: int) -> float:
        """Get learning rate from schedule.

        Args:
            epoch: Current epoch

        Returns:
            Learning rate
        """
        lr = self.lr_schedule[0]["lr"]
        for entry in self.lr_schedule:
            if epoch >= entry["epoch"]:
                lr = entry["lr"]
        return lr
