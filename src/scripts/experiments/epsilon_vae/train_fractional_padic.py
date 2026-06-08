# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train with fractional p-adic interpolation.

Starting from the optimal p=3 baseline (v5_11_progressive with 100% coverage),
this script implements continuous interpolation toward p=6 (full closure).

The key innovation: Instead of discrete jumps between integer p values,
we use fractal dimensions (3.01, 3.05, 3.1, ...) to smoothly transition
the algebraic structure while monitoring coverage.

Mathematical Foundation:
- p=3: 19,683 operations (current)
- p=6: 10,077,696 operations (target - closure over 2 AND 3)
- Fractional p: Interpolated operation space + scaled architecture

Usage:
    python src/scripts/epsilon_vae/train_fractional_padic.py --p_start 3.0 --p_end 3.5 --steps 10
    python src/scripts/epsilon_vae/train_fractional_padic.py --p_start 3.0 --p_end 6.0 --steps 30 --full_run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.data.generation import generate_all_ternary_operations
from src.models.fractional_padic_architecture import (
    compute_padic_dimensions,
    create_interpolation_schedule,
    interpolate_operation_space,
)
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class FractionalPadicVAE(nn.Module):
    """VAE that handles fractional p-adic bases.

    This model can represent operations for any p in [3, 6] by:
    1. Scaling input layer to handle p^2 input dimensions
    2. Interpolating the operation space
    3. Maintaining algebraic completeness through constrained training
    """

    def __init__(
        self,
        p: float = 3.0,
        latent_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.p = p
        self.latent_dim = latent_dim
        self.dims = compute_padic_dimensions(p)

        # Base input dimension (for p=3)
        self.base_input_dim = 9

        # Capacity scaling factor
        self.capacity_scale = self.dims["required_bits"] / 14.3

        # Encoder A (for truth table)
        self.encoder_A = self._build_encoder(hidden_dim)

        # Encoder B (for operation embedding) - same architecture
        self.encoder_B = self._build_encoder(hidden_dim)

        # Decoder
        self.decoder = self._build_decoder(hidden_dim)

        # Projection to hyperbolic space
        self.proj_A = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.proj_B = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def _build_encoder(self, hidden_dim: int) -> nn.Module:
        """Build encoder with scaled capacity."""
        scaled_hidden = int(hidden_dim * self.capacity_scale)

        return nn.ModuleDict(
            {
                "input_proj": nn.Linear(self.dims["input_dim"], self.base_input_dim)
                if self.dims["input_dim"] != self.base_input_dim
                else nn.Identity(),
                "layers": nn.Sequential(
                    nn.Linear(self.base_input_dim, scaled_hidden),
                    nn.LayerNorm(scaled_hidden),
                    nn.GELU(),
                    nn.Linear(scaled_hidden, scaled_hidden // 2),
                    nn.LayerNorm(scaled_hidden // 2),
                    nn.GELU(),
                    nn.Linear(scaled_hidden // 2, scaled_hidden // 4),
                    nn.LayerNorm(scaled_hidden // 4),
                    nn.GELU(),
                ),
                "fc_mu": nn.Linear(scaled_hidden // 4, self.latent_dim),
                "fc_logvar": nn.Linear(scaled_hidden // 4, self.latent_dim),
            }
        )

    def _build_decoder(self, hidden_dim: int) -> nn.Module:
        """Build decoder."""
        scaled_hidden = int(hidden_dim * self.capacity_scale)

        return nn.Sequential(
            nn.Linear(self.latent_dim * 2, scaled_hidden // 2),
            nn.LayerNorm(scaled_hidden // 2),
            nn.GELU(),
            nn.Linear(scaled_hidden // 2, scaled_hidden),
            nn.LayerNorm(scaled_hidden),
            nn.GELU(),
            nn.Linear(scaled_hidden, self.base_input_dim),
        )

    def encode(self, x: torch.Tensor, encoder: nn.ModuleDict):
        """Encode input to latent distribution."""
        x = encoder["input_proj"](x)
        h = encoder["layers"](x)
        mu = encoder["fc_mu"](h)
        logvar = encoder["fc_logvar"](h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def project_to_hyperbolic(self, z: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Project to hyperbolic space (Poincare ball)."""
        z_proj = proj(z)
        # Clamp to Poincare ball
        norm = z_proj.norm(dim=-1, keepdim=True)
        max_norm = 0.99
        z_proj = torch.where(norm > max_norm, z_proj * max_norm / norm, z_proj)
        return z_proj

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # Pad input if needed for higher p
        if x.shape[-1] < self.dims["input_dim"]:
            padding = torch.zeros(*x.shape[:-1], self.dims["input_dim"] - x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)

        # Encode
        mu_A, logvar_A = self.encode(x, self.encoder_A)
        mu_B, logvar_B = self.encode(x, self.encoder_B)

        # Sample
        z_A_euc = self.reparameterize(mu_A, logvar_A)
        z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Project to hyperbolic
        z_A_hyp = self.project_to_hyperbolic(z_A_euc, self.proj_A)
        z_B_hyp = self.project_to_hyperbolic(z_B_euc, self.proj_B)

        # Decode
        z_combined = torch.cat([z_A_euc, z_B_euc], dim=-1)
        x_recon = self.decoder(z_combined)

        # Losses
        recon_loss = nn.functional.mse_loss(x_recon, x[..., : self.base_input_dim])

        kl_loss = -0.5 * torch.mean(
            1 + logvar_A - mu_A.pow(2) - logvar_A.exp() + 1 + logvar_B - mu_B.pow(2) - logvar_B.exp()
        )

        return {
            "z_A_euc": z_A_euc,
            "z_B_euc": z_B_euc,
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
            "x_recon": x_recon,
            "mu_A": mu_A,
            "logvar_A": logvar_A,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "loss": recon_loss + 0.01 * kl_loss,
        }


def compute_coverage_for_p(model, p: float, device: str = "cpu") -> dict:
    """Compute coverage metrics for a given p value.

    Args:
        model: FractionalPadicVAE model
        p: The p value
        device: Device

    Returns:
        Coverage metrics
    """
    model.eval()

    # Get base operations
    base_ops = generate_all_ternary_operations()

    # Interpolate for fractional p
    ops = interpolate_operation_space(p, base_ops) if p > 3.0 else base_ops

    ops_tensor = torch.tensor(ops, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(ops_tensor)
        z_hyp = outputs["z_A_hyp"]

        # Coverage metrics
        norms = z_hyp.norm(dim=-1)
        well_represented = (norms > 0.1) & (norms < 0.99)
        coverage = well_represented.float().mean().item()

        # Uniqueness (check for collisions)
        dists = torch.cdist(z_hyp, z_hyp)
        mask = torch.eye(len(z_hyp), device=device).bool()
        dists.masked_fill_(mask, float("inf"))
        min_dists = dists.min(dim=1)[0]

        collisions = (min_dists < 0.01).sum().item()

    return {
        "p": p,
        "n_operations": len(ops),
        "coverage": coverage,
        "collisions": collisions,
        "collision_rate": collisions / len(ops),
        "mean_norm": norms.mean().item(),
        "mean_min_dist": min_dists.mean().item(),
    }


def transfer_weights(
    source_model,
    target_model,
    p_source: float,
    p_target: float,
):
    """Transfer weights from source to target model.

    For fractional p interpolation, we need to handle:
    1. Input layer scaling (source p^2 -> target p^2)
    2. Hidden layer capacity scaling
    3. Preserving learned representations

    Args:
        source_model: Model trained at p_source
        target_model: New model for p_target
        p_source: Source p value
        p_target: Target p value
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    transferred = 0
    resized = 0

    for key in target_state:
        if key in source_state:
            src_shape = source_state[key].shape
            tgt_shape = target_state[key].shape

            if src_shape == tgt_shape:
                # Direct copy
                target_state[key] = source_state[key].clone()
                transferred += 1
            elif len(src_shape) == len(tgt_shape):
                # Need to resize - use interpolation or padding
                if all(s <= t for s, t in zip(src_shape, tgt_shape, strict=False)):
                    # Target is larger - pad with zeros (preserve source structure)
                    src_data = source_state[key]
                    slices = tuple(slice(0, s) for s in src_shape)
                    target_state[key][slices] = src_data
                    resized += 1
                else:
                    # Target is smaller - truncate
                    slices = tuple(slice(0, t) for t in tgt_shape)
                    target_state[key] = source_state[key][slices].clone()
                    resized += 1

    target_model.load_state_dict(target_state)

    return {"transferred": transferred, "resized": resized}


def train_at_p(
    model,
    p: float,
    device: str = "cpu",
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
) -> dict:
    """Train model at a specific p value.

    Args:
        model: FractionalPadicVAE model
        p: P value to train at
        device: Device
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size

    Returns:
        Training history
    """
    model.to(device)
    model.train()

    # Get operations for this p
    base_ops = generate_all_ternary_operations()
    ops = interpolate_operation_space(p, base_ops) if p > 3.0 else base_ops

    ops_tensor = torch.tensor(ops, dtype=torch.float32)
    dataset = TensorDataset(ops_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": [], "coverage": []}

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for (batch,) in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        # Evaluate coverage periodically
        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = compute_coverage_for_p(model, p, device)
            history["coverage"].append(metrics["coverage"])
            print(
                f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"Coverage: {metrics['coverage']:.4f} | "
                f"Collisions: {metrics['collisions']}"
            )
            model.train()

    return history


def load_baseline_model(checkpoint_path: Path, device: str = "cpu"):
    """Load the v5_11_progressive baseline model.

    This model achieves 100% coverage at p=3 and serves as the starting point
    for fractional p interpolation.
    """
    from src.models.ternary_vae import TernaryVAEV5_11

    ckpt = load_checkpoint_compat(checkpoint_path, map_location=device)
    model_state = get_model_state_dict(ckpt)

    # Infer architecture
    latent_dim = 16
    for key, value in model_state.items():
        if "encoder_A.fc_mu.weight" in key:
            latent_dim = value.shape[0]
            break

    model = TernaryVAEV5_11(
        latent_dim=latent_dim,
        hidden_dim=64,
        use_dual_projection=True,
        use_controller=False,
    )

    model.load_state_dict(model_state, strict=False)
    model.to(device)

    return model


def run_interpolation(
    p_values: list,
    device: str = "cpu",
    epochs_per_p: int = 50,
    output_dir: Path = None,
    baseline_checkpoint: Path = None,
):
    """Run full p-adic interpolation from p_start to p_end.

    Args:
        p_values: List of p values to train at
        device: Device
        epochs_per_p: Epochs at each p
        output_dir: Output directory
        baseline_checkpoint: Path to v5_11_progressive checkpoint

    Returns:
        Interpolation results
    """
    output_dir = output_dir or OUTPUT_DIR / "fractional_padic"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "p_values": p_values,
        "coverage_trajectory": [],
        "collision_trajectory": [],
        "timestamps": [],
    }

    # Load baseline model if this is starting at p=3
    baseline_path = baseline_checkpoint or CHECKPOINTS_DIR / "v5_11_progressive" / "best.pt"
    if p_values[0] <= 3.0 and baseline_path.exists():
        print(f"\nLoading baseline from {baseline_path}")
        baseline_model = load_baseline_model(baseline_path, device)

        # Evaluate baseline
        baseline_model.eval()
        base_ops = generate_all_ternary_operations()
        base_ops_tensor = torch.tensor(base_ops, dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = baseline_model(base_ops_tensor, compute_control=False)
            z_hyp = outputs["z_A_hyp"]
            norms = z_hyp.norm(dim=-1)
            well_represented = (norms > 0.1) & (norms < 0.99)
            baseline_coverage = well_represented.float().mean().item()

        print(f"Baseline coverage: {baseline_coverage:.4f}")

        # Use baseline for p=3
        if p_values[0] == 3.0:
            results["coverage_trajectory"].append(baseline_coverage)
            results["collision_trajectory"].append(0)
            results["timestamps"].append(datetime.now().isoformat())
            p_values = p_values[1:]  # Skip p=3, already have it

            # Save baseline as p=3.0 checkpoint
            torch.save(
                {
                    "p": 3.0,
                    "model_state_dict": baseline_model.state_dict(),
                    "coverage": baseline_coverage,
                },
                output_dir / "checkpoint_p3.00.pt",
            )

    model = None

    for i, p in enumerate(p_values):
        print(f"\n{'=' * 70}")
        print(f"TRAINING AT p = {p:.4f} ({i + 1}/{len(p_values)})")
        print(f"{'=' * 70}")

        dims = compute_padic_dimensions(p)
        print(f"  Operations: {dims['n_operations_int']:,}")
        print(f"  Input dim: {dims['input_dim']}")
        print(f"  Required bits: {dims['required_bits']:.1f}")

        # Create new model for this p
        new_model = FractionalPadicVAE(p=p, latent_dim=16, hidden_dim=256)

        # Transfer weights from previous model
        if model is not None:
            prev_p = p_values[i - 1]
            transfer_info = transfer_weights(model, new_model, prev_p, p)
            print(f"  Transferred {transfer_info['transferred']} weights, resized {transfer_info['resized']}")

        model = new_model

        # Train
        history = train_at_p(
            model=model,
            p=p,
            device=device,
            epochs=epochs_per_p,
        )

        # Final evaluation
        metrics = compute_coverage_for_p(model, p, device)
        results["coverage_trajectory"].append(metrics["coverage"])
        results["collision_trajectory"].append(metrics["collisions"])
        results["timestamps"].append(datetime.now().isoformat())

        print(f"\n  Final coverage at p={p:.4f}: {metrics['coverage']:.4f}")
        print(f"  Collisions: {metrics['collisions']}")

        # Save checkpoint
        torch.save(
            {
                "p": p,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "history": history,
            },
            output_dir / f"checkpoint_p{p:.2f}.pt",
        )

        # Check if we should stop (coverage dropped too much)
        if metrics["coverage"] < 0.8:
            print("\n  WARNING: Coverage dropped below 80%, stopping interpolation")
            break

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(
        results["p_values"][: len(results["coverage_trajectory"])],
        results["coverage_trajectory"],
        "b-o",
        linewidth=2,
        markersize=8,
    )
    ax1.axhline(y=1.0, color="g", linestyle="--", label="Perfect coverage")
    ax1.axhline(y=0.8, color="r", linestyle="--", label="Minimum threshold")
    ax1.set_xlabel("p value")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage vs p-adic Base")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(
        results["p_values"][: len(results["collision_trajectory"])],
        results["collision_trajectory"],
        "r-o",
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("p value")
    ax2.set_ylabel("Collisions")
    ax2.set_title("Latent Space Collisions vs p")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "interpolation_results.png", dpi=150)
    plt.close()

    # Save results
    with open(output_dir / "interpolation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("INTERPOLATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train with fractional p-adic interpolation")
    parser.add_argument("--p_start", type=float, default=3.0)
    parser.add_argument("--p_end", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=6, help="Number of interpolation steps")
    parser.add_argument("--epochs_per_p", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "fractional_padic"))
    parser.add_argument("--full_run", action="store_true", help="Run full interpolation to p=6")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)

    # Create interpolation schedule
    if args.full_run:
        p_values = create_interpolation_schedule(3.0, 6.0, schedule_type="milestone")
    else:
        p_values = list(np.linspace(args.p_start, args.p_end, args.steps))

    print(f"\n{'=' * 70}")
    print("FRACTIONAL P-ADIC INTERPOLATION")
    print(f"{'=' * 70}")
    print(f"P values: {[f'{p:.2f}' for p in p_values]}")
    print(f"Device: {device}")
    print(f"Epochs per p: {args.epochs_per_p}")

    # Run interpolation
    results = run_interpolation(
        p_values=p_values,
        device=device,
        epochs_per_p=args.epochs_per_p,
        output_dir=output_dir,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for i, p in enumerate(p_values[: len(results["coverage_trajectory"])]):
        cov = results["coverage_trajectory"][i]
        col = results["collision_trajectory"][i]
        status = "OK" if cov >= 0.95 else "DEGRADED" if cov >= 0.8 else "FAILED"
        print(f"  p={p:.2f}: coverage={cov:.4f}, collisions={col}, status={status}")


if __name__ == "__main__":
    main()
