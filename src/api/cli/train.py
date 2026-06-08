# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training commands for Ternary VAE CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config.paths import RESULTS_DIR

app = typer.Typer(help="Training commands for Ternary VAE models")
console = Console()


@app.command("run")
def train_run(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        "-b",
        help="Training batch size",
    ),
    learning_rate: float = typer.Option(
        1e-3,
        "--lr",
        help="Learning rate",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Device to train on (cuda/cpu)",
    ),
    save_dir: Path = typer.Option(
        RESULTS_DIR / "training",
        "--save-dir",
        "-o",
        help="Directory to save checkpoints",
    ),
    partial_freeze: bool = typer.Option(
        True,
        "--partial-freeze/--no-partial-freeze",
        help="Use partial freeze architecture (frozen encoder_A)",
    ),
    curvature: float = typer.Option(
        1.0,
        "--curvature",
        help="Hyperbolic curvature parameter",
    ),
    max_radius: float = typer.Option(
        0.95,
        "--max-radius",
        help="Maximum Poincare ball radius",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Train a Ternary VAE model.

    Example:
        ternary-vae train run --config src/configs/ternary.yaml
        ternary-vae train run --epochs 200 --lr 5e-4
    """
    import torch

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"

    console.print("[bold blue]Starting Ternary VAE Training[/bold blue]")
    console.print(f"  Device: {device}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Architecture: {'PartialFreeze' if partial_freeze else 'Standard'}")

    # Load config if provided
    training_config = {}
    if config and config.exists():
        import yaml

        with open(config) as f:
            training_config = yaml.safe_load(f)
        console.print(f"[green]Loaded config from {config}[/green]")

    # Import training components
    from src import TrainingConfig
    from src.data import generate_all_ternary_operations
    from src.models import TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze
    from src.training import TernaryVAETrainer

    # Generate data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating ternary operations...", total=None)
        x, indices = generate_all_ternary_operations()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        progress.update(task, completed=True)

    console.print(f"[green]Generated {len(x_tensor)} ternary operations[/green]")

    # Create model
    ModelClass = TernaryVAEV5_11_PartialFreeze if partial_freeze else TernaryVAEV5_11
    model = ModelClass(
        latent_dim=training_config.get("model", {}).get("latent_dim", 16),
        hidden_dim=training_config.get("model", {}).get("hidden_dim", 64),
        curvature=curvature,
        max_radius=max_radius,
    )
    model = model.to(device)

    console.print(f"[green]Model created with {sum(p.numel() for p in model.parameters()):,} parameters[/green]")

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build training config
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # Create trainer and train
    trainer = TernaryVAETrainer(model, train_cfg, device=device)

    console.print("[bold]Starting training...[/bold]")
    trainer.train(x_tensor)

    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": training_config,
            "epochs": epochs,
        },
        final_path,
    )

    console.print(f"[bold green]Training complete! Model saved to {final_path}[/bold green]")


@app.command("hiv")
def train_hiv(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    save_dir: Path = typer.Option(
        RESULTS_DIR / "hiv_training",
        "--save-dir",
        "-o",
        help="Directory to save results",
    ),
):
    """Train HIV-specific codon VAE model.

    Example:
        ternary-vae train hiv --epochs 200
    """
    console.print("[bold blue]HIV Codon VAE Training[/bold blue]")
    console.print("This command wraps src/scripts/train_codon_vae_hiv.py")

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "src/scripts/train_codon_vae_hiv.py",
        "--epochs",
        str(epochs),
        "--save_dir",
        str(save_dir),
    ]
    if config:
        cmd.extend(["--config", str(config)])

    subprocess.run(cmd)


@app.command("resume")
def train_resume(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint to resume from",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Additional epochs to train",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        "-b",
        help="Training batch size",
    ),
    learning_rate: float | None = typer.Option(
        None,
        "--lr",
        help="Learning rate (default: use checkpoint value)",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Device to train on (cuda/cpu)",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        "-o",
        help="Directory to save checkpoints (default: same as checkpoint)",
    ),
):
    """Resume training from a checkpoint.

    Example:
        ternary-vae train resume results/training/best.pt --epochs 50
        ternary-vae train resume checkpoint.pt --epochs 100 --lr 5e-4
    """
    import torch
    from scipy.stats import spearmanr

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"

    console.print(f"[bold blue]Resuming training from {checkpoint}[/bold blue]")

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    start_epoch = ckpt.get("epoch", 0)
    saved_config = ckpt.get("config", {})

    console.print(f"[green]Loaded checkpoint from epoch {start_epoch}[/green]")

    # Display saved metrics if available
    if "metrics" in ckpt:
        metrics = ckpt["metrics"]
        console.print(f"  Coverage: {metrics.get('coverage', 0) * 100:.1f}%")
        console.print(f"  Radial Corr A: {metrics.get('radial_corr_A', 0):.4f}")

    # Import training components
    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations
    from src.geometry import poincare_distance
    from src.losses import PAdicGeodesicLoss, RadialHierarchyLoss
    from src.models import TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze

    # Determine model type from config or state dict
    use_partial_freeze = saved_config.get("option_c", True)
    use_dual_projection = saved_config.get("dual_projection", False)

    # Create model
    ModelClass = TernaryVAEV5_11_PartialFreeze if use_partial_freeze else TernaryVAEV5_11
    model = ModelClass(
        latent_dim=saved_config.get("latent_dim", 16),
        hidden_dim=saved_config.get("hidden_dim", 64),
        curvature=saved_config.get("curvature", 1.0),
        max_radius=saved_config.get("max_radius", 0.95),
        use_dual_projection=use_dual_projection,
    )

    # Load model state
    model_state = ckpt.get("model_state_dict", ckpt.get("model_state", {}))
    if model_state:
        model.load_state_dict(model_state, strict=False)
        console.print("[green]Model state loaded successfully[/green]")
    else:
        console.print("[yellow]Warning: No model state found in checkpoint[/yellow]")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Model ready with {param_count:,} trainable parameters[/green]")

    # Generate data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating ternary operations...", total=None)
        operations = generate_all_ternary_operations()
        x = torch.tensor(operations, dtype=torch.float32, device=device)
        indices = torch.arange(len(operations), device=device)
        progress.update(task, completed=True)

    console.print(f"[green]Dataset ready: {len(x)} operations[/green]")

    # Create loss functions
    geodesic_loss_fn = PAdicGeodesicLoss(
        curvature=saved_config.get("curvature", 1.0),
        n_pairs=saved_config.get("n_pairs", 2000),
    ).to(device)
    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=saved_config.get("inner_radius", 0.1),
        outer_radius=saved_config.get("outer_radius", 0.85),
    ).to(device)

    # Create optimizer
    lr = learning_rate or saved_config.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        weight_decay=saved_config.get("weight_decay", 1e-4),
    )

    # Load optimizer state if available
    if "optimizer_state_dict" in ckpt or "optimizer_state" in ckpt:
        opt_state = ckpt.get("optimizer_state_dict", ckpt.get("optimizer_state"))
        try:
            optimizer.load_state_dict(opt_state)
            console.print("[green]Optimizer state restored[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not restore optimizer state: {e}[/yellow]")

    # Setup save directory
    if save_dir is None:
        save_dir = checkpoint.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    console.print(f"\n[bold]Resuming training for {epochs} additional epochs...[/bold]")
    end_epoch = start_epoch + epochs
    n_batches = (len(x) + batch_size - 1) // batch_size
    radial_weight = saved_config.get("radial_weight", 2.0)

    best_radial_corr = float("inf")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = 0.0

        # Shuffle data
        perm = torch.randperm(len(x), device=device)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x))
            batch_idx = perm[start_idx:end_idx]

            x_batch = x[batch_idx]
            idx_batch = indices[batch_idx]

            optimizer.zero_grad()

            outputs = model(x_batch, compute_control=False)
            z_A = outputs["z_A_hyp"]
            z_B = outputs["z_B_hyp"]

            geo_loss_A, _ = geodesic_loss_fn(z_A, idx_batch)
            geo_loss_B, _ = geodesic_loss_fn(z_B, idx_batch)
            rad_loss_A, _ = radial_loss_fn(z_A, idx_batch)
            rad_loss_B, _ = radial_loss_fn(z_B, idx_batch)

            tau = min(1.0, epoch / (end_epoch * 0.7))
            loss = tau * (geo_loss_A + geo_loss_B) + radial_weight * (rad_loss_A + rad_loss_B)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(x, compute_control=False)
            z_A = outputs["z_A_hyp"]

            # Coverage
            mu_A = outputs["mu_A"]
            logits = model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == x.long()).float().mean(dim=1)
            coverage = (correct == 1.0).sum().item() / len(x)

            # V5.12.2: Radial correlation using hyperbolic distance
            origin = torch.zeros_like(z_A)
            radii = poincare_distance(z_A, origin, c=1.0).cpu().numpy()
            valuations = TERNARY.valuation(indices).cpu().numpy()
            radial_corr = spearmanr(valuations, radii)[0]

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            console.print(
                f"  Epoch {epoch + 1}/{end_epoch}: loss={avg_loss:.4f}, "
                f"coverage={coverage * 100:.1f}%, radial_corr={radial_corr:.4f}"
            )

        # Save best model
        if radial_corr < best_radial_corr:
            best_radial_corr = radial_corr
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {
                        "coverage": coverage,
                        "radial_corr_A": radial_corr,
                    },
                    "config": saved_config,
                },
                save_dir / "best.pt",
            )

    # Save final checkpoint
    final_path = save_dir / "latest.pt"
    torch.save(
        {
            "epoch": end_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "coverage": coverage,
                "radial_corr_A": radial_corr,
            },
            "config": saved_config,
        },
        final_path,
    )

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Final coverage: {coverage * 100:.1f}%")
    console.print(f"  Final radial correlation: {radial_corr:.4f}")
    console.print(f"  Best radial correlation: {best_radial_corr:.4f}")
    console.print(f"  Checkpoints saved to: {save_dir}")


@app.callback()
def callback():
    """Training commands for Ternary VAE models."""
    pass
