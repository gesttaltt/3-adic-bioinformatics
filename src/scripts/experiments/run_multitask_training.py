"""Multi-Task Training for Drug Resistance Prediction.

Train a single model on all drugs within a class simultaneously.
Benefits:
1. Shared representations capture common resistance patterns
2. Transfer learning from data-rich to data-poor drugs
3. GradNorm for balanced multi-task learning

Target: Improve generalization and cross-drug transfer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training."""

    input_dim: int = 99
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    n_tasks: int = 8  # Number of drugs
    batch_size: int = 64
    epochs: int = 200
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # GradNorm settings
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5  # Asymmetry parameter
    gradnorm_lr: float = 0.01

    # Loss settings
    use_ranking: bool = True
    ranking_weight: float = 0.3


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning."""

    def __init__(self, task_data: dict[str, tuple[np.ndarray, np.ndarray]]):
        """
        Args:
            task_data: Dict mapping task_name -> (X, y)
        """
        self.task_names = list(task_data.keys())
        self.task_to_idx = {name: i for i, name in enumerate(self.task_names)}

        # Combine all data with task labels
        all_X = []
        all_y = []
        all_tasks = []

        for task_name, (X, y) in task_data.items():
            task_idx = self.task_to_idx[task_name]
            all_X.append(X)
            all_y.append(y)
            all_tasks.extend([task_idx] * len(X))

        self.X = np.vstack(all_X)
        self.y = np.hstack(all_y)
        self.tasks = np.array(all_tasks)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.tasks[idx], dtype=torch.long),
        )


class MultiTaskVAE(nn.Module):
    """VAE with task-specific prediction heads."""

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg

        # Shared encoder
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Dropout(0.1),
                ]
            )
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Shared decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                ]
            )
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Task-specific prediction heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.latent_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for _ in range(cfg.n_tasks)
            ]
        )

        # Task embeddings (optional - for conditioning)
        self.task_embedding = nn.Embedding(cfg.n_tasks, 16)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Decode
        x_recon = self.decoder(z)

        # Task-specific predictions
        predictions = torch.zeros(x.size(0), device=x.device)
        for task_idx in range(self.cfg.n_tasks):
            mask = task_ids == task_idx
            if mask.any():
                predictions[mask] = self.task_heads[task_idx](z[mask]).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": predictions,
        }

    def predict_all_tasks(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Get predictions for all tasks."""
        h = self.encoder(x)
        mu = self.fc_mu(h)

        predictions = {}
        for task_idx in range(self.cfg.n_tasks):
            predictions[task_idx] = self.task_heads[task_idx](mu).squeeze(-1)

        return predictions


class GradNorm:
    """GradNorm for balanced multi-task learning.

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive
    Loss Balancing in Deep Multitask Networks", ICML 2018.
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5, lr: float = 0.01):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr

        # Learnable loss weights (initialize to 1)
        self.weights = nn.Parameter(torch.ones(n_tasks))

        # Initial losses (for relative scaling)
        self.initial_losses = None

    def update_weights(
        self,
        task_losses: list[torch.Tensor],
        shared_params: list[nn.Parameter],
    ):
        """Update task weights based on gradient magnitudes."""
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in task_losses]

        # Compute gradient norms for each task
        grad_norms = []
        for _i, loss in enumerate(task_losses):
            # Get gradients w.r.t. shared parameters
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            total_norm = 0
            for g in grads:
                if g is not None:
                    total_norm += g.norm().item() ** 2
            grad_norms.append(total_norm**0.5)

        grad_norms = torch.tensor(grad_norms)
        mean_norm = grad_norms.mean()

        # Compute inverse training rates
        loss_ratios = torch.tensor([l.item() / (self.initial_losses[i] + 1e-8) for i, l in enumerate(task_losses)])
        mean_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (mean_ratio + 1e-8)

        # Target gradient norms
        target_norms = mean_norm * (relative_rates**self.alpha)

        # Update weights
        with torch.no_grad():
            torch.abs(grad_norms - target_norms).sum()
            # Simple gradient step on weights
            weight_grads = 2 * (grad_norms - target_norms)
            self.weights.data = self.weights.data - self.lr * weight_grads

            # Renormalize weights
            self.weights.data = self.weights.data * self.n_tasks / self.weights.data.sum()
            self.weights.data = torch.clamp(self.weights.data, min=0.1, max=10.0)

        return self.weights.clone()


def load_drug_class_data(drug_class: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int, list[str]]:
    """Load all drugs for a class."""
    data_dir = project_root / "data" / "research"

    file_mapping = {
        "pi": "stanford_hivdb_pi.txt",
        "nrti": "stanford_hivdb_nrti.txt",
        "nnrti": "stanford_hivdb_nnrti.txt",
        "ini": "stanford_hivdb_ini.txt",
    }

    drug_columns = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],
    }

    filepath = data_dir / file_mapping[drug_class]
    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Get position columns
    if drug_class == "pi":
        prefix = "P"
    elif drug_class in ["nrti", "nnrti"]:
        prefix = "RT"
    else:
        prefix = "IN"

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    # Encoding params
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    n_aa = len(aa_alphabet)
    n_positions = len(position_cols)
    input_dim = n_positions * n_aa

    # Load each drug
    task_data = {}
    for drug in drug_columns[drug_class]:
        try:
            df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
            if len(df_valid) < 50:
                print(f"  Skipping {drug}: only {len(df_valid)} samples")
                continue

            # Encode
            X = np.zeros((len(df_valid), input_dim), dtype=np.float32)
            for idx, (_, row) in enumerate(df_valid.iterrows()):
                for j, col in enumerate(position_cols):
                    aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                    if aa in aa_to_idx:
                        X[idx, j * n_aa + aa_to_idx[aa]] = 1.0
                    else:
                        X[idx, j * n_aa + aa_to_idx["-"]] = 1.0

            # Resistance values
            y = np.log10(df_valid[drug].values + 1).astype(np.float32)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

            task_data[drug] = (X, y)
            print(f"  Loaded {drug}: {len(X)} samples")

        except Exception as e:
            print(f"  Error loading {drug}: {e}")

    return task_data, input_dim, list(task_data.keys())


def train_multitask(
    cfg: MultiTaskConfig,
    train_data: dict[str, tuple[np.ndarray, np.ndarray]],
    test_data: dict[str, tuple[np.ndarray, np.ndarray]],
    task_names: list[str],
) -> dict[str, float]:
    """Train multi-task model."""
    device = torch.device(cfg.device)

    # Create datasets
    train_dataset = MultiTaskDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create model
    model = MultiTaskVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # GradNorm
    gradnorm = GradNorm(cfg.n_tasks, cfg.gradnorm_alpha, cfg.gradnorm_lr) if cfg.use_gradnorm else None

    best_avg_corr = -1.0
    history = {"epoch": [], "avg_corr": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = [0.0] * cfg.n_tasks
        epoch_counts = [0] * cfg.n_tasks

        for x, y, task_ids in train_loader:
            x, y, task_ids = x.to(device), y.to(device), task_ids.to(device)

            optimizer.zero_grad()
            out = model(x, task_ids)

            # Compute per-task losses
            task_losses = []
            for task_idx in range(cfg.n_tasks):
                mask = task_ids == task_idx
                if mask.sum() > 0:
                    # Reconstruction
                    recon_loss = F.mse_loss(out["x_recon"][mask], x[mask])

                    # KL
                    kl = -0.5 * torch.mean(1 + out["logvar"][mask] - out["mu"][mask].pow(2) - out["logvar"][mask].exp())

                    # Prediction
                    pred_loss = F.mse_loss(out["prediction"][mask], y[mask])

                    # Ranking
                    if cfg.use_ranking and mask.sum() > 2:
                        p = out["prediction"][mask]
                        t = y[mask]
                        p_c = p - p.mean()
                        t_c = t - t.mean()
                        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
                        t_std = torch.sqrt(torch.sum(t_c**2) + 1e-8)
                        corr = torch.sum(p_c * t_c) / (p_std * t_std)
                        rank_loss = cfg.ranking_weight * (-corr)
                    else:
                        rank_loss = 0

                    task_loss = recon_loss + 0.001 * kl + pred_loss + rank_loss
                    task_losses.append(task_loss)
                    epoch_losses[task_idx] += task_loss.item()
                    epoch_counts[task_idx] += 1
                else:
                    task_losses.append(torch.tensor(0.0, device=device))

            # Combine losses
            active_losses = [l for l in task_losses if l.requires_grad]
            if gradnorm is not None and len(active_losses) > 0:
                # Get shared encoder parameters
                shared_params = list(model.encoder.parameters())
                # Only update if we have all tasks (avoid size mismatch)
                if len(active_losses) == gradnorm.n_tasks:
                    gradnorm.update_weights(active_losses, shared_params)
                weights = gradnorm.weights.to(device)
                # Use weights only for matching indices
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                weight_idx = 0
                for l in task_losses:
                    if l.requires_grad and weight_idx < len(weights):
                        total_loss = total_loss + weights[weight_idx] * l
                        weight_idx += 1
            else:
                total_loss = sum(l for l in task_losses if l.requires_grad)

            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        scheduler.step()

        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            task_corrs = {}

            with torch.no_grad():
                for task_name in task_names:
                    if task_name in test_data:
                        X_test, y_test = test_data[task_name]
                        X_t = torch.tensor(X_test).to(device)
                        task_idx = train_dataset.task_to_idx[task_name]
                        task_ids_t = torch.full((len(X_t),), task_idx, dtype=torch.long, device=device)

                        out = model(X_t, task_ids_t)
                        pred = out["prediction"].cpu().numpy()
                        corr = np.corrcoef(pred, y_test)[0, 1]
                        if not np.isnan(corr):
                            task_corrs[task_name] = corr

            avg_corr = np.mean(list(task_corrs.values())) if task_corrs else 0
            history["epoch"].append(epoch + 1)
            history["avg_corr"].append(avg_corr)

            if avg_corr > best_avg_corr:
                best_avg_corr = avg_corr

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}: avg_corr={avg_corr:+.4f}, best={best_avg_corr:+.4f}")
                for task, corr in sorted(task_corrs.items()):
                    print(f"    {task}: {corr:+.4f}")

    # Final evaluation
    model.eval()
    final_results = {}
    with torch.no_grad():
        for task_name in task_names:
            if task_name in test_data:
                X_test, y_test = test_data[task_name]
                X_t = torch.tensor(X_test).to(device)
                task_idx = train_dataset.task_to_idx[task_name]
                task_ids_t = torch.full((len(X_t),), task_idx, dtype=torch.long, device=device)

                out = model(X_t, task_ids_t)
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, y_test)[0, 1]
                final_results[task_name] = corr if not np.isnan(corr) else 0

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--drug-class", type=str, default="pi", choices=["pi", "nrti", "nnrti", "ini"])
    parser.add_argument("--no-gradnorm", action="store_true", help="Disable GradNorm")
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-TASK TRAINING FOR DRUG RESISTANCE")
    print("=" * 80)
    print(f"\nSettings: epochs={args.epochs}, drug_class={args.drug_class}, gradnorm={not args.no_gradnorm}")

    # Load data
    print(f"\nLoading {args.drug_class.upper()} data...")
    task_data, input_dim, task_names = load_drug_class_data(args.drug_class)

    if not task_data:
        print("No data available")
        return

    # Split into train/test
    train_data = {}
    test_data = {}
    for task_name, (X, y) in task_data.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data[task_name] = (X_train, y_train)
        test_data[task_name] = (X_test, y_test)

    # Create config
    cfg = MultiTaskConfig(
        input_dim=input_dim,
        n_tasks=len(task_names),
        epochs=args.epochs,
        use_gradnorm=not args.no_gradnorm,
    )

    # Train
    print("\n--- Training Multi-Task Model ---")
    results = train_multitask(cfg, train_data, test_data, task_names)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\n{'Drug':<8} {'Test Corr':<12}")
    print("-" * 25)

    for drug, corr in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{drug:<8} {corr:+.4f}")

    avg_corr = np.mean(list(results.values()))
    print("-" * 25)
    print(f"{'AVERAGE':<8} {avg_corr:+.4f}")

    # Save
    results_df = pd.DataFrame([{"drug_class": args.drug_class, "drug": d, "test_corr": c} for d, c in results.items()])
    results_path = project_root / "results" / f"multitask_{args.drug_class}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
