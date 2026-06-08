"""Extended training test for top synergies.

Tests whether multi-module configurations improve over simple ranking
with longer training (200+ epochs).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class Config:
    input_dim: int = 9
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    batch_size: int = 32
    epochs: int = 200
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_hyper: bool = False
    use_trop: bool = False
    use_triplet: bool = False
    use_rank: bool = False
    use_contrast: bool = False

    padic_weight: float = 0.5
    ranking_weight: float = 0.3


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        if cfg.use_hyper:
            self.hyper_scale = nn.Parameter(torch.ones(1))
        if cfg.use_trop:
            self.trop_heads = nn.Linear(cfg.latent_dim, 4)

        # Decoder
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h)])
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.cfg.use_trop:
            w = F.softmax(self.trop_heads(mu) / 0.1, dim=-1)
            mu = mu * w.mean(dim=-1, keepdim=True)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        if self.cfg.use_hyper:
            z_norm = torch.norm(z, dim=-1, keepdim=True)
            z = z * (torch.clamp(z_norm, max=0.99) / (z_norm + 1e-8)) * self.hyper_scale

        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


def compute_loss(cfg, out, x, fitness):
    losses = {}
    losses["recon"] = F.mse_loss(out["x_recon"], x)
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    z = out["z"]

    if cfg.use_triplet:
        total = 0.0
        n = 0
        for i in range(min(z.size(0), 32)):
            diff = torch.abs(fitness - fitness[i])
            idx = torch.argsort(diff)
            if len(idx) >= 3:
                d_pos = torch.norm(z[i] - z[idx[1]])
                d_neg = torch.norm(z[i] - z[idx[-1]])
                total += F.relu(d_pos - d_neg + 1.0)
                n += 1
        losses["triplet"] = cfg.padic_weight * total / max(n, 1)

    if cfg.use_rank:
        z_proj = z[:, 0]
        z_c = z_proj - z_proj.mean()
        f_c = fitness - fitness.mean()
        corr = torch.sum(z_c * f_c) / (torch.sqrt(torch.sum(z_c**2) * torch.sum(f_c**2)) + 1e-8)
        losses["rank"] = cfg.ranking_weight * (-corr)

    if cfg.use_contrast:
        z_norm = F.normalize(z, dim=-1)
        sim = torch.mm(z_norm, z_norm.t()) / 0.1
        labels = torch.arange(z.size(0), device=z.device)
        losses["contrast"] = 0.1 * F.cross_entropy(sim, labels)

    losses["total"] = sum(losses.values())
    return losses


def generate_data(n=1000):
    np.random.seed(42)
    n_clusters = 5
    centers = np.random.randn(n_clusters, 9).astype(np.float32)
    cluster_fitness = np.linspace(0, 1, n_clusters)

    seqs = np.zeros((n, 9), dtype=np.float32)
    labels = np.zeros(n)

    for i in range(n):
        c = np.random.randint(0, n_clusters)
        seqs[i] = centers[c] + np.random.randn(9).astype(np.float32) * 0.3
        labels[i] = cluster_fitness[c] + np.random.randn() * 0.1

    seqs = (seqs - seqs.min()) / (seqs.max() - seqs.min() + 1e-8)
    labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

    return torch.tensor(seqs), torch.tensor(labels, dtype=torch.float32)


def train(cfg, train_x, train_y, checkpoints=None):
    if checkpoints is None:
        checkpoints = [50, 100, 150, 200, 250, 300]

    device = torch.device(cfg.device)
    model = VAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    history = {"corr": [], "acc": [], "epoch": []}

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            losses = compute_loss(cfg, out, x, y)
            losses["total"].backward()
            opt.step()

        # Evaluate at checkpoints
        if (epoch + 1) in checkpoints or (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                out = model(train_x.to(device))
                z = out["z"][:, 0].cpu().numpy()
                f = train_y.numpy()
                corr = np.corrcoef(z, f)[0, 1] if np.std(z) > 1e-08 else 0.0

                err = F.mse_loss(out["x_recon"], train_x.to(device), reduction="none").mean(dim=-1)
                acc = (err < 0.1).float().mean().item()

            history["corr"].append(corr)
            history["acc"].append(acc)
            history["epoch"].append(epoch + 1)

    return history


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    print("=" * 80)
    print(f"EXTENDED TRAINING: TOP SYNERGIES ({args.epochs} epochs)")
    print("=" * 80)

    # Generate data
    print("Generating data...")
    train_x, train_y = generate_data(1000)
    print(f"Data: {train_x.shape}")
    print()

    # Configs to test (most promising synergies)
    configs = [
        ("rank", {"use_rank": True}),
        ("rank_contrast", {"use_rank": True, "use_contrast": True}),
        ("trop_rank_contrast", {"use_trop": True, "use_rank": True, "use_contrast": True}),
        ("triplet_rank_contrast", {"use_triplet": True, "use_rank": True, "use_contrast": True}),
        ("hyper_triplet_rank", {"use_hyper": True, "use_triplet": True, "use_rank": True}),
        ("hyper_trop_triplet_rank", {"use_hyper": True, "use_trop": True, "use_triplet": True, "use_rank": True}),
        (
            "all_modules",
            {"use_hyper": True, "use_trop": True, "use_triplet": True, "use_rank": True, "use_contrast": True},
        ),
    ]

    results = {}
    checkpoints = [50, 100, 150, 200, 250, 300]

    for name, flags in configs:
        print(f"Training: {name}...")
        cfg = Config(epochs=args.epochs)
        for k, v in flags.items():
            setattr(cfg, k, v)

        history = train(cfg, train_x, train_y, checkpoints)
        results[name] = history

        # Print progress
        for i, ep in enumerate(history["epoch"]):
            if ep <= args.epochs:
                print(f"  Epoch {ep}: corr={history['corr'][i]:+.4f}, acc={history['acc'][i]:.1%}")
        print()

    # Summary table
    print("=" * 80)
    print("CONVERGENCE COMPARISON")
    print("=" * 80)
    print()

    # Header
    header = f"{'Config':<25}"
    for ep in checkpoints:
        if ep <= args.epochs:
            header += f" @{ep:>4}"
    header += "  Improve"
    print(header)
    print("-" * 80)

    for name in results:
        h = results[name]
        row = f"{name:<25}"
        for i, ep in enumerate(h["epoch"]):
            if ep <= args.epochs and i < len(h["corr"]):
                row += f" {h['corr'][i]:+.4f}"

        # Improvement from first to last
        if len(h["corr"]) >= 2:
            improve = h["corr"][-1] - h["corr"][0]
            row += f"  {improve:+.4f}"
        print(row)

    # Analysis
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Best at final epoch
    final_epoch = min(args.epochs, max(checkpoints))
    final_results = {}
    for name, h in results.items():
        for i, ep in enumerate(h["epoch"]):
            if ep == final_epoch or (ep <= final_epoch and i == len(h["epoch"]) - 1):
                final_results[name] = h["corr"][i]
                break

    best_name = max(final_results, key=final_results.get)
    print(f"Best at epoch {final_epoch}: {best_name} ({final_results[best_name]:+.4f})")
    print()

    # Compare to rank baseline
    rank_final = final_results.get("rank", 0)
    print("Comparison to rank baseline:")
    for name, corr in sorted(final_results.items(), key=lambda x: -x[1]):
        diff = corr - rank_final
        status = "BETTER" if diff > 0.005 else ("SAME" if diff > -0.005 else "WORSE")
        print(f"  {name:<25} {corr:+.4f} ({diff:+.4f}) {status}")

    # Check for late bloomers
    print()
    print("Late bloomer analysis (improvement after epoch 100):")
    for name, h in results.items():
        # Find correlation at 100 and at end
        corr_100 = None
        corr_end = None
        for i, ep in enumerate(h["epoch"]):
            if ep == 100:
                corr_100 = h["corr"][i]
            if ep == h["epoch"][-1]:
                corr_end = h["corr"][i]

        if corr_100 is not None and corr_end is not None:
            late_improve = corr_end - corr_100
            if late_improve > 0.01:
                print(f"  {name}: +{late_improve:.4f} improvement after epoch 100")


if __name__ == "__main__":
    main()
