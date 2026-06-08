#!/usr/bin/env python3
"""Step 3: Optimized Multimodal Architecture.

Combines insights from Step 1 and Step 2:
1. Use pretrained specialist VAE embeddings (Step 1 was better than Step 2)
2. Use strong residual learning from VAE-ProTherm's prediction (baseline insight)
3. Learn attention weights over specialist embeddings
4. Apply MLP refiner with residual connections

Key insight: Don't train from scratch - leverage pretrained specialists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class OptimizedConfig:
    """Configuration for optimized multimodal architecture."""

    # Specialist VAE dimensions
    s669_dim: int = 16
    protherm_dim: int = 32
    wide_dim: int = 64

    # Fusion architecture
    fusion_dim: int = 64
    n_heads: int = 4

    # Training
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 40


class AttentionFusion(nn.Module):
    """Attention-based fusion of specialist embeddings.

    Uses the ProTherm embedding as query to attend over all embeddings.
    This makes ProTherm the "anchor" modality since it's our target task.
    """

    def __init__(self, dims: list[int], fusion_dim: int, n_heads: int = 4):
        super().__init__()

        # Project each modality to fusion dimension
        self.projections = nn.ModuleList([nn.Linear(d, fusion_dim) for d in dims])

        # Multi-head attention
        # Query: ProTherm embedding, Key/Value: all embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.SiLU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse embeddings with ProTherm as anchor.

        Args:
            embeddings: [s669, protherm, wide] tensors

        Returns:
            fused: Fused embedding
            weights: Attention weights for interpretability
        """
        # Project all embeddings
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings, strict=False)]

        # Stack as sequence [batch, 3, fusion_dim]
        kv = torch.stack(projected, dim=1)

        # Use ProTherm as query (it's our target task)
        query = projected[1].unsqueeze(1)  # [batch, 1, fusion_dim]

        # Attention: ProTherm queries other embeddings
        attn_out, attn_weights = self.attention(query, kv, kv)

        # Residual connection with ProTherm
        fused = self.output_proj(attn_out.squeeze(1) + projected[1])

        return fused, attn_weights.squeeze(1)


class OptimizedMultimodal(nn.Module):
    """Optimized multimodal predictor.

    Architecture:
    1. Frozen specialist VAE encoders (pretrained)
    2. Attention-based fusion with ProTherm as anchor
    3. Residual prediction from ProTherm VAE's DDG output
    4. MLP refiner for delta corrections
    """

    def __init__(
        self,
        vae_s669: DDGVAE,
        vae_protherm: DDGVAE,
        vae_wide: DDGVAE,
        config: OptimizedConfig,
    ):
        super().__init__()
        self.config = config

        # Store full VAEs (frozen)
        self.vae_s669 = vae_s669
        self.vae_protherm = vae_protherm
        self.vae_wide = vae_wide

        # Freeze all specialist VAEs
        for vae in [self.vae_s669, self.vae_protherm, self.vae_wide]:
            for param in vae.parameters():
                param.requires_grad = False

        # Get encoder references for convenience
        self.encoder_s669 = vae_s669.encoder
        self.encoder_protherm = vae_protherm.encoder
        self.encoder_wide = vae_wide.encoder

        # Attention fusion
        self.fusion = AttentionFusion(
            dims=[config.s669_dim, config.protherm_dim, config.wide_dim],
            fusion_dim=config.fusion_dim,
            n_heads=config.n_heads,
        )

        # Refiner MLP (learns delta from fused embedding)
        self.refiner = nn.Sequential(
            nn.Linear(config.fusion_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Residual weight (controls blend of baseline + refinement)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def get_baseline_pred(self, x_protherm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get baseline DDG prediction from ProTherm VAE."""
        with torch.no_grad():
            out = self.vae_protherm(x_protherm)
            mu = out["mu"]
            ddg = out["ddg_pred"]
            # Ensure ddg is 1D
            if ddg.dim() > 1:
                ddg = ddg.squeeze()
        return mu, ddg

    def forward(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
    ) -> dict:
        """Forward pass."""
        # Get frozen embeddings
        with torch.no_grad():
            mu_s669, _ = self.encoder_s669(x_s669)
            mu_protherm, _ = self.encoder_protherm(x_protherm)
            mu_wide, _ = self.encoder_wide(x_wide)

        # Get baseline prediction from ProTherm VAE
        _, baseline_ddg = self.get_baseline_pred(x_protherm)

        # Attention fusion
        fused, attn_weights = self.fusion([mu_s669, mu_protherm, mu_wide])

        # Refiner delta
        delta = self.refiner(fused).squeeze(-1)

        # Ensure proper shapes
        if baseline_ddg.dim() > 1:
            baseline_ddg = baseline_ddg.squeeze(-1)
        if delta.dim() > 1:
            delta = delta.squeeze(-1)

        # Residual combination
        weight = torch.sigmoid(self.residual_weight)

        # Debug shapes (remove after fixing)
        # print(f"baseline_ddg: {baseline_ddg.shape}, delta: {delta.shape}")

        # Ensure 1D tensors
        if baseline_ddg.dim() == 0:
            baseline_ddg = baseline_ddg.unsqueeze(0)
        if delta.dim() == 0:
            delta = delta.unsqueeze(0)

        # If still 2D, take first column (output_dim=1)
        if baseline_ddg.dim() == 2:
            baseline_ddg = baseline_ddg[:, 0]
        if delta.dim() == 2:
            delta = delta[:, 0]

        ddg = baseline_ddg + weight * delta

        return {
            "ddg": ddg,
            "baseline_ddg": baseline_ddg,
            "delta": delta,
            "weight": weight,
            "fused": fused,
            "attn_weights": attn_weights,
            "embeddings": {
                "s669": mu_s669,
                "protherm": mu_protherm,
                "wide": mu_wide,
            },
        }


class OptimizedDataset(Dataset):
    """Dataset providing features for all three VAE encoders."""

    def __init__(self, records: list):
        self.features_s669 = []
        self.features_protherm = []
        self.features_wide = []
        self.ddg_values = []

        for record in records:
            feat = compute_features(record.wild_type, record.mutant)
            basic_arr = feat.to_array(include_hyperbolic=False)
            protherm_arr = np.pad(basic_arr, (0, 6), mode="constant")

            self.features_s669.append(basic_arr)
            self.features_protherm.append(protherm_arr)
            self.features_wide.append(basic_arr)
            self.ddg_values.append(record.ddg)

        self.features_s669 = np.array(self.features_s669, dtype=np.float32)
        self.features_protherm = np.array(self.features_protherm, dtype=np.float32)
        self.features_wide = np.array(self.features_wide, dtype=np.float32)
        self.ddg_values = np.array(self.ddg_values, dtype=np.float32)

    def __len__(self):
        return len(self.ddg_values)

    def __getitem__(self, idx):
        return {
            "x_s669": torch.from_numpy(self.features_s669[idx]),
            "x_protherm": torch.from_numpy(self.features_protherm[idx]),
            "x_wide": torch.from_numpy(self.features_wide[idx]),
            "ddg": torch.tensor(self.ddg_values[idx], dtype=torch.float32),
        }


def train_step3():
    """Train optimized multimodal architecture."""
    print("=" * 70)
    print("STEP 3: Optimized Multimodal Architecture")
    print("=" * 70)
    print("\nKey insights applied:")
    print("  1. Use pretrained specialist VAE embeddings (Step 1)")
    print("  2. Use ProTherm VAE's DDG as strong baseline (baseline insight)")
    print("  3. Attention fusion with ProTherm as anchor")
    print("  4. Residual delta learning from fused embedding")

    set_deterministic_mode(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = OptimizedConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/multimodal_step3_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load specialist VAEs
    print("\n[1] Loading specialist VAEs...")

    vae_s669 = DDGVAE.create_s669_variant(use_hyperbolic=False)
    ckpt = torch.load(
        "outputs/ddg_vae_training_20260129_212316/vae_s669/best.pt", map_location=device, weights_only=False
    )
    vae_s669.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-S669")

    vae_protherm = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt = torch.load(
        "outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt", map_location=device, weights_only=False
    )
    vae_protherm.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-ProTherm (will provide baseline DDG)")

    vae_wide = DDGVAE.create_wide_variant(use_hyperbolic=False)
    ckpt = torch.load("outputs/vae_wide_filtered_20260129_220019/best.pt", map_location=device, weights_only=False)
    vae_wide.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-Wide")

    # Create model
    print("\n[2] Creating optimized multimodal model...")
    model = OptimizedMultimodal(vae_s669, vae_protherm, vae_wide, config).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {frozen:,}")

    # Load data
    print("\n[3] Loading ProTherm data...")
    loader = ProThermLoader()
    db = loader.load_curated()
    records = db.records
    print(f"  Loaded {len(records)} mutations")

    dataset = OptimizedDataset(records)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # First, evaluate baseline
    print("\n[4] Evaluating ProTherm VAE baseline...")
    model.eval()
    baseline_preds = []
    baseline_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x_protherm = batch["x_protherm"].to(device)
            ddg = batch["ddg"]

            _, baseline_ddg = model.get_baseline_pred(x_protherm)
            baseline_preds.extend(baseline_ddg.cpu().numpy())
            baseline_targets.extend(ddg.numpy())

    baseline_spearman = spearmanr(baseline_targets, baseline_preds)[0]
    print(f"  ProTherm VAE baseline Spearman: {baseline_spearman:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training
    print("\n" + "=" * 70)
    print("[5] Training Optimized Multimodal")
    print("=" * 70)

    best_spearman = baseline_spearman  # Start from baseline
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_spearman": [],
        "val_pearson": [],
        "delta_contribution": [],
        "attn_weights": [],
    }

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            x_s669 = batch["x_s669"].to(device)
            x_protherm = batch["x_protherm"].to(device)
            x_wide = batch["x_wide"].to(device)
            ddg = batch["ddg"].to(device)

            optimizer.zero_grad()
            out = model(x_s669, x_protherm, x_wide)

            # MSE loss
            loss = F.mse_loss(out["ddg"], ddg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        all_preds = []
        all_baseline = []
        all_delta = []
        all_targets = []
        all_attn = []

        with torch.no_grad():
            for batch in val_loader:
                x_s669 = batch["x_s669"].to(device)
                x_protherm = batch["x_protherm"].to(device)
                x_wide = batch["x_wide"].to(device)
                ddg = batch["ddg"].to(device)

                out = model(x_s669, x_protherm, x_wide)
                loss = F.mse_loss(out["ddg"], ddg)

                val_losses.append(loss.item())
                all_preds.extend(out["ddg"].cpu().numpy())
                all_baseline.extend(out["baseline_ddg"].cpu().numpy())
                all_delta.extend(out["delta"].cpu().numpy())
                all_targets.extend(ddg.cpu().numpy())
                all_attn.append(out["attn_weights"].cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_targets, all_preds)[0]
        val_pearson = pearsonr(all_targets, all_preds)[0]

        # Analyze delta contribution
        delta_std = np.std(all_delta)
        baseline_std = np.std(all_baseline)
        delta_ratio = delta_std / (baseline_std + 1e-8)

        # Average attention weights
        avg_attn = np.concatenate(all_attn, axis=0).mean(axis=0)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_spearman"].append(float(val_spearman))
        history["val_pearson"].append(float(val_pearson))
        history["delta_contribution"].append(float(delta_ratio))
        history["attn_weights"].append(avg_attn.tolist())

        scheduler.step()

        weight = torch.sigmoid(model.residual_weight).item()

        if epoch % 20 == 0 or val_spearman > best_spearman:
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"spearman={val_spearman:.4f} weight={weight:.3f} "
                f"attn=[{avg_attn[0]:.2f},{avg_attn[1]:.2f},{avg_attn[2]:.2f}]"
            )

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_pearson = val_pearson
            best_attn = avg_attn
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                    "val_pearson": val_pearson,
                    "attn_weights": avg_attn.tolist(),
                    "config": config.__dict__,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ========================================
    # Final Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("[6] Multimodality Analysis")
    print("=" * 70)

    # Load best model
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get final predictions
    all_final_preds = []
    all_baseline_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x_s669 = batch["x_s669"].to(device)
            x_protherm = batch["x_protherm"].to(device)
            x_wide = batch["x_wide"].to(device)
            ddg = batch["ddg"]

            out = model(x_s669, x_protherm, x_wide)
            all_final_preds.extend(out["ddg"].cpu().numpy())
            all_baseline_preds.extend(out["baseline_ddg"].cpu().numpy())
            all_targets.extend(ddg.numpy())

    final_spearman = spearmanr(all_targets, all_final_preds)[0]
    final_pearson = pearsonr(all_targets, all_final_preds)[0]
    baseline_spearman_final = spearmanr(all_targets, all_baseline_preds)[0]

    print(f"\n  Attention weights: [S669={best_attn[0]:.3f}, ProTherm={best_attn[1]:.3f}, Wide={best_attn[2]:.3f}]")
    print(f"\n  Baseline (ProTherm VAE):  Spearman={baseline_spearman_final:.4f}")
    print(f"  Optimized Multimodal:     Spearman={final_spearman:.4f}, Pearson={final_pearson:.4f}")
    print(f"\n  Improvement: {(final_spearman - baseline_spearman_final) / abs(baseline_spearman_final) * 100:+.1f}%")

    # Save results
    results = {
        "baseline_spearman": float(baseline_spearman_final),
        "final_spearman": float(final_spearman),
        "final_pearson": float(final_pearson),
        "best_attn_weights": best_attn.tolist(),
        "improvement_pct": float((final_spearman - baseline_spearman_final) / abs(baseline_spearman_final) * 100),
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========================================
    # Step 3b: Add MLP Refiner on top
    # ========================================
    print("\n" + "=" * 70)
    print("[7] Training MLP Refiner on Step 3 Embeddings")
    print("=" * 70)

    # Extract fused embeddings and predictions
    class Step3EmbeddingDataset(Dataset):
        def __init__(self, base_dataset, model, device):
            self.embeddings = []
            self.model_preds = []
            self.labels = []

            model.eval()
            with torch.no_grad():
                for i in range(len(base_dataset)):
                    item = base_dataset[i]
                    x_s669 = item["x_s669"].unsqueeze(0).to(device)
                    x_protherm = item["x_protherm"].unsqueeze(0).to(device)
                    x_wide = item["x_wide"].unsqueeze(0).to(device)

                    out = model(x_s669, x_protherm, x_wide)
                    self.embeddings.append(out["fused"].cpu().squeeze(0))
                    self.model_preds.append(out["ddg"].cpu().item())
                    ddg = item["ddg"]
                    self.labels.append(ddg.item() if hasattr(ddg, "item") else ddg)

            self.embeddings = torch.stack(self.embeddings)
            self.model_preds = torch.tensor(self.model_preds, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

        @property
        def embedding_dim(self):
            return self.embeddings.shape[1]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "embedding": self.embeddings[idx],
                "vae_pred": self.model_preds[idx],
                "label": self.labels[idx],
            }

    # Create embedding datasets
    train_emb_ds = Step3EmbeddingDataset(train_ds, model, device)
    val_emb_ds = Step3EmbeddingDataset(val_ds, model, device)

    print(f"  Fused embedding dim: {train_emb_ds.embedding_dim}")

    # Train MLP Refiner
    refiner_config = RefinerConfig(
        latent_dim=config.fusion_dim,
        hidden_dims=[64, 32],
        dropout=0.1,
        use_residual=True,
        initial_residual_weight=0.3,
    )

    refiner = DDGMLPRefiner(config=refiner_config).to(device)
    refiner_optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3, weight_decay=1e-5)
    refiner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refiner_optimizer, T_max=100)

    refiner_train_loader = DataLoader(train_emb_ds, batch_size=16, shuffle=True)
    refiner_val_loader = DataLoader(val_emb_ds, batch_size=16, shuffle=False)

    best_refiner_spearman = -1

    for epoch in range(100):
        # Train
        refiner.train()
        for batch in refiner_train_loader:
            z = batch["embedding"].to(device)
            vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
            y = batch["label"].to(device).unsqueeze(-1)

            refiner_optimizer.zero_grad()
            loss_dict = refiner.loss(z, y, vae_pred)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            refiner_optimizer.step()

        # Validate
        refiner.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in refiner_val_loader:
                z = batch["embedding"].to(device)
                vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
                y = batch["label"].to(device).unsqueeze(-1)

                out = refiner(z, vae_pred)
                all_preds.extend(out["ddg_pred"].cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        val_spearman = spearmanr(all_preds, all_labels)[0]
        refiner_scheduler.step()

        if epoch % 20 == 0 or val_spearman > best_refiner_spearman:
            rw = torch.sigmoid(refiner.residual_weight).item()
            print(f"Epoch {epoch:3d}: spearman={val_spearman:.4f} res_w={rw:.3f}")

        if val_spearman > best_refiner_spearman:
            best_refiner_spearman = val_spearman
            torch.save(
                {
                    "model_state_dict": refiner.state_dict(),
                    "val_spearman": val_spearman,
                },
                output_dir / "mlp_refiner_best.pt",
            )

    print(f"\n  Step 3 + MLP Refiner Spearman: {best_refiner_spearman:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(f"  Optimized Multimodal Spearman: {best_spearman:.4f}")
    print(f"  Step 3 + MLP Refiner Spearman: {best_refiner_spearman:.4f}")
    print("\nComparison:")
    print("  Step 1 Meta-VAE + Refiner:     0.63")
    print("  Step 2 Combined + Refiner:     0.46")
    print(f"  Step 3 Optimized Multimodal:   {best_spearman:.2f}")
    print(f"  Step 3 + MLP Refiner:          {best_refiner_spearman:.2f}")
    print("  Baseline VAE-ProTherm:         0.64")
    print("  Baseline + MLP Refiner:        0.78")
    print(f"\nCheckpoints: {output_dir}")

    # Determine if true multimodality achieved
    if best_refiner_spearman > 0.78:
        print("\n  MULTIMODAL BEATS BEST BASELINE!")
    elif best_spearman > 0.64:
        print("\n  TRUE MULTIMODALITY: Improves over single VAE baseline!")
    else:
        print("\n  No multimodal benefit over single VAE baseline.")

    return {
        "multimodal_spearman": best_spearman,
        "refiner_spearman": best_refiner_spearman,
        "pearson": best_pearson,
        "attn_weights": best_attn.tolist(),
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    train_step3()
