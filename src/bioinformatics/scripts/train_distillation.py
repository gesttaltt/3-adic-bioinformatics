#!/usr/bin/env python3
"""Knowledge Distillation: Transfer knowledge from best model to cross-dataset model.

Strategy:
1. Use best individual model (Transformer-ProTherm, 0.86) as TEACHER
2. Train a STUDENT model on combined S669+ProTherm data
3. Student learns from both ground truth AND teacher predictions
4. This allows cross-dataset generalization while preserving ProTherm expertise

The key insight: Instead of trying to learn everything from scratch on combined
data (which causes negative transfer), we use the teacher to regularize learning.
"""

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
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    # Model architecture
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Distillation
    temperature: float = 2.0  # Softens teacher predictions
    alpha: float = 0.7  # Weight for distillation loss (vs ground truth)

    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 30

    # Paths
    teacher_checkpoint: str | None = None


class DDGTransformer(nn.Module):
    """Transformer for DDG prediction."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project each feature to d_model
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Project features: (batch, input_dim) -> (batch, input_dim, d_model)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_enc

        # Transformer
        x = self.transformer(x)

        # Use CLS token for prediction
        cls_out = x[:, 0]
        pred = self.head(cls_out).squeeze(-1)

        return {"pred": pred, "embedding": cls_out}


def load_protherm_data() -> tuple[np.ndarray, np.ndarray]:
    """Load ProTherm curated dataset from built-in data."""
    from src.bioinformatics.data.protherm_loader import ProThermLoader

    loader = ProThermLoader()
    db = loader.load_curated()

    features_list = []
    labels_list = []

    for record in db.records:
        feat = compute_features(record.wild_type, record.mutant)
        features_list.append(feat.to_array(include_hyperbolic=False))
        labels_list.append(record.ddg)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


def load_s669_data() -> tuple[np.ndarray, np.ndarray]:
    """Load full S669 dataset."""
    loader = S669Loader()
    records = loader.load_from_csv()

    features_list = []
    labels_list = []

    for record in records:
        feat = compute_features(record.wild_type, record.mutant)
        features_list.append(feat.to_array(include_hyperbolic=False))
        labels_list.append(record.ddg)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


def train_teacher_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: DistillationConfig,
) -> tuple[DDGTransformer, float]:
    """Train teacher model on ProTherm (high-quality data)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDGTransformer(
        input_dim=X_train.shape[1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.mse_loss(out["pred"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Teacher epoch {epoch + 1}: val_spearman={val_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_corr


def distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Combined distillation + ground truth loss.

    Args:
        student_pred: Student model predictions
        teacher_pred: Teacher model predictions (detached)
        target: Ground truth labels
        temperature: Softens predictions (higher = softer)
        alpha: Weight for distillation loss (1-alpha for ground truth)

    Returns:
        Combined loss
    """
    # Distillation loss: match teacher's predictions
    # Using MSE scaled by temperature (similar to soft targets)
    distill_loss = F.mse_loss(student_pred / temperature, teacher_pred / temperature) * (temperature**2)

    # Ground truth loss
    gt_loss = F.mse_loss(student_pred, target)

    # Combine
    return alpha * distill_loss + (1 - alpha) * gt_loss


def train_student_with_distillation(
    teacher: DDGTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    source_train: np.ndarray,  # 0 = S669, 1 = ProTherm
    config: DistillationConfig,
) -> tuple[DDGTransformer, dict]:
    """Train student model on combined data with teacher guidance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.to(device)
    teacher.eval()

    student = DDGTransformer(
        input_dim=X_train.shape[1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    source_train_t = torch.tensor(source_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t, source_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_corr = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_corr": [], "distill_loss": [], "gt_loss": []}

    for epoch in range(config.epochs):
        student.train()
        epoch_losses = {"total": [], "distill": [], "gt": []}

        for batch_x, batch_y, batch_source in loader:
            optimizer.zero_grad()

            # Get teacher predictions
            with torch.no_grad():
                teacher_out = teacher(batch_x)
                teacher_pred = teacher_out["pred"]

            # Student predictions
            student_out = student(batch_x)
            student_pred = student_out["pred"]

            # Adaptive distillation weight based on source
            # Higher weight for S669 samples (where we want teacher knowledge)
            # Lower weight for ProTherm samples (where ground truth is reliable)
            source_alpha = torch.where(
                batch_source == 0,  # S669
                torch.tensor(config.alpha, device=device),  # Use teacher more
                torch.tensor(config.alpha * 0.5, device=device),  # Use ground truth more
            )

            # Compute losses separately for logging
            distill = F.mse_loss(student_pred / config.temperature, teacher_pred / config.temperature, reduction="none")
            distill = (distill * (config.temperature**2)).mean()

            gt = F.mse_loss(student_pred, batch_y)

            # Combined loss with adaptive alpha
            loss = source_alpha.mean() * distill + (1 - source_alpha.mean()) * gt

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["distill"].append(distill.item())
            epoch_losses["gt"].append(gt.item())

        scheduler.step()

        # Evaluate
        student.eval()
        with torch.no_grad():
            val_out = student(X_val_t)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        history["train_loss"].append(np.mean(epoch_losses["total"]))
        history["val_corr"].append(val_corr)
        history["distill_loss"].append(np.mean(epoch_losses["distill"]))
        history["gt_loss"].append(np.mean(epoch_losses["gt"]))

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            print(
                f"    Student epoch {epoch + 1}: val_spearman={val_corr:.4f}, "
                f"distill={np.mean(epoch_losses['distill']):.4f}, gt={np.mean(epoch_losses['gt']):.4f}"
            )

    if best_state:
        student.load_state_dict(best_state)

    return student, {"best_val_corr": best_val_corr, "history": history}


def evaluate_model(model: DDGTransformer, X: np.ndarray, y: np.ndarray, name: str = "Model") -> dict:
    """Evaluate model on a dataset."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(X_t)
        preds = out["pred"].cpu().numpy()

    spearman = spearmanr(y, preds)[0]
    pearson = pearsonr(y, preds)[0]
    mae = np.mean(np.abs(y - preds))
    rmse = np.sqrt(np.mean((y - preds) ** 2))

    return {
        "name": name,
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(y),
    }


def main():
    """Train distillation model."""
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION: ProTherm Teacher → Combined Student")
    print("=" * 70)

    config = DistillationConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/distillation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    print("\n[1] Loading data...")
    X_s669, y_s669 = load_s669_data()
    print(f"  S669: {len(X_s669)} samples")

    X_protherm, y_protherm = load_protherm_data()
    if X_protherm is not None:
        print(f"  ProTherm: {len(X_protherm)} samples")
    else:
        print("  ProTherm not found, using S669 subset")
        X_protherm = X_s669[:100]
        y_protherm = y_s669[:100]

    # Create splits
    np.random.seed(42)

    # ProTherm split (for teacher training)
    protherm_idx = np.random.permutation(len(X_protherm))
    protherm_train_idx = protherm_idx[: int(0.8 * len(protherm_idx))]
    protherm_val_idx = protherm_idx[int(0.8 * len(protherm_idx)) :]

    X_protherm_train, y_protherm_train = X_protherm[protherm_train_idx], y_protherm[protherm_train_idx]
    X_protherm_val, y_protherm_val = X_protherm[protherm_val_idx], y_protherm[protherm_val_idx]

    # S669 split
    s669_idx = np.random.permutation(len(X_s669))
    s669_train_idx = s669_idx[: int(0.8 * len(s669_idx))]
    s669_val_idx = s669_idx[int(0.8 * len(s669_idx)) :]

    X_s669_train, y_s669_train = X_s669[s669_train_idx], y_s669[s669_train_idx]
    X_s669_val, y_s669_val = X_s669[s669_val_idx], y_s669[s669_val_idx]

    print(f"  ProTherm train: {len(X_protherm_train)}, val: {len(X_protherm_val)}")
    print(f"  S669 train: {len(X_s669_train)}, val: {len(X_s669_val)}")

    # Train teacher on ProTherm
    print("\n[2] Training TEACHER model on ProTherm...")
    teacher, teacher_val_corr = train_teacher_model(
        X_protherm_train,
        y_protherm_train,
        X_protherm_val,
        y_protherm_val,
        config,
    )
    print(f"  Teacher ProTherm validation: Spearman = {teacher_val_corr:.4f}")

    # Evaluate teacher on S669
    teacher_s669 = evaluate_model(teacher, X_s669_val, y_s669_val, "Teacher-S669")
    print(f"  Teacher S669 validation: Spearman = {teacher_s669['spearman']:.4f}")

    # Create combined training set
    X_combined_train = np.vstack([X_s669_train, X_protherm_train])
    y_combined_train = np.concatenate([y_s669_train, y_protherm_train])
    source_combined_train = np.concatenate(
        [
            np.zeros(len(X_s669_train)),  # 0 = S669
            np.ones(len(X_protherm_train)),  # 1 = ProTherm
        ]
    )

    # Combined validation
    X_combined_val = np.vstack([X_s669_val, X_protherm_val])
    y_combined_val = np.concatenate([y_s669_val, y_protherm_val])

    print(f"\n  Combined train: {len(X_combined_train)} samples")
    print(f"  Combined val: {len(X_combined_val)} samples")

    # Train student with distillation
    print("\n[3] Training STUDENT model with distillation...")
    student, student_history = train_student_with_distillation(
        teacher,
        X_combined_train,
        y_combined_train,
        X_combined_val,
        y_combined_val,
        source_combined_train,
        config,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Teacher results
    print("\nTEACHER (trained on ProTherm only):")
    teacher_protherm = evaluate_model(teacher, X_protherm_val, y_protherm_val, "Teacher-ProTherm")
    teacher_s669 = evaluate_model(teacher, X_s669_val, y_s669_val, "Teacher-S669")
    teacher_combined = evaluate_model(teacher, X_combined_val, y_combined_val, "Teacher-Combined")
    print(f"  ProTherm: Spearman = {teacher_protherm['spearman']:.4f}")
    print(f"  S669:     Spearman = {teacher_s669['spearman']:.4f}")
    print(f"  Combined: Spearman = {teacher_combined['spearman']:.4f}")

    # Student results
    print("\nSTUDENT (distilled + combined training):")
    student_protherm = evaluate_model(student, X_protherm_val, y_protherm_val, "Student-ProTherm")
    student_s669 = evaluate_model(student, X_s669_val, y_s669_val, "Student-S669")
    student_combined = evaluate_model(student, X_combined_val, y_combined_val, "Student-Combined")
    print(f"  ProTherm: Spearman = {student_protherm['spearman']:.4f}")
    print(f"  S669:     Spearman = {student_s669['spearman']:.4f}")
    print(f"  Combined: Spearman = {student_combined['spearman']:.4f}")

    # Compare with baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print(f"""
| Model                    | ProTherm | S669  | Combined |
|--------------------------|----------|-------|----------|
| Teacher (ProTherm only)  | {teacher_protherm["spearman"]:.4f}   | {teacher_s669["spearman"]:.4f} | {teacher_combined["spearman"]:.4f}    |
| Student (distilled)      | {student_protherm["spearman"]:.4f}   | {student_s669["spearman"]:.4f} | {student_combined["spearman"]:.4f}    |
| Previous best combined   |  0.36    | 0.27  | 0.26     |
""")

    # Knowledge transfer analysis
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRANSFER ANALYSIS")
    print("=" * 70)

    s669_improvement = (student_s669["spearman"] - teacher_s669["spearman"]) / abs(teacher_s669["spearman"]) * 100
    protherm_retention = student_protherm["spearman"] / teacher_protherm["spearman"] * 100

    print(f"  S669 improvement (student vs teacher): {s669_improvement:+.1f}%")
    print(f"  ProTherm knowledge retention: {protherm_retention:.1f}%")

    # Save results
    results = {
        "teacher": {
            "protherm": teacher_protherm,
            "s669": teacher_s669,
            "combined": teacher_combined,
        },
        "student": {
            "protherm": student_protherm,
            "s669": student_s669,
            "combined": student_combined,
            "history": {k: [float(v) for v in vals] for k, vals in student_history["history"].items()},
        },
        "config": config.__dict__,
    }

    with open(output_dir / "distillation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save models
    torch.save({"model_state_dict": teacher.state_dict()}, output_dir / "teacher.pt")
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student.pt")

    print(f"\nResults saved to: {output_dir}")

    return student, results


if __name__ == "__main__":
    main()
