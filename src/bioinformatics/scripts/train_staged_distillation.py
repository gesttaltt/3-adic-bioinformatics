#!/usr/bin/env python3
"""Staged Distillation: Two-phase training for cross-dataset generalization.

Phase 1: Train student to match teacher on ProTherm (preserve expertise)
Phase 2: Fine-tune on S669 with teacher regularization (learn new domain)

This staged approach preserves more ProTherm knowledge while learning S669.
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
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class StagedConfig:
    """Configuration for staged distillation."""

    # Model architecture
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Phase 1: Match teacher on ProTherm
    phase1_epochs: int = 100
    phase1_lr: float = 3e-4

    # Phase 2: Fine-tune on S669
    phase2_epochs: int = 150
    phase2_lr: float = 1e-4  # Lower LR to preserve knowledge
    teacher_reg_weight: float = 0.3  # Regularize toward teacher

    # Common
    batch_size: int = 32
    weight_decay: float = 1e-4
    patience: int = 25


class DDGTransformer(nn.Module):
    """Transformer for DDG prediction."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_enc
        x = self.transformer(x)
        cls_out = x[:, 0]
        pred = self.head(cls_out).squeeze(-1)
        return {"pred": pred, "embedding": cls_out}


def load_data():
    """Load both datasets."""
    # ProTherm
    protherm_loader = ProThermLoader()
    db = protherm_loader.load_curated()

    X_protherm, y_protherm = [], []
    for record in db.records:
        feat = compute_features(record.wild_type, record.mutant)
        X_protherm.append(feat.to_array(include_hyperbolic=False))
        y_protherm.append(record.ddg)
    X_protherm = np.array(X_protherm, dtype=np.float32)
    y_protherm = np.array(y_protherm, dtype=np.float32)

    # S669
    s669_loader = S669Loader()
    records = s669_loader.load_from_csv()

    X_s669, y_s669 = [], []
    for record in records:
        feat = compute_features(record.wild_type, record.mutant)
        X_s669.append(feat.to_array(include_hyperbolic=False))
        y_s669.append(record.ddg)
    X_s669 = np.array(X_s669, dtype=np.float32)
    y_s669 = np.array(y_s669, dtype=np.float32)

    return X_protherm, y_protherm, X_s669, y_s669


def train_teacher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: StagedConfig,
) -> tuple[DDGTransformer, float]:
    """Train teacher on ProTherm."""
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.phase1_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.phase1_epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0

    for epoch in range(config.phase1_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.mse_loss(out["pred"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

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


def phase1_match_teacher(
    student: DDGTransformer,
    teacher: DDGTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: StagedConfig,
) -> tuple[DDGTransformer, float]:
    """Phase 1: Train student to match teacher on ProTherm."""
    device = next(teacher.parameters()).device
    teacher.eval()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Get teacher predictions for training data
    with torch.no_grad():
        teacher_train = teacher(X_train_t)["pred"]
        teacher(X_val_t)["pred"]

    dataset = TensorDataset(X_train_t, y_train_t, teacher_train)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.phase1_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.phase1_epochs // 2)

    best_val_corr = -1
    best_state = None
    patience_counter = 0

    for epoch in range(config.phase1_epochs // 2):
        student.train()
        for batch_x, batch_y, batch_teacher in loader:
            optimizer.zero_grad()
            out = student(batch_x)

            # Match both teacher and ground truth
            teacher_loss = F.mse_loss(out["pred"], batch_teacher)
            gt_loss = F.mse_loss(out["pred"], batch_y)
            loss = 0.5 * teacher_loss + 0.5 * gt_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        student.eval()
        with torch.no_grad():
            val_out = student(X_val_t)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Phase 1 epoch {epoch + 1}: val_spearman={val_corr:.4f}")

    if best_state:
        student.load_state_dict(best_state)
    return student, best_val_corr


def phase2_finetune_s669(
    student: DDGTransformer,
    teacher: DDGTransformer,
    X_s669_train: np.ndarray,
    y_s669_train: np.ndarray,
    X_s669_val: np.ndarray,
    y_s669_val: np.ndarray,
    X_protherm_val: np.ndarray,
    y_protherm_val: np.ndarray,
    config: StagedConfig,
) -> tuple[DDGTransformer, dict]:
    """Phase 2: Fine-tune on S669 with teacher regularization."""
    device = next(teacher.parameters()).device
    teacher.eval()

    X_s669_train_t = torch.tensor(X_s669_train, dtype=torch.float32, device=device)
    y_s669_train_t = torch.tensor(y_s669_train, dtype=torch.float32, device=device)
    X_s669_val_t = torch.tensor(X_s669_val, dtype=torch.float32, device=device)
    y_s669_val_t = torch.tensor(y_s669_val, dtype=torch.float32, device=device)
    X_protherm_val_t = torch.tensor(X_protherm_val, dtype=torch.float32, device=device)
    y_protherm_val_t = torch.tensor(y_protherm_val, dtype=torch.float32, device=device)

    # Pre-compute teacher predictions for S669
    with torch.no_grad():
        teacher_s669 = teacher(X_s669_train_t)["pred"]

    dataset = TensorDataset(X_s669_train_t, y_s669_train_t, teacher_s669)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Use lower LR to preserve ProTherm knowledge
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.phase2_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_combined_corr = -1
    best_state = None
    patience_counter = 0
    history = {"s669_corr": [], "protherm_corr": [], "combined_corr": []}

    for epoch in range(config.phase2_epochs):
        student.train()
        for batch_x, batch_y, batch_teacher in loader:
            optimizer.zero_grad()
            out = student(batch_x)

            # Ground truth loss (primary)
            gt_loss = F.mse_loss(out["pred"], batch_y)

            # Teacher regularization (prevent forgetting)
            teacher_reg = F.mse_loss(out["pred"], batch_teacher)

            loss = gt_loss + config.teacher_reg_weight * teacher_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on both datasets
        student.eval()
        with torch.no_grad():
            s669_out = student(X_s669_val_t)
            protherm_out = student(X_protherm_val_t)

            s669_corr = spearmanr(y_s669_val_t.cpu().numpy(), s669_out["pred"].cpu().numpy())[0]
            protherm_corr = spearmanr(y_protherm_val_t.cpu().numpy(), protherm_out["pred"].cpu().numpy())[0]

            # Combined metric: average of both
            combined_corr = (s669_corr + protherm_corr) / 2

        history["s669_corr"].append(s669_corr)
        history["protherm_corr"].append(protherm_corr)
        history["combined_corr"].append(combined_corr)

        if combined_corr > best_combined_corr:
            best_combined_corr = combined_corr
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            print(
                f"    Phase 2 epoch {epoch + 1}: S669={s669_corr:.4f}, ProTherm={protherm_corr:.4f}, Combined={combined_corr:.4f}"
            )

    if best_state:
        student.load_state_dict(best_state)
    return student, {"best_combined_corr": best_combined_corr, "history": history}


def evaluate(model: DDGTransformer, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Evaluate model."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(X_t)
        preds = out["pred"].cpu().numpy()

    return {
        "name": name,
        "spearman": spearmanr(y, preds)[0],
        "pearson": pearsonr(y, preds)[0],
        "mae": np.mean(np.abs(y - preds)),
        "n": len(y),
    }


def main():
    """Run staged distillation."""
    print("=" * 70)
    print("STAGED DISTILLATION: Phase 1 (Match Teacher) → Phase 2 (Fine-tune S669)")
    print("=" * 70)

    config = StagedConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/staged_distillation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1] Loading data...")
    X_protherm, y_protherm, X_s669, y_s669 = load_data()
    print(f"  ProTherm: {len(X_protherm)} samples")
    print(f"  S669: {len(X_s669)} samples")

    # Splits
    np.random.seed(42)

    protherm_idx = np.random.permutation(len(X_protherm))
    protherm_split = int(0.8 * len(protherm_idx))
    X_protherm_train = X_protherm[protherm_idx[:protherm_split]]
    y_protherm_train = y_protherm[protherm_idx[:protherm_split]]
    X_protherm_val = X_protherm[protherm_idx[protherm_split:]]
    y_protherm_val = y_protherm[protherm_idx[protherm_split:]]

    s669_idx = np.random.permutation(len(X_s669))
    s669_split = int(0.8 * len(s669_idx))
    X_s669_train = X_s669[s669_idx[:s669_split]]
    y_s669_train = y_s669[s669_idx[:s669_split]]
    X_s669_val = X_s669[s669_idx[s669_split:]]
    y_s669_val = y_s669[s669_idx[s669_split:]]

    print(f"  ProTherm train/val: {len(X_protherm_train)}/{len(X_protherm_val)}")
    print(f"  S669 train/val: {len(X_s669_train)}/{len(X_s669_val)}")

    # Train teacher
    print("\n[2] Training TEACHER on ProTherm...")
    teacher, teacher_val_corr = train_teacher(
        X_protherm_train,
        y_protherm_train,
        X_protherm_val,
        y_protherm_val,
        config,
    )
    print(f"  Teacher ProTherm: Spearman = {teacher_val_corr:.4f}")

    teacher_s669 = evaluate(teacher, X_s669_val, y_s669_val, "Teacher-S669")
    print(f"  Teacher S669: Spearman = {teacher_s669['spearman']:.4f}")

    # Phase 1: Match teacher
    print("\n[3] PHASE 1: Training student to match teacher on ProTherm...")
    student = DDGTransformer(
        input_dim=X_protherm.shape[1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    student, phase1_corr = phase1_match_teacher(
        student,
        teacher,
        X_protherm_train,
        y_protherm_train,
        X_protherm_val,
        y_protherm_val,
        config,
    )
    print(f"  Student after Phase 1 (ProTherm): Spearman = {phase1_corr:.4f}")

    phase1_s669 = evaluate(student, X_s669_val, y_s669_val, "Student-Phase1-S669")
    print(f"  Student after Phase 1 (S669): Spearman = {phase1_s669['spearman']:.4f}")

    # Phase 2: Fine-tune on S669
    print("\n[4] PHASE 2: Fine-tuning on S669 with teacher regularization...")
    student, phase2_history = phase2_finetune_s669(
        student,
        teacher,
        X_s669_train,
        y_s669_train,
        X_s669_val,
        y_s669_val,
        X_protherm_val,
        y_protherm_val,
        config,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    teacher_protherm = evaluate(teacher, X_protherm_val, y_protherm_val, "Teacher-ProTherm")
    teacher_s669 = evaluate(teacher, X_s669_val, y_s669_val, "Teacher-S669")
    student_protherm = evaluate(student, X_protherm_val, y_protherm_val, "Student-ProTherm")
    student_s669 = evaluate(student, X_s669_val, y_s669_val, "Student-S669")

    # Combined evaluation
    X_combined_val = np.vstack([X_protherm_val, X_s669_val])
    y_combined_val = np.concatenate([y_protherm_val, y_s669_val])
    teacher_combined = evaluate(teacher, X_combined_val, y_combined_val, "Teacher-Combined")
    student_combined = evaluate(student, X_combined_val, y_combined_val, "Student-Combined")

    print(f"""
| Model           | ProTherm | S669   | Combined |
|-----------------|----------|--------|----------|
| Teacher         | {teacher_protherm["spearman"]:.4f}   | {teacher_s669["spearman"]:.4f} | {teacher_combined["spearman"]:.4f}   |
| Student         | {student_protherm["spearman"]:.4f}   | {student_s669["spearman"]:.4f} | {student_combined["spearman"]:.4f}   |

Knowledge Retention (ProTherm): {student_protherm["spearman"] / teacher_protherm["spearman"] * 100:.1f}%
S669 Improvement: {(student_s669["spearman"] - teacher_s669["spearman"]) / abs(teacher_s669["spearman"]) * 100:+.1f}%
""")

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
        },
        "phase2_history": {k: [float(v) for v in vals] for k, vals in phase2_history["history"].items()},
        "config": config.__dict__,
    }

    with open(output_dir / "staged_distillation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({"model_state_dict": teacher.state_dict()}, output_dir / "teacher.pt")
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student.pt")

    print(f"Results saved to: {output_dir}")

    return student, results


if __name__ == "__main__":
    main()
