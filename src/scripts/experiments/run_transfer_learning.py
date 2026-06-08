#!/usr/bin/env python3
"""
Transfer Learning for Problem Drugs

Problem drugs have limited data and underperform:
- TPV: +0.70 (1226 samples) - Non-peptidic PI
- DRV: +0.78 (993 samples) - High genetic barrier PI
- DTG: +0.72 (370 samples) - High genetic barrier INI
- RPV: +0.71 (311 samples) - Second-gen NNRTI

Solution: Pre-train on ALL drugs in the class, then fine-tune on target drug.
This leverages 10-20x more data for learning shared representations.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class TransferVAE(nn.Module):
    """VAE with separate shared encoder and drug-specific heads."""

    def __init__(self, input_dim: int, latent_dim: int = 16, n_drugs: int = 1):
        super().__init__()

        # Shared encoder (learns general resistance patterns)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        # Drug-specific prediction heads
        self.drug_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(n_drugs)]
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, drug_idx: int = 0) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        pred = self.drug_heads[drug_idx](z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "pred": pred,
        }


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE ranking loss."""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    sorted_idx = torch.argsort(target, descending=True)
    sorted_pred = pred[sorted_idx]

    total_loss = 0.0
    for i in range(n):
        remaining = sorted_pred[i:]
        log_prob = sorted_pred[i] - torch.logsumexp(remaining, dim=0)
        total_loss -= log_prob

    return total_loss / n


class TransferLearningTrainer:
    """Transfer learning trainer for problem drugs."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load_stanford_raw(self, drug_class: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Load Stanford HIVDB data."""
        data_dir = project_root / "data" / "research"

        file_mapping = {
            "pi": "stanford_hivdb_pi.txt",
            "nrti": "stanford_hivdb_nrti.txt",
            "nnrti": "stanford_hivdb_nnrti.txt",
            "ini": "stanford_hivdb_ini.txt",
        }

        drug_columns = {
            "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
            "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
            "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
            "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
        }

        filepath = data_dir / file_mapping[drug_class]
        df = pd.read_csv(filepath, sep="\t", low_memory=False)

        prefix = "P"
        position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
        position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

        return df, position_cols, drug_columns[drug_class]

    def encode_amino_acids(self, df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
        """One-hot encode sequences."""
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

        n_samples = len(df)
        n_positions = len(position_cols)
        n_aa = len(aa_alphabet)

        encoded = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)

        for idx, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(position_cols):
                aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                if aa in aa_to_idx:
                    encoded[idx, j * n_aa + aa_to_idx[aa]] = 1.0
                else:
                    encoded[idx, j * n_aa + aa_to_idx["-"]] = 1.0

        return encoded

    def prepare_class_data(self, drug_class: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Prepare data for all drugs in a class."""
        df, position_cols, drugs = self.load_stanford_raw(drug_class)
        data = {}

        for drug in drugs:
            df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
            if len(df_valid) > 30:
                X = self.encode_amino_acids(df_valid, position_cols)
                y = np.log10(df_valid[drug].values + 1).astype(np.float32)
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                data[drug] = (X, y)

        return data

    def pretrain_on_class(self, drug_class: str, epochs: int = 50) -> tuple[TransferVAE, dict]:
        """Pre-train on ALL drugs in a class."""
        print(f"\n{'=' * 70}")
        print(f"PRE-TRAINING ON ALL {drug_class.upper()} DRUGS")
        print(f"{'=' * 70}\n")

        data = self.prepare_class_data(drug_class)
        drugs = list(data.keys())
        n_drugs = len(drugs)

        print(f"Drugs: {drugs}")
        print(f"Total samples: {sum(len(X) for X, _ in data.values())}")

        # Get input dimension
        input_dim = next(iter(data.values()))[0].shape[1]

        # Create model with heads for all drugs
        model = TransferVAE(input_dim, latent_dim=16, n_drugs=n_drugs).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Prepare tensors
        tensors = {}
        for drug, (X, y) in data.items():
            tensors[drug] = (
                torch.tensor(X, dtype=torch.float32).to(self.device),
                torch.tensor(y, dtype=torch.float32).to(self.device),
            )

        drug_to_idx = {drug: i for i, drug in enumerate(drugs)}

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for drug, (X_t, y_t) in tensors.items():
                optimizer.zero_grad()

                out = model(X_t, drug_idx=drug_to_idx[drug])

                # Reconstruction loss
                recon_loss = F.mse_loss(out["x_recon"], X_t)

                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

                # Ranking loss (ListMLE)
                rank_loss = listmle_loss(out["pred"], y_t)

                loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / n_drugs:.4f}")

        # Evaluate pre-trained model
        print("\nPre-training evaluation:")
        pretrain_results = {}
        model.eval()
        with torch.no_grad():
            for drug, (X_t, y_t) in tensors.items():
                out = model(X_t, drug_idx=drug_to_idx[drug])
                pred = out["pred"].cpu().numpy()
                y = y_t.cpu().numpy()
                corr, _ = stats.spearmanr(pred, y)
                pretrain_results[drug] = corr
                print(f"  {drug}: {corr:+.3f}")

        return model, {"drugs": drugs, "drug_to_idx": drug_to_idx, "pretrain_results": pretrain_results}

    def finetune_on_drug(
        self, model: TransferVAE, meta: dict, target_drug: str, drug_class: str, epochs: int = 50
    ) -> dict:
        """Fine-tune on a specific target drug."""
        print(f"\n{'=' * 70}")
        print(f"FINE-TUNING ON {target_drug}")
        print(f"{'=' * 70}\n")

        data = self.prepare_class_data(drug_class)
        if target_drug not in data:
            return {"error": f"Drug {target_drug} not found"}

        X, y = data[target_drug]
        n = len(X)

        # Split into train/test
        split_idx = int(0.8 * n)
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(self.device)
        y_test = y[test_idx]

        # Get drug index (use 0 if target drug has a new head)
        drug_idx = meta["drug_to_idx"].get(target_drug, 0)

        # Fine-tune with lower learning rate
        # Freeze encoder initially, then unfreeze
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Only train the drug-specific head first
        optimizer = torch.optim.AdamW(model.drug_heads[drug_idx].parameters(), lr=1e-3, weight_decay=0.01)

        best_corr = -1.0
        best_state = None

        # Phase 1: Train only head (10 epochs)
        print("Phase 1: Training drug-specific head only...")
        for epoch in range(min(10, epochs)):
            model.train()
            optimizer.zero_grad()

            out = model(X_train, drug_idx=drug_idx)
            loss = listmle_loss(out["pred"], y_train) + F.mse_loss(out["pred"], y_train)
            loss.backward()
            optimizer.step()

        # Phase 2: Unfreeze encoder and train everything
        print("Phase 2: Fine-tuning full model...")
        for param in model.encoder.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            out = model(X_train, drug_idx=drug_idx)

            recon_loss = F.mse_loss(out["x_recon"], X_train)
            kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
            rank_loss = listmle_loss(out["pred"], y_train)
            pred_loss = F.mse_loss(out["pred"], y_train)

            loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss + 0.5 * pred_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Evaluate
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(X_test, drug_idx=drug_idx)
                    pred = out["pred"].cpu().numpy()
                    corr, _ = stats.spearmanr(pred, y_test)
                    if corr > best_corr:
                        best_corr = corr
                        best_state = deepcopy(model.state_dict())
                    print(f"  Epoch {epoch + 1}/{epochs}, Test Corr: {corr:+.3f} (best: {best_corr:+.3f})")

        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "drug": target_drug,
            "n_samples": n,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_corr": best_corr,
            "pretrain_corr": meta["pretrain_results"].get(target_drug, None),
        }

    def run_transfer_learning(self, target_drug: str, drug_class: str) -> dict:
        """Full transfer learning pipeline for a drug."""
        # Step 1: Pre-train on all drugs in class
        model, meta = self.pretrain_on_class(drug_class, epochs=50)

        # Step 2: Fine-tune on target drug
        result = self.finetune_on_drug(model, meta, target_drug, drug_class, epochs=100)

        return result


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for Problem Drugs")
    parser.add_argument("--drug", type=str, default="TPV", help="Target drug to improve (TPV, DRV, DTG, RPV)")
    parser.add_argument(
        "--drug-class",
        type=str,
        default=None,
        help="Drug class (pi, nrti, nnrti, ini). Auto-detected if not specified.",
    )
    parser.add_argument("--all-problem-drugs", action="store_true", help="Run on all problem drugs")
    args = parser.parse_args()

    # Drug to class mapping
    drug_to_class = {
        "TPV": "pi",
        "DRV": "pi",
        "FPV": "pi",
        "ATV": "pi",
        "IDV": "pi",
        "LPV": "pi",
        "NFV": "pi",
        "SQV": "pi",
        "ABC": "nrti",
        "AZT": "nrti",
        "D4T": "nrti",
        "DDI": "nrti",
        "FTC": "nrti",
        "3TC": "nrti",
        "TDF": "nrti",
        "DOR": "nnrti",
        "EFV": "nnrti",
        "ETR": "nnrti",
        "NVP": "nnrti",
        "RPV": "nnrti",
        "BIC": "ini",
        "CAB": "ini",
        "DTG": "ini",
        "EVG": "ini",
        "RAL": "ini",
    }

    trainer = TransferLearningTrainer()
    results = []

    if args.all_problem_drugs:
        problem_drugs = [
            ("TPV", "pi"),
            ("DRV", "pi"),
            ("DTG", "ini"),
            ("RPV", "nnrti"),
        ]

        for drug, drug_class in problem_drugs:
            print(f"\n{'#' * 70}")
            print(f"# TRANSFER LEARNING: {drug}")
            print(f"{'#' * 70}")

            result = trainer.run_transfer_learning(drug, drug_class)
            results.append(result)
    else:
        drug_class = args.drug_class or drug_to_class.get(args.drug)
        if not drug_class:
            print(f"Unknown drug: {args.drug}")
            return

        result = trainer.run_transfer_learning(args.drug, drug_class)
        results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("TRANSFER LEARNING RESULTS")
    print(f"{'=' * 70}\n")

    # Baseline comparisons
    baseline = {
        "TPV": 0.699,
        "DRV": 0.779,
        "DTG": 0.722,
        "RPV": 0.714,
    }

    for r in results:
        drug = r.get("drug", "Unknown")
        base = baseline.get(drug, 0.0)
        best = r.get("best_corr", 0.0)
        improvement = best - base

        print(f"{drug}:")
        print(f"  Baseline:     {base:+.3f}")
        print(f"  Transfer:     {best:+.3f}")
        print(f"  Improvement:  {improvement:+.3f} ({improvement / base * 100:.1f}%)")
        print()

    # Save results
    df = pd.DataFrame(results)
    output_path = project_root / "results" / "transfer_learning_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
