#!/usr/bin/env python3
"""
Experiment Runner: Amino Acid Encoding Variants (#201-225)

Tests different amino acid encoding schemes to improve feature representation.

From Research Plan:
- #201: One-hot encoding (current baseline)
- #202: BLOSUM62 embedding
- #204: PAM250 embedding
- #214: Physicochemical properties (7-dim)
- #215: AAindex properties
- #216: Hydrophobicity scale
- #217: Charge + polarity
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# Amino acid physicochemical properties
AA_PROPERTIES = {
    # AA: [hydrophobicity, charge, polarity, size, aromaticity, h_bond_donor, h_bond_acceptor]
    "A": [1.8, 0, 0, 0.31, 0, 0, 0],
    "R": [-4.5, 1, 1, 0.77, 0, 1, 1],
    "N": [-3.5, 0, 1, 0.47, 0, 1, 1],
    "D": [-3.5, -1, 1, 0.46, 0, 0, 1],
    "C": [2.5, 0, 0, 0.40, 0, 0, 0],
    "Q": [-3.5, 0, 1, 0.56, 0, 1, 1],
    "E": [-3.5, -1, 1, 0.56, 0, 0, 1],
    "G": [-0.4, 0, 0, 0.22, 0, 0, 0],
    "H": [-3.2, 0.5, 1, 0.59, 1, 1, 1],
    "I": [4.5, 0, 0, 0.53, 0, 0, 0],
    "L": [3.8, 0, 0, 0.53, 0, 0, 0],
    "K": [-3.9, 1, 1, 0.62, 0, 1, 0],
    "M": [1.9, 0, 0, 0.57, 0, 0, 0],
    "F": [2.8, 0, 0, 0.65, 1, 0, 0],
    "P": [-1.6, 0, 0, 0.41, 0, 0, 0],
    "S": [-0.8, 0, 1, 0.32, 0, 1, 1],
    "T": [-0.7, 0, 1, 0.43, 0, 1, 1],
    "W": [-0.9, 0, 0, 0.79, 1, 1, 0],
    "Y": [-1.3, 0, 1, 0.71, 1, 1, 1],
    "V": [4.2, 0, 0, 0.47, 0, 0, 0],
    "*": [0, 0, 0, 0, 0, 0, 0],  # Stop codon
    "-": [0, 0, 0, 0, 0, 0, 0],  # Gap
    "X": [0, 0, 0, 0, 0, 0, 0],  # Unknown
}

# BLOSUM62 substitution matrix (simplified - just diagonal + common substitutions)
BLOSUM62 = {
    "A": {
        "A": 4,
        "R": -1,
        "N": -2,
        "D": -2,
        "C": 0,
        "Q": -1,
        "E": -1,
        "G": 0,
        "H": -2,
        "I": -1,
        "L": -1,
        "K": -1,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 1,
        "T": 0,
        "W": -3,
        "Y": -2,
        "V": 0,
    },
    "R": {
        "A": -1,
        "R": 5,
        "N": 0,
        "D": -2,
        "C": -3,
        "Q": 1,
        "E": 0,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -2,
        "K": 2,
        "M": -1,
        "F": -3,
        "P": -2,
        "S": -1,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -3,
    },
    "N": {
        "A": -2,
        "R": 0,
        "N": 6,
        "D": 1,
        "C": -3,
        "Q": 0,
        "E": 0,
        "G": 0,
        "H": 1,
        "I": -3,
        "L": -3,
        "K": 0,
        "M": -2,
        "F": -3,
        "P": -2,
        "S": 1,
        "T": 0,
        "W": -4,
        "Y": -2,
        "V": -3,
    },
    "D": {
        "A": -2,
        "R": -2,
        "N": 1,
        "D": 6,
        "C": -3,
        "Q": 0,
        "E": 2,
        "G": -1,
        "H": -1,
        "I": -3,
        "L": -4,
        "K": -1,
        "M": -3,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -4,
        "Y": -3,
        "V": -3,
    },
    "C": {
        "A": 0,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": 9,
        "Q": -3,
        "E": -4,
        "G": -3,
        "H": -3,
        "I": -1,
        "L": -1,
        "K": -3,
        "M": -1,
        "F": -2,
        "P": -3,
        "S": -1,
        "T": -1,
        "W": -2,
        "Y": -2,
        "V": -1,
    },
    "Q": {
        "A": -1,
        "R": 1,
        "N": 0,
        "D": 0,
        "C": -3,
        "Q": 5,
        "E": 2,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -2,
        "K": 1,
        "M": 0,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -2,
        "Y": -1,
        "V": -2,
    },
    "E": {
        "A": -1,
        "R": 0,
        "N": 0,
        "D": 2,
        "C": -4,
        "Q": 2,
        "E": 5,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -3,
        "K": 1,
        "M": -2,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "G": {
        "A": 0,
        "R": -2,
        "N": 0,
        "D": -1,
        "C": -3,
        "Q": -2,
        "E": -2,
        "G": 6,
        "H": -2,
        "I": -4,
        "L": -4,
        "K": -2,
        "M": -3,
        "F": -3,
        "P": -2,
        "S": 0,
        "T": -2,
        "W": -2,
        "Y": -3,
        "V": -3,
    },
    "H": {
        "A": -2,
        "R": 0,
        "N": 1,
        "D": -1,
        "C": -3,
        "Q": 0,
        "E": 0,
        "G": -2,
        "H": 8,
        "I": -3,
        "L": -3,
        "K": -1,
        "M": -2,
        "F": -1,
        "P": -2,
        "S": -1,
        "T": -2,
        "W": -2,
        "Y": 2,
        "V": -3,
    },
    "I": {
        "A": -1,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -1,
        "Q": -3,
        "E": -3,
        "G": -4,
        "H": -3,
        "I": 4,
        "L": 2,
        "K": -3,
        "M": 1,
        "F": 0,
        "P": -3,
        "S": -2,
        "T": -1,
        "W": -3,
        "Y": -1,
        "V": 3,
    },
    "L": {
        "A": -1,
        "R": -2,
        "N": -3,
        "D": -4,
        "C": -1,
        "Q": -2,
        "E": -3,
        "G": -4,
        "H": -3,
        "I": 2,
        "L": 4,
        "K": -2,
        "M": 2,
        "F": 0,
        "P": -3,
        "S": -2,
        "T": -1,
        "W": -2,
        "Y": -1,
        "V": 1,
    },
    "K": {
        "A": -1,
        "R": 2,
        "N": 0,
        "D": -1,
        "C": -3,
        "Q": 1,
        "E": 1,
        "G": -2,
        "H": -1,
        "I": -3,
        "L": -2,
        "K": 5,
        "M": -1,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "M": {
        "A": -1,
        "R": -1,
        "N": -2,
        "D": -3,
        "C": -1,
        "Q": 0,
        "E": -2,
        "G": -3,
        "H": -2,
        "I": 1,
        "L": 2,
        "K": -1,
        "M": 5,
        "F": 0,
        "P": -2,
        "S": -1,
        "T": -1,
        "W": -1,
        "Y": -1,
        "V": 1,
    },
    "F": {
        "A": -2,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -2,
        "Q": -3,
        "E": -3,
        "G": -3,
        "H": -1,
        "I": 0,
        "L": 0,
        "K": -3,
        "M": 0,
        "F": 6,
        "P": -4,
        "S": -2,
        "T": -2,
        "W": 1,
        "Y": 3,
        "V": -1,
    },
    "P": {
        "A": -1,
        "R": -2,
        "N": -2,
        "D": -1,
        "C": -3,
        "Q": -1,
        "E": -1,
        "G": -2,
        "H": -2,
        "I": -3,
        "L": -3,
        "K": -1,
        "M": -2,
        "F": -4,
        "P": 7,
        "S": -1,
        "T": -1,
        "W": -4,
        "Y": -3,
        "V": -2,
    },
    "S": {
        "A": 1,
        "R": -1,
        "N": 1,
        "D": 0,
        "C": -1,
        "Q": 0,
        "E": 0,
        "G": 0,
        "H": -1,
        "I": -2,
        "L": -2,
        "K": 0,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 4,
        "T": 1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "T": {
        "A": 0,
        "R": -1,
        "N": 0,
        "D": -1,
        "C": -1,
        "Q": -1,
        "E": -1,
        "G": -2,
        "H": -2,
        "I": -1,
        "L": -1,
        "K": -1,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 1,
        "T": 5,
        "W": -2,
        "Y": -2,
        "V": 0,
    },
    "W": {
        "A": -3,
        "R": -3,
        "N": -4,
        "D": -4,
        "C": -2,
        "Q": -2,
        "E": -3,
        "G": -2,
        "H": -2,
        "I": -3,
        "L": -2,
        "K": -3,
        "M": -1,
        "F": 1,
        "P": -4,
        "S": -3,
        "T": -2,
        "W": 11,
        "Y": 2,
        "V": -3,
    },
    "Y": {
        "A": -2,
        "R": -2,
        "N": -2,
        "D": -3,
        "C": -2,
        "Q": -1,
        "E": -2,
        "G": -3,
        "H": 2,
        "I": -1,
        "L": -1,
        "K": -2,
        "M": -1,
        "F": 3,
        "P": -3,
        "S": -2,
        "T": -2,
        "W": 2,
        "Y": 7,
        "V": -1,
    },
    "V": {
        "A": 0,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -1,
        "Q": -2,
        "E": -2,
        "G": -3,
        "H": -3,
        "I": 3,
        "L": 1,
        "K": -2,
        "M": 1,
        "F": -1,
        "P": -2,
        "S": -2,
        "T": 0,
        "W": -3,
        "Y": -1,
        "V": 4,
    },
}

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class AAEncoders:
    """Collection of amino acid encoding methods."""

    @staticmethod
    def one_hot(sequence: str, seq_length: int) -> np.ndarray:
        """#201: Standard one-hot encoding (21 channels including X/gap)."""
        aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS + ["X"])}
        encoding = np.zeros((seq_length, 21))

        for i, aa in enumerate(sequence[:seq_length]):
            idx = aa_to_idx.get(aa, 20)  # Unknown -> 20
            encoding[i, idx] = 1.0

        return encoding.flatten()

    @staticmethod
    def blosum62(sequence: str, seq_length: int) -> np.ndarray:
        """#202: BLOSUM62 substitution matrix encoding (20-dim per position)."""
        encoding = np.zeros((seq_length, 20))

        for i, aa in enumerate(sequence[:seq_length]):
            if aa in BLOSUM62:
                for j, ref_aa in enumerate(AMINO_ACIDS):
                    encoding[i, j] = BLOSUM62[aa].get(ref_aa, 0) / 10.0  # Normalize

        return encoding.flatten()

    @staticmethod
    def physicochemical(sequence: str, seq_length: int) -> np.ndarray:
        """#214: Physicochemical properties (7-dim per position)."""
        encoding = np.zeros((seq_length, 7))

        for i, aa in enumerate(sequence[:seq_length]):
            props = AA_PROPERTIES.get(aa, AA_PROPERTIES["X"])
            encoding[i] = props

        # Normalize
        encoding = (encoding - encoding.mean(axis=0)) / (encoding.std(axis=0) + 1e-8)
        return encoding.flatten()

    @staticmethod
    def hydrophobicity(sequence: str, seq_length: int) -> np.ndarray:
        """#216: Hydrophobicity scale only (1-dim per position)."""
        encoding = np.zeros((seq_length, 1))

        for i, aa in enumerate(sequence[:seq_length]):
            props = AA_PROPERTIES.get(aa, AA_PROPERTIES["X"])
            encoding[i, 0] = props[0]  # Hydrophobicity is first property

        return encoding.flatten()

    @staticmethod
    def charge_polarity(sequence: str, seq_length: int) -> np.ndarray:
        """#217: Charge + polarity (2-dim per position)."""
        encoding = np.zeros((seq_length, 2))

        for i, aa in enumerate(sequence[:seq_length]):
            props = AA_PROPERTIES.get(aa, AA_PROPERTIES["X"])
            encoding[i, 0] = props[1]  # Charge
            encoding[i, 1] = props[2]  # Polarity

        return encoding.flatten()

    @staticmethod
    def combined(sequence: str, seq_length: int) -> np.ndarray:
        """Combined: One-hot + Physicochemical (28-dim per position)."""
        one_hot = AAEncoders.one_hot(sequence, seq_length).reshape(seq_length, -1)
        physico = AAEncoders.physicochemical(sequence, seq_length).reshape(seq_length, -1)
        combined = np.concatenate([one_hot, physico], axis=1)
        return combined.flatten()

    @staticmethod
    def blosum_physico(sequence: str, seq_length: int) -> np.ndarray:
        """Combined: BLOSUM62 + Physicochemical (27-dim per position)."""
        blosum = AAEncoders.blosum62(sequence, seq_length).reshape(seq_length, -1)
        physico = AAEncoders.physicochemical(sequence, seq_length).reshape(seq_length, -1)
        combined = np.concatenate([blosum, physico], axis=1)
        return combined.flatten()


class AAEncodingExperimentRunner:
    """Runs AA encoding experiments systematically."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load_stanford_raw(self, drug_class: str = "pi") -> tuple[pd.DataFrame, list[str], list[str]]:
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
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        prefix = "P"
        position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
        position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

        return df, position_cols, drug_columns[drug_class]

    def load_sequences(self, drug_class: str = "pi") -> dict[str, tuple[list[str], np.ndarray, int]]:
        """Load raw sequences for re-encoding."""
        data = {}

        try:
            df, position_cols, drugs = self.load_stanford_raw(drug_class)
        except FileNotFoundError as e:
            print(f"  {e}")
            return data

        for drug in drugs:
            try:
                df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
                if len(df_valid) > 50:
                    # Extract sequences as strings
                    sequences = []
                    for _, row in df_valid.iterrows():
                        seq = "".join([str(row[col]).upper() if pd.notna(row[col]) else "-" for col in position_cols])
                        sequences.append(seq)

                    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                    data[drug] = (sequences, y, len(position_cols))
                    print(f"  Loaded {drug}: {len(sequences)} sequences")
            except Exception as e:
                print(f"  Could not load {drug}: {e}")

        return data

    def encode_sequences(self, sequences: list[str], encoder_fn: callable, seq_length: int = 99) -> np.ndarray:
        """Encode sequences using specified encoder."""
        encoded = []
        for seq in sequences:
            # Handle string or array input
            if isinstance(seq, np.ndarray):
                seq = "".join(seq)
            enc = encoder_fn(seq, seq_length)
            encoded.append(enc)
        return np.array(encoded)

    def create_model(self, input_dim: int) -> nn.Module:
        """Create a predictor model."""
        return nn.Sequential(
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
            nn.Linear(64, 1),
        ).to(self.device)

    def train_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
    ) -> float:
        """Train and evaluate model."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        best_corr = -1.0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            pred = model(X_train_t).squeeze()
            loss = F.mse_loss(pred, y_train_t)

            # Add ranking loss
            if len(pred) > 1:
                pred_mean = pred - pred.mean()
                target_mean = y_train_t - y_train_t.mean()
                cov = (pred_mean * target_mean).sum()
                pred_std = torch.sqrt((pred_mean**2).sum() + 1e-8)
                target_std = torch.sqrt((target_mean**2).sum() + 1e-8)
                corr_loss = 1 - cov / (pred_std * target_std)
                loss = loss + 0.5 * corr_loss

            if torch.isnan(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)

            # Evaluate
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t).squeeze().cpu().numpy()
                    corr, _ = stats.spearmanr(test_pred, y_test)
                    if not np.isnan(corr):
                        best_corr = max(best_corr, corr)

        return best_corr

    def run_experiment(self, drug_class: str = "pi") -> pd.DataFrame:
        """Run all AA encoding experiments."""
        print(f"\n{'=' * 70}")
        print(f"AA ENCODING EXPERIMENTS - {drug_class.upper()}")
        print(f"{'=' * 70}\n")

        data = self.load_sequences(drug_class)

        # Define encoders to test
        encoders = {
            "#201 OneHot": AAEncoders.one_hot,
            "#202 BLOSUM62": AAEncoders.blosum62,
            "#214 Physicochemical": AAEncoders.physicochemical,
            "#216 Hydrophobicity": AAEncoders.hydrophobicity,
            "#217 ChargePolarity": AAEncoders.charge_polarity,
            "Combined (OH+Phys)": AAEncoders.combined,
            "Combined (BLOSUM+Phys)": AAEncoders.blosum_physico,
        }

        results = []

        for drug, (sequences, y, seq_length) in data.items():
            print(f"\n--- Drug: {drug} ({len(sequences)} samples) ---")

            # Train/test split
            n = len(sequences)
            split_idx = int(0.8 * n)
            indices = np.random.permutation(n)
            train_idx, test_idx = indices[:split_idx], indices[split_idx:]

            y_train, y_test = y[train_idx], y[test_idx]

            for enc_name, encoder_fn in encoders.items():
                print(f"  Testing {enc_name}...", end=" ")

                try:
                    # Encode sequences
                    X_all = self.encode_sequences(sequences, encoder_fn, seq_length)
                    X_train = X_all[train_idx]
                    X_test = X_all[test_idx]

                    # Create and train model
                    model = self.create_model(X_all.shape[1])
                    corr = self.train_model(model, X_train, y_train, X_test, y_test)

                    results.append(
                        {
                            "drug": drug,
                            "drug_class": drug_class,
                            "encoder": enc_name,
                            "input_dim": X_all.shape[1],
                            "best_corr": corr,
                            "n_samples": len(sequences),
                        }
                    )
                    print(f"corr = {corr:+.3f} (dim={X_all.shape[1]})")

                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append(
                        {
                            "drug": drug,
                            "drug_class": drug_class,
                            "encoder": enc_name,
                            "best_corr": np.nan,
                            "error": str(e),
                        }
                    )

        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="AA Encoding Experiments")
    parser.add_argument("--drug-class", type=str, default="pi", choices=["pi", "nrti", "nnrti", "ini", "all"])
    args = parser.parse_args()

    runner = AAEncodingExperimentRunner()

    if args.drug_class == "all":
        all_results = []
        for dc in ["pi", "nrti", "nnrti", "ini"]:
            results = runner.run_experiment(dc)
            all_results.append(results)
        results = pd.concat(all_results, ignore_index=True)
    else:
        results = runner.run_experiment(args.drug_class)

    # Save results
    output_path = project_root / "results" / "aa_encoding_experiments.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY - Best Encoder per Drug")
    print(f"{'=' * 70}\n")

    for drug in results["drug"].unique():
        drug_results = results[results["drug"] == drug].dropna(subset=["best_corr"])
        if len(drug_results) > 0:
            best = drug_results.loc[drug_results["best_corr"].idxmax()]
            print(f"{drug}: {best['encoder']} -> {best['best_corr']:+.3f}")

    # Overall average
    print(f"\n{'=' * 70}")
    print("OVERALL - Average Correlation by Encoder")
    print(f"{'=' * 70}\n")

    avg_by_enc = results.groupby("encoder")["best_corr"].mean().sort_values(ascending=False)
    for enc, avg_corr in avg_by_enc.items():
        print(f"{enc}: {avg_corr:+.3f}")


if __name__ == "__main__":
    main()
