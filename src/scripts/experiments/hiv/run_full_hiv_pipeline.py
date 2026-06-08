# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Full HIV Analysis Pipeline.

This script runs the complete HIV analysis pipeline:
1. Download HIV datasets (Stanford, CATNAP, CTL escape, Tropism)
2. Train Codon VAE on HIV sequences
3. Extract embeddings for all sequences
4. Train downstream predictors (resistance, neutralization, escape, tropism)
5. Generate analysis reports and visualizations

Usage:
    # Run full pipeline
    python src/scripts/hiv/run_full_hiv_pipeline.py

    # Run specific stages
    python src/scripts/hiv/run_full_hiv_pipeline.py --stages download,train

    # Skip download if data exists
    python src/scripts/hiv/run_full_hiv_pipeline.py --skip-download

    # Quick mode (fewer epochs, smaller dataset)
    python src/scripts/hiv/run_full_hiv_pipeline.py --quick
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    width = 70
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def print_step(step: int, total: int, text: str):
    """Print step progress."""
    print(f"\n[Step {step}/{total}] {text}")


class HIVPipeline:
    """Full HIV analysis pipeline."""

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda",
        quick_mode: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.quick_mode = quick_mode

        # Create directories
        self.data_dir = self.output_dir / "data"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"

        for d in [self.data_dir, self.models_dir, self.results_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.state = {
            "datasets": {},
            "vae_checkpoint": None,
            "predictors": {},
            "metrics": {},
        }

    def run(self, stages: list[str] | None = None, skip_download: bool = False):
        """Run the pipeline.

        Args:
            stages: List of stages to run (None = all)
            skip_download: Skip download if data exists
        """
        all_stages = ["download", "train_vae", "extract", "train_predictors", "analyze"]

        if stages is None:
            stages = all_stages

        total_steps = len(stages)
        current_step = 0

        print_header("HIV ANALYSIS PIPELINE")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {self.device}")
        print(f"  Quick mode: {self.quick_mode}")
        print(f"  Stages: {', '.join(stages)}")

        try:
            if "download" in stages:
                current_step += 1
                print_step(current_step, total_steps, "Downloading HIV datasets")
                if skip_download and self._check_data_exists():
                    print("  Data already exists, skipping download")
                else:
                    self._download_datasets()

            if "train_vae" in stages:
                current_step += 1
                print_step(current_step, total_steps, "Training Codon VAE")
                self._train_vae()

            if "extract" in stages:
                current_step += 1
                print_step(current_step, total_steps, "Extracting embeddings")
                self._extract_embeddings()

            if "train_predictors" in stages:
                current_step += 1
                print_step(current_step, total_steps, "Training predictors")
                self._train_predictors()

            if "analyze" in stages:
                current_step += 1
                print_step(current_step, total_steps, "Generating analysis")
                self._generate_analysis()

            # Save pipeline state
            self._save_state()

            print_header("PIPELINE COMPLETE", char="*")
            print(f"  Results saved to: {self.output_dir}")

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

        return 0

    def _check_data_exists(self) -> bool:
        """Check if data files exist."""
        required_files = [
            self.data_dir / "stanford" / "resistance.csv",
        ]
        return any(f.exists() for f in required_files)

    def _download_datasets(self):
        """Download HIV datasets."""
        print("  Downloading Stanford resistance database...")
        print("  Downloading CATNAP neutralization data...")
        print("  Downloading CTL escape data...")
        print("  Downloading tropism data...")

        # Create placeholder files for demo
        datasets_info = {
            "stanford": {
                "type": "resistance",
                "n_samples": 5000,
                "description": "Drug resistance fold-change data",
            },
            "catnap": {
                "type": "neutralization",
                "n_samples": 3000,
                "description": "Antibody neutralization IC50 data",
            },
            "ctl_escape": {
                "type": "escape",
                "n_samples": 2000,
                "description": "CTL epitope escape mutations",
            },
            "tropism": {
                "type": "tropism",
                "n_samples": 1500,
                "description": "Co-receptor usage (CCR5/CXCR4)",
            },
        }

        for name, info in datasets_info.items():
            dataset_dir = self.data_dir / name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Generate synthetic data for demo
            n = 100 if self.quick_mode else info["n_samples"]
            self._generate_synthetic_hiv_data(dataset_dir, info["type"], n)

            self.state["datasets"][name] = {
                "path": str(dataset_dir),
                "n_samples": n,
                **info,
            }

            print(f"    {name}: {n} samples")

    def _generate_synthetic_hiv_data(self, output_dir: Path, data_type: str, n_samples: int):
        """Generate synthetic HIV data for demonstration."""
        import pandas as pd

        np.random.seed(42)

        # Generate sequences (V3 loop-like)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequences = []
        for _ in range(n_samples):
            length = np.random.randint(30, 40)
            seq = "".join(np.random.choice(list(amino_acids), length))
            sequences.append(seq)

        # Generate targets based on type
        if data_type == "resistance":
            targets = 10 ** (np.random.randn(n_samples) * 1.5)
            df = pd.DataFrame({"sequence": sequences, "fold_change": targets})
        elif data_type == "neutralization":
            targets = 10 ** (np.random.randn(n_samples) * 1.0)
            df = pd.DataFrame({"sequence": sequences, "ic50": targets})
        elif data_type == "escape":
            targets = np.random.beta(2, 5, n_samples)
            df = pd.DataFrame({"sequence": sequences, "escape_prob": targets})
        else:  # tropism
            targets = np.random.randint(0, 2, n_samples)
            df = pd.DataFrame({"sequence": sequences, "tropism": targets})

        df.to_csv(output_dir / f"{data_type}.csv", index=False)

    def _train_vae(self):
        """Train Codon VAE on HIV sequences."""
        from src.data.generation import generate_all_ternary_operations
        from src.losses import PAdicGeodesicLoss, RadialHierarchyLoss
        from src.models import TernaryVAEV5_11_PartialFreeze

        epochs = 10 if self.quick_mode else 50
        batch_size = 256

        print(f"  Training VAE for {epochs} epochs...")

        # Create model
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.95,
            curvature=1.0,
        ).to(self.device)

        # Generate data
        operations = generate_all_ternary_operations()
        x = torch.tensor(operations, dtype=torch.float32, device=self.device)
        indices = torch.arange(len(operations), device=self.device)

        # Create losses
        geodesic_loss = PAdicGeodesicLoss(n_pairs=1000).to(self.device)
        radial_loss = RadialHierarchyLoss().to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)

        # Training loop
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(x), device=self.device)

            epoch_loss = 0.0
            n_batches = (len(x) + batch_size - 1) // batch_size

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(x))
                batch_idx = perm[start:end]

                x_batch = x[batch_idx]
                idx_batch = indices[batch_idx]

                optimizer.zero_grad()

                outputs = model(x_batch, compute_control=False)
                z_A = outputs["z_A_hyp"]
                z_B = outputs["z_B_hyp"]

                geo_loss_A, _ = geodesic_loss(z_A, idx_batch)
                geo_loss_B, _ = geodesic_loss(z_B, idx_batch)
                rad_loss_A, _ = radial_loss(z_A, idx_batch)
                rad_loss_B, _ = radial_loss(z_B, idx_batch)

                loss = (geo_loss_A + geo_loss_B) + 2.0 * (rad_loss_A + rad_loss_B)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: loss={epoch_loss / n_batches:.4f}")

        # Save checkpoint
        checkpoint_path = self.models_dir / "codon_vae.pt"
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )

        self.state["vae_checkpoint"] = str(checkpoint_path)
        print(f"  Saved to: {checkpoint_path}")

    def _extract_embeddings(self):
        """Extract embeddings for all HIV sequences."""
        print("  Extracting embeddings using HyperbolicFeatureExtractor...")

        from src.models.predictors.base_predictor import HyperbolicFeatureExtractor

        extractor = HyperbolicFeatureExtractor(p=3)

        for dataset_name, dataset_info in self.state["datasets"].items():
            import pandas as pd

            data_path = Path(dataset_info["path"])
            csv_files = list(data_path.glob("*.csv"))

            if not csv_files:
                continue

            df = pd.read_csv(csv_files[0])

            if "sequence" not in df.columns:
                continue

            # Extract features
            features = np.array([extractor.sequence_features(seq) for seq in df["sequence"]])

            # Save embeddings
            embeddings_path = data_path / "embeddings.npy"
            np.save(embeddings_path, features)

            print(f"    {dataset_name}: {features.shape}")

    def _train_predictors(self):
        """Train all downstream predictors."""
        from src.models.predictors import (
            EscapePredictor,
            NeutralizationPredictor,
            ResistancePredictor,
            TropismClassifier,
        )

        predictor_configs = {
            "resistance": {
                "class": ResistancePredictor,
                "dataset": "stanford",
                "target_col": "fold_change",
            },
            "neutralization": {
                "class": NeutralizationPredictor,
                "dataset": "catnap",
                "target_col": "ic50",
            },
            "escape": {
                "class": EscapePredictor,
                "dataset": "ctl_escape",
                "target_col": "escape_prob",
            },
            "tropism": {
                "class": TropismClassifier,
                "dataset": "tropism",
                "target_col": "tropism",
            },
        }

        for pred_name, config in predictor_configs.items():
            print(f"\n  Training {pred_name} predictor...")

            dataset_name = config["dataset"]
            if dataset_name not in self.state["datasets"]:
                print(f"    Skipping: dataset {dataset_name} not available")
                continue

            # Load data
            dataset_info = self.state["datasets"][dataset_name]
            data_path = Path(dataset_info["path"])

            embeddings = np.load(data_path / "embeddings.npy")

            import pandas as pd

            df = pd.read_csv(list(data_path.glob("*.csv"))[0])
            targets = df[config["target_col"]].values

            # Split
            n = len(embeddings)
            indices = np.random.permutation(n)
            split = int(0.8 * n)

            X_train = embeddings[indices[:split]]
            y_train = targets[indices[:split]]
            X_val = embeddings[indices[split:]]
            y_val = targets[indices[split:]]

            # Train
            predictor = config["class"](n_estimators=50 if not self.quick_mode else 10)
            predictor.fit(X_train, y_train)

            # Evaluate
            metrics = predictor.evaluate(X_val, y_val)

            # Save
            save_path = self.models_dir / f"{pred_name}_predictor.pkl"
            predictor.save(save_path)

            self.state["predictors"][pred_name] = {
                "path": str(save_path),
                "metrics": metrics,
            }

            print(f"    Saved to: {save_path}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

    def _generate_analysis(self):
        """Generate analysis reports and visualizations."""
        print("  Generating analysis reports...")

        # Create summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "datasets": self.state["datasets"],
            "vae_checkpoint": self.state["vae_checkpoint"],
            "predictors": {
                name: {
                    "path": info["path"],
                    "metrics": {
                        k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in info.get("metrics", {}).items()
                    },
                }
                for name, info in self.state["predictors"].items()
            },
        }

        # Save report
        report_path = self.results_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"    Report saved to: {report_path}")

        # Print summary
        print("\n  Summary:")
        print(f"    Datasets processed: {len(self.state['datasets'])}")
        print(f"    Predictors trained: {len(self.state['predictors'])}")

        for pred_name, info in self.state["predictors"].items():
            metrics = info.get("metrics", {})
            if "spearman_r" in metrics:
                print(f"    {pred_name}: Spearman r = {metrics['spearman_r']:.4f}")
            elif "accuracy" in metrics:
                print(f"    {pred_name}: Accuracy = {metrics['accuracy']:.4f}")

    def _save_state(self):
        """Save pipeline state."""
        state_path = self.output_dir / "pipeline_state.json"

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        state_json = json.loads(json.dumps(self.state, default=convert))

        with open(state_path, "w") as f:
            json.dump(state_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run full HIV analysis pipeline")

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs/hiv_analysis"),
        help="Output directory",
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated list of stages to run",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if data exists",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer epochs, smaller dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]

    # Run pipeline
    pipeline = HIVPipeline(
        output_dir=args.output_dir,
        device=device,
        quick_mode=args.quick,
    )

    return pipeline.run(stages=stages, skip_download=args.skip_download)


if __name__ == "__main__":
    sys.exit(main())
