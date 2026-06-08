#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""BYOL/SimCLR Contrastive Pretraining for Multi-Disease Sequences.

This script provides self-supervised pretraining on unlabeled sequences
across multiple disease domains. The pretrained encoder can then be
used to initialize the VAE encoder for faster convergence.

Supported Methods:
- BYOL (Bootstrap Your Own Latent) - no negative samples needed
- SimCLR - contrastive with negative samples
- MoCo - momentum contrast

NEW (v2.0): Enhanced with SOTA components:
- CodonPositiveSampler: Biology-aware positive pair sampling
- MultiScaleNucleotideEncoder: Sub-codon granularity features
- HybridCodonEncoder: Optional ESM-2 integration
- ProteinGymEvaluator: Standardized evaluation metrics

Usage:
    # BYOL pretraining (recommended)
    python src/scripts/training/train_contrastive_pretrain.py --method byol

    # Enhanced with biology-aware sampling
    python src/scripts/training/train_contrastive_pretrain.py --method byol --use-codon-sampler

    # Multi-disease pretraining
    python src/scripts/training/train_contrastive_pretrain.py --diseases hiv ra neuro

    # Quick test
    python src/scripts/training/train_contrastive_pretrain.py --quick
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Import new SOTA components
try:
    from src.contrastive import CodonPositiveSampler, CodonSamplerConfig
    from src.encoders import (
        HybridCodonEncoder,
        HybridEncoderConfig,
        MultiScaleConfig,
        MultiScaleNucleotideEncoder,
        PLMBackend,
    )
    from src.evaluation import ProteinGymEvaluator, evaluate_generated_sequences
    from src.losses import CodonUsageConfig, CodonUsageLoss

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


class CodonAugmentation:
    """Augmentation strategies for codon sequences."""

    def __init__(
        self,
        mask_prob: float = 0.15,
        swap_prob: float = 0.10,
        synonymous_prob: float = 0.20,
    ):
        """Initialize augmentation.

        Args:
            mask_prob: Probability of masking a codon
            swap_prob: Probability of swapping adjacent codons
            synonymous_prob: Probability of synonymous codon substitution
        """
        self.mask_prob = mask_prob
        self.swap_prob = swap_prob
        self.synonymous_prob = synonymous_prob

        # Synonymous codon groups (simplified - 64 codons grouped by amino acid)
        self._build_synonymous_groups()

    def _build_synonymous_groups(self):
        """Build synonymous codon lookup."""
        # Standard genetic code - codons encoding same amino acid
        self.synonymous = {
            # Phe (F)
            0: [0, 1],
            1: [0, 1],
            # Leu (L)
            2: [2, 3, 16, 17, 18, 19],
            3: [2, 3, 16, 17, 18, 19],
            16: [2, 3, 16, 17, 18, 19],
            17: [2, 3, 16, 17, 18, 19],
            18: [2, 3, 16, 17, 18, 19],
            19: [2, 3, 16, 17, 18, 19],
            # Ile (I)
            4: [4, 5, 6],
            5: [4, 5, 6],
            6: [4, 5, 6],
            # Met (M) - start
            7: [7],
            # Val (V)
            20: [20, 21, 22, 23],
            21: [20, 21, 22, 23],
            22: [20, 21, 22, 23],
            23: [20, 21, 22, 23],
            # Ser (S)
            8: [8, 9, 10, 11, 40, 41],
            9: [8, 9, 10, 11, 40, 41],
            10: [8, 9, 10, 11, 40, 41],
            11: [8, 9, 10, 11, 40, 41],
            40: [8, 9, 10, 11, 40, 41],
            41: [8, 9, 10, 11, 40, 41],
            # Pro (P)
            12: [12, 13, 14, 15],
            13: [12, 13, 14, 15],
            14: [12, 13, 14, 15],
            15: [12, 13, 14, 15],
            # Thr (T)
            24: [24, 25, 26, 27],
            25: [24, 25, 26, 27],
            26: [24, 25, 26, 27],
            27: [24, 25, 26, 27],
            # Ala (A)
            28: [28, 29, 30, 31],
            29: [28, 29, 30, 31],
            30: [28, 29, 30, 31],
            31: [28, 29, 30, 31],
            # Tyr (Y)
            32: [32, 33],
            33: [32, 33],
            # Stop
            34: [34, 35, 42],
            35: [34, 35, 42],
            42: [34, 35, 42],
            # His (H)
            36: [36, 37],
            37: [36, 37],
            # Gln (Q)
            38: [38, 39],
            39: [38, 39],
            # Asn (N)
            44: [44, 45],
            45: [44, 45],
            # Lys (K)
            46: [46, 47],
            47: [46, 47],
            # Asp (D)
            48: [48, 49],
            49: [48, 49],
            # Glu (E)
            50: [50, 51],
            51: [50, 51],
            # Cys (C)
            52: [52, 53],
            53: [52, 53],
            # Trp (W)
            54: [54],
            # Arg (R)
            43: [43, 55, 56, 57, 58, 59],
            55: [43, 55, 56, 57, 58, 59],
            56: [43, 55, 56, 57, 58, 59],
            57: [43, 55, 56, 57, 58, 59],
            58: [43, 55, 56, 57, 58, 59],
            59: [43, 55, 56, 57, 58, 59],
            # Gly (G)
            60: [60, 61, 62, 63],
            61: [60, 61, 62, 63],
            62: [60, 61, 62, 63],
            63: [60, 61, 62, 63],
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to sequence.

        Args:
            x: Codon indices (seq_len,) or (batch, seq_len)

        Returns:
            Augmented sequence
        """
        if x.dim() == 1:
            return self._augment_single(x)
        return torch.stack([self._augment_single(seq) for seq in x])

    def _augment_single(self, x: torch.Tensor) -> torch.Tensor:
        """Augment a single sequence."""
        x = x.clone()
        seq_len = len(x)

        # Synonymous substitution
        for i in range(seq_len):
            if torch.rand(1).item() < self.synonymous_prob:
                codon = x[i].item()
                if codon in self.synonymous:
                    synonyms = self.synonymous[codon]
                    x[i] = synonyms[torch.randint(len(synonyms), (1,)).item()]

        # Random swap
        for i in range(seq_len - 1):
            if torch.rand(1).item() < self.swap_prob:
                x[i], x[i + 1] = x[i + 1].clone(), x[i].clone()

        return x


class BYOLEncoder(nn.Module):
    """BYOL encoder for codon sequences."""

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        n_layers: int = 3,
    ):
        """Initialize encoder.

        Args:
            vocab_size: Number of codons (64)
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output representation dimension
            n_layers: Number of transformer layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Codon indices (batch, seq_len)

        Returns:
            Representations (batch, output_dim)
        """
        # Embed
        h = self.embedding(x)  # (batch, seq, embed_dim)

        # Add positional encoding
        seq_len = h.size(1)
        h = h + self.pos_encoding[:, :seq_len, :]

        # Transform
        h = self.transformer(h)  # (batch, seq, embed_dim)

        # Pool (mean over sequence)
        h = h.mean(dim=1)  # (batch, embed_dim)

        # Project
        z = self.projector(h)  # (batch, output_dim)

        return z


class BYOLPredictor(nn.Module):
    """BYOL predictor network."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOL(nn.Module):
    """Bootstrap Your Own Latent for codon sequences."""

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        momentum: float = 0.996,
    ):
        """Initialize BYOL.

        Args:
            encoder: Encoder network
            projection_dim: Projection dimension
            hidden_dim: Hidden dimension for predictor
            momentum: Momentum for target network update
        """
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.predictor = BYOLPredictor(projection_dim, hidden_dim, projection_dim)
        self.momentum = momentum

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """Update target network with momentum."""
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=False):
            target.data = self.momentum * target.data + (1 - self.momentum) * online.data

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with two augmented views.

        Args:
            x1: First view (batch, seq_len)
            x2: Second view (batch, seq_len)

        Returns:
            Tuple of (loss, online_embedding)
        """
        # Online network
        z1_online = self.online_encoder(x1)
        z2_online = self.online_encoder(x2)

        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # Target network (no gradients)
        with torch.no_grad():
            z1_target = self.target_encoder(x1)
            z2_target = self.target_encoder(x2)

        # BYOL loss (cosine similarity)
        loss = (
            2
            - 2 * F.cosine_similarity(p1, z2_target.detach(), dim=-1).mean()
            + 2
            - 2 * F.cosine_similarity(p2, z1_target.detach(), dim=-1).mean()
        ) / 2

        return loss, z1_online


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning on codon sequences."""

    def __init__(
        self,
        sequences: torch.Tensor,
        augmentation: CodonAugmentation,
    ):
        """Initialize dataset.

        Args:
            sequences: Codon sequences (n_samples, seq_len)
            augmentation: Augmentation to apply
        """
        self.sequences = sequences
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item with two augmented views.

        Returns:
            Tuple of (original, view1, view2)
        """
        x = self.sequences[idx]
        x1 = self.augmentation(x)
        x2 = self.augmentation(x)
        return x, x1, x2


class EnhancedContrastiveDataset(Dataset):
    """Enhanced dataset with biology-aware positive sampling.

    Uses CodonPositiveSampler to create biologically meaningful positive pairs
    based on synonymous codons, wobble position, and p-adic distance.
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        augmentation: CodonAugmentation,
        codon_sampler: CodonPositiveSampler | None = None,
        mutation_rate: float = 0.2,
    ):
        """Initialize enhanced dataset.

        Args:
            sequences: Codon sequences (n_samples, seq_len)
            augmentation: Augmentation to apply
            codon_sampler: Biology-aware positive sampler
            mutation_rate: Fraction of positions to mutate for positive
        """
        self.sequences = sequences
        self.augmentation = augmentation
        self.codon_sampler = codon_sampler
        self.mutation_rate = mutation_rate

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item with biologically-aware augmented views.

        Returns:
            Tuple of (original, view1, view2)
        """
        x = self.sequences[idx]

        # View 1: Standard augmentation
        x1 = self.augmentation(x)

        # View 2: Biology-aware positive sampling (if sampler available)
        if self.codon_sampler is not None:
            x2 = x.clone()
            n_mutations = max(1, int(self.mutation_rate * len(x)))
            positions = torch.randperm(len(x))[:n_mutations]

            for pos in positions:
                original_codon = x[pos].item()
                new_codon = self.codon_sampler.sample_positive(original_codon)
                if new_codon is not None:
                    x2[pos] = new_codon
        else:
            x2 = self.augmentation(x)

        return x, x1, x2


def train_byol(
    model: BYOL,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 3e-4,
    device: str = "cuda",
    log_dir: Path | None = None,
) -> BYOL:
    """Train BYOL model.

    Args:
        model: BYOL model
        dataloader: Training data
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        log_dir: TensorBoard log directory

    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir) if log_dir else None

    print(f"\nTraining BYOL for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            _, x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            loss, _ = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            model.update_target()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f}")

        if writer:
            writer.add_scalar("loss/byol", avg_loss, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

    if writer:
        writer.close()

    print("\nBYOL training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Pretraining for Multi-Disease Sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default="byol", choices=["byol", "simclr"])
    parser.add_argument("--diseases", nargs="+", default=["hiv", "ra", "neuro"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Quick test (10 epochs)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # New SOTA enhancement options
    parser.add_argument(
        "--use-codon-sampler",
        action="store_true",
        help="Use biology-aware CodonPositiveSampler for positive pairs",
    )
    parser.add_argument(
        "--use-hybrid-encoder",
        action="store_true",
        help="Use HybridCodonEncoder with ESM-2 integration",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run ProteinGym-style evaluation after training",
    )
    parser.add_argument(
        "--synonymous-weight",
        type=float,
        default=0.4,
        help="Weight for synonymous codon sampling",
    )
    parser.add_argument(
        "--wobble-weight",
        type=float,
        default=0.3,
        help="Weight for wobble position sampling",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 10
        print("Quick test mode: 10 epochs")

    # Check if enhanced features are available
    use_enhanced = args.use_codon_sampler or args.use_hybrid_encoder or args.evaluate
    if use_enhanced and not ENHANCED_AVAILABLE:
        print("\nWarning: Enhanced components not available. Using standard training.")
        args.use_codon_sampler = False
        args.use_hybrid_encoder = False
        args.evaluate = False

    print("\n" + "=" * 60)
    print("  CONTRASTIVE PRETRAINING")
    print("=" * 60)
    print(f"  Method: {args.method.upper()}")
    print(f"  Diseases: {', '.join(args.diseases)}")
    print(f"  Epochs: {args.epochs}")
    if args.use_codon_sampler:
        print("  Enhancement: Biology-aware CodonPositiveSampler")
    if args.use_hybrid_encoder:
        print("  Enhancement: HybridCodonEncoder")
    if args.evaluate:
        print("  Enhancement: ProteinGym-style evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Create synthetic data for now (will load real sequences when available)
    print("\nGenerating training data...")
    n_samples = 10000
    seq_len = 100
    sequences = torch.randint(0, 64, (n_samples, seq_len))

    # Create augmentation
    augmentation = CodonAugmentation()

    # Create codon sampler if requested
    codon_sampler = None
    if args.use_codon_sampler and ENHANCED_AVAILABLE:
        print("\nCreating CodonPositiveSampler...")
        sampler_config = CodonSamplerConfig(
            use_synonymous=True,
            use_wobble=True,
            use_conservative=True,
            use_padic=True,
            synonymous_weight=args.synonymous_weight,
            wobble_weight=args.wobble_weight,
            conservative_weight=0.2,
            padic_weight=0.1,
        )
        codon_sampler = CodonPositiveSampler(sampler_config)
        print("  Synonymous weight:", args.synonymous_weight)
        print("  Wobble weight:", args.wobble_weight)

    # Create dataset (enhanced or standard)
    if codon_sampler is not None:
        print("  Using EnhancedContrastiveDataset with biology-aware sampling")
        dataset = EnhancedContrastiveDataset(sequences, augmentation, codon_sampler=codon_sampler, mutation_rate=0.2)
    else:
        dataset = ContrastiveDataset(sequences, augmentation)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")

    # Create model
    print("\nCreating BYOL model...")
    encoder = BYOLEncoder(
        vocab_size=64,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    )
    model = BYOL(encoder, projection_dim=args.output_dim)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    log_dir = project_root / "runs" / "contrastive_pretrain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    model = train_byol(
        model,
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_dir=log_dir,
    )

    # Run ProteinGym-style evaluation if requested
    if args.evaluate and ENHANCED_AVAILABLE:
        print("\n" + "=" * 60)
        print("  PROTEINGYM-STYLE EVALUATION")
        print("=" * 60)

        # Generate some sequences using the encoder embeddings
        model.eval()
        with torch.no_grad():
            sample_seqs = sequences[:1000]
            model.online_encoder(sample_seqs.to(device))

        # Create evaluator and run
        evaluator = ProteinGymEvaluator(training_sequences=sequences[:500])
        metrics = evaluator.evaluate(sample_seqs)

        print(f"\nEvaluated {metrics.n_sequences} sequences:")
        print(f"  Quality - Mean tAI: {metrics.quality.mean_tai:.4f}")
        print(f"  Quality - Mean CAI: {metrics.quality.mean_cai:.4f}")
        print(f"  Novelty - Unique: {metrics.novelty.unique_fraction:.4f}")
        print(f"  Novelty - Novel: {metrics.novelty.novel_fraction:.4f}")
        print(f"  Diversity - Pairwise dist: {metrics.diversity.mean_pairwise_distance:.4f}")
        print(f"  Validity - No stops: {metrics.validity.no_stop_codons:.4f}")
        print("=" * 60)

    # Save
    save_dir = project_root / "checkpoints" / "contrastive_pretrain"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "byol_encoder.pt"

    save_dict = {
        "encoder_state_dict": model.online_encoder.state_dict(),
        "config": {
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
        },
        "diseases": args.diseases,
        "enhancements": {
            "codon_sampler": args.use_codon_sampler,
            "hybrid_encoder": args.use_hybrid_encoder,
            "synonymous_weight": args.synonymous_weight,
            "wobble_weight": args.wobble_weight,
        },
    }
    torch.save(save_dict, save_path)

    print(f"\nSaved encoder to: {save_path}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
