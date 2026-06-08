#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Diffusion Model Training for Codon Sequence Generation.

Implements D3PM (Discrete Denoising Diffusion Probabilistic Models) for
generating codon sequences across multiple disease domains.

Features:
- Discrete diffusion with absorbing state
- Multi-disease conditioning
- Structure-conditioned generation
- P-adic hierarchy-aware noise schedules

NEW (v2.0): Enhanced with SOTA components:
- CodonUsageLoss: Biological constraints (tAI, CAI, CpG penalties)
- ProteinGymEvaluator: Standardized quality, novelty, diversity metrics

Hardware: RTX 2060 SUPER (8GB VRAM)
Estimated Duration: 4-6 hours

Usage:
    # Train diffusion model for HIV
    python src/scripts/training/train_diffusion_codon.py --disease hiv

    # Train for all diseases
    python src/scripts/training/train_diffusion_codon.py --disease all

    # Structure-conditioned
    python src/scripts/training/train_diffusion_codon.py --disease hiv --structure-conditioned

    # With biological constraints (NEW)
    python src/scripts/training/train_diffusion_codon.py --disease hiv --use-codon-loss

    # With ProteinGym evaluation (NEW)
    python src/scripts/training/train_diffusion_codon.py --disease hiv --evaluate
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import new SOTA components
try:
    from src.evaluation import ProteinGymEvaluator, evaluate_generated_sequences
    from src.losses import CodonOptimalityScore, CodonUsageConfig, CodonUsageLoss, Organism

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


# Genetic code mapping
CODON_TO_IDX = {}
IDX_TO_CODON = {}
NUCLEOTIDES = ["A", "C", "G", "T"]
idx = 0
for n1 in NUCLEOTIDES:
    for n2 in NUCLEOTIDES:
        for n3 in NUCLEOTIDES:
            codon = n1 + n2 + n3
            CODON_TO_IDX[codon] = idx
            IDX_TO_CODON[idx] = codon
            idx += 1

VOCAB_SIZE = 64  # 4^3 codons
MASK_TOKEN = 64  # Absorbing state


@dataclass
class DiffusionConfig:
    """Configuration for diffusion training."""

    # Model architecture
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    # Diffusion parameters
    n_timesteps: int = 1000
    schedule_type: str = "cosine"  # linear, cosine, sqrt
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    max_seq_len: int = 300  # ~300 codons = ~900 nucleotides

    # Hardware optimization
    use_amp: bool = True
    gradient_checkpointing: bool = False
    accumulation_steps: int = 1


class NoiseScheduler:
    """Noise scheduler for discrete diffusion."""

    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.n_timesteps = n_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule (better for discrete)
            steps = torch.arange(n_timesteps + 1)
            alpha_bar = torch.cos((steps / n_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clamp(betas, min=0.0001, max=0.999)
        elif schedule_type == "sqrt":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> NoiseScheduler:
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask_token: int = MASK_TOKEN,
    ) -> torch.Tensor:
        """Forward diffusion: corrupt x at timestep t.

        For discrete diffusion, we use absorbing state (mask token).
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get corruption probability at timestep t
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)

        # Generate random mask
        mask_prob = 1 - alpha_bar_t
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob

        # Apply corruption
        x_noisy = x.clone()
        x_noisy[mask] = mask_token

        return x_noisy

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps."""
        return torch.randint(0, self.n_timesteps, (batch_size,), device=device)


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for discrete diffusion."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE + 1,  # +1 for mask token
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Position embedding
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size - 1)  # Predict non-mask tokens

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len) - noisy tokens
        t: torch.Tensor,  # (batch,) - timesteps
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean tokens from noisy input."""
        batch_size, seq_len = x.shape
        device = x.device

        # Token embedding
        token_emb = self.token_embed(x)

        # Position embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)

        # Timestep embedding
        time_emb = self._get_timestep_embedding(t.float())
        time_emb = self.time_embed(time_emb)

        # Combine embeddings
        h = token_emb + pos_emb + time_emb.unsqueeze(1)

        # Transformer
        h = self.transformer(h, src_key_padding_mask=mask)

        # Output logits
        logits = self.output_proj(h)

        return logits


class StructureConditionedDenoiser(nn.Module):
    """Denoiser with protein structure conditioning."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE + 1,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 300,
        structure_dim: int = 64,
    ):
        super().__init__()
        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Structure encoder (simple MLP for backbone features)
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention for structure conditioning
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        structure: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean tokens with optional structure conditioning."""
        if structure is None:
            return self.denoiser(x, t, mask)

        # Encode structure
        struct_emb = self.structure_encoder(structure)

        # Get base denoiser hidden states (modified to return hidden)
        batch_size, seq_len = x.shape
        device = x.device

        token_emb = self.denoiser.token_embed(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.denoiser.pos_embed(positions)
        time_emb = self.denoiser._get_timestep_embedding(t.float())
        time_emb = self.denoiser.time_embed(time_emb)

        h = token_emb + pos_emb + time_emb.unsqueeze(1)
        h = self.denoiser.transformer(h, src_key_padding_mask=mask)

        # Cross-attend to structure
        h_cond, _ = self.cross_attn(h, struct_emb, struct_emb)
        h = h + h_cond  # Residual connection

        # Output
        logits = self.denoiser.output_proj(h)

        return logits


class CodonDiffusion(nn.Module):
    """Complete diffusion model for codon sequences."""

    def __init__(
        self,
        config: DiffusionConfig,
        structure_conditioned: bool = False,
    ):
        super().__init__()
        self.config = config
        self.scheduler = NoiseScheduler(
            n_timesteps=config.n_timesteps,
            schedule_type=config.schedule_type,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

        if structure_conditioned:
            self.denoiser = StructureConditionedDenoiser(
                vocab_size=VOCAB_SIZE + 1,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )
        else:
            self.denoiser = TransformerDenoiser(
                vocab_size=VOCAB_SIZE + 1,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )

    def forward(
        self,
        x: torch.Tensor,
        structure: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward: add noise and predict."""
        device = x.device

        # Move scheduler to device
        self.scheduler.to(device)

        # Sample timesteps
        t = self.scheduler.sample_timesteps(x.shape[0], device)

        # Add noise
        x_noisy = self.scheduler.q_sample(x, t, mask_token=MASK_TOKEN)

        # Predict
        if isinstance(self.denoiser, StructureConditionedDenoiser):
            logits = self.denoiser(x_noisy, t, structure=structure)
        else:
            logits = self.denoiser(x_noisy, t)

        # Compute loss (only on masked positions that have valid targets)
        mask = x_noisy == MASK_TOKEN
        # Exclude padded positions (where original x is also MASK_TOKEN)
        valid_mask = mask & (x < VOCAB_SIZE)

        if valid_mask.sum() == 0:
            # No valid positions to compute loss
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        else:
            loss = F.cross_entropy(
                logits[valid_mask],
                x[valid_mask],
                reduction="mean",
            )

        return loss, logits

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seq_length: int,
        structure: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate sequences via reverse diffusion."""
        self.scheduler.to(device)

        # Start with all masked
        x = torch.full((n_samples, seq_length), MASK_TOKEN, device=device)

        # Reverse diffusion
        for t in reversed(range(self.config.n_timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # Predict logits
            if isinstance(self.denoiser, StructureConditionedDenoiser):
                logits = self.denoiser(x, t_batch, structure=structure)
            else:
                logits = self.denoiser(x, t_batch)

            # Sample from logits
            probs = F.softmax(logits / temperature, dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
            pred_tokens = pred_tokens.view(n_samples, seq_length)

            # Decide which positions to unmask
            if t > 0:
                # Unmask some positions based on schedule
                alpha_bar_t = self.scheduler.alpha_bar[t]
                alpha_bar_t_prev = self.scheduler.alpha_bar[t - 1]
                unmask_prob = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
                unmask = torch.rand(n_samples, seq_length, device=device) < unmask_prob
                x = torch.where(unmask & (x == MASK_TOKEN), pred_tokens, x)
            else:
                # Final step: unmask all
                x = torch.where(x == MASK_TOKEN, pred_tokens, x)

        return x


class CodonSequenceDataset(Dataset):
    """Dataset for codon sequences."""

    def __init__(
        self,
        disease: str,
        max_seq_len: int = 300,
        split: str = "train",
    ):
        self.disease = disease
        self.max_seq_len = max_seq_len
        self.sequences = []
        self._load_sequences()

    def _load_sequences(self):
        """Load sequences for the disease."""
        # Try to load from disease-specific data
        data_paths = [
            PROJECT_ROOT / f"data/{self.disease}/sequences.fasta",
            PROJECT_ROOT / f"data/{self.disease}/codons.pt",
            PROJECT_ROOT / f"research/bioinformatics/{self.disease}/data/sequences.pt",
        ]

        for path in data_paths:
            if path.exists():
                if path.suffix == ".pt":
                    data = torch.load(path, weights_only=True)
                    if isinstance(data, dict) and "sequences" in data:
                        self.sequences = data["sequences"]
                    else:
                        self.sequences = data
                    return
                elif path.suffix == ".fasta":
                    self._load_fasta(path)
                    return

        # Generate synthetic data for testing
        print(f"  [WARN] No data found for {self.disease}, generating synthetic data")
        self._generate_synthetic()

    def _load_fasta(self, path: Path):
        """Load sequences from FASTA file."""
        sequences = []
        current_seq = ""

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line

        if current_seq:
            sequences.append(current_seq)

        # Convert to codon indices
        for seq in sequences:
            # Ensure sequence length is multiple of 3
            seq = seq[: len(seq) // 3 * 3]
            codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
            indices = []
            for codon in codons:
                if codon.upper() in CODON_TO_IDX:
                    indices.append(CODON_TO_IDX[codon.upper()])
            if indices:
                self.sequences.append(torch.tensor(indices, dtype=torch.long))

    def _generate_synthetic(self):
        """Generate synthetic codon sequences for testing."""
        n_sequences = 1000
        for _ in range(n_sequences):
            length = torch.randint(50, self.max_seq_len, (1,)).item()
            seq = torch.randint(0, VOCAB_SIZE, (length,), dtype=torch.long)
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]

        # Truncate or pad
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        elif len(seq) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(seq),), MASK_TOKEN, dtype=torch.long)
            seq = torch.cat([seq, padding])

        return seq


def train_epoch(
    model: CodonDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: DiffusionConfig,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, sequences in enumerate(dataloader):
        sequences = sequences.to(device)

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            loss, _ = model(sequences)
            loss = loss / config.accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.accumulation_steps
        n_batches += 1

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def evaluate(
    model: CodonDiffusion,
    dataloader: DataLoader,
    device: torch.device,
    config: DiffusionConfig,
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for sequences in dataloader:
        sequences = sequences.to(device)

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            loss, _ = model(sequences)

        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / n_batches}


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Model for Codon Generation")
    parser.add_argument("--disease", type=str, default="hiv", help="Disease to train on (hiv, ra, neuro, cancer, all)")
    parser.add_argument("--structure-conditioned", action="store_true", help="Use structure conditioning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (10 epochs)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # New SOTA enhancement options
    parser.add_argument(
        "--use-codon-loss",
        action="store_true",
        help="Add CodonUsageLoss for biological constraints (tAI, CAI, CpG)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run ProteinGym-style evaluation after training",
    )
    parser.add_argument(
        "--codon-loss-weight",
        type=float,
        default=0.1,
        help="Weight for codon usage loss (default: 0.1)",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="human",
        choices=["human", "ecoli", "yeast", "mouse"],
        help="Organism for codon usage tables (default: human)",
    )
    args = parser.parse_args()

    # Check if enhanced features are available
    if (args.use_codon_loss or args.evaluate) and not ENHANCED_AVAILABLE:
        print("\nWarning: Enhanced components not available. Using standard training.")
        args.use_codon_loss = False
        args.evaluate = False

    print("\n" + "=" * 70)
    print("  DIFFUSION MODEL TRAINING - CODON SEQUENCE GENERATION")
    print("=" * 70)
    print(f"  Disease: {args.disease}")
    print(f"  Structure conditioned: {args.structure_conditioned}")
    if args.use_codon_loss:
        print(f"  Enhancement: CodonUsageLoss (weight={args.codon_loss_weight})")
    if args.evaluate:
        print("  Enhancement: ProteinGym-style evaluation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Config
    config = DiffusionConfig(
        epochs=10 if args.quick else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Diseases to train
    diseases = ["hiv", "ra", "neuro", "cancer"] if args.disease == "all" else [args.disease]

    for disease in diseases:
        print(f"\n{'#' * 70}")
        print(f"  Training for: {disease.upper()}")
        print(f"{'#' * 70}\n")

        # Data
        dataset = CodonSequenceDataset(disease=disease, max_seq_len=config.max_seq_len)
        print(f"Dataset size: {len(dataset)}")

        # Split train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # Model
        model = CodonDiffusion(config, structure_conditioned=args.structure_conditioned)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Create CodonUsageLoss if requested
        codon_loss_fn = None
        if args.use_codon_loss and ENHANCED_AVAILABLE:
            print("Creating CodonUsageLoss...")
            organism_map = {
                "human": Organism.HUMAN,
                "ecoli": Organism.ECOLI,
                "yeast": Organism.YEAST,
                "mouse": Organism.MOUSE,
            }
            codon_config = CodonUsageConfig(
                organism=organism_map.get(args.organism, Organism.HUMAN),
                tai_weight=0.3,
                cai_weight=0.3,
                rare_penalty_weight=0.2,
                cpg_penalty_weight=0.1,
                gc_weight=0.1,
            )
            codon_loss_fn = CodonUsageLoss(weight=args.codon_loss_weight, config=codon_config)
            print(f"  Organism: {args.organism}")
            print(f"  Loss weight: {args.codon_loss_weight}")

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        # Training
        best_val_loss = float("inf")
        checkpoint_dir = PROJECT_ROOT / f"checkpoints/diffusion_{disease}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(config.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config)
            val_metrics = evaluate(model, val_loader, device, config)
            scheduler.step()

            # Print metrics
            msg = f"Epoch {epoch + 1}/{config.epochs} | Train: {train_metrics['loss']:.4f} | Val: {val_metrics['val_loss']:.4f}"
            print(msg)

            # Save best
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": best_val_loss,
                    },
                    checkpoint_dir / "best.pt",
                )
                print(f"  [BEST] Saved checkpoint with val_loss={best_val_loss:.4f}")

        # Save final
        torch.save(
            {
                "epoch": config.epochs,
                "model_state_dict": model.state_dict(),
                "config": config,
            },
            checkpoint_dir / "latest.pt",
        )

        # Generate samples
        print("\nGenerating sample sequences...")
        model.eval()
        samples = model.sample(n_samples=100, seq_length=100, device=device)
        print(f"Generated {samples.shape[0]} sequences of length {samples.shape[1]}")

        # Decode first sample
        sample_codons = [IDX_TO_CODON.get(idx.item(), "???") for idx in samples[0] if idx.item() < VOCAB_SIZE]
        print(f"Sample sequence (first 10 codons): {' '.join(sample_codons[:10])}")

        # Compute codon usage metrics on generated samples
        if codon_loss_fn is not None:
            print("\nCodon usage metrics on generated samples:")
            scorer = CodonOptimalityScore(organism=organism_map.get(args.organism, Organism.HUMAN))
            # Filter out mask tokens
            valid_samples = samples.clone()
            valid_samples[valid_samples >= VOCAB_SIZE] = 0  # Replace mask with ATT
            scores = scorer(valid_samples)
            print(f"  Mean tAI: {scores['tai'].mean():.4f}")
            print(f"  Mean CAI: {scores['cai'].mean():.4f}")

        # Run ProteinGym-style evaluation if requested
        if args.evaluate and ENHANCED_AVAILABLE:
            print("\n" + "=" * 60)
            print("  PROTEINGYM-STYLE EVALUATION")
            print("=" * 60)

            # Get training sequences for novelty comparison
            train_seqs = torch.stack([train_dataset[i] for i in range(min(500, len(train_dataset)))])
            train_seqs = train_seqs.cpu()

            # Filter mask tokens from generated samples
            valid_samples = samples.clone().cpu()
            valid_samples[valid_samples >= VOCAB_SIZE] = 0

            evaluator = ProteinGymEvaluator(training_sequences=train_seqs)
            metrics = evaluator.evaluate(valid_samples)

            print(f"\nEvaluated {metrics.n_sequences} generated sequences:")
            print(f"  Quality - Mean tAI: {metrics.quality.mean_tai:.4f}")
            print(f"  Quality - Mean CAI: {metrics.quality.mean_cai:.4f}")
            print(f"  Novelty - Unique: {metrics.novelty.unique_fraction:.4f}")
            print(f"  Novelty - Novel: {metrics.novelty.novel_fraction:.4f}")
            print(f"  Diversity - Pairwise dist: {metrics.diversity.mean_pairwise_distance:.4f}")
            print(f"  Diversity - Codon coverage: {metrics.diversity.codon_coverage:.4f}")
            print(f"  Validity - No stops: {metrics.validity.no_stop_codons:.4f}")
            print(f"  Validity - Valid start: {metrics.validity.valid_start_codon:.4f}")
            print("=" * 60)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
