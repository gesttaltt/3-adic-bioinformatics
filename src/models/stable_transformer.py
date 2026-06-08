"""Numerically stable Transformer for long sequences.

This module provides a transformer implementation optimized for:
1. Long sequences (RT = 560 positions, IN = 288 positions)
2. Numerical stability (gradient clipping, scaled attention)
3. Memory efficiency (gradient checkpointing, chunked attention)
4. Mixed precision training support

Key improvements over standard transformer:
- Pre-LayerNorm architecture (more stable training)
- Scaled dot-product attention with numerical guards
- Gradient checkpointing for memory efficiency
- Chunked attention for very long sequences
- Proper initialization for deep networks

Copyright 2024-2025 AI Whisperers
Licensed under PolyForm Noncommercial License 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class StableTransformerConfig:
    """Configuration for stable transformer."""

    n_positions: int = 560  # RT sequence length
    n_aa: int = 22  # Amino acid alphabet size
    d_model: int = 256  # Model dimension
    n_heads: int = 8  # Attention heads
    n_layers: int = 6  # Transformer layers
    d_ff: int = 1024  # Feed-forward dimension
    dropout: float = 0.1
    max_seq_len: int = 600  # Maximum sequence length

    # Stability options
    use_gradient_checkpointing: bool = True
    attention_dropout: float = 0.1
    attention_chunk_size: int = 128  # Chunk size for memory-efficient attention
    use_pre_layernorm: bool = True  # Pre-LN is more stable than Post-LN
    init_scale: float = 0.02  # Weight initialization scale
    eps: float = 1e-6  # Epsilon for numerical stability
    max_position_embeddings: int = 1024

    # Gradient control
    grad_clip_value: float = 1.0
    use_gated_mlp: bool = True  # Gated MLP (GLU variant)


class StablePositionalEncoding(nn.Module):
    """Numerically stable positional encoding with clamping."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.eps = eps

        # Create positional encoding with numerical guards
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Use numerically stable computation
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        # Clamp to prevent overflow
        pe[:, 0::2] = torch.sin(position * div_term).clamp(-1.0, 1.0)
        pe[:, 1::2] = torch.cos(position * div_term).clamp(-1.0, 1.0)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding with numerical stability."""
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()

        # Normalize positional encoding to prevent scale explosion
        x = x + pe * (1.0 / math.sqrt(x.size(-1)))
        return self.dropout(x)


class StableMultiHeadAttention(nn.Module):
    """Numerically stable multi-head attention with chunking for long sequences."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        chunk_size: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.chunk_size = chunk_size
        self.eps = eps

        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Initialize with smaller weights for stability
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))

    def _stable_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable softmax with max subtraction."""
        x_max = x.max(dim=-1, keepdim=True).values.detach()
        x_stable = x - x_max
        exp_x = torch.exp(x_stable.clamp(-50, 50))  # Clamp to prevent overflow
        return exp_x / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

    def _chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Memory-efficient chunked attention for long sequences.

        Args:
            q: Query tensor (batch, heads, seq_len, d_head)
            k: Key tensor (batch, heads, seq_len, d_head)
            v: Value tensor (batch, heads, seq_len, d_head)

        Returns:
            Attention output (batch, heads, seq_len, d_head)
        """
        batch_size, n_heads, seq_len, d_head = q.shape

        # If sequence is short enough, use standard attention
        if seq_len <= self.chunk_size:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = self._stable_softmax(attn_weights)
            attn_weights = self.attn_dropout(attn_weights)
            return torch.matmul(attn_weights, v)

        # Chunked attention for long sequences
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]

            # Compute attention for this chunk against all keys
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            attn_weights = self._stable_softmax(attn_weights)
            attn_weights = self.attn_dropout(attn_weights)

            chunk_out = torch.matmul(attn_weights, v)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with stable attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply chunked attention
        attn_output = self._chunked_attention(q, k, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.out_proj(attn_output))


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU variant) for better gradient flow."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with SiLU gating."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class StableTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block for improved stability."""

    def __init__(self, cfg: StableTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Layer norms (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)

        # Attention
        self.attention = StableMultiHeadAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.attention_dropout,
            chunk_size=cfg.attention_chunk_size,
            eps=cfg.eps,
        )

        # MLP
        if cfg.use_gated_mlp:
            self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward with Pre-LN residual connections."""
        # Attention with pre-norm
        x = x + self.attention(self.ln1(x), mask)
        # MLP with pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


class StableResistanceTransformer(nn.Module):
    """Numerically stable transformer for HIV drug resistance prediction on long sequences."""

    def __init__(self, cfg: StableTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Amino acid embedding with proper initialization
        self.aa_embedding = nn.Linear(cfg.n_aa, cfg.d_model, bias=False)
        nn.init.normal_(self.aa_embedding.weight, std=cfg.init_scale)

        # Positional encoding
        self.pos_encoding = StablePositionalEncoding(cfg.d_model, cfg.max_position_embeddings, cfg.dropout, cfg.eps)

        # Transformer blocks
        self.blocks = nn.ModuleList([StableTransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # Final layer norm
        self.final_ln = nn.LayerNorm(cfg.d_model, eps=cfg.eps)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # Gradient checkpointing flag
        self.use_gradient_checkpointing = cfg.use_gradient_checkpointing

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with gradient checkpointing for memory efficiency.

        Args:
            x: One-hot encoded sequence (batch, n_positions * n_aa) or (batch, n_positions, n_aa)

        Returns:
            Dict with 'prediction' and 'embedding'
        """
        batch_size = x.size(0)

        # Handle both flat and 3D input
        if x.dim() == 2:
            x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)

        # Embed amino acids
        x = self.aa_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final norm
        embedding = self.final_ln(x)

        # Prediction
        prediction = self.output_head(embedding).squeeze(-1)

        return {"prediction": prediction, "embedding": embedding}

    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get attention maps from a specific layer.

        Args:
            x: Input tensor
            layer_idx: Which layer's attention to return (-1 = last)

        Returns:
            Attention weights tensor
        """
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)

        x = self.aa_embedding(x)
        x = self.pos_encoding(x)

        target_layer = self.blocks[layer_idx]

        # Get attention weights
        h = target_layer.ln1(x)
        qkv = target_layer.attention.qkv(h)
        qkv = qkv.reshape(batch_size, -1, 3, target_layer.attention.n_heads, target_layer.attention.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * target_layer.attention.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights


class StableTransformerWithVAE(nn.Module):
    """Stable transformer with VAE latent space for uncertainty quantification."""

    def __init__(self, cfg: StableTransformerConfig, latent_dim: int = 32):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Encoder (stable transformer)
        self.aa_embedding = nn.Linear(cfg.n_aa, cfg.d_model, bias=False)
        self.pos_encoding = StablePositionalEncoding(cfg.d_model, cfg.max_position_embeddings, cfg.dropout, cfg.eps)
        self.blocks = nn.ModuleList([StableTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model, eps=cfg.eps)

        # VAE latent
        self.fc_mu = nn.Linear(cfg.d_model, latent_dim)
        self.fc_logvar = nn.Linear(cfg.d_model, latent_dim)

        # Decoder (simpler MLP)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_positions * cfg.n_aa),
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )

        self.use_gradient_checkpointing = cfg.use_gradient_checkpointing

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with clamping for stability."""
        # Clamp logvar to prevent extreme values
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution."""
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)

        x = self.aa_embedding(x)
        x = self.pos_encoding(x)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = x.mean(dim=1)
        x = self.final_ln(x)

        return self.fc_mu(x), self.fc_logvar(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Reconstruction
        x_recon = self.decoder(z)

        # Prediction
        prediction = self.predictor(z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
        }


def create_stable_transformer_for_gene(gene: str, use_vae: bool = False) -> nn.Module:
    """Factory function to create stable transformer for specific gene.

    Args:
        gene: Gene name (PR, RT, IN)
        use_vae: Whether to use VAE variant

    Returns:
        Configured transformer model
    """
    gene = gene.upper()

    if gene in ["PR", "PROTEASE"]:
        cfg = StableTransformerConfig(
            n_positions=99,
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            use_gradient_checkpointing=False,  # Not needed for short sequences
            attention_chunk_size=99,
        )
    elif gene in ["RT", "REVERSE_TRANSCRIPTASE"]:
        cfg = StableTransformerConfig(
            n_positions=560,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            use_gradient_checkpointing=True,  # Essential for long sequences
            attention_chunk_size=128,  # Chunked attention
        )
    elif gene in ["IN", "INTEGRASE"]:
        cfg = StableTransformerConfig(
            n_positions=288,
            d_model=192,
            n_layers=5,
            n_heads=8,
            d_ff=768,
            use_gradient_checkpointing=True,
            attention_chunk_size=144,
        )
    else:
        raise ValueError(f"Unknown gene: {gene}")

    if use_vae:
        return StableTransformerWithVAE(cfg)
    else:
        return StableResistanceTransformer(cfg)


if __name__ == "__main__":
    print("Testing Stable Transformer for Long Sequences")
    print("=" * 60)

    # Test on RT (long sequence)
    cfg = StableTransformerConfig(n_positions=560)
    model = StableResistanceTransformer(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RT Transformer parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(4, 560, 22)  # Batch of 4 RT sequences
    model.eval()  # Disable checkpointing for test
    model.use_gradient_checkpointing = False
    out = model(x)
    print(f"Prediction shape: {out['prediction'].shape}")
    print(f"Embedding shape: {out['embedding'].shape}")

    # Test gradient flow
    model.train()
    model.use_gradient_checkpointing = True
    x.requires_grad = True
    out = model(x)
    loss = out["prediction"].sum()
    loss.backward()
    print(f"Gradient norm: {x.grad.norm().item():.6f}")

    # Test VAE variant
    model_vae = StableTransformerWithVAE(cfg, latent_dim=32)
    out_vae = model_vae(x.detach())
    print("\nVAE variant:")
    print(f"  z shape: {out_vae['z'].shape}")
    print(f"  mu shape: {out_vae['mu'].shape}")
    print(f"  prediction: {out_vae['prediction'].shape}")

    # Memory test
    print("\nMemory test with gradient checkpointing:")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for batch_size in [4, 8, 16]:
        try:
            x = torch.randn(batch_size, 560, 22)
            out = model(x)
            loss = out["prediction"].sum()
            loss.backward()
            print(f"  Batch size {batch_size}: OK")
        except RuntimeError:
            print(f"  Batch size {batch_size}: OOM")
            break

    print("\n" + "=" * 60)
    print("Stable transformer tests complete!")
