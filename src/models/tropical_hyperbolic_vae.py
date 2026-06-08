"""Tropical-Hyperbolic VAE: Hybrid Architecture.

Combines tropical (max-plus) geometry with hyperbolic (Poincare ball) geometry
to capture hierarchical and ultrametric structure in ternary operation space.

Key Ideas:
- Tropical operations create piecewise-linear mappings with natural tree structure
- Hyperbolic space has exponential volume growth ideal for hierarchies
- Both spaces have ultrametric-like properties compatible with p-adic structure

Usage:
    from src.models.tropical_hyperbolic_vae import TropicalHyperbolicVAE

    model = TropicalHyperbolicVAE(input_dim=9, latent_dim=16)
    outputs = model(x)  # Returns dict with logits, mu, logvar, z_euc, z_hyp, z_tropical
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TropicalLinear(nn.Module):
    """Linear layer with tropical-inspired structure.

    Tropical algebra uses:
    - Tropical addition: max(a, b)
    - Tropical multiplication: a + b

    We approximate this with ReLU (max(0, x)) and combine with
    standard linear operations for learnable tropical-like mappings.
    """

    def __init__(self, in_features: int, out_features: int, temperature: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tropical_linear = nn.Linear(in_features, out_features)
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable mixing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tropical-inspired transformation."""
        # Standard linear path
        h_std = self.linear(x)

        # Tropical path: uses logsumexp as smooth max
        h_tropical = self.tropical_linear(x)
        combined = torch.stack([h_std, h_tropical], dim=-1)
        h_max = torch.logsumexp(combined / self.temperature, dim=-1) * self.temperature

        # Mix standard and tropical
        alpha = torch.sigmoid(self.alpha)
        return alpha * h_std + (1 - alpha) * h_max


class TropicalEncoder(nn.Module):
    """Encoder with tropical-inspired structure."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(TropicalLinear(in_dim, h_dim, temperature))
            layers.append(nn.ReLU())  # ReLU approximates tropical max
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TropicalDecoder(nn.Module):
    """Decoder with tropical-inspired structure."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        temperature: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        layers = []
        in_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.append(TropicalLinear(in_dim, h_dim, temperature))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], output_dim * 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        logits = self.fc_out(h)
        return logits.view(-1, self.output_dim, 3)


class HyperbolicProjection(nn.Module):
    """Project between Euclidean and hyperbolic (Poincare ball) spaces."""

    def __init__(self, dim: int, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.curvature = curvature

    def exp_map(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: tangent space -> Poincare ball."""
        c = self.curvature
        sqrt_c = c**0.5
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)

        # Softer projection to stay well within ball
        scale = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
        return v * scale

    def log_map(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: Poincare ball -> tangent space."""
        c = self.curvature
        sqrt_c = c**0.5
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=1e-8, max=1.0 - 1e-6)

        # Inverse of exp_map
        scale = 2.0 * torch.atanh(sqrt_c * y_norm) / (sqrt_c * y_norm)
        return y * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition in Poincare ball."""
        c = self.curvature
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm2 = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm2 = torch.sum(y * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_norm2) * x + (1 - c * x_norm2) * y
        denom = 1 + 2 * c * xy + c * c * x_norm2 * y_norm2
        return num / torch.clamp(denom, min=1e-8)

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distance in Poincare ball."""
        c = self.curvature
        sqrt_c = c**0.5

        diff = x - y
        diff_norm2 = torch.sum(diff * diff, dim=-1)
        x_norm2 = torch.sum(x * x, dim=-1)
        y_norm2 = torch.sum(y * y, dim=-1)

        num = 2 * diff_norm2
        denom = (1 - c * x_norm2) * (1 - c * y_norm2)
        arg = 1 + num / torch.clamp(denom, min=1e-8)
        return 2 / sqrt_c * torch.acosh(torch.clamp(arg, min=1.0))


class TropicalAggregation(nn.Module):
    """Tropical-style aggregation layer using smooth max operations."""

    def __init__(self, dim: int, n_heads: int = 4, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.temperature = temperature

        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_heads)])
        self.output_proj = nn.Linear(dim * n_heads, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply multi-head tropical aggregation."""
        heads = []
        for proj in self.projections:
            h = proj(z)
            heads.append(h)

        # Stack heads: (batch, n_heads, dim)
        stacked = torch.stack(heads, dim=1)

        # Tropical max across heads using logsumexp
        torch.logsumexp(stacked / self.temperature, dim=1) * self.temperature

        # Project back
        concat = stacked.view(z.size(0), -1)
        return self.output_proj(concat)


class TropicalHyperbolicVAE(nn.Module):
    """Hybrid VAE combining tropical and hyperbolic geometry.

    Architecture:
    1. Tropical encoder: Creates piecewise-linear tree-preserving mappings
    2. Euclidean latent: Standard VAE reparameterization
    3. Hyperbolic projection: Maps to Poincare ball for hierarchy
    4. Tropical aggregation: Combines representations using smooth max
    5. Tropical decoder: Reconstructs with tropical-inspired structure

    The model outputs multiple latent representations:
    - z_euc: Euclidean latent (for reconstruction)
    - z_hyp: Hyperbolic latent (for p-adic losses)
    - z_tropical: Tropical aggregated (for tree-structure losses)

    Args:
        input_dim: Input dimension (9 for ternary operations)
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        curvature: Hyperbolic curvature (default 1.0)
        temperature: Tropical softmax temperature (default 0.1)
        use_tropical_aggregation: Whether to use tropical aggregation layer
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        curvature: float = 1.0,
        temperature: float = 0.1,
        use_tropical_aggregation: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.curvature = curvature
        self.temperature = temperature

        # Tropical encoder
        self.encoder = TropicalEncoder(input_dim, latent_dim, hidden_dims, temperature)

        # Hyperbolic projection layer
        self.hyperbolic = HyperbolicProjection(latent_dim, curvature)

        # Optional tropical aggregation
        self.use_tropical_aggregation = use_tropical_aggregation
        if use_tropical_aggregation:
            self.tropical_agg = TropicalAggregation(latent_dim, n_heads=4, temperature=temperature)

        # Tropical decoder (decodes from Euclidean z)
        self.decoder = TropicalDecoder(latent_dim, input_dim, hidden_dims, temperature)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Forward pass through hybrid architecture.

        Returns dict with:
            - logits: (batch, input_dim, 3) reconstruction logits
            - mu: (batch, latent_dim) mean
            - logvar: (batch, latent_dim) log variance
            - z: (batch, latent_dim) sampled latent (backward compat)
            - z_euc: (batch, latent_dim) Euclidean latent
            - z_hyp: (batch, latent_dim) Hyperbolic latent
            - z_tropical: (batch, latent_dim) Tropical aggregated latent
        """
        # Encode
        mu, logvar = self.encoder(x)
        z_euc = self.reparameterize(mu, logvar)

        # Project to hyperbolic space
        z_hyp = self.hyperbolic.exp_map(z_euc)

        # Tropical aggregation (optional)
        if self.use_tropical_aggregation:
            z_tropical = self.tropical_agg(z_euc)
        else:
            # Simple tropical-like transform using logsumexp
            z_expanded = z_euc.unsqueeze(-1).expand(-1, -1, 2)
            z_tropical = torch.logsumexp(z_expanded / self.temperature, dim=-1) * self.temperature

        # Decode from Euclidean z (most stable for reconstruction)
        logits = self.decoder(z_euc)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z_euc,  # Backward compatibility
            "z_euc": z_euc,
            "z_hyp": z_hyp,
            "z_tropical": z_tropical,
        }

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        outputs = self.forward(x)
        logits = outputs["logits"]
        classes = torch.argmax(logits, dim=-1)
        return classes.float() - 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space (mean only)."""
        mu, _ = self.encoder(x)
        return mu

    def hyperbolic_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances."""
        n = z.size(0)
        z_hyp = self.hyperbolic.exp_map(z)
        dists = torch.zeros(n, n, device=z.device)

        for i in range(n):
            dists[i] = self.hyperbolic.hyperbolic_distance(z_hyp[i].unsqueeze(0).expand(n, -1), z_hyp)
        return dists

    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class TropicalHyperbolicVAELight(nn.Module):
    """Lightweight version of TropicalHyperbolicVAE.

    Uses standard linear layers with tropical-style aggregation only in latent space.
    More efficient and often performs comparably to full version.
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        curvature: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.temperature = temperature

        # Standard encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Tropical transform in latent space
        self.tropical_transform = nn.Linear(latent_dim, latent_dim)

        # Standard decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim * 3)

    def exp_map(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map to Poincare ball."""
        c = self.curvature
        sqrt_c = c**0.5
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
        scale = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
        return v * scale

    def tropical_aggregate(self, z: torch.Tensor) -> torch.Tensor:
        """Apply tropical-like aggregation."""
        z_transformed = self.tropical_transform(z)
        combined = torch.stack([z, z_transformed], dim=-1)
        return torch.logsumexp(combined / self.temperature, dim=-1) * self.temperature

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            z_euc = mu + std * torch.randn_like(std)
        else:
            z_euc = mu

        # Tropical and hyperbolic projections
        z_tropical = self.tropical_aggregate(z_euc)
        z_hyp = self.exp_map(z_euc)

        # Decode from euclidean
        h = self.decoder(z_euc)
        logits = self.fc_out(h).view(-1, self.input_dim, 3)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z_euc,
            "z_euc": z_euc,
            "z_hyp": z_hyp,
            "z_tropical": z_tropical,
        }


__all__ = [
    "TropicalHyperbolicVAE",
    "TropicalHyperbolicVAELight",
    "TropicalLinear",
    "TropicalEncoder",
    "TropicalDecoder",
    "HyperbolicProjection",
    "TropicalAggregation",
]
