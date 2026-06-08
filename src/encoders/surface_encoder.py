# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""MaSIF-style surface encoder for protein analysis.

This module implements a surface-based deep learning approach for
protein representation, inspired by MaSIF (Molecular Surface Interaction
Fingerprinting). The p-adic framework is integrated to capture
hierarchical surface relationships.

Key components:
- Surface patch extraction and encoding
- Geodesic convolution on molecular surfaces
- P-adic distance integration for surface point relationships
- Chemical feature extraction (hydrophobicity, charge, etc.)

References:
- 2020_Gainza_MaSIF_Surfaces.md: Original MaSIF method
- 2012_Scalco_Protein_Ultrametricity.md: Ultrametric geometry
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurfacePoint:
    """Represents a point on the molecular surface."""

    position: torch.Tensor  # (3,) xyz coordinates
    normal: torch.Tensor  # (3,) surface normal
    curvature: float  # Local curvature
    charge: float  # Electrostatic charge
    hydrophobicity: float  # Hydrophobicity score


@dataclass
class SurfacePatch:
    """A local patch on the molecular surface."""

    center: torch.Tensor  # (3,) center position
    points: torch.Tensor  # (n_points, 3) local point coordinates
    normals: torch.Tensor  # (n_points, 3) surface normals
    features: torch.Tensor  # (n_points, n_features) chemical features
    geodesic_distances: torch.Tensor  # (n_points,) distances from center


@dataclass
class SurfaceEncoderOutput:
    """Output from surface encoder."""

    patch_embeddings: torch.Tensor  # (batch, n_patches, embed_dim)
    surface_embedding: torch.Tensor  # (batch, embed_dim)
    attention_weights: torch.Tensor  # (batch, n_patches)
    padic_structure: torch.Tensor | None  # Optional p-adic distances


# Chemical property scales
HYDROPHOBICITY_SCALE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

CHARGE_SCALE = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0.5,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
}

VOLUME_SCALE = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}


class GeodesicConv(nn.Module):
    """Geodesic convolution layer for surface features.

    Performs convolution on the molecular surface using geodesic
    distances to define local neighborhoods.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_rings: int = 5,
        n_orientations: int = 8,
    ):
        """Initialize geodesic convolution.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            n_rings: Number of radial rings
            n_orientations: Number of angular bins
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_rings = n_rings
        self.n_orientations = n_orientations

        # Learnable filter bank
        self.filters = nn.Parameter(torch.randn(n_rings, n_orientations, in_channels, out_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        features: torch.Tensor,
        geodesic_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Apply geodesic convolution.

        Args:
            features: Input features (batch, n_points, in_channels)
            geodesic_coords: Geodesic coordinates (batch, n_points, 2)
                            [radial, angular]

        Returns:
            Output features (batch, n_points, out_channels)
        """
        batch_size, n_points, _ = features.shape

        # Discretize geodesic coordinates
        radial = geodesic_coords[:, :, 0]  # (batch, n_points)
        angular = geodesic_coords[:, :, 1]  # (batch, n_points)

        # Normalize to ring/orientation indices
        ring_idx = (radial * self.n_rings).long().clamp(0, self.n_rings - 1)
        orient_idx = (angular * self.n_orientations / (2 * 3.14159)).long()
        orient_idx = orient_idx.clamp(0, self.n_orientations - 1)

        # Gather filters for each point
        output = torch.zeros(batch_size, n_points, self.out_channels, device=features.device)

        for b in range(batch_size):
            for p in range(n_points):
                r = int(ring_idx[b, p].item())
                o = int(orient_idx[b, p].item())
                filter_weights = self.filters[r, o]  # (in_channels, out_channels)
                output[b, p] = features[b, p] @ filter_weights + self.bias

        return output


class SurfaceFeatureExtractor(nn.Module):
    """Extracts chemical features for surface points.

    Given residue assignments and positions, computes local
    chemical environment features.
    """

    def __init__(self, feature_dim: int = 16):
        """Initialize feature extractor.

        Args:
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Raw features: hydrophobicity, charge, volume, curvature
        raw_dim = 4

        self.feature_proj = nn.Sequential(
            nn.Linear(raw_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def extract_residue_features(self, residue: str) -> torch.Tensor:
        """Extract features for a single residue.

        Args:
            residue: Single-letter amino acid code

        Returns:
            Feature tensor (4,)
        """
        residue = residue.upper()
        hydro = HYDROPHOBICITY_SCALE.get(residue, 0.0) / 4.5  # Normalize
        charge = CHARGE_SCALE.get(residue, 0.0)
        volume = VOLUME_SCALE.get(residue, 100.0) / 200.0  # Normalize

        return torch.tensor([hydro, charge, volume, 0.0])  # Curvature added later

    def forward(
        self,
        residue_features: torch.Tensor,
        curvatures: torch.Tensor,
    ) -> torch.Tensor:
        """Extract surface features.

        Args:
            residue_features: Raw residue features (batch, n_points, 3)
            curvatures: Local curvatures (batch, n_points)

        Returns:
            Surface features (batch, n_points, feature_dim)
        """
        # Combine with curvature
        combined = torch.cat([residue_features, curvatures.unsqueeze(-1)], dim=-1)
        return self.feature_proj(combined)


class PAdicSurfaceAttention(nn.Module):
    """Attention mechanism using p-adic distances on surfaces.

    Computes attention weights based on p-adic structure of
    surface point indices, capturing hierarchical relationships.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        p: int = 3,
    ):
        """Initialize p-adic surface attention.

        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.p = p
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable p-adic distance scaling
        self.padic_scale = nn.Parameter(torch.ones(1))

    def compute_padic_distances(self, n_points: int, device: torch.device) -> torch.Tensor:
        """Compute p-adic distance matrix for point indices.

        Args:
            n_points: Number of surface points
            device: Device for tensor

        Returns:
            Distance matrix (n_points, n_points)
        """
        indices = torch.arange(n_points, device=device)
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))

        # Compute valuations
        valuations = torch.zeros_like(diff, dtype=torch.float)
        for k in range(1, 10):
            divisible = (diff % (self.p**k) == 0) & (diff > 0)
            valuations[divisible] = k

        distances = torch.where(
            diff == 0,
            torch.zeros_like(valuations),
            torch.pow(float(self.p), -valuations),
        )

        return distances

    def forward(
        self,
        x: torch.Tensor,
        padic_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply p-adic surface attention.

        Args:
            x: Input features (batch, n_points, embed_dim)
            padic_bias: Optional precomputed p-adic distances

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, n_points, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, n_points, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, n_points, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, n_points, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, n_points, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Add p-adic bias
        if padic_bias is None:
            padic_bias = self.compute_padic_distances(n_points, x.device)
        padic_bias = padic_bias * self.padic_scale
        scores = scores - padic_bias.unsqueeze(0).unsqueeze(1)

        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, n_points, self.embed_dim)
        output = self.out_proj(output)

        # Average attention weights across heads for return
        avg_weights = attn_weights.mean(dim=1)

        return output, avg_weights


class SurfacePatchEncoder(nn.Module):
    """Encodes local surface patches into fixed-size embeddings.

    Processes points within a local patch using geodesic
    convolutions and attention.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 32,
        n_layers: int = 2,
    ):
        """Initialize patch encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            n_layers: Number of convolution layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Geodesic convolution layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(GeodesicConv(dims[i], dims[i + 1]))

        # Aggregation to fixed size
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        geodesic_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a surface patch.

        Args:
            features: Point features (batch, n_points, input_dim)
            geodesic_coords: Geodesic coordinates (batch, n_points, 2)

        Returns:
            Patch embedding (batch, output_dim)
        """
        x = features

        for conv in self.convs:
            x = conv(x, geodesic_coords)
            x = F.relu(x)

        # Pool over points
        x = x.transpose(1, 2)  # (batch, hidden, n_points)
        x = self.pool(x).squeeze(-1)  # (batch, hidden)

        # Project to output
        x = self.proj(x)

        return x


class MaSIFEncoder(nn.Module):
    """Full MaSIF-style encoder for protein surfaces.

    Combines patch encoding with p-adic attention for
    global surface representation.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        patch_dim: int = 64,
        output_dim: int = 128,
        n_heads: int = 4,
        p: int = 3,
        use_padic: bool = True,
    ):
        """Initialize MaSIF encoder.

        Args:
            feature_dim: Dimension of input features
            patch_dim: Dimension of patch embeddings
            output_dim: Final output dimension
            n_heads: Number of attention heads
            p: Prime for p-adic calculations
            use_padic: Whether to use p-adic attention bias
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim
        self.use_padic = use_padic

        # Feature extraction
        self.feature_extractor = SurfaceFeatureExtractor(feature_dim)

        # Patch encoding
        self.patch_encoder = SurfacePatchEncoder(
            input_dim=feature_dim,
            hidden_dim=patch_dim,
            output_dim=patch_dim,
        )

        # P-adic attention over patches
        self.padic_attention = PAdicSurfaceAttention(
            embed_dim=patch_dim,
            n_heads=n_heads,
            p=p,
        )

        # Global pooling with attention
        self.global_attention = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 4),
            nn.Tanh(),
            nn.Linear(patch_dim // 4, 1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(patch_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_geodesics: torch.Tensor,
        curvatures: torch.Tensor | None = None,
    ) -> SurfaceEncoderOutput:
        """Encode a protein surface.

        Args:
            patch_features: Features for each patch (batch, n_patches, n_points, feat_dim)
            patch_geodesics: Geodesic coords per patch (batch, n_patches, n_points, 2)
            curvatures: Local curvatures (batch, n_patches, n_points)

        Returns:
            SurfaceEncoderOutput with embeddings
        """
        batch_size, n_patches, n_points, _ = patch_features.shape

        # Default curvatures
        if curvatures is None:
            curvatures = torch.zeros(batch_size, n_patches, n_points, device=patch_features.device)

        # Encode each patch
        patch_embeddings = []
        for p in range(n_patches):
            features = self.feature_extractor(
                patch_features[:, p, :, :3],  # hydro, charge, volume
                curvatures[:, p],
            )
            embedding = self.patch_encoder(features, patch_geodesics[:, p])
            patch_embeddings.append(embedding)

        patch_embeddings_stacked = torch.stack(patch_embeddings, dim=1)  # (batch, n_patches, patch_dim)

        # Apply p-adic attention
        if self.use_padic:
            attended, attn_weights = self.padic_attention(patch_embeddings_stacked)
            padic_dist = self.padic_attention.compute_padic_distances(n_patches, patch_features.device)
        else:
            attended = patch_embeddings_stacked
            torch.ones(batch_size, n_patches, n_patches, device=patch_features.device) / n_patches
            padic_dist = None

        # Global pooling with attention
        attn_scores = self.global_attention(attended).squeeze(-1)  # (batch, n_patches)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        surface_embedding = torch.einsum("bp,bpd->bd", attn_probs, attended)
        surface_embedding = self.output_proj(surface_embedding)

        return SurfaceEncoderOutput(
            patch_embeddings=attended,
            surface_embedding=surface_embedding,
            attention_weights=attn_probs,
            padic_structure=padic_dist,
        )


class SurfaceInteractionPredictor(nn.Module):
    """Predicts interactions between protein surfaces.

    Uses surface embeddings to predict binding sites,
    interface residues, and interaction scores.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 64,
    ):
        """Initialize predictor.

        Args:
            embed_dim: Surface embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Pairwise interaction scoring
        self.interaction_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Binding site prediction (per patch)
        self.binding_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [not_binding, binding]
        )

    def forward(
        self,
        surface1: torch.Tensor,
        surface2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict interaction between two surfaces.

        Args:
            surface1: First surface embedding (batch, embed_dim)
            surface2: Second surface embedding (batch, embed_dim)

        Returns:
            Dictionary with interaction predictions
        """
        # Concatenate for interaction scoring
        combined = torch.cat([surface1, surface2], dim=-1)
        interaction_score = self.interaction_scorer(combined)

        # Binding site predictions
        binding1 = self.binding_classifier(surface1)
        binding2 = self.binding_classifier(surface2)

        return {
            "interaction_score": interaction_score.squeeze(-1),
            "binding_probs_1": F.softmax(binding1, dim=-1)[:, 1],
            "binding_probs_2": F.softmax(binding2, dim=-1)[:, 1],
        }


class SurfaceComplementarity(nn.Module):
    """Analyzes shape and chemical complementarity between surfaces.

    Useful for understanding protein-protein interfaces and
    designing binding partners.
    """

    def __init__(self, p: int = 3):
        """Initialize complementarity analyzer.

        Args:
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.p = p

    def shape_complementarity(
        self,
        surface1_curvatures: torch.Tensor,
        surface2_curvatures: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shape complementarity score.

        Surfaces are complementary when local curvatures
        are opposite (convex matches concave).

        Args:
            surface1_curvatures: Curvatures of first surface
            surface2_curvatures: Curvatures of second surface

        Returns:
            Complementarity score (higher = more complementary)
        """
        # Negative correlation indicates complementarity
        correlation = torch.corrcoef(torch.stack([surface1_curvatures.flatten(), surface2_curvatures.flatten()]))[0, 1]

        # Convert to complementarity score (0-1)
        complementarity = (1 - correlation) / 2

        return complementarity

    def chemical_complementarity(
        self,
        surface1_features: torch.Tensor,
        surface2_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chemical complementarity score.

        Analyzes hydrophobicity and charge complementarity.

        Args:
            surface1_features: Features of first surface (n, 4)
            surface2_features: Features of second surface (m, 4)

        Returns:
            Complementarity score
        """
        # Extract hydrophobicity and charge
        hydro1 = surface1_features[:, 0]
        hydro2 = surface2_features[:, 0]
        charge1 = surface1_features[:, 1]
        charge2 = surface2_features[:, 1]

        # Hydrophobic matching (similar hydrophobicity is good)
        hydro_match = 1 - torch.abs(hydro1.mean() - hydro2.mean())

        # Charge complementarity (opposite charges attract)
        charge_comp = -charge1.mean() * charge2.mean()
        charge_comp = (charge_comp + 1) / 2  # Normalize to 0-1

        # Combined score
        return (hydro_match + charge_comp) / 2

    def forward(
        self,
        surface1: dict[str, torch.Tensor],
        surface2: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Analyze complementarity between two surfaces.

        Args:
            surface1: Dict with 'curvatures' and 'features'
            surface2: Dict with 'curvatures' and 'features'

        Returns:
            Dictionary with complementarity scores
        """
        shape_score = self.shape_complementarity(surface1["curvatures"], surface2["curvatures"])
        chem_score = self.chemical_complementarity(surface1["features"], surface2["features"])

        overall = (shape_score + chem_score) / 2

        return {
            "shape_complementarity": shape_score,
            "chemical_complementarity": chem_score,
            "overall_complementarity": overall,
        }
