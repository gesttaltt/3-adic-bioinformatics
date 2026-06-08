"""Interpretability analysis for drug resistance models.

Black box predictions limit clinical trust. This module provides:

1. Feature importance via integrated gradients and SHAP
2. Attention visualization for transformer models
3. Comparison with known resistance mutations
4. Latent space visualization

Clinical value: Understanding WHY a prediction was made,
not just WHAT the prediction is.

References:
- Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks
- Lundberg & Lee (2017): SHAP Values
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FeatureImportance:
    """Container for feature importance scores."""

    position_scores: np.ndarray  # Importance per position (n_positions,)
    aa_scores: np.ndarray  # Importance per AA at each position (n_positions, n_aa)
    top_positions: list[int]  # Most important positions
    top_mutations: list[tuple[int, str, float]]  # (position, AA, score)

    def get_mutation_importance(self, position: int, aa: str) -> float:
        """Get importance score for specific mutation."""
        aa_idx = "ACDEFGHIKLMNPQRSTVWY*-".index(aa)
        return self.aa_scores[position, aa_idx]


class IntegratedGradients:
    """Compute feature importance via integrated gradients.

    Integrated gradients provide axiomatic attribution by
    integrating gradients along a path from baseline to input.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        baseline: torch.Tensor | None = None,
    ):
        """Initialize.

        Args:
            model: Neural network model
            n_steps: Number of integration steps
            baseline: Baseline input (default: zeros)
        """
        self.model = model
        self.n_steps = n_steps
        self.baseline = baseline

    def attribute(
        self,
        x: torch.Tensor,
        target_fn: Callable[[dict], torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attributions for input.

        Args:
            x: Input tensor (batch, features)
            target_fn: Function to extract target from model output

        Returns:
            Attributions tensor same shape as input
        """
        if target_fn is None:

            def target_fn(out):
                return out.get("prediction", out.get("z", out["mu"])[:, 0])

        device = x.device
        baseline = self.baseline if self.baseline is not None else torch.zeros_like(x)
        baseline = baseline.to(device)

        # Scale inputs
        scaled_inputs = [baseline + (float(i) / self.n_steps) * (x - baseline) for i in range(1, self.n_steps + 1)]
        scaled_inputs = torch.stack(scaled_inputs)  # (n_steps, batch, features)

        # Compute gradients
        scaled_inputs.requires_grad_(True)
        grads = []

        for i in range(self.n_steps):
            self.model.zero_grad()
            out = self.model(scaled_inputs[i])
            target = target_fn(out)

            # Backward
            target.sum().backward(retain_graph=True)
            grads.append(scaled_inputs.grad[i].clone())
            scaled_inputs.grad.zero_()

        grads = torch.stack(grads)  # (n_steps, batch, features)

        # Integrate
        avg_grads = grads.mean(dim=0)  # (batch, features)
        attributions = (x - baseline) * avg_grads

        return attributions


class GradientSHAP:
    """Gradient SHAP for feature importance.

    Combines SHAP values with gradient-based attribution
    for efficient computation.
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        n_samples: int = 50,
    ):
        """Initialize.

        Args:
            model: Neural network model
            background: Background dataset for SHAP
            n_samples: Number of samples for estimation
        """
        self.model = model
        self.background = background
        self.n_samples = n_samples

    def shap_values(
        self,
        x: torch.Tensor,
        target_fn: Callable[[dict], torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SHAP values.

        Args:
            x: Input tensor
            target_fn: Function to extract target from output

        Returns:
            SHAP values tensor
        """
        if target_fn is None:

            def target_fn(out):
                return out.get("prediction", out.get("z", out["mu"])[:, 0])

        device = x.device
        batch_size = x.size(0)

        # Sample random baselines
        baseline_idx = torch.randint(0, len(self.background), (self.n_samples,))
        baselines = self.background[baseline_idx].to(device)

        attributions = torch.zeros_like(x)

        for baseline in baselines:
            baseline = baseline.unsqueeze(0).expand_as(x)

            # Random interpolation
            alpha = torch.rand(batch_size, 1, device=device)
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)

            # Forward and backward
            self.model.zero_grad()
            out = self.model(interpolated)
            target = target_fn(out)
            target.sum().backward()

            # Accumulate
            attributions += interpolated.grad * (x - baseline)

        return attributions / self.n_samples


def compute_position_importance(
    attributions: torch.Tensor,
    n_positions: int,
    n_aa: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert attributions to position-level importance.

    Args:
        attributions: Attribution tensor (batch, n_positions * n_aa)
        n_positions: Number of sequence positions
        n_aa: Number of amino acids

    Returns:
        (position_scores, aa_scores)
    """
    # Reshape to (batch, n_positions, n_aa)
    attr = attributions.detach().cpu().numpy()
    attr = attr.reshape(-1, n_positions, n_aa)

    # Aggregate over batch
    attr = np.abs(attr).mean(axis=0)  # (n_positions, n_aa)

    # Position scores = max importance at each position
    position_scores = attr.max(axis=1)

    return position_scores, attr


def extract_top_mutations(
    aa_scores: np.ndarray,
    top_k: int = 20,
    aa_alphabet: str = "ACDEFGHIKLMNPQRSTVWY*-",
) -> list[tuple[int, str, float]]:
    """Extract top important mutations.

    Returns:
        List of (position, amino_acid, score)
    """
    n_positions, n_aa = aa_scores.shape

    # Flatten and get top indices
    flat_scores = aa_scores.flatten()
    top_indices = np.argsort(flat_scores)[::-1][:top_k]

    mutations = []
    for idx in top_indices:
        pos = idx // n_aa
        aa_idx = idx % n_aa
        aa = aa_alphabet[aa_idx]
        score = flat_scores[idx]
        mutations.append((pos, aa, score))

    return mutations


class AttentionAnalyzer:
    """Analyze attention patterns in transformer models."""

    def __init__(self, model: nn.Module):
        """Initialize with transformer model."""
        self.model = model
        self.attention_maps: list[torch.Tensor] = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(self._attention_hook(name))

    def _attention_hook(self, name: str):
        """Create hook to capture attention."""

        def hook(module, input, output):
            # output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                self.attention_maps.append(output[1].detach())

        return hook

    def get_attention(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get attention maps for input.

        Args:
            x: Input tensor

        Returns:
            List of attention weight tensors per layer
        """
        self.attention_maps = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.attention_maps

    def visualize_attention(
        self,
        attention_maps: list[torch.Tensor],
        layer_idx: int = -1,
    ) -> np.ndarray:
        """Get attention heatmap for visualization.

        Args:
            attention_maps: List of attention tensors
            layer_idx: Which layer to visualize

        Returns:
            Attention matrix (n_positions, n_positions)
        """
        attn = attention_maps[layer_idx]

        # Average over heads and batch
        if attn.dim() == 4:  # (batch, heads, seq, seq)
            attn = attn.mean(dim=(0, 1))
        elif attn.dim() == 3:  # (batch, seq, seq)
            attn = attn.mean(dim=0)

        return attn.cpu().numpy()


class ResistanceMutationValidator:
    """Validate model importance against known resistance mutations.

    Compares learned feature importance with clinically validated
    resistance mutations from databases like Stanford HIVDB.
    """

    # Known PI resistance mutations (position, amino acids)
    PI_MUTATIONS = {
        10: ["F", "I", "R", "V"],
        20: ["M", "R", "T"],
        24: ["I"],
        30: ["N"],
        32: ["I"],
        33: ["F", "I", "V"],
        36: ["I", "L", "V"],
        46: ["I", "L"],
        47: ["A", "V"],
        48: ["V"],
        50: ["L", "V"],
        53: ["L"],
        54: ["L", "M", "T", "V"],
        71: ["I", "L", "T", "V"],
        73: ["A", "C", "S", "T"],
        76: ["V"],
        82: ["A", "F", "L", "M", "S", "T"],
        84: ["A", "C", "V"],
        88: ["D", "S"],
        89: ["V"],
        90: ["M"],
    }

    def __init__(self, drug_class: str = "PI"):
        """Initialize with drug class."""
        self.drug_class = drug_class
        if drug_class == "PI":
            self.known_mutations = self.PI_MUTATIONS
        else:
            self.known_mutations = {}

    def validate(self, feature_importance: FeatureImportance) -> dict[str, float]:
        """Validate feature importance against known mutations.

        Returns:
            Dict with validation metrics
        """
        if not self.known_mutations:
            return {"error": "No known mutations for drug class"}

        # Check if top positions match known resistance positions
        top_positions = set(feature_importance.top_positions[:20])
        known_positions = set(self.known_mutations.keys())

        overlap = top_positions & known_positions
        precision = len(overlap) / len(top_positions) if top_positions else 0
        recall = len(overlap) / len(known_positions) if known_positions else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Check specific mutations
        mutation_hits = 0
        mutation_total = 0
        for pos, aas in self.known_mutations.items():
            for aa in aas:
                mutation_total += 1
                score = feature_importance.aa_scores[pos, "ACDEFGHIKLMNPQRSTVWY*-".index(aa)]
                if score > feature_importance.aa_scores.mean():
                    mutation_hits += 1

        mutation_recall = mutation_hits / mutation_total if mutation_total > 0 else 0

        return {
            "position_precision": precision,
            "position_recall": recall,
            "position_f1": f1,
            "mutation_recall": mutation_recall,
            "n_overlapping_positions": len(overlap),
        }


class LatentSpaceAnalyzer:
    """Analyze latent space structure and drug clustering."""

    def __init__(self, model: nn.Module):
        self.model = model

    def get_latent_embeddings(
        self,
        x: torch.Tensor,
        drug_labels: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract latent embeddings.

        Returns:
            (embeddings, labels) arrays
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            z = out.get("z", out.get("mu"))

        z = z.cpu().numpy()
        labels = drug_labels.cpu().numpy() if drug_labels is not None else None

        return z, labels

    def compute_cluster_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute clustering quality metrics.

        Returns:
            Dict with silhouette score, calinski-harabasz, etc.
        """
        from sklearn.metrics import calinski_harabasz_score, silhouette_score

        if len(np.unique(labels)) < 2:
            return {"error": "Need at least 2 classes"}

        return {
            "silhouette": silhouette_score(embeddings, labels),
            "calinski_harabasz": calinski_harabasz_score(embeddings, labels),
        }

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2,
    ) -> np.ndarray:
        """Reduce to 2D/3D for visualization.

        Args:
            embeddings: High-dimensional embeddings
            method: 'umap', 'tsne', or 'pca'
            n_components: Output dimensions

        Returns:
            Reduced embeddings
        """
        if method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, perplexity=30)
        elif method == "umap":
            try:
                import umap

                reducer = umap.UMAP(n_components=n_components)
            except ImportError:
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)


def compute_feature_importance(
    model: nn.Module,
    x: torch.Tensor,
    n_positions: int = 99,
    n_aa: int = 22,
    method: str = "integrated_gradients",
    background: torch.Tensor | None = None,
) -> FeatureImportance:
    """Compute feature importance using specified method.

    Args:
        model: Neural network model
        x: Input tensor
        n_positions: Number of sequence positions
        n_aa: Number of amino acids
        method: 'integrated_gradients' or 'gradient_shap'
        background: Background data for SHAP

    Returns:
        FeatureImportance object
    """
    if method == "integrated_gradients":
        ig = IntegratedGradients(model)
        attributions = ig.attribute(x)
    elif method == "gradient_shap":
        if background is None:
            background = torch.zeros(100, n_positions * n_aa)
        shap = GradientSHAP(model, background)
        attributions = shap.shap_values(x)
    else:
        raise ValueError(f"Unknown method: {method}")

    position_scores, aa_scores = compute_position_importance(attributions, n_positions, n_aa)
    top_positions = np.argsort(position_scores)[::-1][:20].tolist()
    top_mutations = extract_top_mutations(aa_scores, top_k=20)

    return FeatureImportance(
        position_scores=position_scores,
        aa_scores=aa_scores,
        top_positions=top_positions,
        top_mutations=top_mutations,
    )


if __name__ == "__main__":
    print("Testing Interpretability Analysis")
    print("=" * 60)

    # Create simple model for testing
    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(99 * 22, 64), nn.ReLU(), nn.Linear(64, 16))
            self.predictor = nn.Linear(16, 1)

        def forward(self, x):
            z = self.encoder(x)
            return {"z": z, "prediction": self.predictor(z).squeeze(-1)}

    model = SimpleVAE()
    x = torch.randn(8, 99 * 22)

    # Test integrated gradients
    ig = IntegratedGradients(model, n_steps=10)
    attr = ig.attribute(x)
    print(f"Attributions shape: {attr.shape}")

    # Test feature importance
    importance = compute_feature_importance(model, x, n_positions=99)
    print(f"Top 5 positions: {importance.top_positions[:5]}")
    print(f"Top 3 mutations: {importance.top_mutations[:3]}")

    # Test validation
    validator = ResistanceMutationValidator("PI")
    validation = validator.validate(importance)
    print(f"\nValidation metrics: {validation}")

    print("\n" + "=" * 60)
    print("Interpretability analysis working!")
