"""Generate Attention Visualization Figures.

Creates publication-quality figures showing:
1. Attention weight heatmaps per drug
2. Position importance comparison with known mutations
3. Novel mutation candidates
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")


# Known resistance mutations from Stanford HIVDB
KNOWN_MUTATIONS = {
    "pi": {
        "LPV": [10, 20, 24, 32, 33, 46, 47, 48, 50, 53, 54, 63, 71, 73, 76, 82, 84, 90],
        "DRV": [11, 32, 33, 47, 50, 54, 74, 76, 84, 89],
        "ATV": [10, 16, 20, 24, 32, 33, 34, 36, 46, 48, 50, 53, 54, 60, 62, 64, 71, 73, 82, 84, 85, 88, 90, 93],
    },
    "nrti": {
        "AZT": [41, 44, 62, 65, 67, 69, 70, 74, 75, 115, 116, 118, 151, 184, 210, 215, 219],
        "3TC": [44, 65, 68, 69, 74, 75, 118, 151, 184],
        "TDF": [41, 65, 67, 69, 70, 74, 75, 115, 151, 210, 215, 219],
    },
    "nnrti": {
        "EFV": [98, 100, 101, 103, 106, 108, 138, 179, 181, 188, 190, 221, 225, 227, 230],
        "NVP": [98, 100, 101, 103, 106, 108, 179, 181, 188, 190, 221, 227, 230],
    },
}


def create_attention_heatmap(
    attention_weights: np.ndarray,
    drug: str,
    drug_class: str,
    output_dir: Path,
) -> None:
    """Create attention weight heatmap."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Get known mutations for highlighting
    known_pos = KNOWN_MUTATIONS.get(drug_class, {}).get(drug, [])

    # Plot attention weights
    n_positions = len(attention_weights)
    positions = np.arange(1, n_positions + 1)

    # Color by known vs unknown
    colors = ["#e74c3c" if p in known_pos else "#3498db" for p in positions]

    ax.bar(positions, attention_weights, color=colors, alpha=0.7, edgecolor="none")

    # Highlight top attended positions
    top_k = 15
    top_indices = np.argsort(attention_weights)[-top_k:]

    for idx in top_indices:
        if attention_weights[idx] > np.percentile(attention_weights, 90):
            ax.annotate(
                str(idx + 1),
                (idx + 1, attention_weights[idx]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(f"{drug} Attention Weights - {drug_class.upper()}", fontsize=14)

    # Legend
    known_patch = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Known Mutation Sites")
    other_patch = mpatches.Patch(color="#3498db", alpha=0.7, label="Other Positions")
    ax.legend(handles=[known_patch, other_patch], loc="upper right")

    # Save
    output_path = output_dir / f"attention_{drug_class}_{drug}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def create_mutation_comparison(
    attention_weights: dict[str, np.ndarray],
    drug_class: str,
    output_dir: Path,
) -> None:
    """Create comparison of top attended positions vs known mutations."""
    if not MATPLOTLIB_AVAILABLE:
        return

    n_drugs = len(attention_weights)
    fig, axes = plt.subplots(1, n_drugs, figsize=(5 * n_drugs, 6))

    if n_drugs == 1:
        axes = [axes]

    for ax, (drug, weights) in zip(axes, attention_weights.items(), strict=False):
        # Get top 20 positions
        top_k = 20
        top_indices = np.argsort(weights)[-top_k:][::-1]
        top_positions = [i + 1 for i in top_indices]
        top_weights = weights[top_indices]

        # Known mutations
        known = set(KNOWN_MUTATIONS.get(drug_class, {}).get(drug, []))

        # Color by known vs novel
        colors = ["#27ae60" if p in known else "#f39c12" for p in top_positions]

        ax.barh(range(top_k), top_weights, color=colors, alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([str(p) for p in top_positions])
        ax.invert_yaxis()
        ax.set_xlabel("Attention Weight")
        ax.set_title(f"{drug}")

        # Calculate overlap
        overlap = len(set(top_positions) & known)
        ax.text(0.95, 0.05, f"Overlap: {overlap}/{len(known)}", transform=ax.transAxes, ha="right", fontsize=10)

    # Legend
    known_patch = mpatches.Patch(color="#27ae60", alpha=0.8, label="Known Mutation")
    novel_patch = mpatches.Patch(color="#f39c12", alpha=0.8, label="Potential Novel")
    fig.legend(handles=[known_patch, novel_patch], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f"{drug_class.upper()} - Top Attended Positions vs Known Mutations", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"mutation_comparison_{drug_class}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def create_cross_drug_attention(
    attention_weights: dict[str, np.ndarray],
    drug_class: str,
    output_dir: Path,
) -> None:
    """Create cross-drug attention correlation matrix."""
    if not MATPLOTLIB_AVAILABLE:
        return

    drugs = list(attention_weights.keys())
    n_drugs = len(drugs)

    # Compute correlation matrix
    corr_matrix = np.zeros((n_drugs, n_drugs))
    for i, drug1 in enumerate(drugs):
        for j, drug2 in enumerate(drugs):
            w1 = attention_weights[drug1]
            w2 = attention_weights[drug2]
            # Align lengths
            min_len = min(len(w1), len(w2))
            corr = np.corrcoef(w1[:min_len], w2[:min_len])[0, 1]
            corr_matrix[i, j] = corr

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n_drugs))
    ax.set_yticks(range(n_drugs))
    ax.set_xticklabels(drugs)
    ax.set_yticklabels(drugs)

    # Add values
    for i in range(n_drugs):
        for j in range(n_drugs):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f"{drug_class.upper()} - Cross-Drug Attention Correlation", fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    plt.tight_layout()

    output_path = output_dir / f"cross_drug_attention_{drug_class}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def simulate_attention_weights(drug_class: str, drug: str) -> np.ndarray:
    """Simulate attention weights for visualization testing."""
    position_counts = {"pi": 99, "nrti": 240, "nnrti": 318, "ini": 288}
    n_pos = position_counts.get(drug_class, 99)

    # Base random weights
    weights = np.random.exponential(0.1, n_pos)

    # Boost known mutation positions
    known = KNOWN_MUTATIONS.get(drug_class, {}).get(drug, [])
    for pos in known:
        if pos <= n_pos:
            weights[pos - 1] += np.random.uniform(0.3, 0.8)

    # Normalize
    weights = weights / weights.sum()

    return weights


def main():
    print("=" * 70)
    print("ATTENTION VISUALIZATION")
    print("=" * 70)

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    output_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations for each drug class
    for drug_class in ["pi", "nrti", "nnrti"]:
        print(f"\n{drug_class.upper()}:")

        drugs = list(KNOWN_MUTATIONS.get(drug_class, {}).keys())
        if not drugs:
            continue

        attention_weights = {}
        for drug in drugs:
            weights = simulate_attention_weights(drug_class, drug)
            attention_weights[drug] = weights

            # Individual heatmap
            create_attention_heatmap(weights, drug, drug_class, output_dir)

        # Comparison figure
        create_mutation_comparison(attention_weights, drug_class, output_dir)

        # Cross-drug correlation
        create_cross_drug_attention(attention_weights, drug_class, output_dir)

    print("\n" + "=" * 70)
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
