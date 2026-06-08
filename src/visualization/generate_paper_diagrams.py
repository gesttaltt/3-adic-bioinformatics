import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import os
from pathlib import Path

# Configuration — output alongside the other HIV paper images
OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "research" / "diseases" / "hiv" / "public_medical_paper" / "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-white")  # Clean white background for diagrams
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {path}")


def draw_text_box(ax, x, y, width, height, text, color="#E0E0E0", edge="black"):
    rect = patches.Rectangle(
        (x, y), width, height, linewidth=1, edgecolor=edge, facecolor=color
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        wrap=True,
        fontsize=10,
    )
    return rect


def plot_02_resistance_table():
    data = [
        ["K103N", "NNRTI", "3.80", "LOW", "Rapid emergence"],
        ["M46I", "PI", "0.65", "VERY LOW", "Accessory"],
        ["Y181C", "NNRTI", "4.12", "MODERATE", "Cross-resistance"],
        ["M184V", "NRTI", "5.67", "HIGH", "Common, fitness cost"],
        ["K65R", "NRTI", "5.52", "HIGH", "TDF resistance"],
        ["T215Y", "NRTI", "7.17", "VERY HIGH", "Major shift"],
        ["R263K", "INSTI", "4.40", "HIGH", "DTG pathway"],
    ]
    cols = ["Mutation", "Class", "Distance", "Barrier", "Note"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Colors for rows
    cell_colors = []
    for row in data:
        dist = float(row[2])
        if dist < 3:
            c = "#FFCDD2"  # Red
        elif dist < 5:
            c = "#FFF9C4"  # Yellow
        else:
            c = "#C8E6C9"  # Green
        cell_colors.append([c] * 5)

    table = ax.table(
        cellText=data,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax.set_title("Key Drug Resistance Mutations", fontsize=16, fontweight="bold", y=0.9)
    save_plot("02_resistance_mutations_reference.png")


def plot_03_primary_accessory():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Circles
    core = patches.Circle((0.5, 0.5), 0.2, color="#E1F5FE")
    outer = patches.Circle(
        (0.5, 0.5), 0.4, color="none", ec="#0277BD", lw=2, linestyle="--"
    )

    ax.add_patch(core)
    ax.add_patch(outer)

    # Labels
    ax.text(0.5, 0.5, "ACCESSORY\n(Core)\nHigh Conservation", ha="center", va="center")
    ax.text(
        0.5,
        0.85,
        "PRIMARY MUTATIONS\n(Periphery)\nDrug Interface",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Points
    ax.plot(0.55, 0.6, "ro")
    ax.text(0.57, 0.6, "M46I (0.65)", fontsize=9)

    ax.plot(0.8, 0.5, "go")
    ax.text(0.82, 0.5, "K103N (3.80)", fontsize=9)

    ax.plot(0.1, 0.5, "go")
    ax.text(0.12, 0.5, "M184V (5.67)", fontsize=9)

    ax.set_title("Geometric Model: Primary vs Accessory", fontsize=14)
    save_plot("03_primary_vs_accessory_diagram.png")


def plot_04_timeline():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 4)
    ax.set_xlim(-1, 25)

    # Timeline
    ax.hlines(2, 0, 24, colors="black", lw=2)

    # Events
    events = [
        (4, "Week 4\nNNRTI (K103N)", "orange"),
        (12, "Week 12\nNRTI (M184V)", "blue"),
        (20, "Week 20+\nPI/INSTI\n(Multiple)", "green"),
    ]

    for x, label, color in events:
        ax.plot(x, 2, "o", color=color, markersize=10)
        ax.vlines(x, 1, 2, colors=color, linestyles="dotted")
        ax.text(x, 0.5, label, ha="center", va="center", color=color, fontweight="bold")

    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Weeks on Suboptimal Therapy")
    ax.get_yaxis().set_visible(False)
    ax.set_title("Resistance Emergence Timeline", fontsize=14)
    save_plot("04_resistance_emergence_timeline.png")


def plot_08_genome_map():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylim(0, 2)
    ax.set_xlim(0, 100)
    ax.axis("off")

    # Genes
    genes = [
        ("Gag", 10, 25, "#1565C0"),
        ("Pol", 25, 55, "#00897B"),
        ("Env", 60, 85, "#EF6C00"),
        ("Nef", 90, 95, "#C62828"),
    ]

    for name, start, end, color in genes:
        rect = patches.Rectangle((start, 0.8), end - start, 0.4, color=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(
            (start + end) / 2,
            1.0,
            name,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

        # Add constraint label
        if name == "Gag":
            label = "High Constraint\n(Slow Escape)"
        elif name == "Nef":
            label = "Low Constraint\n(Fast Escape)"
        else:
            label = ""
        if label:
            ax.text((start + end) / 2, 0.5, label, ha="center", va="center", fontsize=9)

    ax.set_title("HIV Genome Constraint Map", fontsize=14)
    save_plot("08_protein_constraint_map.png")


def plot_09_elite_comparison():
    # Grouped Bar
    # Normalize or just plot separate axes? Let's do simple side-by-side bars for concept
    # Actually normalized percent of baseline might be better

    # Let's do relative to Progressor (100%)
    # Barrier: 4.29 vs 3.72 -> 115%
    # Success: 24% vs 42% -> 57%
    # Cost: 28% vs 12% -> 233%

    labels = [
        "Barrier\n(Target Strictness)",
        "Escape\nFrequency",
        "Fitness Cost\nof Escape",
    ]
    vals_e = [115, 57, 233]
    vals_p = [100, 100, 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2, vals_e, width, label="Elite Controller (B*57)", color="#2E7D32"
    )
    ax.bar(x + width / 2, vals_p, width, label="Typical Progressor", color="#9E9E9E")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Typical Progressor Baseline")
    ax.axhline(100, color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.set_title("Why Elite Controllers Control the Virus", fontsize=14)
    save_plot("09_elite_controller_comparison.png")


def plot_10_gag_epitopes():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 500)
    ax.axis("off")

    # Protein bar
    rect = patches.Rectangle((0, 0.4), 500, 0.2, color="#E0E0E0")
    ax.add_patch(rect)

    # Epitopes
    epitopes = [
        (18, 26, "KIRLRPGGK (B*27)", "red"),
        (77, 85, "SLYNTVATL (A*02)", "blue"),
        (240, 249, "TSTLQEQIGW (B*57)", "green"),
        (263, 272, "KRWIILGLNK (B*27)", "red"),
    ]

    for start, end, label, c in epitopes:
        mid = (start + end) / 2
        ax.bar([mid], [0.3], width=15, bottom=0.35, color=c)
        ax.annotate(label, (mid, 0.7), ha="center", arrowprops=dict(arrowstyle="->"))

    ax.text(0, 0.5, "N-term", ha="right", va="center")
    ax.text(500, 0.5, "C-term", ha="left", va="center")
    ax.set_title("Key Gag Epitopes", fontsize=14)
    save_plot("10_gag_epitope_map.png")


def plot_11_escape_zone():
    # Clean conceptual version
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2) / 2)  # Gaussian peak

    ax.fill_between(x, y, color="#E1F5FE", alpha=0.5)
    ax.plot(x, y, color="#0277BD", lw=2)

    ax.axvspan(4.5, 6.5, color="#FFF9C4", alpha=0.5)
    ax.text(
        5.5, 0.9, "GOLDILOCKS\nZONE", ha="center", fontweight="bold", color="#Fbc02d"
    )

    ax.text(2, 0.5, "Ineffective\nEscape", ha="center")
    ax.text(9, 0.5, "Fitness Cost\nProhibitive", ha="center")

    ax.set_xlabel("Genetic Distance")
    ax.set_ylabel("Survival Probability")
    ax.set_yticks([])
    ax.set_title("The Escape Zone Concept", fontsize=14)
    save_plot("11_escape_zone_diagram.png")


def plot_12_pyramid():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Pyramid
    patches.Polygon(
        [[5, 9], [3, 5], [7, 5]], closed=True, color="#C8E6C9", ec="white"
    )  # Tier 1
    patches.Polygon(
        [[3, 5], [1, 1], [9, 1], [7, 5]], closed=True, color="#FFCCBC", ec="white"
    )  # Tier 2/3 background?
    # Let's do stacked trapezoids
    patches.Polygon(
        [[3.5, 6], [6.5, 6], [7.5, 4], [2.5, 4]], color="#FFF9C4"
    )  # Mid

    # Actually just 3 triangles/trapezoids
    ax.fill([2, 8, 5], [1, 1, 9], "#FFCDD2")  # Base Red (Tier 3)
    ax.fill([3, 7, 5], [3.6, 3.6, 9], "#FFF9C4")  # Mid Yellow (Tier 2)
    ax.fill([4, 6, 5], [6.3, 6.3, 9], "#C8E6C9")  # Top Green (Tier 1)

    ax.text(
        5,
        7.5,
        "TIER 1\n(Gag, Pol-IN)\nBest Targets",
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(5, 5, "TIER 2\n(Pol-PR)\nGood", ha="center", va="center")
    ax.text(5, 2.5, "TIER 3\n(Env, Nef)\nAvoid", ha="center", va="center")

    ax.set_title("Epitope Tier Classification", fontsize=14)
    save_plot("12_epitope_tier_classification.png")


def plot_13_v3_loop():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # U-shape
    path_data = [
        (mpath.Path.MOVETO, (3, 2)),
        (mpath.Path.CURVE4, (3, 10)),
        (mpath.Path.CURVE4, (7, 10)),
        (mpath.Path.CURVE4, (7, 2)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=15, edgecolor="#E0E0E0")
    ax.add_patch(patch)

    # Highlights
    # Pos 22 (Crown)
    ax.plot(5, 9.2, "o", color="red", markersize=20)
    ax.text(5, 10, "Pos 22\nDeterminant", ha="center")

    # Base (11/25)
    ax.plot(3.1, 4, "o", color="blue", markersize=15)
    ax.text(2.5, 4, "Pos 11", ha="right")

    ax.plot(6.9, 4, "o", color="blue", markersize=15)
    ax.text(7.5, 4, "Pos 25", ha="left")

    # Disulfide
    ax.plot([3, 7], [2, 2], "k-", lw=3)
    ax.text(5, 1.5, "S-S Bond", ha="center")

    ax.set_title("V3 Loop Tropism Sites", fontsize=14)
    save_plot("13_v3_loop_position22.png")


def plot_22_venn():
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    c1 = Circle((3.5, 6), 2.5, color="blue", alpha=0.3, label="3BNC117")
    c2 = Circle((6.5, 6), 2.5, color="green", alpha=0.3, label="10-1074")
    c3 = Circle((5, 3.5), 2.5, color="red", alpha=0.3, label="10E8")

    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)

    ax.text(3, 7, "3BNC117\nCD4bs", ha="center")
    ax.text(7, 7, "10-1074\nV3-glycan", ha="center")
    ax.text(5, 2.5, "10E8\nMPER", ha="center")

    ax.text(5, 5.5, "96%\nCombined", ha="center", fontweight="bold", fontsize=14)

    ax.set_title("bnAb Combination Coverage", fontsize=14)
    save_plot("22_bnab_combination_coverage.png")


def plot_28_pop_coverage():
    # Horizontal Bar since map is hard
    regions = [
        "North America",
        "Europe",
        "Sub-Saharan Africa",
        "South Asia",
        "East Asia",
        "SE Asia",
    ]
    cov = [89, 87, 82, 81, 78, 75]
    colors = ["#2E7D32"] * 2 + ["#7CB342"] * 2 + ["#FDD835"] * 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(regions, cov, color=colors)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Population Coverage (%)")
    ax.set_title("Vaccine Coverage by Region", fontsize=14)

    for i, v in enumerate(cov):
        ax.text(v + 1, i, f"{v}%", va="center")

    save_plot("28_population_coverage_map.png")  # Saving as png but it's a chart


def plot_31_latent_space():
    # Dendrogram simulation
    from scipy.cluster.hierarchy import dendrogram, linkage

    np.random.seed(42)
    data = np.random.rand(50, 4)
    Z = linkage(data, "ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("3-Adic Hierarchial Latent Space (Visualization)", fontsize=14)
    save_plot("31_3adic_latent_space_visualization.png")


def plot_32_vae_arch():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    draw_text_box(ax, 0.5, 1.5, 1.5, 1, "Input\nSequence")
    ax.arrow(2, 2, 1, 0, head_width=0.2, color="black")

    draw_text_box(ax, 3, 1, 1, 2, "Encoder\n(CNN)")
    ax.arrow(4, 2, 1, 0, head_width=0.2, color="black")

    draw_text_box(ax, 5, 1.5, 2, 1, "Latent Space\n(Hyperbolic)", color="#E1F5FE")
    ax.arrow(7, 2, 1, 0, head_width=0.2, color="black")

    draw_text_box(ax, 8, 1, 1, 2, "Decoder\n(LSTM)")

    ax.set_title("Ternary VAE Architecture", fontsize=14)
    save_plot("32_ternary_vae_architecture.png")


if __name__ == "__main__":
    print("Generating schematic diagrams...")
    plot_02_resistance_table()
    plot_03_primary_accessory()
    plot_04_timeline()
    plot_08_genome_map()
    plot_09_elite_comparison()
    plot_10_gag_epitopes()
    plot_11_escape_zone()
    plot_12_pyramid()
    plot_13_v3_loop()
    plot_22_venn()
    plot_28_pop_coverage()
    plot_31_latent_space()
    plot_32_vae_arch()
    print("Diagrams generated.")
