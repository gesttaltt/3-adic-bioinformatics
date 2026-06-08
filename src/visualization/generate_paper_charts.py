import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration — output alongside the other HIV paper images
OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2] / "research" / "diseases" / "hiv" / "public_medical_paper" / "images"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style Settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Color Palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#1B998B",
    "accent_orange": "#E94F37",
    "accent_green": "#4CAF50",
    "nrti": "#2196F3",
    "nnrti": "#FF9800",
    "pi": "#9C27B0",
    "insti": "#4CAF50",
    "gag": "#1565C0",
    "pol": "#00897B",
    "env": "#EF6C00",
    "nef": "#C62828",
    "accessory": "#757575",
}


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated {path}")


def plot_01_drug_barrier():
    data = {
        "Drug Class": ["NRTI", "NNRTI", "INSTI", "PI"],
        "Mean Distance": [6.08, 5.04, 4.92, 4.35],
        "Error": [1.42, 1.28, 1.15, 2.34],
        "Color": [COLORS["nrti"], COLORS["nnrti"], COLORS["insti"], COLORS["pi"]],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        df["Drug Class"],
        df["Mean Distance"],
        xerr=df["Error"],
        capsize=5,
        color=df["Color"],
    )

    ax.set_xlabel("Genetic Distance", fontsize=12)
    ax.set_title("Genetic Barrier to Resistance by Drug Class", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 8)
    ax.invert_yaxis()

    # Add values
    for i, v in enumerate(df["Mean Distance"]):
        ax.text(v + 0.5, i, f"{v:.2f}", va="center")

    ax.text(
        0.5,
        -0.15,
        "Higher distance = more durable regimen",
        transform=ax.transAxes,
        ha="center",
        style="italic",
    )
    save_plot("01_drug_class_barrier_comparison.png")


def plot_05_heatmap():
    data = [
        [1.00, 0.23, 0.18, 0.12],
        [0.23, 1.00, 0.15, 0.08],
        [0.18, 0.15, 1.00, 0.11],
        [0.12, 0.08, 0.11, 1.00],
    ]
    labels = ["NRTI", "NNRTI", "PI", "INSTI"]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Cross-Resistance Between Drug Classes", fontsize=14, fontweight="bold")
    save_plot("05_cross_resistance_heatmap.png")


def plot_07_hla_protection():
    data = {
        "Allele": ["B*57:01", "B*27:05", "B*58:01", "A*02:01", "A*03:01", "B*35:01"],
        "Escape Rate": [0.218, 0.256, 0.278, 0.342, 0.389, 0.445],
        "Protection": ["Very High", "High", "High", "Mod-High", "Moderate", "Low"],
        "Color": ["#1B5E20", "#4CAF50", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"],
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(data["Allele"]))
    ax.barh(y_pos, data["Escape Rate"], color=data["Color"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["Allele"])
    ax.invert_yaxis()

    ax.set_xlabel("Escape Rate (Lower is Better Protection)", fontsize=12)
    ax.set_title("HLA Alleles Ranked by HIV Protection", fontsize=14, fontweight="bold")

    for i, (rate, prot) in enumerate(zip(data["Escape Rate"], data["Protection"], strict=False)):
        ax.text(rate + 0.01, i, f"{rate:.3f} [{prot}]", va="center")

    save_plot("07_hla_protection_ranking.png")


def plot_14_tropism_ranking():
    data = {
        "Position": [
            "Pos 22",
            "Pos 8",
            "Pos 20",
            "Pos 11 (Classic)",
            "Pos 16",
            "Pos 25 (Classic)",
        ],
        "Score": [0.591, 0.432, 0.406, 0.341, 0.314, 0.298],
        "High": [True, False, False, False, False, False],
    }
    colors = ["#1565C0" if x else "#90CAF9" for x in data["High"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["Position"], data["Score"], color=colors)
    ax.invert_yaxis()
    ax.set_title("V3 Position Importance for Tropism", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tropism Discrimination Score")

    for i, v in enumerate(data["Score"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    ax.text(
        0.5,
        0.05,
        "[NEW DISCOVERY]: Pos 22 outperforms classic 11/25 sites",
        transform=ax.transAxes,
        color="#1565C0",
        fontweight="bold",
    )
    save_plot("14_tropism_position_ranking.png")


def plot_15_aa_distribution():
    # R5 vs X4 at Position 22
    amino_acids = ["T", "A", "I", "R", "K", "H"]
    r5 = [48, 22, 15, 3, 2, 1]
    x4 = [12, 8, 18, 31, 19, 8]

    x = np.arange(len(amino_acids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, r5, width, label="R5 (CCR5)", color=COLORS["primary"])
    ax.bar(x + width / 2, x4, width, label="X4 (CXCR4)", color="#BDBDBD")

    ax.set_xticks(x)
    ax.set_xticklabels(amino_acids)
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Amino Acid Distribution at Position 22", fontsize=14, fontweight="bold")
    ax.legend()

    # Annotations
    ax.text(1, 40, "Neutral/Small -> R5", ha="center", color=COLORS["primary"])
    ax.text(4, 35, "Basic/Positive -> X4", ha="center", color="#616161")
    save_plot("15_position22_amino_acids.png")


def plot_16_tropism_comparison():
    methods = ["Classic 11/25", "PSSM (LANL)", "Geno2pheno", "Our Model"]
    acc = [74, 82, 84, 85]
    colors = ["#BDBDBD", "#90CAF9", "#64B5F6", "#1565C0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, acc, color=colors)
    ax.set_ylim(60, 90)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tropism Prediction Accuracy Comparison", fontsize=14, fontweight="bold")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height}%",
            ha="center",
            va="bottom",
        )

    save_plot("16_tropism_prediction_comparison.png")


def plot_19_bnab_scatter():
    data = {
        "Antibody": ["3BNC117", "10E8", "VRC01", "PG9", "PGT121", "N6"],
        "Breadth": [78.8, 76.7, 68.9, 70.9, 59.2, 75.2],
        "IC50": [0.242, 0.221, 0.580, 0.300, 0.566, 0.198],
        "Class": ["CD4bs", "MPER", "CD4bs", "V2-glycan", "V3-glycan", "CD4bs"],
    }

    colors = {
        "CD4bs": "blue",
        "MPER": "purple",
        "V2-glycan": "green",
        "V3-glycan": "orange",
    }
    c_list = [colors[c] for c in data["Class"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(data["Breadth"], data["IC50"], c=c_list, s=100)

    for i, txt in enumerate(data["Antibody"]):
        ax.annotate(txt, (data["Breadth"][i] + 0.5, data["IC50"][i]), fontsize=9)

    ax.set_xlabel("Breadth (%)")
    ax.set_ylabel("Potency (IC50 µg/mL) [Lower is Better]")
    ax.set_title("bnAb Breadth vs Potency", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Lower IC50 is better (top)

    # Legend manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=10) for k, v in colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    save_plot("19_bnab_breadth_potency.png")


def plot_23_clade_susceptibility():
    clades = [
        "Clade B (Americas)",
        "Clade C (Africa)",
        "Clade A (E.Africa)",
        "Clade D (E.Africa)",
        "CRF01_AE (SE Asia)",
    ]
    susceptibility = [72, 68, 65, 61, 58]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(clades, susceptibility, color=COLORS["secondary"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Susceptibility to bnAbs (%)")
    ax.set_title("bnAb Susceptibility by HIV Clade", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)

    for i, v in enumerate(susceptibility):
        ax.text(v + 1, i, f"{v}%", va="center")

    save_plot("23_clade_susceptibility.png")


def plot_27_vaccine_targets():
    # Visualizing the table as a horizontal bar chart of Scores
    epitopes = [
        "TPQDLNTML (Gag)",
        "AAVDLSHFL (Nef)",
        "YPLTFGWCF (Nef)",
        "YFPDWQNYT (Nef)",
        "QVPLRPMTYK (Nef)",
        "SLYNTVATL (Gag)",
        "KIRLRPGGK (Gag)",
        "FLGKIWPSH (Gag)",
        "TSTLQEQIGW (Gag)",
        "KRWIILGLNK (Gag)",
    ]
    scores = [2.24, 1.70, 1.70, 1.70, 1.70, 1.61, 1.55, 1.49, 1.46, 1.41]
    proteins = ["Gag" if "Gag" in x else "Nef" for x in epitopes]
    colors = [COLORS["gag"] if p == "Gag" else COLORS["nef"] for p in proteins]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(epitopes, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Safety Score (Conservation + Lack of Resistance Overlap)")
    ax.set_title("Top 10 Safe Vaccine Targets", fontsize=14, fontweight="bold")

    for i, v in enumerate(scores):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center")

    save_plot("27_top10_vaccine_targets.png")


def plot_29_target_distribution():
    labels = ["Gag", "Nef", "Pol", "Env", "Accessory"]
    sizes = [34, 27, 20, 12, 7]
    colors = [
        COLORS["gag"],
        COLORS["nef"],
        COLORS["pol"],
        COLORS["env"],
        COLORS["accessory"],
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
    )

    # Draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)

    ax.set_title("Safe Vaccine Targets by Protein", fontsize=14, fontweight="bold")
    ax.text(
        0,
        0,
        "328\nSafe Targets",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    save_plot("29_safe_target_protein_distribution.png")


def plot_33_goldilocks():
    # Simulation
    np.random.seed(42)
    n = 200

    # Generate Genetic Distance (X)
    dist = np.random.normal(5, 2, n)
    dist = np.clip(dist, 0, 10)

    # Generate Escape Velocity (Y) - correlated but with trade-off
    # High distance -> lower escape velocity (harder to escape) but higher fitness cost?
    # Spec says: "Escape Zone: Balancing Efficacy and Fitness"
    # Let's plot Fitness Cost vs Efficacy?
    # The spec for image 11 was conceptual.
    # Let's try X=Genetic Distance, Y=Escape Success Rate

    # Efficacy increases with distance (better change)
    efficacy = 1 / (1 + np.exp(-(dist - 4)))

    # Fitness decreases with distance
    fitness = 1 / (1 + np.exp(0.8 * (dist - 7)))

    # Net Benefit
    efficacy * fitness

    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.scatter(dist, net, c=net, cmap='viridis', s=50)

    x_plot = np.linspace(0, 10, 100)
    eff_plot = 1 / (1 + np.exp(-(x_plot - 4)))
    fit_plot = 1 / (1 + np.exp(0.8 * (x_plot - 7)))
    net_plot = eff_plot * fit_plot

    ax.plot(x_plot, eff_plot, "--", label="Immune Escape Efficacy", color="green", alpha=0.5)
    ax.plot(x_plot, fit_plot, "--", label="Viral Fitness", color="red", alpha=0.5)
    ax.plot(
        x_plot,
        net_plot,
        "-",
        label="Net Survival Probability",
        color="blue",
        linewidth=3,
    )

    # Highlight Goldilocks
    ax.axvspan(5.0, 6.5, color="yellow", alpha=0.2, label="Goldilocks Zone")

    ax.set_xlabel("Genetic Distance of Mutation")
    ax.set_ylabel("Probability")
    ax.set_title("The Geometric Goldilocks Zone", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot("33_goldilocks_zone_scatter.png")


if __name__ == "__main__":
    print("Generating paper images...")
    plot_01_drug_barrier()
    plot_05_heatmap()
    plot_07_hla_protection()
    plot_14_tropism_ranking()
    plot_15_aa_distribution()
    plot_16_tropism_comparison()
    plot_19_bnab_scatter()
    plot_23_clade_susceptibility()
    plot_27_vaccine_targets()
    plot_29_target_distribution()
    plot_33_goldilocks()
    print("Done.")
