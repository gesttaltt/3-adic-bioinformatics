import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Configuration — output alongside the other HIV paper images
OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2] / "research" / "diseases" / "hiv" / "public_medical_paper" / "images"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {path}")


def draw_box(ax, x, y, w, h, text, color="#E0E0E0", ec="black"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec=ec)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", wrap=True, fontsize=9)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))


def plot_06_barrier_decision():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    draw_box(ax, 4, 8, 2, 1, "Assess Adherence Risk")

    draw_arrow(ax, 5, 8, 2, 6)  # Left
    draw_arrow(ax, 5, 8, 5, 6)  # Center
    draw_arrow(ax, 5, 8, 8, 6)  # Right

    draw_box(ax, 1, 5, 2, 1, "LOW Risk", color="#C8E6C9")
    draw_box(ax, 4, 5, 2, 1, "MODERATE Risk", color="#FFF9C4")
    draw_box(ax, 7, 5, 2, 1, "HIGH Risk", color="#FFCDD2")

    draw_arrow(ax, 2, 5, 2, 3)
    draw_arrow(ax, 5, 5, 5, 3)
    draw_arrow(ax, 8, 5, 8, 3)

    draw_box(ax, 1, 2, 2, 1, "Any Regimen", color="#E0E0E0")
    draw_box(ax, 4, 2, 2, 1, "Prefer INSTI/PI", color="#E0E0E0")
    draw_box(ax, 7, 2, 2, 1, "Require DTG/PI", color="#E0E0E0")

    ax.set_title("Genetic Barrier Clinical Decision Guide", fontsize=14)
    save_plot("06_barrier_clinical_decision.png")


def plot_17_tropism_flowchart():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    draw_box(ax, 3.5, 8.5, 3, 1, "V3 Sequence")
    draw_arrow(ax, 5, 8.5, 5, 7.5)

    draw_box(ax, 3.5, 6.5, 3, 1, "Check Position 22", color="#B3E5FC")

    draw_arrow(ax, 5, 6.5, 2, 5)  # Left
    draw_arrow(ax, 5, 6.5, 5, 5)  # Center
    draw_arrow(ax, 5, 6.5, 8, 5)  # Right

    draw_box(ax, 0.5, 4, 3, 1, "T, A, S (Neutral)\nR5 Likely", color="#C8E6C9")
    draw_box(ax, 3.5, 4, 3, 1, "I, V, L (Bulky)\nCheck Pos 11/25", color="#FFF9C4")
    draw_box(ax, 6.5, 4, 3, 1, "R, K, H (Basic)\nX4 Likely", color="#FFCDD2")

    ax.set_title("Tropism Clinical Flowchart", fontsize=14)
    save_plot("17_tropism_clinical_flowchart.png")


def plot_18_transition_pathway():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.arrow(1, 4, 10, 0, head_width=0.2, color="black")  # Timeline

    # Stages
    ax.text(2, 4.2, "EARLY\n(R5)", ha="center")
    ax.text(6, 4.2, "INTERMEDIATE\n(Dual)", ha="center")
    ax.text(10, 4.2, "LATE\n(X4)", ha="center")

    # Text bubbles below
    draw_box(ax, 1, 2, 2, 1.5, "Pos 11: N\nPos 22: T\nR5 Tropism", color="#E0E0E0")
    draw_box(ax, 5, 2, 2, 1.5, "Pos 11: R\nPos 22: T\nTransitional", color="#FFF9C4")
    draw_box(ax, 9, 2, 2, 1.5, "Pos 11: R\nPos 22: R\nX4 Tropism", color="#FFCDD2")

    ax.text(6, 1, "Pos 22 T->R Switch is Tipping Point", ha="center", style="italic")
    ax.set_title("R5 to X4 Transition Pathway", fontsize=14)
    save_plot("18_tropism_transition_pathway.png")


def plot_20_env_trimer():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Schematic Trimer (Mushroom shape)
    lobe1 = patches.Circle((3, 6), 1.5, color="#B3E5FC")
    lobe2 = patches.Circle((7, 6), 1.5, color="#B3E5FC")
    lobe3 = patches.Circle((5, 8), 1.5, color="#B3E5FC")

    stem = patches.Rectangle((3.5, 2), 3, 4, color="#E0E0E0")  # gp41

    ax.add_patch(stem)
    ax.add_patch(lobe1)
    ax.add_patch(lobe2)
    ax.add_patch(lobe3)

    ax.text(5, 8, "V2-Apex\n(PG9)", ha="center", fontsize=8)
    ax.text(3, 6, "CD4bs\n(VRC01)", ha="center", fontsize=8)
    ax.text(7, 6, "V3-Glycan\n(PGT121)", ha="center", fontsize=8)
    ax.text(5, 3, "MPER\n(10E8)", ha="center", fontsize=8)

    ax.set_title("Env Trimer Epitopes (Schematic)", fontsize=14)
    save_plot("20_env_trimer_epitopes.png")


def plot_24_bnab_selection():
    # Just a text table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    data = [
        ["Prevention", "3BNC117 + 10-1074", "High breadth (91%)"],
        ["Reservoir", "N6 + 10E8", "Broadest (93%)"],
        ["Cure", "Triple Combo", "Minimize escape (96%)"],
        ["Pediatric", "VRC01-LS", "Long half-life"],
    ]
    cols = ["Goal", "Recommended", "Rationale"]
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="left")
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax.set_title("bnAb Clinical Selection Guide", fontsize=14, fontweight="bold", y=0.8)
    save_plot("24_bnab_clinical_selection.png")


def plot_25_genome_safe():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_ylim(0, 2)
    ax.set_xlim(0, 100)
    ax.axis("off")

    # Safe Zones (Gag)
    patches.Rectangle((10, 0.5), 20, 1, color="#4CAF50")  # Gag
    ax.add_patch(patches.Rectangle((10, 0.5), 15, 1, color="#4CAF50", label="Safe"))

    # Unsafe (Pol PR/RT)
    ax.add_patch(patches.Rectangle((25, 0.5), 20, 1, color="#EF5350", label="Unsafe"))

    # Pol IN (Safe)
    ax.add_patch(patches.Rectangle((45, 0.5), 10, 1, color="#4CAF50"))

    # Env (Unsafe)
    ax.add_patch(patches.Rectangle((60, 0.5), 25, 1, color="#EF5350"))

    # Nef (Split)
    ax.add_patch(patches.Rectangle((90, 0.5), 5, 1, color="#FFE0B2"))

    ax.text(17, 1, "Gag (Safe)", color="white", fontweight="bold", ha="center")
    ax.text(35, 1, "Pol-RT (Unsafe)", color="white", fontweight="bold", ha="center")

    ax.set_title("Safe vs Unsafe Vaccine Targets", fontsize=14)
    save_plot("25_safe_unsafe_genome_map.png")


def plot_26_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    draw_box(ax, 3.5, 7, 3, 2, "Overlapping Position\n(e.g. K103N)")

    draw_arrow(ax, 2, 5, 3.5, 7.5)  # From Left
    draw_arrow(ax, 8, 5, 6.5, 7.5)  # From Right

    draw_box(ax, 0.5, 3, 3, 2, "Drug Pressure\nSelects FOR", color="#FFCDD2")
    draw_box(ax, 6.5, 3, 3, 2, "Immune Pressure\nSelects AGAINST", color="#C8E6C9")

    ax.text(5, 1, "Conflict: Treatment drives immune escape", ha="center", style="italic")
    ax.set_title("Drug vs Immune Trade-off", fontsize=14)
    save_plot("26_tradeoff_conflict_diagram.png")


def plot_30_vaccine_framework():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    draw_box(ax, 3.5, 9, 3, 0.8, "Candidate Epitope")
    draw_arrow(ax, 5, 9, 5, 8)

    draw_box(ax, 3, 7, 4, 1, "Step 1: Conservation >90%?")
    draw_arrow(ax, 5, 7, 5, 6)  # Yes

    draw_box(ax, 3, 5, 4, 1, "Step 2: No Resistance Overlap?")
    draw_arrow(ax, 5, 5, 5, 4)  # Yes

    draw_box(ax, 3, 3, 4, 1, "Step 3: Escape Velocity Low?")
    draw_arrow(ax, 5, 3, 5, 2)  # Yes

    draw_box(ax, 3.5, 1, 3, 1, "ACCEPT\n(Safe Target)", color="#4CAF50")

    # Rejects
    ax.text(8, 7.5, "NO -> Reject")
    ax.text(8, 5.5, "NO -> Reject")
    ax.text(8, 3.5, "NO -> Reject")

    ax.set_title("Vaccine Target Selection Framework", fontsize=14)
    save_plot("30_vaccine_design_framework.png")


if __name__ == "__main__":
    print("Generating flowcharts...")
    plot_06_barrier_decision()
    plot_17_tropism_flowchart()
    plot_18_transition_pathway()
    plot_20_env_trimer()
    plot_24_bnab_selection()
    plot_25_genome_safe()
    plot_26_tradeoff()
    plot_30_vaccine_framework()
    print("Done.")
