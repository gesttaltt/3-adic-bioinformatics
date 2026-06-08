import os

import matplotlib.pyplot as plt

OUTPUT_DIR = r"c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\research\bioinformatics\codon_encoder_research\hiv\public_medical_paper\images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {path}")


def plot_21_epitope_class():
    data = [
        ["CD4bs", "72%", "1.1", "Low", "Very High", "Best Breadth"],
        ["MPER", "68%", "1.8", "V.Low", "Very High", ""],
        ["V2-glycan", "58%", "0.69", "Mod", "Moderate", "Best Potency"],
        ["V3-glycan", "51%", "0.75", "Mod", "Moderate", ""],
    ]
    cols = ["Class", "Breadth", "Potency", "Access", "Conservation", "Note"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    ax.set_title("bnAb Epitope Class Comparison", fontsize=14, fontweight="bold", y=0.8)
    save_plot("21_epitope_class_comparison.png")


if __name__ == "__main__":
    plot_21_epitope_class()
