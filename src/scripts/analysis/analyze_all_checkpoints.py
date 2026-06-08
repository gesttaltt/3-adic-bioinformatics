#!/usr/bin/env python3
"""Analyze ALL checkpoints to understand the mess.

This script scans every checkpoint directory and extracts:
- Stored metrics (if any)
- Model architecture info
- Training config
- Status (complete/partial/crashed)

Output: JSON and markdown summary for documentation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def analyze_checkpoint(ckpt_path: Path) -> dict:
    """Analyze a single checkpoint file."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": str(e), "path": str(ckpt_path)}

    info = {
        "path": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "keys": list(ckpt.keys()),
        "file_size_mb": ckpt_path.stat().st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(ckpt_path.stat().st_mtime).isoformat(),
    }

    # Extract epoch
    if "epoch" in ckpt:
        info["epoch"] = ckpt["epoch"]

    # Extract metrics (various formats used across versions)
    metrics = {}

    # Format 1: Direct metrics dict
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        if isinstance(m, dict):
            metrics.update(m)

    # Format 2: Separate metric keys
    for key in [
        "coverage",
        "hierarchy",
        "hierarchy_A",
        "hierarchy_B",
        "radial_corr_A",
        "radial_corr_B",
        "richness",
        "dist_corr",
        "Q",
        "r_v0",
        "r_v9",
        "radius_v0",
        "radius_v9",
    ]:
        if key in ckpt:
            metrics[key] = ckpt[key]

    # Format 3: Nested in eval_metrics
    if "eval_metrics" in ckpt and isinstance(ckpt["eval_metrics"], dict):
        metrics.update(ckpt["eval_metrics"])

    # Format 4: ComprehensiveMetrics object
    if "comprehensive_metrics" in ckpt:
        cm = ckpt["comprehensive_metrics"]
        if isinstance(cm, dict):
            metrics.update(cm)

    info["metrics"] = metrics

    # Extract config if present
    if "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            info["config_keys"] = list(cfg.keys())
            # Extract key training params
            if "epochs" in cfg:
                info["config_epochs"] = cfg["epochs"]
            if "lr" in cfg:
                info["config_lr"] = cfg["lr"]
            if "batch_size" in cfg:
                info["config_batch_size"] = cfg["batch_size"]

    # Extract model architecture info
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        info["n_params"] = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
        # Check for dual encoder
        has_encoder_a = any("encoder_A" in k for k in state)
        has_encoder_b = any("encoder_B" in k for k in state)
        has_projection_a = any("projection_A" in k or "hyperbolic_projection_A" in k for k in state)
        has_projection_b = any("projection_B" in k or "hyperbolic_projection_B" in k for k in state)
        has_controller = any("controller" in k for k in state)

        info["architecture"] = {
            "has_encoder_A": has_encoder_a,
            "has_encoder_B": has_encoder_b,
            "has_projection_A": has_projection_a,
            "has_projection_B": has_projection_b,
            "has_controller": has_controller,
            "dual_encoder": has_encoder_a and has_encoder_b,
            "dual_projection": has_projection_a and has_projection_b,
        }

    # Homeostasis state
    if "homeostasis_state" in ckpt:
        info["has_homeostasis"] = True
        info["homeostasis_summary"] = str(ckpt["homeostasis_state"])[:200]

    return info


def analyze_directory(dir_path: Path) -> dict:
    """Analyze all checkpoints in a directory."""
    result = {
        "name": dir_path.name,
        "path": str(dir_path.relative_to(PROJECT_ROOT)),
        "checkpoints": {},
    }

    # Find checkpoint files
    pt_files = list(dir_path.glob("*.pt"))
    result["n_checkpoints"] = len(pt_files)

    # Analyze key checkpoints
    for name in ["best.pt", "latest.pt", "best_Q.pt"]:
        ckpt_path = dir_path / name
        if ckpt_path.exists():
            result["checkpoints"][name] = analyze_checkpoint(ckpt_path)

    # If no best/latest, check for epoch checkpoints
    if not result["checkpoints"]:
        epoch_files = sorted([f for f in pt_files if "epoch" in f.name])
        if epoch_files:
            result["checkpoints"]["last_epoch"] = analyze_checkpoint(epoch_files[-1])

    # Determine status
    if (dir_path / "best.pt").exists():
        result["status"] = "complete"
    elif (dir_path / "latest.pt").exists():
        result["status"] = "partial"
    elif pt_files:
        result["status"] = "crashed"
    else:
        result["status"] = "empty"

    return result


def categorize_checkpoint(dir_info: dict) -> str:
    """Categorize a checkpoint directory."""
    name = dir_info["name"]

    # Production versions
    if name.startswith("v5_") and "_" not in name[3:].replace("_", "", 1):
        return "PRODUCTION"

    # Named experiments
    if any(x in name for x in ["homeostatic", "homeostasis", "structural"]):
        return "HOMEOSTATIC_EXPERIMENT"
    if any(x in name for x in ["sweep", "stability", "stable"]):
        return "SWEEP_TEST"
    if any(x in name for x in ["test", "validation", "debug"]):
        return "TEST"
    if any(x in name for x in ["progressive", "annealing", "learnable"]):
        return "TRAINING_EXPERIMENT"
    if any(x in name for x in ["radial", "hierarchy", "rich"]):
        return "LOSS_EXPERIMENT"
    if any(x in name for x in ["scratch", "final"]):
        return "FINAL_PUSH"

    return "OTHER"


def extract_key_metrics(ckpt_info: dict) -> dict:
    """Extract key metrics from checkpoint info."""
    metrics = ckpt_info.get("metrics", {})
    return {
        "coverage": metrics.get("coverage", metrics.get("accuracy", None)),
        "hierarchy_A": metrics.get("hierarchy_A", metrics.get("radial_corr_A", None)),
        "hierarchy_B": metrics.get("hierarchy_B", metrics.get("radial_corr_B", metrics.get("hierarchy", None))),
        "richness": metrics.get("richness", None),
        "dist_corr": metrics.get("dist_corr", metrics.get("distance_correlation", None)),
        "Q": metrics.get("Q", None),
        "r_v0": metrics.get("r_v0", metrics.get("radius_v0", None)),
        "r_v9": metrics.get("r_v9", metrics.get("radius_v9", None)),
    }


def main():
    print("=" * 60)
    print("CHECKPOINT ANALYSIS - FULL SCAN")
    print("=" * 60)

    # Get all checkpoint directories
    dirs = sorted([d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(dirs)} checkpoint directories\n")

    all_results = []
    categories = {}

    for i, dir_path in enumerate(dirs):
        print(f"[{i + 1}/{len(dirs)}] Analyzing {dir_path.name}...", end=" ")
        try:
            result = analyze_directory(dir_path)
            result["category"] = categorize_checkpoint(result)
            all_results.append(result)

            # Track categories
            cat = result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

            print(f"[{result['status']}] {result['category']}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"name": dir_path.name, "error": str(e)})

    # Save full results
    output_path = CHECKPOINTS_DIR / "checkpoint_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")

    # Generate summary
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)

    summary_lines = []
    summary_lines.append("# Checkpoint Analysis Summary\n")
    summary_lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    summary_lines.append(f"**Total Directories:** {len(dirs)}\n\n")

    for cat in sorted(categories.keys()):
        items = categories[cat]
        print(f"\n{cat}: {len(items)} directories")
        summary_lines.append(f"## {cat} ({len(items)} directories)\n\n")

        for item in items:
            name = item["name"]
            status = item.get("status", "unknown")

            # Get metrics from best.pt if available
            best = item.get("checkpoints", {}).get("best.pt", {})
            metrics = extract_key_metrics(best)

            # Format metrics string
            metrics_str = []
            if metrics.get("coverage") is not None:
                cov = metrics["coverage"]
                if isinstance(cov, float):
                    metrics_str.append(f"cov={cov * 100:.1f}%")
            if metrics.get("hierarchy_B") is not None:
                metrics_str.append(f"hier_B={metrics['hierarchy_B']:.4f}")
            if metrics.get("richness") is not None:
                metrics_str.append(f"rich={metrics['richness']:.6f}")
            if metrics.get("Q") is not None:
                metrics_str.append(f"Q={metrics['Q']:.3f}")

            metrics_display = ", ".join(metrics_str) if metrics_str else "no metrics"

            print(f"  - {name}: [{status}] {metrics_display}")
            summary_lines.append(f"- **{name}**: [{status}] {metrics_display}\n")

        summary_lines.append("\n")

    # Save summary
    summary_path = CHECKPOINTS_DIR / "CHECKPOINT_ANALYSIS.md"
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\nSummary saved to: {summary_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    statuses = {}
    for r in all_results:
        s = r.get("status", "error")
        statuses[s] = statuses.get(s, 0) + 1

    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    return all_results


if __name__ == "__main__":
    main()
