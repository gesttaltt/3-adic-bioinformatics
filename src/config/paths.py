# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized path configuration - Single Source of Truth for all paths.

This module provides Path objects for all project directories, with support
for environment variable overrides. Use these instead of hardcoding paths.

Usage:
    from src.config.paths import (
        PROJECT_ROOT,
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        OUTPUT_DIR, RESULTS_DIR, CHECKPOINTS_DIR,
    )

    # Use in scripts
    output_file = RESULTS_DIR / "analysis.json"

    # All paths support environment variable overrides
    # Set TERNARY_DATA_DIR=/custom/path to override DATA_DIR

Environment Variables:
    TERNARY_PROJECT_ROOT: Override project root detection
    TERNARY_DATA_DIR: Override data directory
    TERNARY_OUTPUT_DIR: Override output directory
    TERNARY_CONFIG_DIR: Override config directory
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root by looking for marker files.

    Searches upward from this file's location for common project markers.

    Returns:
        Path to project root directory
    """
    # Check environment variable first
    env_root = os.environ.get("TERNARY_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Start from this file's directory
    current = Path(__file__).resolve().parent

    # Search upward for project markers
    markers = ["pyproject.toml", "setup.py", ".git", "LICENSE"]

    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    # Fallback: assume src/config/paths.py structure
    return Path(__file__).resolve().parents[2]


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = _find_project_root()

# =============================================================================
# CONFIGURATION DIRECTORIES
# =============================================================================

CONFIG_DIR = Path(os.environ.get("TERNARY_CONFIG_DIR", PROJECT_ROOT / "src" / "configs"))

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_DIR = Path(os.environ.get("TERNARY_DATA_DIR", PROJECT_ROOT / "data"))

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CACHE_DIR = DATA_DIR / "cache"

# =============================================================================
# OUTPUT DIRECTORIES (all generated artifacts)
# =============================================================================

OUTPUT_DIR = Path(os.environ.get("TERNARY_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Output subdirectories
RESULTS_DIR = OUTPUT_DIR / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"  # Main checkpoint storage
RUNS_DIR = OUTPUT_DIR / "runs"
REPORTS_DIR = OUTPUT_DIR / "reports"
VIZ_DIR = OUTPUT_DIR / "visualizations"
LOGS_DIR = OUTPUT_DIR / "logs"

# =============================================================================
# LEGACY DIRECTORIES (for backward compatibility during migration)
# =============================================================================

# These map old paths to new structure
# Use these during migration, then remove

LEGACY_RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_RUNS_DIR = PROJECT_ROOT / "runs"
LEGACY_REPORTS_DIR = PROJECT_ROOT / "reports"
# Note: LEGACY_CHECKPOINTS_DIR removed - CHECKPOINTS_DIR now points to checkpoints/

# =============================================================================
# DOCUMENTATION DIRECTORIES
# =============================================================================

DOCS_DIR = PROJECT_ROOT / "docs"
DOCUMENTATION_DIR = PROJECT_ROOT / "DOCUMENTATION"  # Legacy

# =============================================================================
# SOURCE AND TEST DIRECTORIES
# =============================================================================

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = SRC_DIR / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESEARCH_DIR = PROJECT_ROOT / "research"

# =============================================================================
# DELIVERABLES
# =============================================================================

DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist.

    Args:
        *dirs: Path objects to create
    """
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root
    """
    return PROJECT_ROOT


def get_checkpoint_path(name: str, version: str | None = None) -> Path:
    """Get path to a model checkpoint.

    Args:
        name: Checkpoint name (e.g., "ternary_vae")
        version: Optional version string (e.g., "v5_11")

    Returns:
        Path to checkpoint file
    """
    if version:
        return CHECKPOINTS_DIR / version / f"{name}.pt"
    return CHECKPOINTS_DIR / f"{name}.pt"


def get_results_path(name: str, ext: str = "json") -> Path:
    """Get path for a results file.

    Args:
        name: Results file name (without extension)
        ext: File extension (default: json)

    Returns:
        Path to results file
    """
    return RESULTS_DIR / f"{name}.{ext}"


def get_data_path(filename: str, processed: bool = True) -> Path:
    """Get path to a data file.

    Args:
        filename: Data file name
        processed: If True, use processed directory; else raw

    Returns:
        Path to data file
    """
    base = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR
    return base / filename


def get_research_path(subpath: str = "") -> Path:
    """Get path to research directory or subdirectory.

    Args:
        subpath: Optional subdirectory path (e.g., "bioinformatics/hiv")

    Returns:
        Path to research directory or subdirectory
    """
    if subpath:
        return RESEARCH_DIR / subpath
    return RESEARCH_DIR


def get_config_path(config_name: str) -> Path | None:
    """Get path to a configuration file.

    Args:
        config_name: Name of config file (e.g., "ternary.yaml")

    Returns:
        Path to config file if exists, None otherwise
    """
    config_path = CONFIG_DIR / config_name
    return config_path if config_path.exists() else None


def list_research_experiments() -> list[str]:
    """List available research experiments.

    Returns:
        List of experiment directory names
    """
    if not RESEARCH_DIR.exists():
        return []

    experiments = []
    for item in RESEARCH_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            experiments.append(item.name)
    return sorted(experiments)


def list_datasets() -> list[str]:
    """List available datasets in data/ directory.

    Returns:
        List of dataset directory names
    """
    if not DATA_DIR.exists():
        return []

    datasets = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            datasets.append(item.name)
    return sorted(datasets)


def resolve_legacy_path(path: str | Path) -> Path:
    """Resolve a legacy path to the new structure.

    This function helps migrate old hardcoded paths to the new structure.

    Args:
        path: Legacy path string or Path object

    Returns:
        Resolved path in new structure
    """
    path_str = str(path)

    # Map legacy paths to new structure
    mappings = {
        "results/": str(RESULTS_DIR) + "/",
        "checkpoints/": str(CHECKPOINTS_DIR) + "/",
        "runs/": str(RUNS_DIR) + "/",
        "reports/": str(REPORTS_DIR) + "/",
        "outputs/viz/": str(VIZ_DIR) + "/",
        "outputs/": str(OUTPUT_DIR) + "/",
        "data/raw/": str(RAW_DATA_DIR) + "/",
        "data/processed/": str(PROCESSED_DATA_DIR) + "/",
        "data/": str(DATA_DIR) + "/",
    }

    for old_prefix, new_prefix in mappings.items():
        if path_str.startswith(old_prefix):
            return Path(path_str.replace(old_prefix, new_prefix, 1))

    # If no mapping found, return as-is resolved against project root
    return PROJECT_ROOT / path


# =============================================================================
# INITIALIZATION
# =============================================================================


def init_project_dirs() -> None:
    """Initialize all project directories.

    Call this at project startup to ensure all directories exist.
    """
    ensure_dirs(
        # Data directories
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        CACHE_DIR,
        # Output directories
        RESULTS_DIR,
        CHECKPOINTS_DIR,
        RUNS_DIR,
        REPORTS_DIR,
        VIZ_DIR,
        LOGS_DIR,
    )


__all__ = [
    # Root
    "PROJECT_ROOT",
    # Config
    "CONFIG_DIR",
    # Data
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "CACHE_DIR",
    # Output
    "OUTPUT_DIR",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "VIZ_DIR",
    "LOGS_DIR",
    # Legacy (for migration)
    "LEGACY_RESULTS_DIR",
    "LEGACY_RUNS_DIR",
    "LEGACY_REPORTS_DIR",
    # Docs
    "DOCS_DIR",
    "DOCUMENTATION_DIR",
    # Source
    "SRC_DIR",
    "TESTS_DIR",
    "SCRIPTS_DIR",
    "NOTEBOOKS_DIR",
    "RESEARCH_DIR",
    # Deliverables
    "DELIVERABLES_DIR",
    # Functions
    "ensure_dirs",
    "get_project_root",
    "get_checkpoint_path",
    "get_results_path",
    "get_data_path",
    "get_research_path",
    "get_config_path",
    "list_research_experiments",
    "list_datasets",
    "resolve_legacy_path",
    "init_project_dirs",
]
