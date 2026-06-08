# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CLI Entry Point for Ternary VAE."""

import sys

from src.train import main


def app():
    """Main entry point for the CLI."""
    sys.exit(main())


if __name__ == "__main__":
    app()
