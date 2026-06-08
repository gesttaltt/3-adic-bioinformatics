"""Hugging Face Hub integration for checkpoint storage.

This module provides utilities for uploading and downloading model checkpoints
from Hugging Face Hub, keeping large files out of the git repository.

Usage:
    # Upload checkpoints
    python -m src.utils.checkpoint_hub upload --checkpoint outputs/models/v5_11/best.pt

    # Download checkpoints
    python -m src.utils.checkpoint_hub download --checkpoint v5_11/best.pt

    # List available checkpoints
    python -m src.utils.checkpoint_hub list

Configuration:
    Set HF_REPO_ID environment variable or use default: ai-whisperers/ternary-vae-checkpoints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Hub configuration
DEFAULT_REPO_ID = "ivanpol/ternary-vae-checkpoints"
CHECKPOINT_SUBDIRS = [
    "outputs/models",
    "checkpoints",
    "checkpoints",
]


def get_repo_id() -> str:
    """Get the Hugging Face repo ID from environment or default."""
    return os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)


def upload_checkpoint(
    local_path: str | Path,
    repo_id: str | None = None,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a checkpoint file to Hugging Face Hub.

    Args:
        local_path: Local path to the checkpoint file
        repo_id: Hugging Face repo ID (default from env or DEFAULT_REPO_ID)
        path_in_repo: Path within the repo (default: preserves local structure)
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded file
    """
    from huggingface_hub import HfApi, create_repo

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local_path}")

    repo_id = repo_id or get_repo_id()
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: Could not create repo (may already exist): {e}")

    # Determine path in repo
    if path_in_repo is None:
        # Try to preserve meaningful structure
        for subdir in CHECKPOINT_SUBDIRS:
            if subdir in str(local_path):
                path_in_repo = str(local_path).split(subdir)[-1].lstrip("/\\")
                break
        if path_in_repo is None:
            path_in_repo = local_path.name

    # Upload
    commit_message = commit_message or f"Upload {path_in_repo}"
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
    )

    print(f"Uploaded: {local_path} -> {repo_id}/{path_in_repo}")
    return url


def download_checkpoint(
    path_in_repo: str,
    local_dir: str | Path = "checkpoints",
    repo_id: str | None = None,
    force: bool = False,
) -> Path:
    """Download a checkpoint from Hugging Face Hub.

    Args:
        path_in_repo: Path within the repo (e.g., "v5_11/best.pt")
        local_dir: Local directory to save to
        repo_id: Hugging Face repo ID
        force: Overwrite existing files

    Returns:
        Path to the downloaded file
    """
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or get_repo_id()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / path_in_repo
    if local_path.exists() and not force:
        print(f"Already exists (use --force to overwrite): {local_path}")
        return local_path

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        local_dir=str(local_dir),
    )

    print(f"Downloaded: {repo_id}/{path_in_repo} -> {downloaded}")
    return Path(downloaded)


def list_checkpoints(repo_id: str | None = None) -> list[str]:
    """List all checkpoints in the Hugging Face repo.

    Args:
        repo_id: Hugging Face repo ID

    Returns:
        List of checkpoint paths
    """
    from huggingface_hub import HfApi

    repo_id = repo_id or get_repo_id()
    api = HfApi()

    try:
        files = api.list_repo_files(repo_id=repo_id)
        checkpoints = [f for f in files if f.endswith(".pt")]
        return checkpoints
    except Exception as e:
        print(f"Could not list files: {e}")
        return []


def upload_directory(
    local_dir: str | Path,
    repo_id: str | None = None,
    pattern: str = "*.pt",
) -> list[str]:
    """Upload all checkpoints from a directory.

    Args:
        local_dir: Local directory containing checkpoints
        repo_id: Hugging Face repo ID
        pattern: Glob pattern for files to upload

    Returns:
        List of uploaded URLs
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    files = list(local_dir.rglob(pattern))
    print(f"Found {len(files)} files matching '{pattern}' in {local_dir}")

    urls = []
    for f in files:
        try:
            url = upload_checkpoint(f, repo_id=repo_id)
            urls.append(url)
        except Exception as e:
            print(f"Failed to upload {f}: {e}")

    return urls


def ensure_checkpoint(
    checkpoint_name: str,
    local_dir: str | Path = "checkpoints",
    repo_id: str | None = None,
) -> Path:
    """Ensure a checkpoint is available locally, downloading if needed.

    This is the main function to use in training/inference code.

    Args:
        checkpoint_name: Name of checkpoint (e.g., "v5_11/best.pt")
        local_dir: Local directory for checkpoints
        repo_id: Hugging Face repo ID

    Returns:
        Path to local checkpoint file

    Example:
        from src.utils.checkpoint_hub import ensure_checkpoint

        ckpt_path = ensure_checkpoint("homeostatic_rich/best.pt")
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    """
    local_dir = Path(local_dir)
    local_path = local_dir / checkpoint_name

    if local_path.exists():
        return local_path

    print("Checkpoint not found locally, downloading from Hub...")
    return download_checkpoint(checkpoint_name, local_dir, repo_id)


# CLI
def main():
    parser = argparse.ArgumentParser(description="Manage checkpoints on Hugging Face Hub")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Upload
    upload_parser = subparsers.add_parser("upload", help="Upload checkpoint(s)")
    upload_parser.add_argument("--checkpoint", "-c", help="Single checkpoint file")
    upload_parser.add_argument("--directory", "-d", help="Directory of checkpoints")
    upload_parser.add_argument("--repo-id", help="Hugging Face repo ID")

    # Download
    download_parser = subparsers.add_parser("download", help="Download checkpoint")
    download_parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path in repo")
    download_parser.add_argument("--output", "-o", default="checkpoints", help="Output directory")
    download_parser.add_argument("--repo-id", help="Hugging Face repo ID")
    download_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    # List
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument("--repo-id", help="Hugging Face repo ID")

    args = parser.parse_args()

    if args.command == "upload":
        if args.checkpoint:
            upload_checkpoint(args.checkpoint, repo_id=args.repo_id)
        elif args.directory:
            upload_directory(args.directory, repo_id=args.repo_id)
        else:
            print("Specify --checkpoint or --directory")

    elif args.command == "download":
        download_checkpoint(
            args.checkpoint,
            local_dir=args.output,
            repo_id=args.repo_id,
            force=args.force,
        )

    elif args.command == "list":
        checkpoints = list_checkpoints(repo_id=args.repo_id)
        if checkpoints:
            print(f"\nAvailable checkpoints ({len(checkpoints)}):")
            for c in sorted(checkpoints):
                print(f"  {c}")
        else:
            print("No checkpoints found (repo may not exist yet)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
