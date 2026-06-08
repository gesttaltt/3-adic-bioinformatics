#!/usr/bin/env python3
"""
HIV Dataset Downloader with Custom DNS Resolution

Bypasses local DNS issues by using Google DNS (8.8.8.8) directly.
"""

import json
import os
import random
import socket
import ssl
import struct
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external"

# DNS Cache
DNS_CACHE = {}


def dns_query(domain: str, dns_server: str = "8.8.8.8") -> str:
    """Query DNS server directly."""
    if domain in DNS_CACHE:
        return DNS_CACHE[domain]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10)

    # Build DNS query
    transaction_id = random.randint(0, 65535)
    flags = 0x0100  # Standard query
    questions = 1
    header = struct.pack(">HHHHHH", transaction_id, flags, questions, 0, 0, 0)

    # Build question
    question = b""
    for part in domain.split("."):
        question += bytes([len(part)]) + part.encode()
    question += b"\x00"  # End of name
    question += struct.pack(">HH", 1, 1)  # Type A, Class IN

    # Send query
    sock.sendto(header + question, (dns_server, 53))
    response, _ = sock.recvfrom(512)
    sock.close()

    # Parse response
    offset = 12
    while response[offset] != 0:
        offset += response[offset] + 1
    offset += 5

    # Read answer
    if len(response) > offset + 12:
        offset += 12
        if len(response) >= offset + 4:
            ip = ".".join(str(b) for b in response[offset : offset + 4])
            DNS_CACHE[domain] = ip
            return ip

    raise Exception(f"Failed to resolve {domain}")


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download file using custom DNS resolution."""
    try:
        # Parse URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.netloc

        # Resolve DNS
        ip = dns_query(host)
        print(f"    Resolved {host} -> {ip}")

        # Create custom opener that connects to IP but sends Host header
        class CustomHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(self.get_connection, req)

            def get_connection(self, host, timeout=300):
                return socket.create_connection((ip, 80), timeout)

        class CustomHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(self.get_connection, req)

            def get_connection(self, host, timeout=300):
                conn = socket.create_connection((ip, 443), timeout)
                context = ssl.create_default_context()
                return context.wrap_socket(conn, server_hostname=host)

        # Use requests library if available (handles this better)
        try:
            import requests

            # Monkey-patch socket.getaddrinfo temporarily
            original_getaddrinfo = socket.getaddrinfo

            def custom_getaddrinfo(host, port, *args, **kwargs):
                if host in DNS_CACHE:
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS_CACHE[host], port))]
                return original_getaddrinfo(host, port, *args, **kwargs)

            socket.getaddrinfo = custom_getaddrinfo

            try:
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                size_kb = dest_path.stat().st_size / 1024
                print(f"    -> {dest_path.name} ({size_kb:.1f} KB)")
                return True
            finally:
                socket.getaddrinfo = original_getaddrinfo

        except ImportError:
            # Fallback to urllib with modified connection
            print("    Using urllib fallback...")
            opener = urllib.request.build_opener(CustomHTTPSHandler())
            urllib.request.install_opener(opener)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest_path)

            size_kb = dest_path.stat().st_size / 1024
            print(f"    -> {dest_path.name} ({size_kb:.1f} KB)")
            return True

    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def download_csv_files():
    """Download CSV files."""
    print("\n" + "=" * 60)
    print("DOWNLOADING CSV FILES")
    print("=" * 60)

    csv_dir = DATA_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("corgis_aids.csv", "https://corgis-edu.github.io/corgis/datasets/csv/aids/aids.csv"),
    ]

    for name, url in files:
        dest = csv_dir / name
        if dest.exists():
            print(f"  [SKIP] {name} already exists")
            continue
        print(f"  Downloading: {name}")
        download_file(url, dest, name)


def download_zenodo_datasets():
    """Download Zenodo datasets."""
    print("\n" + "=" * 60)
    print("DOWNLOADING ZENODO DATASETS")
    print("=" * 60)

    zenodo_dir = DATA_DIR / "zenodo"

    datasets = [
        ("cview_gp120", "6475667"),
        ("hiv_genome_to_genome", "7139"),
    ]

    for name, record_id in datasets:
        dataset_dir = zenodo_dir / name
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  [SKIP] {name} already exists")
            continue

        print(f"\n  Dataset: {name} (Record {record_id})")

        # Resolve zenodo.org
        try:
            ip = dns_query("zenodo.org")
            print(f"    Resolved zenodo.org -> {ip}")
        except Exception as e:
            print(f"    [ERROR] DNS resolution failed: {e}")
            continue

        # Get record metadata
        api_url = f"https://zenodo.org/api/records/{record_id}"
        try:
            import requests

            # Patch DNS
            original_getaddrinfo = socket.getaddrinfo

            def custom_getaddrinfo(host, port, *args, **kwargs):
                if host in DNS_CACHE:
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS_CACHE[host], port))]
                return original_getaddrinfo(host, port, *args, **kwargs)

            socket.getaddrinfo = custom_getaddrinfo

            try:
                response = requests.get(api_url, timeout=30)
                record = response.json()

                dataset_dir.mkdir(parents=True, exist_ok=True)

                for file_info in record.get("files", []):
                    file_url = file_info["links"]["self"]
                    file_name = file_info["key"]
                    file_path = dataset_dir / file_name

                    if file_path.exists():
                        print(f"    [SKIP] {file_name}")
                        continue

                    # Skip large files
                    size_mb = file_info.get("size", 0) / (1024 * 1024)
                    if size_mb > 100:
                        print(f"    [SKIP] {file_name} ({size_mb:.1f} MB - too large)")
                        continue

                    print(f"    Downloading: {file_name} ({size_mb:.2f} MB)")

                    r = requests.get(file_url, timeout=120, stream=True)
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("      -> Success")

            finally:
                socket.getaddrinfo = original_getaddrinfo

        except Exception as e:
            print(f"    [ERROR] {e}")


def download_github_repos():
    """Clone GitHub repositories using custom DNS."""
    print("\n" + "=" * 60)
    print("DOWNLOADING GITHUB REPOSITORIES")
    print("=" * 60)

    github_dir = DATA_DIR / "github"
    github_dir.mkdir(parents=True, exist_ok=True)

    repos = [
        ("HIV-data", "https://github.com/malabz/HIV-data.git"),
        ("HIV-DRM-machine-learning", "https://github.com/lucblassel/HIV-DRM-machine-learning.git"),
        ("HIV-1_Paper", "https://github.com/pauloluniyi/HIV-1_Paper.git"),
    ]

    # Resolve github.com first
    try:
        github_ip = dns_query("github.com")
        print(f"  Resolved github.com -> {github_ip}")
    except Exception as e:
        print(f"  [ERROR] Cannot resolve github.com: {e}")
        return

    # Add to hosts-style resolution
    for name, url in repos:
        repo_dir = github_dir / name
        if repo_dir.exists():
            print(f"  [SKIP] {name} already exists")
            continue

        print(f"\n  Cloning: {name}")

        # Try using git with explicit IP (via config)
        try:
            # Set git to use resolved IP
            env = os.environ.copy()

            # Use git's http.proxy or direct clone
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

            if result.returncode == 0:
                print("    -> Success")
            else:
                print(f"    [ERROR] {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print("    [ERROR] Clone timed out")
        except Exception as e:
            print(f"    [ERROR] {e}")


def download_huggingface():
    """Download HuggingFace datasets."""
    print("\n" + "=" * 60)
    print("DOWNLOADING HUGGING FACE DATASETS")
    print("=" * 60)

    hf_dir = DATA_DIR / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("human_hiv_ppi", "damlab/human_hiv_ppi"),
        ("HIV_V3_coreceptor", "damlab/HIV_V3_coreceptor"),
    ]

    # Resolve huggingface.co
    try:
        hf_ip = dns_query("huggingface.co")
        print(f"  Resolved huggingface.co -> {hf_ip}")

        # Also resolve cdn-lfs
        cdn_ip = dns_query("cdn-lfs.huggingface.co")
        print(f"  Resolved cdn-lfs.huggingface.co -> {cdn_ip}")
    except Exception as e:
        print(f"  [WARN] DNS resolution issue: {e}")

    try:
        from huggingface_hub import snapshot_download

        # Patch DNS
        original_getaddrinfo = socket.getaddrinfo

        def custom_getaddrinfo(host, port, *args, **kwargs):
            if host in DNS_CACHE:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS_CACHE[host], port))]
            # Try to resolve unknown hosts
            try:
                ip = dns_query(host)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
            except:
                pass
            return original_getaddrinfo(host, port, *args, **kwargs)

        socket.getaddrinfo = custom_getaddrinfo

        try:
            for name, repo in datasets:
                local_dir = hf_dir / name
                if local_dir.exists() and any(local_dir.iterdir()):
                    print(f"  [SKIP] {name} already exists")
                    continue

                print(f"\n  Downloading: {name}")
                try:
                    snapshot_download(
                        repo,
                        repo_type="dataset",
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                    )
                    print("    -> Success")
                except Exception as e:
                    print(f"    [ERROR] {e}")

        finally:
            socket.getaddrinfo = original_getaddrinfo

    except ImportError:
        print("  [WARN] huggingface_hub not installed")
        print("  Run: pip install huggingface_hub")


def create_index():
    """Create dataset index."""
    print("\n" + "=" * 60)
    print("CREATING DATASET INDEX")
    print("=" * 60)

    index = {"description": "HIV External Datasets", "sources": {}}

    for source_dir in DATA_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        if source_dir.name.startswith("."):
            continue

        datasets = []
        for item in source_dir.iterdir():
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                if file_count > 0:
                    datasets.append(
                        {
                            "name": item.name,
                            "type": "directory",
                            "file_count": file_count,
                            "size_mb": round(total_size / (1024 * 1024), 2),
                        }
                    )
            else:
                datasets.append(
                    {
                        "name": item.name,
                        "type": "file",
                        "size_mb": round(item.stat().st_size / (1024 * 1024), 2),
                    }
                )

        if datasets:
            index["sources"][source_dir.name] = datasets

    index_path = DATA_DIR / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Index saved to: {index_path}")

    # Print summary
    print("\n  Downloaded datasets:")
    for source, datasets in index["sources"].items():
        if datasets:
            print(f"\n  [{source}]")
            for ds in datasets:
                if ds["type"] == "directory":
                    print(f"    - {ds['name']}: {ds['file_count']} files ({ds['size_mb']} MB)")
                else:
                    print(f"    - {ds['name']}: {ds['size_mb']} MB")


def main():
    print("=" * 60)
    print("HIV DATASET DOWNLOADER (Custom DNS)")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print("Using Google DNS: 8.8.8.8")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Test DNS resolution
    print("\nTesting DNS resolution...")
    for domain in ["github.com", "zenodo.org", "huggingface.co"]:
        try:
            ip = dns_query(domain)
            print(f"  {domain} -> {ip}")
        except Exception as e:
            print(f"  {domain} -> FAILED: {e}")

    # Download everything
    download_csv_files()
    download_zenodo_datasets()
    download_github_repos()
    download_huggingface()

    # Create index
    create_index()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
