# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Manifold Resolution Benchmark
Measures the "sharpness" of the continuous learned manifold vs discrete ternary space
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import BenchmarkBase, create_v5_6_model, get_device, load_checkpoint_safe, load_config, save_results

from src.config.paths import CHECKPOINTS_DIR


class ManifoldResolutionBenchmark(BenchmarkBase):
    """Measures resolution quality between discrete operations and continuous latent space"""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        super().__init__(model, device)

    @torch.no_grad()
    def measure_reconstruction_fidelity(self, vae="A", batch_size=256) -> dict:
        """Measure exact reconstruction accuracy"""
        correct = 0
        total = 0
        bit_errors = []

        for i in range(0, self.n_ops, batch_size):
            batch = self.all_ops[i : i + batch_size]

            # Encode-decode
            if vae == "A":
                mu, logvar = self.model.encoder_A(batch)
                z = mu  # Use mean for deterministic reconstruction
                logits = self.model.decoder_A(z)
            else:
                mu, logvar = self.model.encoder_B(batch)
                z = mu
                logits = self.model.decoder_B(z)

            # Convert to operations
            recon = torch.argmax(logits, dim=-1) - 1  # {0,1,2} -> {-1,0,1}

            # Exact match
            matches = (recon == batch).all(dim=1)
            correct += matches.sum().item()
            total += len(batch)

            # Bit-wise errors
            bit_diffs = (recon != batch).sum(dim=1)
            bit_errors.extend(bit_diffs.cpu().numpy())

        bit_errors = np.array(bit_errors)

        return {
            "exact_match_rate": correct / total,
            "mean_bit_error": bit_errors.mean(),
            "median_bit_error": np.median(bit_errors),
            "max_bit_error": bit_errors.max(),
            "zero_error_count": (bit_errors == 0).sum(),
            "error_histogram": {
                int(k): int(v) for k, v in zip(*np.unique(bit_errors, return_counts=True), strict=False)
            },
        }

    @torch.no_grad()
    def measure_latent_separation(self, vae="A", batch_size=256) -> dict:
        """Measure how well-separated operations are in latent space"""
        # Encode all operations
        latents = []
        for i in range(0, self.n_ops, batch_size):
            batch = self.all_ops[i : i + batch_size]
            if vae == "A":
                mu, _ = self.model.encoder_A(batch)
            else:
                mu, _ = self.model.encoder_B(batch)
            latents.append(mu)

        latents = torch.cat(latents, dim=0)  # (19683, latent_dim)

        # Compute pairwise distances (sample for efficiency)
        sample_size = min(1000, self.n_ops)
        indices = torch.randperm(self.n_ops)[:sample_size]
        sample_latents = latents[indices]

        # Pairwise L2 distances
        dists = torch.cdist(sample_latents, sample_latents)

        # Remove diagonal
        mask = torch.eye(sample_size, device=self.device).bool()
        dists_no_diag = dists[~mask]

        return {
            "mean_distance": dists_no_diag.mean().item(),
            "min_distance": dists_no_diag.min().item(),
            "std_distance": dists_no_diag.std().item(),
            "latent_dim": latents.shape[1],
            "latent_norm_mean": torch.norm(latents, dim=1).mean().item(),
            "latent_norm_std": torch.norm(latents, dim=1).std().item(),
        }

    @torch.no_grad()
    def measure_sampling_coverage(self, vae="A", n_samples=50000, batch_size=1000) -> dict:
        """Measure what fraction of discrete operations can be sampled"""
        sampled_ops = set()

        for _ in range(n_samples // batch_size):
            # Sample from prior
            samples = self.model.sample(batch_size, self.device, use_vae=vae)

            # Convert to tuples for set storage
            for sample in samples:
                op_tuple = tuple(sample.cpu().numpy().astype(int))
                sampled_ops.add(op_tuple)

        coverage = len(sampled_ops) / self.n_ops

        return {
            "unique_sampled": len(sampled_ops),
            "coverage_rate": coverage,
            "n_samples": n_samples,
            "diversity": len(sampled_ops) / n_samples,  # unique / total
        }

    @torch.no_grad()
    def measure_interpolation_quality(self, vae="A", n_pairs=100) -> dict:
        """Measure smoothness of interpolation between operations"""
        # Sample random pairs
        indices = torch.randperm(self.n_ops)[: n_pairs * 2].reshape(n_pairs, 2)

        valid_interpolations = 0
        mean_errors = []

        for idx1, idx2 in indices:
            op1, op2 = (
                self.all_ops[idx1 : idx1 + 1],
                self.all_ops[idx2 : idx2 + 1],
            )

            # Encode
            if vae == "A":
                mu1, _ = self.model.encoder_A(op1)
                mu2, _ = self.model.encoder_A(op2)
            else:
                mu1, _ = self.model.encoder_B(op1)
                mu2, _ = self.model.encoder_B(op2)

            # Interpolate in latent space
            alphas = torch.linspace(0, 1, 11, device=self.device)
            errors = []

            for alpha in alphas[1:-1]:  # Exclude endpoints
                z_interp = (1 - alpha) * mu1 + alpha * mu2

                # Decode
                logits = self.model.decoder_A(z_interp) if vae == "A" else self.model.decoder_B(z_interp)

                recon = torch.argmax(logits, dim=-1) - 1

                # Check if valid operation
                if tuple(recon[0].cpu().numpy().astype(int)) in set(
                    tuple(op.cpu().numpy().astype(int)) for op in self.all_ops
                ):
                    valid_interpolations += 1

                # Measure error relative to linear interpolation in discrete space
                discrete_interp = torch.round((1 - alpha) * op1 + alpha * op2)
                error = (recon - discrete_interp).abs().float().mean()
                errors.append(error.item())

            mean_errors.append(np.mean(errors))

        return {
            "valid_interpolation_rate": valid_interpolations / (n_pairs * 9),  # 9 intermediate points
            "mean_interpolation_error": np.mean(mean_errors),
            "std_interpolation_error": np.std(mean_errors),
        }

    @torch.no_grad()
    def measure_nearest_neighbor_consistency(self, vae="A", n_queries=500) -> dict:
        """Measure if nearest neighbors in latent space map to similar operations"""
        # Encode all operations
        latents = []
        batch_size = 256
        for i in range(0, self.n_ops, batch_size):
            batch = self.all_ops[i : i + batch_size]
            if vae == "A":
                mu, _ = self.model.encoder_A(batch)
            else:
                mu, _ = self.model.encoder_B(batch)
            latents.append(mu)

        latents = torch.cat(latents, dim=0)

        # Sample query points
        indices = torch.randperm(self.n_ops)[:n_queries]

        hamming_dists = []
        latent_dists = []

        for idx in indices:
            query_latent = latents[idx : idx + 1]
            query_op = self.all_ops[idx : idx + 1]

            # Find nearest neighbor in latent space
            dists = torch.cdist(query_latent, latents)[0]
            dists[idx] = float("inf")  # Exclude self
            nn_idx = torch.argmin(dists)

            # Compute Hamming distance in operation space
            nn_op = self.all_ops[nn_idx : nn_idx + 1]
            hamming = (query_op != nn_op).sum().item()

            hamming_dists.append(hamming)
            latent_dists.append(dists[nn_idx].item())

        return {
            "mean_hamming_to_nn": np.mean(hamming_dists),
            "median_hamming_to_nn": np.median(hamming_dists),
            "mean_latent_dist_to_nn": np.mean(latent_dists),
            "hamming_1_rate": (np.array(hamming_dists) == 1).mean(),  # Single bit flip
        }

    @torch.no_grad()
    def measure_manifold_dimensionality(self, vae="A", n_samples=1000) -> dict:
        """Estimate effective dimensionality of learned manifold"""
        # Sample operations
        indices = torch.randperm(self.n_ops)[:n_samples]
        batch = self.all_ops[indices]

        # Encode
        if vae == "A":
            mu, logvar = self.model.encoder_A(batch)
        else:
            mu, logvar = self.model.encoder_B(batch)

        # PCA on latent representations
        latents = mu.cpu().numpy()
        centered = latents - latents.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Effective dimensionality (participation ratio)
        participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues**2).sum()

        # Cumulative variance explained
        cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
        dim_95 = np.argmax(cumvar >= 0.95) + 1
        dim_99 = np.argmax(cumvar >= 0.99) + 1

        return {
            "nominal_dim": len(eigenvalues),
            "effective_dim": participation_ratio,
            "dim_for_95pct_var": dim_95,
            "dim_for_99pct_var": dim_99,
            "max_eigenvalue": eigenvalues[0],
            "min_eigenvalue": eigenvalues[-1],
            "eigenvalue_ratio": eigenvalues[0] / eigenvalues[-1],
        }

    def run_full_benchmark(self) -> dict:
        """Run all benchmark measurements"""
        results = {
            "model_info": {
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "latent_dim": self.model.latent_dim,
                "n_operations": self.n_ops,
            }
        }

        print("Benchmarking VAE-A...")
        results["vae_a"] = {
            "reconstruction": self.measure_reconstruction_fidelity("A"),
            "latent_separation": self.measure_latent_separation("A"),
            "sampling_coverage": self.measure_sampling_coverage("A", n_samples=50000),
            "interpolation": self.measure_interpolation_quality("A", n_pairs=100),
            "nearest_neighbor": self.measure_nearest_neighbor_consistency("A", n_queries=500),
            "dimensionality": self.measure_manifold_dimensionality("A", n_samples=1000),
        }

        print("Benchmarking VAE-B...")
        results["vae_b"] = {
            "reconstruction": self.measure_reconstruction_fidelity("B"),
            "latent_separation": self.measure_latent_separation("B"),
            "sampling_coverage": self.measure_sampling_coverage("B", n_samples=50000),
            "interpolation": self.measure_interpolation_quality("B", n_pairs=100),
            "nearest_neighbor": self.measure_nearest_neighbor_consistency("B", n_queries=500),
            "dimensionality": self.measure_manifold_dimensionality("B", n_samples=1000),
        }

        # Compute manifold resolution score
        results["resolution_score"] = self.compute_resolution_score(results)

        return results

    def compute_resolution_score(self, results: dict) -> dict:
        """Compute aggregate resolution quality score"""

        def score_vae(vae_results):
            # Higher is better for all normalized metrics
            recon_score = vae_results["reconstruction"]["exact_match_rate"]
            coverage_score = vae_results["sampling_coverage"]["coverage_rate"]
            interp_score = vae_results["interpolation"]["valid_interpolation_rate"]
            nn_score = vae_results["nearest_neighbor"]["hamming_1_rate"]  # Close neighbors

            # Dimensionality efficiency (want effective_dim close to but less than nominal)
            dim_ratio = vae_results["dimensionality"]["effective_dim"] / vae_results["dimensionality"]["nominal_dim"]
            dim_score = 1.0 - abs(dim_ratio - 0.7)  # Target ~70% effective usage

            # Weighted average
            overall = (
                0.30 * recon_score
                + 0.25 * coverage_score
                + 0.20 * interp_score
                + 0.15 * nn_score
                + 0.10 * max(0, dim_score)
            )

            return {
                "reconstruction": recon_score,
                "coverage": coverage_score,
                "interpolation": interp_score,
                "nearest_neighbor": nn_score,
                "dimensionality": dim_score,
                "overall": overall,
            }

        return {
            "vae_a": score_vae(results["vae_a"]),
            "vae_b": score_vae(results["vae_b"]),
            "combined": (score_vae(results["vae_a"])["overall"] + score_vae(results["vae_b"])["overall"]) / 2,
        }


def main():
    # Setup
    config = load_config("src/configs/ternary_v5_6.yaml")
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model
    print("Initializing model...")
    model = create_v5_6_model(config)
    checkpoint = load_checkpoint_safe(model, str(CHECKPOINTS_DIR / "v5_6"), device)

    # Run benchmark
    print("\nRunning manifold resolution benchmark...")
    benchmark = ManifoldResolutionBenchmark(model, device)
    results = benchmark.run_full_benchmark()

    # Save results
    save_results(results, "manifold_resolution", checkpoint.get("epoch", "init"))

    # Print summary
    print("\n" + "=" * 80)
    print("MANIFOLD RESOLUTION SUMMARY")
    print("=" * 80)
    print(f"Model: {results['model_info']['total_params']} parameters")
    print(f"Latent dim: {results['model_info']['latent_dim']}")
    print(f"Operations: {results['model_info']['n_operations']}")

    print("\nVAE-A Resolution Scores:")
    for key, value in results["resolution_score"]["vae_a"].items():
        print(f"  {key:20s}: {value:.4f}")

    print("\nVAE-B Resolution Scores:")
    for key, value in results["resolution_score"]["vae_b"].items():
        print(f"  {key:20s}: {value:.4f}")

    print(f"\nCombined Resolution Score: {results['resolution_score']['combined']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
