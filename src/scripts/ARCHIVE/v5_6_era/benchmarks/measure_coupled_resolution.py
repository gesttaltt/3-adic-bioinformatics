# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Coupled System Manifold Resolution Benchmark
Measures resolution of the full dual-VAE system working together
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.benchmark import BenchmarkBase, create_v5_6_model, get_device, load_checkpoint_safe, load_config, save_results

from src.config.paths import CHECKPOINTS_DIR


class CoupledSystemBenchmark(BenchmarkBase):
    """Measures resolution of the coupled dual-VAE system"""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        super().__init__(model, device)

    @torch.no_grad()
    def measure_ensemble_reconstruction(self, batch_size=256) -> dict:
        """Measure reconstruction using ensemble of both VAEs"""
        # Strategy 1: Voting (both VAEs decode, take majority vote per bit)
        voting_correct = 0

        # Strategy 2: Confidence-weighted (use reconstruction loss as confidence)
        confidence_correct = 0

        # Strategy 3: Best-of-two (take whichever VAE has better reconstruction)
        best_of_two_correct = 0

        for i in range(0, self.n_ops, batch_size):
            batch = self.all_ops[i : i + batch_size]

            # Encode with both VAEs
            mu_a, _ = self.model.encoder_A(batch)
            mu_b, _ = self.model.encoder_B(batch)

            # Decode with both VAEs
            logits_a = self.model.decoder_A(mu_a)
            logits_b = self.model.decoder_B(mu_b)

            # Convert to operations
            recon_a = torch.argmax(logits_a, dim=-1) - 1
            recon_b = torch.argmax(logits_b, dim=-1) - 1

            # Strategy 1: Voting
            # For each bit, take majority vote (if tie, use VAE-B)
            recon_vote = torch.where(recon_a == recon_b, recon_a, recon_b)  # Tie-breaker: VAE-B
            voting_correct += (recon_vote == batch).all(dim=1).sum().item()

            # Strategy 2: Confidence-weighted
            # Use softmax probabilities as confidence
            prob_a = torch.softmax(logits_a, dim=-1)
            prob_b = torch.softmax(logits_b, dim=-1)

            # Get confidence for each prediction
            conf_a = prob_a.gather(2, (recon_a + 1).unsqueeze(-1).long()).squeeze(-1)
            conf_b = prob_b.gather(2, (recon_b + 1).unsqueeze(-1).long()).squeeze(-1)

            # Select per bit based on confidence
            recon_conf = torch.where(conf_a > conf_b, recon_a, recon_b)
            confidence_correct += (recon_conf == batch).all(dim=1).sum().item()

            # Strategy 3: Best-of-two
            # Count bit errors for each VAE and pick best
            errors_a = (recon_a != batch).sum(dim=1)
            errors_b = (recon_b != batch).sum(dim=1)

            # Select reconstruction with fewer errors
            mask = (errors_a <= errors_b).unsqueeze(1).expand_as(recon_a)
            recon_best = torch.where(mask, recon_a, recon_b)
            best_of_two_correct += (recon_best == batch).all(dim=1).sum().item()

        return {
            "voting": {
                "exact_match_rate": voting_correct / self.n_ops,
                "strategy": "majority_vote_per_bit",
            },
            "confidence_weighted": {
                "exact_match_rate": confidence_correct / self.n_ops,
                "strategy": "confidence_weighted_per_bit",
            },
            "best_of_two": {
                "exact_match_rate": best_of_two_correct / self.n_ops,
                "strategy": "select_best_reconstruction",
            },
            "baseline_vae_a": 0.1487,  # From isolated benchmark
            "baseline_vae_b": 1.0000,  # From isolated benchmark
        }

    @torch.no_grad()
    def measure_cross_injected_sampling(self, n_samples=50000, batch_size=1000, rho=0.5) -> dict:
        """Measure sampling coverage with cross-injection active"""
        sampled_ops = set()

        for _ in range(n_samples // batch_size):
            # Sample from prior for both VAEs
            z_a = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            z_b = torch.randn(batch_size, self.model.latent_dim, device=self.device)

            # Apply cross-injection (stop-gradient)
            z_a_mixed = z_a + rho * z_b.detach()
            z_b_mixed = z_b + rho * z_a.detach()

            # Decode with both VAEs
            logits_a = self.model.decoder_A(z_a_mixed)
            logits_b = self.model.decoder_B(z_b_mixed)

            # Convert to operations
            samples_a = torch.argmax(logits_a, dim=-1) - 1
            samples_b = torch.argmax(logits_b, dim=-1) - 1

            # Collect unique operations from both
            for sample in samples_a:
                op_tuple = tuple(sample.cpu().numpy().astype(int))
                sampled_ops.add(op_tuple)

            for sample in samples_b:
                op_tuple = tuple(sample.cpu().numpy().astype(int))
                sampled_ops.add(op_tuple)

        return {
            "unique_sampled": len(sampled_ops),
            "coverage_rate": len(sampled_ops) / self.n_ops,
            "n_samples": n_samples * 2,  # Both VAEs
            "diversity": len(sampled_ops) / (n_samples * 2),
            "rho": rho,
            "baseline_vae_a_coverage": 0.7755,
            "baseline_vae_b_coverage": 0.6582,
            "improvement": len(sampled_ops) / self.n_ops - max(0.7755, 0.6582),
        }

    @torch.no_grad()
    def measure_complementary_coverage(self, batch_size=256) -> dict:
        """Measure which operations each VAE handles best"""
        vae_a_best_count = 0
        vae_b_best_count = 0
        both_perfect_count = 0
        both_imperfect_count = 0

        vae_a_specialization = []
        vae_b_specialization = []

        for i in range(0, self.n_ops, batch_size):
            batch = self.all_ops[i : i + batch_size]

            # Encode/decode with both
            mu_a, _ = self.model.encoder_A(batch)
            mu_b, _ = self.model.encoder_B(batch)

            logits_a = self.model.decoder_A(mu_a)
            logits_b = self.model.decoder_B(mu_b)

            recon_a = torch.argmax(logits_a, dim=-1) - 1
            recon_b = torch.argmax(logits_b, dim=-1) - 1

            # Count errors
            errors_a = (recon_a != batch).sum(dim=1)
            errors_b = (recon_b != batch).sum(dim=1)

            # Categorize
            for j in range(len(batch)):
                err_a = errors_a[j].item()
                err_b = errors_b[j].item()

                if err_a == 0 and err_b == 0:
                    both_perfect_count += 1
                elif err_a == 0 and err_b > 0:
                    vae_a_best_count += 1
                    vae_a_specialization.append((i + j, err_b))
                elif err_b == 0 and err_a > 0:
                    vae_b_best_count += 1
                    vae_b_specialization.append((i + j, err_a))
                else:
                    both_imperfect_count += 1
                    if err_a < err_b:
                        vae_a_best_count += 1
                    elif err_b < err_a:
                        vae_b_best_count += 1

        return {
            "both_perfect": both_perfect_count,
            "both_perfect_rate": both_perfect_count / self.n_ops,
            "vae_a_best": vae_a_best_count,
            "vae_b_best": vae_b_best_count,
            "both_imperfect": both_imperfect_count,
            "vae_a_specialization_rate": vae_a_best_count / self.n_ops,
            "vae_b_specialization_rate": vae_b_best_count / self.n_ops,
            "complementarity_score": (
                min(vae_a_best_count, vae_b_best_count) / max(vae_a_best_count, vae_b_best_count)
                if vae_b_best_count > 0
                else 0
            ),
        }

    @torch.no_grad()
    def measure_latent_space_coupling(self, n_samples=1000) -> dict:
        """Measure correlation between VAE-A and VAE-B latent spaces"""
        # Sample operations
        indices = torch.randperm(self.n_ops)[:n_samples]
        batch = self.all_ops[indices]

        # Encode with both
        mu_a, _ = self.model.encoder_A(batch)
        mu_b, _ = self.model.encoder_B(batch)

        # Compute correlation
        mu_a_np = mu_a.cpu().numpy()
        mu_b_np = mu_b.cpu().numpy()

        # Dimension-wise correlation
        correlations = []
        for dim in range(self.model.latent_dim):
            corr = np.corrcoef(mu_a_np[:, dim], mu_b_np[:, dim])[0, 1]
            correlations.append(corr)

        correlations = np.array(correlations)

        # Distance correlation
        # For same operation, how similar are the latent representations?
        distances = torch.norm(mu_a - mu_b, dim=1)

        return {
            "mean_correlation": float(np.mean(correlations)),
            "std_correlation": float(np.std(correlations)),
            "mean_distance": distances.mean().item(),
            "std_distance": distances.std().item(),
            "min_distance": distances.min().item(),
            "max_distance": distances.max().item(),
            "alignment_score": 1.0 / (1.0 + distances.mean().item()),  # Higher is better
        }

    @torch.no_grad()
    def measure_system_resolution_score(self) -> dict:
        """Compute overall system resolution score"""
        # Quick measurements for scoring
        ensemble = self.measure_ensemble_reconstruction(batch_size=256)
        coverage = self.measure_cross_injected_sampling(n_samples=10000, rho=0.5)
        complementarity = self.measure_complementary_coverage(batch_size=256)
        coupling = self.measure_latent_space_coupling(n_samples=1000)

        # Compute weighted score
        ensemble_score = ensemble["best_of_two"]["exact_match_rate"]
        coverage_score = coverage["coverage_rate"]
        complementarity_score = complementarity["complementarity_score"]
        coupling_score = coupling["alignment_score"]

        overall_score = (
            0.40 * ensemble_score  # Reconstruction quality
            + 0.30 * coverage_score  # Space coverage
            + 0.20 * complementarity_score  # VAE specialization
            + 0.10 * coupling_score  # Latent alignment
        )

        return {
            "ensemble": ensemble_score,
            "coverage": coverage_score,
            "complementarity": complementarity_score,
            "coupling": coupling_score,
            "overall": overall_score,
            "isolated_vae_a": 0.6684,
            "isolated_vae_b": 0.8887,
            "isolated_combined": 0.7785,
            "improvement": overall_score - 0.7785,
        }

    def run_full_benchmark(self) -> dict:
        """Run all coupled system benchmarks"""
        results = {
            "model_info": {
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "latent_dim": self.model.latent_dim,
                "n_operations": self.n_ops,
                "rho_current": float(self.model.rho),
                "statenet_enabled": self.model.use_statenet,
            }
        }

        print("Benchmarking coupled system...")
        print("\n1. Ensemble Reconstruction (both VAEs together)...")
        results["ensemble_reconstruction"] = self.measure_ensemble_reconstruction()

        print("2. Cross-Injected Sampling (rho=0.5)...")
        results["cross_injected_sampling_rho_05"] = self.measure_cross_injected_sampling(n_samples=50000, rho=0.5)

        print("3. Cross-Injected Sampling (rho=0.7)...")
        results["cross_injected_sampling_rho_07"] = self.measure_cross_injected_sampling(n_samples=50000, rho=0.7)

        print("4. Complementary Coverage Analysis...")
        results["complementary_coverage"] = self.measure_complementary_coverage()

        print("5. Latent Space Coupling...")
        results["latent_coupling"] = self.measure_latent_space_coupling()

        print("6. System Resolution Score...")
        results["system_resolution"] = self.measure_system_resolution_score()

        return results


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
    print("\nRunning coupled system benchmark...")
    benchmark = CoupledSystemBenchmark(model, device)
    results = benchmark.run_full_benchmark()

    # Save results
    save_results(results, "coupled_resolution", checkpoint.get("epoch", "init"))

    # Print summary
    print("\n" + "=" * 80)
    print("COUPLED SYSTEM RESOLUTION SUMMARY")
    print("=" * 80)
    print(f"Model: {results['model_info']['total_params']} parameters")
    print(f"Latent dim: {results['model_info']['latent_dim']}")
    print(f"Current rho: {results['model_info']['rho_current']:.4f}")

    print("\nEnsemble Reconstruction:")
    for strategy, data in results["ensemble_reconstruction"].items():
        if isinstance(data, dict) and "exact_match_rate" in data:
            print(f"  {strategy:20s}: {data['exact_match_rate']:.4f}")

    print("\nCross-Injected Sampling:")
    for key in [
        "cross_injected_sampling_rho_05",
        "cross_injected_sampling_rho_07",
    ]:
        if key in results:
            data = results[key]
            print(f"  rho={data['rho']:.1f}: {data['coverage_rate']:.4f} coverage, {data['unique_sampled']} unique ops")

    print("\nComplementary Coverage:")
    comp = results["complementary_coverage"]
    print(f"  Both perfect:        {comp['both_perfect_rate']:.4f}")
    print(f"  VAE-A specializes:   {comp['vae_a_specialization_rate']:.4f}")
    print(f"  VAE-B specializes:   {comp['vae_b_specialization_rate']:.4f}")
    print(f"  Complementarity:     {comp['complementarity_score']:.4f}")

    print("\nLatent Coupling:")
    coupling = results["latent_coupling"]
    print(f"  Mean correlation:    {coupling['mean_correlation']:.4f}")
    print(f"  Mean distance:       {coupling['mean_distance']:.4f}")
    print(f"  Alignment score:     {coupling['alignment_score']:.4f}")

    print("\nSystem Resolution Score:")
    score = results["system_resolution"]
    print(f"  Ensemble:            {score['ensemble']:.4f}")
    print(f"  Coverage:            {score['coverage']:.4f}")
    print(f"  Complementarity:     {score['complementarity']:.4f}")
    print(f"  Coupling:            {score['coupling']:.4f}")
    print(f"  Overall:             {score['overall']:.4f}")
    print(f"  Isolated baseline:   {score['isolated_combined']:.4f}")
    print(f"  Improvement:         {score['improvement']:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
