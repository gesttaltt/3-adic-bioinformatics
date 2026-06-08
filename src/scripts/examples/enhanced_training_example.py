#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Enhanced Training Example with All New Components.

This script demonstrates how to integrate all the new SOTA-inspired
components into the training pipeline:

1. HybridCodonEncoder - ESM-2 + P-adic codon embeddings
2. MultiScaleNucleotideEncoder - Sub-codon granularity features
3. CodonUsageLoss - Biological codon constraints (tAI, CAI, CpG)
4. ProteinGymEvaluator - Standardized evaluation metrics
5. VariantEscapeHead - HIV/viral escape prediction
6. CodonPositiveSampler - Biology-aware contrastive learning

Usage:
    python src/scripts/examples/enhanced_training_example.py --quick
"""

from __future__ import annotations

import argparse

import torch

# New enhanced components
from src.contrastive import CodonPositiveSampler, CodonSamplerConfig
from src.diseases import MetaLearningEscapeHead, VariantEscapeHead
from src.encoders import (
    HybridCodonEncoder,
    HybridEncoderConfig,
    MultiScaleConfig,
    MultiScaleNucleotideEncoder,
    PLMBackend,
)
from src.evaluation import ProteinGymEvaluator
from src.losses import CodonOptimalityScore, CodonUsageConfig, CodonUsageLoss, Organism


def demo_hybrid_encoder():
    """Demonstrate HybridCodonEncoder usage."""
    print("\n" + "=" * 60)
    print("DEMO: HybridCodonEncoder")
    print("=" * 60)

    # Create codon-only encoder (no PLM, works without ESM-2 installed)
    config = HybridEncoderConfig(
        codon_dim=64,
        output_dim=64,
        plm_backend=PLMBackend.NONE,  # No PLM for demo
    )
    encoder = HybridCodonEncoder(config)

    # Create sample codon sequence
    batch_size = 2
    seq_len = 30  # 30 codons = 90 nucleotides
    codon_indices = torch.randint(0, 64, (batch_size, seq_len))

    # Forward pass
    embeddings = encoder(codon_indices)
    print(f"Input shape: {codon_indices.shape}")
    print(f"Output shape: {embeddings.shape}")

    # P-adic loss
    padic_loss = encoder.compute_padic_loss()
    print(f"P-adic structure loss: {padic_loss.item():.4f}")

    # Factory methods
    print("\nFactory methods available:")
    print("  - HybridEncoderFactory.create_codon_only()")
    print("  - HybridEncoderFactory.create_esm2_small()  # Requires ESM-2")
    print("  - HybridEncoderFactory.create_esm2_medium() # Requires ESM-2")

    return encoder


def demo_multiscale_encoder():
    """Demonstrate MultiScaleNucleotideEncoder usage."""
    print("\n" + "=" * 60)
    print("DEMO: MultiScaleNucleotideEncoder (Sub-Codon Granularity)")
    print("=" * 60)

    # Create encoder with RiboNN-style features
    config = MultiScaleConfig(
        nucleotide_dim=16,
        dinucleotide_dim=32,
        position_dim=16,
        output_dim=64,
        use_dinucleotide=True,
        use_position=True,  # Wobble position features
        use_gc_content=True,
        use_cpg_density=True,
    )
    encoder = MultiScaleNucleotideEncoder(config)

    # Create sample nucleotide sequence (0=A, 1=C, 2=G, 3=T)
    batch_size = 2
    seq_len = 90  # 30 codons * 3
    nucleotides = torch.randint(0, 4, (batch_size, seq_len))

    # Forward pass
    outputs = encoder(nucleotides)

    print(f"Input nucleotides shape: {nucleotides.shape}")
    print("\nOutput features:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\nSub-codon features captured:")
    print("  - Dinucleotide effects (AUG, GG, UU, AA, UA)")
    print("  - Wobble position (3rd codon position)")
    print("  - GC content (per-position)")
    print("  - CpG density (immune activation)")

    return encoder


def demo_codon_usage_loss():
    """Demonstrate CodonUsageLoss usage."""
    print("\n" + "=" * 60)
    print("DEMO: CodonUsageLoss (Biological Constraints)")
    print("=" * 60)

    # Create loss with human codon usage
    config = CodonUsageConfig(
        organism=Organism.HUMAN,
        tai_weight=0.3,
        cai_weight=0.3,
        rare_penalty_weight=0.2,
        cpg_penalty_weight=0.1,
        gc_weight=0.1,
    )
    loss_fn = CodonUsageLoss(weight=0.1, config=config)

    # Sample generated codons
    batch_size = 4
    seq_len = 100
    generated_codons = torch.randint(0, 64, (batch_size, seq_len))

    # Compute loss
    outputs = {"generated_codons": generated_codons}
    result = loss_fn(outputs, targets=torch.zeros(1))

    print(f"Total codon usage loss: {result.loss.item():.4f}")
    print("\nComponent metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")

    # Optimality scorer
    scorer = CodonOptimalityScore(organism=Organism.HUMAN)
    scores = scorer(generated_codons)
    print(f"\nBatch mean tAI: {scores['tai'].mean():.4f}")
    print(f"Batch mean CAI: {scores['cai'].mean():.4f}")

    return loss_fn


def demo_proteingym_evaluator():
    """Demonstrate ProteinGym-style evaluation."""
    print("\n" + "=" * 60)
    print("DEMO: ProteinGym-Style Evaluation Metrics")
    print("=" * 60)

    # Create sample training set
    n_train = 50
    seq_len = 30
    training_set = torch.randint(0, 64, (n_train, seq_len))

    # Create evaluator
    evaluator = ProteinGymEvaluator(
        training_sequences=training_set,
        organism=Organism.HUMAN,
    )

    # Generate some sequences
    n_generated = 20
    generated = torch.randint(0, 64, (n_generated, seq_len))

    # Evaluate
    metrics = evaluator.evaluate(generated)

    print(f"Evaluated {metrics.n_sequences} sequences")
    print(f"Mean length: {metrics.mean_length}")

    print("\nQuality Metrics:")
    print(f"  Mean tAI: {metrics.quality.mean_tai:.4f}")
    print(f"  Mean CAI: {metrics.quality.mean_cai:.4f}")

    print("\nNovelty Metrics:")
    print(f"  Unique fraction: {metrics.novelty.unique_fraction:.4f}")
    print(f"  Novel fraction: {metrics.novelty.novel_fraction:.4f}")

    print("\nDiversity Metrics:")
    print(f"  Mean pairwise distance: {metrics.diversity.mean_pairwise_distance:.4f}")
    print(f"  Codon coverage: {metrics.diversity.codon_coverage:.4f}")
    print(f"  AA coverage: {metrics.diversity.amino_acid_coverage:.4f}")

    print("\nBiological Validity:")
    print(f"  No internal stops: {metrics.validity.no_stop_codons:.4f}")
    print(f"  Valid start codon: {metrics.validity.valid_start_codon:.4f}")

    return metrics


def demo_variant_escape():
    """Demonstrate VariantEscapeHead usage."""
    print("\n" + "=" * 60)
    print("DEMO: VariantEscapeHead (EVEscape-Inspired)")
    print("=" * 60)

    # Create escape prediction head for HIV
    head = VariantEscapeHead(
        latent_dim=64,
        hidden_dim=128,
        disease="hiv",
        n_drug_classes=6,  # HIV drug classes
        n_antibody_classes=10,
        n_tcell_epitopes=20,
    )

    # Sample latent representations
    batch_size = 8
    latent_z = torch.randn(batch_size, 64)

    # Predict escape
    predictions = head(latent_z, return_components=True)

    print(f"Input latent shape: {latent_z.shape}")
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape} (mean={value.mean():.3f})")

    # Meta-learning compatible head
    meta_head = MetaLearningEscapeHead(latent_dim=64, hidden_dim=64)
    meta_preds = meta_head(latent_z)
    print("\nMeta-learning head predictions:")
    for key, value in meta_preds.items():
        print(f"  {key}: {value.shape}")

    print("\nDesigned for few-shot adaptation to new variants!")

    return head


def demo_codon_sampler():
    """Demonstrate CodonPositiveSampler for contrastive learning."""
    print("\n" + "=" * 60)
    print("DEMO: CodonPositiveSampler (Biology-Aware Contrastive)")
    print("=" * 60)

    # Create sampler
    config = CodonSamplerConfig(
        use_synonymous=True,
        use_wobble=True,
        use_conservative=True,
        use_padic=True,
        synonymous_weight=0.4,
        wobble_weight=0.3,
        conservative_weight=0.2,
        padic_weight=0.1,
    )
    sampler = CodonPositiveSampler(config)

    # Example: Find positives for ATG (start codon, index ~14)
    anchor_idx = 14  # ATG
    positives = sampler.get_positive_candidates(anchor_idx)

    from src.biology.codons import codon_index_to_triplet

    anchor_triplet = codon_index_to_triplet(anchor_idx)
    print(f"Anchor codon: {anchor_triplet} (index {anchor_idx})")
    print(f"\nPositive candidates ({len(positives)} codons):")

    for pos_idx in list(positives)[:10]:
        triplet = codon_index_to_triplet(pos_idx)
        weight = sampler.get_positive_weights(anchor_idx, pos_idx)
        print(f"  {triplet} (index {pos_idx}, weight={weight:.2f})")

    # Sample a positive
    sampled = sampler.sample_positive(anchor_idx)
    if sampled is not None:
        print(f"\nSampled positive: {codon_index_to_triplet(sampled)}")

    # Get hard negatives
    negatives = sampler.sample_negatives(anchor_idx, n_negatives=5)
    print(f"\nHard negatives: {[codon_index_to_triplet(n) for n in negatives]}")

    return sampler


def run_all_demos():
    """Run all component demonstrations."""
    print("\n" + "=" * 60)
    print("ENHANCED TRAINING COMPONENTS DEMONSTRATION")
    print("Based on SOTA Research (2024-2025)")
    print("=" * 60)

    print("\nComponents implemented:")
    print("1. HybridCodonEncoder - ESM-2 + P-adic fusion")
    print("2. MultiScaleNucleotideEncoder - Sub-codon features")
    print("3. CodonUsageLoss - tAI/CAI/CpG constraints")
    print("4. ProteinGymEvaluator - Standardized metrics")
    print("5. VariantEscapeHead - Viral escape prediction")
    print("6. CodonPositiveSampler - Biology-aware contrastive")

    # Run demos
    demo_hybrid_encoder()
    demo_multiscale_encoder()
    demo_codon_usage_loss()
    demo_proteingym_evaluator()
    demo_variant_escape()
    demo_codon_sampler()

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)

    print("\nReferences:")
    print("- RiboNN (Nature Biotech 2024): Dinucleotide features")
    print("- CodonTransformer (Nature Comm 2025): Context-aware optimization")
    print("- Riboformer (Nature Comm 2024): Ribosome profiling")
    print("- EVEscape (Nature 2023): Viral escape prediction")
    print("- ProteinGym (ICLR 2023): Evaluation benchmarks")
    print("- ESM-2 (Science 2023): Protein language models")

    print("\nSources:")
    print("- https://www.nature.com/articles/s41587-025-02712-x")
    print("- https://www.nature.com/articles/s41467-025-58588-7")
    print("- https://www.nature.com/articles/s41467-024-46241-8")
    print("- https://www.nature.com/articles/s41586-023-06004-9")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Training Demo")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode")
    parser.parse_args()

    run_all_demos()


if __name__ == "__main__":
    main()
