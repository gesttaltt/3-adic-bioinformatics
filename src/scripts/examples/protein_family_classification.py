#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Protein family classification with meta-learning.

This script demonstrates few-shot protein family classification:
1. Create synthetic protein family data
2. Pretrain with contrastive learning
3. Fine-tune with MAML for few-shot classification
4. Evaluate on novel protein families

This approach is useful when:
- You have limited examples of each protein family
- New families are discovered that need quick classification
- You want a model that generalizes to unseen families

Usage:
    python src/scripts/examples/protein_family_classification.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


def create_protein_encoder(input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 32) -> nn.Module:
    """Create a simple protein sequence encoder.

    Args:
        input_dim: Input feature dimension (e.g., amino acid embeddings)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension

    Returns:
        Encoder neural network
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def create_synthetic_protein_data(
    n_families: int = 10,
    n_proteins_per_family: int = 20,
    seq_length: int = 50,
    feature_dim: int = 20,
) -> tuple:
    """Create synthetic protein family data.

    Each family has a distinct "signature" pattern that proteins
    in that family share, plus random noise.

    Args:
        n_families: Number of protein families
        n_proteins_per_family: Proteins per family
        seq_length: Sequence length
        feature_dim: Feature dimension (e.g., amino acid embedding)

    Returns:
        Tuple of (data, labels, family_signatures)
    """
    # Create family signatures (distinct patterns)
    signatures = torch.randn(n_families, feature_dim) * 2

    data = []
    labels = []

    for family_id in range(n_families):
        signature = signatures[family_id]
        for _ in range(n_proteins_per_family):
            # Create protein with family signature + noise
            protein = signature.unsqueeze(0).expand(seq_length, -1) + torch.randn(seq_length, feature_dim) * 0.5
            # Average pool to fixed-size representation
            protein_repr = protein.mean(dim=0)
            data.append(protein_repr)
            labels.append(family_id)

    data = torch.stack(data)
    labels = torch.tensor(labels)

    return data, labels, signatures


def contrastive_pretrain(
    encoder: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 50,
) -> nn.Module:
    """Pretrain encoder with contrastive learning.

    Uses p-adic contrastive loss to learn good representations
    before few-shot fine-tuning.

    Args:
        encoder: Encoder network
        data: (n_samples, feature_dim) protein data
        labels: (n_samples,) family labels
        n_epochs: Number of pretraining epochs

    Returns:
        Pretrained encoder
    """
    from src.contrastive import PAdicContrastiveLoss, SimCLREncoder

    # Wrap encoder with projection head
    simclr = SimCLREncoder(
        base_encoder=encoder,
        representation_dim=32,  # Output of encoder
        projection_dim=16,
        hidden_dim=32,
    )

    # Contrastive loss
    loss_fn = PAdicContrastiveLoss(temperature=0.1, prime=3)

    optimizer = torch.optim.Adam(simclr.parameters(), lr=1e-3)

    print("   Contrastive pretraining...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        embeddings = simclr(data)

        # Use labels as p-adic indices (similar families have close indices)
        indices = labels

        loss = loss_fn(embeddings, indices)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

    # Return the base encoder (without projection head)
    return encoder


def create_few_shot_tasks(
    data: torch.Tensor,
    labels: torch.Tensor,
    n_tasks: int = 20,
    n_way: int = 5,  # Classes per task
    n_support: int = 5,  # Examples per class in support set
    n_query: int = 10,  # Examples per class in query set
) -> list:
    """Create few-shot learning tasks.

    Each task has:
    - Support set: n_way * n_support examples (for adaptation)
    - Query set: n_way * n_query examples (for evaluation)

    Args:
        data: (n_samples, feature_dim) all data
        labels: (n_samples,) all labels
        n_tasks: Number of tasks to create
        n_way: Number of classes per task
        n_support: Support examples per class
        n_query: Query examples per class

    Returns:
        List of Task objects
    """
    from src.meta import Task

    n_classes = labels.max().item() + 1
    tasks = []

    for _ in range(n_tasks):
        # Sample n_way classes for this task
        selected_classes = torch.randperm(n_classes)[:n_way]

        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, old_label in enumerate(selected_classes):
            # Get all examples of this class
            class_mask = labels == old_label.item()
            class_indices = class_mask.nonzero().squeeze(-1)
            perm = torch.randperm(len(class_indices))

            # Split into support and query
            support_indices = class_indices[perm[:n_support]]
            query_indices = class_indices[perm[n_support : n_support + n_query]]

            support_x.append(data[support_indices])
            support_y.extend([new_label] * n_support)
            query_x.append(data[query_indices])
            query_y.extend([new_label] * n_query)

        task = Task(
            support_x=torch.cat(support_x),
            support_y=torch.tensor(support_y),
            query_x=torch.cat(query_x),
            query_y=torch.tensor(query_y),
        )
        tasks.append(task)

    return tasks


def train_maml(
    encoder: nn.Module,
    tasks: list,
    n_way: int = 5,
    n_epochs: int = 30,
) -> tuple:
    """Train MAML for few-shot classification.

    Args:
        encoder: Pretrained encoder
        tasks: List of few-shot tasks
        n_way: Number of classes per task
        n_epochs: Number of meta-training epochs

    Returns:
        Tuple of (trained MAML, training history)
    """
    from src.meta import MAML

    # Create classifier (encoder + linear head)
    classifier = nn.Sequential(
        encoder,
        nn.Linear(32, n_way),
    )

    maml = MAML(
        model=classifier,
        inner_lr=0.1,
        n_inner_steps=5,
        first_order=True,  # Faster than second-order
    )

    optimizer = torch.optim.Adam(maml.model.parameters(), lr=1e-3)

    history = {"loss": [], "accuracy": []}

    print("   MAML meta-training...")
    for epoch in range(n_epochs):
        # Sample tasks for this epoch
        epoch_tasks = [tasks[i % len(tasks)] for i in range(8)]

        metrics = maml.meta_train_step(epoch_tasks, optimizer)

        history["loss"].append(metrics["meta_loss"])
        history["accuracy"].append(metrics["meta_accuracy"])

        if (epoch + 1) % 5 == 0:
            print(
                f"     Epoch {epoch + 1}/{n_epochs}, "
                f"Loss: {metrics['meta_loss']:.4f}, "
                f"Acc: {metrics['meta_accuracy']:.2%}"
            )

    return maml, history


def evaluate_few_shot(
    maml,
    test_tasks: list,
) -> dict:
    """Evaluate few-shot performance on held-out tasks.

    Args:
        maml: Trained MAML model
        test_tasks: List of test tasks

    Returns:
        Dictionary with evaluation metrics
    """
    accuracies = []

    for task in test_tasks:
        # Adapt to task
        query_loss, query_output = maml(task)

        # Compute accuracy
        predictions = query_output.argmax(dim=-1)
        correct = (predictions == task.query_y).float().mean()
        accuracies.append(correct.item())

    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "std_accuracy": (sum((a - sum(accuracies) / len(accuracies)) ** 2 for a in accuracies) / len(accuracies))
        ** 0.5,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
    }


def main():
    """Run protein family classification example."""
    print("=" * 60)
    print("PROTEIN FAMILY CLASSIFICATION WITH META-LEARNING")
    print("=" * 60)

    # Configuration
    n_families = 20  # Total protein families
    n_proteins_per_family = 30
    n_way = 5  # Classes per task
    n_support = 5  # Support examples per class
    n_query = 10  # Query examples per class

    # Step 1: Create synthetic data
    print("\n1. Creating synthetic protein family data...")
    data, labels, _ = create_synthetic_protein_data(
        n_families=n_families,
        n_proteins_per_family=n_proteins_per_family,
    )
    print(f"   Total proteins: {len(data)}")
    print(f"   Number of families: {n_families}")

    # Split into train and test families
    train_families = n_families - 5
    train_mask = labels < train_families
    test_mask = ~train_mask

    train_data, train_labels = data[train_mask], labels[train_mask]
    test_data, test_labels = data[test_mask], labels[test_mask] - train_families  # Re-index

    print(f"   Train families: {train_families}, Test families: {n_families - train_families}")

    # Step 2: Create encoder
    print("\n2. Creating protein encoder...")
    encoder = create_protein_encoder(
        input_dim=20,
        hidden_dim=64,
        output_dim=32,
    )
    print(f"   Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")

    # Step 3: Contrastive pretraining
    print("\n3. Contrastive pretraining...")
    encoder = contrastive_pretrain(encoder, train_data, train_labels, n_epochs=30)

    # Step 4: Create few-shot tasks
    print("\n4. Creating few-shot tasks...")
    train_tasks = create_few_shot_tasks(
        train_data,
        train_labels,
        n_tasks=50,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
    )
    test_tasks = create_few_shot_tasks(
        test_data,
        test_labels,
        n_tasks=20,
        n_way=min(n_way, n_families - train_families),
        n_support=n_support,
        n_query=n_query,
    )
    print(f"   Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

    # Step 5: MAML training
    print("\n5. MAML meta-training...")
    maml, history = train_maml(encoder, train_tasks, n_way=n_way, n_epochs=20)

    # Step 6: Evaluation
    print("\n6. Evaluating on novel protein families...")
    results = evaluate_few_shot(maml, test_tasks)
    print(f"   Mean accuracy: {results['mean_accuracy']:.2%}")
    print(f"   Std accuracy: {results['std_accuracy']:.2%}")
    print(f"   Range: [{results['min_accuracy']:.2%}, {results['max_accuracy']:.2%}]")

    print("\n" + "=" * 60)
    print("Classification complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
