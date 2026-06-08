"""
Toxicity Regressor Trainer (Brizuela MVP)

Trains a simple MLP to predict Hemolytic Toxicity from Hyperbolic Latent Vectors.
Input: (batch, 16) Latent Vectors
Output: (batch, 1) Probability of Toxicity (0-1)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


class ToxicityRegressor(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_regressor(data_path: str, output_model_path: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    """
    Trains the toxicity regressor.
    Expects .pt file:
    - Standard: "embeddings" key (tensor)
    - Trajectories: Dict of tensors (flattened)
    """
    input_file = Path(data_path)
    if not input_file.exists():
        print(f"E: Input file {input_file} not found.")
        return

    print("I: Loading data...")
    data = torch.load(input_file)

    embeddings = None
    labels = None

    # Handle different input formats
    if isinstance(data, dict):
        if "embeddings" in data:
            # Case A: Standard Pipeline (Ingest StarPep)
            embeddings = data["embeddings"]

            # extract labels if available
            if "metadata" in data and "activity" in data["metadata"]:
                try:
                    labels = torch.tensor(data["metadata"]["activity"].values, dtype=torch.float32).unsqueeze(1)
                except:
                    print("W: Could not extract activity labels. Using random.")
        else:
            # Case B: Sliding Window Output (Dict of Tensors)
            # Flatten all trajectories into one big dataset for pre-training/testing
            print("I: Detected Dictionary of Tensors (Sliding Window format). Flattening...")
            tensor_list = []
            for _k, v in data.items():
                if isinstance(v, torch.Tensor):
                    tensor_list.append(v)
            if tensor_list:
                embeddings = torch.cat(tensor_list, dim=0)
            else:
                print("E: No tensors found in dictionary.")
                return
    elif isinstance(data, torch.Tensor):
        # Case C: Raw Tensor Input
        embeddings = data
    else:
        print(f"E: Unknown data type {type(data)}")
        return

    if embeddings is None:
        print("E: Failed to extract embeddings.")
        return

    print(f"I: Training on {embeddings.shape[0]} samples.")

    # Create dummy labels if needed
    if labels is None:
        print("I: Generating dummy labels for infrastructure test.")
        labels = torch.rand((len(embeddings), 1))

    dataset = TensorDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ToxicityRegressor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print("I: Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), output_model_path)
    print(f"S: Model saved to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed .pt data")
    parser.add_argument("--output", default="models/predictors/toxicity_regressor.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_regressor(args.input, args.output, args.epochs)
