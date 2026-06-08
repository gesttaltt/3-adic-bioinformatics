# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""ESM-2 Protein Language Model encoder.

ESM-2 (Evolutionary Scale Modeling 2) from Meta AI provides
state-of-the-art protein sequence representations.

Reference:
    Lin et al. (2023) "Evolutionary-scale prediction of atomic-level
    protein structure with a language model"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.plm.base import PLMEncoderBase


@dataclass
class ESM2Config:
    """Configuration for ESM-2 encoder.

    Attributes:
        model_name: HuggingFace model name or path
        layers_to_use: Which transformer layers to extract (mid-to-late best)
        pooling: How to pool sequence embeddings
        freeze_backbone: Whether to freeze ESM-2 weights
        use_flash_attention: Use FlashAttention for efficiency
        quantize: Quantization level (None, 8, 4)
    """

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    layers_to_use: list[int] = field(default_factory=lambda: [20, 24, 28, 32])
    pooling: str = "mean"
    freeze_backbone: bool = True
    use_flash_attention: bool = False
    quantize: int | None = None
    max_length: int = 1024


# Check for transformers availability
try:
    from transformers import AutoModel, AutoTokenizer, EsmModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ESM2Encoder(PLMEncoderBase):
    """ESM-2 encoder for protein sequences.

    Extracts embeddings from multiple transformer layers and combines
    them using learned attention weights. Mid-to-late layers (20-33)
    typically provide the best representations.

    Example:
        >>> config = ESM2Config(model_name="facebook/esm2_t12_35M_UR50D")
        >>> encoder = ESM2Encoder(config)
        >>> embeddings = encoder.encode(["MKWVTFISLLLLFSSAYS"])
        >>> print(embeddings.shape)  # (1, 480)
    """

    # Model dimension lookup
    MODEL_DIMS = {
        "facebook/esm2_t6_8M_UR50D": 320,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
        "facebook/esm2_t36_3B_UR50D": 2560,
        "facebook/esm2_t48_15B_UR50D": 5120,
    }

    def __init__(
        self,
        config: ESM2Config | None = None,
        device: str = "cuda",
    ):
        """Initialize ESM-2 encoder.

        Args:
            config: ESM-2 configuration
            device: Device for computation
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required for ESM-2. Install with: pip install transformers")

        self.config = config or ESM2Config()

        # Determine output dimension
        embed_dim = self.MODEL_DIMS.get(self.config.model_name, 1280)
        super().__init__(output_dim=embed_dim, device=device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model with optional quantization
        self._load_model()

        # Layer attention for combining multi-layer embeddings
        n_layers = len(self.config.layers_to_use)
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

        # Move to device
        self.to(device)

    def _load_model(self):
        """Load ESM-2 model with optional optimizations."""
        load_kwargs = {}

        # Quantization
        if self.config.quantize == 8:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                pass
        elif self.config.quantize == 4:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                pass

        # Load model
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            output_hidden_states=True,
            **load_kwargs,
        )

        # Freeze backbone if specified
        if self.config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Enable FlashAttention if available
        if self.config.use_flash_attention:
            try:
                self.model = self.model.to_bettertransformer()
            except Exception:
                pass  # FlashAttention not available

    def encode(
        self,
        sequences: str | list[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode protein sequences to embeddings.

        Args:
            sequences: Single sequence or list of sequences
            return_attention: Whether to return attention weights

        Returns:
            Pooled embeddings of shape (batch, output_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
            outputs = self.model(**inputs)

        # Extract specified layers
        hidden_states = outputs.hidden_states
        layer_embeddings = torch.stack(
            [hidden_states[l] for l in self.config.layers_to_use],
            dim=0,
        )

        # Weighted combination of layers
        weights = torch.softmax(self.layer_weights, dim=0)
        combined = (layer_embeddings * weights.view(-1, 1, 1, 1)).sum(dim=0)

        # Pool to fixed size
        attention_mask = inputs.get("attention_mask")
        pooled = self.pool_embeddings(combined, attention_mask, self.config.pooling)

        if return_attention:
            attentions = outputs.attentions if hasattr(outputs, "attentions") else None
            return pooled, attentions

        return pooled

    def encode_batch(
        self,
        sequences: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode large batch of sequences efficiently.

        Args:
            sequences: List of protein sequences
            batch_size: Processing batch size

        Returns:
            Embeddings tensor of shape (n_sequences, output_dim)
        """
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def get_layer_embeddings(
        self,
        sequences: str | list[str],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Get embeddings from specific transformer layers.

        Args:
            sequences: Input sequences
            layers: List of layer indices

        Returns:
            Dictionary mapping layer index to embeddings
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract requested layers
        hidden_states = outputs.hidden_states
        attention_mask = inputs.get("attention_mask")

        result = {}
        for layer in layers:
            embeddings = hidden_states[layer]
            pooled = self.pool_embeddings(embeddings, attention_mask, self.config.pooling)
            result[layer] = pooled

        return result

    def get_per_residue_embeddings(
        self,
        sequences: str | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get per-residue embeddings without pooling.

        Args:
            sequences: Input sequences

        Returns:
            Tuple of (embeddings, attention_mask)
            embeddings: (batch, seq_len, output_dim)
            attention_mask: (batch, seq_len)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Combine layers
        hidden_states = outputs.hidden_states
        layer_embeddings = torch.stack(
            [hidden_states[l] for l in self.config.layers_to_use],
            dim=0,
        )
        weights = torch.softmax(self.layer_weights, dim=0)
        combined = (layer_embeddings * weights.view(-1, 1, 1, 1)).sum(dim=0)

        return combined, inputs["attention_mask"]


class ESM2EncoderLite(PLMEncoderBase):
    """Lightweight ESM-2 encoder for resource-constrained environments.

    Uses the smallest ESM-2 model (8M params) with additional optimizations
    for fast inference on limited hardware.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize lightweight encoder."""
        config = ESM2Config(
            model_name="facebook/esm2_t6_8M_UR50D",
            layers_to_use=[3, 4, 5, 6],
            freeze_backbone=True,
            quantize=8,
        )
        super().__init__(output_dim=320, device=device)

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required")

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(
            config.model_name,
            output_hidden_states=True,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.to(device)

    def encode(
        self,
        sequences: str | list[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fast encoding with minimal overhead."""
        if isinstance(sequences, str):
            sequences = [sequences]

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use last hidden state directly
        embeddings = outputs.last_hidden_state
        pooled = self.pool_embeddings(embeddings, inputs.get("attention_mask"), "mean")

        return pooled

    def encode_batch(
        self,
        sequences: list[str],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Batch encoding."""
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def get_layer_embeddings(
        self,
        sequences: str | list[str],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Get layer embeddings."""
        if isinstance(sequences, str):
            sequences = [sequences]

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = {}
        for layer in layers:
            if layer < len(outputs.hidden_states):
                embeddings = outputs.hidden_states[layer]
                pooled = self.pool_embeddings(embeddings, inputs.get("attention_mask"), "mean")
                result[layer] = pooled

        return result
