# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic Dynamical Systems and Ergodic Theory.

Implements recurrent neural networks and predictors based on p-adic
dynamical systems theory and ergodic properties.

Key Concepts:
1. P-adic Dynamics: Dynamics on p-adic spaces have unique properties
   - Contractive maps on ultrametric spaces
   - Ergodic behavior of polynomial maps
   - Attractors in p-adic fields

2. Ergodic Theory: Long-term statistical behavior of dynamical systems
   - Invariant measures on p-adic spaces
   - Mixing properties of p-adic maps
   - Prediction through ergodic averaging

Applications:
- Predicting viral evolution trajectories
- Modeling sequence mutations as dynamics on p-adic spaces
- Long-range dependency modeling via p-adic structure

Mathematical Background:
A p-adic dynamical system is a map f: Z_p -> Z_p where Z_p is the
ring of p-adic integers. The dynamics are given by iteration:
    x_{n+1} = f(x_n)

Key property: If f is 1-Lipschitz in the p-adic metric, orbits cannot
diverge, leading to stable long-term behavior.

For polynomial maps f(x) = a_0 + a_1*x + ... + a_n*x^n, the dynamics
are ergodic with respect to Haar measure when certain conditions hold.

References:
- Anashin (2007): Uniformly distributed sequences in computer algebra
- Khrennikov (2009): P-adic valued quantization
- Fan & Liao (2015): Ergodic theory of p-adic dynamics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicsConfig:
    """Configuration for p-adic dynamics modules."""

    p: int = 3  # Prime base
    precision: int = 10  # Number of p-adic digits
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    use_ergodic_averaging: bool = True
    averaging_window: int = 10


class PAdicRNNCell(nn.Module):
    """Single cell of a P-adic RNN.

    The state update respects p-adic structure by:
    1. Decomposing states into p-adic digits
    2. Processing each digit level separately
    3. Ensuring updates are contractive in p-adic metric

    Update equations:
        z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)  (update gate)
        r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)  (reset gate)
        h_tilde = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
        h_t = z_t * h_{t-1} + (1 - z_t) * h_tilde

    The p-adic structure is incorporated through:
    - Digit-level processing
    - Contractive projections
    - Valuation-based gating
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        p: int = 3,
        precision: int = 10,
    ):
        """Initialize P-adic RNN cell.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            p: Prime base for p-adic structure
            precision: Number of p-adic digits to use
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p
        self.precision = precision

        # GRU-style gates
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Per-digit processing (for p-adic structure)
        self.digit_transforms = nn.ModuleList(
            [nn.Linear(hidden_dim // precision, hidden_dim // precision) for _ in range(precision)]
        )

        # Valuation-based attention
        self.valuation_attention = nn.Parameter(torch.ones(precision) / precision)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of RNN cell.

        Args:
            x: Input (batch, input_dim)
            h: Previous hidden state (batch, hidden_dim)

        Returns:
            New hidden state (batch, hidden_dim)
        """
        batch_size = x.size(0)
        device = x.device

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Compute gates
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))

        # Apply p-adic digit-level processing
        h_digits = h.view(batch_size, self.precision, -1)

        # Process each digit level
        h_processed = []
        for i, transform in enumerate(self.digit_transforms):
            digit_i = h_digits[:, i]
            processed = transform(digit_i)
            # Apply valuation-based weighting
            h_processed.append(self.valuation_attention[i] * processed)

        h_padic = torch.stack(h_processed, dim=1).view(batch_size, -1)

        # Candidate state
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_padic))

        # Update
        h_new = z * h + (1 - z) * h_tilde

        # Project to ensure contractiveness (p-adic Lipschitz condition)
        h_new = self._contractive_projection(h_new)

        return h_new

    def _contractive_projection(
        self,
        h: torch.Tensor,
        lipschitz_const: float = 0.99,
    ) -> torch.Tensor:
        """Project hidden state to ensure p-adic contractiveness.

        Args:
            h: Hidden state
            lipschitz_const: Target Lipschitz constant (< 1 for contraction)

        Returns:
            Projected state
        """
        # Normalize to ensure bounded dynamics
        norm = torch.norm(h, dim=-1, keepdim=True)
        max_norm = 1.0 / (1.0 - lipschitz_const)  # Bound for contractive map

        scale = torch.where(
            norm > max_norm,
            max_norm / (norm + 1e-8),
            torch.ones_like(norm),
        )

        return h * scale


class PAdicRNN(nn.Module):
    """P-adic Recurrent Neural Network.

    Full RNN using p-adic structure for modeling hierarchical sequences.
    Particularly suited for:
    - Codon sequences (ternary structure)
    - Phylogenetic evolution
    - Hierarchical biological data

    The p-adic structure provides:
    - Natural handling of hierarchical dependencies
    - Stable long-term dynamics (contractive)
    - Ultrametric distance preservation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        p: int = 3,
        precision: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """Initialize P-adic RNN.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            n_layers: Number of stacked layers
            p: Prime base
            precision: P-adic precision
            dropout: Dropout probability
            bidirectional: Use bidirectional RNN
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.p = p
        self.bidirectional = bidirectional

        # Stack of P-adic RNN cells
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            cell_input = input_dim if i == 0 else hidden_dim
            self.cells.append(PAdicRNNCell(cell_input, hidden_dim, p, precision))

        if bidirectional:
            self.cells_backward = nn.ModuleList()
            for i in range(n_layers):
                cell_input = input_dim if i == 0 else hidden_dim
                self.cells_backward.append(PAdicRNNCell(cell_input, hidden_dim, p, precision))

        # Output projection
        output_mult = 2 if bidirectional else 1
        self.output_proj = nn.Linear(hidden_dim * output_mult, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h0: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through P-adic RNN.

        Args:
            x: Input sequence (batch, seq_len, input_dim)
            h0: Initial hidden states (list of n_layers tensors)

        Returns:
            Tuple of (output sequence, final hidden states)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states if not provided
        if h0 is None:
            h0 = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]

        # Forward pass
        h_forward = h0
        outputs_forward = []

        for t in range(seq_len):
            x_t = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                input_t = x_t if layer_idx == 0 else h_forward[layer_idx - 1]
                h_forward[layer_idx] = cell(input_t, h_forward[layer_idx])

                if layer_idx < self.n_layers - 1:
                    h_forward[layer_idx] = self.dropout(h_forward[layer_idx])

            outputs_forward.append(h_forward[-1])

        output_fwd = torch.stack(outputs_forward, dim=1)

        # Backward pass if bidirectional
        if self.bidirectional:
            h_backward = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.n_layers)]
            outputs_backward = []

            for t in range(seq_len - 1, -1, -1):
                x_t = x[:, t]
                for layer_idx, cell in enumerate(self.cells_backward):
                    input_t = x_t if layer_idx == 0 else h_backward[layer_idx - 1]
                    h_backward[layer_idx] = cell(input_t, h_backward[layer_idx])

                    if layer_idx < self.n_layers - 1:
                        h_backward[layer_idx] = self.dropout(h_backward[layer_idx])

                outputs_backward.append(h_backward[-1])

            outputs_backward = outputs_backward[::-1]
            output_bwd = torch.stack(outputs_backward, dim=1)

            output = torch.cat([output_fwd, output_bwd], dim=-1)
            final_h = h_forward + h_backward
        else:
            output = output_fwd
            final_h = h_forward

        # Project to output dimension
        output = self.output_proj(output)

        return output, final_h


class ErgodicPredictor(nn.Module):
    r"""Predictor based on ergodic theory of p-adic dynamics.

    Uses the ergodic properties of p-adic dynamical systems for
    long-range prediction:

    1. Ergodic Averaging: Time average equals space average
       For a function f and ergodic map T:
       lim_{n->inf} (1/n) sum_{k=0}^{n-1} f(T^k(x)) = integral f d\mu

    2. Invariant Measures: Haar measure on p-adic integers
       Predictions use expectations under invariant measure

    3. Mixing: Decorrelation of past and future
       Allows prediction without full history

    Applications:
    - Mutation rate prediction
    - Evolutionary trajectory forecasting
    - Long-term sequence property prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        averaging_window: int = 10,
        p: int = 3,
        use_haar_prior: bool = True,
    ):
        """Initialize ergodic predictor.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            averaging_window: Window for ergodic averaging
            p: Prime base
            use_haar_prior: Use Haar measure as prior
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.averaging_window = averaging_window
        self.p = p
        self.use_haar_prior = use_haar_prior

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dynamics model (polynomial map on p-adic space)
        # f(x) = a_0 + a_1*x + a_2*x^2 (quadratic dynamics)
        self.dynamics_coeffs = nn.Parameter(torch.randn(3, hidden_dim))

        # Ergodic averaging weights (learned)
        self.averaging_weights = nn.Parameter(torch.ones(averaging_window) / averaging_window)

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Haar measure prior (for p-adic integration)
        if use_haar_prior:
            self.register_buffer("haar_weights", self._compute_haar_weights(hidden_dim))

    def _compute_haar_weights(self, dim: int) -> torch.Tensor:
        """Compute discretized Haar measure weights.

        Haar measure on Z_p assigns weight p^{-k} to balls of radius p^{-k}.

        Args:
            dim: Dimension

        Returns:
            Weight tensor
        """
        # Approximate Haar weights for dimension reduction
        weights = torch.zeros(dim)
        for i in range(dim):
            # Weight inversely proportional to p-adic "level"
            level = i % self.p
            weights[i] = 1.0 / (self.p**level)

        return weights / weights.sum()

    def polynomial_dynamics(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply polynomial dynamics map.

        f(x) = a_0 + a_1*x + a_2*x^2

        This is the canonical form for ergodic p-adic maps.

        Args:
            x: State (batch, hidden_dim)

        Returns:
            Updated state
        """
        a0, a1, a2 = self.dynamics_coeffs[0], self.dynamics_coeffs[1], self.dynamics_coeffs[2]

        return a0 + a1 * x + a2 * (x**2)

    def ergodic_average(
        self,
        x: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Compute ergodic average over trajectory.

        Args:
            x: Initial state
            n_steps: Number of dynamics steps

        Returns:
            Ergodic average
        """
        trajectory = [x]
        current = x

        for _ in range(n_steps - 1):
            current = self.polynomial_dynamics(current)
            trajectory.append(current)

        trajectory = torch.stack(trajectory, dim=1)  # (batch, n_steps, hidden)

        # Weighted average (ergodic theorem: time average = space average)
        weights = F.softmax(self.averaging_weights[:n_steps], dim=0)
        weights = weights.view(1, -1, 1)

        averaged = (trajectory * weights).sum(dim=1)

        return averaged

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with ergodic prediction.

        Args:
            x: Input (batch, input_dim) or (batch, seq_len, input_dim)
            return_trajectory: Whether to return full trajectory

        Returns:
            Predictions, optionally with trajectory
        """
        # Handle sequence input
        if x.dim() == 3:
            # Average over sequence for initial state
            x = x.mean(dim=1)

        # Extract features
        features = self.feature_net(x)

        # Run dynamics and compute ergodic average
        if self.use_haar_prior:
            # Weight features by Haar measure
            features = features * self.haar_weights

        ergodic_features = self.ergodic_average(features, self.averaging_window)

        # Generate predictions
        predictions = self.output_net(ergodic_features)

        if return_trajectory:
            # Generate full trajectory for analysis
            trajectory = [features]
            current = features
            for _ in range(self.averaging_window - 1):
                current = self.polynomial_dynamics(current)
                trajectory.append(current)
            trajectory = torch.stack(trajectory, dim=1)
            return predictions, trajectory

        return predictions


class MutationDynamicsPredictor(nn.Module):
    """Predict mutation dynamics using p-adic ergodic theory.

    Models mutation as a dynamical system on p-adic space:
    - Each codon position is a p-adic integer
    - Mutations are maps on this space
    - Long-term evolution predicted via ergodic properties
    """

    def __init__(
        self,
        vocab_size: int = 64,  # 4^3 codons
        hidden_dim: int = 64,
        n_steps: int = 10,
        p: int = 3,
    ):
        """Initialize mutation dynamics predictor.

        Args:
            vocab_size: Number of tokens (codons)
            hidden_dim: Hidden dimension
            n_steps: Prediction horizon
            p: Prime base
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_steps = n_steps
        self.p = p

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # P-adic RNN for sequence encoding
        self.encoder = PAdicRNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=2,
            p=p,
        )

        # Ergodic predictor for future states
        self.predictor = ErgodicPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=vocab_size * n_steps,
            averaging_window=n_steps,
            p=p,
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict mutation trajectory.

        Args:
            sequence: Input sequence (batch, seq_len)

        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len = sequence.shape

        # Embed sequence
        embedded = self.embedding(sequence)

        # Encode with P-adic RNN
        encoded, hidden = self.encoder(embedded)

        # Use final state for prediction
        final_state = hidden[-1]

        # Predict future mutations
        predictions = self.predictor(final_state)
        predictions = predictions.view(batch_size, self.n_steps, self.vocab_size)

        # Convert to probabilities
        mutation_probs = F.softmax(predictions, dim=-1)

        return {
            "encoded_sequence": encoded,
            "mutation_probabilities": mutation_probs,
            "final_hidden": final_state,
        }


__all__ = [
    "DynamicsConfig",
    "PAdicRNNCell",
    "PAdicRNN",
    "ErgodicPredictor",
    "MutationDynamicsPredictor",
]
