# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tree-Structured LSTM for Hierarchical Sequence Modeling.

Implements Tree-LSTM architectures for processing tree-structured data,
particularly suited for:
- Hierarchical protein domain structures
- Parse trees from peptide grammars
- Phylogenetic trees

Tree-LSTMs generalize LSTMs to tree structures by having:
- Multiple child cell states
- Gating mechanisms that combine children appropriately

Variants:
1. Child-Sum Tree-LSTM: Sum children (order-invariant)
2. N-ary Tree-LSTM: Position-specific weights (order-sensitive)

References:
- Tai, Socher, Manning (2015): Improved Semantic Representations
  from Tree-Structured Long Short-Term Memory Networks
- Zhu, Sobhani, Guo (2015): Long Short-Term Memory Over Tree Structures
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src._experimental.linguistics.peptide_grammar import ParseTree


@dataclass
class TreeNode:
    """Node in a tree structure for Tree-LSTM processing."""

    id: int
    embedding: torch.Tensor
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = None
    label: str | None = None
    hidden: torch.Tensor | None = None
    cell: torch.Tensor | None = None


class TreeLSTM(nn.Module):
    """Base Tree-LSTM module.

    Abstract base class defining the interface for Tree-LSTM variants.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize Tree-LSTM.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden state
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tree: TreeNode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process tree and return root hidden state.

        Args:
            tree: Root node of tree

        Returns:
            Tuple of (hidden_state, cell_state) for root
        """
        raise NotImplementedError


class ChildSumTreeLSTM(TreeLSTM):
    """Child-Sum Tree-LSTM.

    Computes child states by summing over all children (order-invariant).
    Suitable for trees with varying and unordered branching factor.

    Equations:
        h_tilde = sum_j(h_j)  (sum of child hidden states)
        i = sigmoid(W_i x + U_i h_tilde + b_i)
        f_j = sigmoid(W_f x + U_f h_j + b_f)  (per-child forget)
        o = sigmoid(W_o x + U_o h_tilde + b_o)
        u = tanh(W_u x + U_u h_tilde + b_u)
        c = i * u + sum_j(f_j * c_j)
        h = o * tanh(c)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize Child-Sum Tree-LSTM.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden state dimension
            dropout: Dropout probability
        """
        super().__init__(input_dim, hidden_dim, dropout)

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Forget gate (shared weights, but applied per-child)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cell input
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def node_forward(
        self,
        x: torch.Tensor,
        child_h: torch.Tensor,
        child_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hidden state for a single node.

        Args:
            x: Node input embedding (hidden_dim,)
            child_h: Child hidden states (n_children, hidden_dim)
            child_c: Child cell states (n_children, hidden_dim)

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        n_children = child_h.size(0) if child_h.dim() > 1 else (1 if child_h.numel() > 0 else 0)

        if n_children == 0:
            # Leaf node
            h_tilde = torch.zeros(self.hidden_dim, device=x.device)

            i = torch.sigmoid(self.W_i(x))
            o = torch.sigmoid(self.W_o(x))
            u = torch.tanh(self.W_u(x))

            c = i * u
            h = o * torch.tanh(c)

            return h, c

        # Sum of child hidden states
        if child_h.dim() == 1:
            child_h = child_h.unsqueeze(0)
            child_c = child_c.unsqueeze(0)

        h_tilde = child_h.sum(dim=0)

        # Gates
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_tilde))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_tilde))
        u = torch.tanh(self.W_u(x) + self.U_u(h_tilde))

        # Per-child forget gates
        f = torch.sigmoid(self.W_f(x).unsqueeze(0).expand(n_children, -1) + self.U_f(child_h))

        # Cell state
        fc = (f * child_c).sum(dim=0)
        c = i * u + fc

        # Hidden state
        h = o * torch.tanh(c)

        return self.dropout(h), c

    def forward(
        self,
        tree: TreeNode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process tree bottom-up.

        Args:
            tree: Root node of tree

        Returns:
            Root (hidden_state, cell_state)
        """

        # Post-order traversal (children before parents)
        def process_node(node: TreeNode) -> tuple[torch.Tensor, torch.Tensor]:
            if not node.children:
                # Leaf node
                h, c = self.node_forward(
                    node.embedding,
                    torch.zeros(0, self.hidden_dim, device=node.embedding.device),
                    torch.zeros(0, self.hidden_dim, device=node.embedding.device),
                )
            else:
                # Process children first
                child_states = [process_node(child) for child in node.children]
                child_h = torch.stack([s[0] for s in child_states])
                child_c = torch.stack([s[1] for s in child_states])

                h, c = self.node_forward(node.embedding, child_h, child_c)

            node.hidden = h
            node.cell = c
            return h, c

        return process_node(tree)


class NaryTreeLSTM(TreeLSTM):
    """N-ary Tree-LSTM with position-specific weights.

    Uses separate weight matrices for each child position.
    Suitable for trees with fixed, ordered branching factor.

    More expressive than Child-Sum but requires fixed branching.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_children: int = 2,  # Binary tree by default
        dropout: float = 0.0,
    ):
        """Initialize N-ary Tree-LSTM.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden state dimension
            n_children: Maximum number of children
            dropout: Dropout probability
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.n_children = n_children

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_children)])

        # Forget gates (one per child position)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_children)])

        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_children)])

        # Cell input
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_children)])

    def node_forward(
        self,
        x: torch.Tensor,
        child_h: list[torch.Tensor],
        child_c: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hidden state for a single node.

        Args:
            x: Node input embedding
            child_h: List of child hidden states (may be shorter than n_children)
            child_c: List of child cell states

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        device = x.device

        # Pad to n_children
        while len(child_h) < self.n_children:
            child_h.append(torch.zeros(self.hidden_dim, device=device))
            child_c.append(torch.zeros(self.hidden_dim, device=device))

        # Input gate
        i = self.W_i(x)
        for k in range(self.n_children):
            i = i + self.U_i[k](child_h[k])
        i = torch.sigmoid(i)

        # Output gate
        o = self.W_o(x)
        for k in range(self.n_children):
            o = o + self.U_o[k](child_h[k])
        o = torch.sigmoid(o)

        # Cell input
        u = self.W_u(x)
        for k in range(self.n_children):
            u = u + self.U_u[k](child_h[k])
        u = torch.tanh(u)

        # Per-child forget gates and cell contribution
        fc_sum = torch.zeros(self.hidden_dim, device=device)
        for k in range(self.n_children):
            f_k = torch.sigmoid(self.W_f(x) + self.U_f[k](child_h[k]))
            fc_sum = fc_sum + f_k * child_c[k]

        # Cell state
        c = i * u + fc_sum

        # Hidden state
        h = o * torch.tanh(c)

        return self.dropout(h), c

    def forward(
        self,
        tree: TreeNode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process tree bottom-up.

        Args:
            tree: Root node of tree

        Returns:
            Root (hidden_state, cell_state)
        """

        def process_node(node: TreeNode) -> tuple[torch.Tensor, torch.Tensor]:
            if not node.children:
                # Leaf node
                h, c = self.node_forward(node.embedding, [], [])
            else:
                # Process children first
                child_states = [process_node(child) for child in node.children]
                child_h = [s[0] for s in child_states]
                child_c = [s[1] for s in child_states]

                h, c = self.node_forward(node.embedding, child_h, child_c)

            node.hidden = h
            node.cell = c
            return h, c

        return process_node(tree)


class ProteinTreeEncoder(nn.Module):
    """Encode protein sequences using Tree-LSTM on parse trees.

    Combines:
    1. Amino acid embedding
    2. Parse tree construction (from grammar)
    3. Tree-LSTM encoding
    """

    def __init__(
        self,
        vocab_size: int = 21,  # 20 AAs + padding
        embed_dim: int = 64,
        hidden_dim: int = 128,
        tree_type: str = "child_sum",
        dropout: float = 0.1,
    ):
        """Initialize protein tree encoder.

        Args:
            vocab_size: Size of amino acid vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Tree-LSTM hidden dimension
            tree_type: 'child_sum' or 'nary'
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Non-terminal embedding (for grammar symbols)
        self.symbol_embedding = nn.Embedding(50, embed_dim)  # 50 non-terminal types

        # Projection from embed to hidden
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Tree-LSTM
        if tree_type == "child_sum":
            self.tree_lstm = ChildSumTreeLSTM(hidden_dim, hidden_dim, dropout)
        else:
            self.tree_lstm = NaryTreeLSTM(hidden_dim, hidden_dim, dropout=dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Symbol to index mapping
        self.symbol_to_idx: dict[str, int] = {
            "PROTEIN": 1,
            "REGION": 2,
            "GLYCO": 3,
            "PHOSPHO": 4,
            "BINDING": 5,
            "NLS": 6,
            "TM": 7,
            "HELIX": 8,
            "SHEET": 9,
        }

    def parse_tree_to_tree_node(
        self,
        parse_tree: ParseTree,
        sequence: str,
    ) -> TreeNode:
        """Convert ParseTree to TreeNode with embeddings.

        Args:
            parse_tree: Parse tree from grammar
            sequence: Original sequence for embedding lookup

        Returns:
            TreeNode ready for Tree-LSTM
        """
        # Amino acid to index
        aa_to_idx = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        node_id = [0]  # Counter for unique IDs

        def convert_node(pt: ParseTree) -> TreeNode:
            current_id = node_id[0]
            node_id[0] += 1

            # Get embedding
            if pt.is_leaf():
                # Terminal (amino acid)
                aa = pt.sequence[0] if pt.sequence else "X"
                idx = aa_to_idx.get(aa, 0)
                embed = self.embedding(torch.tensor([idx]))[0]
            else:
                # Non-terminal
                symbol_idx = self.symbol_to_idx.get(pt.symbol, 0)
                embed = self.symbol_embedding(torch.tensor([symbol_idx]))[0]

            # Project to hidden dim
            embed = self.input_proj(embed)

            # Create node
            node = TreeNode(
                id=current_id,
                embedding=embed,
                label=pt.symbol,
            )

            # Convert children
            for child_pt in pt.children:
                child_node = convert_node(child_pt)
                child_node.parent = node
                node.children.append(child_node)

            return node

        return convert_node(parse_tree)

    def forward(
        self,
        parse_tree: ParseTree,
        sequence: str,
    ) -> torch.Tensor:
        """Encode protein using Tree-LSTM on parse tree.

        Args:
            parse_tree: Parse tree from peptide grammar
            sequence: Original amino acid sequence

        Returns:
            Protein embedding (hidden_dim,)
        """
        # Convert parse tree to Tree-LSTM input
        tree_node = self.parse_tree_to_tree_node(parse_tree, sequence)

        # Run Tree-LSTM
        h, c = self.tree_lstm(tree_node)

        # Project output
        output = self.output_proj(h)

        return output

    def encode_batch(
        self,
        parse_trees: list[ParseTree],
        sequences: list[str],
    ) -> torch.Tensor:
        """Encode batch of proteins.

        Args:
            parse_trees: List of parse trees
            sequences: List of sequences

        Returns:
            Batch of embeddings (batch, hidden_dim)
        """
        embeddings = []
        for pt, seq in zip(parse_trees, sequences, strict=False):
            emb = self.forward(pt, seq)
            embeddings.append(emb)

        return torch.stack(embeddings)


def collect_tree_states(
    root: TreeNode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all hidden and cell states from a processed tree.

    Args:
        root: Root of processed tree (after Tree-LSTM forward)

    Returns:
        Tuple of (all_hidden, all_cell) tensors
    """
    hidden_states = []
    cell_states = []

    def collect(node: TreeNode):
        if node.hidden is not None:
            hidden_states.append(node.hidden)
        if node.cell is not None:
            cell_states.append(node.cell)
        for child in node.children:
            collect(child)

    collect(root)

    if hidden_states:
        return torch.stack(hidden_states), torch.stack(cell_states)
    else:
        return torch.zeros(0), torch.zeros(0)


__all__ = [
    "TreeNode",
    "TreeLSTM",
    "ChildSumTreeLSTM",
    "NaryTreeLSTM",
    "ProteinTreeEncoder",
    "collect_tree_states",
]
