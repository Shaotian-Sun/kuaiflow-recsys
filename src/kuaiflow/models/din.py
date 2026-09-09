"""DIN and DIN+MMoE with shared target/history embeddings.

A DIN-style local activation unit weights history without softmax normalization.
This controlled implementation uses PReLU rather than paper-specific Dice and
AdamW rather than mini-batch-aware embedding regularization.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn
from kuaiflow.models.mmoe import _mlp


class LocalActivation(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: Sequence[int] = (64, 32)):
        super().__init__()
        if any(int(n) <= 0 for n in hidden_dims):
            raise ValueError("attention hidden dimensions must be positive")
        layers = []
        width = 4 * embedding_dim
        for size in hidden_dims:
            layers.extend([nn.Linear(width, int(size)), nn.PReLU()])
            width = int(size)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, query, keys, mask):
        query = query[:, None, :].expand_as(keys)
        interactions = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        weights = self.network(interactions).squeeze(-1).masked_fill(~mask, 0.0)
        return (keys * weights.unsqueeze(-1)).sum(dim=1), weights


class DIN(nn.Module):
    def __init__(self, cardinalities, numeric_dim, video_field_index,
                 embedding_dim=16, hidden_dims=(128, 64), dropout=0.1,
                 attention_hidden_dims=(64, 32)):
        super().__init__()
        if not cardinalities or any(int(c) <= 1 for c in cardinalities):
            raise ValueError("DIN requires categorical fields with at least two IDs")
        if not 0 <= video_field_index < len(cardinalities):
            raise ValueError("video_field_index must identify the target video field")
        if numeric_dim < 0 or embedding_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid numeric_dim, embedding_dim or dropout")
        if not hidden_dims or any(int(n) <= 0 for n in hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        self.cardinalities = tuple(cardinalities)
        self.numeric_dim = int(numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.video_field_index = int(video_field_index)
        self.feature_embeddings = nn.ModuleList(
            nn.Embedding(c, embedding_dim, padding_idx=0) for c in cardinalities
        )
        self.numeric_embeddings = nn.Parameter(torch.empty(numeric_dim, embedding_dim))
        self.attention = LocalActivation(embedding_dim, attention_hidden_dims)
        self.input_dim = (len(cardinalities) + numeric_dim + 1) * embedding_dim
        self.deep = nn.Sequential(
            _mlp(self.input_dim, hidden_dims, dropout), nn.Linear(hidden_dims[-1], 1)
        )
        for embedding in self.feature_embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.normal_(self.numeric_embeddings, std=0.01)

    def representation(self, categorical, numeric, history):
        if categorical.ndim != 2 or categorical.shape[1] != len(self.cardinalities):
            raise ValueError("categorical has the wrong number of fields")
        if numeric.ndim != 2 or numeric.shape != (len(categorical), self.numeric_dim):
            raise ValueError("numeric has the wrong shape")
        if history.ndim != 2 or len(history) != len(categorical):
            raise ValueError("history must be a batch of video ID sequences")
        fields = [emb(categorical[:, i]) for i, emb in enumerate(self.feature_embeddings)]
        video_embedding = self.feature_embeddings[self.video_field_index]
        interest, weights = self.attention(
            fields[self.video_field_index], video_embedding(history), history > 1
        )
        fields.extend(numeric[:, i:i+1] * self.numeric_embeddings[i]
                      for i in range(self.numeric_dim))
        return torch.cat([*fields, interest], dim=1), weights

    def forward(self, categorical, numeric, history):
        representation, _ = self.representation(categorical, numeric, history)
        return self.deep(representation).squeeze(1)


class DINMMoE(DIN):
    def __init__(self, cardinalities, numeric_dim, video_field_index, task_names,
                 embedding_dim=16, num_experts=4, expert_hidden_dims=(128, 64),
                 tower_hidden_dim=32, dropout=0.1, attention_hidden_dims=(64, 32)):
        if not task_names or len(set(task_names)) != len(task_names) or any(not t for t in task_names):
            raise ValueError("task_names must be non-empty and unique")
        if num_experts <= 0 or tower_hidden_dim <= 0:
            raise ValueError("Expert count and tower width must be positive")
        super().__init__(cardinalities, numeric_dim, video_field_index, embedding_dim,
                         expert_hidden_dims, dropout, attention_hidden_dims)
        del self.deep
        self.task_names = tuple(task_names)
        self.num_experts = int(num_experts)
        self.expert_hidden_dims = tuple(expert_hidden_dims)
        self.tower_hidden_dim = int(tower_hidden_dim)
        self.experts = nn.ModuleList(_mlp(self.input_dim, expert_hidden_dims, dropout)
                                     for _ in range(num_experts))
        self.gates = nn.ModuleList(nn.Linear(self.input_dim, num_experts) for _ in task_names)
        self.towers = nn.ModuleList(nn.Sequential(
            nn.Linear(expert_hidden_dims[-1], tower_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(tower_hidden_dim, 1)) for _ in task_names)
        self.task_biases = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in task_names)

    def component_logits(self, categorical, numeric, history):
        representation, attention = self.representation(categorical, numeric, history)
        experts = torch.stack([expert(representation) for expert in self.experts], dim=1)
        gates = torch.stack([torch.softmax(gate(representation), dim=1) for gate in self.gates], dim=1)
        mixtures = torch.einsum('bte,beh->bth', gates, experts)
        logits = torch.stack([tower(mixtures[:, i]).squeeze(1) + self.task_biases[i]
                              for i, tower in enumerate(self.towers)], dim=1)
        return {"logits": logits, "gate_weights": gates, "attention_weights": attention}

    def forward(self, categorical, numeric, history):
        return self.component_logits(categorical, numeric, history)["logits"]
