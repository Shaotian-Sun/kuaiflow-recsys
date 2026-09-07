"""A transparent DeepFM + multi-gate mixture-of-experts ranker."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for output in hidden_dims:
        layers.extend(
            [nn.Linear(current, int(output)), nn.ReLU(), nn.Dropout(dropout)]
        )
        current = int(output)
    return nn.Sequential(*layers)


class DeepFMMMoE(nn.Module):
    """Share DeepFM embeddings/FM and let every task gate shared experts.

    For task ``k`` the model computes

    ``z_k = linear_k + fm_scale_k * shared_fm + tower_k(mixture_k)``.

    ``mixture_k`` is a task-specific convex combination of the common expert
    outputs.  Task modules are stored by position instead of task name because
    a valid outcome such as ``forward`` conflicts with a built-in PyTorch
    module method when used directly as a ``ModuleDict`` key.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        numeric_dim: int,
        task_names: Sequence[str],
        embedding_dim: int = 16,
        num_experts: int = 4,
        expert_hidden_dims: Sequence[int] = (128, 64),
        tower_hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not cardinalities and numeric_dim <= 0:
            raise ValueError("DeepFMMMoE requires at least one feature field")
        if any(int(cardinality) <= 1 for cardinality in cardinalities):
            raise ValueError("Categorical cardinalities must include at least two IDs")
        if numeric_dim < 0 or embedding_dim <= 0:
            raise ValueError("numeric_dim must be non-negative and embedding_dim positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not expert_hidden_dims or any(int(value) <= 0 for value in expert_hidden_dims):
            raise ValueError("expert_hidden_dims must contain positive values")
        if tower_hidden_dim <= 0:
            raise ValueError("tower_hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        normalized_tasks = tuple(str(name) for name in task_names)
        if not normalized_tasks or any(not name for name in normalized_tasks):
            raise ValueError("task_names must contain at least one non-empty name")
        if len(set(normalized_tasks)) != len(normalized_tasks):
            raise ValueError("task_names must be unique")

        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.numeric_dim = int(numeric_dim)
        self.task_names = normalized_tasks
        self.embedding_dim = int(embedding_dim)
        self.num_experts = int(num_experts)
        self.expert_hidden_dims = tuple(int(value) for value in expert_hidden_dims)
        self.tower_hidden_dim = int(tower_hidden_dim)

        # Shared by the FM interaction branch and every MMoE expert/gate.
        self.feature_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, self.embedding_dim, padding_idx=0)
            for cardinality in self.cardinalities
        )
        self.numeric_embeddings = nn.Parameter(
            torch.empty(self.numeric_dim, self.embedding_dim)
        )

        # Each outcome keeps its own first-order memorization terms.
        self.first_order_embeddings = nn.ModuleList(
            nn.ModuleList(
                nn.Embedding(cardinality, 1, padding_idx=0)
                for cardinality in self.cardinalities
            )
            for _ in self.task_names
        )
        self.numeric_first_order = nn.ParameterList(
            nn.Parameter(torch.zeros(self.numeric_dim)) for _ in self.task_names
        )
        self.task_biases = nn.ParameterList(
            nn.Parameter(torch.zeros(1)) for _ in self.task_names
        )
        self.fm_scales = nn.ParameterList(
            nn.Parameter(torch.ones(1)) for _ in self.task_names
        )

        field_count = len(self.cardinalities) + self.numeric_dim
        input_dim = field_count * self.embedding_dim
        expert_output_dim = self.expert_hidden_dims[-1]
        self.experts = nn.ModuleList(
            _mlp(input_dim, self.expert_hidden_dims, dropout)
            for _ in range(self.num_experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(input_dim, self.num_experts) for _ in self.task_names
        )
        self.towers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(expert_output_dim, self.tower_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.tower_hidden_dim, 1),
            )
            for _ in self.task_names
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.feature_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        if self.numeric_dim:
            nn.init.normal_(self.numeric_embeddings, mean=0.0, std=0.01)
        for task_embeddings in self.first_order_embeddings:
            for embedding in task_embeddings:
                nn.init.zeros_(embedding.weight)
        for parameter in self.numeric_first_order:
            nn.init.zeros_(parameter)
        for parameter in self.task_biases:
            nn.init.zeros_(parameter)
        for parameter in self.fm_scales:
            nn.init.ones_(parameter)

    def _validate_inputs(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> None:
        if categorical.ndim != 2 or categorical.shape[1] != len(self.cardinalities):
            raise ValueError("categorical has the wrong number of fields")
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim:
            raise ValueError("numeric has the wrong number of fields")
        if len(categorical) != len(numeric):
            raise ValueError("categorical and numeric batch sizes must match")

    def _field_embeddings(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        fields = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.feature_embeddings)
        ]
        if self.numeric_dim:
            fields.extend(
                numeric[:, index : index + 1] * self.numeric_embeddings[index]
                for index in range(self.numeric_dim)
            )
        return torch.stack(fields, dim=1)

    def component_logits(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expose every branch and gate for learning/debugging inspections."""
        self._validate_inputs(categorical, numeric)
        fields = self._field_embeddings(categorical, numeric)
        flattened = fields.flatten(start_dim=1)

        summed = fields.sum(dim=1)
        shared_fm = 0.5 * (
            summed.square() - fields.square().sum(dim=1)
        ).sum(dim=1)
        expert_outputs = torch.stack(
            [expert(flattened) for expert in self.experts], dim=1
        )

        linear_rows: list[torch.Tensor] = []
        fm_rows: list[torch.Tensor] = []
        mmoe_rows: list[torch.Tensor] = []
        gate_rows: list[torch.Tensor] = []
        for task_index in range(len(self.task_names)):
            linear = self.task_biases[task_index].expand(len(categorical)).clone()
            for field_index, embedding in enumerate(
                self.first_order_embeddings[task_index]
            ):
                linear = linear + embedding(
                    categorical[:, field_index]
                ).squeeze(1)
            if self.numeric_dim:
                linear = linear + (
                    numeric * self.numeric_first_order[task_index]
                ).sum(dim=1)

            gate = torch.softmax(self.gates[task_index](flattened), dim=1)
            mixture = torch.einsum("be,beh->bh", gate, expert_outputs)
            tower_logit = self.towers[task_index](mixture).squeeze(1)
            linear_rows.append(linear)
            fm_rows.append(self.fm_scales[task_index] * shared_fm)
            mmoe_rows.append(tower_logit)
            gate_rows.append(gate)

        return {
            "linear": torch.stack(linear_rows, dim=1),
            "fm": torch.stack(fm_rows, dim=1),
            "mmoe": torch.stack(mmoe_rows, dim=1),
            "gate_weights": torch.stack(gate_rows, dim=1),
            "shared_fm": shared_fm,
            "expert_outputs": expert_outputs,
        }

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        components = self.component_logits(categorical, numeric)
        return components["linear"] + components["fm"] + components["mmoe"]


__all__ = ["DeepFMMMoE"]
