"""DeepFM for pointwise click-through-rate prediction.

The same field embeddings feed both the explicit second-order FM interaction
and the deep network. This is the defining parameter-sharing idea in DeepFM.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DeepFM(nn.Module):
    """Combine linear, factorization-machine, and deep logits.

    Categorical and numerical values are both represented as fields. A numeric
    field ``x_i`` has a learned vector ``v_i`` and contributes ``x_i * v_i`` to
    the FM/deep branches, matching the standard FM formulation.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        numeric_dim: int,
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not cardinalities and numeric_dim <= 0:
            raise ValueError("DeepFM requires at least one feature field")
        if any(cardinality <= 1 for cardinality in cardinalities):
            raise ValueError("Categorical cardinalities must include at least two IDs")
        if numeric_dim < 0 or embedding_dim <= 0:
            raise ValueError("numeric_dim must be non-negative and embedding_dim positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.numeric_dim = int(numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        if any(value <= 0 for value in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")

        # First-order terms memorize the independent contribution of each value.
        self.first_order_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, 1, padding_idx=0)
            for cardinality in self.cardinalities
        )
        self.numeric_first_order = nn.Parameter(torch.zeros(self.numeric_dim))
        self.bias = nn.Parameter(torch.zeros(1))

        # These embeddings are shared by the FM and deep branches.
        self.feature_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, self.embedding_dim, padding_idx=0)
            for cardinality in self.cardinalities
        )
        self.numeric_embeddings = nn.Parameter(
            torch.empty(self.numeric_dim, self.embedding_dim)
        )

        field_count = len(self.cardinalities) + self.numeric_dim
        layers: list[nn.Module] = []
        input_dim = field_count * self.embedding_dim
        for output_dim in self.hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.first_order_embeddings:
            nn.init.zeros_(embedding.weight)
        for embedding in self.feature_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        if self.numeric_dim:
            nn.init.normal_(self.numeric_embeddings, mean=0.0, std=0.01)
        nn.init.zeros_(self.numeric_first_order)
        nn.init.zeros_(self.bias)

    def _field_embeddings(
        self, categorical: torch.Tensor, numeric: torch.Tensor
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
        self, categorical: torch.Tensor, numeric: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return each branch separately for inspection and testing."""
        if categorical.ndim != 2 or categorical.shape[1] != len(self.cardinalities):
            raise ValueError("categorical has the wrong number of fields")
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim:
            raise ValueError("numeric has the wrong number of fields")
        if len(categorical) != len(numeric):
            raise ValueError("categorical and numeric batch sizes must match")

        linear = self.bias.expand(len(categorical)).clone()
        for index, embedding in enumerate(self.first_order_embeddings):
            linear = linear + embedding(categorical[:, index]).squeeze(1)
        if self.numeric_dim:
            linear = linear + (numeric * self.numeric_first_order).sum(dim=1)

        fields = self._field_embeddings(categorical, numeric)
        summed = fields.sum(dim=1)
        fm = 0.5 * (summed.square() - fields.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(fields.flatten(start_dim=1)).squeeze(1)
        return {"linear": linear, "fm": fm, "deep": deep}

    def forward(
        self, categorical: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        components = self.component_logits(categorical, numeric)
        return components["linear"] + components["fm"] + components["deep"]
