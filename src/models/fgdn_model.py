from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import ChebConv, global_mean_pool


class FGDNBranch(nn.Module):
    """
    One graph-processing branch:
    subject node features + one class-specific template graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.prelu1 = nn.PReLU(hidden_channels)

        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)
        self.prelu2 = nn.PReLU(out_channels)

        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level embedding
        x = global_mean_pool(x, batch)
        return x


class FGDNModel(nn.Module):
    """
    FGDN-style dual-template model.

    Each subject graph is evaluated under:
      - ASD template graph
      - HC template graph

    Then the two graph-level embeddings are fused for final classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        branch_out_channels: int = 64,
        cheb_k: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.asd_branch = FGDNBranch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=branch_out_channels,
            K=cheb_k,
            dropout=dropout,
        )

        self.hc_branch = FGDNBranch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=branch_out_channels,
            K=cheb_k,
            dropout=dropout,
        )

        fusion_dim = 2 * branch_out_channels

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.PReLU(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, data) -> Tuple[Tensor, Tensor]:
        """
        Expected data fields:
          data.x
          data.batch
          data.edge_index_asd
          data.edge_index_hc
        """
        x = data.x
        batch = data.batch

        z_asd = self.asd_branch(x, data.edge_index_asd, batch)
        z_hc = self.hc_branch(x, data.edge_index_hc, batch)

        z = torch.cat([z_asd, z_hc], dim=1)
        logits = self.classifier(z)

        return logits, z


def build_fgdn_model(num_node_features: int) -> FGDNModel:
    """
    Convenience builder using paper-like defaults.
    """
    return FGDNModel(
        in_channels=num_node_features,
        hidden_channels=64,
        branch_out_channels=64,
        cheb_k=3,
        dropout=0.1,
        num_classes=2,
    )