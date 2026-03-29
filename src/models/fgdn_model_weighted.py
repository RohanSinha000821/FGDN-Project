from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import ChebConv


class FGDNBranchWeighted(nn.Module):
    """
    One FGDN branch using weighted graph templates.

    Pipeline:
      ChebConv -> PReLU -> Dropout
      ChebConv -> PReLU -> Dropout
      Flatten all node features
      Linear -> scalar branch score
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes: int,
        hidden_channels: int = 64,
        K: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.prelu1 = nn.PReLU(hidden_channels)

        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=K)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.fc = nn.Linear(num_nodes * hidden_channels, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        expected_total_nodes = batch_size * self.num_nodes
        if x.size(0) != expected_total_nodes:
            raise ValueError(
                f"FGDNBranchWeighted reshape mismatch: got total nodes {x.size(0)}, "
                f"expected {expected_total_nodes} = "
                f"batch_size({batch_size}) * num_nodes({self.num_nodes})."
            )

        x = x.view(batch_size, self.num_nodes * self.hidden_channels)
        branch_score = self.fc(x)
        return branch_score


class FGDNModelWeighted(nn.Module):
    """
    Weighted dual-template FGDN model.

    Project label convention:
      HC = 0
      ASD = 1

    Therefore CrossEntropy logits must be returned as:
      [HC_logit, ASD_logit]

    branch_scores are returned separately as:
      [ASD_branch_score, HC_branch_score]
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes: int,
        hidden_channels: int = 64,
        cheb_k: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        if num_classes != 2:
            raise ValueError("FGDNModelWeighted supports binary ASD/HC classification only.")

        self.num_nodes = num_nodes

        self.asd_branch = FGDNBranchWeighted(
            in_channels=in_channels,
            num_nodes=num_nodes,
            hidden_channels=hidden_channels,
            K=cheb_k,
            dropout=dropout,
        )

        self.hc_branch = FGDNBranchWeighted(
            in_channels=in_channels,
            num_nodes=num_nodes,
            hidden_channels=hidden_channels,
            K=cheb_k,
            dropout=dropout,
        )

    def forward(self, data) -> Tuple[Tensor, Tensor]:
        x = data.x
        batch = data.batch

        asd_score = self.asd_branch(
            x,
            data.edge_index_asd,
            data.edge_weight_asd,
            batch,
        )

        hc_score = self.hc_branch(
            x,
            data.edge_index_hc,
            data.edge_weight_hc,
            batch,
        )

        logits = torch.cat([hc_score, asd_score], dim=1)        # [HC, ASD]
        branch_scores = torch.cat([asd_score, hc_score], dim=1) # [ASD, HC]

        return logits, branch_scores


def build_fgdn_model_weighted(num_node_features: int, num_nodes: int) -> FGDNModelWeighted:
    return FGDNModelWeighted(
        in_channels=num_node_features,
        num_nodes=num_nodes,
        hidden_channels=64,
        cheb_k=3,
        dropout=0.1,
        num_classes=2,
    )