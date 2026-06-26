import torch
import torch.nn as nn


class EdgeGraphRefiner(nn.Module):
    """V2 module: refine pairwise relation logits with graph message passing.

    Inputs:
        node_features: [B, N, Dn]
        edge_features: [B, N, N, De]

    Output:
        refined edge logits: [B, N, N, C]
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3, num_classes=3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(
                            nn.Linear(hidden_dim * 3, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        "node": nn.GRUCell(hidden_dim, hidden_dim),
                        "edge": nn.Sequential(
                            nn.Linear(hidden_dim * 3, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_features, edge_features, node_mask=None):
        node = self.node_proj(node_features)
        edge = self.edge_proj(edge_features)
        b, n, h = node.shape

        for layer in self.layers:
            src = node[:, :, None, :].expand(b, n, n, h)
            dst = node[:, None, :, :].expand(b, n, n, h)
            edge_input = torch.cat([src, dst, edge], dim=-1)
            msg = layer["msg"](edge_input)

            if node_mask is not None:
                pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
                msg = msg * pair_mask[..., None].float()

            agg = msg.sum(dim=1) / max(1, n)
            node = layer["node"](agg.reshape(b * n, h), node.reshape(b * n, h)).reshape(b, n, h)

            src = node[:, :, None, :].expand(b, n, n, h)
            dst = node[:, None, :, :].expand(b, n, n, h)
            edge = edge + layer["edge"](torch.cat([src, dst, edge], dim=-1))

        return self.out(edge)
