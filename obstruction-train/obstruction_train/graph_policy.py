import math

import torch
import torch.nn as nn


class EdgeAwareGraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, hidden_dim // num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edges, node_mask):
        batch_size, num_nodes, hidden_dim = nodes.shape
        normalized = self.norm(nodes)
        shape = (batch_size, num_nodes, self.num_heads, self.head_dim)
        query, key, value = (layer(normalized).view(shape) for layer in (self.query, self.key, self.value))
        attention = torch.einsum("bnhd,bmhd->bhnm", query, key) / math.sqrt(self.head_dim)
        attention = attention + self.edge_bias(edges).permute(0, 3, 1, 2)
        attention = attention.masked_fill(~node_mask[:, None, None, :], float("-inf"))
        messages = torch.einsum("bhnm,bmhd->bnhd", self.dropout(torch.softmax(attention, dim=-1)), value).reshape(batch_size, num_nodes, hidden_dim)
        nodes = nodes + self.dropout(self.output(messages))
        return (nodes + self.dropout(self.ffn(self.ffn_norm(nodes)))) * node_mask[..., None]


class TaskConditionedGraphPolicy(nn.Module):
    """Scores every valid object as the next high-level grasp action."""

    TASK_TARGET = 0
    TASK_CLEAR_TABLE = 1

    def __init__(self, node_dim, edge_dim, task_dim=0, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.task_type_embedding = nn.Embedding(2, hidden_dim)
        self.target_embedding = nn.Embedding(2, hidden_dim)
        self.task_proj = nn.Linear(task_dim, hidden_dim) if task_dim else None
        self.layers = nn.ModuleList([EdgeAwareGraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_features, edge_features, node_mask, task_type, target_mask=None, task_features=None):
        if edge_features.shape[:3] != node_features.shape[:2] + (node_features.shape[1],):
            raise ValueError("edge_features must be [B,N,N,D] for node_features [B,N,D]")
        if target_mask is None:
            target_mask = torch.zeros_like(node_mask)
        nodes = self.node_proj(node_features) + self.task_type_embedding(task_type)[:, None, :] + self.target_embedding(target_mask.long())
        if task_features is not None:
            if self.task_proj is None:
                raise ValueError("task_features require task_dim > 0")
            nodes = nodes + self.task_proj(task_features)[:, None, :]
        edges = self.edge_proj(edge_features)
        for layer in self.layers:
            nodes = layer(nodes, edges, node_mask)
        return self.action_head(nodes).squeeze(-1).masked_fill(~node_mask, float("-inf"))
