import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obstruction_train.graph_policy import TaskConditionedGraphPolicy


def test_graph_policy_masks_invalid_actions_and_backpropagates():
    model = TaskConditionedGraphPolicy(node_dim=10, edge_dim=5, hidden_dim=32, num_heads=4, num_layers=2)
    nodes = torch.randn(2, 4, 10, requires_grad=True)
    edges = torch.randn(2, 4, 4, 5)
    mask = torch.tensor([[True, True, True, False], [True, False, True, False]])
    task_type = torch.tensor([model.TASK_TARGET, model.TASK_CLEAR_TABLE])
    targets = torch.tensor([[False, True, False, False], [False, False, False, False]])
    logits = model(nodes, edges, mask, task_type, targets)
    assert logits.shape == (2, 4)
    assert torch.isneginf(logits[0, 3]) and torch.isneginf(logits[1, 1])
    logits[mask].sum().backward()
    assert nodes.grad is not None


def test_graph_adapter_encodes_target_and_direct_blocker():
    from obstruction_train.plan_graph_policy import build_graph

    ids, nodes, edges = build_graph(
        [{"id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}, {"id": 2, "bbox": [10, 0, 20, 10], "score": 0.8}],
        [{"blocker": 1, "blocked": 2, "confidence": 0.7, "mask_ratio": 0.4}],
        target_id=2,
    )
    assert ids == [1, 2]
    assert nodes[1, 8] == 1 and nodes[0, 9] == 1
    assert torch.allclose(edges[0, 1], torch.tensor([1.0, 0.7, 0.4, 0.0]))
