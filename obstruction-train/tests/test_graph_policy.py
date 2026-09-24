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
    assert torch.allclose(edges[0, 1, :4], torch.tensor([1.0, 0.7, 0.4, 0.0]))
    assert torch.allclose(edges[0, 1, 4:], torch.zeros(3))


def test_hash_task_encoder_is_batched_and_trainable():
    from obstruction_train.task_text_encoder import HashTaskEncoder

    encoder = HashTaskEncoder(embedding_dim=16, num_buckets=128)
    vectors = encoder(["grasp the red cup", "clear table", "grasp the red cup"])
    assert vectors.shape == (3, 16)
    assert torch.allclose(vectors[0], vectors[2])
    vectors.sum().backward()
    assert encoder.embedding.weight.grad is not None


def test_graph_adapter_encodes_support_relation():
    from obstruction_train.plan_graph_policy import build_graph

    _, _, edges = build_graph(
        [{"id": 1, "bbox": [0, 0, 10, 10]}, {"id": 2, "bbox": [0, 10, 10, 20]}],
        [{"source": 1, "target": 2, "relation_type": "support", "confidence": 0.8}],
    )
    assert edges[0, 1, 4] == 0.8


def test_graph_ablation_disables_requested_relation_families():
    from obstruction_train.plan_graph_policy import build_graph

    objects = [{"id": 1, "bbox": [0, 0, 10, 10]}, {"id": 2, "bbox": [5, 0, 15, 10]}]
    relations = [{"blocker": 1, "blocked": 2, "confidence": 0.8}, {"source": 1, "target": 2, "relation_type": "support", "confidence": 0.7}]
    _, _, edges = build_graph(objects, relations, features={"obstruction": False, "support": False, "nearby": False})
    assert torch.count_nonzero(edges) == 0


def test_graph_consistency_removes_weakest_cycle_edge():
    from obstruction_train.graph_consistency import repair_obstruction_relations

    relations = [
        {"blocker": 1, "blocked": 2, "confidence": 0.9},
        {"blocker": 2, "blocked": 3, "confidence": 0.8},
        {"blocker": 3, "blocked": 1, "confidence": 0.2},
    ]

    repaired, report = repair_obstruction_relations(relations)

    assert [(row["blocker"], row["blocked"]) for row in repaired] == [(1, 2), (2, 3)]
    assert report["removed_cycles"] == [{"blocker": 3, "blocked": 1, "confidence": 0.2}]
    assert report["is_acyclic"] is True

def test_build_graph_optionally_repairs_obstruction_cycles():
    from obstruction_train.plan_graph_policy import build_graph

    objects = [
        {"id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "bbox": [20, 0, 30, 10]},
        {"id": 3, "bbox": [40, 0, 50, 10]},
    ]
    relations = [
        {"blocker": 1, "blocked": 2, "confidence": 0.9},
        {"blocker": 2, "blocked": 3, "confidence": 0.8},
        {"blocker": 3, "blocked": 1, "confidence": 0.2},
    ]

    _, _, repaired_edges = build_graph(
        objects,
        relations,
        repair_graph=True,
        min_relation_confidence=0.0,
    )
    assert repaired_edges[0, 1, 0] == 1
    assert repaired_edges[1, 2, 0] == 1
    assert repaired_edges[2, 0, 0] == 0

    _, _, unrepaired_edges = build_graph(
        objects,
        relations,
        repair_graph=False,
        min_relation_confidence=0.0,
    )
    assert unrepaired_edges[0, 1, 0] == 1
    assert unrepaired_edges[1, 2, 0] == 1
    assert unrepaired_edges[2, 0, 0] == 1
