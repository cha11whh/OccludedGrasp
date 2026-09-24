OBSTRUCTION_TYPES = {"obstruction", "occlusion", "blocks", "covers"}


def _confidence(relation):
    return float(relation.get("confidence", relation.get("relation_confidence", 0.0)))


def _is_obstruction(relation):
    relation_type = relation.get("relation_type", relation.get("type"))
    return "blocker" in relation and "blocked" in relation and (
        relation_type is None or str(relation_type).lower() in OBSTRUCTION_TYPES
    )


def _would_create_cycle(adjacency, source, destination):
    if source == destination:
        return True
    pending = [destination]
    visited = set()
    while pending:
        node = pending.pop()
        if node == source:
            return True
        if node not in visited:
            visited.add(node)
            pending.extend(adjacency.get(node, ()))
    return False


def repair_obstruction_relations(relations, min_confidence=0.0):
    indexed = list(enumerate(relations))
    candidates = [(index, row) for index, row in indexed if _is_obstruction(row)]
    strongest = {}
    for index, row in candidates:
        edge = (row["blocker"], row["blocked"])
        current = strongest.get(edge)
        if current is None or _confidence(row) > _confidence(current[1]):
            strongest[edge] = (index, row)

    strongest_indexes = {index for index, _ in strongest.values()}
    removed_duplicates = [
        row for index, row in candidates if index not in strongest_indexes
    ]
    removed_low_confidence = [
        row for _, row in strongest.values() if _confidence(row) < min_confidence
    ]
    eligible = [
        item for item in strongest.values() if _confidence(item[1]) >= min_confidence
    ]
    eligible.sort(key=lambda item: (-_confidence(item[1]), item[0]))

    adjacency = {}
    accepted_indexes = set()
    removed_cycle_indexes = set()
    for index, row in eligible:
        source, destination = row["blocker"], row["blocked"]
        if source in adjacency.get(destination, ()) or _would_create_cycle(
            adjacency, source, destination
        ):
            removed_cycle_indexes.add(index)
        else:
            adjacency.setdefault(source, set()).add(destination)
            accepted_indexes.add(index)

    repaired = [
        row
        for index, row in indexed
        if not _is_obstruction(row) or index in accepted_indexes
    ]
    report = {
        "input_relations": len(relations),
        "output_relations": len(repaired),
        "removed_low_confidence": removed_low_confidence,
        "removed_duplicates": removed_duplicates,
        "removed_cycles": [
            row for index, row in indexed if index in removed_cycle_indexes
        ],
        "is_acyclic": True,
    }
    return repaired, report
