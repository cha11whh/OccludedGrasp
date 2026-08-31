import hashlib
import re

import torch
import torch.nn as nn


class HashTaskEncoder(nn.Module):
    """Dependency-free trainable encoder for target and clear-table instructions."""

    def __init__(self, embedding_dim=64, num_buckets=4096):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, embedding_dim, mode="mean")

    def token_ids(self, instructions, device=None):
        values = []
        offsets = [0]
        for instruction in instructions:
            tokens = re.findall(r"[\w]+", instruction.lower(), flags=re.UNICODE) or ["<empty>"]
            values.extend(int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little") % self.num_buckets for token in tokens)
            offsets.append(len(values))
        return torch.tensor(values, dtype=torch.long, device=device), torch.tensor(offsets[:-1], dtype=torch.long, device=device)

    def forward(self, instructions):
        values, offsets = self.token_ids(instructions, self.embedding.weight.device)
        return self.embedding(values, offsets)
