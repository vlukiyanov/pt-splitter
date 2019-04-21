import torch
import torch.nn as nn


class SplitterEmbedding(nn.Module):
    def __init__(self,
                 node_count: int,
                 ego_node_count: int,
                 embedding_dimension: int,
                 initial_embedding=None) -> None:
        super(SplitterEmbedding, self).__init__()
        self.node_count = node_count
        self.ego_node_count = ego_node_count
        self.embedding_dimension = embedding_dimension
        # embedding layers
        self.embedding = nn.Embedding(
            num_embeddings=self.node_count,
            embedding_dim=self.embedding_dimension,
            padding_idx=0
        )
        self.persona_embedding = nn.Embedding(
            num_embeddings=self.ego_node_count,
            embedding_dim=self.embedding_dimension,
            padding_idx=0
        )
        if initial_embedding is None:
            initial_embedding = torch.ones(node_count, embedding_dimension)
        else:
            initial_embedding = torch.from_numpy(initial_embedding)
        self.embedding.weight.data = torch.nn.Parameter(initial_embedding)
        self.persona_embedding.weight.data = torch.nn.Parameter(initial_embedding, requires_grad=False)

    def forward(self, batch, persona_batch):
        return {
            'embedding': self.embedding(batch),
            'persona_embedding': self.persona_embedding(persona_batch),
        }
