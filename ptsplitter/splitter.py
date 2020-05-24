import torch
import torch.nn as nn


class SplitterEmbedding(nn.Module):
    def __init__(
        self,
        node_count: int,
        persona_node_count: int,
        embedding_dimension: int,
        lamb: float = 0.1,
        initial_embedding=None,  # TODO add types
        initial_persona_embedding=None,
    ) -> None:  # TODO add types
        super(SplitterEmbedding, self).__init__()
        self.node_count = node_count
        self.persona_node_count = persona_node_count
        self.embedding_dimension = embedding_dimension
        self.lamb = lamb
        # embedding layers
        self.persona_embedding = nn.Embedding(
            num_embeddings=self.persona_node_count,
            embedding_dim=self.embedding_dimension,
            padding_idx=0,
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.node_count,
            embedding_dim=self.embedding_dimension,
            padding_idx=0,
        )
        if initial_embedding is None:
            initial_embedding = torch.ones(
                node_count, embedding_dimension
            )  # TODO change to random
        else:
            initial_embedding = torch.from_numpy(initial_embedding)
        if initial_persona_embedding is None:
            initial_persona_embedding = torch.ones(
                persona_node_count, embedding_dimension
            )  # TODO change to random
        else:
            initial_persona_embedding = torch.from_numpy(initial_persona_embedding)
        self.persona_embedding.weight.data = torch.nn.Parameter(
            initial_persona_embedding
        )
        self.embedding.weight.data = torch.nn.Parameter(
            initial_embedding, requires_grad=False
        )
        # adaptive softmax layers
        self.predict_persona_embedding = torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=embedding_dimension,
            n_classes=persona_node_count,
            cutoffs=[
                round(node_count / 10),
                2 * round(node_count / 10),
                3 * round(node_count / 10),
            ],
        )
        self.predict_embedding = torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=embedding_dimension,
            n_classes=node_count,
            cutoffs=[
                round(node_count / 10),
                2 * round(node_count / 10),
                3 * round(node_count / 10),
            ],
        )

    def loss(self, persona_batch, pure_node_batch, context_node_batch):
        # main loss
        main_loss = self.predict_persona_embedding(
            self.persona_embedding(persona_batch), context_node_batch
        )
        # regularisation loss
        regularisation_loss = self.predict_embedding(
            self.persona_embedding(persona_batch), pure_node_batch
        )
        return main_loss.loss + self.lamb * regularisation_loss.loss

    def forward(self, persona_batch):
        return self.persona_embedding(persona_batch)
