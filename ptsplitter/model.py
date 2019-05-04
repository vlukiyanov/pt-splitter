from typing import Any, Optional, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptsplitter.persona import PersonaNode


def train(dataset: torch.utils.data.Dataset,
          model: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          scheduler: Any = None,
          cuda: bool = True,
          sampler: Optional[torch.utils.data.sampler.Sampler] = None,
          silent: bool = False) -> None:
    """
    Function to train an model using the provided dataset.

    :param dataset: training Dataset
    :param model: model to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True
    )
    model.train()
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit='batch',
            postfix={
                'epo': epoch,
                'lss': '%.6f' % 0.0,
            },
            disable=silent
        )
        for index, (persona_batch, pure_node_batch, context_node_batch) in enumerate(data_iterator):
            if cuda:
                persona_batch = persona_batch.cuda(non_blocking=True)
                pure_node_batch = pure_node_batch.cuda(non_blocking=True)
                context_node_batch = context_node_batch.cuda(non_blocking=True)
            loss = model.loss(persona_batch, pure_node_batch, context_node_batch)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch,
                lss='%.6f' % loss_value,
            )


def predict(reverse_persona: Dict[int, PersonaNode], model: torch.nn.Module) -> Dict[str, Any]:
    persona_embedding = model.persona_embedding.weight.detach().cpu()
    data = {
        'persona_node': [],
        'node': [],
        'index': [],
        'embedding_vector': []
    }
    for index in reverse_persona:
        data['persona_node'].append(reverse_persona[index])
        data['node'].append(reverse_persona[index].node)
        data['index'].append(reverse_persona[index].index)
        data['embedding_vector'].append(persona_embedding[index].numpy())
    return data
