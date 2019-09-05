import numpy as np
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple
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
          silent: bool = False,
          epoch_callback: Optional[Callable[[int, float, float], None]] = None) -> None:
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
    :param epoch_callback: function of epoch, learning rate and average loss called per epoch, default disabled.
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
        losses = []
        for index, (persona_batch, pure_node_batch, context_node_batch) in enumerate(data_iterator):
            if cuda:
                persona_batch = persona_batch.cuda(non_blocking=True)
                pure_node_batch = pure_node_batch.cuda(non_blocking=True)
                context_node_batch = context_node_batch.cuda(non_blocking=True)
            loss = model.loss(persona_batch, pure_node_batch, context_node_batch)
            loss_value = float(loss.item())
            losses.append(loss_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch,
                lss='%.6f' % loss_value,
            )
        if epoch_callback is not None:
            epoch_callback(epoch, optimizer.param_groups[0]['lr'], np.mean(losses))


def predict(reverse_persona: Dict[int, PersonaNode],
            model: torch.nn.Module) -> Tuple[List[PersonaNode], List[Hashable], List[int], List[np.ndarray]]:
    """
    Utility function to run the given model to obtain embeddings for all the nodes
    with some associated metadata. The output can all be zipped up by index.

    :param reverse_persona: lookup from index to PersonaNode
    :param model: instance of the PyTorch model
    :return: 4-tuple of list of PersonaNode object, original nodes, index nodes, and embeddings
    """
    persona_embedding = model.persona_embedding.weight.detach().cpu()
    persona_node_list = []
    node_list = []
    index_list = []
    persona_embedding_list = []
    for index in reverse_persona:
        persona_node_list.append(reverse_persona[index])
        node_list.append(reverse_persona[index].node)
        index_list.append(reverse_persona[index].index)
        persona_embedding_list.append(persona_embedding[index].numpy())
    return persona_node_list, node_list, index_list, persona_embedding_list
