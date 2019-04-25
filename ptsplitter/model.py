from typing import Any, Callable, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(dataset: torch.utils.data.Dataset,
          model: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          scheduler: Any = None,
          validation: Optional[torch.utils.data.Dataset] = None,
          cuda: bool = True,
          sampler: Optional[torch.utils.data.sampler.Sampler] = None,
          silent: bool = False,
          update_freq: Optional[int] = 1,
          update_callback: Optional[Callable[[float, float], None]] = None,
          epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None) -> None:
    """
    Function to train an model using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.

    :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
    :param model: model to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
    :param update_callback: optional function of loss and validation loss to update
    :param epoch_callback: optional function of epoch and model
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False
        )
    else:
        validation_loader = None
    model.train()
    validation_loss_value = -1
    loss_value = 0
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
                'vls': '%.6f' % -1,
            },
            disable=silent,
        )
        for index, (persona_batch, pure_node_batch, context_node_batch) in enumerate(data_iterator):
            if cuda:
                persona_batch = persona_batch.cuda(non_blocking=True)
                pure_node_batch = pure_node_batch.cuda(non_blocking=True)
                context_node_batch = context_node_batch.cuda(non_blocking=True)
            loss = model.loss(persona_batch, pure_node_batch, context_node_batch)
            # accuracy = pretrain_accuracy(output, batch)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch,
                lss='%.6f' % loss_value,
                vls='%.6f' % validation_loss_value,
            )
        if update_freq is not None and epoch % update_freq == 0:
            if validation_loader is not None:
                pass
                # TODO
                # validation_output = predict(
                #     validation,
                #     model,
                #     batch_size,
                #     cuda=cuda,
                #     silent=True,
                #     encode=False
                # )
                # validation_inputs = []
                # for val_batch in validation_loader:
                #     if (isinstance(val_batch, tuple) or isinstance(val_batch, list)) and len(val_batch) in [1, 2]:
                #         validation_inputs.append(val_batch[0])
                #     else:
                #         validation_inputs.append(val_batch)
                # validation_actual = torch.cat(validation_inputs)
                # if cuda:
                #     validation_actual = validation_actual.cuda(non_blocking=True)
                #     validation_output = validation_output.cuda(non_blocking=True)
                # validation_loss = loss_function(validation_output, validation_actual)
                # # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                # validation_loss_value = float(validation_loss.item())
                # data_iterator.set_postfix(
                #     epo=epoch,
                #     lss='%.6f' % loss_value,
                #     vls='%.6f' % validation_loss_value,
                # )
                # model.train()
            else:
                validation_loss_value = -1
                # validation_accuracy = -1
                data_iterator.set_postfix(
                    epo=epoch,
                    lss='%.6f' % loss_value,
                    vls='%.6f' % -1,
                )
            if update_callback is not None:
                update_callback(epoch, optimizer.param_groups[0]['lr'], loss_value, validation_loss_value)
        if epoch_callback is not None:
            model.eval()
            epoch_callback(epoch, model)
            model.train()


# TODO
# def predict(
#         dataset: torch.utils.data.Dataset,
#         model: torch.nn.Module,
#         batch_size: int,
#         cuda: bool = True,
#         silent: bool = False,
#         encode: bool = True) -> torch.Tensor:
#     """
#     Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
#     output.
#
#     :param dataset: evaluation Dataset
#     :param model: model for prediction
#     :param batch_size: batch size
#     :param cuda: whether CUDA is used, defaults to True
#     :param silent: set to True to prevent printing out summary statistics, defaults to False
#     :param encode: whether to encode or use the full model
#     :return: predicted features from the Dataset
#     """
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         pin_memory=False,
#         shuffle=False
#     )
#     data_iterator = tqdm(
#         dataloader,
#         leave=False,
#         unit='batch',
#         disable=silent,
#     )
#     features = []
#     if isinstance(model, torch.nn.Module):
#         model.eval()
#     for index, batch in enumerate(data_iterator):
#         if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
#             batch = batch[0]
#         if cuda:
#             batch = batch.cuda(non_blocking=True)
#         batch = batch.squeeze(1).view(batch.size(0), -1)
#         if encode:
#             output = model.encode(batch)
#         else:
#             output = model(batch)
#         features.append(output.detach().cpu())  # move to the CPU to prevent out of memory on the GPU
#     return torch.cat(features)
