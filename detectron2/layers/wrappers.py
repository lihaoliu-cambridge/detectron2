# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

from typing import List, Optional
import torch
from torch.nn import functional as F
from typing import Optional, Sequence, Union
from torch import Tensor
from torch import nn


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, reduction=reduction, **kwargs)


# def cross_entropy(input, target, *, reduction="mean", **kwargs):
#     """
#     Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
#     for empty inputs.
#     """
#     if target.numel() == 0 and reduction == "mean":
#         return input.sum() * 0.0  # connect the gradient
#     # # Tried 4     : 
#     weights = torch.tensor([0.1, 20, 1, 1, 4, 20, 1], dtype=torch.float32).to(input.device)

#     # # 3 is not helpful: https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
#     # # Tried 3_max1: 
#     # weights = torch.tensor([0.0109, 0.4778, 0.0109, 0.0200, 0.0925, 1.0000, 0.0230], dtype=torch.float32).to(input.device)
    
#     # # 2 looks helpful:  https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514 (answer frombanikr) 
#     # # Tried 2_min1:    weights = torch.tensor([1.0, 1.4507, 1.0, 1.2104, 1.4069, 1.4562, 1.24308], dtype=torch.float32).to(input.device)
#     # weights = torch.tensor([1.0, 1.4507, 1.0, 1.2104, 1.4069, 1.4562, 1.2430], dtype=torch.float32).to(input.device)
#     # # Tried 2     :    weights = torch.tensor([0.5388, 0.9895,  0.5388, 0.7492, 0.9457, 0.9950, 0.7818], dtype=torch.float32).to(input.device)
#     # weights = torch.tensor([0.5388, 0.9895,  0.5388, 0.7492, 0.9457, 0.9950, 0.7818], dtype=torch.float32).to(input.device)
    
#     # # 1 is not helpful: https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514 (answer from beaupreda)
#     # # Tried 1_min1:    weights = torch.tensor([1.0,    43.9264, 1.0,    1.839,  8.5005, 91.939, 2.1144], dtype=torch.float32).to(input.device)
#     # weights = torch.tensor([1.0,    43.9264, 1.0,    1.839,  8.5005, 91.939, 2.1144], dtype=torch.float32).to(input.device)
#     # # Tried 1     :    weights = torch.tensor([0.0067, 0.2942,  0.0067, 0.0123, 0.0569, 0.6157, 0.0142], dtype=torch.float32).to(input.device)  # should be [0.0402, ...] not [1, ...]
#     # weights = torch.tensor([0.0067, 0.2942,  0.0067, 0.0123, 0.0569, 0.6157, 0.0142], dtype=torch.float32).to(input.device) 

#     return F.cross_entropy(input, target, weight=weights, reduction=reduction, **kwargs)


# def cross_entropy(input, target, *, reduction="mean", **kwargs):
#     """
#     Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
#     for empty inputs.
#     """
#     if target.numel() == 0 and reduction == "mean":
#         return input.sum() * 0.0  # connect the gradient

#     # focal_loss = FocalLoss(alpha=torch.tensor([0.2222222222222222, 0.05555555555555555, 0.2222222222222222, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.16666666666666666]), gamma=2)
#     # return focal_loss(input, target)

#     seasaw_loss = SeesawLoss(class_counts=[230689, 3912, 230689, 92986, 25919, 3290, 101643])
#     return seasaw_loss(input, target)


# class FocalLoss(nn.Module):
#     def __init__(self,
#                  alpha: Optional[Tensor] = None,
#                  gamma: float = 0.,
#                  reduction: str = 'mean',
#                  ignore_index: int = -100):
#         """Constructor.

#         Args:
#             alpha (Tensor, optional): Weights for each class. Defaults to None.
#             gamma (float, optional): A constant, as described in the paper.
#                 Defaults to 0.
#             reduction (str, optional): 'mean', 'sum' or 'none'.
#                 Defaults to 'mean'.
#             ignore_index (int, optional): class label to ignore.
#                 Defaults to -100.
#         """
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(
#                 'Reduction must be one of: "mean", "sum", "none".')

#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#         self.nll_loss = nn.NLLLoss(
#             weight=alpha, reduction='none', ignore_index=ignore_index)

#     def __repr__(self):
#         arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
#         arg_vals = [self.__dict__[k] for k in arg_keys]
#         arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
#         arg_str = ', '.join(arg_strs)
#         return f'{type(self).__name__}({arg_str})'

#     def forward(self, x: Tensor, y: Tensor) -> Tensor:
#         if x.ndim > 2:
#             # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
#             y = y.view(-1)

#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0:
#             return 0.
#         x = x[unignored_mask]

#         # compute weighted cross entropy term: -alpha * log(pt)
#         # (alpha is already part of self.nll_loss)
#         log_p = F.log_softmax(x, dim=-1)
#         self.nll_loss = self.nll_loss.to(x.device)
#         ce = self.nll_loss(log_p, y)

#         # get true class column from each row
#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]

#         # compute focal term: (1 - pt)^gamma
#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma

#         # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#         loss = focal_term * ce

#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()

#         return loss


# class SeesawLoss(nn.Module):
#     def __init__(self, class_counts: list, p: float = 0.8):
#         super().__init__()

#         class_counts = torch.FloatTensor(class_counts)
#         conditions = class_counts[:, None] > class_counts[None, :]
#         trues = (class_counts[None, :] / class_counts[:, None]) ** p
#         falses = torch.ones(len(class_counts), len(class_counts))
#         self.s = torch.where(conditions, trues, falses)
        
#         self.eps = 1.0e-6

#     def forward(self, logits, targets):
#         targets = F.one_hot(targets.long(), num_classes=7)
        
#         self.s = self.s.to(targets.device)
#         max_element, _ = logits.max(axis=-1)
#         logits = logits - max_element[:, None]  # to prevent overflow

#         numerator = torch.exp(logits)

#         denominator = (
#             (1 - targets)[:, None, :]
#             * self.s[None, :, :]
#             * torch.exp(logits)[:, None, :]).sum(axis=-1) \
#             + torch.exp(logits)

#         sigma = numerator / (denominator + self.eps)
#         loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
#         return loss.mean()

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate
Linear = torch.nn.Linear


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
