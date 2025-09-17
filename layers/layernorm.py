import torch
from torch import nn
import torch.nn.functional as F

"""
Taken from the minGPT repo.
"""
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. """

    def __init__(self, ndim, eps, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
