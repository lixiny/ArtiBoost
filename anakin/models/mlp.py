from operator import mod
from typing import Dict

import torch
import torch.nn as nn
from anakin.utils.builder import MODEL
from anakin.utils.misc import enable_lower_param


@MODEL.register_module
class MLP_O(nn.Module):

    def __init__(self, **cfg):
        super().__init__()
        layers_n = cfg["LAYERS_N"]
        out_channel = cfg["OUT_CHANNEL"]
        layers = nn.ModuleList()
        for (in_n, out_n) in zip(layers_n[:-1], layers_n[1:]):
            layers.append(nn.Linear(in_n, out_n))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layers_n[-1], out_channel))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
