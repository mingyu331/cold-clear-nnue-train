# responsible for the structure and serialization of nnues
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy

import json

# ENCODE_LEN = ((8 * 37) << 12) + ((1 * 40) << 10)
ENCODE_LEN = 1
print(ENCODE_LEN)


# data for training
class data:
    def __init__(self) -> data:
        pass

    # should return a sparse tensor for performance
    def encode(self) -> torch.Tensor:
        pass


class nnue(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(ENCODE_LEN, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 2)

    def forward(self, d: data):
        encoded = d.encode()
        layer1 = self.encode(encoded)
        clamp1 = torch.clamp(layer1, 0, 1)
        layer2 = self.linear1(clamp1)
        clamp2 = torch.clamp(layer2, 0, 1)
        layer3 = self.linear2(clamp2)
        clamp3 = torch.clamp(layer3, 0, 1)
        layer4 = self.linear3(clamp3)
        # scale to respective sizes
        layer4[0] = 1000 * layer4[0].clamp(max=0)
        layer4[1] = layer4[1].clamp(min=0)
        return layer4

    def serialize_layer(l) -> str:
        params = list(l.parameters())
        params_flattened = numpy.concatenate(
            [
                torch.transpose(params[0], 0, 1).detach().numpy().flatten(),
                params[1].detach().numpy(),
            ]
        ).tolist()
        return json.dumps(params_flattened)

    def serialize_json(self):
        return f'{{"encode": {nnue.serialize_layer(self.encode)}, "layer1": {nnue.serialize_layer(self.linear1)}, "layer2": {nnue.serialize_layer(self.linear2)}, "layer3": {nnue.serialize_layer(self.linear3)}}}'
