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
class Data:
    def __init__(self, s: dict):
        # current piece
        self.piece_type = s["mv"]["kind"][0]
        # locks
        self.placement_kind = s["lock"]["placement_kind"]
        self.b2b = s["lock"]["b2b"]
        self.pc = s["lock"]["perfect_clear"]
        self.combo = s["lock"]["combo"]
        self.garbage_sent = s["lock"]["garbage_sent"]
        self.cleared_lines = s["lock"]["cleared_lines"]

        # board
        self.board = s["board"]["cells"]  # must add current piece to it
        self.column_heights = s["board"]["column_heights"]
        self.bag = s["board"]["bag"]
        self.hold = s["board"]["hold_piece"]
        self.queue = s["board"]["next_pieces"]

        # eval (the target)
        self.eval = s["evaluation"]  # (value, spike)

        pass

    def to_idx(value):
        return {"I": 0, "O": 1, "T": 2, "L": 3, "J": 4, "S": 5, "Z": 6}[value]

    def convolute(self, m, n, offset=0):
        ret = [[(row >> j) & 1 for j in range(10)] for row in self.board]
        for row in ret:
            for i in range(11 - m):
                for j in range(1, m + 1):
                    row[i] += row[i + j] << j

        for j in range(10):
            for i in range(41 - n):
                for k in range(1, n + 1):
                    ret[i][j] += ret[i + k][j] << (m * k)
        indices = []
        for i in range(41 - n):
            for j in range(11 - m):
                indices.append((1 << (n * m)) * (i * (11 - m) + j) + ret[i][j] + offset)
        return indices

    # should return a sparse tensor for performance
    def encode(self) -> torch.Tensor:
        offset = 0
        indices = self.convolute(3, 4, offset)
        offset += (1 << (3 * 4)) * (11 - 3) * (41 - 4)
        indices += self.convolute(10, 1, offset)
        offset += (1 << (10 * 1)) * (11 - 10) * (41 - 1)
        for i in [0, 1, 2, 3, 4, 5, 6]:
            if (self.bag >> i) & 1:
                indices.append(offset + i)
        offset += 7
        if self.b2b:
            indices.append(offset)
        offset += 1
        indices.append(offset + min(self.combo, 19))
        offset += 20
        for i, v in enumerate(self.queue):
            indices.append(offset + 7 * i + Data.to_idx(v))
        offset += 35
        # I know this is not a sufficient check, but I don't care
        if self.hold in "IOTLJSZ":
            indices.append(offset + Data.to_idx(self.hold))
        offset += 7
        values = [1] * len(indices)
        print(offset)
        return torch.sparse_coo_tensor([indices], values, offset)


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(ENCODE_LEN, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 2)

    def forward(self, d: Data):
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
        return f'{{"encode": {NNUE.serialize_layer(self.encode)}, "layer1": {NNUE.serialize_layer(self.linear1)}, "layer2": {NNUE.serialize_layer(self.linear2)}, "layer3": {NNUE.serialize_layer(self.linear3)}}}'
