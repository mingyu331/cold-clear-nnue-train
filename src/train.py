# responsible for training of nnue given data
import torch

from src.model import NNUE, Data
from src.ser_de import load_train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded = load_train('./train_data/out_high.txt.edited.txt')
net = NNUE()
for epoch in range(10):
    for i in map(lambda x:map(lambda y: Data(y), x), loaded):
        for j, data in enumerate(i):
            inputs, evaluation = data.encode(), data.eval
            outputs = net(inputs)
