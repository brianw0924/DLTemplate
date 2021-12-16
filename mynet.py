import torch.nn as nn

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

    def forward(self, x):