from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import *

CH = 192


class Bias(nn.Module, ABC):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(shape))

    def forward(self, x):
        return x + self.bias


class PolicyNetwork(nn.Module, ABC):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Conv2d(in_channels=PIECE_NUM, out_channels=CH, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l4 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l5 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l6 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l7 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l8 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l9 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l10 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l11 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l12 = nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, padding=1)
        self.l13 = nn.Conv2d(in_channels=CH, out_channels=MOVE_DIRECTION_LABEL_NUM, kernel_size=1, bias=False)
        self.l13_bias = Bias(9 * 9 * MOVE_DIRECTION_LABEL_NUM)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.relu(self.l12(h11))
        h13 = self.l13(h12)
        return self.l13_bias(h13.view(-1, 9 * 9 * MOVE_DIRECTION_LABEL_NUM))
