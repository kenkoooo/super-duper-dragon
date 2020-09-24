import torch.nn.functional as F
import torch
import torch.nn as nn
from constants import *

ch = 192


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(shape))

    def forward(self, x):
        return x + self.bias


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Conv2d(PIECE_NUM, ch, 3, padding=1)
        self.l2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l4 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l5 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l6 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l7 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l8 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l9 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l10 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l11 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l12 = nn.Conv2d(ch, ch, 3, padding=1)
        self.l13 = nn.Conv2d(ch, MOVE_DIRECTION_LABEL_NUM, 1, bias=False)
        self.l13_bias = Bias(9*9*MOVE_DIRECTION_LABEL_NUM)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(x))
        h3 = F.relu(self.l3(x))
        h4 = F.relu(self.l4(x))
        h5 = F.relu(self.l5(x))
        h6 = F.relu(self.l6(x))
        h7 = F.relu(self.l7(x))
        h8 = F.relu(self.l8(x))
        h9 = F.relu(self.l9(x))
        h10 = F.relu(self.l10(x))
        h11 = F.relu(self.l11(x))
        h12 = F.relu(self.l12(x))
        h13 = self.l13(x)
        return self.l13_bias(h13.view(-1, 9*9*MOVE_DIRECTION_LABEL_NUM))
