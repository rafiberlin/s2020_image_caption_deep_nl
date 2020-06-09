import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter


class ImageToHiddenState(nn.Module):
    """
    We try to transform each image to an hidden state with 120 values...
    """
    def __init__(self):
        super(ImageToHiddenState, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=3)
        self.out = nn.Linear(in_features=12 * 7 * 7, out_features=120)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        t = t.reshape(-1, 12 * 7 * 7)
        t = self.out(t)
        t = F.relu(t)

        return t
