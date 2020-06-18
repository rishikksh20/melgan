import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResStack(nn.Module):
    def __init__(self, channel, dilation=1):
        super(ResStack, self).__init__()

        self.block = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(dilation),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=dilation)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
           

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
