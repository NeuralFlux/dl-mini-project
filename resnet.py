from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor



class ResNetModule(nn.Module):

    def __init__(self,
                blocks: List[int]):

        super(ResNetModule, self).__init__()


class BasicBlock(nn.Module):

    def __init__(self,
                in_channels: int,
                num_channels: int,
                kernel: Tuple[int, int],
                stride: int,
                padding: str = "same"):

        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.relu = nn.ReLU()

        self.conv_layer = nn.Sequential([
            nn.Conv2d(self.in_channels,
                            self.num_channels,
                            self.kernel,
                            self.stride,
                            self.padding,
                            bias=False),
            nn.BatchNorm2d(self.num_channels),
            self.relu,
            nn.Conv2d(self.in_channels,
                            self.num_channels,
                            self.kernel,
                            self.stride,
                            self.padding,
                            bias=False),
            nn.BatchNorm2d(self.num_channels)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_layer(x)
        out = self.relu(out + x)

        return out
