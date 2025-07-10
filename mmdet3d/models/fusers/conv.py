from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser","invarEncoder"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

@FUSERS.register_module()
class invarEncoder(nn.Sequential):
    def __init__(self, camera_in_channels: int, lidar_in_channels: int, invariant_out_channels: int, hidden_channel: int, block_nums=2) -> None:
        super().__init__()
        self.camera_invariant = nn.Conv2d(camera_in_channels, hidden_channel, kernel_size=3, padding=1)
        self.lidar_invariant = nn.Conv2d(lidar_in_channels, hidden_channel, kernel_size=3, padding=1)
        if block_nums == 1:
            self.out = ResidualBlock(hidden_channel, invariant_out_channels)
        else:
            self.out = nn.Sequential(
                *[ResidualBlock(hidden_channel, hidden_channel) for _ in range(block_nums-1)],
                ResidualBlock(hidden_channel, invariant_out_channels)
            )
    
    def forward(self, camera: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        camera_out = self.out(self.camera_invariant(camera))
        lidar_out = self.out(self.lidar_invariant(lidar)) 
        return camera_out , lidar_out