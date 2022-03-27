import torch
import torch.nn as nn
import torch.nn.functional as F

# Common modules

class BlockWidth1d(nn.Module):

  def __init__(self, width) -> None:
      super().__init__()
      self.conv = nn.Conv1d(width, width, kernel_size=5, padding=2)

  def forward(self, x):
     x = x + F.leaky_relu(self.conv(x))
     return x


class BlockWidth2d(nn.Module):

  def __init__(self, width) -> None:
      super().__init__()
      self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)

  def forward(self, x):
     x = x + F.leaky_relu(self.conv(x))
     return x

class Downsample1d(nn.Module):

  def __init__(self, width, scale) -> None:
      super().__init__()
      self.blocks = nn.Sequential(
          BlockWidth1d(width),
          BlockWidth1d(width),
          BlockWidth1d(width),
          BlockWidth1d(width)
      )

      self.conv = nn.Conv1d(width, width*2, kernel_size=scale, stride=scale)

  def forward(self, x):

    return self.conv(self.blocks(x))


class Upsample1d(nn.Module):

  def __init__(self, width, scale) -> None:
      super().__init__()
      self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
      self.conv = nn.Conv1d(width*2, width, kernel_size=1)
      self.blocks = nn.Sequential(
          BlockWidth1d(width),
          BlockWidth1d(width),
          BlockWidth1d(width),
          BlockWidth1d(width)
      )
      self.out = nn.Conv1d(width*2, width, kernel_size=1)

  def forward(self, x, skip):

    x = self.blocks(self.conv(self.upsample(x)))

    return self.out(torch.cat([x, skip], dim=1))

class Downsample2d(nn.Module):

  def __init__(self, width, out_width, scale) -> None:
      super().__init__()
      self.blocks = nn.Sequential(
          BlockWidth2d(width),
          BlockWidth2d(width),
          BlockWidth2d(width),
          BlockWidth2d(width)
      )

      self.conv = nn.Conv2d(width, out_width, kernel_size=scale, stride=scale)

  def forward(self, x):

    return self.conv(self.blocks(x))

class Upsample2d(nn.Module):

  def __init__(self, in_width, width, scale) -> None:
      super().__init__()
      self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
      self.conv = nn.Conv2d(in_width, width, kernel_size=1)
      self.blocks = nn.Sequential(
          BlockWidth2d(width),
          BlockWidth2d(width),
          BlockWidth2d(width),
          BlockWidth2d(width)
      )
      self.out = nn.Conv2d(width*2, width, kernel_size=1)

  def forward(self, x, skip):

    x = self.blocks(self.conv(self.upsample(x)))

    return self.out(torch.cat([x, skip], dim=1))


# SpectralUnet

class SpectralUnet(nn.Module):

  def __init__(self, in_channels, out_channels) -> None:
      super().__init__()
      self.input = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)

      self.down1 = Downsample2d(8, 12, 2)
      self.down2 = Downsample2d(12, 24, 2)
      self.down3 = Downsample2d(24, 32, 2)
      self.bottleneck = nn.Sequential(
          BlockWidth2d(32),
          BlockWidth2d(32),
          BlockWidth2d(32),
          BlockWidth2d(32)
      )
      self.up3 = Upsample2d(32, 24, 2)
      self.up2 = Upsample2d(24, 12, 2)
      self.up1 = Upsample2d(12, 8, 2)

      self.output = nn.Conv2d(8, out_channels=out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    skip1 = self.input(x)
    skip2 = self.down1(skip1)
    skip3 = self.down2(skip2)
    bottleneck = self.bottleneck(self.down3(skip3))
    up3 = self.up3(bottleneck, skip3)
    up2 = self.up2(up3, skip2)
    up1 = self.up1(up2, skip1)

    return self.output(up1)

# WaveUnet

class WaveUnet(nn.Module):

  def __init__(self, in_channels, out_channels) -> None:
      super().__init__()
      self.input = nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=5, padding=2)

      self.down1 = Downsample1d(10, 4)
      self.down2 = Downsample1d(20, 4)
      self.down3 = Downsample1d(40, 4)
      self.bottleneck = nn.Sequential(
          BlockWidth1d(80),
          BlockWidth1d(80),
          BlockWidth1d(80),
          BlockWidth1d(80)
      )
      self.up3 = Upsample1d(40, 4)
      self.up2 = Upsample1d(20, 4)
      self.up1 = Upsample1d(10, 4)

      self.output = nn.Conv1d(10, out_channels=out_channels, kernel_size=5, padding=2)

  def forward(self, x):
    skip1 = self.input(x)
    skip2 = self.down1(skip1)
    skip3 = self.down2(skip2)
    bottleneck = self.bottleneck(self.down3(skip3))
    up3 = self.up3(bottleneck, skip3)
    up2 = self.up2(up3, skip2)
    up1 = self.up1(up2, skip1)

    return self.output(up1)