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


class SpectralMaskNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(80, 513, 1)
        self.spectralunet = SpectralUnet(1, 1)

    def forward(self, x, m):
        mag, phase, _, _ = self.stft(x)
        print(m.shape)
        m = self.conv1d(m)
        print(m.shape)
        inp = torch.cat([mag, m.unsqueeze(1)], dim=2)
        print(inp.shape)
        mul = F.softplus(self.spectralunet(inp))
        mag_ = mag * mul
        out = self.istft((mag_, phase))
        return out

    def stft(self, y, n_fft=1024, hop_length=256, win_length=1024):
        """
        Wrapper of the official torch.stft for single-channel and multi-channel
        Args:
            y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
            n_fft: num of FFT
            hop_length: hop length
            win_length: hanning window size
        Shapes:
            mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]
        Returns:
            mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
        """
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)

        complex_stft = torch.stft(y, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=y.device),
                                  return_complex=True)
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

        mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
        real, imag = complex_stft.real, complex_stft.imag
        return mag, phase, real, imag

    def istft(self, features, n_fft=1024, hop_length=256, win_length=1024, length=None, input_type="mag_phase"):
        """
        Wrapper of the official torch.istft
        Args:
            features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
            n_fft: num of FFT
            hop_length: hop length
            win_length: hanning window size
            length: expected length of istft
            use_mag_phase: use mag and phase as the input ("features")
        Returns:
            single-channel speech of shape [B, T]
        """
        if input_type == "real_imag":
            # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert isinstance(features, torch.ComplexType)
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError("Only 'real_imag', 'complex', and 'mag_phase' are supported")

        return torch.istft(features, n_fft, hop_length, win_length,
                           window=torch.hann_window(n_fft, device=features.device),
                           length=length)