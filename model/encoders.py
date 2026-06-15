import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock2d(nn.Module):
    """Residual convolution block for high-fidelity shape encoding."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return torch.nn.functional.gelu(x + self.net(x))


class SpectralConv2d(nn.Module):
    """2D Fourier Layer for capturing global spectral dependencies."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralTrunkFNO(nn.Module):
    """
    Novel Spectral Trunk: FNO-based spatial feature extractor.
    Outputs a 64x64 feature map instead of a global vector.
    """
    def __init__(self, in_channels=1, out_channels=64, modes=12, width=64):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels) # Final feature map depth

    def forward(self, x):
        # x: (B, 1, 64, 64)
        x = x.permute(0, 2, 3, 1) # (B, 64, 64, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (B, 64, 64, 64)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 3, 1) # (B, 64, 64, 64)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2) # (B, C_out, 64, 64)


class ResidualShapeEncoder(nn.Module):
    """
    V10 'SOTA-Focus': 5-block ResNet encoder for high-fidelity shape capture.
    Designed to resolve extreme leading-edge curvature (suction peaks).
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # ResBlocks: Increasingly deep abstraction
        self.layer1 = nn.Sequential(ResBlock2d(32), nn.MaxPool2d(2)) # 32x32
        self.layer2 = nn.Sequential(ResBlock2d(32), nn.MaxPool2d(2)) # 16x16
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), ResBlock2d(64), nn.MaxPool2d(2)) # 8x8
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), ResBlock2d(128), nn.MaxPool2d(2)) # 4x4
        self.layer5 = ResBlock2d(128) # Final shaping at 4x4
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, latent_dim)

    def forward(self, sdf_grid):
        h = self.stem(sdf_grid)
        h = self.layer1(h)
        p1 = self.layer2(h) # (B, 32, 32, 32)
        h = self.layer3(p1)
        h = self.layer4(h)
        p2 = self.layer5(h) # (B, 128, 4, 4)
        
        mu = self.avgpool(p2).flatten(1)
        mu = self.head(mu)
        
        # Return global latent and the multiscale pyramid
        return mu, [p1, p2]


class MappingNetwork(nn.Module):
    """
    StyleGAN-style Mapping Network: (AoA_trig, Re_log) → W-space latent (w_dim).

    Projects the 4-D flow condition into a high-dimensional W-space that the
    hypernetwork can use to modulate the decoders without entangling geometry.

    Input:  aoa_enc (B, 3) = [sin(α), cos(α), sin(2α)], re_enc (B, 1) = [log10(Re/1e6)]
    Output: w (B, w_dim=64)
    """
    def __init__(self, cond_dim=4, w_dim=64, depth=4):
        super().__init__()
        layers, in_d = [], cond_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, w_dim), nn.LeakyReLU(0.2)]
            in_d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, aoa_enc, re_enc):
        return self.net(torch.cat([aoa_enc, re_enc], dim=-1))


class SurfaceCpDiscriminator(nn.Module):
    """
    V8 1D-CNN Discriminator: Critiques 1D surface pressure profiles (Cp) 
    sampled at N=128 points along normalized arc length.
    Goal: Capture local gradient sharpness that MSE loss smears.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4)), # 1024 -> 512
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)), # 512 -> 256
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)), # 256 -> 128
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(128, 256, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.utils.spectral_norm(nn.Linear(256, 1))

    def forward(self, cp):
        """cp shape: (B, 256)"""
        if cp.dim() == 2:
            cp = cp.unsqueeze(1)
        feat = self.conv(cp).squeeze(-1) # (B, 128)
        return self.head(feat)


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Passes x through forward, but multiplies gradient by -lambda during backward.
    """
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None

def grad_reverse(x, lam=1.0):
    return GradientReversal.apply(x, lam)


class AoAAdversary(nn.Module):
    """
    Adversarial predictor for AoA disentanglement.
    Tries to reconstruct AoA trig features from mu_geom.
    Combined with GRL, it forces mu_geom to be AoA-invariant.
    """
    def __init__(self, geom_dim=128, out_dim=3): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geom_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, mu_geom, lam=1.0):
        # Apply gradient reversal to mu_geom
        mu_rev = grad_reverse(mu_geom, lam)
        return self.net(mu_rev)
