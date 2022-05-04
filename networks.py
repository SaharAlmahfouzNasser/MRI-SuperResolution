import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from deform_conv_3d import DeformConv3D


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class ConcatPool3d(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


class ConvLreluBlock(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv = nn.Conv3d(inf, outf, kernel, stride, padding)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class ConvBlock(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv = nn.Conv3d(inf, outf, kernel, stride, padding)
        self.bn = nn.BatchNorm3d(outf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, nf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(nf, nf, kernel, stride, padding)
        self.bn1 = nn.BatchNorm3d(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.conv2 = nn.Conv3d(nf, nf, kernel, stride, padding)
        self.bn2 = nn.BatchNorm3d(nf)

    def forward(self, x):
        z = self.lrelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return x + z


class ShiftAndScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.scale * x + self.shift


class SwitchableResBlock(nn.Module):
    def __init__(self, nf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(nf, nf, kernel, stride, padding)
        self.bn1 = nn.BatchNorm3d(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.conv2 = nn.Conv3d(nf, nf, kernel, stride, padding)
        self.bn2 = nn.BatchNorm3d(nf)
        self.sas = ShiftAndScale()

    def forward(self, x):
        z = self.lrelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return self.sas(x) + z


class UpConv(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, factor, slope=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv3d(inf, outf, kernel, stride, padding),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Upsample(scale_factor=factor),
            nn.Conv3d(outf, outf, kernel, stride, padding),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

    def forward(self, x):
        x = self.trunk(x)
        return x


class Generator(nn.Module):

    def __init__(self, kernel, inc, outc, ngf, n_blocks, scheme, tanh=True):
        super().__init__()
        self.conv1 = ConvBlock(inc, ngf, kernel, 1, 1)

        resblocks = []
        for _ in range(n_blocks):
            resblocks.append(ResBlock(ngf, kernel, 1, 1))

        self.trunk = nn.Sequential(*resblocks)
        self.conv2 = ConvBlock(ngf, ngf, kernel, 1, 1)

        if scheme == 'isotropic':
            self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 2.0, 2.0))
        elif scheme == 'anisotropic':
            self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 1.0, 1.0))
        else:
            raise ValueError(f'Scheme {scheme} not understood. Must be `isotropic` or `anisotropic`')

        if tanh:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf * 2, outc, kernel, 1, 1),
                nn.Tanh()
            )
        else:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf * 2, outc, kernel, 1, 1)
            )

    def forward(self, x):
        z = self.conv1(x)
        y = self.conv2(self.trunk(z))
        y = y + z
        y = self.upsample(y)
        y = self.final_conv(y)
        return y


class GeneratorSwitchSkip(nn.Module):

    def __init__(self, kernel, inc, outc, ngf, n_blocks, scheme, tanh=True):
        super().__init__()
        self.conv1 = ConvBlock(inc, ngf, kernel, 1, 1)

        resblocks = []
        for _ in range(n_blocks):
            resblocks.append(SwitchableResBlock(ngf, kernel, 1, 1))

        self.trunk = nn.Sequential(*resblocks)
        self.conv2 = ConvBlock(ngf, ngf, kernel, 1, 1)
        self.sas = ShiftAndScale()

        if scheme == 'isotropic':
            self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 2.0, 2.0))
        elif scheme == 'anisotropic':
            self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 1.0, 1.0))
        else:
            raise ValueError(f'Scheme {scheme} not understood. Must be `isotropic` or `anisotropic`')

        if tanh:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf * 2, outc, kernel, 1, 1),
                nn.Tanh()
            )
        else:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf * 2, outc, kernel, 1, 1)
            )

    def forward(self, x):
        z = self.conv1(x)
        y = self.conv2(self.trunk(z))
        # Switchable Global Skip Connnection
        y = y + self.sas(z)
        y = self.upsample(y)
        y = self.final_conv(y)
        return y


class Discriminator(nn.Module):

    def __init__(self, kernel, inc, ndf, logits=False):
        super().__init__()
        self.features = nn.Sequential(
            ConvLreluBlock(inc, ndf, kernel, 1, 1),
            ConvBlock(ndf, ndf, kernel, 1, 1),
            ConvBlock(ndf, ndf * 2, kernel, 1, 1),
            ConvBlock(ndf * 2, ndf * 2, kernel, 2, 1),
            ConvBlock(ndf * 2, ndf * 4, kernel, 1, 1),
            ConvBlock(ndf * 4, ndf * 4, kernel, 2, 1),
            ConvBlock(ndf * 4, ndf * 8, kernel, 1, 1),
            ConvBlock(ndf * 8, ndf * 8, kernel, 2, 1),
        )
        self.pool = ConcatPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 2, 512),
            nn.LeakyReLU(negative_slope=0.3, inplace=False),
            nn.Linear(512, 1)
        )
        if not logits:
            self.fc.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x.view(x.size(0), x.size(1)))
        return x


class ProjectionDiscriminator(nn.Module):

    def __init__(self, kernel, inc, ndf, scheme, logits=False):
        super().__init__()
        if scheme == 'isotropic':
            stride = (2, 2, 2)
        elif scheme == 'anisotropic':
            stride = (2, 1, 1)
        else:
            raise ValueError(f'Scheme {scheme} not understood. Must be `isotropic` or `anisotropic`')
        self.features1 = nn.Sequential(
            ConvLreluBlock(inc, ndf, kernel, 1, 1),
            ConvBlock(ndf, ndf, kernel, 1, 1),
            ConvBlock(ndf, ndf * 2, kernel, 1, 1),
            ConvBlock(ndf * 2, ndf * 2, kernel, stride, 1))

        self.projection_conv = nn.Conv3d(ndf * 2, inc, 3, 1, 1)

        self.features2 = nn.Sequential(
            ConvBlock(ndf * 2, ndf * 4, kernel, 1, 1),
            ConvBlock(ndf * 4, ndf * 4, kernel, 2, 1),
            ConvBlock(ndf * 4, ndf * 8, kernel, 1, 1),
            ConvBlock(ndf * 8, ndf * 8, kernel, 2, 1),
        )
        self.pool = ConcatPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 2, 512),
            nn.LeakyReLU(negative_slope=0.3, inplace=False),
            nn.Linear(512, 1)
        )
        self.act = nn.Identity() if logits else nn.Sigmoid()

    def forward(self, x, lr):
        x = self.features1(x)  # Image size reduces to half

        # Dot product
        h = self.projection_conv(x)
        h = torch.sum(h * lr, dim=(1, 2, 3, 4))

        x = self.features2(x)
        x = self.pool(x)
        x = self.fc(x.view(x.size(0), x.size(1)))
        x = x + h.unsqueeze(-1)
        x = self.act(x)

        return x
